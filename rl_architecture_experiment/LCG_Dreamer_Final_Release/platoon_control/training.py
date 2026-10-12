"""Training orchestration, metrics, and versioned checkpoints."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .agents import PlatoonController
from .config import SystemConfig


class MetricLogger:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.json_path = self.directory / "metrics.jsonl"
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(str(self.directory / "tensorboard"))
        except ImportError:
            self.tensorboard = None

    def write(self, step: int, metrics: Mapping[str, float]) -> None:
        record = {"step": step, **metrics}
        with self.json_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        if self.tensorboard is not None:
            for name, value in metrics.items():
                self.tensorboard.add_scalar(name, value, step)

    def close(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.close()


class PlatoonTrainer:
    def __init__(self, controller: PlatoonController) -> None:
        self.controller = controller
        training = controller.config.training
        parameters = {
            "delay": controller.delay.parameters(),
            "trajectory": controller.trajectory.parameters(),
            "leader": [
                parameter
                for name, parameter in controller.leader.named_parameters()
                if not name.startswith(("actor.", "critic."))
            ],
            "leader_actor": list(controller.leader.actor.parameters())
            + list(controller.leader.critic.parameters()),
            "follower_actor": controller.follower.actor.parameters(),
            "follower_critic": list(controller.follower.critic1.parameters())
            + list(controller.follower.critic2.parameters()),
        }
        self.optimizers = {
            name: torch.optim.AdamW(
                values,
                lr=(
                    training.learning_rate * 0.01
                    if name == "follower_actor"
                    else (
                        training.learning_rate * 0.1
                        if name == "leader_actor"
                        else training.learning_rate
                    )
                ),
                weight_decay=training.weight_decay,
            )
            for name, values in parameters.items()
        }
        self.gradient_clip = training.gradient_clip

    def _update(self, name: str, loss: Tensor) -> float:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        gradient_clip = (
            min(self.gradient_clip, 5.0)
            if name == "follower_actor"
            else self.gradient_clip
        )
        clip_grad_norm_(parameters, gradient_clip)
        optimizer.step()
        return float(loss.detach())

    def train_delay(self, batch: Mapping[str, Tensor]) -> dict[str, float]:
        self.controller.delay.train()
        batch = {key: value.to(self.controller.device) for key, value in batch.items()}
        prediction = self.controller.delay(batch["v2v_history"], batch["local_predecessor"])
        losses = self.controller.delay.loss(
            prediction, batch["current_target"], batch["future_target"]
        )
        self._update("delay", losses["loss"])
        return {f"delay/{key}": float(value.detach()) for key, value in losses.items()}

    def train_trajectory(self, batch: Mapping[str, Tensor]) -> dict[str, float]:
        self.controller.trajectory.train()
        batch = {key: value.to(self.controller.device) for key, value in batch.items()}
        prediction = self.controller.trajectory(batch["history"], batch["mask"])
        losses = self.controller.trajectory.loss(
            prediction,
            batch["future"],
            batch["mask"],
            batch["cutin"],
            batch.get("lane_change"),
        )
        self._update("trajectory", losses["loss"])
        return {f"trajectory/{key}": float(value.detach()) for key, value in losses.items()}

    def train_followers(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        continues: Tensor,
        behavior_actions: Tensor,
        requested_behavior_actions: Tensor,
        update_actor: bool = True,
        behavior_weight: float | None = None,
        actor_rl_weight: float = 1.0,
        actor_stability_weight: float = 0.0,
    ) -> dict[str, float]:
        self.controller.follower.train()
        states = states.to(self.controller.device)
        actions = actions.to(self.controller.device)
        rewards = rewards.to(self.controller.device)
        next_states = next_states.to(self.controller.device)
        continues = continues.to(self.controller.device)
        behavior_actions = behavior_actions.to(self.controller.device)
        requested_behavior_actions = requested_behavior_actions.to(
            self.controller.device
        )

        critic_loss, critic_metrics = self.controller.follower.critic_loss(
            states, actions, rewards, next_states, continues
        )
        self._update("follower_critic", critic_loss)
        self.controller.follower.soft_update_targets()
        output = {
            f"follower/{key}": float(value.detach())
            for key, value in critic_metrics.items()
        }
        if update_actor:
            actor_loss, actor_metrics = self.controller.follower.actor_loss(
                states,
                behavior_actions,
                requested_behavior_actions=requested_behavior_actions,
                behavior_weight=behavior_weight,
                rl_weight=actor_rl_weight,
                stability_weight=actor_stability_weight,
            )
            self._update("follower_actor", actor_loss)
            output.update(
                {
                    f"follower/{key}": float(value.detach())
                    for key, value in actor_metrics.items()
                }
            )
            output["follower/loss"] = float(
                critic_loss.detach() + actor_loss.detach()
            )
        else:
            output["follower/loss"] = float(critic_loss.detach())
            output["follower/actor_deferred"] = 1.0
        return output

    def train_follower_actor(
        self,
        states: Tensor,
        behavior_actions: Tensor,
        requested_behavior_actions: Tensor,
        behavior_weight: float | None = None,
        rl_weight: float = 1.0,
        stability_weight: float = 0.0,
    ) -> dict[str, float]:
        """Update only the follower actor between fixed-policy episodes."""

        self.controller.follower.train()
        states = states.to(self.controller.device)
        behavior_actions = behavior_actions.to(self.controller.device)
        requested_behavior_actions = requested_behavior_actions.to(
            self.controller.device
        )
        actor_loss, actor_metrics = self.controller.follower.actor_loss(
            states,
            behavior_actions,
            requested_behavior_actions=requested_behavior_actions,
            behavior_weight=behavior_weight,
            rl_weight=rl_weight,
            stability_weight=stability_weight,
        )
        self._update("follower_actor", actor_loss)
        return {
            f"follower/{key}": float(value.detach())
            for key, value in actor_metrics.items()
        }

    def train_leader(
        self,
        observations: Mapping[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        continues: Tensor,
        update_actor: bool = True,
        behavior_weight: float = 0.0,
    ) -> dict[str, float]:
        self.controller.leader.train()
        observations = {
            key: value.to(self.controller.device) for key, value in observations.items()
        }
        losses = self.controller.leader.world_model_loss(
            observations,
            actions.to(self.controller.device),
            rewards.to(self.controller.device),
            continues.to(self.controller.device),
        )
        self._update("leader", losses["loss"])
        output = {f"leader/{key}": float(value.detach()) for key, value in losses.items()}
        if not update_actor:
            output["leader_imagination/actor_deferred"] = 1.0
            return output
        with torch.no_grad():
            posterior_sequence, _ = self.controller.leader.observe_sequence(
                observations, actions.to(self.controller.device)
            )
            posterior = posterior_sequence[-1]
        imagination = self.controller.leader.imagination_loss(posterior)
        behavior_loss = self.controller.leader.behavior_loss(
            posterior_sequence, actions.to(self.controller.device)
        )
        actor_loss = imagination["loss"] + float(behavior_weight) * behavior_loss
        self._update("leader_actor", actor_loss)
        output.update(
            {f"leader_imagination/{key}": float(value.detach()) for key, value in imagination.items()}
        )
        output["leader_imagination/behavior"] = float(behavior_loss.detach())
        output["leader_imagination/behavior_weight"] = float(behavior_weight)
        output["leader_imagination/loss"] = float(actor_loss.detach())
        return output

    def train_leader_front(self, batch: Mapping[str, Tensor]) -> dict[str, float]:
        history = batch["front_history"].to(self.controller.device)
        mask = batch["front_mask"].to(self.controller.device)
        future = batch["front_future"].to(self.controller.device)
        ego = torch.zeros(history.shape[0], 8, device=self.controller.device)
        ego[:, 1] = history[:, -1, -1, 2]
        losses = self.controller.leader.front_prediction_loss(history, ego, mask, future)
        self._update("leader", losses["loss"])
        return {f"leader_front/{key}": float(value.detach()) for key, value in losses.items()}

    def save_checkpoint(
        self,
        path: str | Path,
        step: int,
        metrics: Mapping[str, float],
        data_sources: list[str],
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload: dict[str, Any] = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "step": step,
            "config": self.controller.config.to_dict(),
            "models": self.controller.state_dict(),
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
            "metrics": dict(metrics),
            "data_sources": data_sources,
        }
        torch.save(payload, temporary)
        temporary.replace(path)
        return path


def load_controller(path: str | Path, device: str = "auto") -> Any:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("control_architecture") == "ma_latent_graph_dreamer_v1":
        from .graph_rl import load_graph_controller

        return load_graph_controller(path, device)
    config = SystemConfig.from_dict(checkpoint["config"])
    config.training.device = device
    controller = PlatoonController(config)
    controller.load_state_dict(checkpoint["models"])
    controller.eval()
    return controller

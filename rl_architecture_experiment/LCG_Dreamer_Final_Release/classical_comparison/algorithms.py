"""DDPG-family, PPO, and SAC implementations over the frozen protocol.

The DDPG-family baseline uses the standard twin-critic and target-smoothing
stabilizers; checkpoints and metrics identify that implementation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from .common import ComparisonProtocol, RunningNormalizer
from .models import (
    DeterministicActor,
    QNetwork,
    SquashedGaussianActor,
    ValueNetwork,
)


ACTION_DIM = 6


@dataclass
class ActionSample:
    action: np.ndarray
    auxiliary: dict[str, Any]


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.states = np.empty((capacity, state_dim), np.float32)
        self.actions = np.empty((capacity, ACTION_DIM), np.float32)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.next_states = np.empty((capacity, state_dim), np.float32)
        self.continues = np.empty((capacity, 1), np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def append(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index, 0] = reward
        self.next_states[index] = next_state
        self.continues[index, 0] = float(not done)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.continues[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        size = self.size
        if size < self.capacity:
            indices = np.arange(size)
        else:
            indices = np.concatenate(
                (np.arange(self.position, self.capacity), np.arange(self.position))
            )
        return {
            "capacity": self.capacity,
            "states": self.states[indices].astype(np.float16),
            "actions": self.actions[indices].astype(np.float16),
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices].astype(np.float16),
            "continues": self.continues[indices],
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        states = np.asarray(state["states"], np.float32)
        size = min(len(states), self.capacity)
        self.states[:size] = states[-size:]
        self.actions[:size] = np.asarray(state["actions"], np.float32)[-size:]
        self.rewards[:size] = np.asarray(state["rewards"], np.float32)[-size:]
        self.next_states[:size] = np.asarray(
            state["next_states"], np.float32
        )[-size:]
        self.continues[:size] = np.asarray(
            state["continues"], np.float32
        )[-size:]
        self.size = size
        self.position = size % self.capacity
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


class ClassicalAlgorithm:
    algorithm = "base"

    def __init__(
        self,
        state_dim: int,
        protocol: ComparisonProtocol,
        device: str,
        seed: int,
    ) -> None:
        self.state_dim = int(state_dim)
        self.protocol = protocol
        self.device = torch.device(device)
        self.seed = int(seed)
        self.normalizer = RunningNormalizer(state_dim)
        self.normalizer_frozen = False
        self.total_updates = 0
        self.rng = np.random.default_rng(seed)

    def normalized_tensor(self, state: np.ndarray) -> Tensor:
        normalized = self.normalizer.normalize(state)
        return torch.as_tensor(
            normalized, dtype=torch.float32, device=self.device
        )

    def observe_state(self, state: np.ndarray) -> None:
        if not self.normalizer_frozen:
            self.normalizer.update(state)

    def act(
        self, state: np.ndarray, deterministic: bool, total_steps: int
    ) -> ActionSample:
        raise NotImplementedError

    def record(
        self,
        state: np.ndarray,
        sample: ActionSample,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def update_if_ready(self, total_steps: int, force: bool = False) -> dict[str, float]:
        raise NotImplementedError

    def state_dict(self, include_replay: bool = True) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.lerp_(parameter, tau)

    @staticmethod
    def _clip(parameters: Any, maximum: float = 10.0) -> None:
        nn.utils.clip_grad_norm_(parameters, maximum)


class DDPG(ClassicalAlgorithm):
    algorithm = "ddpg"
    implementation_variant = "proximal_twin_critic_stabilized"

    def __init__(self, state_dim: int, protocol: ComparisonProtocol, device: str, seed: int) -> None:
        super().__init__(state_dim, protocol, device, seed)
        self.actor = DeterministicActor(state_dim, ACTION_DIM).to(self.device)
        self.reference_actor = DeterministicActor(state_dim, ACTION_DIM).to(self.device)
        self.anchor_actor = DeterministicActor(state_dim, ACTION_DIM).to(self.device)
        self.actor_target = DeterministicActor(state_dim, ACTION_DIM).to(self.device)
        self.critic = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.critic2 = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.critic_target = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.critic2_target = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.requires_grad_(False)
        self.anchor_actor.load_state_dict(self.actor.state_dict())
        self.anchor_actor.requires_grad_(False)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.replay = ReplayBuffer(protocol.replay_capacity, state_dim, seed)
        self.learning_starts = 5_000
        self.actor_learning_starts = self.learning_starts
        self.policy_delay = 1
        self.policy_objective_weight = 1.0
        self.target_policy_coefficient = 0.0
        self.reference_policy_coefficient = 0.0
        self.normalize_actor_q = False
        self.anchor_coefficient = 0.0
        self.anchor_decay_steps = 400_000
        self.tau = 0.005
        self.target_policy_noise = 0.10
        self.target_noise_clip = 0.20
        self.exploration_initial = 0.25
        self.exploration_final = 0.05
        self.exploration_decay_steps = 150_000

    def reset_policy_reference(self) -> None:
        """Freeze the current learned policy as the next-stage trust reference."""

        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.requires_grad_(False)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool, total_steps: int) -> ActionSample:
        if not deterministic:
            self.observe_state(state)
        tensor = self.normalized_tensor(state).unsqueeze(0)
        action = self.actor(tensor).squeeze(0).cpu().numpy()
        if not deterministic:
            progress = min(1.0, total_steps / self.exploration_decay_steps)
            standard_deviation = self.exploration_initial + progress * (
                self.exploration_final - self.exploration_initial
            )
            common = self.rng.normal(0.0, standard_deviation)
            independent = self.rng.normal(0.0, standard_deviation * 0.25, ACTION_DIM)
            action = action + 0.8 * common + independent
        return ActionSample(np.clip(action, -1.0, 1.0).astype(np.float32), {})

    def record(self, state: np.ndarray, sample: ActionSample, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.append(state, sample.action, reward, next_state, done)

    def update_if_ready(self, total_steps: int, force: bool = False) -> dict[str, float]:
        del force
        if total_steps < self.learning_starts or self.replay.size < self.protocol.batch_size:
            return {}
        states, actions, rewards, next_states, continues = self.replay.sample(
            self.protocol.batch_size
        )
        state = self.normalized_tensor(states)
        next_state = self.normalized_tensor(next_states)
        action = torch.as_tensor(actions, device=self.device)
        reward = torch.as_tensor(rewards, device=self.device) * self.protocol.reward_scale
        cont = torch.as_tensor(continues, device=self.device)
        with torch.no_grad():
            target_action = self.actor_target(next_state)
            smoothing_noise = torch.randn_like(target_action).mul_(
                self.target_policy_noise
            ).clamp_(-self.target_noise_clip, self.target_noise_clip)
            target_action = (target_action + smoothing_noise).clamp_(-1.0, 1.0)
            target_q = torch.minimum(
                self.critic_target(next_state, target_action),
                self.critic2_target(next_state, target_action),
            )
            target = reward + self.protocol.discount * cont * target_q
        predicted = self.critic(state, action)
        predicted2 = self.critic2(state, action)
        critic_loss = F.mse_loss(predicted, target) + F.mse_loss(
            predicted2, target
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(
            list(self.critic.parameters()) + list(self.critic2.parameters())
        )
        self.critic_optimizer.step()
        self.critic2_optimizer.step()

        self._soft_update(self.critic_target, self.critic, self.tau)
        self._soft_update(self.critic2_target, self.critic2, self.tau)
        self.total_updates += 1
        update_actor = (
            total_steps >= self.actor_learning_starts
            and self.total_updates % self.policy_delay == 0
        )
        actor_loss = torch.zeros((), device=self.device)
        target_policy_loss = torch.zeros((), device=self.device)
        reference_policy_loss = torch.zeros((), device=self.device)
        policy_q_scale = torch.ones((), device=self.device)
        if update_actor:
            current_action = self.actor(state)
            with torch.no_grad():
                anchor_action = self.anchor_actor(state)
                target_policy_action = self.actor_target(state)
                reference_action = self.reference_actor(state)
            anchor_loss = F.mse_loss(current_action, anchor_action)
            target_policy_loss = F.mse_loss(
                current_action, target_policy_action
            )
            reference_policy_loss = F.mse_loss(
                current_action, reference_action
            )
            anchor_weight = self.anchor_coefficient * max(
                0.0, 1.0 - total_steps / max(self.anchor_decay_steps, 1)
            )
            policy_q = self.critic(state, current_action)
            policy_q_scale = (
                policy_q.abs().mean().detach().clamp(min=1.0)
                if self.normalize_actor_q
                else torch.ones((), device=self.device)
            )
            policy_loss = -policy_q.mean() / policy_q_scale
            actor_loss = (
                self.policy_objective_weight * policy_loss
                + anchor_weight * anchor_loss
                + self.target_policy_coefficient * target_policy_loss
                + self.reference_policy_coefficient * reference_policy_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip(self.actor.parameters())
            self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor, self.tau)
        return {
            "loss/actor": float(actor_loss.detach()),
            "loss/anchor": float(anchor_loss.detach()) if update_actor else 0.0,
            "loss/target_policy_trust": float(target_policy_loss.detach()),
            "loss/reference_policy_trust": float(
                reference_policy_loss.detach()
            ),
            "train/anchor_coefficient": float(anchor_weight) if update_actor else 0.0,
            "train/target_policy_coefficient": float(
                self.target_policy_coefficient if update_actor else 0.0
            ),
            "train/reference_policy_coefficient": float(
                self.reference_policy_coefficient if update_actor else 0.0
            ),
            "train/policy_q_scale": float(policy_q_scale.detach()),
            "loss/critic": float(critic_loss.detach()),
            "train/actor_updated": float(update_actor),
            "train/q": float(torch.minimum(predicted, predicted2).detach().mean()),
            "train/q_disagreement": float(
                (predicted - predicted2).abs().detach().mean()
            ),
            "train/twin_critic": 1.0,
        }

    def state_dict(self, include_replay: bool = True) -> dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "reference_actor": self.reference_actor.state_dict(),
            "anchor_actor": self.anchor_actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "implementation_variant": self.implementation_variant,
            "normalizer": self.normalizer.state_dict(),
            "total_updates": self.total_updates,
            "rng_state": self.rng.bit_generator.state,
        }
        if include_replay:
            state["replay"] = self.replay.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in ("actor", "actor_target", "critic", "critic_target"):
            getattr(self, name).load_state_dict(state[name])
        self.reference_actor.load_state_dict(
            state.get("reference_actor", state["actor"])
        )
        if "critic2" in state:
            self.critic2.load_state_dict(state["critic2"])
            self.critic2_target.load_state_dict(state["critic2_target"])
        self.anchor_actor.load_state_dict(state.get("anchor_actor", state["actor"]))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "critic2_optimizer" in state:
            self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.total_updates = int(state["total_updates"])
        self.rng.bit_generator.state = state["rng_state"]
        if "replay" in state:
            self.replay.load_state_dict(state["replay"])


class SAC(ClassicalAlgorithm):
    algorithm = "sac"

    def __init__(self, state_dim: int, protocol: ComparisonProtocol, device: str, seed: int) -> None:
        super().__init__(state_dim, protocol, device, seed)
        self.actor = SquashedGaussianActor(
            state_dim, ACTION_DIM, state_dependent_std=True
        ).to(self.device)
        self.anchor_actor = SquashedGaussianActor(
            state_dim, ACTION_DIM, state_dependent_std=True
        ).to(self.device)
        self.reference_actor = SquashedGaussianActor(
            state_dim, ACTION_DIM, state_dependent_std=True
        ).to(self.device)
        self.critic1 = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.critic2 = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.target1 = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.target2 = QNetwork(state_dim, ACTION_DIM).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.anchor_actor.load_state_dict(self.actor.state_dict())
        self.anchor_actor.requires_grad_(False)
        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4
        )
        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=3e-4)
        self.target_entropy = -float(ACTION_DIM)
        self.maximum_alpha: float | None = None
        self.behavior_noise_scale = 1.0
        # Raw shared rewards are scaled by 0.1 and observed in approximately
        # [-54, 12]. These discounted bounds retain the physical return range
        # while preventing optimistic OOD bootstraps after curriculum changes.
        self.critic_target_min = -600.0
        self.critic_target_max = 150.0
        self.replay = ReplayBuffer(protocol.replay_capacity, state_dim, seed)
        self.learning_starts = 5_000
        self.actor_learning_starts = self.learning_starts
        self.policy_delay = 1
        self.policy_objective_weight = 1.0
        self.anchor_coefficient = 0.0
        self.reference_policy_coefficient = 0.0
        self.anchor_decay_steps = 400_000
        self.conservative_weight = 0.0
        self.conservative_samples = 4
        self.conservative_temperature = 1.0
        self.normalize_actor_q = False
        self.tau = 0.005

    def reset_policy_reference(self) -> None:
        """Freeze the current policy as a trust-region reference."""

        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.requires_grad_(False)

    def reset_value_learning(self) -> None:
        """Reset OOD-sensitive value state while preserving the repaired actor."""

        critic_lr = float(self.critic_optimizer.param_groups[0]["lr"])
        self.critic1 = QNetwork(self.state_dim, ACTION_DIM).to(self.device)
        self.critic2 = QNetwork(self.state_dim, ACTION_DIM).to(self.device)
        self.target1 = QNetwork(self.state_dim, ACTION_DIM).to(self.device)
        self.target2 = QNetwork(self.state_dim, ACTION_DIM).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.replay = ReplayBuffer(
            self.protocol.replay_capacity, self.state_dim, self.seed + 97_000
        )
        self.total_updates = 0

    def _policy_distribution(self, state: Tensor) -> Normal:
        """Return the same exploration policy used online and in backups."""

        base = self.actor.distribution(state)
        scale = max(float(self.behavior_noise_scale), 1e-4)
        return Normal(base.loc, base.scale * scale)

    def _sample_policy(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self._policy_distribution(state)
        raw = distribution.rsample()
        unit = raw.tanh().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_prob = self.actor._squashed_log_prob_from_raw(distribution, raw)
        deterministic = distribution.mean.tanh()
        return unit, log_prob, deterministic

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool, total_steps: int) -> ActionSample:
        if not deterministic:
            self.observe_state(state)
        tensor = self.normalized_tensor(state).unsqueeze(0)
        if deterministic or self.behavior_noise_scale <= 0.0:
            action = self.actor.deterministic(tensor)
        else:
            action, _, _ = self._sample_policy(tensor)
        return ActionSample(action.squeeze(0).cpu().numpy().astype(np.float32), {})

    def record(self, state: np.ndarray, sample: ActionSample, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.append(state, sample.action, reward, next_state, done)

    def update_if_ready(self, total_steps: int, force: bool = False) -> dict[str, float]:
        del force
        if total_steps < self.learning_starts or self.replay.size < self.protocol.batch_size:
            return {}
        states, actions, rewards, next_states, continues = self.replay.sample(
            self.protocol.batch_size
        )
        state = self.normalized_tensor(states)
        next_state = self.normalized_tensor(next_states)
        action = torch.as_tensor(actions, device=self.device)
        reward = torch.as_tensor(rewards, device=self.device) * self.protocol.reward_scale
        cont = torch.as_tensor(continues, device=self.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_log_prob, _ = self._sample_policy(next_state)
            next_q = torch.minimum(
                self.target1(next_state, next_action),
                self.target2(next_state, next_action),
            ) - alpha * next_log_prob
            target = reward + self.protocol.discount * cont * next_q
            unclipped_target = target
            target = target.clamp(
                self.critic_target_min, self.critic_target_max
            )
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        td_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        conservative_loss = torch.zeros((), device=self.device)
        if self.conservative_weight > 0.0:
            repeats = max(1, int(self.conservative_samples))
            repeated_state = state[:, None, :].expand(
                -1, repeats, -1
            ).reshape(-1, state.shape[-1])
            with torch.no_grad():
                policy_actions, _, _ = self._sample_policy(repeated_state)
                random_actions = torch.empty_like(policy_actions).uniform_(
                    -1.0, 1.0
                )
            q1_ood = torch.cat(
                (
                    self.critic1(repeated_state, policy_actions),
                    self.critic1(repeated_state, random_actions),
                ),
                dim=1,
            ).reshape(state.shape[0], -1)
            q2_ood = torch.cat(
                (
                    self.critic2(repeated_state, policy_actions),
                    self.critic2(repeated_state, random_actions),
                ),
                dim=1,
            ).reshape(state.shape[0], -1)
            temperature = max(float(self.conservative_temperature), 1e-4)
            conservative_loss = (
                temperature
                * torch.logsumexp(q1_ood / temperature, dim=1).mean()
                - q1.mean()
                + temperature
                * torch.logsumexp(q2_ood / temperature, dim=1).mean()
                - q2.mean()
            )
        critic_loss = td_loss + self.conservative_weight * conservative_loss
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("non-finite SAC critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(list(self.critic1.parameters()) + list(self.critic2.parameters()))
        self.critic_optimizer.step()

        self._soft_update(self.target1, self.critic1, self.tau)
        self._soft_update(self.target2, self.critic2, self.tau)
        self.total_updates += 1
        update_actor = (
            total_steps >= self.actor_learning_starts
            and self.total_updates % self.policy_delay == 0
        )
        actor_loss = torch.zeros((), device=self.device)
        alpha_loss = torch.zeros((), device=self.device)
        reference_loss = torch.zeros((), device=self.device)
        if update_actor:
            sampled_action, log_prob, _ = self._sample_policy(state)
            policy_q = torch.minimum(
                self.critic1(state, sampled_action),
                self.critic2(state, sampled_action),
            ).clamp(self.critic_target_min, self.critic_target_max)
            policy_q_scale = (
                policy_q.abs().mean().detach().clamp(min=1.0)
                if self.normalize_actor_q
                else torch.ones((), device=self.device)
            )
            policy_loss = (alpha * log_prob - policy_q / policy_q_scale).mean()
            current_distribution = self.actor.distribution(state)
            with torch.no_grad():
                anchor_distribution = self.anchor_actor.distribution(state)
                reference_distribution = self.reference_actor.distribution(state)
            anchor_loss = F.mse_loss(
                current_distribution.mean, anchor_distribution.mean
            ) + 0.1 * F.mse_loss(
                current_distribution.scale.log(),
                anchor_distribution.scale.log(),
            )
            anchor_weight = self.anchor_coefficient * max(
                0.0, 1.0 - total_steps / max(self.anchor_decay_steps, 1)
            )
            reference_loss = F.mse_loss(
                current_distribution.mean, reference_distribution.mean
            ) + 0.1 * F.mse_loss(
                current_distribution.scale.log(),
                reference_distribution.scale.log(),
            )
            actor_loss = (
                self.policy_objective_weight * policy_loss
                + anchor_weight * anchor_loss
                + self.reference_policy_coefficient * reference_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip(self.actor.parameters())
            self.actor_optimizer.step()

            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if self.maximum_alpha is not None:
                self.log_alpha.data.clamp_(max=float(np.log(self.maximum_alpha)))
        return {
            "loss/actor": float(actor_loss.detach()),
            "loss/anchor": float(anchor_loss.detach()) if update_actor else 0.0,
            "loss/reference_policy_trust": float(reference_loss.detach()),
            "train/anchor_coefficient": float(anchor_weight) if update_actor else 0.0,
            "train/reference_policy_coefficient": float(
                self.reference_policy_coefficient if update_actor else 0.0
            ),
            "loss/critic": float(critic_loss.detach()),
            "loss/critic_td": float(td_loss.detach()),
            "loss/conservative": float(conservative_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "train/actor_updated": float(update_actor),
            "train/alpha": float(self.log_alpha.exp().detach()),
            "train/q": float(torch.minimum(q1, q2).detach().mean()),
            "train/policy_q": (
                float(policy_q.detach().mean()) if update_actor else 0.0
            ),
            "train/policy_q_scale": (
                float(policy_q_scale.detach()) if update_actor else 1.0
            ),
            "train/q_abs_max": float(
                torch.maximum(q1.abs(), q2.abs()).detach().max()
            ),
            "train/target_abs_max": float(target.detach().abs().max()),
            "train/target_clipped_fraction": float(
                (target != unclipped_target).float().detach().mean()
            ),
        }

    def state_dict(self, include_replay: bool = True) -> dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "anchor_actor": self.anchor_actor.state_dict(),
            "reference_actor": self.reference_actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "total_updates": self.total_updates,
            "rng_state": self.rng.bit_generator.state,
        }
        if include_replay:
            state["replay"] = self.replay.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in ("actor", "critic1", "critic2", "target1", "target2"):
            getattr(self, name).load_state_dict(state[name])
        self.anchor_actor.load_state_dict(state.get("anchor_actor", state["actor"]))
        self.reference_actor.load_state_dict(
            state.get("reference_actor", state["actor"])
        )
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.total_updates = int(state["total_updates"])
        self.rng.bit_generator.state = state["rng_state"]
        if "replay" in state:
            self.replay.load_state_dict(state["replay"])


class PPO(ClassicalAlgorithm):
    algorithm = "ppo"

    def __init__(self, state_dim: int, protocol: ComparisonProtocol, device: str, seed: int) -> None:
        super().__init__(state_dim, protocol, device, seed)
        self.actor = SquashedGaussianActor(
            state_dim,
            ACTION_DIM,
            state_dependent_std=False,
            initial_log_std=-1.25,
            mean_limit=5.0,
        ).to(self.device)
        self.anchor_actor = SquashedGaussianActor(
            state_dim,
            ACTION_DIM,
            state_dependent_std=False,
            initial_log_std=-1.25,
            mean_limit=5.0,
        ).to(self.device)
        self.anchor_actor.load_state_dict(self.actor.state_dict())
        self.anchor_actor.requires_grad_(False)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()), lr=3e-4
        )
        self.rollout_size = 2_048
        self.epochs = 10
        self.clip_ratio = 0.2
        self.gae_lambda = 0.95
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.target_kl: float | None = None
        self.actor_learning_starts = 0
        self.anchor_coefficient = 0.0
        self.anchor_decay_steps = 400_000
        self.rollout: list[dict[str, Any]] = []

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool, total_steps: int) -> ActionSample:
        if not deterministic:
            self.observe_state(state)
        normalized = self.normalizer.normalize(state)
        tensor = torch.as_tensor(normalized, device=self.device).unsqueeze(0)
        sampled, _, mean = self.actor.sample(tensor)
        action = mean if deterministic else sampled
        # PPO later reconstructs the pre-tanh value from the bounded action.
        # Store the likelihood of that exact action so saturated means do not
        # create a spurious KL jump between collection and the first update.
        log_prob, _ = self.actor.log_prob(tensor, action)
        value = self.value(tensor)
        return ActionSample(
            action.squeeze(0).cpu().numpy().astype(np.float32),
            {
                "normalized_state": normalized,
                "log_prob": float(log_prob.squeeze().cpu()),
                "value": float(value.squeeze().cpu()),
            },
        )

    @torch.no_grad()
    def record(self, state: np.ndarray, sample: ActionSample, reward: float, next_state: np.ndarray, done: bool) -> None:
        del state
        normalized_next = self.normalizer.normalize(next_state)
        if done:
            next_value = 0.0
        else:
            tensor = torch.as_tensor(
                normalized_next, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            next_value = float(self.value(tensor).squeeze().cpu())
        self.rollout.append(
            {
                "state": sample.auxiliary["normalized_state"],
                "action": sample.action,
                "reward": float(reward) * self.protocol.reward_scale,
                "continue": float(not done),
                "log_prob": sample.auxiliary["log_prob"],
                "value": sample.auxiliary["value"],
                "next_value": next_value,
            }
        )

    def update_if_ready(self, total_steps: int, force: bool = False) -> dict[str, float]:
        if len(self.rollout) < self.rollout_size and not (force and self.rollout):
            return {}
        update_actor = total_steps >= self.actor_learning_starts
        records = self.rollout
        self.rollout = []
        states = np.stack([record["state"] for record in records]).astype(np.float32)
        actions = np.stack([record["action"] for record in records]).astype(np.float32)
        rewards = np.asarray([record["reward"] for record in records], np.float32)
        continues = np.asarray([record["continue"] for record in records], np.float32)
        old_log_probs = np.asarray([record["log_prob"] for record in records], np.float32)
        values = np.asarray([record["value"] for record in records], np.float32)
        next_values = np.asarray([record["next_value"] for record in records], np.float32)
        deltas = rewards + self.protocol.discount * continues * next_values - values
        advantages = np.zeros_like(deltas)
        running = 0.0
        for index in reversed(range(len(records))):
            running = deltas[index] + (
                self.protocol.discount
                * self.gae_lambda
                * continues[index]
                * running
            )
            advantages[index] = running
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        state_tensor = torch.as_tensor(states, device=self.device)
        action_tensor = torch.as_tensor(actions, device=self.device)
        old_log_tensor = torch.as_tensor(old_log_probs, device=self.device)[:, None]
        advantage_tensor = torch.as_tensor(advantages, device=self.device)[:, None]
        return_tensor = torch.as_tensor(returns, device=self.device)[:, None]
        losses: dict[str, list[float]] = {
            "actor": [], "anchor": [], "value": [], "entropy": [], "kl": []
        }
        anchor_weight = self.anchor_coefficient * max(
            0.0, 1.0 - total_steps / max(self.anchor_decay_steps, 1)
        )
        indices = np.arange(len(records))
        stop_for_kl = False
        actor_batches = 0
        for _ in range(self.epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.protocol.batch_size):
                batch = indices[start : start + self.protocol.batch_size]
                actor_loss = torch.zeros((), device=self.device)
                anchor_loss = torch.zeros((), device=self.device)
                entropy_mean = torch.zeros((), device=self.device)
                approximate_kl = torch.zeros((), device=self.device)
                update_actor_batch = update_actor and not stop_for_kl
                if update_actor_batch:
                    log_prob, entropy = self.actor.log_prob(
                        state_tensor[batch], action_tensor[batch]
                    )
                    log_ratio = log_prob - old_log_tensor[batch]
                    ratio = log_ratio.exp()
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    if (
                        self.target_kl is not None
                        and float(approximate_kl.detach())
                        > self.target_kl * 1.5
                    ):
                        stop_for_kl = True
                        update_actor_batch = False
                    else:
                        unclipped = ratio * advantage_tensor[batch]
                        clipped = ratio.clamp(
                            1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                        ) * advantage_tensor[batch]
                        actor_loss = -torch.minimum(unclipped, clipped).mean()
                        entropy_mean = entropy.mean()
                        current_distribution = self.actor.distribution(
                            state_tensor[batch]
                        )
                        with torch.no_grad():
                            anchor_distribution = self.anchor_actor.distribution(
                                state_tensor[batch]
                            )
                        anchor_loss = kl_divergence(
                            current_distribution, anchor_distribution
                        ).sum(-1).mean()
                        actor_batches += 1
                value_prediction = self.value(state_tensor[batch])
                value_loss = F.mse_loss(value_prediction, return_tensor[batch])
                loss = (
                    actor_loss
                    + anchor_weight * anchor_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._clip(
                    list(self.actor.parameters()) + list(self.value.parameters()),
                    0.5,
                )
                self.optimizer.step()
                losses["actor"].append(float(actor_loss.detach()))
                losses["anchor"].append(float(anchor_loss.detach()))
                losses["value"].append(float(value_loss.detach()))
                losses["entropy"].append(float(entropy_mean.detach()))
                losses["kl"].append(float(approximate_kl.detach()))
        self.total_updates += 1
        return {
            "loss/actor": float(np.mean(losses["actor"])),
            "loss/anchor": float(np.mean(losses["anchor"])),
            "loss/value": float(np.mean(losses["value"])),
            "train/actor_updated": float(actor_batches > 0),
            "train/anchor_coefficient": (
                float(anchor_weight) if actor_batches > 0 else 0.0
            ),
            "train/entropy": float(np.mean(losses["entropy"])),
            "train/approx_kl": float(np.mean(losses["kl"])),
            "train/kl_stopped": float(stop_for_kl),
        }

    def state_dict(self, include_replay: bool = True) -> dict[str, Any]:
        del include_replay
        return {
            "actor": self.actor.state_dict(),
            "anchor_actor": self.anchor_actor.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "total_updates": self.total_updates,
            "rng_state": self.rng.bit_generator.state,
            "rollout": self.rollout,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.anchor_actor.load_state_dict(state.get("anchor_actor", state["actor"]))
        self.value.load_state_dict(state["value"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.total_updates = int(state["total_updates"])
        self.rng.bit_generator.state = state["rng_state"]
        self.rollout = list(state.get("rollout", []))


ALGORITHMS = {"ddpg": DDPG, "ppo": PPO, "sac": SAC}


def create_algorithm(
    name: str,
    state_dim: int,
    protocol: ComparisonProtocol,
    device: str,
    seed: int,
) -> ClassicalAlgorithm:
    try:
        algorithm_type = ALGORITHMS[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown classical algorithm: {name}") from error
    return algorithm_type(state_dim, protocol, device, seed)

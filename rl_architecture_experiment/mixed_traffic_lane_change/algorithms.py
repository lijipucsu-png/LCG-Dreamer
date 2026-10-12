"""Native focused lateral learners for the prediction-augmented baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from classical_comparison.models import (
    DeterministicActor,
    QNetwork,
    SquashedGaussianActor,
    ValueNetwork,
)

from .protocol import MixedLaneProtocol


NODE_COUNT = 6
NODE_FEATURE_DIM = 32
STATE_DIM = NODE_COUNT * NODE_FEATURE_DIM
ACTION_DIM = 11


def _masked_imitation_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Match Graph BC semantics: lateral intent is supervised only when valid."""

    lane_weights = (target[:, :NODE_COUNT] > 0.0).to(prediction.dtype)
    shape_weights = torch.ones_like(target[:, NODE_COUNT:])
    weights = torch.cat((lane_weights, shape_weights), dim=-1)
    squared_error = (prediction - target).square() * weights
    return squared_error.sum() / weights.sum().clamp_min(1.0)


@dataclass
class NativeActionSample:
    action: np.ndarray
    log_prob: float = 0.0
    value: float = 0.0
    entropy: float = 0.0


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.states = np.empty((capacity, STATE_DIM), np.float32)
        self.actions = np.empty((capacity, ACTION_DIM), np.float32)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.next_states = np.empty((capacity, STATE_DIM), np.float32)
        self.continues = np.empty((capacity, 1), np.float32)
        self.expert_targets = np.empty((capacity, ACTION_DIM), np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def append_episode(self, transitions: list[dict[str, Any]]) -> None:
        for transition in transitions:
            index = self.position
            self.states[index] = transition["state"]
            self.actions[index] = transition["action_unit"]
            self.rewards[index, 0] = transition["reward"]
            self.next_states[index] = transition["next_state"]
            self.continues[index, 0] = float(not transition["done"])
            self.expert_targets[index] = transition["expert_target"]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        index = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.continues[index],
            self.expert_targets[index],
        )

    def state_dict(self) -> dict[str, Any]:
        if self.size < self.capacity:
            index = np.arange(self.size)
        else:
            index = np.concatenate(
                (np.arange(self.position, self.capacity), np.arange(self.position))
            )
        return {
            "states": self.states[index].astype(np.float16),
            "actions": self.actions[index].astype(np.float16),
            "rewards": self.rewards[index],
            "next_states": self.next_states[index].astype(np.float16),
            "continues": self.continues[index],
            "expert_targets": self.expert_targets[index].astype(np.float16),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        size = min(len(state["states"]), self.capacity)
        self.states[:size] = np.asarray(state["states"], np.float32)[-size:]
        self.actions[:size] = np.asarray(state["actions"], np.float32)[-size:]
        self.rewards[:size] = np.asarray(state["rewards"], np.float32)[-size:]
        self.next_states[:size] = np.asarray(
            state["next_states"], np.float32
        )[-size:]
        self.continues[:size] = np.asarray(
            state["continues"], np.float32
        )[-size:]
        self.expert_targets[:size] = np.asarray(
            state["expert_targets"], np.float32
        )[-size:]
        self.size = size
        self.position = size % self.capacity
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


class DelaySwitchingLaneActor(nn.Module):
    """SAC actor with separate 0/100/200 ms lateral experts."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                SquashedGaussianActor(
                    STATE_DIM, ACTION_DIM, state_dependent_std=True
                )
                for _ in range(3)
            ]
        )
        with torch.no_grad():
            for expert in self.experts:
                expert.network[-1].bias[ACTION_DIM:].fill_(-2.0)

    @staticmethod
    def _mode(state: Tensor) -> Tensor:
        nodes = state.reshape(*state.shape[:-1], NODE_COUNT, NODE_FEATURE_DIM)
        age = nodes[..., 20:24].amax(dim=(-1, -2))
        return torch.where(
            age < 0.25,
            torch.zeros_like(age, dtype=torch.long),
            torch.where(
                age < 0.75,
                torch.ones_like(age, dtype=torch.long),
                torch.full_like(age, 2, dtype=torch.long),
            ),
        )

    def distribution(self, state: Tensor) -> torch.distributions.Normal:
        distributions = [expert.distribution(state) for expert in self.experts]
        locations = torch.stack([item.loc for item in distributions], -2)
        scales = torch.stack([item.scale for item in distributions], -2)
        mode = self._mode(state)
        selector = F.one_hot(mode, num_classes=3).to(locations.dtype)
        selector = selector[..., :, None]
        return torch.distributions.Normal(
            (locations * selector).sum(-2),
            (scales * selector).sum(-2),
        )

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(state)
        raw = distribution.rsample()
        unit = raw.tanh().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_prob = SquashedGaussianActor._squashed_log_prob_from_raw(
            distribution, raw
        )
        return unit, log_prob, distribution.mean.tanh()

    def deterministic(self, state: Tensor) -> Tensor:
        return self.distribution(state).mean.tanh()


class NativeLaneAlgorithm:
    algorithm = "base"

    def __init__(self, protocol: MixedLaneProtocol, device: str, seed: int) -> None:
        self.protocol = protocol
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.episodes_updated = 0

    def imitation_prior_weight(self) -> float:
        return float(
            self.protocol.imitation_prior_initial_weight
            * np.exp(
                -self.episodes_updated
                / self.protocol.imitation_prior_decay_episodes
            )
        )

    @staticmethod
    def tensor(values: np.ndarray, device: torch.device) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def act(self, state: np.ndarray, deterministic: bool) -> NativeActionSample:
        raise NotImplementedError

    def behavior_clone(
        self, states: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        raise NotImplementedError

    def update_episode(
        self, transitions: list[dict[str, Any]]
    ) -> dict[str, float]:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.lerp_(parameter, tau)


class DDPGLaneAlgorithm(NativeLaneAlgorithm):
    algorithm = "ddpg_p"

    def __init__(self, protocol: MixedLaneProtocol, device: str, seed: int) -> None:
        super().__init__(protocol, device, seed)
        self.actor = DeterministicActor(STATE_DIM, ACTION_DIM).to(self.device)
        self.target_actor = DeterministicActor(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic = QNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.target_critic = QNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=protocol.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=protocol.learning_rate
        )
        self.replay = ReplayBuffer(protocol.replay_capacity, seed)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool) -> NativeActionSample:
        tensor = self.tensor(state, self.device).unsqueeze(0)
        action = self.actor(tensor).squeeze(0).cpu().numpy()
        if not deterministic:
            action += self.rng.normal(0.0, 0.10, ACTION_DIM)
        return NativeActionSample(np.clip(action, -1.0, 1.0).astype(np.float32))

    def behavior_clone(
        self, states: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        state = self.tensor(states, self.device)
        target = self.tensor(targets, self.device)
        losses = []
        for _ in range(self.protocol.imitation_epochs):
            loss = _masked_imitation_loss(self.actor(state), target)
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.protocol.max_grad_norm)
            self.actor_optimizer.step()
            losses.append(float(loss.detach()))
        self.target_actor.load_state_dict(self.actor.state_dict())
        return {"imitation_loss": float(np.mean(losses))}

    def update_episode(
        self, transitions: list[dict[str, Any]]
    ) -> dict[str, float]:
        self.episodes_updated += 1
        prior_weight = self.imitation_prior_weight()
        self.replay.append_episode(transitions)
        if self.replay.size < self.protocol.minibatch_size:
            return {}
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        updates = min(len(transitions), 256)
        for _ in range(updates):
            states, actions, rewards, next_states, continues, expert_targets = self.replay.sample(
                self.protocol.minibatch_size
            )
            state = self.tensor(states, self.device)
            action = self.tensor(actions, self.device)
            reward = self.tensor(rewards, self.device) * 0.1
            next_state = self.tensor(next_states, self.device)
            cont = self.tensor(continues, self.device)
            expert_target = self.tensor(expert_targets, self.device)
            with torch.no_grad():
                target = reward + self.protocol.discount * cont * self.target_critic(
                    next_state, self.target_actor(next_state)
                )
            q = self.critic(state, action)
            critic_loss = F.smooth_l1_loss(q, target)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.protocol.max_grad_norm)
            self.critic_optimizer.step()
            actor_prediction = self.actor(state)
            prior_loss = _masked_imitation_loss(actor_prediction, expert_target)
            actor_loss = (
                -self.critic(state, actor_prediction).mean()
                + prior_weight * prior_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.protocol.max_grad_norm)
            self.actor_optimizer.step()
            self.soft_update(self.target_actor, self.actor, 0.01)
            self.soft_update(self.target_critic, self.critic, 0.01)
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
        return {
            "loss_actor": float(np.mean(actor_losses)),
            "loss_critic": float(np.mean(critic_losses)),
            "imitation_prior_weight": prior_weight,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "episodes_updated": self.episodes_updated,
        }


class PPOLaneAlgorithm(NativeLaneAlgorithm):
    algorithm = "ppo_p"

    def __init__(self, protocol: MixedLaneProtocol, device: str, seed: int) -> None:
        super().__init__(protocol, device, seed)
        self.actor = SquashedGaussianActor(
            STATE_DIM,
            ACTION_DIM,
            state_dependent_std=False,
            initial_log_std=-2.0,
        ).to(self.device)
        self.value = ValueNetwork(STATE_DIM).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()),
            lr=protocol.learning_rate,
        )

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool) -> NativeActionSample:
        tensor = self.tensor(state, self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(tensor)
            log_prob, entropy = self.actor.log_prob(tensor, action)
        else:
            action, log_prob, _ = self.actor.sample(tensor)
            _, entropy = self.actor.log_prob(tensor, action)
        value = self.value(tensor)
        return NativeActionSample(
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
            float(entropy.item()),
        )

    def behavior_clone(
        self, states: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        state = self.tensor(states, self.device)
        target = self.tensor(targets, self.device)
        losses = []
        for _ in range(self.protocol.imitation_epochs):
            loss = _masked_imitation_loss(
                self.actor.deterministic(state), target
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.protocol.max_grad_norm)
            self.optimizer.step()
            losses.append(float(loss.detach()))
        return {"imitation_loss": float(np.mean(losses))}

    def update_episode(
        self, transitions: list[dict[str, Any]]
    ) -> dict[str, float]:
        self.episodes_updated += 1
        prior_weight = self.imitation_prior_weight()
        rewards = np.asarray([item["reward"] for item in transitions], np.float32)
        values = np.asarray([item["value"] for item in transitions], np.float32)
        dones = np.asarray([item["done"] for item in transitions], np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for index in reversed(range(len(transitions))):
            continuation = 1.0 - dones[index]
            delta = rewards[index] + self.protocol.discount * next_value * continuation - values[index]
            gae = delta + self.protocol.discount * self.protocol.gae_lambda * continuation * gae
            advantages[index] = gae
            next_value = values[index]
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
        states = self.tensor(np.stack([item["state"] for item in transitions]), self.device)
        actions = self.tensor(np.stack([item["action_unit"] for item in transitions]), self.device)
        old_log_prob = self.tensor(np.asarray([item["log_prob"] for item in transitions], np.float32), self.device)
        advantage = self.tensor(advantages, self.device)
        target_return = self.tensor(returns, self.device)
        expert_target = self.tensor(
            np.stack([item["expert_target"] for item in transitions]),
            self.device,
        )
        losses: list[float] = []
        for _ in range(self.protocol.ppo_epochs):
            log_prob, entropy = self.actor.log_prob(states, actions)
            log_prob = log_prob.squeeze(-1)
            ratio = (log_prob - old_log_prob).exp()
            policy_loss = -torch.minimum(
                ratio * advantage,
                ratio.clamp(1.0 - self.protocol.clip_ratio, 1.0 + self.protocol.clip_ratio) * advantage,
            ).mean()
            value_loss = 0.5 * (self.value(states).squeeze(-1) - target_return).pow(2).mean()
            prior_loss = _masked_imitation_loss(
                self.actor.deterministic(states), expert_target
            )
            loss = (
                policy_loss
                + self.protocol.value_weight * value_loss
                - self.protocol.entropy_weight * entropy.mean()
                + prior_weight * prior_loss
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.value.parameters()),
                self.protocol.max_grad_norm,
            )
            self.optimizer.step()
            losses.append(float(loss.detach()))
        return {
            "ppo_loss": float(np.mean(losses)),
            "imitation_prior_weight": prior_weight,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_updated": self.episodes_updated,
        }


class SACLaneAlgorithm(NativeLaneAlgorithm):
    algorithm = "sac_p"

    def __init__(
        self,
        protocol: MixedLaneProtocol,
        device: str,
        seed: int,
        knowledge_guided: bool = False,
    ) -> None:
        super().__init__(protocol, device, seed)
        self.knowledge_guided = bool(knowledge_guided)
        self.actor: nn.Module = (
            DelaySwitchingLaneActor()
            if self.knowledge_guided
            else SquashedGaussianActor(
                STATE_DIM, ACTION_DIM, state_dependent_std=True
            )
        )
        self.actor.to(self.device)
        if not self.knowledge_guided:
            with torch.no_grad():
                self.actor.network[-1].bias[ACTION_DIM:].fill_(-2.0)
        self.q1 = QNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.q2 = QNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.target1 = QNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.target2 = QNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.target1.load_state_dict(self.q1.state_dict())
        self.target2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=protocol.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=protocol.learning_rate,
        )
        self.replay = ReplayBuffer(protocol.replay_capacity, seed)
        self.alpha = 0.05

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool) -> NativeActionSample:
        tensor = self.tensor(state, self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(tensor)
            log_prob = torch.zeros(1, 1, device=self.device)
        else:
            action, log_prob, _ = self.actor.sample(tensor)
        return NativeActionSample(
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
        )

    def behavior_clone(
        self, states: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        state = self.tensor(states, self.device)
        target = self.tensor(targets, self.device)
        losses = []
        for _ in range(self.protocol.imitation_epochs):
            loss = _masked_imitation_loss(
                self.actor.deterministic(state), target
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.protocol.max_grad_norm)
            self.actor_optimizer.step()
            losses.append(float(loss.detach()))
        return {"imitation_loss": float(np.mean(losses))}

    def update_episode(
        self, transitions: list[dict[str, Any]]
    ) -> dict[str, float]:
        self.episodes_updated += 1
        prior_weight = self.imitation_prior_weight()
        self.replay.append_episode(transitions)
        if self.replay.size < self.protocol.minibatch_size:
            return {}
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        updates = min(len(transitions), 256)
        for _ in range(updates):
            states, actions, rewards, next_states, continues, expert_targets = self.replay.sample(
                self.protocol.minibatch_size
            )
            state = self.tensor(states, self.device)
            action = self.tensor(actions, self.device)
            reward = self.tensor(rewards, self.device) * 0.1
            next_state = self.tensor(next_states, self.device)
            cont = self.tensor(continues, self.device)
            expert_target = self.tensor(expert_targets, self.device)
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                target_q = torch.minimum(
                    self.target1(next_state, next_action),
                    self.target2(next_state, next_action),
                ) - self.alpha * next_log_prob
                target = reward + self.protocol.discount * cont * target_q
            q1 = self.q1(state, action)
            q2 = self.q2(state, action)
            critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.protocol.max_grad_norm,
            )
            self.critic_optimizer.step()
            sampled, log_prob, _ = self.actor.sample(state)
            actor_loss = (
                self.alpha * log_prob
                - torch.minimum(self.q1(state, sampled), self.q2(state, sampled))
            ).mean() + prior_weight * _masked_imitation_loss(
                self.actor.deterministic(state), expert_target
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.protocol.max_grad_norm)
            self.actor_optimizer.step()
            self.soft_update(self.target1, self.q1, 0.01)
            self.soft_update(self.target2, self.q2, 0.01)
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
        return {
            "loss_actor": float(np.mean(actor_losses)),
            "loss_critic": float(np.mean(critic_losses)),
            "imitation_prior_weight": prior_weight,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "knowledge_guided": self.knowledge_guided,
            "episodes_updated": self.episodes_updated,
        }


def create_native_algorithm(
    method: str, protocol: MixedLaneProtocol, device: str, seed: int
) -> NativeLaneAlgorithm:
    if method == "ddpg_p":
        return DDPGLaneAlgorithm(protocol, device, seed)
    if method == "ppo_p":
        return PPOLaneAlgorithm(protocol, device, seed)
    if method == "sac_p":
        return SACLaneAlgorithm(protocol, device, seed)
    if method == "nc_kg_sac_p":
        algorithm = SACLaneAlgorithm(
            protocol, device, seed, knowledge_guided=True
        )
        algorithm.algorithm = "nc_kg_sac_p"
        return algorithm
    raise ValueError(f"no native mixed-lane algorithm for {method}")


def load_native_algorithm(
    method: str,
    protocol: MixedLaneProtocol,
    device: str,
    state: Mapping[str, Any],
) -> NativeLaneAlgorithm:
    algorithm = create_native_algorithm(method, protocol, device, protocol.seed)
    if isinstance(algorithm, DDPGLaneAlgorithm):
        for name in ("actor", "target_actor", "critic", "target_critic"):
            getattr(algorithm, name).load_state_dict(state[name])
    elif isinstance(algorithm, PPOLaneAlgorithm):
        algorithm.actor.load_state_dict(state["actor"])
        algorithm.value.load_state_dict(state["value"])
    elif isinstance(algorithm, SACLaneAlgorithm):
        for name in ("actor", "q1", "q2", "target1", "target2"):
            getattr(algorithm, name).load_state_dict(state[name])
    algorithm.episodes_updated = int(state.get("episodes_updated", 0))
    return algorithm


def restore_native_algorithm(
    method: str,
    protocol: MixedLaneProtocol,
    device: str,
    seed: int,
    state: Mapping[str, Any],
) -> NativeLaneAlgorithm:
    """Restore model, optimizer, target, and replay state for training."""

    algorithm = create_native_algorithm(method, protocol, device, seed)
    if isinstance(algorithm, DDPGLaneAlgorithm):
        for name in ("actor", "target_actor", "critic", "target_critic"):
            getattr(algorithm, name).load_state_dict(state[name])
        algorithm.actor_optimizer.load_state_dict(state["actor_optimizer"])
        algorithm.critic_optimizer.load_state_dict(state["critic_optimizer"])
        algorithm.replay.load_state_dict(state["replay"])
    elif isinstance(algorithm, PPOLaneAlgorithm):
        algorithm.actor.load_state_dict(state["actor"])
        algorithm.value.load_state_dict(state["value"])
        algorithm.optimizer.load_state_dict(state["optimizer"])
    elif isinstance(algorithm, SACLaneAlgorithm):
        for name in ("actor", "q1", "q2", "target1", "target2"):
            getattr(algorithm, name).load_state_dict(state[name])
        algorithm.actor_optimizer.load_state_dict(state["actor_optimizer"])
        algorithm.critic_optimizer.load_state_dict(state["critic_optimizer"])
        algorithm.replay.load_state_dict(state["replay"])
    algorithm.episodes_updated = int(state.get("episodes_updated", 0))
    return algorithm

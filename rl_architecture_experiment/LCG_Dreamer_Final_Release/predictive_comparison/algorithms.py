"""Prediction-enabled algorithms and the NC knowledge-guided SAC baseline."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal
import torch.nn.functional as F

from classical_comparison.algorithms import (
    ACTION_DIM,
    ClassicalAlgorithm,
    DDPG,
    PPO,
    SAC,
)
from classical_comparison.models import SquashedGaussianActor

from .protocol import PredictiveComparisonProtocol


class DelaySwitchingActor(nn.Module):
    """Three SAC experts selected per vehicle for 0/100/200 ms packet age."""

    def __init__(self, state_dim: int, gate_dimension: int = ACTION_DIM * 3) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.gate_dimension = int(gate_dimension)
        self.experts = nn.ModuleList(
            [
                SquashedGaussianActor(
                    state_dim, ACTION_DIM, state_dependent_std=True
                )
                for _ in range(3)
            ]
        )

    def distribution(self, state: Tensor) -> Normal:
        distributions = [expert.distribution(state) for expert in self.experts]
        locations = torch.stack([item.loc for item in distributions], -1)
        scales = torch.stack([item.scale for item in distributions], -1)
        gate = state[..., -self.gate_dimension :].reshape(
            *state.shape[:-1], ACTION_DIM, 3
        )
        gate = gate.clamp_min(0.0)
        total = gate.sum(-1, keepdim=True)
        fallback = torch.zeros_like(gate)
        fallback[..., 0] = 1.0
        gate = torch.where(
            total > 1e-6,
            gate / total.clamp_min(1e-6),
            fallback,
        )
        return Normal((locations * gate).sum(-1), (scales * gate).sum(-1))

    def deterministic(self, state: Tensor) -> Tensor:
        return self.distribution(state).mean.tanh()

    @staticmethod
    def _squashed_log_prob_from_raw(distribution: Normal, raw: Tensor) -> Tensor:
        return SquashedGaussianActor._squashed_log_prob_from_raw(distribution, raw)


class NCKnowledgeGuidedSAC(SAC):
    """Adaptation of Wang et al. (2025) with delay-conditioned SAC experts."""

    algorithm = "nc_kg_sac_p"

    def __init__(
        self,
        state_dim: int,
        protocol: PredictiveComparisonProtocol,
        device: str,
        seed: int,
    ) -> None:
        super().__init__(state_dim, protocol, device, seed)
        gate_dimension = ACTION_DIM * 3
        self.actor = DelaySwitchingActor(state_dim, gate_dimension).to(self.device)
        self.anchor_actor = DelaySwitchingActor(state_dim, gate_dimension).to(self.device)
        self.reference_actor = DelaySwitchingActor(state_dim, gate_dimension).to(
            self.device
        )
        self.anchor_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.anchor_actor.requires_grad_(False)
        self.reference_actor.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.gate_dimension = gate_dimension

    def normalized_tensor(self, state: np.ndarray) -> Tensor:
        tensor = super().normalized_tensor(state)
        raw = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        tensor[..., -self.gate_dimension :] = raw[..., -self.gate_dimension :]
        return tensor


class CanonicalPredictiveDDPG(DDPG):
    """Canonical single-critic DDPG update with the shared predictive state."""

    algorithm = "ddpg_p"
    implementation_variant = "canonical_single_critic"

    def update_if_ready(
        self, total_steps: int, force: bool = False
    ) -> dict[str, float]:
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
            target = reward + self.protocol.discount * cont * self.critic_target(
                next_state, target_action
            )
        q = self.critic(state, action)
        critic_loss = F.smooth_l1_loss(q, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(self.critic.parameters())
        self.critic_optimizer.step()

        actor_loss = torch.zeros((), device=self.device)
        self.total_updates += 1
        actor_updated = total_steps >= self.actor_learning_starts
        if actor_updated:
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip(self.actor.parameters())
            self.actor_optimizer.step()
            self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)
        return {
            "loss/actor": float(actor_loss.detach()),
            "loss/critic": float(critic_loss.detach()),
            "train/actor_updated": float(actor_updated),
            "train/q": float(q.detach().mean()),
            "train/target_abs_max": float(target.detach().abs().max()),
        }


def create_predictive_algorithm(
    method: str,
    state_dim: int,
    protocol: PredictiveComparisonProtocol,
    device: str,
    seed: int,
) -> ClassicalAlgorithm:
    if method == "ddpg_p":
        return CanonicalPredictiveDDPG(state_dim, protocol, device, seed)
    if method == "ppo_p":
        return PPO(state_dim, protocol, device, seed)
    if method == "sac_p":
        return SAC(state_dim, protocol, device, seed)
    if method == "nc_kg_sac_p":
        return NCKnowledgeGuidedSAC(state_dim, protocol, device, seed)
    raise ValueError(f"unknown predictive algorithm: {method}")


def _expanded_state_dict(
    module: nn.Module,
    source: Mapping[str, Tensor],
    old_state_dim: int,
    q_network: bool = False,
) -> dict[str, Tensor]:
    target = module.state_dict()
    for key, value in target.items():
        source_value = source.get(key)
        if source_value is None:
            continue
        source_value = source_value.to(dtype=value.dtype)
        if source_value.shape == value.shape:
            target[key] = source_value.clone()
            continue
        if key == "network.0.weight" and value.ndim == source_value.ndim == 2:
            expanded = torch.zeros_like(value)
            expanded[:, :old_state_dim] = source_value[:, :old_state_dim]
            if q_network:
                expanded[:, -ACTION_DIM:] = source_value[:, -ACTION_DIM:]
            target[key] = expanded
    return target


def _load_expanded(
    module: nn.Module,
    source: Mapping[str, Tensor],
    old_state_dim: int,
    q_network: bool = False,
) -> None:
    module.load_state_dict(
        _expanded_state_dict(module, source, old_state_dim, q_network)
    )


def initialize_from_classical_checkpoint(
    algorithm: ClassicalAlgorithm,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate a capable policy while zero-initializing new prediction columns."""

    source = checkpoint["algorithm_state"]
    old_state_dim = int(checkpoint["state_dim"])
    if isinstance(algorithm, NCKnowledgeGuidedSAC):
        for expert in algorithm.actor.experts:
            _load_expanded(expert, source["actor"], old_state_dim)
        algorithm.anchor_actor.load_state_dict(algorithm.actor.state_dict())
        algorithm.reference_actor.load_state_dict(algorithm.actor.state_dict())
    else:
        _load_expanded(algorithm.actor, source["actor"], old_state_dim)
        if hasattr(algorithm, "anchor_actor"):
            algorithm.anchor_actor.load_state_dict(algorithm.actor.state_dict())
        if hasattr(algorithm, "reference_actor"):
            algorithm.reference_actor.load_state_dict(algorithm.actor.state_dict())
        if hasattr(algorithm, "actor_target"):
            algorithm.actor_target.load_state_dict(algorithm.actor.state_dict())

    for name in (
        "critic",
        "critic2",
        "critic_target",
        "critic2_target",
        "critic1",
        "target1",
        "target2",
    ):
        if hasattr(algorithm, name) and name in source:
            _load_expanded(
                getattr(algorithm, name), source[name], old_state_dim, q_network=True
            )
    if hasattr(algorithm, "value") and "value" in source:
        _load_expanded(algorithm.value, source["value"], old_state_dim)

    normalizer = source["normalizer"]
    mean = np.zeros(algorithm.state_dim, np.float64)
    variance = np.ones(algorithm.state_dim, np.float64)
    mean[:old_state_dim] = np.asarray(normalizer["mean"], np.float64)
    variance[:old_state_dim] = np.asarray(normalizer["variance"], np.float64)
    algorithm.normalizer.mean = mean
    algorithm.normalizer.variance = variance
    algorithm.normalizer.count = float(normalizer["count"])
    algorithm.normalizer.clip = float(normalizer["clip"])
    algorithm.normalizer_frozen = True

    # Existing value estimates refer to a state without prediction channels.
    # Fresh optimizers and replay avoid carrying stale moments/transitions.
    if isinstance(algorithm, DDPG):
        algorithm.actor_learning_starts = 5_000
        algorithm.learning_starts = 2_000
        algorithm.exploration_initial = 0.08
        algorithm.exploration_final = 0.02
        for group in algorithm.actor_optimizer.param_groups:
            group["lr"] = 5e-6
        for group in algorithm.critic_optimizer.param_groups:
            group["lr"] = 1e-4
    elif isinstance(algorithm, PPO):
        algorithm.actor_learning_starts = 0
        algorithm.epochs = 2
        algorithm.clip_ratio = 0.05
        algorithm.target_kl = 0.01
        algorithm.anchor_coefficient = 0.1
        for group in algorithm.optimizer.param_groups:
            group["lr"] = 1e-5
    elif isinstance(algorithm, SAC):
        algorithm.actor_learning_starts = 5_000
        algorithm.learning_starts = 2_000
        algorithm.policy_delay = 2
        algorithm.behavior_noise_scale = 0.25
        algorithm.maximum_alpha = 0.10
        algorithm.reference_policy_coefficient = 1.0
        algorithm.normalize_actor_q = True
        for group in algorithm.actor_optimizer.param_groups:
            group["lr"] = 1e-5
        for group in algorithm.critic_optimizer.param_groups:
            group["lr"] = 5e-5
    return {
        "source_algorithm": checkpoint["algorithm"],
        "source_state_dim": old_state_dim,
        "target_state_dim": algorithm.state_dim,
        "new_input_columns_initialized_to_zero": True,
        "source_replay_migrated": False,
        "source_optimizer_moments_migrated": False,
    }

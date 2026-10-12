"""Shared longitudinal SAC policy with centralized critics for followers."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

from platoon_sumo.env import FOLLOWER_IDS

from ..config import FollowerRLConfig


FOLLOWER_POLICY_STATE_DIM = 42
FOLLOWER_MEMORY_STEPS = 10
FOLLOWER_ACTION_DIM = 1
FOLLOWER_EGO_SPEED_INDEX = 1
FOLLOWER_PHYSICAL_FRONT_GAP_INDEX = 3
FOLLOWER_LOCAL_PREDECESSOR_ACCELERATION_INDEX = 5
FOLLOWER_DESIRED_GAP_INDEX = 6
FOLLOWER_RECOVERY_ROLE_INDEX = 16
FOLLOWER_RECOVERY_GAP_INDEX = 17
FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX = 23
FOLLOWER_PREDICTED_SPEED_INDEX = 35
FOLLOWER_PREDICTED_ACCELERATION_INDEX = 36
FOLLOWER_LOCAL_RELATIVE_SPEED_INDEX = 40
FOLLOWER_RECOVERY_TARGET_MAX_ACCELERATION = 0.6
FOLLOWER_RECOVERY_PREDICTED_ACCELERATION_WEIGHT = 1.0


def symlog(value: Tensor) -> Tensor:
    return value.sign() * torch.log1p(value.abs())


def symexp(value: Tensor) -> Tensor:
    return value.sign() * torch.expm1(value.abs())


@dataclass
class FollowerActionSample:
    acceleration: Tensor
    log_probability: Tensor

    @property
    def critic_action(self) -> Tensor:
        return self.acceleration


class SharedFollowerActor(nn.Module):
    """One decentralized policy shared by all follower vehicles."""

    def __init__(self, config: FollowerRLConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.accel_min = config.accel_min
        self.accel_max = config.accel_max
        self.residual_accel_scale = config.residual_accel_scale
        self.register_buffer(
            "state_scale",
            torch.tensor(
                [
                    3500.0,
                    35.0,
                    4.0,
                    60.0,
                    35.0,
                    4.0,
                    20.0,
                    0.02,
                    35.0,
                    1.0,
                    250.0,
                    35.0,
                    35.0,
                    4.0,
                    0.5,
                    5.0,
                    1.0,
                    500.0,
                    60.0,
                    1.0,
                    40.0,
                    240.0,
                    1.0,
                    4.0,
                    35.0,
                    35.0,
                    35.0,
                    35.0,
                    0.02,
                    8.0,
                    800.0,
                    35.0,
                    35.0,
                    15.0,
                    80.0,
                    35.0,
                    4.0,
                    6.0,
                    250.0,
                    10.0,
                    35.0,
                    10.0,
                ],
                dtype=torch.float32,
            ),
        )
        self.temporal = nn.GRU(
            FOLLOWER_POLICY_STATE_DIM, hidden, batch_first=True
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.follower_mean = nn.Linear(hidden, 1)
        self.follower_log_std = nn.Linear(hidden, 1)
        self.subleader_mean = nn.Linear(hidden, 1)
        self.subleader_log_std = nn.Linear(hidden, 1)
        with torch.no_grad():
            for mean_head, std_head in (
                (self.follower_mean, self.follower_log_std),
                (self.subleader_mean, self.subleader_log_std),
            ):
                mean_head.weight.zero_()
                # The actor learns a residual around CPIDLM's predicted
                # predecessor acceleration. Zero therefore means synchronous
                # feed-forward, not zero physical acceleration.
                mean_head.bias.zero_()
                std_head.weight.zero_()
                std_head.bias.fill_(-2.5)

    def normalize(self, states: Tensor) -> Tensor:
        return states / self.state_scale.clamp_min(1e-6)

    @property
    def acceleration_mean(self) -> nn.Linear:
        return self.follower_mean

    def _sequence(self, states: Tensor) -> tuple[Tensor, Tensor, tuple[int, ...]]:
        if states.shape[-1] != FOLLOWER_POLICY_STATE_DIM:
            raise ValueError(
                f"Follower state must end in {FOLLOWER_POLICY_STATE_DIM} features"
            )
        if states.ndim == 2:
            sequence = states.unsqueeze(-2)
            leading = states.shape[:-1]
        elif states.ndim == 3 and states.shape[0] == len(FOLLOWER_IDS) and states.shape[-2] == FOLLOWER_MEMORY_STEPS:
            sequence = states
            leading = states.shape[:-2]
        elif states.ndim == 3:
            leading = states.shape[:-1]
            sequence = states.reshape(-1, 1, states.shape[-1])
        elif states.ndim == 4:
            leading = states.shape[:-2]
            sequence = states.reshape(-1, states.shape[-2], states.shape[-1])
        else:
            raise ValueError("Follower actor states must have 2, 3, or 4 dimensions")
        role = states[..., -1, FOLLOWER_RECOVERY_ROLE_INDEX] if states.ndim in (3, 4) and states.shape[-2] == FOLLOWER_MEMORY_STEPS else states[..., FOLLOWER_RECOVERY_ROLE_INDEX]
        return sequence, role, leading

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        sequence, role, leading = self._sequence(states)
        normalized = self.normalize(sequence)
        _, hidden = self.temporal(normalized)
        encoded = self.encoder(hidden[-1]).reshape(*leading, -1)
        blend = role.clamp(0.0, 1.0).unsqueeze(-1)
        follower_mean = self.follower_mean(encoded)
        subleader_mean = self.subleader_mean(encoded)
        follower_std = self.follower_log_std(encoded)
        subleader_std = self.subleader_log_std(encoded)
        mean = ((1.0 - blend) * follower_mean + blend * subleader_mean).clamp(-5.0, 5.0)
        log_std = ((1.0 - blend) * follower_std + blend * subleader_std).clamp(-5.0, 1.0)
        return mean, log_std

    def sample(self, states: Tensor, deterministic: bool = False) -> FollowerActionSample:
        mean, log_std = self(states)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        normalized = torch.tanh(raw)
        if states.ndim >= 3 and states.shape[-2] == FOLLOWER_MEMORY_STEPS:
            current_states = states[..., -1, :]
        else:
            current_states = states
        gap_error = (
            current_states[
                ..., FOLLOWER_RECOVERY_GAP_INDEX : FOLLOWER_RECOVERY_GAP_INDEX + 1
            ]
            - current_states[..., 6:7]
        ).abs()
        relative_speed = current_states[
            ...,
            FOLLOWER_LOCAL_RELATIVE_SPEED_INDEX : FOLLOWER_LOCAL_RELATIVE_SPEED_INDEX
            + 1,
        ].abs()
        cutin_probability = current_states[..., 9:10].clamp(0.0, 1.0)
        recovery_role = current_states[
            ..., FOLLOWER_RECOVERY_ROLE_INDEX : FOLLOWER_RECOVERY_ROLE_INDEX + 1
        ].clamp(0.0, 1.0)
        leader_plan_acceleration = current_states[
            ...,
            FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX : FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX
            + 1,
        ].clamp(self.accel_min, self.accel_max)
        predecessor_acceleration = current_states[
            ...,
            FOLLOWER_LOCAL_PREDECESSOR_ACCELERATION_INDEX :
            FOLLOWER_LOCAL_PREDECESSOR_ACCELERATION_INDEX + 1,
        ].clamp(self.accel_min, self.accel_max)
        # A predecessor may add a learned recovery residual while the local gap
        # is still nominal. Keep the actor responsive to that V2V feed-forward
        # signal so recovery does not propagate one vehicle at a time.
        predecessor_plan_deviation = (
            predecessor_acceleration - leader_plan_acceleration
        ).abs()
        control_need = (
            gap_error / 4.0
            + relative_speed / 2.0
            + cutin_probability
            + recovery_role
            + predecessor_plan_deviation
        ).clamp(0.0, 1.0)
        residual = normalized * self.residual_accel_scale * control_need
        # CPIDLM predictions remain explicit actor inputs. They must not be a
        # hard execution baseline: an offline prediction bias would otherwise
        # become a self-reinforcing brake command as soon as recovery activates.
        acceleration = (leader_plan_acceleration + residual).clamp(
            self.accel_min, self.accel_max
        )
        continuous_log_probability = distribution.log_prob(raw)
        continuous_log_probability -= torch.log(
            self.residual_accel_scale * (1.0 - normalized.square()) + 1e-6
        )

        return FollowerActionSample(
            acceleration=acceleration,
            log_probability=continuous_log_probability,
        )


class CentralizedCritic(nn.Module):
    """Training-only critic over all follower states and joint accelerations."""

    def __init__(self, config: FollowerRLConfig, state_scale: Tensor) -> None:
        super().__init__()
        self.accel_min = config.accel_min
        self.accel_max = config.accel_max
        self.register_buffer("state_scale", state_scale.detach().clone())
        hidden = config.hidden_dim
        self.temporal = nn.GRU(
            FOLLOWER_POLICY_STATE_DIM, hidden, batch_first=True
        )
        inputs = len(FOLLOWER_IDS) * (hidden + FOLLOWER_ACTION_DIM)
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(FOLLOWER_IDS)),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        if states.ndim == 3:
            states = states.unsqueeze(-2)
        if states.ndim != 4:
            raise ValueError(
                "Centralized critic states must have shape [batch, agents, time, features]"
            )
        batch, agents, time, features = states.shape
        normalized_states = states / self.state_scale.clamp_min(1e-6)
        _, hidden = self.temporal(
            normalized_states.reshape(batch * agents, time, features)
        )
        encoded_states = hidden[-1].reshape(batch, agents, -1)
        normalized_acceleration = 2.0 * (
            actions[..., :1] - self.accel_min
        ) / (self.accel_max - self.accel_min) - 1.0
        joint = torch.cat((encoded_states, normalized_acceleration), dim=-1)
        return self.network(joint.flatten(-2))


class MultiAgentFollowerSAC(nn.Module):
    """Decentralized shared actor and twin centralized SAC critics."""

    def __init__(self, config: FollowerRLConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = SharedFollowerActor(config)
        self.critic1 = CentralizedCritic(config, self.actor.state_scale)
        self.critic2 = CentralizedCritic(config, self.actor.state_scale)
        self.target_critic1 = copy.deepcopy(self.critic1).requires_grad_(False)
        self.target_critic2 = copy.deepcopy(self.critic2).requires_grad_(False)

    def act(self, states: Tensor, deterministic: bool = True) -> FollowerActionSample:
        return self.actor.sample(states, deterministic)

    def critic_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        continues: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1).expand(-1, len(FOLLOWER_IDS))
        if continues.ndim == 1:
            continues = continues.unsqueeze(-1)
        with torch.no_grad():
            next_sample = self.actor.sample(next_states, deterministic=False)
            next_action = next_sample.critic_action
            next_entropy = next_sample.log_probability.squeeze(-1)
            next_q_symlog = torch.minimum(
                self.target_critic1(next_states, next_action),
                self.target_critic2(next_states, next_action),
            )
            next_q = symexp(next_q_symlog.clamp(-8.0, 8.0))
            target_value = self.config.reward_scale * rewards + self.config.gamma * continues * (
                next_q - self.config.entropy_coefficient * next_entropy
            )
            target = symlog(target_value)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        bellman_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        conservative1 = self._conservative_penalty(
            self.critic1, states, q1
        )
        conservative2 = self._conservative_penalty(
            self.critic2, states, q2
        )
        conservative_loss = conservative1 + conservative2
        loss = (
            bellman_loss
            + self.config.conservative_weight * conservative_loss
        )
        return loss, {
            "critic": loss,
            "bellman": bellman_loss,
            "conservative": conservative_loss,
            "q": 0.5 * (
                symexp(q1.clamp(-8.0, 8.0)).mean()
                + symexp(q2.clamp(-8.0, 8.0)).mean()
            ),
            "target_q": target_value.mean(),
            "q_symlog": 0.5 * (q1.mean() + q2.mean()),
            "target_q_symlog": target.mean(),
        }

    def _conservative_penalty(
        self,
        critic: CentralizedCritic,
        states: Tensor,
        data_q: Tensor,
        samples: int = 4,
    ) -> Tensor:
        batch, agents = states.shape[:2]
        expanded_states = states.unsqueeze(1).expand(
            batch, samples, *states.shape[1:]
        )
        random_actions = torch.empty(
            batch,
            samples,
            agents,
            FOLLOWER_ACTION_DIM,
            device=states.device,
            dtype=states.dtype,
        ).uniform_(self.config.accel_min, self.config.accel_max)
        random_q = critic(
            expanded_states.reshape(batch * samples, *states.shape[1:]),
            random_actions.reshape(batch * samples, agents, FOLLOWER_ACTION_DIM),
        ).reshape(batch, samples, agents)
        log_mean_exp = torch.logsumexp(random_q, dim=1) - math.log(samples)
        return F.softplus(log_mean_exp - data_q).mean()

    def actor_loss(
        self,
        states: Tensor,
        behavior_actions: Tensor | None = None,
        requested_behavior_actions: Tensor | None = None,
        behavior_weight: float | None = None,
        rl_weight: float = 1.0,
        stability_weight: float = 0.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        sample = self.actor.sample(states, deterministic=False)
        action = sample.critic_action
        entropy = sample.log_probability.squeeze(-1)
        q_symlog = torch.minimum(
            self.critic1(states, action), self.critic2(states, action)
        )
        # The critic predicts symlog returns. Maximizing this monotonic transform
        # preserves action ordering while keeping actor gradients O(1).
        rl_loss = (
            self.config.entropy_coefficient * entropy - q_symlog
        ).mean()
        behavior_loss = torch.zeros((), device=states.device)
        shield_fraction = torch.zeros((), device=states.device)
        mean_action = self.actor.sample(
            states, deterministic=True
        ).acceleration
        if behavior_actions is not None:
            # Acceleration is already expressed in a compact physical range.
            # A second division by the full 4.2 m/s2 action span reduced the
            # nominal BC weight by 17.6x and let the immature critic drive the
            # actor into the safety shield during foundation training.
            squared_error = (mean_action - behavior_actions).square()
            if requested_behavior_actions is not None:
                shield_mask = (
                    (behavior_actions - requested_behavior_actions).abs() > 0.05
                ).to(squared_error.dtype)
                shield_fraction = shield_mask.mean()
                unshielded = 1.0 - shield_mask
                behavior_loss = (squared_error * unshielded).sum() / (
                    unshielded.sum().clamp_min(1.0)
                )
            else:
                behavior_loss = squared_error.mean()
        current_states = (
            states[..., -1, :]
            if states.ndim >= 3
            and states.shape[-2] == FOLLOWER_MEMORY_STEPS
            else states
        )
        desired_gap = current_states[
            ..., FOLLOWER_DESIRED_GAP_INDEX : FOLLOWER_DESIRED_GAP_INDEX + 1
        ]
        gap = current_states[
            ..., FOLLOWER_RECOVERY_GAP_INDEX : FOLLOWER_RECOVERY_GAP_INDEX + 1
        ]
        relative_speed = current_states[
            ...,
            FOLLOWER_LOCAL_RELATIVE_SPEED_INDEX : FOLLOWER_LOCAL_RELATIVE_SPEED_INDEX
            + 1,
        ]
        leader_plan = current_states[
            ...,
            FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX : FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX
            + 1,
        ]
        predecessor_acceleration = current_states[
            ...,
            FOLLOWER_LOCAL_PREDECESSOR_ACCELERATION_INDEX :
            FOLLOWER_LOCAL_PREDECESSOR_ACCELERATION_INDEX + 1,
        ].clamp(self.config.accel_min, self.config.accel_max)
        predicted_acceleration = current_states[
            ...,
            FOLLOWER_PREDICTED_ACCELERATION_INDEX :
            FOLLOWER_PREDICTED_ACCELERATION_INDEX + 1,
        ].clamp(self.config.accel_min, self.config.accel_max)
        residual = mean_action - leader_plan
        predecessor_residual = predecessor_acceleration - leader_plan
        recovery_residual = residual - predecessor_residual
        gap_error = gap - desired_gap
        recovery_role = current_states[
            ...,
            FOLLOWER_RECOVERY_ROLE_INDEX : FOLLOWER_RECOVERY_ROLE_INDEX + 1,
        ]
        physical_front_gap = current_states[
            ...,
            FOLLOWER_PHYSICAL_FRONT_GAP_INDEX :
            FOLLOWER_PHYSICAL_FRONT_GAP_INDEX + 1,
        ]
        recovery_clearance = torch.where(
            physical_front_gap > 0.5,
            torch.sigmoid(
                (physical_front_gap - desired_gap - 2.0) / 2.0
            ),
            torch.ones_like(physical_front_gap),
        )
        # This is a training-only Lyapunov direction constraint, not an
        # execution-time fallback controller. It prevents a young critic from
        # rewarding residual actions that actively enlarge an existing gap.
        restoring_signal = torch.tanh(gap_error / 4.0 + relative_speed / 2.0)
        direction_penalty = recovery_clearance * F.relu(
            -recovery_residual * restoring_signal
        )
        direction_loss = direction_penalty.mean()
        if direction_penalty.ndim >= 3:
            direction_loss = direction_loss + direction_penalty.amax(
                dim=-2
            ).mean()
        equilibrium_weight = torch.exp(
            -gap_error.abs() / 2.0 - relative_speed.abs()
        )
        equilibrium_penalty = equilibrium_weight * recovery_residual.square()
        equilibrium_loss = equilibrium_penalty.mean()
        if equilibrium_penalty.ndim >= 3:
            equilibrium_loss = equilibrium_loss + equilibrium_penalty.amax(
                dim=-2
            ).mean()
        recovery_boost = recovery_clearance * (
            0.18 * gap_error + 0.35 * relative_speed
        ).clamp(
            0.0,
            min(
                self.config.accel_max,
                FOLLOWER_RECOVERY_TARGET_MAX_ACCELERATION,
            ),
        )
        recovery_target = (
            FOLLOWER_RECOVERY_PREDICTED_ACCELERATION_WEIGHT
            * predicted_acceleration
            + recovery_boost
        ).clamp(self.config.accel_min, self.config.accel_max)
        recovery_mask = recovery_role * (gap_error > 0.0).to(
            recovery_role.dtype
        )
        recovery_magnitude_loss = (
            recovery_mask * (mean_action - recovery_target).square()
        ).sum() / recovery_mask.sum().clamp_min(1.0)
        stability_loss = (
            direction_loss
            + equilibrium_loss
            + 0.10 * recovery_magnitude_loss
        )
        cloning_weight = (
            self.config.behavior_cloning_weight
            if behavior_weight is None
            else float(behavior_weight)
        )
        loss = (
            float(rl_weight) * rl_loss
            + cloning_weight * behavior_loss
            + float(stability_weight) * stability_loss
        )
        return loss, {
            "actor": loss,
            "actor_rl": rl_loss,
            "actor_rl_weight": rl_loss.new_tensor(float(rl_weight)),
            "behavior": behavior_loss,
            "behavior_weight": behavior_loss.new_tensor(cloning_weight),
            "behavior_shield_fraction": shield_fraction,
            "stability": stability_loss,
            "stability_direction": direction_loss,
            "stability_equilibrium": equilibrium_loss,
            "stability_recovery_magnitude": recovery_magnitude_loss,
            "stability_recovery_clearance": recovery_clearance.mean(),
            "stability_recovery_boost_mean": recovery_boost.mean(),
            "stability_recovery_boost_max": recovery_boost.max(),
            "stability_recovery_target_mean": recovery_target.mean(),
            "stability_recovery_target_max": recovery_target.max(),
            "stability_weight": stability_loss.new_tensor(
                float(stability_weight)
            ),
            "policy_q": symexp(q_symlog.clamp(-8.0, 8.0)).mean(),
            "policy_q_symlog": q_symlog.mean(),
            "entropy": -entropy.mean(),
            "acceleration_mean": sample.acceleration.mean(),
        }

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        tau = self.config.target_tau
        for target, source in (
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.lerp_(source_parameter, tau)

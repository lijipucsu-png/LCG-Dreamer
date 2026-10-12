"""PyTorch DreamerV3 world model adapted for the platoon leader.

The implementation follows DreamerV3's recurrent categorical state-space model,
KL balancing/free bits, symlog prediction targets, and imagination actor-critic.
It adds an explicit multimodal head for multiple human vehicles ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from third_party.r2dreamer.rssm import RSSM as UpstreamRSSM


LEADER_RECOVERY_TRAINING_SPEED = 15.0
LEADER_RECOVERY_ACCELERATION_GAIN = 0.06
LEADER_RECOVERY_MAX_ACCELERATION = 0.6
LEADER_RECOVERY_MIN_GAP = 3.5
LEADER_RECOVERY_PHYSICAL_ACCELERATION = 1.2


def symlog(value: Tensor) -> Tensor:
    return value.sign() * torch.log1p(value.abs())


def symexp(value: Tensor) -> Tensor:
    return value.sign() * torch.expm1(value.abs())


@dataclass
class RSSMState:
    deterministic: Tensor
    stochastic: Tensor
    logits: Tensor


@dataclass
class FrontPrediction:
    trajectories: Tensor
    mode_logits: Tensor


class LeaderObservationEncoder(nn.Module):
    """Encode causal leader observations without flattening temporal structure."""

    def __init__(self, hidden_dim: int = 128, embedding_dim: int = 256) -> None:
        super().__init__()
        self.register_buffer(
            "ego_scale",
            torch.tensor([3500.0, 35.0, 6.0, 15.0, 3.0, 1.0, 0.01, 1.0]),
        )
        self.register_buffer(
            "platoon_scale",
            torch.tensor([3500.0, 35.0, 6.0, 3.0, 100.0, 1.0]),
        )
        self.register_buffer(
            "front_scale",
            torch.tensor([800.0, 3.0, 35.0, 6.0, 20.0, 1.0]),
        )
        self.register_buffer("v2i_scale", torch.tensor([35.0, 300.0, 15.0, 1.0]))
        self.register_buffer("road_scale", torch.tensor([500.0, 0.01, 40.0]))
        self.ego = nn.Sequential(nn.Linear(8, 32), nn.SiLU())
        self.platoon = nn.Sequential(nn.Linear(36, 64), nn.SiLU())
        self.front_temporal = nn.GRU(6, 64, batch_first=True)
        self.front_future = nn.Sequential(nn.Linear(8 * 3, 64), nn.SiLU())
        self.infrastructure = nn.Sequential(nn.Linear(32 + 60 + 5, 64), nn.SiLU())
        self.output = nn.Sequential(
            nn.Linear(32 + 64 + 64 + 64 + 64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(
        self, observation: Mapping[str, Tensor], predicted_front: Tensor
    ) -> Tensor:
        ego = self.ego(observation["ego"] / self.ego_scale)
        platoon = self.platoon(
            (observation["platoon_state"] / self.platoon_scale).flatten(-2)
        )
        front = observation["front_history"] / self.front_scale
        prefix = front.shape[:-3]
        vehicles, steps, features = front.shape[-3:]
        encoded, _ = self.front_temporal(front.reshape(-1, steps, features))
        nodes = encoded[:, -1].reshape(*prefix, vehicles, -1)
        mask = observation["front_mask"][..., None]
        pooled = (nodes * mask).sum(-2) / mask.sum(-2).clamp_min(1.0)
        displacement = predicted_front[..., -1, :] - predicted_front[..., 0, :]
        maximum_lateral = predicted_front[..., 1].abs().amax(-1, keepdim=True)
        future_summary = torch.cat((displacement, maximum_lateral), dim=-1)
        future_summary = future_summary * observation["front_mask"][..., None]
        future_context = self.front_future(
            (future_summary / future_summary.new_tensor([100.0, 10.0, 10.0])).flatten(-2)
        )
        infrastructure = torch.cat(
            (
                (observation["v2i_flow"] / self.v2i_scale).flatten(-2),
                (observation["road_profile"] / self.road_scale).flatten(-2),
                observation["signal"],
                observation["signal_mask"],
            ),
            dim=-1,
        )
        return self.output(
            torch.cat(
                (ego, platoon, pooled, future_context, self.infrastructure(infrastructure)),
                dim=-1,
            )
        )


class FrontTrajectoryHead(nn.Module):
    def __init__(self, hidden_dim: int, modes: int, horizon: int) -> None:
        super().__init__()
        self.modes = modes
        self.horizon = horizon
        self.temporal = nn.GRU(6, hidden_dim, batch_first=True)
        self.context = nn.Sequential(nn.Linear(hidden_dim + 8, hidden_dim), nn.SiLU())
        self.trajectory = nn.Linear(hidden_dim, modes * horizon * 2)
        self.mode = nn.Linear(hidden_dim, modes)
        self.register_buffer("input_scale", torch.tensor([150.0, 15.0, 35.0, 6.0, 20.0, 6.0]))
        self.register_buffer("residual_scale", torch.tensor([10.0, 2.0]))
        nn.init.zeros_(self.mode.weight)
        nn.init.zeros_(self.mode.bias)
        with torch.no_grad():
            self.mode.bias[0] = 2.0

    def forward(self, front_history: Tensor, ego: Tensor) -> FrontPrediction:
        prefix = front_history.shape[:-3]
        vehicles, steps, features = front_history.shape[-3:]
        normalized = front_history / self.input_scale
        encoded, _ = self.temporal(normalized.reshape(-1, steps, features))
        nodes = encoded[:, -1].reshape(*prefix, vehicles, -1)
        # The supervised front predictor is trained with ego speed as context.
        # Excluding absolute station prevents a distribution shift at inference.
        ego_context = torch.zeros_like(ego)
        ego_context[..., 1] = ego[..., 1]
        expanded_ego = ego_context.unsqueeze(-2).expand(
            *prefix, vehicles, ego.shape[-1]
        )
        context = self.context(torch.cat((nodes, expanded_ego), dim=-1))
        residual = self.trajectory(context).view(
            *prefix, vehicles, self.modes, self.horizon, 2
        ) * self.residual_scale
        mode_residual_mask = residual.new_ones(self.modes)
        mode_residual_mask[0] = 0.0
        residual = residual * mode_residual_mask.view(
            *((1,) * len(prefix)), 1, self.modes, 1, 1
        )
        seconds = torch.arange(
            1, self.horizon + 1, device=front_history.device, dtype=front_history.dtype
        ) * 0.1
        latest = front_history[..., -1, :]
        base = front_history.new_zeros(*prefix, vehicles, self.horizon, 2)
        base[..., 0] = latest[..., 0, None] + latest[..., 2, None] * seconds
        base[..., 1] = latest[..., 1, None]
        trajectories = base.unsqueeze(-3) + residual
        return FrontPrediction(trajectories, self.mode(context))


class CategoricalRSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embedding_dim: int,
        deterministic_dim: int,
        stochastic_dim: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.classes = classes
        config = SimpleNamespace(
            stoch=stochastic_dim,
            deter=deterministic_dim,
            hidden=deterministic_dim,
            discrete=classes,
            act="SiLU",
            unimix_ratio=0.01,
            initial="zeros",
            device="cpu",
            obs_layers=1,
            img_layers=1,
            dyn_layers=1,
            blocks=1,
        )
        self.core = UpstreamRSSM(config, embedding_dim, action_dim)

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deterministic = torch.zeros(batch_size, self.deterministic_dim, device=device)
        logits = torch.zeros(batch_size, self.stochastic_dim, self.classes, device=device)
        stochastic = F.one_hot(
            torch.zeros(batch_size, self.stochastic_dim, dtype=torch.long, device=device),
            self.classes,
        ).float()
        return RSSMState(deterministic, stochastic, logits)

    def _sample(self, logits: Tensor) -> Tensor:
        if self.training:
            return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        return F.one_hot(logits.argmax(-1), self.classes).float()

    def observe(self, previous: RSSMState, action: Tensor, embedding: Tensor) -> tuple[RSSMState, RSSMState]:
        reset = torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)
        stochastic, deterministic, posterior_logits = self.core.obs_step(
            previous.stochastic, previous.deterministic, action, embedding, reset
        )
        stochastic = self._sample(posterior_logits)
        prior_stochastic, prior_logits = self.core.prior(deterministic)
        prior_stochastic = self._sample(prior_logits)
        prior = RSSMState(deterministic, prior_stochastic, prior_logits)
        posterior = RSSMState(deterministic, stochastic, posterior_logits)
        return posterior, prior

    def imagine(self, previous: RSSMState, action: Tensor) -> RSSMState:
        stochastic, deterministic = self.core.img_step(
            previous.stochastic, previous.deterministic, action
        )
        logits = self.core._img_net(deterministic)
        stochastic = self._sample(logits)
        return RSSMState(deterministic, stochastic, logits)


class DreamerV3Leader(nn.Module):
    """Leader world model, multimodal predictor, actor, and critic."""

    def __init__(
        self,
        hidden_dim: int = 128,
        stochastic_dim: int = 16,
        stochastic_classes: int = 16,
        trajectory_modes: int = 3,
        prediction_horizon: int = 30,
        action_min: float = -3.0,
        action_max: float = 1.2,
    ) -> None:
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max
        embedding_dim = hidden_dim * 2
        self.encoder = LeaderObservationEncoder(hidden_dim, embedding_dim)
        self.rssm = CategoricalRSSM(
            1, embedding_dim, hidden_dim, stochastic_dim, stochastic_classes
        )
        feature_dim = hidden_dim + stochastic_dim * stochastic_classes
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 44)
        )
        self.road_profile_decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 60)
        )
        self.v2i_plan_decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 32)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.continue_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1), nn.Tanh()
        )
        nn.init.zeros_(self.actor[-2].weight)
        nn.init.zeros_(self.actor[-2].bias)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.front_predictor = FrontTrajectoryHead(hidden_dim, trajectory_modes, prediction_horizon)

    @staticmethod
    def features(state: RSSMState) -> Tensor:
        return torch.cat((state.deterministic, state.stochastic.flatten(-2)), dim=-1)

    def scale_action(self, normalized: Tensor) -> Tensor:
        return torch.where(
            normalized >= 0.0,
            normalized * self.action_max,
            -normalized * self.action_min,
        )

    def initial(self, batch_size: int, device: torch.device | str) -> RSSMState:
        return self.rssm.initial(batch_size, torch.device(device))

    def infer(
        self,
        observation: Mapping[str, Tensor],
        previous: RSSMState,
        previous_action: Tensor,
        deterministic: bool = True,
    ) -> tuple[Tensor, RSSMState, FrontPrediction]:
        embedding, front = self.encode(observation)
        posterior, _ = self.rssm.observe(previous, previous_action, embedding)
        raw_action = self.actor(self.features(posterior))
        action = self.scale_action(raw_action)
        if not deterministic:
            action = (action + torch.randn_like(action) * 0.35).clamp(
                self.action_min, self.action_max
            )
        return action, posterior, front

    def encode(self, observation: Mapping[str, Tensor]) -> tuple[Tensor, FrontPrediction]:
        front = self.front_predictor(observation["front_history"], observation["ego"])
        selected_mode = front.mode_logits.argmax(-1)
        selected = front.trajectories.gather(
            -3,
            selected_mode[..., None, None, None].expand(
                *selected_mode.shape, 1, front.trajectories.shape[-2], 2
            ),
        ).squeeze(-3)
        return self.encoder(observation, selected), front

    def world_model_loss(
        self,
        observations: Mapping[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        continues: Tensor,
        front_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Train on sequences with observation tensors shaped [batch, time, ...]."""

        batch, time = actions.shape[:2]
        posterior_states, prior_states = self.observe_sequence(observations, actions)
        features = torch.stack([self.features(item) for item in posterior_states], dim=1)
        reconstruction_target = torch.cat(
            (observations["ego"], observations["platoon_state"].flatten(-2)), dim=-1
        )
        reconstruction = F.mse_loss(self.decoder(features), symlog(reconstruction_target))
        road_target = observations["road_profile"] / self.encoder.road_scale
        road_profile_reconstruction = F.mse_loss(
            self.road_profile_decoder(features).reshape_as(road_target),
            road_target,
        )
        v2i_target = observations["v2i_flow"] / self.encoder.v2i_scale
        v2i_plan_reconstruction = F.mse_loss(
            self.v2i_plan_decoder(features).reshape_as(v2i_target),
            v2i_target,
        )
        reward_loss = F.mse_loss(self.reward_head(features).squeeze(-1), symlog(rewards))
        continue_loss = F.binary_cross_entropy_with_logits(
            self.continue_head(features).squeeze(-1), continues
        )
        posterior_logits = torch.stack([item.logits for item in posterior_states], dim=1)
        prior_logits = torch.stack([item.logits for item in prior_states], dim=1)
        posterior_distribution = posterior_logits.log_softmax(-1).exp()
        dynamics_kl = (
            posterior_distribution.detach()
            * (posterior_logits.detach().log_softmax(-1) - prior_logits.log_softmax(-1))
        ).sum(-1).mean()
        representation_kl = (
            posterior_distribution
            * (posterior_logits.log_softmax(-1) - prior_logits.detach().log_softmax(-1))
        ).sum(-1).mean()
        kl = 0.5 * torch.maximum(dynamics_kl, dynamics_kl.new_tensor(1.0))
        kl = kl + 0.1 * torch.maximum(representation_kl, representation_kl.new_tensor(1.0))
        trajectory_loss = features.new_zeros(())
        if front_target is not None:
            flat_observation = {
                key: value.reshape(batch * time, *value.shape[2:])
                for key, value in observations.items()
            }
            prediction = self.front_predictor(
                flat_observation["front_history"], flat_observation["ego"]
            )
            target = front_target.reshape(batch * time, *front_target.shape[2:])
            errors = (
                (prediction.trajectories - target[:, :, None])
                / target.new_tensor([10.0, 2.0])
            ).square().mean((-1, -2))
            best = errors.detach().argmin(-1)
            regression = errors.gather(-1, best[..., None]).squeeze(-1)
            mask = flat_observation["front_mask"]
            trajectory_loss = (regression * mask).sum() / mask.sum().clamp_min(1.0)
            classification = F.cross_entropy(
                prediction.mode_logits.flatten(0, 1), best.flatten(), reduction="none"
            ).view_as(mask)
            trajectory_loss = trajectory_loss + 0.2 * (
                (classification * mask).sum() / mask.sum().clamp_min(1.0)
            )
        total = (
            reconstruction
            + road_profile_reconstruction
            + v2i_plan_reconstruction
            + reward_loss
            + continue_loss
            + kl
            + trajectory_loss
        )
        return {
            "loss": total,
            "reconstruction": reconstruction,
            "road_profile_reconstruction": road_profile_reconstruction,
            "v2i_plan_reconstruction": v2i_plan_reconstruction,
            "reward": reward_loss,
            "continue": continue_loss,
            "kl": kl,
            "front_trajectory": trajectory_loss,
        }

    def observe_sequence(
        self,
        observations: Mapping[str, Tensor],
        actions: Tensor,
    ) -> tuple[list[RSSMState], list[RSSMState]]:
        """Infer a posterior sequence using the action preceding each observation."""

        batch, time = actions.shape[:2]
        state = self.initial(batch, actions.device)
        previous_actions = torch.cat((torch.zeros_like(actions[:, :1]), actions[:, :-1]), dim=1)
        posterior_states: list[RSSMState] = []
        prior_states: list[RSSMState] = []
        for index in range(time):
            step_observation = {
                key: value[:, index] for key, value in observations.items()
            }
            embedding, _ = self.encode(step_observation)
            state, prior = self.rssm.observe(
                state, previous_actions[:, index], embedding
            )
            posterior_states.append(state)
            prior_states.append(prior)
        return posterior_states, prior_states

    def front_prediction_loss(
        self,
        front_history: Tensor,
        ego: Tensor,
        mask: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:
        prediction = self.front_predictor(front_history, ego)
        errors = (
            (prediction.trajectories - target[:, :, None])
            / target.new_tensor([10.0, 2.0])
        ).square().mean((-1, -2))
        best = errors.detach().argmin(-1)
        regression = (errors.gather(-1, best[..., None]).squeeze(-1) * mask).sum()
        regression = regression / mask.sum().clamp_min(1.0)
        classification = F.cross_entropy(
            prediction.mode_logits.flatten(0, 1), best.flatten(), reduction="none"
        ).view_as(mask)
        classification = (classification * mask).sum() / mask.sum().clamp_min(1.0)
        total = regression + 0.2 * classification
        return {"loss": total, "regression": regression, "mode": classification}

    def imagination_loss(
        self,
        start: RSSMState,
        horizon: int = 15,
        discount: float = 0.997,
        lambda_: float = 0.95,
    ) -> dict[str, Tensor]:
        """Dreamer actor-critic objective through latent imagined rollouts."""

        state = RSSMState(
            start.deterministic.detach(), start.stochastic.detach(), start.logits.detach()
        )
        features: list[Tensor] = []
        rewards: list[Tensor] = []
        discounts: list[Tensor] = []
        actions: list[Tensor] = []
        for _ in range(horizon):
            feature = self.features(state)
            normalized_action = self.actor(feature)
            action = self.scale_action(normalized_action)
            state = self.rssm.imagine(state, action)
            imagined_feature = self.features(state)
            features.append(imagined_feature)
            actions.append(action)
            reward_symlog = self.reward_head(imagined_feature).squeeze(-1)
            rewards.append(symexp(reward_symlog.clamp(-5.0, 5.0)))
            continuation = self.continue_head(imagined_feature).squeeze(-1).sigmoid()
            discounts.append(discount * continuation)
        feature_sequence = torch.stack(features, dim=1)
        reward_sequence = torch.stack(rewards, dim=1)
        discount_sequence = torch.stack(discounts, dim=1)
        value_symlog = self.critic(feature_sequence).squeeze(-1)
        value_sequence = symexp(value_symlog.clamp(-5.0, 5.0))
        bootstrap = value_sequence[:, -1].detach()
        returns: list[Tensor] = []
        for index in reversed(range(horizon)):
            next_value = bootstrap if index == horizon - 1 else value_sequence[:, index + 1].detach()
            bootstrap = reward_sequence[:, index] + discount_sequence[:, index] * (
                (1.0 - lambda_) * next_value + lambda_ * bootstrap
            )
            returns.append(bootstrap)
        returns.reverse()
        return_sequence = torch.stack(returns, dim=1)
        weights = torch.cumprod(
            torch.cat((torch.ones_like(discount_sequence[:, :1]), discount_sequence[:, :-1]), 1),
            dim=1,
        ).detach()
        action_sequence = torch.stack(actions, dim=1)
        detached_returns = return_sequence.detach()
        low = torch.quantile(detached_returns, 0.05)
        high = torch.quantile(detached_returns, 0.95)
        return_scale = (high - low).clamp_min(1.0)
        advantage = (return_sequence - value_sequence.detach()) / return_scale
        actor_loss = -(weights * advantage).mean() + 1e-4 * action_sequence.square().mean()
        critic_loss = (
            weights * (value_symlog - symlog(detached_returns)).square()
        ).mean()
        return {
            "loss": actor_loss + critic_loss,
            "actor": actor_loss,
            "critic": critic_loss,
            "imagined_return": return_sequence.mean(),
        }

    def behavior_loss(
        self, posterior_states: list[RSSMState], actions: Tensor
    ) -> Tensor:
        """Anchor imagined control to applied actions while the model matures."""

        features = torch.stack(
            [self.features(state).detach() for state in posterior_states], dim=1
        )
        predicted = self.scale_action(self.actor(features))
        return F.mse_loss(predicted, actions)

    def restart_loss(
        self,
        posterior_states: list[RSSMState],
        observations: Mapping[str, Tensor],
    ) -> Tensor:
        """Train the RL actor to recover toward a physically safe cruise speed."""

        features = torch.stack(
            [self.features(state).detach() for state in posterior_states],
            dim=1,
        )
        predicted = self.scale_action(self.actor(features))
        speed = observations["ego"][..., 1:2]
        opportunity = observations["signal"][..., 0:1]
        signal_valid = observations["signal_mask"][..., 0:1]
        front_gap = observations["signal"][..., 2:3] * 300.0
        front_speed = observations["signal"][..., 3:4] * 30.0
        road_target = observations["road_profile"][..., 2].amin(
            dim=-1, keepdim=True
        ).clamp(5.0, 30.0)
        safe_space = F.relu(front_gap - LEADER_RECOVERY_MIN_GAP)
        safe_target = torch.sqrt(
            front_speed.square()
            + 2.0 * LEADER_RECOVERY_PHYSICAL_ACCELERATION * safe_space
        )
        target_speed = torch.minimum(road_target, safe_target)
        speed_deficit = F.relu(target_speed - speed)
        follower_chain = observations["platoon_state"][..., 1:, :]
        chain_valid = follower_chain[..., 5:6].clamp(0.0, 1.0).amin(dim=-2)
        pair_gaps = follower_chain[..., 4:5]
        cohesion_gate = torch.exp(
            -0.5 * F.relu(pair_gaps - 12.0).square()
        ).amin(dim=-2)
        restart_mask = (
            opportunity
            * signal_valid
            * chain_valid
            * cohesion_gate
            * (speed_deficit > 0.25).to(speed.dtype)
        )
        target = (
            (LEADER_RECOVERY_ACCELERATION_GAIN * speed_deficit).clamp(
                0.0, LEADER_RECOVERY_MAX_ACCELERATION
            )
            * chain_valid
            * cohesion_gate
        )
        return (
            restart_mask * (predicted - target).square()
        ).sum() / restart_mask.sum().clamp_min(1.0)

"""Physics-guided probabilistic ensemble used by follower model-based RL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from mbrl.models import Model as MBRLModel
except ImportError:  # The core model remains usable without the archived MBRL-Lib package.
    class MBRLModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, device: torch.device | str) -> None:
            super().__init__()
            self.device = torch.device(device)


FOLLOWER_STATE_DIM = 16
ACTION_DIM = 1


@dataclass
class DynamicsPrediction:
    mean: Tensor
    log_variance: Tensor


class ResidualMember(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + ACTION_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim * 2),
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        output = self.network(torch.cat((state, action), dim=-1))
        mean, raw_log_variance = output.chunk(2, dim=-1)
        log_variance = -10.0 + F.softplus(raw_log_variance + 10.0)
        log_variance = 0.5 - F.softplus(0.5 - log_variance)
        return mean, log_variance


class StructuredEnsembleDynamics(nn.Module):
    """MBRL-Lib compatible ensemble with an analytic truck-motion backbone.

    State layout:
      station, speed, acceleration, gap, predecessor speed/acceleration,
      desired gap, curvature, speed advisory, cut-in probability,
      nearest-human relative x/speed, leader speed/acceleration,
      communication age, normalized follower index.
    """

    def __init__(
        self,
        state_dim: int = FOLLOWER_STATE_DIM,
        hidden_dim: int = 128,
        ensemble_size: int = 5,
        step_length: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.ensemble_size = ensemble_size
        self.step_length = step_length
        self.members = nn.ModuleList(
            ResidualMember(state_dim, hidden_dim) for _ in range(ensemble_size)
        )
        self.register_buffer("input_mean", torch.zeros(state_dim + ACTION_DIM))
        self.register_buffer("input_std", torch.ones(state_dim + ACTION_DIM))
        self.register_buffer("delta_std", torch.ones(state_dim))
        self.register_buffer("normalizer_initialized", torch.zeros((), dtype=torch.bool))

    def physics_step(self, state: Tensor, action: Tensor) -> Tensor:
        dt = self.step_length
        acceleration = action[..., 0].clamp(-3.0, 1.2)
        next_state = state.clone()
        next_state[..., 0] = state[..., 0] + state[..., 1] * dt + 0.5 * acceleration * dt * dt
        next_state[..., 1] = (state[..., 1] + acceleration * dt).clamp(0.0, 40.0)
        next_state[..., 2] = acceleration
        next_state[..., 3] = (
            state[..., 3]
            + (state[..., 4] - state[..., 1]) * dt
            + 0.5 * (state[..., 5] - acceleration) * dt * dt
        )
        next_state[..., 4] = (state[..., 4] + state[..., 5] * dt).clamp(0.0, 40.0)
        next_state[..., 10] = state[..., 10] + state[..., 11] * dt
        next_state[..., 13] = state[..., 13]
        next_state[..., 14] = state[..., 14] + dt
        return next_state

    def forward(self, state: Tensor, action: Tensor) -> DynamicsPrediction:
        physics = self.physics_step(state, action)
        normalized = (torch.cat((state, action), -1) - self.input_mean) / self.input_std
        normalized_state = normalized[..., : self.state_dim]
        normalized_action = normalized[..., self.state_dim :]
        means: list[Tensor] = []
        log_variances: list[Tensor] = []
        for member in self.members:
            residual, log_variance = member(normalized_state, normalized_action)
            means.append(physics + residual * self.delta_std * 0.1)
            physical_log_variance = log_variance + 2.0 * self.delta_std.log()
            log_variances.append(physical_log_variance.clamp(-6.0, 3.0))
        return DynamicsPrediction(torch.stack(means), torch.stack(log_variances))

    def predict(self, state: Tensor, action: Tensor, sample: bool = False) -> Tensor:
        prediction = self(state, action)
        if sample:
            member = torch.randint(self.ensemble_size, (state.shape[0],), device=state.device)
            index = torch.arange(state.shape[0], device=state.device)
            mean = prediction.mean[member, index]
            log_variance = prediction.log_variance[member, index]
            return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        return prediction.mean.mean(0)

    def loss(self, state: Tensor, action: Tensor, next_state: Tensor) -> dict[str, Tensor]:
        prediction = self(state, action)
        target = next_state.unsqueeze(0)
        inverse_variance = torch.exp(-prediction.log_variance)
        standardized_error = (
            (prediction.mean - target).square() * inverse_variance
        ).clamp_max(25.0)
        nll = (standardized_error + prediction.log_variance).mean()
        disagreement = prediction.mean.var(dim=0).mean()
        return {"loss": nll, "nll": nll, "disagreement": disagreement}

    @torch.no_grad()
    def update_normalizer(self, state: Tensor, action: Tensor, next_state: Tensor) -> None:
        inputs = torch.cat((state, action), -1)
        physics = self.physics_step(state, action)
        batch_mean = inputs.mean(0)
        batch_std = inputs.std(0, unbiased=False).clamp_min(0.05)
        batch_delta_std = (next_state - physics).std(0, unbiased=False).clamp_min(0.05)
        if not bool(self.normalizer_initialized):
            self.input_mean.copy_(batch_mean)
            self.input_std.copy_(batch_std)
            self.delta_std.copy_(batch_delta_std)
            self.normalizer_initialized.fill_(True)
        else:
            self.input_mean.lerp_(batch_mean, 0.01)
            self.input_std.lerp_(batch_std, 0.01)
            self.delta_std.lerp_(batch_delta_std, 0.01)


class MBRLStructuredAdapter(MBRLModel):
    """Expose the structured model through MBRL-Lib's ``Model`` contract."""

    def __init__(self, model: StructuredEnsembleDynamics, device: torch.device | str) -> None:
        super().__init__(device)
        self.model = model

    def forward(self, model_in: Tensor, **_: object) -> tuple[Tensor, Tensor]:
        state = model_in[..., : self.model.state_dim]
        action = model_in[..., self.model.state_dim : self.model.state_dim + ACTION_DIM]
        prediction = self.model(state, action)
        return prediction.mean, prediction.log_variance

    def loss(self, model_in: Tensor, target: Tensor | None = None):
        if target is None:
            raise ValueError("MBRL loss requires a next-state target")
        state = model_in[..., : self.model.state_dim]
        action = model_in[..., self.model.state_dim : self.model.state_dim + ACTION_DIM]
        losses = self.model.loss(state, action, target)
        return losses["loss"], {
            "nll": float(losses["nll"].detach()),
            "disagreement": float(losses["disagreement"].detach()),
        }

    @torch.no_grad()
    def eval_score(self, model_in: Tensor, target: Tensor | None = None):
        if target is None:
            raise ValueError("MBRL evaluation requires a next-state target")
        means, _ = self(model_in)
        score = (means.mean(0) - target).square()
        return score, {}

    def reset(self, obs: Tensor, rng: torch.Generator | None = None) -> dict[str, Tensor]:
        del rng
        return {"obs": obs}

    @torch.no_grad()
    def sample(
        self,
        act: Tensor,
        model_state: dict[str, Tensor],
        deterministic: bool = False,
        rng: torch.Generator | None = None,
    ) -> tuple[Tensor, None, None, dict[str, Tensor]]:
        del rng
        next_observation = self.model.predict(
            model_state["obs"], act, sample=not deterministic
        )
        return next_observation, None, None, {"obs": next_observation}

    def predict(self, state: Tensor, action: Tensor, sample: bool = False) -> Tensor:
        model_state = self.reset(state)
        next_state, _, _, _ = self.sample(action, model_state, deterministic=not sample)
        return next_state

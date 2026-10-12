"""Neural networks shared by the classical continuous-control baselines."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
import torch.nn.functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    network = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(network[-1].weight, 0.01)
    return network


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = mlp(state_dim, action_dim)

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state).tanh()


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_dependent_std: bool,
        initial_log_std: float = -1.0,
        mean_limit: float | None = None,
    ) -> None:
        super().__init__()
        output_dim = action_dim * 2 if state_dependent_std else action_dim
        self.network = mlp(state_dim, output_dim)
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        self.mean_limit = mean_limit
        if state_dependent_std:
            self.log_std = None
        else:
            self.log_std = nn.Parameter(
                torch.full((action_dim,), float(initial_log_std))
            )

    def distribution(self, state: Tensor) -> Normal:
        output = self.network(state)
        if self.state_dependent_std:
            mean, log_std = output.chunk(2, -1)
            log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        else:
            mean = output
            assert self.log_std is not None
            log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        if self.mean_limit is not None:
            mean = mean.clamp(-self.mean_limit, self.mean_limit)
        return Normal(mean, log_std.exp())

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw: Tensor, unit: Tensor) -> Tensor:
        correction = torch.log(1.0 - unit.square() + 1e-6)
        return (distribution.log_prob(raw) - correction).sum(-1, keepdim=True)

    @staticmethod
    def _squashed_log_prob_from_raw(
        distribution: Normal, raw: Tensor
    ) -> Tensor:
        correction = 2.0 * (
            math.log(2.0) - raw - F.softplus(-2.0 * raw)
        )
        return (distribution.log_prob(raw) - correction).sum(-1, keepdim=True)

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(state)
        raw = distribution.rsample()
        unit = raw.tanh().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_prob = self._squashed_log_prob_from_raw(distribution, raw)
        deterministic = distribution.mean.tanh()
        return unit, log_prob, deterministic

    def log_prob(self, state: Tensor, unit: Tensor) -> tuple[Tensor, Tensor]:
        clipped = unit.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(clipped)
        distribution = self.distribution(state)
        log_prob = self._squashed_log_prob(distribution, raw, clipped)
        entropy = distribution.entropy().sum(-1, keepdim=True)
        return log_prob, entropy

    def deterministic(self, state: Tensor) -> Tensor:
        return self.distribution(state).mean.tanh()


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = mlp(state_dim + action_dim, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat((state, action), -1))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.network = mlp(state_dim, 1)

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)

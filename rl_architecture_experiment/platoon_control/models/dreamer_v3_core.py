"""DreamerV3 distributional prediction and return-normalization primitives.

The graph controller historically used scalar MSE reward and value heads.  The
budgeted architecture ablation uses the DreamerV3 formulation instead: targets
are transformed with symlog, encoded between adjacent categorical bins, and
optimized by cross entropy.  Keeping these primitives in a small module makes
the DreamerV3 baseline and graph variants use exactly the same objective.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def symlog(value: Tensor) -> Tensor:
    return value.sign() * torch.log1p(value.abs())


def symexp(value: Tensor) -> Tensor:
    return value.sign() * torch.expm1(value.abs())


class SymLogTwoHot(nn.Module):
    """Categorical scalar head with DreamerV3-style symlog two-hot targets."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bins: int = 255,
        lower: float = -20.0,
        upper: float = 20.0,
    ) -> None:
        super().__init__()
        if bins < 3 or bins % 2 == 0:
            raise ValueError("two-hot head requires an odd number of at least 3 bins")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bins),
        )
        self.register_buffer("bins", torch.linspace(lower, upper, bins))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)

    def target(self, value: Tensor) -> Tensor:
        transformed = symlog(value).clamp(self.bins[0], self.bins[-1])
        position = (transformed - self.bins[0]) / (
            self.bins[-1] - self.bins[0]
        ) * (self.bins.numel() - 1)
        lower = position.floor().long().clamp(0, self.bins.numel() - 1)
        upper = position.ceil().long().clamp(0, self.bins.numel() - 1)
        upper_weight = position - lower.to(position.dtype)
        lower_weight = 1.0 - upper_weight
        target = F.one_hot(lower, self.bins.numel()).to(position.dtype)
        target = target * lower_weight.unsqueeze(-1)
        target = target + F.one_hot(upper, self.bins.numel()).to(
            position.dtype
        ) * upper_weight.unsqueeze(-1)
        return target

    def loss(self, logits: Tensor, value: Tensor) -> Tensor:
        target = self.target(value).detach()
        return -(target * logits.log_softmax(-1)).sum(-1).mean()

    def mode(self, logits: Tensor) -> Tensor:
        transformed = (logits.softmax(-1) * self.bins).sum(-1)
        return symexp(transformed)


class PercentileReturnNormalizer(nn.Module):
    """EMA of the 5th--95th percentile return range used by DreamerV3."""

    def __init__(
        self,
        rate: float = 0.01,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        minimum_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < rate <= 1.0:
            raise ValueError("return-normalizer rate must be in (0, 1]")
        self.rate = float(rate)
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.minimum_scale = float(minimum_scale)
        self.register_buffer("percentiles", torch.zeros(2))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, returns: Tensor) -> tuple[Tensor, Tensor]:
        quantiles = torch.quantile(
            returns.detach().float().reshape(-1),
            returns.new_tensor(
                [self.lower_percentile, self.upper_percentile], dtype=torch.float32
            ),
        ).to(self.percentiles.dtype)
        if not bool(self.initialized):
            self.percentiles.copy_(quantiles)
            self.initialized.fill_(True)
        else:
            self.percentiles.lerp_(quantiles, self.rate)
        offset = self.percentiles[0]
        scale = (self.percentiles[1] - self.percentiles[0]).clamp_min(
            self.minimum_scale
        )
        return offset.detach(), scale.detach()

    def current(self) -> tuple[Tensor, Tensor]:
        offset = self.percentiles[0]
        scale = (self.percentiles[1] - self.percentiles[0]).clamp_min(
            self.minimum_scale
        )
        return offset.detach(), scale.detach()

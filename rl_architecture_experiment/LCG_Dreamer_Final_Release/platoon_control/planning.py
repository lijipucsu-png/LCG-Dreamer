"""Batched cross-entropy model-predictive control for all follower agents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import PlannerConfig
from .models.dynamics import StructuredEnsembleDynamics


@dataclass
class PlanResult:
    action: Tensor
    sequence: Tensor
    expected_cost: Tensor


class CEMPlanner:
    def __init__(self, dynamics: StructuredEnsembleDynamics, config: PlannerConfig) -> None:
        self.dynamics = dynamics
        self.config = config
        self._previous_mean: Tensor | None = None

    def reset(self) -> None:
        self._previous_mean = None

    def _rollout_cost(self, states: Tensor, actions: Tensor) -> Tensor:
        """Return costs shaped [population, followers]."""

        population, followers, horizon, _ = actions.shape
        current = states.unsqueeze(0).expand(population, -1, -1).reshape(-1, states.shape[-1])
        total = current.new_zeros(population * followers)
        previous_action = current[..., 2:3]
        for step in range(horizon):
            command = actions[:, :, step].reshape(-1, 1)
            current = self.dynamics.predict(current, command, sample=False)
            speed_error = (current[:, 1] - current[:, 8]).square() / 100.0
            gap_error = (current[:, 3] - current[:, 6]).square() / 100.0
            unsafe_gap = torch.relu(5.0 - current[:, 3]).square()
            jerk = (command - previous_action).square()
            string_error = (current[:, 2] - current[:, 5]).square()
            cutin_risk = current[:, 9] * torch.relu(15.0 - current[:, 10]).square() / 25.0
            traction = torch.relu(command[:, 0]) * current[:, 1] / 30.0
            total = total + (
                0.7 * speed_error
                + 1.2 * gap_error
                + 8.0 * unsafe_gap
                + 0.08 * jerk[:, 0]
                + 0.15 * string_error
                + 2.0 * cutin_risk
                + 0.03 * traction
            )
            previous_action = command
        total = total - 0.02 * current[:, 0]
        return total.view(population, followers)

    @torch.no_grad()
    def plan(self, initial_states: Tensor) -> PlanResult:
        followers = initial_states.shape[0]
        cfg = self.config
        if self._previous_mean is not None and self._previous_mean.shape[0] == followers:
            mean = torch.cat(
                (self._previous_mean[:, 1:], self._previous_mean[:, -1:]), dim=1
            )
        else:
            mean = initial_states.new_zeros(followers, cfg.horizon, 1)
        std = initial_states.new_full(mean.shape, (cfg.accel_max - cfg.accel_min) * 0.35)
        expected_cost = initial_states.new_zeros(followers)
        for _ in range(cfg.iterations):
            noise = torch.randn(cfg.population, *mean.shape, device=initial_states.device)
            candidates = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                cfg.accel_min, cfg.accel_max
            )
            costs = self._rollout_cost(initial_states, candidates)
            elite_indices = costs.topk(cfg.elites, dim=0, largest=False).indices
            candidates_by_follower = candidates.permute(1, 0, 2, 3)
            gather_index = elite_indices.transpose(0, 1)[..., None, None].expand(
                followers, cfg.elites, cfg.horizon, 1
            )
            elites = candidates_by_follower.gather(1, gather_index)
            updated_mean = elites.mean(1)
            updated_std = elites.std(1).clamp_min(cfg.min_std)
            mean = cfg.momentum * mean + (1.0 - cfg.momentum) * updated_mean
            std = cfg.momentum * std + (1.0 - cfg.momentum) * updated_std
            expected_cost = costs.gather(0, elite_indices[:1]).squeeze(0)
        self._previous_mean = mean.clone()
        return PlanResult(mean[:, 0], mean, expected_cost)

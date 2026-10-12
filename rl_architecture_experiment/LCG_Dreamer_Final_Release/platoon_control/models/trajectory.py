"""Spatio-temporal graph network for human cut-in and trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TrajectoryPrediction:
    trajectories: Tensor
    mode_logits: Tensor
    cutin_logits: Tensor
    lane_change_logits: Tensor


class STGNNTrajectoryPredictor(nn.Module):
    """Predict multimodal neighbor trajectories and cut-in intent."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        horizon: int = 30,
        modes: int = 3,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.modes = modes
        self.output_dim = output_dim
        self.temporal = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.trajectory_head = nn.Linear(hidden_dim, modes * horizon * output_dim)
        self.mode_head = nn.Linear(hidden_dim, modes)
        self.cutin_head = nn.Linear(hidden_dim, 1)
        self.lane_change_head = nn.Linear(hidden_dim, 3)
        self.register_buffer(
            "history_scale", torch.tensor([150.0, 15.0, 35.0, 6.0, 20.0, 6.0, 1.0, 1.0])
        )
        self.register_buffer("residual_scale", torch.tensor([10.0, 2.0, 5.0, 2.0]))
        nn.init.zeros_(self.mode_head.weight)
        nn.init.zeros_(self.mode_head.bias)
        with torch.no_grad():
            self.mode_head.bias[0] = 2.0

    def forward(self, history: Tensor, mask: Tensor) -> TrajectoryPrediction:
        if history.ndim != 4 or history.shape[-1] != 8:
            raise ValueError("history must have shape [batch, neighbors, time, 8]")
        batch, neighbors, steps, features = history.shape
        normalized = history / self.history_scale
        encoded, _ = self.temporal(normalized.reshape(batch * neighbors, steps, features))
        nodes = encoded[:, -1].reshape(batch, neighbors, -1)
        scores = torch.matmul(self.query(nodes), self.key(nodes).transpose(1, 2))
        scores = scores / (nodes.shape[-1] ** 0.5)
        valid_keys = mask[:, None, :].bool()
        scores = scores.masked_fill(~valid_keys, -1e4)
        attention = scores.softmax(dim=-1)
        message = torch.matmul(attention, self.value(nodes))
        fused = self.fusion(torch.cat((nodes, message), dim=-1))
        fused = fused * mask[..., None]
        residual = self.trajectory_head(fused).view(
            batch, neighbors, self.modes, self.horizon, self.output_dim
        ) * self.residual_scale
        mode_residual_mask = residual.new_ones(self.modes)
        mode_residual_mask[0] = 0.0
        residual = residual * mode_residual_mask[None, None, :, None, None]
        seconds = torch.arange(
            1, self.horizon + 1, device=history.device, dtype=history.dtype
        ) * 0.1
        latest = history[:, :, -1]
        base = history.new_zeros(batch, neighbors, self.horizon, self.output_dim)
        base[..., 0] = latest[..., 0, None] + latest[..., 2, None] * seconds
        base[..., 1] = latest[..., 1, None]
        base[..., 2] = latest[..., 2, None]
        base[..., 3] = latest[..., 3, None]
        trajectories = base[:, :, None] + residual
        mode_logits = self.mode_head(fused)
        cutin_logits = self.cutin_head(fused).squeeze(-1)
        lane_change_logits = self.lane_change_head(fused)
        return TrajectoryPrediction(
            trajectories, mode_logits, cutin_logits, lane_change_logits
        )

    @staticmethod
    def loss(
        prediction: TrajectoryPrediction,
        target: Tensor,
        mask: Tensor,
        cutin_target: Tensor,
        lane_change_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        valid = mask[..., None]
        target_scale = target.new_tensor([10.0, 2.0, 5.0, 2.0])
        mode_error = (
            (prediction.trajectories - target[:, :, None]) / target_scale
        ).square().mean((-1, -2))
        best_mode = mode_error.detach().argmin(dim=-1)
        chosen_error = mode_error.gather(-1, best_mode[..., None]).squeeze(-1)
        trajectory_loss = (chosen_error * mask).sum() / mask.sum().clamp_min(1.0)
        classification = F.cross_entropy(
            prediction.mode_logits.reshape(-1, prediction.mode_logits.shape[-1]),
            best_mode.reshape(-1),
            reduction="none",
        ).reshape_as(mask)
        mode_loss = (classification * mask).sum() / mask.sum().clamp_min(1.0)
        cutin = F.binary_cross_entropy_with_logits(
            prediction.cutin_logits, cutin_target, reduction="none"
        )
        cutin_loss = (cutin * mask).sum() / mask.sum().clamp_min(1.0)
        lane_change_loss = trajectory_loss.new_zeros(())
        if lane_change_target is not None:
            flat_target = lane_change_target.reshape(-1)
            flat_mask = mask.reshape(-1)
            class_weights = prediction.lane_change_logits.new_tensor([0.05, 1.0, 1.0])
            lane_ce = F.cross_entropy(
                prediction.lane_change_logits.reshape(-1, 3),
                flat_target,
                weight=class_weights,
                reduction="none",
            )
            probabilities = prediction.lane_change_logits.reshape(-1, 3).softmax(-1)
            target_probability = probabilities.gather(1, flat_target[:, None]).squeeze(1)
            focal = (1.0 - target_probability).square() * lane_ce
            lane_change_loss = (focal * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
        total = trajectory_loss + 0.2 * mode_loss + 0.5 * cutin_loss + 0.5 * lane_change_loss
        return {
            "loss": total,
            "trajectory": trajectory_loss,
            "mode": mode_loss,
            "cutin": cutin_loss,
            "lane_change": lane_change_loss,
        }

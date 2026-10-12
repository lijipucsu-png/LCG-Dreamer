"""CPIDLM-MORE inspired delayed-state reconstruction network.

The original project artifact contains no importable model definition matching its
orphaned weights.  This implementation keeps the useful structure explicit: causal
history encoding, learned interaction weights, local-radar fusion, and probabilistic
multi-step state prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DelayPrediction:
    current_state: Tensor
    future_mean: Tensor
    future_log_std: Tensor
    fusion_gate: Tensor


class CausalFeatureMixer(nn.Module):
    """Learned directed feature graph applied independently at each time step."""

    def __init__(self, features: int, hidden_dim: int) -> None:
        super().__init__()
        self.adjacency_logits = nn.Parameter(torch.eye(features) * 2.0)
        self.input_projection = nn.Linear(features, hidden_dim)
        self.graph_projection = nn.Linear(features, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, history: Tensor) -> Tensor:
        adjacency = self.adjacency_logits.softmax(dim=-1)
        graph_features = torch.einsum("btf,fg->btg", history, adjacency)
        return self.norm(self.input_projection(history) + self.graph_projection(graph_features))


class CPIDLMDelayCompensator(nn.Module):
    """Reconstruct the predecessor now-state and predict its short-horizon future."""

    def __init__(
        self,
        history_features: int = 8,
        local_features: int = 6,
        hidden_dim: int = 128,
        horizon: int = 30,
        state_dim: int = 3,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.mixer = CausalFeatureMixer(history_features, hidden_dim)
        self.temporal = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.local_encoder = nn.Sequential(
            nn.Linear(local_features, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.current_head = nn.Linear(hidden_dim, state_dim)
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * state_dim * 2),
        )
        self.register_buffer(
            "history_scale", torch.tensor([500.0, 30.0, 5.0, 0.2, 500.0, 30.0, 5.0, 0.2])
        )
        self.register_buffer(
            "local_scale", torch.tensor([100.0, 30.0, 5.0, 1.0, 30.0, 1.0])
        )
        self.register_buffer("state_scale", torch.tensor([500.0, 30.0, 5.0]))
        self.register_buffer("residual_scale", torch.tensor([20.0, 5.0, 2.0]))

    def forward(self, v2v_history: Tensor, local_predecessor: Tensor) -> DelayPrediction:
        if v2v_history.ndim != 3 or v2v_history.shape[-1] != 8:
            raise ValueError("v2v_history must have shape [batch, time, 8]")
        encoded, _ = self.temporal(self.mixer(v2v_history / self.history_scale))
        delayed_context = encoded[:, -1]
        local_context = self.local_encoder(local_predecessor / self.local_scale)
        gate = self.gate(torch.cat((delayed_context, local_context), dim=-1))
        fused = gate * local_context + (1.0 - gate) * delayed_context
        delayed_state = v2v_history[:, -1, 4:7]
        packet_age = v2v_history[:, -1, 7]
        current_base = delayed_state.clone()
        current_base[:, 0] = (
            delayed_state[:, 0]
            + delayed_state[:, 1] * packet_age
            + 0.5 * delayed_state[:, 2] * packet_age.square()
        )
        current_base[:, 1] = delayed_state[:, 1] + delayed_state[:, 2] * packet_age
        correction_mask = delayed_state.new_tensor([0.0, 0.0, 1.0])
        current = current_base + self.current_head(fused) * self.residual_scale * correction_mask
        distribution = self.future_head(fused).view(
            fused.shape[0], self.horizon, self.state_dim, 2
        )
        seconds = torch.arange(
            1, self.horizon + 1, device=current.device, dtype=current.dtype
        )[None] * 0.1
        base = current[:, None].expand(-1, self.horizon, -1).clone()
        base[..., 0] = (
            current[:, 0, None]
            + current[:, 1, None] * seconds
            + 0.5 * current[:, 2, None] * seconds.square()
        )
        base[..., 1] = current[:, 1, None] + current[:, 2, None] * seconds
        mean = base + distribution[..., 0] * self.residual_scale
        log_std = (
            distribution[..., 1].clamp(-5.0, 2.0) + self.residual_scale.log()
        ).clamp(-5.0, 7.0)
        return DelayPrediction(current, mean, log_std, gate)

    @staticmethod
    def loss(
        prediction: DelayPrediction,
        current_target: Tensor,
        future_target: Tensor,
    ) -> dict[str, Tensor]:
        state_scale = current_target.new_tensor([20.0, 5.0, 2.0])
        current_loss = F.smooth_l1_loss(
            prediction.current_state / state_scale, current_target / state_scale
        )
        inv_variance = torch.exp(-2.0 * prediction.future_log_std)
        future_nll = (
            0.5 * (future_target - prediction.future_mean).square() * inv_variance
            + prediction.future_log_std
        ).mean()
        smoothness = (prediction.future_mean / state_scale).diff(dim=1, n=2).square().mean()
        total = current_loss + future_nll + 0.02 * smoothness
        return {
            "loss": total,
            "current": current_loss,
            "future_nll": future_nll,
            "smoothness": smoothness,
        }

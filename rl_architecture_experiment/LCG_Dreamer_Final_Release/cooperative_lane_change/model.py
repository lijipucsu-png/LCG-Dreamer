"""Parameter-shared graph actor-critic for lateral intent and formation shape."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Normal


NODE_COUNT = 6
PAIR_COUNT = 5
NODE_FEATURE_DIM = 26
HIDDEN_DIM = 96


@dataclass
class LanePolicySample:
    change: Tensor
    gap_raw: Tensor
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    change_logits: Tensor
    gap_mean: Tensor


class CooperativeLaneActorCritic(nn.Module):
    """Small team-level adapter; the transferred longitudinal policy is frozen."""

    def __init__(self, feature_dim: int = NODE_FEATURE_DIM) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
                    nn.LayerNorm(HIDDEN_DIM),
                    nn.SiLU(),
                )
                for _ in range(2)
            ]
        )
        self.change_head = nn.Sequential(
            nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.gap_head = nn.Sequential(
            nn.Linear(4 * HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.gap_log_std = nn.Parameter(torch.full((PAIR_COUNT,), -1.0))

        adjacency = torch.eye(NODE_COUNT)
        for index in range(NODE_COUNT - 1):
            adjacency[index, index + 1] = 1.0
            adjacency[index + 1, index] = 1.0
        adjacency = adjacency / adjacency.sum(-1, keepdim=True)
        self.register_buffer("adjacency", adjacency)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        nodes = self.node_encoder(features)
        for layer in self.message_layers:
            messages = torch.einsum("ij,bjh->bih", self.adjacency, nodes)
            nodes = nodes + layer(torch.cat((nodes, messages), dim=-1))
        pooled = torch.cat((nodes.mean(1), nodes.amax(1)), dim=-1)
        global_nodes = pooled[:, None, :].expand(-1, NODE_COUNT, -1)
        change_logits = self.change_head(
            torch.cat((nodes, global_nodes), dim=-1)
        ).squeeze(-1)
        pair_global = pooled[:, None, :].expand(-1, PAIR_COUNT, -1)
        gap_mean = self.gap_head(
            torch.cat(
                (nodes[:, :-1], nodes[:, 1:], pair_global), dim=-1
            )
        ).squeeze(-1)
        value = self.value_head(pooled).squeeze(-1)
        return change_logits, gap_mean, value

    def sample(
        self,
        features: Tensor,
        change_mask: Tensor,
        gap_active: Tensor,
        deterministic: bool,
    ) -> LanePolicySample:
        change_logits, gap_mean, value = self(features)
        if change_mask.ndim == 1:
            change_mask = change_mask.unsqueeze(0)
        if gap_active.ndim == 0:
            gap_active = gap_active.unsqueeze(0)
        mask_float = change_mask.float()
        raw_valid_count = mask_float.sum(-1)
        valid_count = raw_valid_count.clamp_min(1.0)
        team_logits = (change_logits * mask_float).sum(-1) / valid_count
        change_distribution = Bernoulli(logits=team_logits)
        gap_distribution = Normal(
            gap_mean,
            self.gap_log_std.exp().clamp(0.08, 1.0),
        )
        team_change = (
            (team_logits > 0.0).float()
            if deterministic
            else change_distribution.sample()
        )
        change = team_change[:, None] * mask_float
        gap_raw = gap_mean if deterministic else gap_distribution.rsample()
        change_active = (raw_valid_count > 0.5).float()
        change_log_prob = change_distribution.log_prob(team_change) * change_active
        gap_log_prob = gap_distribution.log_prob(gap_raw).sum(-1) * gap_active.float()
        entropy = change_distribution.entropy() * change_active
        entropy = entropy + gap_distribution.entropy().sum(-1) * gap_active.float()
        return LanePolicySample(
            change=change,
            gap_raw=gap_raw,
            log_prob=change_log_prob + gap_log_prob,
            value=value,
            entropy=entropy,
            change_logits=change_logits,
            gap_mean=gap_mean,
        )

    def evaluate_actions(
        self,
        features: Tensor,
        change_mask: Tensor,
        gap_active: Tensor,
        change: Tensor,
        gap_raw: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        change_logits, gap_mean, value = self(features)
        mask_float = change_mask.float()
        raw_valid_count = mask_float.sum(-1)
        valid_count = raw_valid_count.clamp_min(1.0)
        team_logits = (change_logits * mask_float).sum(-1) / valid_count
        change_distribution = Bernoulli(logits=team_logits)
        gap_distribution = Normal(
            gap_mean,
            self.gap_log_std.exp().clamp(0.08, 1.0),
        )
        team_change = change.amax(-1)
        change_active = (raw_valid_count > 0.5).float()
        log_prob = change_distribution.log_prob(team_change) * change_active
        log_prob = log_prob + gap_distribution.log_prob(gap_raw).sum(-1) * gap_active.float()
        entropy = change_distribution.entropy() * change_active
        entropy = entropy + gap_distribution.entropy().sum(-1) * gap_active.float()
        return log_prob, entropy, value


def gap_raw_to_target(gap_raw: Tensor) -> Tensor:
    """Map the unconstrained shape action to a physical 9.5--12 m pair gap."""

    return 9.5 + 2.5 * torch.sigmoid(gap_raw)


def gap_target_to_raw(target: Tensor) -> Tensor:
    scaled = ((target - 9.5) / 2.5).clamp(0.02, 0.98)
    return torch.logit(scaled)

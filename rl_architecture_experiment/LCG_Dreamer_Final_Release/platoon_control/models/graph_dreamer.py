"""Distributed graph world model with quantized latent V2V communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dreamer import CategoricalRSSM, RSSMState, symlog


NODE_COUNT = 6
FORMATION_BREAK_GAP = 13.0
FORMATION_OPENING_SPEED = 0.05
FORMATION_LOOKAHEAD_SECONDS = 1.5
FORMATION_CONSENSUS_OPENING_SPEED = 0.5
FRONT_COUNT = 8
NEIGHBOR_COUNT = 12
NODE_FEATURE_DIM = 46
PHYSICAL_DIM = 7
STEP_LENGTH = 0.1


@dataclass
class GraphDreamerConfig:
    hidden_dim: int = 128
    stochastic_dim: int = 16
    stochastic_classes: int = 16
    embedding_dim: int = 256
    communication_dim: int = 16
    communication_bits: int = 8
    graph_layers: int = 2
    graph_heads: int = 4
    edge_belief_dim: int = 128
    communication_history: int = 3
    prediction_horizon: int = 30
    action_min: float = -3.0
    action_max: float = 1.2
    emergency_action_min: float = -6.0
    maximum_jerk: float = 1.5
    discount: float = 0.997
    lambda_return: float = 0.95
    entropy_coefficient: float = 1e-3
    barrier_coefficient: float = 0.1
    restart_sync_coefficient: float = 1.0
    motion_consensus_coefficient: float = 0.5
    wave_attenuation_coefficient: float = 2.0
    wave_guidance_target_gap: float = 60.0
    wave_guidance_gap_gain: float = 0.015
    wave_guidance_relative_speed_gain: float = 0.12
    wave_guidance_front_acceleration_gain: float = 0.10
    wave_guidance_speed_gain: float = 0.0
    wave_guidance_acceleration_bias: float = 0.0
    wave_guidance_blend: float = 0.0
    nominal_cruise_speed: float = 25.0
    cruise_coefficient: float = 2.0
    rejoin_coefficient: float = 1.0
    kl_free_nats: float = 0.6
    kl_dynamic_weight: float = 0.8
    kl_representation_weight: float = 0.2


@dataclass
class GraphRSSMState:
    deterministic: Tensor
    stochastic: Tensor
    logits: Tensor
    code_history: Tensor
    intention_history: Tensor
    edge_belief: Tensor

    @property
    def feature(self) -> Tensor:
        return torch.cat(
            (self.deterministic, self.stochastic.flatten(-2)), dim=-1
        )

    def detach(self) -> "GraphRSSMState":
        return GraphRSSMState(
            self.deterministic.detach(),
            self.stochastic.detach(),
            self.logits.detach(),
            self.code_history.detach(),
            self.intention_history.detach(),
            self.edge_belief.detach(),
        )


@dataclass
class GraphPolicyOutput:
    raw_acceleration: Tensor
    safe_acceleration: Tensor
    mean_acceleration: Tensor
    standard_deviation: Tensor
    entropy: Tensor
    barrier_correction: Tensor


def physical_actor_priors(
    raw_acceleration: Tensor,
    physical: Tensor,
    nominal_cruise_speed: float = 25.0,
) -> dict[str, Tensor]:
    """Physical action priors for cruise, synchronized restart, and rejoin."""

    action = raw_acceleration.squeeze(-1)
    speed = physical[..., 0].clamp_min(0.0)
    gap = physical[..., 2]
    relative_speed = physical[..., 3]
    advisory = physical[..., 6].clamp(1.0, nominal_cruise_speed)
    front_acceleration = physical[..., 4]
    ttc = physical[..., 5]

    leader_without_front = gap[:, 0] <= 0.5
    cruise_target = (
        0.35 * (advisory[:, 0] - speed[:, 0])
    ).clamp(-1.0, 1.2)
    cruise_mask = leader_without_front.to(action.dtype)
    cruise_loss = cruise_mask * (action[:, 0] - cruise_target).square()

    pair_gap = gap[:, 1:]
    compact = ((pair_gap >= 5.0) & (pair_gap <= 15.0)).all(-1)
    stopped = (speed <= 0.3).all(-1)
    leader_can_move = leader_without_front | (relative_speed[:, 0] >= 0.15)
    restart_mask = (compact & stopped & leader_can_move).to(action.dtype)
    restart_loss = restart_mask * (action - 1.0).square().mean(-1)

    recovery_needed = (pair_gap > 12.5) & (pair_gap < 500.0)
    recovery_mask = recovery_needed.to(action.dtype)
    relative_target = (
        0.08 * (pair_gap - 12.0) + 0.30 * relative_speed[:, 1:]
    ).clamp(0.0, 1.2)
    relative_action = action[:, 1:] - action[:, :-1]
    rejoin_pair_loss = (
        (relative_action - relative_target).square() * recovery_mask
    ).sum(-1) / recovery_mask.sum(-1).clamp_min(1.0)
    predicted_pair_gap = pair_gap + FORMATION_LOOKAHEAD_SECONDS * (
        relative_speed[:, 1:].clamp_min(0.0)
    )
    # Keep the upstream sub-platoon yielding until the broken pair is back
    # inside the formation threshold. Opening speed increases the penalty,
    # but a momentary closing trend must not release the hold prematurely.
    needs_upstream_slowdown = (
        (predicted_pair_gap > FORMATION_BREAK_GAP)
        & (pair_gap > 0.0)
        & (pair_gap < 500.0)
        & (relative_speed[:, 1:] > FORMATION_OPENING_SPEED)
    )
    rejoin_mask = (
        recovery_needed | needs_upstream_slowdown
    ).any(-1).to(action.dtype)
    upstream_hold_mask = torch.stack(
        [needs_upstream_slowdown[:, node:].any(-1) for node in range(NODE_COUNT - 1)]
        + [torch.zeros_like(rejoin_mask, dtype=torch.bool)],
        dim=-1,
    ).to(action.dtype)
    hold_strength = (
        0.05
        + 0.05
        * (predicted_pair_gap - FORMATION_BREAK_GAP).clamp_min(0.0)
        + 0.35
        * (relative_speed[:, 1:] - FORMATION_OPENING_SPEED).clamp_min(0.0)
    ).clamp(0.05, 1.1)
    upstream_min_speed = torch.stack(
        [speed[:, : pair + 1].amin(-1) for pair in range(NODE_COUNT - 1)],
        dim=-1,
    )
    upstream_hold_target = torch.stack(
        [
            -(
                (
                    hold_strength[:, node:]
                    + 0.6
                    * (
                        speed[:, node, None] - upstream_min_speed[:, node:]
                    ).clamp_min(0.0)
                ).clamp_max(1.2)
                * needs_upstream_slowdown[:, node:].to(action.dtype)
            ).amax(-1)
            for node in range(NODE_COUNT - 1)
        ]
        + [torch.zeros_like(rejoin_mask)],
        dim=-1,
    )
    upstream_hold_loss = (
        (action - upstream_hold_target).square() * upstream_hold_mask
    ).sum(-1) / upstream_hold_mask.sum(-1).clamp_min(1.0)
    rejoin_loss = rejoin_mask * (rejoin_pair_loss + upstream_hold_loss)
    leader_front_present = gap[:, 0] > 0.5
    # Older protocol-compatible observations do not expose the certified
    # leader's predecessor acceleration. In that case relative speed provides
    # a measured disturbance trigger; when acceleration is available it adds
    # the faster feed-forward response without changing the state schema.
    wave_active = (
        (front_acceleration[:, 0].abs() >= 0.4)
        | (relative_speed[:, 0].abs() >= 0.4)
    )
    buffer_available = (gap[:, 0] >= 20.0) & (ttc[:, 0] >= 3.5)
    wave_mask = (
        leader_front_present & wave_active & buffer_available & compact
    ).to(action.dtype)
    # Absorb the source disturbance in a controllable front-gap buffer. The
    # feed-forward term transmits only part of the source acceleration; gap
    # and relative-speed feedback preserve mobility and close the buffer after
    # the pulse. All six actions follow the smoothed leader target so internal
    # synchronization no longer implies full disturbance transmission.
    wave_target = (
        0.35 * front_acceleration[:, 0]
        + 0.06 * (gap[:, 0] - 30.0)
        + 0.30 * relative_speed[:, 0]
    ).clamp(-1.2, 0.8)
    wave_loss = wave_mask * (
        action - wave_target[:, None]
    ).square().mean(-1)
    return {
        "cruise_loss": cruise_loss,
        "cruise_mask": cruise_mask,
        "restart_loss": restart_loss,
        "restart_mask": restart_mask,
        "rejoin_loss": rejoin_loss,
        "rejoin_mask": rejoin_mask,
        "wave_loss": wave_loss,
        "wave_mask": wave_mask,
    }


def apply_compact_cruise_guidance(
    acceleration: Tensor,
    physical: Tensor,
    previous_action: Tensor,
    previous_intention: Tensor | None = None,
    nominal_cruise_speed: float = 25.0,
    recovery_latch: Tensor | None = None,
) -> Tensor:
    """Provide a shared progress floor for a compact, low-risk platoon.

    The learned actor remains responsible for traffic following, curves, and
    formation recovery. In addition to clear-road cruise, a distant stable
    front vehicle may provide a bounded traffic-speed target. The floor is
    shared by all six nodes and is disabled by braking intent, fast closing,
    front braking, or a latched formation recovery. The downstream CBF still
    projects every requested action into its local safe interval.
    """

    speed = physical[..., 0].clamp_min(0.0)
    gap = physical[..., 2]
    relative_speed = physical[..., 3]
    front_acceleration = physical[..., 4]
    advisory = physical[..., 6].clamp(1.0, nominal_cruise_speed)
    leader_without_front = gap[:, 0] <= 0.5
    compact = ((gap[:, 1:] >= 5.0) & (gap[:, 1:] <= 15.0)).all(-1)
    # Broadcast the last actually applied leader brake to the followers.  The
    # leader itself is never capped by its previous action, otherwise a single
    # exogenous brake could latch the platoon in a permanent stopped state.
    # Using applied action also avoids treating an uncalibrated actor residual
    # as a genuine high-level braking intention during initial handoff.
    if previous_intention is None:
        previous_intention = previous_action
    brake_reference = previous_action[:, 0, 0]
    intended_brake = previous_intention[:, 0, 0] < -0.1
    recovery_active = (
        recovery_latch.to(torch.bool).any(-1)
        if recovery_latch is not None
        else torch.zeros_like(compact)
    )
    stable_pair_motion = relative_speed[:, 1:].abs().amax(-1) < 1.0
    leader_front_speed = (speed[:, 0] + relative_speed[:, 0]).clamp_min(0.0)
    no_braking_intent = ~intended_brake & (brake_reference >= -0.1)
    distant_stable_front = (
        (gap[:, 0] >= 35.0)
        & (relative_speed[:, 0] >= -1.0)
        & (front_acceleration[:, 0] >= -0.5)
        & stable_pair_motion
        & no_braking_intent
        & ~recovery_active
    )
    braking_team_compact = compact & ~recovery_active
    shared_braking = (
        braking_team_compact & intended_brake & (brake_reference < -0.1)
    )
    follower_ceiling = brake_reference[:, None, None].expand(
        -1, NODE_COUNT - 1, 1
    )
    guided_followers = torch.where(
        shared_braking[:, None, None],
        torch.minimum(acceleration[:, 1:], follower_ceiling),
        acceleration[:, 1:],
    )
    acceleration = torch.cat((acceleration[:, :1], guided_followers), dim=1)

    distant_front_target = torch.minimum(
        advisory[:, 0],
        leader_front_speed + 0.15 * (gap[:, 0] - 20.0).clamp_min(0.0),
    )
    leader_target = torch.where(
        leader_without_front,
        advisory[:, 0],
        distant_front_target,
    ).clamp(0.0, nominal_cruise_speed)
    team_target = leader_target[:, None]
    speed_error = team_target - speed
    cruise_team_compact = (
        compact
        & ~recovery_active
        & (leader_without_front | distant_stable_front)
    )
    cruise_enabled = torch.cat(
        (
            cruise_team_compact[:, None],
            (cruise_team_compact & ~shared_braking)[:, None].expand(
                -1, NODE_COUNT - 1
            ),
        ),
        dim=-1,
    )
    node_enabled = cruise_enabled & (speed_error > 0.25)
    acceleration_floor = (0.35 * speed_error).clamp(0.0, 1.2).unsqueeze(-1)
    guided = torch.where(
        node_enabled.unsqueeze(-1),
        torch.maximum(acceleration, acceleration_floor),
        acceleration,
    )

    # On a clear road, a compact platoon has one unambiguous physical target.
    # Use the same bounded speed controller for every node, with local spacing
    # and relative-speed feedback for followers. This removes seed-dependent
    # actor residuals around the cruise setpoint that can slowly open one link,
    # trip formation recovery, and create a needless long braking cycle.
    # Any external front, commanded braking, or latched recovery disables this
    # narrow prior so the learned actor and CBF retain authority in traffic.
    free_road_regulation = (
        compact
        & leader_without_front
        & ~recovery_active
        & ~shared_braking
    )
    team_mean_speed = speed.mean(-1, keepdim=True)
    speed_regulation = (
        0.45 * (team_target - speed)
        + 0.25 * (team_mean_speed - speed)
    )
    follower_regulation = (
        speed_regulation[:, 1:]
        + 0.30 * relative_speed[:, 1:]
        + 0.06 * (gap[:, 1:] - 12.0)
    )
    physical_regulation = torch.cat(
        (speed_regulation[:, :1], follower_regulation), dim=-1
    ).clamp(-0.8, 1.2)
    guided = torch.where(
        free_road_regulation[:, None, None],
        physical_regulation.unsqueeze(-1),
        guided,
    )

    team_stopped = (speed <= 0.3).all(-1)
    restart_formation = (
        (gap[:, 1:] >= 5.0) & (gap[:, 1:] <= 20.0)
    ).all(-1)
    leader_can_restart = leader_without_front | (relative_speed[:, 0] >= 0.15)
    synchronized_restart = team_stopped & restart_formation & leader_can_restart
    pair_gap = gap[:, 1:]
    communicated_expansion = (
        recovery_latch.to(torch.bool)
        if recovery_latch is not None
        else pair_gap > FORMATION_BREAK_GAP
    )
    predicted_pair_gap = pair_gap + FORMATION_LOOKAHEAD_SECONDS * (
        relative_speed[:, 1:].clamp_min(0.0)
    )
    predicted_expansion = (
        (pair_gap > 0.0)
        & (pair_gap < 500.0)
        & (relative_speed[:, 1:] > FORMATION_OPENING_SPEED)
        & (predicted_pair_gap > FORMATION_BREAK_GAP)
    )
    expanded = (communicated_expansion | predicted_expansion) & (
        pair_gap < 500.0
    )
    widening = expanded & (relative_speed[:, 1:] > FORMATION_OPENING_SPEED)
    upstream_masks = torch.stack(
        [expanded[:, node:].any(-1) for node in range(NODE_COUNT - 1)]
        + [torch.zeros_like(expanded[:, 0])],
        dim=-1,
    )
    downstream_masks = torch.stack(
        [torch.zeros_like(expanded[:, 0])]
        + [expanded[:, :node].any(-1) for node in range(1, NODE_COUNT)],
        dim=-1,
    )
    recovery_waiting = team_stopped & ~leader_can_restart
    upstream_masks = upstream_masks & ~recovery_waiting[:, None]
    downstream_masks = downstream_masks & ~recovery_waiting[:, None]
    upstream_masks = upstream_masks & ~synchronized_restart[:, None]
    downstream_masks = downstream_masks & ~synchronized_restart[:, None]
    gap_excess = (
        torch.maximum(pair_gap, predicted_pair_gap) - 12.5
    ).clamp_min(0.0)
    pair_brake_strength = (
        0.05 * gap_excess
        + 0.35
        * (relative_speed[:, 1:] - FORMATION_OPENING_SPEED).clamp_min(0.0)
    ) * widening.to(gap_excess.dtype)
    upstream_strength = torch.stack(
        [pair_brake_strength[:, node:].amax(-1) for node in range(NODE_COUNT - 1)]
        + [torch.zeros_like(gap_excess[:, 0])],
        dim=-1,
    )
    downstream_strength = torch.stack(
        [torch.zeros_like(gap_excess[:, 0])]
        + [gap_excess[:, :node].amax(-1) for node in range(1, NODE_COUNT)],
        dim=-1,
    )
    upstream_target = -upstream_strength.clamp_max(0.8)
    downstream_target = (0.15 + 0.12 * downstream_strength).clamp_max(1.2)
    guided = torch.where(
        upstream_masks.unsqueeze(-1),
        torch.minimum(guided, upstream_target.unsqueeze(-1)),
        guided,
    )
    guided = torch.where(
        downstream_masks.unsqueeze(-1),
        torch.maximum(guided, downstream_target.unsqueeze(-1)),
        guided,
    )
    # Once every split link is closing, a free-road leader only needs to creep
    # while downstream nodes finish rejoining. A full stop is unnecessary and
    # is indistinguishable from a clear-path policy stall to the environment.
    recovering = expanded.any(-1)
    widening_any = widening.any(-1)
    free_leader_creep = (
        leader_without_front
        & recovering
        & ~widening_any
        & (speed[:, 0] < 2.0)
    )
    creep_target = (
        0.15 + 0.45 * (2.0 - speed[:, 0]).clamp_min(0.0)
    ).clamp_max(0.9)
    guided_leader = torch.where(
        free_leader_creep[:, None, None],
        torch.maximum(guided[:, :1], creep_target[:, None, None]),
        guided[:, :1],
    )
    guided = torch.cat((guided_leader, guided[:, 1:]), dim=1)
    guided = torch.where(
        recovery_waiting[:, None, None],
        torch.minimum(guided, torch.zeros_like(guided)),
        guided,
    )
    guided = torch.where(
        synchronized_restart[:, None, None],
        torch.maximum(guided, torch.full_like(guided, 1.0)),
        guided,
    )
    return guided


def apply_wave_buffer_guidance(
    acceleration: Tensor,
    physical: Tensor,
    config: GraphDreamerConfig,
) -> Tensor:
    """Use a safe front-gap buffer to attenuate a measured traffic wave."""

    if config.wave_guidance_blend <= 0.0:
        return acceleration
    gap = physical[..., 2]
    relative_speed = physical[..., 3]
    front_acceleration = physical[..., 4]
    ttc = physical[..., 5]
    speed = physical[..., 0].clamp_min(0.0)
    advisory = physical[..., 6].clamp(1.0, config.nominal_cruise_speed)
    compact = ((gap[:, 1:] >= 5.0) & (gap[:, 1:] <= 15.0)).all(-1)
    disturbance = (
        (relative_speed[:, 0].abs() >= 0.25)
        | (front_acceleration[:, 0].abs() >= 0.4)
        | (
            (gap[:, 0] - config.wave_guidance_target_gap).abs()
            >= 8.0
        )
    )
    active = (
        compact
        & (gap[:, 0] >= 20.0)
        & (ttc[:, 0] >= 3.5)
        & disturbance
    )
    target = (
        config.wave_guidance_acceleration_bias
        + config.wave_guidance_speed_gain
        * (advisory[:, 0] - speed.mean(-1))
        +
        config.wave_guidance_gap_gain
        * (gap[:, 0] - config.wave_guidance_target_gap)
        + config.wave_guidance_relative_speed_gain * relative_speed[:, 0]
        + config.wave_guidance_front_acceleration_gain
        * front_acceleration[:, 0]
    ).clamp(-0.8, 0.8)
    shared_target = target[:, None, None].expand_as(acceleration)
    blend = float(min(1.0, max(0.0, config.wave_guidance_blend)))
    guided = (1.0 - blend) * acceleration + blend * shared_target
    return torch.where(active[:, None, None], guided, acceleration)


def apply_team_motion_consensus(
    acceleration: Tensor,
    physical: Tensor,
    recovery_latch: Tensor | None = None,
) -> Tensor:
    """Coordinate compact low-speed motion behind an external front vehicle.

    The leader's already projected action is the causal team reference. A
    bounded front-to-rear speed profile opens compressed stop gaps without
    allowing a launch wave to accumulate across the five follower links.
    Local hard-risk nodes retain the more conservative CBF action.
    """

    speed = physical[..., 0]
    gap = physical[..., 2]
    relative_speed = physical[..., 3]
    pair_gap = gap[:, 1:]
    pair_closing = (-relative_speed[:, 1:]).clamp_min(0.0)
    leader_has_front = gap[:, 0] > 0.5
    connected = ((pair_gap >= 5.0) & (pair_gap <= 15.0)).all(-1)
    if recovery_latch is not None:
        latched_links = recovery_latch.to(torch.bool)
        recovery_active = latched_links.any(-1)
        recovery_still_opening = (
            latched_links
            & (
                relative_speed[:, 1:]
                > FORMATION_CONSENSUS_OPENING_SPEED
            )
        ).any(-1)
        recovery_settled = recovery_active & ~recovery_still_opening
    else:
        recovery_settled = torch.zeros_like(leader_has_front)
    # Formation recovery intentionally creates a bounded relative acceleration
    # across a split link.  Applying the compact-team consensus at the same
    # time erases that closing command once speeds become nearly equal and can
    # latch a harmless 13 m gap indefinitely.  The causal CBF remains active;
    # only the competing low-speed synchronization prior yields here.
    consensus_active = leader_has_front & connected & ~recovery_settled

    launch_scale = (speed.amin(-1) / 1.0).clamp(0.0, 1.0)
    desired_pair_opening = (
        0.025 * (12.0 - pair_gap).clamp(0.0, 4.0)
    ).clamp_max(0.08) * launch_scale[:, None]
    cumulative_opening = torch.cat(
        (
            torch.zeros_like(speed[:, :1]),
            desired_pair_opening.cumsum(-1),
        ),
        dim=-1,
    )
    desired_speed = speed[:, :1] - cumulative_opening
    speed_feedback = 1.2 * (desired_speed - speed)
    leader_reference = acceleration[:, :1, 0]
    coordinated = (
        leader_reference + speed_feedback
    ).clamp(-3.0, 1.2)

    hard_risk = torch.cat(
        (
            torch.zeros_like(speed[:, :1], dtype=torch.bool),
            (pair_gap < 5.5)
            | (pair_closing > 0.5)
            | (pair_gap / pair_closing.clamp_min(1e-3) < 2.0),
        ),
        dim=-1,
    )
    coordinated = torch.where(
        hard_risk,
        torch.minimum(coordinated, acceleration[..., 0]),
        coordinated,
    )
    return torch.where(
        consensus_active[:, None, None],
        coordinated.unsqueeze(-1),
        acceleration,
    )


def _masked_pool(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.unsqueeze(-1).to(values.dtype)
    mean = (values * mask).sum(-2) / mask.sum(-2).clamp_min(1.0)
    maximum = values.masked_fill(mask <= 0.0, -1e4).amax(-2)
    maximum = torch.where(mask.any(-2), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class GraphObservationEncoder(nn.Module):
    """Encode compact per-node observations without repeated history windows."""

    def __init__(self, config: GraphDreamerConfig) -> None:
        super().__init__()
        self.register_buffer(
            "node_scale",
            torch.tensor(
                [
                    3500.0,
                    60.0,
                    15.0,
                    30.0,
                    3.0,
                    1.0,
                    0.1,
                    1.0,
                    *([500.0, 60.0, 15.0, 3.0, 500.0, 1.0] * 2),
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    500.0,
                    60.0,
                    1.0,
                    40.0,
                    240.0,
                    6.0,
                    *([60.0] * 10),
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        )
        self.register_buffer("road_scale", torch.tensor([500.0, 0.1, 40.0]))
        self.register_buffer("v2i_scale", torch.tensor([60.0, 300.0, 60.0, 1.0]))
        self.register_buffer(
            "front_scale", torch.tensor([1000.0, 3.0, 60.0, 15.0, 30.0, 1.0])
        )
        self.register_buffer(
            "neighbor_scale",
            torch.tensor([500.0, 3.0, 60.0, 15.0, 30.0, 3.0, 1.0, 1.0]),
        )
        self.node = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, 96), nn.SiLU(), nn.LayerNorm(96)
        )
        self.road = nn.Sequential(nn.Linear(60, 48), nn.SiLU())
        self.v2i = nn.Sequential(nn.Linear(32, 48), nn.SiLU())
        self.front = nn.Sequential(nn.Linear(6, 32), nn.SiLU())
        self.neighbor = nn.Sequential(nn.Linear(8, 32), nn.SiLU())
        self.role = nn.Embedding(2, 16)
        self.output = nn.Sequential(
            nn.Linear(96 + 48 + 48 + 64 + 64 + 16, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        node = self.node(observation["node_features"] / self.node_scale)
        road = self.road(
            (observation["road_profile"] / self.road_scale).flatten(-2)
        )
        v2i = self.v2i((observation["v2i_flow"] / self.v2i_scale).flatten(-2))
        front = self.front(observation["front_entities"] / self.front_scale)
        front = _masked_pool(front, observation["front_mask"])
        neighbors = self.neighbor(
            observation["neighbor_entities"] / self.neighbor_scale
        )
        neighbors = _masked_pool(neighbors, observation["neighbor_mask"])
        role = self.role(observation["role"].long())
        return self.output(torch.cat((node, road, v2i, front, neighbors, role), -1))


class QuantizedSemanticChannel(nn.Module):
    def __init__(self, feature_dim: int, config: GraphDreamerConfig) -> None:
        super().__init__()
        self.bits = config.communication_bits
        self.quantization_enabled = True
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.communication_dim),
            nn.Tanh(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        code = self.encoder(feature)
        if not self.quantization_enabled:
            return code
        levels = float(2**self.bits - 1)
        quantized = torch.round((code + 1.0) * 0.5 * levels)
        quantized = quantized * (2.0 / levels) - 1.0
        if self.training:
            return code + (quantized - code).detach()
        return quantized


class LatentGraphMessenger(nn.Module):
    """Aggregate delayed semantic packets and maintain receiver-side edge beliefs."""

    def __init__(self, config: GraphDreamerConfig) -> None:
        super().__init__()
        self.config = config
        edge_input = config.communication_dim + 1 + 1 + 2
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input, config.edge_belief_dim), nn.SiLU()
        )
        self.edge_gru = nn.GRUCell(config.edge_belief_dim, config.edge_belief_dim)
        self.relation_gate = nn.Sequential(
            nn.Linear(config.edge_belief_dim + 2, config.edge_belief_dim),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(config.edge_belief_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )

    def _select_history(self, history: Tensor, ages: Tensor) -> Tensor:
        # history: [B, K, sender, C], ages: [B, receiver, sender, relation]
        delay = torch.round(ages / STEP_LENGTH).long().clamp(
            0, history.shape[1] - 1
        )
        selected = history.new_zeros(
            *delay.shape, history.shape[-1]
        )
        for lag in range(history.shape[1]):
            sender_value = history[:, history.shape[1] - 1 - lag]
            expanded = sender_value[:, None, :, None, :].expand_as(selected)
            selected = torch.where((delay == lag).unsqueeze(-1), expanded, selected)
        return selected

    def forward(
        self,
        code_history: Tensor,
        intention_history: Tensor,
        adjacency: Tensor,
        ages: Tensor,
        previous_belief: Tensor,
        communication_dropout: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        codes = self._select_history(code_history, ages)
        intentions = self._select_history(intention_history, ages)
        relation = torch.eye(2, device=codes.device, dtype=codes.dtype)
        relation = relation.view(1, 1, 1, 2, 2).expand(
            *codes.shape[:-1], 2
        )
        if self.training and communication_dropout > 0.0:
            drop = torch.rand_like(adjacency[..., 1]) < communication_dropout
            keep = torch.ones_like(adjacency)
            keep[..., 1] = (~drop).to(adjacency.dtype)
            adjacency = adjacency * keep
        edge_input = torch.cat((codes, intentions, ages.unsqueeze(-1), relation), -1)
        encoded = self.edge_encoder(edge_input)
        # A dropped packet may evolve the receiver-side belief from memory,
        # but it must not expose the sender's current code or intention.
        encoded = encoded * adjacency.unsqueeze(-1)
        flat_encoded = encoded.flatten(0, -2)
        flat_previous = previous_belief.flatten(0, -2)
        belief = self.edge_gru(flat_encoded, flat_previous).view_as(previous_belief)
        gate_input = torch.cat((belief, relation), -1)
        belief = belief * self.relation_gate(gate_input)
        masked = belief * adjacency.unsqueeze(-1)
        degree = adjacency.sum(dim=(-2, -1), keepdim=False).clamp_min(1.0)
        context = masked.sum(dim=(-3, -2)) / degree.unsqueeze(-1)
        return self.output(context), belief


class SharedRoleActor(nn.Module):
    def __init__(self, feature_dim: int, config: GraphDreamerConfig) -> None:
        super().__init__()
        self.action_min = config.action_min
        self.action_max = config.action_max
        self.role = nn.Embedding(2, 16)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + config.hidden_dim + 16, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.physical_trunk = nn.Sequential(
            nn.Linear(PHYSICAL_DIM, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        nn.init.zeros_(self.physical_trunk[-1].weight)
        nn.init.zeros_(self.physical_trunk[-1].bias)
        self.mean = nn.Linear(config.hidden_dim, 1)
        self.log_std = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.mean.weight)
        normalized_zero = 2.0 * (0.0 - config.action_min) / (
            config.action_max - config.action_min
        ) - 1.0
        nn.init.constant_(
            self.mean.bias,
            0.5 * torch.log(
                torch.tensor((1.0 + normalized_zero) / (1.0 - normalized_zero))
            ).item(),
        )

    def forward(
        self,
        feature: Tensor,
        graph_context: Tensor,
        physical: Tensor,
        role: Tensor,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = self.trunk(
            torch.cat((feature, graph_context, self.role(role.long())), -1)
        ) + self.physical_trunk(physical)
        mean = self.mean(hidden).clamp(-5.0, 5.0)
        std = self.log_std(hidden).clamp(-5.0, 1.0).exp()
        latent = mean if deterministic else mean + std * torch.randn_like(std)
        squashed = torch.tanh(latent)
        acceleration = self.action_min + 0.5 * (squashed + 1.0) * (
            self.action_max - self.action_min
        )
        mean_squashed = torch.tanh(mean)
        mean_acceleration = self.action_min + 0.5 * (mean_squashed + 1.0) * (
            self.action_max - self.action_min
        )
        entropy = 0.5 * torch.log(2.0 * torch.pi * torch.e * std.square())
        return acceleration, mean_acceleration, std, entropy


class DifferentiableCBFProjector(nn.Module):
    """Smoothly project raw acceleration into a conservative feasible interval."""

    def __init__(self, config: GraphDreamerConfig, beta: float = 12.0) -> None:
        super().__init__()
        self.config = config
        self.beta = beta

    def _smooth_min(self, *values: Tensor) -> Tensor:
        return -torch.logsumexp(
            torch.stack(tuple(-self.beta * item for item in values), 0), 0
        ) / self.beta

    def _smooth_max(self, *values: Tensor) -> Tensor:
        return torch.logsumexp(
            torch.stack(tuple(self.beta * item for item in values), 0), 0
        ) / self.beta

    def forward(
        self, raw_action: Tensor, physical: Tensor, previous_action: Tensor
    ) -> tuple[Tensor, Tensor]:
        speed = physical[..., 0:1].clamp_min(0.0)
        gap = physical[..., 2:3]
        relative_speed = physical[..., 3:4]
        observed_front_acceleration = physical[..., 4:5].clamp(-6.0, 3.0)
        advisory = physical[..., 6:7].clamp(
            1.0, self.config.nominal_cruise_speed
        )
        advisory_upper = torch.where(
            speed > advisory + 0.2,
            torch.full_like(raw_action, -1.0),
            torch.full_like(raw_action, self.config.action_max),
        )
        valid_front = (gap > 0.0).to(raw_action.dtype)
        compact_team = (
            (gap[:, 1:] >= 5.0) & (gap[:, 1:] <= 15.0)
        ).all(dim=1, keepdim=True)
        rejoined_team = (
            (gap[:, 1:] >= 5.0) & (gap[:, 1:] <= 12.5)
        ).all(dim=1, keepdim=True)
        restart_team = (
            (gap[:, 1:] >= 5.0) & (gap[:, 1:] <= 20.0)
        ).all(dim=1, keepdim=True)
        team_stopped = (speed <= 0.3).all(dim=1, keepdim=True)
        team_stationary = (speed <= 0.1).all(dim=1, keepdim=True)
        team_restart_candidate = restart_team & team_stopped
        closing = (-relative_speed).clamp_min(0.0)
        front_speed = (speed + relative_speed).clamp_min(0.0)
        required_gap = torch.maximum(
            gap.new_tensor(5.0), 3.0 + 0.35 * speed + 0.8 * closing
        )
        cbf_horizon = 1.0
        cbf_margin = gap - 5.0 - 1.5 * closing
        ttc = gap / closing.clamp_min(1e-3)
        raw_gap = (gap - 2.0).clamp_min(0.0)
        required_deceleration = (
            (speed.square() - front_speed.square()).clamp_min(0.0)
            / (2.0 * (raw_gap - 5.0).clamp_min(0.5))
        )
        # Bound closing speed by the kinetic energy that can be dissipated
        # before the minimum gap. The short-horizon CBF alone permits a split
        # follower to catch too aggressively and then hit the environment's
        # hard required-deceleration shield, creating a braking shockwave.
        kinetic_horizon = 1.0
        # physical gap includes SUMO's 2 m platoon minGap; the environment's
        # required-deceleration shield reserves another 5 m safety margin.
        usable_gap = (gap - 7.0).clamp_min(0.5)
        local_restart_release = (speed <= 0.3) & (
            (valid_front < 0.5) | (relative_speed >= 0.15)
        )
        leader_node = torch.zeros_like(valid_front, dtype=torch.bool)
        leader_node[:, :1] = True
        opening_brake_release = (
            leader_node
            & (valid_front > 0.5)
            & (relative_speed >= 0.2)
            & (
                gap
                >= torch.maximum(
                    required_gap + 2.0,
                    torch.full_like(required_gap, 10.0),
                )
            )
            & (observed_front_acceleration >= -0.5)
            & (raw_action > 0.05)
            & (previous_action < -0.05)
        )
        leader_restart = local_restart_release[:, :1]
        team_restart_release = team_stopped & restart_team & leader_restart
        restart_release = (
            local_restart_release
            | team_restart_release
            | opening_brake_release
        )
        jerk_delta = torch.where(
            restart_release,
            torch.full_like(raw_action, self.config.action_max - self.config.action_min),
            torch.full_like(raw_action, self.config.maximum_jerk * STEP_LENGTH),
        )
        pair_gap = gap[:, 1:, 0]
        pair_opening_speed = relative_speed[:, 1:, 0]
        predicted_pair_gap = pair_gap + FORMATION_LOOKAHEAD_SECONDS * (
            pair_opening_speed.clamp_min(0.0)
        )
        expanded = (
            (predicted_pair_gap > FORMATION_BREAK_GAP)
            & (pair_gap > 0.0)
            & (pair_gap < 500.0)
            & (pair_opening_speed > FORMATION_OPENING_SPEED)
        )
        # Once the leader has a safe restart release, a latched formation gap
        # must not keep commanding upstream braking while the whole team is
        # stopped. Cohesion limits resume as soon as the vehicles are moving.
        expanded = expanded & ~team_restart_release[:, 0]
        slowdown_strength = (
            0.05
            + 0.05
            * (predicted_pair_gap - FORMATION_BREAK_GAP).clamp_min(0.0)
            + 0.35
            * (pair_opening_speed - FORMATION_OPENING_SPEED).clamp_min(0.0)
        ).clamp(0.05, 1.1)
        node_speed = speed[..., 0]
        upstream_min_speed = torch.stack(
            [
                node_speed[:, : pair + 1].amin(-1)
                for pair in range(NODE_COUNT - 1)
            ],
            dim=-1,
        )
        cohesion_limits = []
        for node in range(NODE_COUNT - 1):
            downstream_active = expanded[:, node:]
            strongest = (
                (
                    slowdown_strength[:, node:]
                    + 0.6
                    * (
                        node_speed[:, node, None]
                        - upstream_min_speed[:, node:]
                    ).clamp_min(0.0)
                ).clamp_max(1.2)
                * downstream_active.to(raw_action.dtype)
            ).amax(-1, keepdim=True)
            cohesion_limits.append(
                torch.where(
                    downstream_active.any(-1, keepdim=True),
                    -strongest,
                    torch.full_like(strongest, self.config.action_max),
                )
            )
        cohesion_limits.append(
            torch.full_like(raw_action[:, -1], self.config.action_max)
        )
        cohesion_upper = torch.stack(cohesion_limits, dim=1)
        mild_recovery = (
            (pair_gap > 12.5) & (pair_gap <= FORMATION_BREAK_GAP)
        )
        mild_limits = []
        for node in range(NODE_COUNT - 1):
            active = mild_recovery[:, node:]
            targets = (
                raw_action[:, node + 1 :, 0]
                - 0.15 * (pair_gap[:, node:] - 12.5).clamp_min(0.0)
            )
            targets = torch.where(
                active,
                targets,
                torch.full_like(targets, self.config.action_max),
            )
            mild_limits.append(targets.amin(-1, keepdim=True))
        mild_limits.append(
            torch.full_like(raw_action[:, -1], self.config.action_max)
        )
        mild_recovery_upper = torch.stack(mild_limits, dim=1)
        team_min_speed, slowest_index = speed.min(dim=1, keepdim=True)
        slowest_raw_action = torch.gather(raw_action, 1, slowest_index)
        slowest_previous_action = torch.gather(
            previous_action, 1, slowest_index
        )
        slowest_reachable_action = torch.minimum(
            slowest_raw_action,
            slowest_previous_action
            + self.config.maximum_jerk * STEP_LENGTH,
        )
        speed_excess = (speed - team_min_speed).clamp_min(0.0)
        stable_team_motion = (
            (team_min_speed > 5.0)
            & (relative_speed[:, 1:].abs().amax(dim=1, keepdim=True) < 1.0)
            & (
                previous_action.amax(dim=1, keepdim=True)
                - previous_action.amin(dim=1, keepdim=True)
                < 0.5
            )
        )
        consensus_upper = torch.where(
            rejoined_team & stable_team_motion & (speed_excess > 0.05),
            slowest_reachable_action - 0.8 * speed_excess,
            torch.full_like(raw_action, self.config.action_max),
        )
        # With no external front vehicle, compact-team consensus should make
        # a slightly faster node coast, not initiate a braking wave. A negative
        # slowest reachable action can otherwise feed back through the jerk
        # bound on every tick and drag an otherwise clear-road platoon from
        # cruise speed to a near stop. Real front-vehicle braking, formation
        # cohesion limits, and the local CBF remain allowed to command braking.
        leader_without_front = valid_front[:, :1] < 0.5
        consensus_upper = torch.where(
            leader_without_front,
            consensus_upper.clamp_min(0.0),
            consensus_upper,
        )
        # SUMO accepts commands in causal front-to-rear order.  Mirror that
        # order inside the differentiable projector: each follower consumes the
        # final projected predecessor action from this same tick, rather than a
        # parallel raw intention or a fixed conservative margin.
        projected_nodes: list[Tensor] = []
        for node in range(NODE_COUNT):
            node_slice = slice(node, node + 1)
            node_raw = raw_action[:, node_slice]
            node_previous = previous_action[:, node_slice]
            node_jerk_delta = jerk_delta[:, node_slice]
            node_valid_front = valid_front[:, node_slice]
            node_front_acceleration = observed_front_acceleration[:, node_slice]
            if node > 0:
                node_front_acceleration = torch.minimum(
                    node_front_acceleration, projected_nodes[-1]
                )

            node_predictive_upper = node_front_acceleration + 2.0 * (
                gap[:, node_slice]
                - required_gap[:, node_slice]
                - closing[:, node_slice] * cbf_horizon
            ) / (cbf_horizon**2)
            node_cooperative_braking = torch.zeros_like(
                node_predictive_upper, dtype=torch.bool
            )
            if node > 0:
                node_cooperative_braking = (
                    (gap[:, node_slice] <= 15.0)
                    & (speed[:, node_slice] > 0.1)
                    & (projected_nodes[-1] < -0.05)
                )
                node_predictive_upper = torch.where(
                    node_cooperative_braking,
                    torch.minimum(
                        node_predictive_upper, projected_nodes[-1]
                    ),
                    node_predictive_upper,
                )
            node_predictive_guard = (
                (cbf_margin[:, node_slice] < 2.0)
                | (
                    (closing[:, node_slice] > 0.5)
                    & (ttc[:, node_slice] < 4.0)
                )
                | (required_deceleration[:, node_slice] > 2.0)
                | (node_predictive_upper < -0.1)
                | node_cooperative_braking
            ) & ~team_restart_candidate
            node_action_max = torch.full_like(
                node_raw, self.config.action_max
            )
            node_cbf_upper = torch.where(
                (node_valid_front > 0.5) & node_predictive_guard,
                node_predictive_upper,
                node_action_max,
            )

            node_predicted_front_speed = (
                front_speed[:, node_slice]
                + node_front_acceleration * kinetic_horizon
            ).clamp_min(0.0)
            node_recoverable_speed = torch.sqrt(
                node_predicted_front_speed.square()
                + 2.0 * 2.5 * usable_gap[:, node_slice]
            )
            node_kinetic_upper = (
                node_recoverable_speed - speed[:, node_slice]
            ) / kinetic_horizon
            node_kinetic_upper = (
                node_valid_front * node_kinetic_upper
                + (1.0 - node_valid_front) * self.config.action_max
            )

            node_upper = self._smooth_min(
                node_raw,
                node_kinetic_upper,
                advisory_upper[:, node_slice],
                cohesion_upper[:, node_slice],
                mild_recovery_upper[:, node_slice],
                consensus_upper[:, node_slice],
                node_previous + node_jerk_delta,
                node_action_max,
            )
            node_regular_lower = node_previous - node_jerk_delta
            node_regular_upper = node_previous + node_jerk_delta
            node_safe = torch.minimum(
                torch.maximum(node_upper, node_regular_lower),
                node_regular_upper,
            )
            node_safe = torch.where(
                (node_valid_front > 0.5) & node_predictive_guard,
                torch.minimum(node_safe, node_cbf_upper),
                node_safe,
            )
            node_hard_risk = (
                (cbf_margin[:, node_slice] < 0.0)
                | (ttc[:, node_slice] < 1.5)
                | (required_deceleration[:, node_slice] > 3.5)
                | (gap[:, node_slice] < 5.0)
            ) & (node_valid_front > 0.5)
            node_safe = torch.where(
                node_hard_risk,
                torch.minimum(node_safe, node_kinetic_upper),
                node_safe,
            ).clamp(
                self.config.emergency_action_min, self.config.action_max
            )
            projected_nodes.append(node_safe)

        safe = torch.cat(projected_nodes, dim=1)
        safe = torch.where(
            team_stationary & compact_team & ~leader_restart,
            torch.zeros_like(safe),
            safe,
        )
        return safe, (raw_action - safe).abs()


class MALatentGraphDreamer(nn.Module):
    """Shared Graph-RSSM, latent communication, and joint imagination policy."""

    architecture_id = "ma_latent_graph_dreamer_v1"

    def __init__(self, config: GraphDreamerConfig | None = None) -> None:
        super().__init__()
        self.config = config or GraphDreamerConfig()
        config = self.config
        self.encoder = GraphObservationEncoder(config)
        self.feature_dim = config.hidden_dim + config.stochastic_dim * config.stochastic_classes
        self.channel = QuantizedSemanticChannel(self.feature_dim, config)
        self.messenger = LatentGraphMessenger(config)
        self.rssm = CategoricalRSSM(
            1 + config.hidden_dim,
            config.embedding_dim,
            config.hidden_dim,
            config.stochastic_dim,
            config.stochastic_classes,
        )
        self.actor = SharedRoleActor(self.feature_dim, config)
        self.projector = DifferentiableCBFProjector(config)
        head_input = self.feature_dim + config.hidden_dim
        self.physical_head = self._head(head_input, PHYSICAL_DIM)
        self.road_head = self._head(head_input, 60)
        self.v2i_head = self._head(head_input, 32)
        self.delay_head = self._head(head_input, config.prediction_horizon * 3)
        self.cutin_head = self._head(head_input, 1)
        self.front_head = self._head(head_input, 4)
        self.reward_head = self._head(head_input, 1)
        self.continue_head = self._head(head_input, 1)
        self.value_head = self._head(head_input, 1)
        self.register_buffer(
            "physical_scale",
            torch.tensor([60.0, 15.0, 500.0, 60.0, 15.0, 10.0, 40.0]),
        )
        self.register_buffer("delay_scale", torch.tensor([3500.0, 60.0, 15.0]))
        self.register_buffer("role_template", torch.tensor([0, 1, 1, 1, 1, 1]))

    def _head(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, output_dim),
        )

    def initial(self, batch_size: int, device: torch.device) -> GraphRSSMState:
        base = self.rssm.initial(batch_size * NODE_COUNT, device)
        deterministic = base.deterministic.view(batch_size, NODE_COUNT, -1)
        stochastic = base.stochastic.view(
            batch_size,
            NODE_COUNT,
            self.config.stochastic_dim,
            self.config.stochastic_classes,
        )
        logits = base.logits.view_as(stochastic)
        return GraphRSSMState(
            deterministic,
            stochastic,
            logits,
            torch.zeros(
                batch_size,
                self.config.communication_history,
                NODE_COUNT,
                self.config.communication_dim,
                device=device,
            ),
            torch.zeros(
                batch_size,
                self.config.communication_history,
                NODE_COUNT,
                1,
                device=device,
            ),
            torch.zeros(
                batch_size,
                NODE_COUNT,
                NODE_COUNT,
                2,
                self.config.edge_belief_dim,
                device=device,
            ),
        )

    def _append_messages(
        self, state: GraphRSSMState, semantic_intention: Tensor
    ) -> tuple[Tensor, Tensor]:
        code = self.channel(state.feature)
        communicated_intention = semantic_intention.to(torch.float16).to(
            semantic_intention.dtype
        )
        code_history = torch.cat((state.code_history[:, 1:], code[:, None]), 1)
        intention_history = torch.cat(
            (state.intention_history[:, 1:], communicated_intention[:, None]), 1
        )
        return code_history, intention_history

    def _context(
        self,
        state: GraphRSSMState,
        semantic_intention: Tensor,
        adjacency: Tensor,
        ages: Tensor,
        communication_dropout: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        code_history, intention_history = self._append_messages(
            state, semantic_intention
        )
        context, belief = self.messenger(
            code_history,
            intention_history,
            adjacency,
            ages,
            state.edge_belief,
            communication_dropout,
        )
        return context, belief, code_history, intention_history

    def observe_step(
        self,
        previous: GraphRSSMState,
        previous_applied_action: Tensor,
        observation: Mapping[str, Tensor],
        adjacency: Tensor,
        ages: Tensor,
        communication_dropout: float = 0.0,
        previous_intention: Tensor | None = None,
    ) -> tuple[GraphRSSMState, GraphRSSMState, Tensor]:
        embedding = self.encoder(observation)
        semantic_intention = (
            previous_applied_action
            if previous_intention is None
            else previous_intention
        )
        context, belief, codes, intentions = self._context(
            previous,
            semantic_intention,
            adjacency,
            ages,
            communication_dropout,
        )
        batch = embedding.shape[0]
        flat_previous = RSSMState(
            previous.deterministic.reshape(batch * NODE_COUNT, -1),
            previous.stochastic.reshape(
                batch * NODE_COUNT,
                self.config.stochastic_dim,
                self.config.stochastic_classes,
            ),
            previous.logits.reshape(
                batch * NODE_COUNT,
                self.config.stochastic_dim,
                self.config.stochastic_classes,
            ),
        )
        transition = torch.cat((previous_applied_action, context), -1)
        posterior, prior = self.rssm.observe(
            flat_previous,
            transition.reshape(batch * NODE_COUNT, -1),
            embedding.reshape(batch * NODE_COUNT, -1),
        )

        def graph_state(value: RSSMState) -> GraphRSSMState:
            return GraphRSSMState(
                value.deterministic.view(batch, NODE_COUNT, -1),
                value.stochastic.view(
                    batch,
                    NODE_COUNT,
                    self.config.stochastic_dim,
                    self.config.stochastic_classes,
                ),
                value.logits.view(
                    batch,
                    NODE_COUNT,
                    self.config.stochastic_dim,
                    self.config.stochastic_classes,
                ),
                codes,
                intentions,
                belief,
            )

        return graph_state(posterior), graph_state(prior), context

    def imagine_step(
        self,
        previous: GraphRSSMState,
        action: Tensor,
        adjacency: Tensor,
        ages: Tensor,
        communication_dropout: float = 0.0,
        intention: Tensor | None = None,
    ) -> tuple[GraphRSSMState, Tensor]:
        semantic_intention = action if intention is None else intention
        context, belief, codes, intentions = self._context(
            previous,
            semantic_intention,
            adjacency,
            ages,
            communication_dropout,
        )
        batch = action.shape[0]
        flat_previous = RSSMState(
            previous.deterministic.reshape(batch * NODE_COUNT, -1),
            previous.stochastic.reshape(
                batch * NODE_COUNT,
                self.config.stochastic_dim,
                self.config.stochastic_classes,
            ),
            previous.logits.reshape(
                batch * NODE_COUNT,
                self.config.stochastic_dim,
                self.config.stochastic_classes,
            ),
        )
        transition = torch.cat((action, context), -1)
        imagined = self.rssm.imagine(
            flat_previous, transition.reshape(batch * NODE_COUNT, -1)
        )
        return (
            GraphRSSMState(
                imagined.deterministic.view(batch, NODE_COUNT, -1),
                imagined.stochastic.view(
                    batch,
                    NODE_COUNT,
                    self.config.stochastic_dim,
                    self.config.stochastic_classes,
                ),
                imagined.logits.view(
                    batch,
                    NODE_COUNT,
                    self.config.stochastic_dim,
                    self.config.stochastic_classes,
                ),
                codes,
                intentions,
                belief,
            ),
            context,
        )

    def _features(self, state: GraphRSSMState, context: Tensor) -> Tensor:
        return torch.cat((state.feature, context), -1)

    def decoded_physical(self, state: GraphRSSMState, context: Tensor) -> Tensor:
        return self.physical_head(self._features(state, context)) * self.physical_scale

    def policy(
        self,
        state: GraphRSSMState,
        context: Tensor,
        physical: Tensor,
        previous_action: Tensor,
        deterministic: bool,
        previous_intention: Tensor | None = None,
        recovery_latch: Tensor | None = None,
    ) -> GraphPolicyOutput:
        role = self.role_template.expand(state.feature.shape[0], -1)
        raw, mean, std, entropy = self.actor(
            state.feature,
            context,
            physical / self.physical_scale,
            role,
            deterministic,
        )
        raw = apply_compact_cruise_guidance(
            raw,
            physical,
            previous_action,
            previous_intention,
            self.config.nominal_cruise_speed,
            recovery_latch,
        )
        mean = apply_compact_cruise_guidance(
            mean,
            physical,
            previous_action,
            previous_intention,
            self.config.nominal_cruise_speed,
            recovery_latch,
        )
        raw = apply_wave_buffer_guidance(raw, physical, self.config)
        mean = apply_wave_buffer_guidance(mean, physical, self.config)
        safe, correction = self.projector(raw, physical, previous_action)
        safe = apply_team_motion_consensus(safe, physical, recovery_latch)
        correction = (raw - safe).abs()
        return GraphPolicyOutput(raw, safe, mean, std, entropy, correction)

    @staticmethod
    def _categorical_kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
        p = p_logits.softmax(-1)
        return (p * (p.clamp_min(1e-8).log() - q_logits.log_softmax(-1))).sum(-1)

    def observe_sequence(
        self,
        batch: Mapping[str, Tensor | Mapping[str, Tensor]],
        communication_dropout: float = 0.0,
    ) -> tuple[list[GraphRSSMState], list[GraphRSSMState], list[Tensor]]:
        observations = batch["observations"]
        actions = batch["applied_actions"]
        raw_intentions = batch.get("raw_actions", actions)
        assert isinstance(observations, Mapping)
        assert isinstance(actions, Tensor)
        assert isinstance(raw_intentions, Tensor)
        batch_size, transitions = actions.shape[:2]
        state = self.initial(batch_size, actions.device)
        zero_action = torch.zeros_like(actions[:, 0])
        initial_obs = {key: value[:, 0] for key, value in observations.items()}
        state, prior, context = self.observe_step(
            state,
            zero_action,
            initial_obs,
            initial_obs["adjacency"],
            initial_obs["edge_ages"],
            communication_dropout,
            zero_action,
        )
        states = [state]
        priors = [prior]
        contexts = [context]
        for step in range(transitions):
            current_obs = {
                key: value[:, step + 1] for key, value in observations.items()
            }
            state, prior, context = self.observe_step(
                state,
                actions[:, step],
                current_obs,
                current_obs["adjacency"],
                current_obs["edge_ages"],
                communication_dropout,
                raw_intentions[:, step],
            )
            states.append(state)
            priors.append(prior)
            contexts.append(context)
        return states, priors, contexts

    def world_model_loss(
        self,
        batch: Mapping[str, Tensor | Mapping[str, Tensor]],
        communication_dropout: float = 0.0,
    ) -> tuple[dict[str, Tensor], GraphRSSMState, Tensor]:
        observations = batch["observations"]
        assert isinstance(observations, Mapping)
        actions = batch["applied_actions"]
        rewards = batch["rewards"]
        continues = batch["continues"]
        assert isinstance(actions, Tensor)
        assert isinstance(rewards, Tensor)
        assert isinstance(continues, Tensor)
        batch_size, transitions = actions.shape[:2]
        states, priors, contexts = self.observe_sequence(
            batch, communication_dropout
        )
        totals = {
            name: actions.new_zeros(())
            for name in (
                "physical",
                "road",
                "v2i",
                "delay",
                "cutin",
                "front",
                "reward",
                "continue",
                "kl",
                "communication",
            )
        }
        for step in range(transitions):
            current_obs = {
                key: value[:, step + 1] for key, value in observations.items()
            }
            state = states[step + 1]
            prior = priors[step + 1]
            context = contexts[step + 1]
            features = self._features(state, context)
            totals["physical"] = totals["physical"] + F.smooth_l1_loss(
                self.physical_head(features),
                current_obs["physical_target"] / self.physical_scale,
            )
            totals["road"] = totals["road"] + F.smooth_l1_loss(
                self.road_head(features),
                (current_obs["road_profile"] / self.encoder.road_scale).flatten(-2),
            )
            totals["v2i"] = totals["v2i"] + F.smooth_l1_loss(
                self.v2i_head(features),
                (current_obs["v2i_flow"] / self.encoder.v2i_scale).flatten(-2),
            )
            follower_mask = current_obs["follower_mask"].unsqueeze(-1).unsqueeze(-1)
            delay_prediction = self.delay_head(features).view(
                batch_size, NODE_COUNT, self.config.prediction_horizon, 3
            )
            delay_error = F.smooth_l1_loss(
                delay_prediction,
                current_obs["delay_target"] / self.delay_scale,
                reduction="none",
            )
            totals["delay"] = totals["delay"] + (
                delay_error * follower_mask
            ).sum() / follower_mask.sum().clamp_min(1.0) / (
                self.config.prediction_horizon * 3
            )
            totals["cutin"] = totals["cutin"] + F.binary_cross_entropy_with_logits(
                self.cutin_head(features).squeeze(-1),
                current_obs["cutin_target"],
            )
            totals["front"] = totals["front"] + F.smooth_l1_loss(
                self.front_head(features), current_obs["front_target"]
            )
            totals["reward"] = totals["reward"] + F.mse_loss(
                self.reward_head(features).squeeze(-1), symlog(rewards[:, step])
            )
            totals["continue"] = totals["continue"] + F.binary_cross_entropy_with_logits(
                self.continue_head(features).squeeze(-1),
                continues[:, step, None].expand(-1, NODE_COUNT),
            )
            dynamic_kl = self._categorical_kl(
                state.logits.detach(), prior.logits
            ).mean()
            representation_kl = self._categorical_kl(
                state.logits, prior.logits.detach()
            ).mean()
            totals["kl"] = totals["kl"] + (
                self.config.kl_dynamic_weight
                * dynamic_kl.clamp_min(self.config.kl_free_nats)
                + self.config.kl_representation_weight
                * representation_kl.clamp_min(self.config.kl_free_nats)
            )
            totals["communication"] = totals["communication"] + state.code_history[
                :, -1
            ].square().mean()
        totals = {name: value / transitions for name, value in totals.items()}
        loss = (
            totals["physical"]
            + 0.5 * totals["road"]
            + 0.5 * totals["v2i"]
            + 0.5 * totals["delay"]
            + 0.25 * totals["cutin"]
            + 0.25 * totals["front"]
            + totals["reward"]
            + totals["continue"]
            + totals["kl"]
            + 1e-3 * totals["communication"]
        )
        return {"loss": loss, **totals}, states[-1], contexts[-1]

    def imagination_loss(
        self,
        start: GraphRSSMState,
        adjacency: Tensor,
        ages: Tensor,
        horizon: int,
        communication_dropout: float = 0.0,
    ) -> dict[str, Tensor]:
        state = start.detach()
        context = torch.zeros(
            state.feature.shape[0],
            NODE_COUNT,
            self.config.hidden_dim,
            device=state.feature.device,
        )
        previous_action = state.feature.new_zeros(state.feature.shape[0], NODE_COUNT, 1)
        rewards: list[Tensor] = []
        continues: list[Tensor] = []
        values: list[Tensor] = []
        entropies: list[Tensor] = []
        corrections: list[Tensor] = []
        restart_sync_losses: list[Tensor] = []
        restart_sync_masks: list[Tensor] = []
        motion_consensus_losses: list[Tensor] = []
        motion_consensus_masks: list[Tensor] = []
        cruise_losses: list[Tensor] = []
        cruise_masks: list[Tensor] = []
        rejoin_losses: list[Tensor] = []
        rejoin_masks: list[Tensor] = []
        wave_losses: list[Tensor] = []
        wave_masks: list[Tensor] = []
        for _ in range(horizon):
            features = self._features(state, context)
            physical = self.physical_head(features) * self.physical_scale
            output = self.policy(
                state, context, physical, previous_action, deterministic=False
            )
            state, context = self.imagine_step(
                state,
                output.safe_acceleration,
                adjacency,
                ages,
                communication_dropout,
                output.raw_acceleration,
            )
            features = self._features(state, context)
            next_physical = self.physical_head(features) * self.physical_scale
            # The reward head is fitted to symlog(reward). Keeping imagined
            # returns in that same bounded domain prevents the actor from
            # exploiting small out-of-distribution decoder errors through the
            # exponentially expanding symexp transform.
            rewards.append(self.reward_head(features).squeeze(-1))
            continues.append(self.continue_head(features).squeeze(-1).sigmoid())
            # The critic fits imagined targets, but its regression loss must not
            # update the actor through the imagined feature trajectory.
            values.append(self.value_head(features.detach()).squeeze(-1))
            entropies.append(output.entropy.squeeze(-1))
            corrections.append(output.barrier_correction.squeeze(-1))
            priors = physical_actor_priors(
                output.raw_acceleration,
                physical,
                self.config.nominal_cruise_speed,
            )
            cruise_losses.append(priors["cruise_loss"])
            cruise_masks.append(priors["cruise_mask"])
            restart_sync_losses.append(priors["restart_loss"])
            restart_sync_masks.append(priors["restart_mask"])
            rejoin_losses.append(priors["rejoin_loss"])
            rejoin_masks.append(priors["rejoin_mask"])
            wave_losses.append(priors["wave_loss"])
            wave_masks.append(priors["wave_mask"])
            compact_pairs = (
                (physical[:, 1:, 2] >= 5.0)
                & (physical[:, 1:, 2] <= 15.0)
            )
            pair_mask = compact_pairs.to(next_physical.dtype)
            relative_speed = next_physical[:, 1:, 3] / 3.0
            acceleration_error = (
                next_physical[:, 1:, 1]
                - next_physical[:, :-1, 1]
            ) / 3.0
            pair_consensus = (
                relative_speed.square() + acceleration_error.square()
            )
            motion_consensus_losses.append(
                (pair_consensus * pair_mask).sum(-1)
                / pair_mask.sum(-1).clamp_min(1.0)
            )
            motion_consensus_masks.append(
                (pair_mask.sum(-1) > 0.0).to(next_physical.dtype)
            )
            previous_action = output.safe_acceleration
        reward = torch.stack(rewards)
        continuation = torch.stack(continues)
        value = torch.stack(values)
        entropy = torch.stack(entropies)
        correction = torch.stack(corrections)
        restart_sync = torch.stack(restart_sync_losses).sum() / torch.stack(
            restart_sync_masks
        ).sum().clamp_min(1.0)
        cruise = torch.stack(cruise_losses).sum() / torch.stack(
            cruise_masks
        ).sum().clamp_min(1.0)
        rejoin = torch.stack(rejoin_losses).sum() / torch.stack(
            rejoin_masks
        ).sum().clamp_min(1.0)
        wave_attenuation = torch.stack(wave_losses).sum() / torch.stack(
            wave_masks
        ).sum().clamp_min(1.0)
        motion_consensus = torch.stack(motion_consensus_losses).sum() / torch.stack(
            motion_consensus_masks
        ).sum().clamp_min(1.0)
        returns = torch.zeros_like(value)
        bootstrap = value[-1].detach()
        for step in reversed(range(horizon)):
            next_value = value[step + 1].detach() if step + 1 < horizon else bootstrap
            bootstrap = reward[step] + self.config.discount * continuation[step] * (
                (1.0 - self.config.lambda_return) * next_value
                + self.config.lambda_return * bootstrap
            )
            returns[step] = bootstrap
        actor_loss = (
            -returns.mean()
            - self.config.entropy_coefficient * entropy.mean()
            + self.config.barrier_coefficient * correction.square().mean()
            + self.config.restart_sync_coefficient * restart_sync
            + self.config.motion_consensus_coefficient * motion_consensus
            + self.config.wave_attenuation_coefficient * wave_attenuation
            + self.config.cruise_coefficient * cruise
            + self.config.rejoin_coefficient * rejoin
        )
        critic_loss = F.mse_loss(value, returns.detach())
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "return": returns.mean(),
            "entropy": entropy.mean(),
            "barrier_correction": correction.mean(),
            "restart_sync": restart_sync,
            "motion_consensus": motion_consensus,
            "wave_attenuation": wave_attenuation,
            "cruise": cruise,
            "rejoin": rejoin,
        }

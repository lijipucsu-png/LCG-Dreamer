"""Training and replay orchestration for MA-Latent-Graph-Dreamer."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from platoon_sumo.env import (
    FOLLOWER_IDS,
    MAX_ACCELERATION,
    PLATOON_IDS,
    PREDECESSOR,
    TARGET_GAP,
    PlatoonParallelEnv,
)
from platoon_sumo.run_demo import baseline_actions

from .graph_agents import GraphPlatoonController
from .graph_validation import evaluate_foundation_gates, _restart_timing_metrics
from .models.graph_dreamer import (
    FORMATION_BREAK_GAP,
    FORMATION_OPENING_SPEED,
    GraphDreamerConfig,
    GraphRSSMState,
    NODE_COUNT,
    physical_actor_priors,
)
from .rl import (
    CURRICULUM_MIN_STAGE_STEPS,
    CURRICULUM_NAMES,
    CURRICULUM_WINDOW,
    RLConfig,
    _apply_correlated_follower_exploration,
    _attach_follower_predictions,
    _behavior_capability_fields,
    _candidate_validation_passes,
    _capability_meets_criteria,
    _curriculum_options,
    _stage_capability_passes,
    evaluate_controller,
)
from .training import MetricLogger


GRAPH_CHECKPOINT_FORMAT = 1
GRAPH_ACTOR_OBJECTIVE_VERSION = 11
GRAPH_ENVIRONMENT_DYNAMICS_SCHEMA_VERSION = 2
GRAPH_RESTART_METRIC_SCHEMA_VERSION = 2
GRAPH_FOUNDATION_MIN_STAGE_STEPS = 36_000
GRAPH_STABLE_STAGE_FRACTION = 0.5


@dataclass
class GraphRLConfig:
    seed: int = 42
    max_sumo_steps: int = 1_000_000
    warmup_steps: int = 20_000
    actor_warmup_steps: int = 20_000
    policy_handoff_steps: int = GRAPH_FOUNDATION_MIN_STAGE_STEPS
    # Stage-0 online learning already supplies roughly 4k world updates before
    # handoff. A short consolidation pass avoids overfitting the simple regime.
    world_pretrain_updates: int = 500
    replay_capacity: int = 30_000
    sequence_length: int = 50
    batch_size: int = 16
    update_every: int = 5
    actor_update_every: int = 20
    online_world_updates_per_episode: int = 16
    online_actor_updates_per_episode: int = 4
    candidate_rollback_cooldown_episodes: int = 1
    behavior_initial_weight: float = 3.0
    behavior_decay_updates: int = 5_000
    behavior_cutoff_updates: int = 50_000
    evaluation_interval: int = 50_000
    checkpoint_interval: int = 10_000
    evaluation_episodes: int = 5
    min_steps_before_convergence: int = 300_000
    world_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-5
    foundation_imagination_weight: float = 0.0
    imagination_weight: float = 0.05
    observed_prior_weight: float = 5.0
    foundation_event_fraction: float = 0.5
    weight_decay: float = 1e-5
    gradient_clip: float = 100.0
    device: str = "auto"
    checkpoint_dir: str = "artifacts/graph_rl_seed42"
    predictor_directory: str = "artifacts/checkpoints"
    calibration_path: str = "artifacts/calibration/calibrated_parameters.json"
    wave_model_path: str = "artifacts/checkpoints/wave_best.pt"
    episode_max_time: float = 240.0
    formation_curriculum_steps: int = 30_000
    disturbance_curriculum_steps: int = 100_000
    downstream_wave_training: bool = False
    downstream_wave_training_probability: float = 0.25
    variant: str = "full"

    def evaluation_config(self) -> RLConfig:
        return RLConfig(
            seed=self.seed,
            max_sumo_steps=self.max_sumo_steps,
            evaluation_interval=self.evaluation_interval,
            checkpoint_interval=self.checkpoint_interval,
            evaluation_episodes=self.evaluation_episodes,
            min_steps_before_convergence=self.min_steps_before_convergence,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
            calibration_path=self.calibration_path,
            wave_model_path=self.wave_model_path,
            episode_max_time=self.episode_max_time,
            formation_curriculum_steps=self.formation_curriculum_steps,
            disturbance_curriculum_steps=self.disturbance_curriculum_steps,
            downstream_wave_training=self.downstream_wave_training,
            downstream_wave_training_probability=(
                self.downstream_wave_training_probability
            ),
        )


def _stack_observations(
    observations: list[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


class GraphEpisodeReplay:
    def __init__(self, capacity_steps: int, seed: int) -> None:
        self.capacity_steps = capacity_steps
        self.episodes: list[dict[str, Any]] = []
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self._foundation_events: dict[str, list[tuple[int, int]]] | None = None

    @staticmethod
    def _upgrade_episode(episode: dict[str, Any]) -> None:
        if "behavior_valid" in episode:
            return
        requested = np.asarray(episode["requested_actions"], np.float32)
        raw = np.asarray(episode["raw_actions"], np.float32)
        episode["behavior_valid"] = list(
            (np.abs(requested - raw) <= 1e-4).astype(np.float32).squeeze(-1)
        )

    def add(self, episode: dict[str, Any]) -> None:
        length = len(episode["applied_actions"])
        if length == 0:
            return
        if len(episode["observations"]) != length + 1:
            raise ValueError("graph replay requires one more observation than transitions")
        self._upgrade_episode(episode)
        self.episodes.append(episode)
        self._foundation_events = None
        self.steps += length
        while self.steps > self.capacity_steps and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.steps -= len(removed["applied_actions"])

    def can_sample(self, sequence_length: int, batch_size: int) -> bool:
        usable = sum(
            max(0, len(episode["applied_actions"]) - sequence_length + 1)
            for episode in self.episodes
        )
        return usable >= batch_size

    def _foundation_event_candidates(self) -> dict[str, list[tuple[int, int]]]:
        if self._foundation_events is not None:
            return self._foundation_events
        events: dict[str, list[tuple[int, int]]] = {
            "restart": [],
            "rejoin": [],
            "consensus": [],
            "wave": [],
        }
        for episode_index, episode in enumerate(self.episodes):
            physical = np.asarray(
                [
                    observation["physical_target"]
                    for observation in episode["observations"][:-1]
                ],
                np.float32,
            )
            requested = np.asarray(episode["requested_actions"], np.float32)
            requested = requested.squeeze(-1)
            restart = (
                (physical[:, :, 0].max(-1) <= 3.0)
                & (requested.mean(-1) > 0.2)
            )
            rejoin = physical[:, 1:, 2].max(-1) > 15.0
            compact = (
                (physical[:, 1:, 2] >= 5.0)
                & (physical[:, 1:, 2] <= 15.0)
            ).all(-1)
            consensus = compact & (np.ptp(physical[:, :, 0], axis=1) > 0.3)
            wave = (
                (physical[:, 0, 2] > 0.5)
                & (
                    (np.abs(physical[:, 0, 4]) >= 0.4)
                    | (np.abs(physical[:, 0, 3]) >= 0.4)
                )
                & (physical[:, 0, 2] >= 20.0)
                & (physical[:, 0, 5] >= 3.5)
                & compact
            )
            for name, mask in (
                ("restart", restart),
                ("rejoin", rejoin),
                ("consensus", consensus),
                ("wave", wave),
            ):
                events[name].extend(
                    (episode_index, int(anchor))
                    for anchor in np.flatnonzero(mask)
                )
        self._foundation_events = events
        return events

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        foundation_event_fraction: float = 0.0,
        event_name: str | None = None,
    ) -> dict[str, Any]:
        candidates = [
            episode
            for episode in self.episodes
            if len(episode["applied_actions"]) >= sequence_length
        ]
        if not candidates:
            raise RuntimeError("graph replay does not contain a complete sequence")
        observation_batches: list[dict[str, np.ndarray]] = []
        transition_batches: dict[str, list[np.ndarray]] = {
            "applied_actions": [],
            "requested_actions": [],
            "raw_actions": [],
            "rewards": [],
            "continues": [],
            "shielded": [],
            "behavior_valid": [],
        }
        foundation_events = (
            self._foundation_event_candidates()
            if foundation_event_fraction > 0.0 or event_name is not None
            else {}
        )
        if event_name is not None:
            if event_name not in foundation_events:
                raise ValueError(f"unknown graph replay event: {event_name}")
            if not foundation_events[event_name]:
                raise RuntimeError(
                    f"graph replay does not contain a {event_name} event"
                )
            foundation_events = {event_name: foundation_events[event_name]}
            foundation_event_fraction = 1.0
        for _ in range(batch_size):
            event_names = [name for name, events in foundation_events.items() if events]
            use_event = bool(
                event_names
                and self.rng.random() < float(foundation_event_fraction)
            )
            if use_event:
                event_name = event_names[int(self.rng.integers(len(event_names)))]
                events = foundation_events[event_name]
                episode_index, anchor = events[int(self.rng.integers(len(events)))]
                episode = self.episodes[episode_index]
                maximum_start = len(episode["applied_actions"]) - sequence_length
                minimum_event_start = max(0, anchor - sequence_length + 1)
                maximum_event_start = min(anchor, maximum_start)
                start = int(
                    self.rng.integers(minimum_event_start, maximum_event_start + 1)
                )
            else:
                episode = candidates[int(self.rng.integers(len(candidates)))]
                maximum_start = len(episode["applied_actions"]) - sequence_length
                start = int(self.rng.integers(maximum_start + 1))
            stop = start + sequence_length
            observation_batches.append(
                _stack_observations(episode["observations"][start : stop + 1])
            )
            for key in transition_batches:
                transition_batches[key].append(np.asarray(episode[key][start:stop]))
        observations = {
            key: torch.from_numpy(
                np.stack([batch[key] for batch in observation_batches]).astype(
                    np.int64 if key == "role" else np.float32
                )
            )
            for key in observation_batches[0]
        }
        return {
            "observations": observations,
            **{
                key: torch.from_numpy(np.stack(values).astype(np.float32))
                for key, values in transition_batches.items()
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_steps": self.capacity_steps,
            "episodes": self.episodes,
            "steps": self.steps,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.capacity_steps = int(state["capacity_steps"])
        self.episodes = list(state["episodes"])
        for episode in self.episodes:
            self._upgrade_episode(episode)
        self.steps = int(state["steps"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._foundation_events = None


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _role_balanced_behavior_loss(
    behavior_error: Tensor, behavior_valid: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Give the single leader role equal weight to the five shared followers."""

    leader_loss = (
        behavior_error[:, 0] * behavior_valid[:, 0]
    ).sum() / behavior_valid[:, 0].sum().clamp_min(1.0)
    follower_loss = (
        behavior_error[:, 1:] * behavior_valid[:, 1:]
    ).sum() / behavior_valid[:, 1:].sum().clamp_min(1.0)
    return 0.5 * (leader_loss + follower_loss), leader_loss, follower_loss


class GraphDreamerTrainer:
    def __init__(self, controller: GraphPlatoonController, config: GraphRLConfig) -> None:
        self.controller = controller
        self.config = config
        model = controller.model
        actor_parameters = list(model.actor.parameters())
        critic_parameters = list(model.value_head.parameters())
        excluded = {id(parameter) for parameter in actor_parameters + critic_parameters}
        world_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in excluded
        ]
        self.world_parameters = world_parameters
        self.actor_parameters = actor_parameters
        self.critic_parameters = critic_parameters
        self.optimizers = {
            "world": torch.optim.AdamW(
                world_parameters,
                lr=config.world_learning_rate,
                weight_decay=config.weight_decay,
            ),
            "actor": torch.optim.AdamW(
                actor_parameters,
                lr=config.actor_learning_rate,
                weight_decay=config.weight_decay,
            ),
            "critic": torch.optim.AdamW(
                critic_parameters,
                lr=config.actor_learning_rate,
                weight_decay=config.weight_decay,
            ),
        }
        self.actor_updates = 0

    def _variant_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        device_batch = _to_device(batch, self.controller.device)
        if self.config.variant != "no_semantic_communication":
            return device_batch
        observations = dict(device_batch["observations"])
        adjacency = observations["adjacency"].clone()
        adjacency[..., 1] = 0.0
        observations["adjacency"] = adjacency
        return {**device_batch, "observations": observations}

    def _step(
        self, names: tuple[str, ...], loss: Tensor, parameters: list[Tensor]
    ) -> None:
        for name in names:
            self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(parameters, self.config.gradient_clip)
        for name in names:
            self.optimizers[name].step()

    def train_world(
        self, batch: Mapping[str, Any], communication_dropout: float
    ) -> dict[str, float]:
        self.controller.model.train()
        device_batch = self._variant_batch(batch)
        losses, _, _ = self.controller.model.world_model_loss(
            device_batch, communication_dropout
        )
        objective = losses["loss"]
        if self.config.variant == "no_auxiliary_distillation":
            objective = (
                objective
                - 0.5 * losses["delay"]
                - 0.25 * losses["cutin"]
                - 0.25 * losses["front"]
            )
        self._step(("world",), objective, self.world_parameters)
        losses = {**losses, "optimized_loss": objective.detach()}
        return {f"graph/world_{key}": float(value.detach()) for key, value in losses.items()}

    def train_actor_critic(
        self,
        batch: Mapping[str, Any],
        horizon: int,
        communication_dropout: float,
        imagination_weight: float | None = None,
    ) -> dict[str, float]:
        self.controller.model.train()
        device_batch = self._variant_batch(batch)
        model = self.controller.model
        frozen = [
            parameter
            for parameter in self.world_parameters
            if parameter.requires_grad
        ]
        for parameter in frozen:
            parameter.requires_grad_(False)
        try:
            with torch.no_grad():
                states, _, contexts = model.observe_sequence(
                    device_batch, communication_dropout
                )
                state = states[-1]
            observations = device_batch["observations"]
            adjacency = observations["adjacency"][:, -1]
            ages = observations["edge_ages"][:, -1]
            effective_imagination_weight = (
                self.config.imagination_weight
                if imagination_weight is None
                else float(imagination_weight)
            )
            if effective_imagination_weight > 0.0:
                imagination = model.imagination_loss(
                    state, adjacency, ages, horizon, communication_dropout
                )
            else:
                zero = state.feature.new_zeros(())
                imagination = {
                    "actor_loss": zero,
                    "critic_loss": zero,
                    "return": zero,
                    "entropy": zero,
                    "barrier_correction": zero,
                    "restart_sync": zero,
                    "motion_consensus": zero,
                    "wave_attenuation": zero,
                    "cruise": zero,
                    "rejoin": zero,
                }
            behavior_states = states[:-1]

            def flatten_state(name: str) -> Tensor:
                return torch.stack(
                    [getattr(state, name) for state in behavior_states], dim=1
                ).flatten(0, 1)

            behavior_state = GraphRSSMState(
                deterministic=flatten_state("deterministic"),
                stochastic=flatten_state("stochastic"),
                logits=flatten_state("logits"),
                code_history=flatten_state("code_history"),
                intention_history=flatten_state("intention_history"),
                edge_belief=flatten_state("edge_belief"),
            )
            behavior_context = torch.stack(contexts[:-1], dim=1).flatten(0, 1)
            applied_actions = device_batch["applied_actions"]
            previous_behavior_action = torch.cat(
                (torch.zeros_like(applied_actions[:, :1]), applied_actions[:, :-1]),
                dim=1,
            ).flatten(0, 1)
            policy = model.policy(
                behavior_state,
                behavior_context,
                observations["physical_target"][:, :-1].flatten(0, 1),
                previous_behavior_action,
                deterministic=True,
            )
            requested_behavior = device_batch["requested_actions"].flatten(0, 1)
            applied_behavior = device_batch["applied_actions"].flatten(0, 1)
            shielded = device_batch["shielded"].flatten(0, 1)
            behavior_valid = device_batch["behavior_valid"].flatten(0, 1)
            requested_error = (
                policy.mean_acceleration - requested_behavior
            ).square().squeeze(-1)
            applied_error = (
                policy.safe_acceleration - applied_behavior
            ).square().squeeze(-1)
            behavior_error = torch.where(
                shielded > 0.5, applied_error, requested_error
            )
            behavior_loss, leader_behavior_loss, follower_behavior_loss = (
                _role_balanced_behavior_loss(behavior_error, behavior_valid)
            )
            observed_physical = observations["physical_target"][:, :-1].flatten(
                0, 1
            )
            observed_priors = physical_actor_priors(
                policy.mean_acceleration,
                observed_physical,
                model.config.nominal_cruise_speed,
            )

            def masked_prior(name: str, mask_name: str) -> Tensor:
                values = observed_priors[name]
                mask = observed_priors[mask_name]
                return values.sum() / mask.sum().clamp_min(1.0)

            observed_cruise = masked_prior("cruise_loss", "cruise_mask")
            observed_restart = masked_prior("restart_loss", "restart_mask")
            observed_rejoin = masked_prior("rejoin_loss", "rejoin_mask")
            observed_wave_attenuation = masked_prior(
                "wave_loss", "wave_mask"
            )
            observed_actions = policy.mean_acceleration.squeeze(-1)
            compact_pairs = (
                (observed_physical[:, 1:, 2] >= 5.0)
                & (observed_physical[:, 1:, 2] <= 15.0)
            )
            consensus_target = (
                0.5 * observed_physical[:, 1:, 3]
            ).clamp(-0.5, 0.5)
            consensus_error = (
                observed_actions[:, 1:]
                - observed_actions[:, :-1]
                - consensus_target
            ).square()
            observed_consensus = (
                consensus_error * compact_pairs.to(consensus_error.dtype)
            ).sum() / compact_pairs.sum().clamp_min(1.0)
            observed_prior_loss = (
                model.config.cruise_coefficient * observed_cruise
                + model.config.restart_sync_coefficient * observed_restart
                + model.config.rejoin_coefficient * observed_rejoin
                + model.config.motion_consensus_coefficient * observed_consensus
                + model.config.wave_attenuation_coefficient
                * observed_wave_attenuation
            )
            behavior_weight = (
                self.config.behavior_initial_weight
                * math.exp(
                    -self.actor_updates
                    / max(float(self.config.behavior_decay_updates), 1.0)
                )
                if self.actor_updates < self.config.behavior_cutoff_updates
                else 0.0
            )
            actor_loss = (
                effective_imagination_weight * imagination["actor_loss"]
                + behavior_weight * behavior_loss
                + self.config.observed_prior_weight * observed_prior_loss
            )
            if effective_imagination_weight > 0.0:
                total = actor_loss + imagination["critic_loss"]
                self._step(
                    ("actor", "critic"),
                    total,
                    self.actor_parameters + self.critic_parameters,
                )
            else:
                self._step(("actor",), actor_loss, self.actor_parameters)
            self.actor_updates += 1
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        return {
            "graph/actor_loss": float(actor_loss.detach()),
            "graph/critic_loss": float(imagination["critic_loss"].detach()),
            "graph/imagined_return": float(imagination["return"].detach()),
            "graph/entropy": float(imagination["entropy"].detach()),
            "graph/barrier_correction": float(
                imagination["barrier_correction"].detach()
            ),
            "graph/restart_sync_loss": float(
                imagination["restart_sync"].detach()
            ),
            "graph/motion_consensus_loss": float(
                imagination["motion_consensus"].detach()
            ),
            "graph/wave_attenuation_loss": float(
                imagination["wave_attenuation"].detach()
            ),
            "graph/cruise_loss": float(imagination["cruise"].detach()),
            "graph/rejoin_loss": float(imagination["rejoin"].detach()),
            "graph/behavior_loss": float(behavior_loss.detach()),
            "graph/leader_behavior_loss": float(leader_behavior_loss.detach()),
            "graph/follower_behavior_loss": float(follower_behavior_loss.detach()),
            "graph/behavior_weight": behavior_weight,
            "graph/imagination_weight": effective_imagination_weight,
            "graph/observed_prior_loss": float(observed_prior_loss.detach()),
            "graph/observed_cruise_loss": float(observed_cruise.detach()),
            "graph/observed_restart_loss": float(observed_restart.detach()),
            "graph/observed_rejoin_loss": float(observed_rejoin.detach()),
            "graph/observed_consensus_loss": float(observed_consensus.detach()),
            "graph/observed_wave_attenuation_loss": float(
                observed_wave_attenuation.detach()
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
            "actor_updates": self.actor_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name, optimizer_state in state["optimizers"].items():
            try:
                self.optimizers[name].load_state_dict(optimizer_state)
            except ValueError:
                if name != "actor":
                    raise
                self.optimizers[name].state.clear()
        self.actor_updates = int(state.get("actor_updates", 0))


def _graph_restart_capability_passes(
    capability: Mapping[str, float],
) -> bool:
    """Reject sequential or oscillatory launches even when aggregate metrics pass."""

    if capability.get("restart_metric_valid", 1.0) < 0.5:
        return True
    if capability.get("restart_detected", 0.0) < 0.5:
        return True
    return bool(
        capability.get("restart_response_seconds", float("inf")) <= 0.3
        and capability.get("restart_spread_seconds", float("inf")) <= 0.8
        and capability.get("restart_maximum_speed_spread", float("inf")) <= 0.5
    )


def _migrate_restart_capability_records(
    records: list[dict[str, float]], schema_version: int
) -> list[dict[str, float]]:
    """Quarantine legacy post-launch peaks without discarding other evidence."""

    migrated = [dict(record) for record in records]
    if schema_version < GRAPH_RESTART_METRIC_SCHEMA_VERSION:
        for record in migrated:
            record["restart_metric_valid"] = 0.0
    else:
        for record in migrated:
            record.setdefault("restart_metric_valid", 1.0)
    return migrated


def _graph_capability_meets_criteria(
    stage: int, capability: Mapping[str, float]
) -> bool:
    return bool(
        _capability_meets_criteria(stage, capability)
        and _graph_restart_capability_passes(capability)
    )


class GraphCandidateGuard:
    """Keep deployed graph policies fixed until a paired rollout accepts them."""

    def __init__(
        self,
        controller: GraphPlatoonController,
        trainer: GraphDreamerTrainer,
        rollback_cooldown_episodes: int,
    ) -> None:
        self.controller = controller
        self.trainer = trainer
        self.rollback_cooldown_episodes = max(
            0, int(rollback_cooldown_episodes)
        )
        self.accepted_model_state = copy.deepcopy(controller.state_dict())
        self.accepted_trainer_state = copy.deepcopy(trainer.state_dict())
        self.validation: dict[str, Any] | None = None
        self.cooldown_episodes = 0

    @property
    def pending(self) -> bool:
        return self.validation is not None

    def accept_current(self) -> None:
        if self.pending:
            raise RuntimeError("cannot accept a graph model before validation")
        self.accepted_model_state = copy.deepcopy(self.controller.state_dict())
        self.accepted_trainer_state = copy.deepcopy(self.trainer.state_dict())

    def begin(
        self,
        scenario_options: Mapping[str, Any],
        reset_seed: int,
        origin_capability: Mapping[str, float],
    ) -> None:
        if self.pending:
            raise RuntimeError("cannot replace an unvalidated graph candidate")
        self.validation = {
            "scenario_options": copy.deepcopy(dict(scenario_options)),
            "reset_seed": int(reset_seed),
            "origin_capability": copy.deepcopy(dict(origin_capability)),
        }

    def resolve(
        self,
        stage: int,
        capability: Mapping[str, float],
    ) -> tuple[bool, bool]:
        if self.validation is None:
            return False, False
        accepted = _candidate_validation_passes(
            stage,
            capability,
            self.validation["origin_capability"],
            actor_kind="joint",
        ) and _graph_restart_capability_passes(capability)
        if accepted:
            self.accepted_model_state = copy.deepcopy(
                self.controller.state_dict()
            )
            self.accepted_trainer_state = copy.deepcopy(
                self.trainer.state_dict()
            )
        else:
            self.controller.load_state_dict(self.accepted_model_state)
            self.trainer.load_state_dict(self.accepted_trainer_state)
            self.cooldown_episodes = self.rollback_cooldown_episodes
        self.validation = None
        return accepted, not accepted

    def consume_cooldown_episode(self, rolled_back: bool) -> bool:
        if rolled_back:
            return True
        if self.cooldown_episodes <= 0:
            return False
        self.cooldown_episodes -= 1
        return True


def _new_graph_episode(initial_observation: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "observations": [
            {key: np.asarray(value).copy() for key, value in initial_observation.items()}
        ],
        "applied_actions": [],
        "requested_actions": [],
        "raw_actions": [],
        "rewards": [],
        "continues": [],
        "shielded": [],
        "behavior_valid": [],
    }


def _append_graph_transition(
    episode: dict[str, Any],
    next_observation: Mapping[str, np.ndarray],
    actions: Mapping[str, Mapping[str, Any]],
    raw_actions: np.ndarray,
    rewards: Mapping[str, float],
    continued: bool,
    infos: Mapping[str, Mapping[str, Any]],
    behavior_invalid: bool = False,
    behavior_supervised: bool = True,
) -> None:
    requested = np.asarray(
        [float(np.asarray(actions[agent]["accel"]).reshape(-1)[0]) for agent in PLATOON_IDS],
        np.float32,
    )[:, None]
    applied = np.asarray(
        [float(infos[agent].get("applied_acceleration", requested[index, 0])) for index, agent in enumerate(PLATOON_IDS)],
        np.float32,
    )[:, None]
    shielded = np.asarray(
        [float(infos[agent].get("safety_intervened", False)) for agent in PLATOON_IDS],
        np.float32,
    )
    if behavior_invalid:
        shielded.fill(1.0)
    behavior_valid = np.ones(NODE_COUNT, np.float32)
    if not behavior_supervised:
        behavior_valid.fill(0.0)
    elif behavior_invalid:
        behavior_valid = (
            np.abs(requested - np.asarray(raw_actions, np.float32)) <= 1e-4
        ).astype(np.float32).squeeze(-1)
    episode["observations"].append(
        {key: np.asarray(value).copy() for key, value in next_observation.items()}
    )
    episode["applied_actions"].append(applied)
    episode["requested_actions"].append(requested)
    episode["raw_actions"].append(np.asarray(raw_actions, np.float32))
    episode["rewards"].append(
        np.asarray([rewards[agent] for agent in PLATOON_IDS], np.float32)
    )
    episode["continues"].append(float(continued))
    episode["shielded"].append(shielded)
    episode["behavior_valid"].append(behavior_valid)


def _communication_dropout(stage: int) -> float:
    if stage < 6:
        return 0.0
    return 0.05 if stage == 6 else float(np.random.uniform(0.0, 0.15))


def _uses_collection_expert(
    total_steps: int,
    config: GraphRLConfig,
    curriculum_stage: int,
    foundation_passed: bool,
) -> bool:
    """Keep expert collection active until the actor passes the safety gates."""

    return total_steps < config.policy_handoff_steps or (
        curriculum_stage == 0 and not foundation_passed
    )


def _collection_update_intervals(
    config: GraphRLConfig, pretraining_done: bool
) -> tuple[int, int]:
    """Reduce redundant Stage-0 gradients after world-model consolidation."""

    if not pretraining_done:
        return config.update_every, config.actor_update_every
    return max(config.update_every, 20), max(config.actor_update_every, 40)


def _post_episode_actor_update_count(
    config: GraphRLConfig, curriculum_stage: int
) -> int:
    """Use one-step trust-region candidates in disturbance-heavy stages."""

    updates = max(0, config.online_actor_updates_per_episode)
    return min(updates, 1) if curriculum_stage >= 4 else updates


def _final_evaluation_requires_refinement(
    final_stage_candidate: bool,
    evaluation: Mapping[str, float],
) -> bool:
    """Reopen learning when a frozen final candidate misses its evaluation."""

    return bool(
        final_stage_candidate
        and (
            evaluation["evaluation/capability_pass"] < 0.5
            or evaluation["evaluation/collision_rate"] > 0.0
            or evaluation["evaluation/failure_rate"] > 0.0
        )
    )


def _resume_final_refinement_pending(
    metrics_path: Path,
    checkpoint_step: int,
    final_stage_window_passed: bool,
) -> bool:
    """Recover an unsaved final-refinement request from the metric journal."""

    if not final_stage_window_passed or not metrics_path.exists():
        return False
    latest_evaluation: dict[str, Any] | None = None
    with metrics_path.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "evaluation/capability_pass" in record:
                latest_evaluation = record
    if latest_evaluation is None:
        return False
    if int(latest_evaluation.get("step", -1)) != checkpoint_step:
        return False
    return _final_evaluation_requires_refinement(True, latest_evaluation)


def _graph_stage_capability_passes(
    stage: int,
    stage_steps: int,
    records: list[dict[str, float]],
) -> tuple[bool, dict[str, float]]:
    """Gate stages on either their full budget or a stable no-update plateau."""

    effective_steps = stage_steps
    if stage == 0 and stage_steps >= GRAPH_FOUNDATION_MIN_STAGE_STEPS:
        effective_steps = 40_000
    elif 0 < stage < len(CURRICULUM_MIN_STAGE_STEPS):
        stable_step_floor = math.ceil(
            CURRICULUM_MIN_STAGE_STEPS[stage] * GRAPH_STABLE_STAGE_FRACTION
        )
        stable_records = records[-CURRICULUM_WINDOW:]
        stable_plateau = bool(
            stage_steps >= stable_step_floor
            and len(stable_records) == CURRICULUM_WINDOW
            and all(
                _graph_capability_meets_criteria(stage, record)
                for record in stable_records
            )
        )
        if stable_plateau:
            effective_steps = CURRICULUM_MIN_STAGE_STEPS[stage]
    passed, summary = _stage_capability_passes(
        stage, effective_steps, records
    )
    restart_window = records[-CURRICULUM_WINDOW:]
    if passed and not all(
        _graph_restart_capability_passes(record)
        for record in restart_window
    ):
        passed = False
    return passed, summary


def _formation_recovery_context(
    observations: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pair gaps, opening speeds, and pairs outside the break limit."""

    platoon = np.asarray(observations["leader"]["platoon_state"], np.float32)
    if platoon.shape != (NODE_COUNT, 6):
        raise ValueError(
            f"platoon_state must have shape {(NODE_COUNT, 6)}, got {platoon.shape}"
        )
    valid = (platoon[:-1, 5] > 0.5) & (platoon[1:, 5] > 0.5)
    pair_gaps = platoon[1:, 4]
    opening_speeds = platoon[:-1, 1] - platoon[1:, 1]
    needs_slowdown = (
        valid
        & (pair_gaps > FORMATION_BREAK_GAP)
        & (pair_gaps < 500.0)
        & (opening_speeds > FORMATION_OPENING_SPEED)
    )
    return pair_gaps, opening_speeds, needs_slowdown


def _apply_cohesive_warmup_control(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: dict[str, dict[str, Any]],
) -> bool:
    """Keep the collection expert from escaping a disconnected formation."""

    platoon = np.asarray(observations["leader"]["platoon_state"], np.float32)
    followers = platoon[1:]
    if not np.any(followers[:, 5] > 0.5):
        return False
    pair_gaps, opening_speeds, needs_slowdown = _formation_recovery_context(
        observations
    )
    leader_speed = float(platoon[0, 1])
    current = float(np.asarray(actions["leader"]["accel"]).reshape(-1)[0])
    changed = False
    if np.any(needs_slowdown):
        slowdown_strength = np.clip(
            0.05
            + 0.05 * np.maximum(pair_gaps - FORMATION_BREAK_GAP, 0.0)
            + 0.35
            * np.maximum(opening_speeds - FORMATION_OPENING_SPEED, 0.0),
            0.05,
            1.1,
        )
        node_speeds = platoon[:, 1]
        upstream_min_speed = np.asarray(
            [
                np.min(node_speeds[: pair_index + 1])
                for pair_index in range(NODE_COUNT - 1)
            ],
            np.float32,
        )
        for node, agent in enumerate(PLATOON_IDS[:-1]):
            downstream = needs_slowdown[node:]
            if not np.any(downstream) or agent not in actions:
                continue
            speed_consensus = 0.6 * np.maximum(
                node_speeds[node] - upstream_min_speed[node:], 0.0
            )
            target = -float(
                np.max(
                    np.minimum(
                        slowdown_strength[node:] + speed_consensus,
                        1.2,
                    )[downstream]
                )
            )
            requested = float(
                np.asarray(actions[agent]["accel"]).reshape(-1)[0]
            )
            actions[agent]["accel"] = np.asarray(
                [min(requested, target)], np.float32
            )
            changed = True
    local_front = np.asarray(
        observations["leader"].get("local_front", np.zeros(6, np.float32)),
        np.float32,
    )
    front_blocks_restart = False
    if local_front.shape[-1] >= 6 and float(local_front[5]) > 0.5:
        front_speed = leader_speed + float(local_front[1])
        front_blocks_restart = (
            front_speed <= 0.3
            and float(local_front[4]) <= TARGET_GAP + 28.0
        )
    if not changed and leader_speed < 5.0 and not front_blocks_restart:
        # A remote gap must not hold the front of a stopped platoon at zero.
        # Every follower receives the same launch intent below, while vehicles
        # with a large local gap retain their stronger CACC recovery command.
        current = max(current, min(1.0, 0.3 * (5.0 - leader_speed)))
        changed = True
        actions["leader"]["accel"] = np.asarray(
            [np.clip(current, -3.0, 1.2)], np.float32
        )
    return changed


def _apply_synchronized_restart_warmup_control(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: dict[str, dict[str, Any]],
) -> bool:
    """Track positive leader intent until the whole acceleration phase settles."""

    platoon = np.asarray(observations["leader"]["platoon_state"], np.float32)
    followers = platoon[1:]
    if not np.all(followers[:, 5] > 0.5):
        return False
    leader_speed = float(platoon[0, 1])
    follower_speeds = followers[:, 1]
    leader_intention = float(
        np.asarray(actions["leader"]["accel"]).reshape(-1)[0]
    )
    if leader_intention <= 0.15:
        return False
    for index, follower in enumerate(FOLLOWER_IDS):
        current = float(np.asarray(actions[follower]["accel"]).reshape(-1)[0])
        gap = float(followers[index, 4])
        synchronized = (
            leader_intention
            + 0.8 * (leader_speed - float(follower_speeds[index]))
            + 0.15 * (gap - TARGET_GAP)
        )
        # A split vehicle may need more acceleration than the global motion
        # consensus command. Compact vehicles use the consensus value exactly
        # so small speed errors are damped instead of carried down the chain.
        if gap > TARGET_GAP + 3.0:
            synchronized = max(current, synchronized)
        actions[follower]["accel"] = np.asarray(
            [np.clip(synchronized, -3.0, MAX_ACCELERATION)],
            np.float32,
        )
    return True


def _apply_synchronized_braking_warmup_control(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: dict[str, dict[str, Any]],
) -> bool:
    """Broadcast measured leader braking before a compression wave forms."""

    platoon = np.asarray(observations["leader"]["platoon_state"], np.float32)
    followers = platoon[1:]
    if not np.all(followers[:, 5] > 0.5):
        return False
    local_front = np.asarray(
        observations["leader"].get("local_front", np.zeros(6, np.float32)),
        np.float32,
    )
    if local_front.shape[-1] < 6 or float(local_front[5]) <= 0.5:
        # Cohesion feedback can legitimately brake the leader when a rear gap
        # opens. Broadcasting that command as an emergency brake creates a
        # self-reinforcing all-stop loop. Only an actual front obstacle is a
        # braking event that should be propagated through the whole platoon.
        return False
    leader_acceleration = float(platoon[0, 2])
    if leader_acceleration >= -0.15:
        return False
    for follower in FOLLOWER_IDS:
        current = float(np.asarray(actions[follower]["accel"]).reshape(-1)[0])
        actions[follower]["accel"] = np.asarray(
            [min(current, leader_acceleration)], np.float32
        )
    return True


def _shape_coordinated_restart_rewards(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: Mapping[str, Mapping[str, Any]],
    rewards: dict[str, float],
    infos: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, float]:
    """Penalize launch intent that reaches followers only after gaps open."""

    platoon = np.asarray(observations["leader"]["platoon_state"], np.float32)
    followers = platoon[1:]
    if not np.all(followers[:, 5] > 0.5):
        return False, 0.0
    leader_intention = float(
        np.asarray(actions["leader"]["accel"]).reshape(-1)[0]
    )
    restart_available = bool(
        infos.get("leader", {}).get("leader_restart_opportunity", False)
    )
    if (
        float(platoon[0, 1]) >= 3.0
        or float(np.max(followers[:, 1])) >= 3.0
        or leader_intention <= 0.15
        or not restart_available
    ):
        return False, 0.0
    intentions = np.asarray(
        [
            float(np.asarray(actions[agent]["accel"]).reshape(-1)[0])
            for agent in PLATOON_IDS
        ],
        np.float32,
    )
    penalties: list[float] = []
    for index, follower in enumerate(FOLLOWER_IDS, start=1):
        lag = max(0.0, leader_intention - float(intentions[index]))
        penalty = 4.0 * (lag / MAX_ACCELERATION) ** 2
        rewards[follower] = float(rewards[follower] - penalty)
        penalties.append(penalty)
    rewards["leader"] = float(rewards["leader"] - np.mean(penalties))
    return True, float(np.max(intentions) - np.min(intentions))


def _shape_expanding_formation_rewards(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: Mapping[str, Mapping[str, Any]],
    rewards: dict[str, float],
) -> tuple[bool, float, float]:
    """Penalize upstream actions that let an already broken pair escape."""

    pair_gaps, opening_speeds, needs_slowdown = _formation_recovery_context(
        observations
    )
    if not np.any(needs_slowdown):
        return False, 0.0, 0.0
    intentions = np.asarray(
        [
            float(np.asarray(actions[agent]["accel"]).reshape(-1)[0])
            for agent in PLATOON_IDS
        ],
        np.float32,
    )
    platoon = np.asarray(observations["leader"]["platoon_state"], np.float32)
    node_speeds = platoon[:, 1]
    upstream_min_speed = np.asarray(
        [
            np.min(node_speeds[: pair_index + 1])
            for pair_index in range(NODE_COUNT - 1)
        ],
        np.float32,
    )
    maximum_penalty = 0.0
    for pair_index in np.flatnonzero(needs_slowdown):
        gap_excess = float(pair_gaps[pair_index] - FORMATION_BREAK_GAP)
        opening_speed = max(0.0, float(opening_speeds[pair_index]))
        slowdown_target = -float(
            np.clip(
                0.05
                + 0.05 * gap_excess
                + 0.35 * max(
                    0.0, opening_speed - FORMATION_OPENING_SPEED
                ),
                0.05,
                1.1,
            )
        )
        state_penalty = (
            0.5 * (gap_excess / 12.0) ** 2
            + (opening_speed / 3.0) ** 2
        )
        for upstream_index in range(pair_index + 1):
            synchronized_target = max(
                -1.2,
                slowdown_target
                - 0.6
                * max(
                    0.0,
                    float(
                        node_speeds[upstream_index]
                        - upstream_min_speed[pair_index]
                    ),
                ),
            )
            action_shortfall = max(
                0.0,
                float(intentions[upstream_index]) - synchronized_target,
            )
            penalty = state_penalty + 2.0 * (action_shortfall / 2.0) ** 2
            agent = PLATOON_IDS[upstream_index]
            rewards[agent] = float(rewards[agent] - penalty)
            maximum_penalty = max(maximum_penalty, penalty)
    return (
        True,
        float(np.max(np.maximum(opening_speeds[needs_slowdown], 0.0))),
        maximum_penalty,
    )


def _foundation_probe(episode_index: int, stage: int) -> str | None:
    if stage != 0:
        return None
    return {1: "A2", 2: "A3", 3: "A4"}.get(episode_index % 8)


def _apply_foundation_probe(
    probe: str | None,
    env: PlatoonParallelEnv,
    episode_step: int,
    actions: dict[str, dict[str, Any]],
) -> bool:
    if probe == "A2":
        if 50 <= episode_step < 80:
            actions["leader"]["accel"] = np.asarray([-2.0], np.float32)
            return True
        if 80 <= episode_step < 120:
            actions["leader"]["accel"] = np.asarray([1.0], np.float32)
            return True
    if probe == "A3" and env._front_ids:
        front = env._front_ids[0]
        active = set(env.connection.vehicle.getIDList())
        if front in active:
            if episode_step < 50:
                target_speed = 25.0
                target_acceleration = 0.0
            elif episode_step < 134:
                target_speed = max(0.0, 25.0 - 0.3 * (episode_step - 49))
                target_acceleration = -3.0
            elif episode_step < 174:
                target_speed = 0.0
                target_acceleration = 0.0
            elif episode_step < 450:
                restart_step = episode_step - 173
                if restart_step <= 8:
                    target_speed = 0.015 * restart_step * (restart_step + 1) / 2.0
                else:
                    # The passenger-car front clears the heavier platoon after
                    # sharing the same launch onset. This preserves headroom
                    # while the trucks follow their 1.2 m/s^2 acceleration
                    # limit and prevents a shield-induced braking wave.
                    target_speed = 0.54 + 0.16 * (restart_step - 8)
                target_speed = min(25.0, target_speed)
                target_acceleration = 1.2 if restart_step < 212 else 0.0
            else:
                target_speed = 25.0
                target_acceleration = 0.0
            env.connection.vehicle.setSpeed(front, target_speed)
            if 50 <= episode_step < 450:
                # The probe represents a communicated front-vehicle plan. Use
                # one feasible team motion profile so the replay contains a
                # genuine simultaneous stop/restart demonstration instead of
                # six shield-corrected cruise commands.
                for agent in PLATOON_IDS:
                    if agent not in active or agent not in actions:
                        continue
                    speed = float(env.connection.vehicle.getSpeed(agent))
                    acceleration = target_acceleration + 0.8 * (
                        target_speed - speed
                    )
                    actions[agent]["accel"] = np.asarray(
                        [np.clip(acceleration, -3.0, MAX_ACCELERATION)],
                        np.float32,
                    )
                return True
    if probe == "A4" and 50 <= episode_step < 80:
        actions["follower_3"]["accel"] = np.asarray([-2.0], np.float32)
        return True
    return False


def _episode_capability(
    pair_gaps: list[float],
    shields: list[float],
    recoveries: list[float],
    applied: Mapping[str, list[float]],
    final_info: Mapping[str, Any],
    options: Mapping[str, Any],
    reward_components: Mapping[str, float],
    steps: int,
    background_peak: int,
    cutin_states: set[str],
    observed_delay: float,
) -> dict[str, float]:
    leader_rms = float(np.sqrt(np.mean(np.square(applied["leader"])))) if applied["leader"] else 0.0
    follower_rms = max(
        (float(np.sqrt(np.mean(np.square(values)))) for agent, values in applied.items() if agent != "leader" and values),
        default=0.0,
    )
    return {
        "success": float(final_info.get("success", False)),
        "collision": float(final_info.get("collision", False)),
        "failure": float(final_info.get("failure", False)),
        "active_count": float(final_info.get("platoon_active_count", 0)),
        "connected_pair_count": float(
            final_info.get("platoon_connected_pair_count", 0)
        ),
        "leader_station": float(final_info.get("leader_station", 0.0)),
        "min_gap": float(min(pair_gaps, default=0.0)),
        "max_gap": float(max(pair_gaps, default=500.0)),
        "recovery": float(np.mean(recoveries or [0.0])),
        "shield": float(np.mean(shields or [0.0])),
        "string": follower_rms / max(leader_rms, 0.1),
        **_behavior_capability_fields(
            options,
            reward_components,
            steps,
            final_info,
            background_peak,
            cutin_states,
            observed_delay,
        ),
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_graph_training_state(
    directory: Path,
    controller: GraphPlatoonController,
    trainer: GraphDreamerTrainer,
    replay: GraphEpisodeReplay,
    config: GraphRLConfig,
    total_steps: int,
    episode_index: int,
    curriculum_stage: int,
    stage_steps: int,
    capability_records: list[dict[str, float]],
    foundation_passed: bool,
    pretrain_updates_completed: int,
    complete: bool = False,
) -> Path:
    payload = {
        "format_version": GRAPH_CHECKPOINT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "control_architecture": controller.architecture_id,
        "graph_model_config": controller.checkpoint_config(),
        "graph_rl_config": asdict(config),
        "models": controller.state_dict(),
        "trainer": trainer.state_dict(),
        "total_steps": total_steps,
        "episodes": episode_index,
        "curriculum_stage": curriculum_stage,
        "stage_steps": stage_steps,
        "capability_records": capability_records,
        "foundation_passed": foundation_passed,
        "pretrain_updates_completed": pretrain_updates_completed,
        "actor_objective_version": GRAPH_ACTOR_OBJECTIVE_VERSION,
        "environment_dynamics_schema_version": (
            GRAPH_ENVIRONMENT_DYNAMICS_SCHEMA_VERSION
        ),
        "restart_metric_schema_version": GRAPH_RESTART_METRIC_SCHEMA_VERSION,
        "complete": complete,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    latest = directory / "graph_latest.pt"
    _atomic_torch_save(payload, latest)
    _atomic_torch_save(replay.state_dict(), directory / "graph_replay_latest.pt")
    if complete:
        complete_path = directory / "graph_complete.pt"
        _atomic_torch_save(payload, complete_path)
        (directory / "training_complete.json").write_text(
            json.dumps(
                {
                    "architecture": controller.architecture_id,
                    "steps": total_steps,
                    "episodes": episode_index,
                    "curriculum_stage": curriculum_stage,
                    "seeds": [config.seed],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return complete_path
    return latest


def load_graph_controller(
    path: str | Path, device: str = "auto"
) -> GraphPlatoonController:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("control_architecture") != "ma_latent_graph_dreamer_v1":
        raise ValueError("checkpoint is not an MA-Latent-Graph-Dreamer checkpoint")
    rl_config = checkpoint.get("graph_rl_config", {})
    predictor_directory = rl_config.get("predictor_directory")
    if predictor_directory is not None:
        predictor_path = Path(predictor_directory)
        required = (
            predictor_path / "cpidlm_best.pt",
            predictor_path / "trajectory_best.pt",
            predictor_path / "leader_front_best.pt",
        )
        if not all(item.exists() for item in required):
            predictor_directory = None
    controller = GraphPlatoonController(
        GraphDreamerConfig(**checkpoint["graph_model_config"]),
        device=device,
        predictor_directory=predictor_directory,
        variant=rl_config.get("variant", "full"),
    )
    controller.load_state_dict(checkpoint["models"])
    controller.eval()
    return controller


def train_graph_rl(
    config: GraphRLConfig,
    model_config: GraphDreamerConfig | None = None,
    resume: str | Path | None = None,
    *,
    reset_replay_on_resume: bool = False,
    reset_curriculum_on_resume: bool = False,
) -> Path:
    # Multiple SUMO workers share the host. Limiting intra-op parallelism keeps
    # CPU graph workers and frozen teacher inference from oversubscribing it.
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    output = Path(config.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    controller = GraphPlatoonController(
        model_config,
        config.device,
        predictor_directory=config.predictor_directory,
        variant=config.variant,
    )
    trainer = GraphDreamerTrainer(controller, config)
    replay = GraphEpisodeReplay(config.replay_capacity, config.seed)
    total_steps = 0
    episode_index = 0
    curriculum_stage = 0
    stage_steps = 0
    capability_records: list[dict[str, float]] = []
    foundation_passed = False
    pretrain_updates_completed = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("control_architecture") != controller.architecture_id:
            raise ValueError("cannot resume graph training from a different architecture")
        if (
            int(checkpoint.get("environment_dynamics_schema_version", 1))
            < GRAPH_ENVIRONMENT_DYNAMICS_SCHEMA_VERSION
        ):
            raise ValueError(
                "checkpoint replay predates continuous SUMO route coordinates; "
                "run scripts.reset_graph_after_station_fix before resuming"
            )
        controller.load_state_dict(checkpoint["models"])
        trainer.load_state_dict(checkpoint["trainer"])
        if int(checkpoint.get("actor_objective_version", 1)) < GRAPH_ACTOR_OBJECTIVE_VERSION:
            trainer.optimizers["actor"].state.clear()
            trainer.optimizers["critic"].state.clear()
        total_steps = int(checkpoint["total_steps"])
        episode_index = int(checkpoint["episodes"])
        curriculum_stage = int(checkpoint["curriculum_stage"])
        stage_steps = int(checkpoint["stage_steps"])
        capability_records = _migrate_restart_capability_records(
            list(checkpoint.get("capability_records", [])),
            int(checkpoint.get("restart_metric_schema_version", 1)),
        )
        foundation_passed = bool(checkpoint.get("foundation_passed", False))
        pretrain_updates_completed = int(
            checkpoint.get(
                "pretrain_updates_completed",
                config.world_pretrain_updates
                if total_steps >= config.warmup_steps
                else 0,
            )
        )
        replay_path = Path(resume).with_name("graph_replay_latest.pt")
        if reset_replay_on_resume:
            replay = GraphEpisodeReplay(config.replay_capacity, config.seed)
        else:
            if not replay_path.is_file():
                raise FileNotFoundError(
                    f"resume replay not found: {replay_path}; use "
                    "reset_replay_on_resume=True when adapting a deployment "
                    "checkpoint to a new scenario"
                )
            replay.load_state_dict(
                torch.load(replay_path, map_location="cpu", weights_only=False)
            )
        if reset_curriculum_on_resume:
            episode_index = 0
            curriculum_stage = 0
            stage_steps = 0
            capability_records = []
            foundation_passed = False
    logger = MetricLogger(output / "logs")
    next_checkpoint = ((total_steps // config.checkpoint_interval) + 1) * config.checkpoint_interval
    next_evaluation = ((total_steps // config.evaluation_interval) + 1) * config.evaluation_interval
    pretraining_done = (
        pretrain_updates_completed >= config.world_pretrain_updates
    )
    controller.use_teacher_reward_predictions = not pretraining_done
    candidate_guard = GraphCandidateGuard(
        controller,
        trainer,
        config.candidate_rollback_cooldown_episodes,
    )
    resume_refinement_pending = _resume_final_refinement_pending(
        output / "logs" / "metrics.jsonl",
        total_steps,
        _graph_stage_capability_passes(
            curriculum_stage, stage_steps, capability_records
        )[0],
    )
    if resume_refinement_pending and capability_records:
        capability_records.pop()
        logger.write(
            total_steps,
            {
                "training/final_refinement_resumed": 1.0,
                "training/final_refinement_window_size": float(
                    len(capability_records)
                ),
            },
        )
    try:
        while total_steps < config.max_sumo_steps:
            env = PlatoonParallelEnv(
                calibration_path=config.calibration_path,
                wave_model_path=config.wave_model_path,
                max_time=config.episode_max_time,
            )
            episode_curriculum_stage = curriculum_stage
            candidate_validation = candidate_guard.validation
            if candidate_validation is None:
                scenario_rng = np.random.default_rng(
                    config.seed + 50_000 + episode_index
                )
                options, _ = _curriculum_options(
                    episode_curriculum_stage,
                    scenario_rng,
                    config.evaluation_config(),
                    total_steps,
                )
                episode_reset_seed = config.seed + episode_index
                foundation_probe = _foundation_probe(
                    episode_index, episode_curriculum_stage
                )
                if foundation_probe == "A3":
                    options = {**options, "front_vehicle_count": 1}
            else:
                options = copy.deepcopy(candidate_validation["scenario_options"])
                episode_reset_seed = int(candidate_validation["reset_seed"])
                foundation_probe = None
            episode_rng = np.random.default_rng(
                config.seed + 100_000 + episode_index
            )
            observations, _ = env.reset(
                seed=episode_reset_seed,
                options=options,
            )
            episode_collection_expert = _uses_collection_expert(
                total_steps,
                config,
                episode_curriculum_stage,
                foundation_passed,
            )
            controller.eval()
            controller.reset()
            policy_states, _ = controller.follower_policy_states(observations)
            _attach_follower_predictions(env, policy_states)
            initial_compact = controller.replay_observation()
            episode = _new_graph_episode(initial_compact)
            pair_gaps: list[float] = []
            shields: list[float] = []
            recoveries: list[float] = []
            applied: dict[str, list[float]] = {agent: [] for agent in PLATOON_IDS}
            reward_components: dict[str, float] = {}
            background_peak = 0
            cutin_states: set[str] = set()
            observed_delay = 0.0
            final_info: dict[str, Any] = {}
            episode_steps = 0
            background_slow_runs: dict[str, int] = {}
            background_max_slow_runs: dict[str, int] = {}
            background_gridlock_run = 0
            background_max_gridlock_run = 0
            coordinated_restart_events = 0
            restart_intent_spreads: list[float] = []
            formation_slowdown_events = 0
            maximum_formation_opening_speed = 0.0
            maximum_formation_penalty = 0.0
            platoon_speeds: dict[str, list[float]] = {
                agent: [] for agent in PLATOON_IDS
            }
            maximum_pair_gaps_by_step: list[float] = []
            leader_front_speeds: list[float] = []
            episode_events: tuple[dict[str, Any], ...] = ()
            try:
                while env.agents and total_steps < config.max_sumo_steps:
                    graph_actions = controller.act(observations, deterministic=True)
                    # Keep one controller for the complete SUMO trajectory.
                    # Switching at a global-step boundary mid-episode creates
                    # an artificial transient and invalid replay semantics.
                    collection_expert = episode_collection_expert
                    if collection_expert:
                        actions = baseline_actions(observations)
                        _apply_correlated_follower_exploration(
                            actions,
                            episode_rng,
                            common_std=0.035,
                            independent_std=0.004,
                            episode_step=episode_steps,
                        )
                        _apply_cohesive_warmup_control(observations, actions)
                        _apply_synchronized_braking_warmup_control(
                            observations, actions
                        )
                        _apply_synchronized_restart_warmup_control(
                            observations, actions
                        )
                        raw_actions = np.asarray(
                            [actions[agent]["accel"] for agent in PLATOON_IDS], np.float32
                        )
                    else:
                        actions = graph_actions
                        raw_actions, _ = controller.requested_action_triplet()
                    behavior_invalid = _apply_foundation_probe(
                        foundation_probe, env, episode_steps, actions
                    )
                    if collection_expert and foundation_probe == "A3" and behavior_invalid:
                        # A3 is an explicit semantic-plan teacher rather than
                        # an adversarial action perturbation. Supervise the
                        # shared actor on the synchronized team command.
                        raw_actions = np.asarray(
                            [actions[agent]["accel"] for agent in PLATOON_IDS],
                            np.float32,
                        )
                        behavior_invalid = False
                    next_observations, rewards, terminations, truncations, infos = env.step(actions)
                    restart_active, restart_spread = (
                        _shape_coordinated_restart_rewards(
                            observations, actions, rewards, infos
                        )
                    )
                    if restart_active:
                        coordinated_restart_events += 1
                        restart_intent_spreads.append(restart_spread)
                    (
                        formation_slowdown_active,
                        formation_opening_speed,
                        formation_penalty,
                    ) = _shape_expanding_formation_rewards(
                        observations, actions, rewards
                    )
                    if formation_slowdown_active:
                        formation_slowdown_events += 1
                        maximum_formation_opening_speed = max(
                            maximum_formation_opening_speed,
                            formation_opening_speed,
                        )
                        maximum_formation_penalty = max(
                            maximum_formation_penalty,
                            formation_penalty,
                        )
                    controller.record_applied_actions(
                        infos,
                        actions if collection_expert else None,
                    )
                    continued = not (
                        any(terminations.values()) or any(truncations.values())
                    )
                    if next_observations:
                        next_policy_states, _ = controller.follower_policy_states(next_observations)
                        _attach_follower_predictions(env, next_policy_states)
                        next_compact = controller.replay_observation()
                    else:
                        next_compact = episode["observations"][-1]
                    _append_graph_transition(
                        episode,
                        next_compact,
                        actions,
                        raw_actions,
                        rewards,
                        continued,
                        infos,
                        behavior_invalid,
                        behavior_supervised=collection_expert,
                    )
                    total_steps += 1
                    stage_steps += 1
                    episode_steps += 1
                    leader_info = infos["leader"]
                    final_info = leader_info
                    for name, value in leader_info["reward_components"].items():
                        reward_components[name] = reward_components.get(name, 0.0) + float(value)
                    background_peak = max(background_peak, int(leader_info.get("background_active_count", 0)))
                    cutin_states.add(str(leader_info.get("cutin_state", "idle")))
                    recoveries.append(float(leader_info["reward_components"].get("diagnostic_recovery_active_ratio", 0.0)))
                    active = set(env.connection.vehicle.getIDList())
                    background_ids = sorted(
                        vehicle_id
                        for vehicle_id in active
                        if vehicle_id.startswith("background_")
                    )
                    background_speeds = [
                        float(env.connection.vehicle.getSpeed(vehicle_id))
                        for vehicle_id in background_ids
                    ]
                    for vehicle_id, speed in zip(
                        background_ids, background_speeds
                    ):
                        run = (
                            background_slow_runs.get(vehicle_id, 0) + 1
                            if speed < 0.2
                            else 0
                        )
                        background_slow_runs[vehicle_id] = run
                        background_max_slow_runs[vehicle_id] = max(
                            background_max_slow_runs.get(vehicle_id, 0), run
                        )
                    if (
                        len(background_speeds) >= 3
                        and float(np.mean(background_speeds)) < 1.0
                    ):
                        background_gridlock_run += 1
                    else:
                        background_gridlock_run = 0
                    background_max_gridlock_run = max(
                        background_max_gridlock_run, background_gridlock_run
                    )
                    current_pair_gaps: list[float] = []
                    for follower in FOLLOWER_IDS:
                        if follower in active:
                            gap = env._platoon_gap(follower, active)
                            if gap >= 0.0:
                                pair_gaps.append(gap)
                                current_pair_gaps.append(gap)
                        shields.append(float(infos[follower].get("collision_avoidance_intervened", False) or infos[follower].get("predictive_safety_intervened", False)))
                        observed_delay = max(
                            observed_delay,
                            float(infos[follower].get("communication_delay", 0.0)),
                            float(infos[follower].get("predecessor_communication_delay", 0.0)),
                        )
                    for agent in PLATOON_IDS:
                        applied[agent].append(float(infos[agent].get("applied_acceleration", 0.0)))
                        if agent in active:
                            platoon_speeds[agent].append(
                                float(env.connection.vehicle.getSpeed(agent))
                            )
                    maximum_pair_gaps_by_step.append(
                        max(current_pair_gaps, default=0.0)
                    )
                    if "leader" in active:
                        leader_front = env.connection.vehicle.getLeader(
                            "leader", 500.0
                        )
                        leader_front_speeds.append(
                            float(
                                env.connection.vehicle.getSpeed(
                                    leader_front[0]
                                )
                            )
                            if leader_front and leader_front[0] in active
                            else 30.0
                        )
                    else:
                        # SUMO may remove the leader on the same terminal step
                        # that provides final infos. Diagnostics must not query
                        # a vehicle after its lifecycle has ended.
                        leader_front_speeds.append(30.0)

                    if total_steps % config.update_every == 0:
                        metrics: dict[str, float] = {}
                        (
                            collection_world_interval,
                            collection_actor_interval,
                        ) = _collection_update_intervals(
                            config, pretraining_done
                        )
                        if (
                            collection_expert
                            and total_steps % collection_world_interval == 0
                            and replay.can_sample(
                                config.sequence_length, config.batch_size
                            )
                        ):
                            batch = replay.sample(
                                config.batch_size,
                                config.sequence_length,
                                config.foundation_event_fraction
                                if curriculum_stage == 0
                                and not foundation_passed
                                else 0.0,
                            )
                            metrics = trainer.train_world(
                                batch, _communication_dropout(curriculum_stage)
                            )
                            if (
                                total_steps >= config.actor_warmup_steps
                                and curriculum_stage > 0
                                and total_steps % collection_actor_interval == 0
                            ):
                                horizon = 15 if curriculum_stage < 3 else 30
                                metrics.update(
                                    trainer.train_actor_critic(
                                        batch,
                                        horizon,
                                        _communication_dropout(curriculum_stage),
                                        (
                                            config.foundation_imagination_weight
                                            if curriculum_stage == 0
                                            and not foundation_passed
                                            else config.imagination_weight
                                        ),
                                    )
                                )
                        current_speeds = [
                            values[-1]
                            for values in platoon_speeds.values()
                            if values
                        ]
                        metrics.update(
                            {
                                "training/live_episode_steps": float(episode_steps),
                                "training/live_min_gap": float(
                                    min(pair_gaps, default=0.0)
                                ),
                                "training/live_max_gap": float(
                                    max(pair_gaps, default=0.0)
                                ),
                                "training/live_current_min_gap": float(
                                    min(current_pair_gaps, default=0.0)
                                ),
                                "training/live_current_max_gap": float(
                                    max(current_pair_gaps, default=0.0)
                                ),
                                "training/live_speed_spread": float(
                                    max(current_speeds, default=0.0)
                                    - min(current_speeds, default=0.0)
                                ),
                                "training/live_background_gridlock_seconds": float(
                                    background_gridlock_run * env.step_length
                                ),
                                "training/fixed_policy_rollout": float(
                                    not collection_expert
                                ),
                                "training/candidate_pending": float(
                                    candidate_guard.pending
                                ),
                            }
                        )
                        logger.write(total_steps, metrics)
                    if not continued:
                        break
                    observations = next_observations
            finally:
                episode_events = env.event_log
                env.close()
            replay.add(episode)
            episode_index += 1

            if total_steps >= config.warmup_steps and not pretraining_done:
                while (
                    pretrain_updates_completed < config.world_pretrain_updates
                    and replay.can_sample(config.sequence_length, config.batch_size)
                ):
                    batch = replay.sample(
                        config.batch_size,
                        config.sequence_length,
                        config.foundation_event_fraction
                        if curriculum_stage == 0 and not foundation_passed
                        else 0.0,
                    )
                    metrics = trainer.train_world(batch, 0.0)
                    pretrain_updates_completed += 1
                    if pretrain_updates_completed % 100 == 0:
                        logger.write(total_steps, {**metrics, "graph/pretrain_updates": float(pretrain_updates_completed)})
                    if pretrain_updates_completed % 500 == 0:
                        save_graph_training_state(
                            output,
                            controller,
                            trainer,
                            replay,
                            config,
                            total_steps,
                            episode_index,
                            curriculum_stage,
                            stage_steps,
                            capability_records,
                            foundation_passed,
                            pretrain_updates_completed,
                            False,
                        )
                pretraining_done = (
                    pretrain_updates_completed >= config.world_pretrain_updates
                )
                controller.use_teacher_reward_predictions = not pretraining_done

            capability = _episode_capability(
                pair_gaps,
                shields,
                recoveries,
                applied,
                final_info,
                options,
                reward_components,
                episode_steps,
                background_peak,
                cutin_states,
                observed_delay,
            )
            event_names = [str(event.get("event", "")) for event in episode_events]
            collision_details = [
                detail
                for event in episode_events
                if event.get("event") == "collision"
                for detail in event.get("details", [])
            ]
            background_collision_details = [
                detail
                for detail in collision_details
                if str(detail.get("collider", "")).startswith("background_")
                or str(detail.get("victim", "")).startswith("background_")
            ]
            capability.update(
                {
                    **_restart_timing_metrics(
                        platoon_speeds,
                        maximum_pair_gaps_by_step,
                        leader_front_speeds,
                    ),
                    "restart_metric_valid": 1.0,
                    "background_peak_count": float(background_peak),
                    "background_stuck_30s_count": float(
                        sum(
                            run * env.step_length >= 30.0
                            for run in background_max_slow_runs.values()
                        )
                    ),
                    "background_max_stuck_seconds": float(
                        max(background_max_slow_runs.values(), default=0)
                        * env.step_length
                    ),
                    "background_max_gridlock_seconds": float(
                        background_max_gridlock_run * env.step_length
                    ),
                    "background_collision_events": float(
                        len(background_collision_details)
                    ),
                    "collision_detail_count": float(len(collision_details)),
                    "ramp_meter_closed_events": float(
                        event_names.count("ramp_meter_closed")
                    ),
                    "ramp_queue_braking_events": float(
                        event_names.count("ramp_queue_braking")
                    ),
                    "ramp_merge_waiting_events": float(
                        event_names.count("ramp_merge_waiting")
                    ),
                    "ramp_merge_completed_events": float(
                        event_names.count("ramp_merge_completed")
                    ),
                    "leader_stall_events": float(
                        event_names.count("leader_stalled_with_clear_path")
                    ),
                    "coordinated_restart_steps": float(
                        coordinated_restart_events
                    ),
                    "restart_intent_spread_max": float(
                        max(restart_intent_spreads, default=0.0)
                    ),
                    "formation_slowdown_steps": float(
                        formation_slowdown_events
                    ),
                    "formation_opening_speed_max": float(
                        maximum_formation_opening_speed
                    ),
                    "formation_expansion_penalty_max": float(
                        maximum_formation_penalty
                    ),
                    "foundation_probe_active": float(
                        foundation_probe is not None
                    ),
                }
            )
            candidate_accepted, candidate_rolled_back = candidate_guard.resolve(
                episode_curriculum_stage, capability
            )
            cooldown_blocked = candidate_guard.consume_cooldown_episode(
                candidate_rolled_back
            )
            capability.update(
                {
                    "candidate_accepted": float(candidate_accepted),
                    "candidate_rolled_back": float(candidate_rolled_back),
                    "candidate_cooldown_episodes": float(
                        candidate_guard.cooldown_episodes
                    ),
                }
            )
            if foundation_probe is None and not candidate_rolled_back:
                capability_records.append(capability)
                capability_records = capability_records[-8:]
            stage_passed, summary = _graph_stage_capability_passes(
                episode_curriculum_stage, stage_steps, capability_records
            )
            if stage_passed and curriculum_stage == 0 and not foundation_passed:
                teachers = controller.adapter.teachers
                controller.adapter.teachers = None
                previous_teacher_reward = controller.use_teacher_reward_predictions
                controller.use_teacher_reward_predictions = False
                try:
                    foundation = evaluate_foundation_gates(
                        controller,
                        config.calibration_path,
                        config.wave_model_path,
                        config.seed,
                        output / "foundation_validation.json",
                    )
                finally:
                    controller.adapter.teachers = teachers
                    controller.use_teacher_reward_predictions = previous_teacher_reward
                foundation_passed = bool(foundation["passed"])
                stage_passed = stage_passed and foundation_passed
                if foundation_passed:
                    candidate_guard.accept_current()
                logger.write(
                    total_steps,
                    {
                        "validation/foundation_passed": float(foundation_passed),
                        **{
                            f"validation/{name}_passed": float(value)
                            for name, value in foundation["gates"].items()
                        },
                    },
                )
            logger.write(
                total_steps,
                {
                    "training/episode": float(episode_index),
                    "training/curriculum_stage": float(curriculum_stage),
                    "training/stage_steps": float(stage_steps),
                    **{f"training/{key}": value for key, value in capability.items()},
                },
            )
            complete = False
            final_stage_candidate = False
            if stage_passed:
                if curriculum_stage == len(CURRICULUM_NAMES) - 1:
                    final_stage_candidate = (
                        total_steps >= config.min_steps_before_convergence
                    )
                else:
                    curriculum_stage += 1
                    stage_steps = 0
                    capability_records = []
            if resume_refinement_pending:
                stage_passed = False
                final_stage_candidate = False
            completion_refinement_required = resume_refinement_pending
            if total_steps >= next_evaluation or final_stage_candidate:
                teachers = controller.adapter.teachers
                controller.adapter.teachers = None
                previous_teacher_reward = controller.use_teacher_reward_predictions
                controller.use_teacher_reward_predictions = False
                logger.write(total_steps, {"evaluation/active": 1.0})
                try:
                    evaluation = evaluate_controller(
                        controller,
                        config.evaluation_config(),
                        total_steps,
                        curriculum_stage,
                        progress_callback=lambda metrics: logger.write(
                            total_steps, metrics
                        ),
                    )
                finally:
                    controller.adapter.teachers = teachers
                    controller.use_teacher_reward_predictions = previous_teacher_reward
                logger.write(
                    total_steps,
                    {**evaluation, "evaluation/active": 0.0},
                )
                complete = bool(
                    final_stage_candidate
                    and evaluation["evaluation/capability_pass"] >= 0.5
                    and evaluation["evaluation/collision_rate"] == 0.0
                    and evaluation["evaluation/failure_rate"] == 0.0
                )
                completion_refinement_required = (
                    _final_evaluation_requires_refinement(
                        final_stage_candidate, evaluation
                    )
                )
                if completion_refinement_required:
                    # A passing eight-episode training window freezes updates.
                    # If its deterministic completion evaluation then fails,
                    # remove one window record and allow one guarded refinement
                    # update below. Otherwise the identical frozen checkpoint
                    # is evaluated forever and cannot improve.
                    if capability_records:
                        capability_records.pop()
                    stage_passed = False
                    final_stage_candidate = False
                    logger.write(
                        total_steps,
                        {
                            "training/final_refinement_requested": 1.0,
                            "training/final_refinement_window_size": float(
                                len(capability_records)
                            ),
                        },
                    )
                if total_steps >= next_evaluation:
                    next_evaluation += config.evaluation_interval
            if complete or total_steps >= next_checkpoint:
                checkpoint = save_graph_training_state(
                    output,
                    controller,
                    trainer,
                    replay,
                    config,
                    total_steps,
                    episode_index,
                    curriculum_stage,
                    stage_steps,
                    capability_records,
                    foundation_passed,
                    pretrain_updates_completed,
                    complete,
                )
                next_checkpoint += config.checkpoint_interval
                if complete:
                    return checkpoint

            episode_capability_meets_criteria = _graph_capability_meets_criteria(
                episode_curriculum_stage, capability
            )
            post_world_updates = 0
            post_actor_updates = 0
            if (
                not episode_collection_expert
                and foundation_probe is None
                and (
                    not episode_capability_meets_criteria
                    or completion_refinement_required
                )
                and not stage_passed
                and not final_stage_candidate
                and not cooldown_blocked
                and total_steps < config.max_sumo_steps
                and replay.can_sample(config.sequence_length, config.batch_size)
            ):
                update_metrics: dict[str, float] = {}
                dropout = _communication_dropout(episode_curriculum_stage)
                for _ in range(
                    max(0, config.online_world_updates_per_episode)
                ):
                    batch = replay.sample(
                        config.batch_size,
                        config.sequence_length,
                        0.0,
                    )
                    update_metrics.update(trainer.train_world(batch, dropout))
                    post_world_updates += 1
                if total_steps >= config.actor_warmup_steps:
                    horizon = 15 if episode_curriculum_stage < 3 else 30
                    for _ in range(
                        _post_episode_actor_update_count(
                            config, episode_curriculum_stage
                        )
                    ):
                        batch = replay.sample(
                            config.batch_size,
                            config.sequence_length,
                            0.0,
                        )
                        update_metrics.update(
                            trainer.train_actor_critic(
                                batch,
                                horizon,
                                dropout,
                                config.imagination_weight,
                            )
                        )
                        post_actor_updates += 1
                if post_world_updates or post_actor_updates:
                    candidate_guard.begin(
                        options,
                        episode_reset_seed,
                        capability,
                    )
                    resume_refinement_pending = False
                    logger.write(
                        total_steps,
                        {
                            **update_metrics,
                            "training/post_episode_world_updates": float(
                                post_world_updates
                            ),
                            "training/post_episode_actor_updates": float(
                                post_actor_updates
                            ),
                            "training/candidate_pending": 1.0,
                        },
                    )
        return save_graph_training_state(
            output,
            controller,
            trainer,
            replay,
            config,
            total_steps,
            episode_index,
            curriculum_stage,
            stage_steps,
            capability_records,
            foundation_passed,
            pretrain_updates_completed,
            False,
        )
    finally:
        logger.close()

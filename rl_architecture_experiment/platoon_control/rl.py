"""Resumable SUMO training for the Dreamer leader and CTDE MARL followers."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor

from platoon_sumo.env import (
    FOLLOWER_IDS,
    PLATOON_IDS,
    PREDECESSOR,
    REJOIN_GAP_TOLERANCE,
    TARGET_GAP,
    PlatoonParallelEnv,
)

from .agents import PlatoonController
from .config import FollowerRLConfig, SystemConfig
from .models import (
    FOLLOWER_RECOVERY_GAP_INDEX,
    FOLLOWER_RECOVERY_ROLE_INDEX,
)
from .models.dreamer import LEADER_RECOVERY_TRAINING_SPEED
from .training import MetricLogger, PlatoonTrainer


LEADER_KEYS = (
    "ego",
    "platoon_state",
    "front_history",
    "front_mask",
    "v2i_flow",
    "road_profile",
    "signal",
    "signal_mask",
)
CONTROL_ARCHITECTURE = "dreamer_gated_pbrs_leader_cpidlm_residual_ctde_sac_v7"
FOLLOWER_ACTOR_OBJECTIVE_VERSION = 3
FOLLOWER_STATE_SCHEMA_VERSION = 2
TERMINAL_SUCCESS_SCHEMA_VERSION = 1
CAPABILITY_SCHEMA_VERSION = 2
ACTOR_COOLDOWN_SCHEMA_VERSION = 2

CURRICULUM_NAMES = (
    "formation_stability",
    "front_traffic_adaptation",
    "wave_absorption",
    "v2i_curve_planning",
    "background_traffic",
    "cutin_defense",
    "mild_communication_delay",
    "full_mixed_scenario",
)
CURRICULUM_MIN_STAGE_STEPS = (
    40_000,
    20_000,
    25_000,
    20_000,
    25_000,
    30_000,
    30_000,
    40_000,
)
CURRICULUM_WINDOW = 8
CURRICULUM_CRITERIA = (
    {"success": 0.90, "failure": 0.05, "min_gap": 5.0, "max_gap": 15.0, "recovery": 0.02, "shield": 0.08, "string": 1.30},
    {"success": 0.85, "failure": 0.10, "min_gap": 4.5, "max_gap": 18.0, "recovery": 0.05, "shield": 0.15, "string": 1.50},
    {"success": 0.80, "failure": 0.15, "min_gap": 3.5, "max_gap": 22.0, "recovery": 0.12, "shield": 0.30, "string": 1.80},
    {"success": 0.85, "failure": 0.10, "min_gap": 4.0, "max_gap": 20.0, "recovery": 0.10, "shield": 0.25, "string": 1.60},
    {"success": 0.80, "failure": 0.15, "min_gap": 3.5, "max_gap": 24.0, "recovery": 0.15, "shield": 0.30, "string": 1.80},
    {"success": 0.75, "failure": 0.20, "min_gap": 3.0, "max_gap": 30.0, "recovery": 0.22, "shield": 0.35, "string": 2.00},
    {"success": 0.75, "failure": 0.20, "min_gap": 3.0, "max_gap": 30.0, "recovery": 0.22, "shield": 0.35, "string": 2.00},
    {"success": 0.75, "failure": 0.20, "min_gap": 3.0, "max_gap": 32.0, "recovery": 0.25, "shield": 0.35, "string": 2.00},
)
# Physical completion alone is insufficient for the later curriculum stages:
# the disturbance must be observed and the corresponding behavior must pass.
CURRICULUM_BEHAVIOR_CRITERIA = (
    {},
    {},
    {},
    {
        "v2i_score_min": 0.78,
        "curve_overspeed_max": 0.08,
        "curve_wasted_acceleration_max": 0.08,
        "leader_jerk_max": 0.02,
    },
    {
        "v2i_score_min": 0.75,
        "curve_overspeed_max": 0.10,
        "curve_wasted_acceleration_max": 0.10,
        "leader_jerk_max": 0.025,
        "background_exposure_min": 0.75,
    },
    {
        "v2i_score_min": 0.70,
        "curve_overspeed_max": 0.12,
        "curve_wasted_acceleration_max": 0.12,
        "leader_jerk_max": 0.03,
        "background_exposure_min": 0.75,
        "cutin_exposure_min": 0.75,
        "cutin_resolution_min": 0.50,
    },
    {
        "v2i_score_min": 0.68,
        "curve_overspeed_max": 0.14,
        "curve_wasted_acceleration_max": 0.14,
        "leader_jerk_max": 0.035,
        "background_exposure_min": 0.75,
        "cutin_exposure_min": 0.75,
        "cutin_resolution_min": 0.50,
        "delay_exposure_min": 1.0,
        "observed_delay_min": 0.09,
        "observed_delay_max": 0.11,
    },
    {
        "v2i_score_min": 0.62,
        "curve_overspeed_max": 0.18,
        "curve_wasted_acceleration_max": 0.18,
        "leader_jerk_max": 0.04,
        "background_exposure_min": 0.90,
        "cutin_exposure_min": 0.75,
        "cutin_resolution_min": 0.50,
        "delay_exposure_min": 1.0,
        "observed_delay_min": 0.09,
        "observed_delay_max": 0.21,
    },
)
FOLLOWER_ACTOR_RL_WEIGHTS = (0.01, 0.01, 0.05, 0.15, 0.30, 0.50, 0.75, 1.00)
FOLLOWER_ACTOR_STABILITY_WEIGHTS = (50.0, 300.0, 150.0, 75.0, 40.0, 25.0, 15.0, 10.0)
FOLLOWER_ACTOR_UPDATES_PER_EPISODE = (1, 1, 32, 8, 4, 24, 24, 24)
LEADER_ACTOR_UPDATES_PER_EPISODE = (1, 1, 1, 1, 4, 8, 8, 8)
# Focused restart batches need to move the requested acceleration far enough
# to survive a paired rollout; four batches only changed it by about 0.03 m/s2.
LEADER_RESTART_UPDATES_PER_EPISODE = (0, 0, 0, 0, 8, 8, 8, 8)
LEADER_RESTART_AUX_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0)
LEADER_RECOVERY_MIN_PROGRESS_STATION = 3_300.0
FOLLOWER_ACTOR_LR_MULTIPLIERS = (0.01, 0.01, 0.01, 0.01, 0.10, 0.10, 0.10, 0.10)
LEADER_ACTOR_LR_MULTIPLIERS = (0.10, 0.10, 0.10, 0.10, 1.00, 1.00, 1.00, 1.00)
FOLLOWER_ACTOR_ADAPTATION_STAGE = 2
FOUNDATION_MAX_GAP_DRIFT = 0.25
# Recovery observations saturate at 500 m, which is sufficient to represent
# the direction and severity of a split. Keep collision-free separations up to
# the full route length so the follower can learn to rejoin after an extreme
# dropout; paired candidate validation still requires material improvement.
REPLAY_MAX_GAP = 3_500.0
CANDIDATE_MIN_BEHAVIOR_DEFICIT_IMPROVEMENT = 0.0075
ACTOR_ROLLBACK_COOLDOWN_EPISODES = 2


@dataclass
class RLConfig:
    seed: int = 42
    max_sumo_steps: int = 1_000_000
    warmup_steps: int = 5_000
    actor_warmup_steps: int = 10_000
    leader_warmup_steps: int = 5_000
    leader_actor_warmup_steps: int = 30_000
    replay_capacity: int = 30_000
    sequence_length: int = 50
    batch_size: int = 32
    update_every: int = 1
    policy_update_interval: int = 8
    leader_update_interval: int = 20
    evaluation_interval: int = 50_000
    checkpoint_interval: int = 10_000
    evaluation_episodes: int = 5
    convergence_patience: int = 5
    min_steps_before_convergence: int = 300_000
    device: str = "auto"
    checkpoint_dir: str = "artifacts/rl"
    calibration_path: str = "artifacts/calibration/calibrated_parameters.json"
    wave_model_path: str = "artifacts/checkpoints/wave_best.pt"
    episode_max_time: float = 240.0
    formation_curriculum_steps: int = 30_000
    disturbance_curriculum_steps: int = 100_000
    downstream_wave_training: bool = False
    downstream_wave_training_probability: float = 0.25


class EpisodeReplay:
    def __init__(self, capacity_steps: int, seed: int) -> None:
        self.capacity_steps = capacity_steps
        self.episodes: list[dict[str, Any]] = []
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def add(self, episode: dict[str, Any]) -> None:
        length = len(episode["rewards"])
        if length == 0:
            return
        self.episodes.append(episode)
        self.steps += length
        while self.steps > self.capacity_steps and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.steps -= len(removed["rewards"])

    def _candidates(
        self, extra_episode: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        candidates = list(self.episodes)
        if extra_episode is not None and extra_episode["rewards"]:
            candidates.append(extra_episode)
        return candidates

    def can_sample(
        self,
        sequence_length: int,
        batch_size: int,
        extra_episode: dict[str, Any] | None = None,
    ) -> bool:
        usable = sum(
            max(0, len(ep["rewards"]) - sequence_length + 1)
            for ep in self._candidates(extra_episode)
        )
        return usable >= batch_size

    def sample_leader(
        self,
        batch_size: int,
        sequence_length: int,
        extra_episode: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        candidates = [
            ep
            for ep in self._candidates(extra_episode)
            if len(ep["rewards"]) >= sequence_length
        ]
        return self._sample_leader_candidates(
            candidates, batch_size, sequence_length
        )

    def sample_leader_episode(
        self,
        episode: dict[str, Any],
        batch_size: int,
        sequence_length: int,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        if len(episode["rewards"]) < sequence_length:
            raise ValueError("episode is shorter than the leader sequence")
        return self._sample_leader_candidates(
            [episode], batch_size, sequence_length
        )

    def sample_leader_restart_episode(
        self,
        episode: dict[str, Any],
        batch_size: int,
        sequence_length: int,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        """Sample windows containing a clear-path restart with intact cohesion."""

        if len(episode["rewards"]) < sequence_length:
            raise ValueError("episode is shorter than the leader sequence")
        eligible_starts: list[int] = []
        last_start = len(episode["rewards"]) - sequence_length
        for start in range(last_start + 1):
            stop = start + sequence_length
            for index in range(start, stop):
                ego = episode["leader_observations"]["ego"][index]
                signal = episode["leader_observations"]["signal"][index]
                signal_mask = episode["leader_observations"]["signal_mask"][index]
                follower_chain = episode["leader_observations"][
                    "platoon_state"
                ][index][1:]
                if (
                    float(ego[1]) < LEADER_RECOVERY_TRAINING_SPEED
                    and float(signal[0]) > 0.5
                    and float(signal_mask[0]) > 0.5
                    and np.all(follower_chain[:, 5] > 0.5)
                    and np.all(follower_chain[:, 4] <= 14.0)
                ):
                    eligible_starts.append(start)
                    break
        if not eligible_starts:
            raise ValueError("episode has no cohesive leader restart windows")
        selected_starts = self.rng.choice(
            eligible_starts, batch_size, replace=True
        )
        return self._stack_leader_windows(
            [(episode, int(start)) for start in selected_starts],
            sequence_length,
        )

    def _sample_leader_candidates(
        self,
        candidates: list[dict[str, Any]],
        batch_size: int,
        sequence_length: int,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        if not candidates:
            raise ValueError("no leader episodes are available for sampling")
        selected = self.rng.choice(len(candidates), batch_size, replace=True)
        windows: list[tuple[dict[str, Any], int]] = []
        for candidate_index in selected:
            episode = candidates[int(candidate_index)]
            start = int(
                self.rng.integers(
                    0, len(episode["rewards"]) - sequence_length + 1
                )
            )
            windows.append((episode, start))
        return self._stack_leader_windows(windows, sequence_length)

    @staticmethod
    def _stack_leader_windows(
        windows: list[tuple[dict[str, Any], int]],
        sequence_length: int,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        observation_batches: dict[str, list[np.ndarray]] = {key: [] for key in LEADER_KEYS}
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        continues: list[np.ndarray] = []
        for episode, start in windows:
            stop = start + sequence_length
            for key in LEADER_KEYS:
                observation_batches[key].append(np.stack(episode["leader_observations"][key][start:stop]))
            actions.append(np.asarray(episode["leader_actions"][start:stop], np.float32))
            rewards.append(np.asarray(episode["rewards"][start:stop], np.float32))
            continues.append(np.asarray(episode["continues"][start:stop], np.float32))
        observations = {
            key: torch.from_numpy(np.stack(values)) for key, values in observation_batches.items()
        }
        return (
            observations,
            torch.from_numpy(np.stack(actions)),
            torch.from_numpy(np.stack(rewards)),
            torch.from_numpy(np.stack(continues)),
        )

    def sample_followers(
        self,
        batch_size: int,
        extra_episode: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        candidates = [
            ep
            for ep in self._candidates(extra_episode)
            if len(ep["follower_states"])
        ]
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        behavior_actions: list[np.ndarray] = []
        requested_actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        continues: list[float] = []
        recovery_candidates: list[
            tuple[dict[str, Any], int, np.ndarray]
        ] = []
        for episode in candidates:
            episode_states = np.asarray(episode["follower_states"], np.float32)
            current_states = (
                episode_states[:, :, -1]
                if episode_states.ndim == 4
                else episode_states
            )
            for follower_index in range(current_states.shape[1]):
                recovery_role = (
                    current_states[
                        :, follower_index, FOLLOWER_RECOVERY_ROLE_INDEX
                    ]
                    > 0.5
                )
                disconnected = (
                    current_states[
                        :, follower_index, FOLLOWER_RECOVERY_GAP_INDEX
                    ]
                    > TARGET_GAP + REJOIN_GAP_TOLERANCE
                )
                indices = np.flatnonzero(recovery_role | disconnected)
                if len(indices):
                    recovery_candidates.append(
                        (episode, follower_index, indices)
                    )
        for _ in range(batch_size):
            collision_candidates = [
                episode
                for episode in candidates
                if episode["continues"]
                and not episode["continues"][-1]
                and np.min(episode["follower_rewards"][-1]) <= -50.0
            ]
            sample_kind = self.rng.random()
            if collision_candidates and sample_kind < 0.10:
                episode = collision_candidates[
                    int(self.rng.integers(len(collision_candidates)))
                ]
                lower = max(0, len(episode["follower_states"]) - 30)
                step = int(
                    self.rng.integers(lower, len(episode["follower_states"]))
                )
            elif recovery_candidates and sample_kind < 0.80:
                episode, _, indices = recovery_candidates[
                    int(self.rng.integers(len(recovery_candidates)))
                ]
                step = int(indices[int(self.rng.integers(len(indices)))])
            else:
                episode = candidates[int(self.rng.integers(len(candidates)))]
                step = int(self.rng.integers(len(episode["follower_states"])))
            states.append(episode["follower_states"][step])
            actions.append(
                np.asarray(episode["follower_actions"][step], np.float32)
            )
            behavior_actions.append(
                np.asarray(
                    episode.get(
                        "follower_behavior_actions", episode["follower_actions"]
                    )[step],
                    np.float32,
                )
            )
            requested_actions.append(
                np.asarray(
                    episode.get(
                        "follower_requested_actions", episode["follower_actions"]
                    )[step],
                    np.float32,
                )
            )
            rewards.append(
                np.asarray(episode["follower_rewards"][step], np.float32)
            )
            next_states.append(episode["follower_next_states"][step])
            continues.append(float(episode["continues"][step]))
        return (
            torch.from_numpy(np.asarray(states, np.float32)),
            torch.from_numpy(np.asarray(actions, np.float32)),
            torch.from_numpy(np.asarray(rewards, np.float32)),
            torch.from_numpy(np.asarray(next_states, np.float32)),
            torch.from_numpy(np.asarray(continues, np.float32)),
            torch.from_numpy(np.asarray(behavior_actions, np.float32)),
            torch.from_numpy(np.asarray(requested_actions, np.float32)),
        )

    def state_dict(self) -> dict[str, Any]:
        return {"capacity_steps": self.capacity_steps, "episodes": self.episodes, "steps": self.steps, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.capacity_steps = int(state["capacity_steps"])
        self.episodes = list(state["episodes"])
        self.steps = int(state["steps"])
        self.rng.bit_generator.state = state["rng"]


def _load_predictors(controller: PlatoonController, directory: str | Path) -> None:
    directory = Path(directory)
    delay = torch.load(directory / "cpidlm_best.pt", map_location="cpu", weights_only=False)
    trajectory = torch.load(directory / "trajectory_best.pt", map_location="cpu", weights_only=False)
    leader_front = torch.load(directory / "leader_front_best.pt", map_location="cpu", weights_only=False)
    controller.delay.load_state_dict(delay["model"])
    controller.trajectory.load_state_dict(trajectory["model"])
    front_state = {
        key.removeprefix("front_predictor."): value
        for key, value in leader_front["model"].items()
        if key.startswith("front_predictor.")
    }
    controller.leader.front_predictor.load_state_dict(front_state)
    controller.delay.requires_grad_(False)
    controller.trajectory.requires_grad_(False)
    controller.leader.front_predictor.requires_grad_(False)


def _curriculum_options(
    stage: int,
    rng: np.random.Generator,
    config: RLConfig,
    total_steps: int = 0,
) -> tuple[dict[str, Any], int]:
    if not 0.0 <= config.downstream_wave_training_probability <= 1.0:
        raise ValueError(
            "downstream_wave_training_probability must be in [0, 1]"
        )
    stage = int(np.clip(stage, 0, len(CURRICULUM_NAMES) - 1))
    options: dict[str, Any] = {
        "front_vehicle_count": 0,
        "cutin_enabled": False,
        "wave_mode": "off",
        "v2i_enabled": False,
        "speed_reference": 25.0,
        "communication_delay_probability": 0.0,
        "communication_delay_min": 0.1,
        "communication_delay_max": 0.1,
        "background_initial_scale": 0.0,
        "background_demand_scale": 0.0,
    }
    if stage >= 1:
        options["front_vehicle_count"] = int(rng.integers(0, 3))
    if stage >= 2:
        options["front_vehicle_count"] = int(rng.integers(1, 5))
        options["wave_mode"] = "learned" if rng.random() < 0.65 else "off"
        options.pop("speed_reference", None)
    if stage >= 3:
        options["v2i_enabled"] = True
    if stage >= 4:
        traffic_scale = float(rng.uniform(0.3, 0.7))
        options["front_vehicle_count"] = int(rng.integers(0, 5))
        options["background_initial_scale"] = traffic_scale
        options["background_demand_scale"] = traffic_scale
    if stage >= 5:
        traffic_scale = float(rng.uniform(0.6, 1.0))
        options["front_vehicle_count"] = int(rng.integers(0, 6))
        options["cutin_enabled"] = True
        options["background_initial_scale"] = traffic_scale
        options["background_demand_scale"] = traffic_scale
    if stage >= 6:
        options["communication_delay_probability"] = 0.05
        options["communication_delay_min"] = 0.1
        options["communication_delay_max"] = 0.1
    if stage >= 7:
        options.update(
            {
                "front_vehicle_count": int(rng.integers(0, 9)),
                "wave_mode": "learned",
                "background_initial_scale": 1.0,
                "background_demand_scale": 1.0,
                "communication_delay_probability": 0.15,
                "communication_delay_min": 0.1,
                "communication_delay_max": 0.2,
            }
        )
        if (
            config.downstream_wave_training
            and rng.random() < config.downstream_wave_training_probability
        ):
            options.update(
                {
                    "front_vehicle_count": 1,
                    "wave_mode": "fixed",
                    "wave_min_source_distance_m": 0.0,
                    "wave_trigger_stations": [2_400.0, 2_650.0, 2_900.0],
                    "wave_trigger_reference": "source",
                    "cutin_enabled": False,
                    "v2i_enabled": False,
                    "communication_delay_probability": 0.0,
                    "background_initial_scale": 0.0,
                    "background_demand_scale": 0.0,
                    "initial_position_offset_m": 300.0,
                    "tail_background_count": 5,
                    "tail_background_gap_m": 45.0,
                    "tail_background_depart_speed_mps": 22.22,
                    "tail_background_type": "human_aggressive",
                    "downstream_wave_reward_enabled": True,
                }
            )
    return options, stage


def _follower_actor_curriculum_enabled(stage: int) -> bool:
    """Keep foundation control fixed only while the leader learns basic traffic adaptation."""

    return stage == 0 or stage >= FOLLOWER_ACTOR_ADAPTATION_STAGE


def _follower_actor_update_turn(
    stage: int,
    episode_number: int,
    enabled: bool,
    capability: Mapping[str, float],
) -> bool:
    if not enabled:
        return False
    criteria = CURRICULUM_CRITERIA[
        min(stage, len(CURRICULUM_CRITERIA) - 1)
    ]
    formation_recovery_needed = bool(
        stage >= 4
        and (
            capability.get("connected_pair_count", 0.0)
            < len(FOLLOWER_IDS)
            or capability.get("max_gap", float("inf"))
            > criteria["max_gap"]
        )
    )
    return bool(
        stage == 0
        or episode_number % 2 == 0
        or formation_recovery_needed
    )


def _set_actor_learning_rates(
    trainer: PlatoonTrainer,
    base_learning_rate: float,
    stage: int,
) -> tuple[float, float]:
    stage_index = min(stage, len(LEADER_ACTOR_LR_MULTIPLIERS) - 1)
    follower_rate = (
        base_learning_rate * FOLLOWER_ACTOR_LR_MULTIPLIERS[stage_index]
    )
    leader_rate = (
        base_learning_rate * LEADER_ACTOR_LR_MULTIPLIERS[stage_index]
    )
    for group in trainer.optimizers["follower_actor"].param_groups:
        group["lr"] = follower_rate
    for group in trainer.optimizers["leader_actor"].param_groups:
        group["lr"] = leader_rate
    return follower_rate, leader_rate


def _capability_summary(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    summary = {
        "success": float(np.mean([row["success"] for row in records])),
        "collision": float(np.mean([row["collision"] for row in records])),
        "failure": float(np.mean([row["failure"] for row in records])),
        "min_gap": float(min(row["min_gap"] for row in records)),
        "max_gap": float(max(row["max_gap"] for row in records)),
        "recovery": float(np.mean([row["recovery"] for row in records])),
        "shield": float(np.mean([row["shield"] for row in records])),
        "string": float(np.mean([row["string"] for row in records])),
    }
    for name in (
        "v2i_exposure",
        "v2i_score",
        "curve_overspeed",
        "curve_wasted_acceleration",
        "leader_jerk",
        "background_exposure",
        "cutin_exposure",
        "cutin_resolution",
        "delay_exposure",
    ):
        summary[name] = float(
            np.mean([row.get(name, 0.0) for row in records])
        )
    summary["observed_delay"] = float(
        max(row.get("observed_delay", 0.0) for row in records)
    )
    return summary


def _behavior_capability_meets_criteria(
    stage: int, summary: Mapping[str, float]
) -> bool:
    behavior = CURRICULUM_BEHAVIOR_CRITERIA[
        min(stage, len(CURRICULUM_BEHAVIOR_CRITERIA) - 1)
    ]
    if stage >= 3 and summary.get("v2i_exposure", 0.0) < 1.0:
        return False
    for criterion, limit in behavior.items():
        if criterion.endswith("_min"):
            metric = criterion.removesuffix("_min")
            if summary.get(metric, 0.0) < limit:
                return False
        elif criterion.endswith("_max"):
            metric = criterion.removesuffix("_max")
            if summary.get(metric, float("inf")) > limit:
                return False
        else:  # pragma: no cover - guards malformed local configuration
            raise ValueError(f"unsupported behavior criterion: {criterion}")
    return True


def _behavior_capability_deficits(
    stage: int, summary: Mapping[str, float]
) -> dict[str, float]:
    behavior = CURRICULUM_BEHAVIOR_CRITERIA[
        min(stage, len(CURRICULUM_BEHAVIOR_CRITERIA) - 1)
    ]
    deficits: dict[str, float] = {}
    if stage >= 3:
        deficits["v2i_exposure_min"] = max(
            0.0, 1.0 - summary.get("v2i_exposure", 0.0)
        )
    for criterion, limit in behavior.items():
        scale = max(abs(float(limit)), 1e-6)
        if criterion.endswith("_min"):
            metric = criterion.removesuffix("_min")
            deficits[criterion] = max(
                0.0, (limit - summary.get(metric, 0.0)) / scale
            )
        elif criterion.endswith("_max"):
            metric = criterion.removesuffix("_max")
            deficits[criterion] = max(
                0.0,
                (summary.get(metric, float("inf")) - limit) / scale,
            )
    return deficits


def _behavior_capability_fields(
    scenario_options: Mapping[str, Any],
    reward_component_sums: Mapping[str, float],
    sample_count: int,
    final_info: Mapping[str, Any],
    background_active_peak: int,
    cutin_states_seen: set[str],
    observed_delay: float,
) -> dict[str, float]:
    denominator = max(int(sample_count), 1)
    downstream_activity = max(
        float(reward_component_sums.get("downstream_wave_active", 0.0)),
        1e-6,
    )
    cutin_exposed = any(
        state in {"aligning", "blocked", "success", "failed"}
        for state in cutin_states_seen
    )
    cutin_resolved = any(
        state in {"blocked", "success"} for state in cutin_states_seen
    )
    return {
        "v2i_exposure": float(scenario_options.get("v2i_enabled", False)),
        "v2i_score": float(
            reward_component_sums.get("v2i_speed_reward", 0.0)
            / denominator
        ),
        "curve_overspeed": float(
            reward_component_sums.get("v2i_curve_overspeed_error", 0.0)
            / denominator
        ),
        "curve_wasted_acceleration": float(
            reward_component_sums.get(
                "v2i_wasted_curve_acceleration", 0.0
            )
            / denominator
        ),
        "leader_jerk": abs(
            float(
                reward_component_sums.get("leader_jerk", 0.0)
                / denominator
            )
        ),
        "background_exposure": float(background_active_peak > 0),
        "cutin_exposure": float(cutin_exposed),
        "cutin_resolution": float(cutin_resolved),
        "delay_exposure": float(
            int(final_info.get("communication_delay_events", 0)) > 0
        ),
        "observed_delay": float(observed_delay),
        "downstream_wave_exposure": float(
            scenario_options.get("downstream_wave_reward_enabled", False)
        ),
        "downstream_wave_penalty": abs(
            float(
                reward_component_sums.get("downstream_wave_control", 0.0)
                / denominator
            )
        ),
        "downstream_wave_f5_speed_gain": float(
            reward_component_sums.get("downstream_wave_f5_speed_gain", 0.0)
            / downstream_activity
        ),
        "downstream_wave_b5_speed_gain": float(
            reward_component_sums.get("downstream_wave_b5_speed_gain", 0.0)
            / downstream_activity
        ),
        "downstream_wave_acceleration_gain": float(
            reward_component_sums.get(
                "downstream_wave_acceleration_gain", 0.0
            )
            / downstream_activity
        ),
        "downstream_wave_mobility_ratio": float(
            reward_component_sums.get(
                "downstream_wave_mobility_weighted", 0.0
            )
            / downstream_activity
        ),
    }


def _capability_meets_criteria(
    stage: int, summary: Mapping[str, float]
) -> bool:
    if not summary or stage >= len(CURRICULUM_CRITERIA):
        return False
    criteria = CURRICULUM_CRITERIA[stage]
    physical_pass = bool(
        summary["success"] >= criteria["success"]
        and summary["collision"] == 0.0
        and summary["failure"] <= criteria["failure"]
        and summary["min_gap"] >= criteria["min_gap"]
        and summary["max_gap"] <= criteria["max_gap"]
        and summary["recovery"] <= criteria["recovery"]
        and summary["shield"] <= criteria["shield"]
        and summary["string"] <= criteria["string"]
    )
    return physical_pass and _behavior_capability_meets_criteria(stage, summary)


def _leader_restart_failure_episode_eligible(
    stage: int,
    capability: Mapping[str, float],
    has_restart_samples: bool,
) -> bool:
    """Admit cohesive stalled or materially slow episodes to recovery updates."""

    criteria = CURRICULUM_CRITERIA[
        min(stage, len(CURRICULUM_CRITERIA) - 1)
    ]
    recovery_failure = bool(
        capability["failure"] >= 0.5
        or (
            capability.get("success", 0.0) < 0.5
            and capability.get("leader_station", 0.0)
            < LEADER_RECOVERY_MIN_PROGRESS_STATION
        )
    )
    return bool(
        has_restart_samples
        and recovery_failure
        and capability["collision"] == 0.0
        and capability["active_count"] == len(PLATOON_IDS)
        and capability["connected_pair_count"] == len(FOLLOWER_IDS)
        and capability["min_gap"] >= max(2.0, 0.8 * criteria["min_gap"])
        # Window sampling and the loss both require every pair gap <= 14 m.
        # A later split must not discard earlier cohesive low-speed samples.
        and capability["max_gap"] <= REPLAY_MAX_GAP
    )


def _candidate_validation_passes(
    stage: int,
    candidate: Mapping[str, float],
    origin: Mapping[str, float],
    actor_kind: str = "generic",
) -> bool:
    """Accept a safe paired candidate that passes outright or improves a weak baseline."""

    criteria = CURRICULUM_CRITERIA[
        min(stage, len(CURRICULUM_CRITERIA) - 1)
    ]
    strongly_recovered_gap = bool(
        origin["max_gap"] > criteria["max_gap"]
        and candidate["max_gap"]
        <= min(
            0.75 * origin["max_gap"],
            1.20 * criteria["max_gap"],
        )
        and candidate["recovery"] <= 0.60
    )
    recovery_safe = bool(
        candidate["recovery"]
        <= max(criteria["recovery"], origin["recovery"] + 0.02)
        or strongly_recovered_gap
    )
    completed_safe = bool(
        candidate["success"] >= 0.5
        and candidate["collision"] == 0.0
        and candidate["failure"] == 0.0
        and candidate["min_gap"]
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate["shield"]
        <= min(0.60, max(criteria["shield"], origin["shield"] + 0.02))
        and candidate["string"]
        <= max(criteria["string"], origin["string"] + 0.10)
        and recovery_safe
    )
    if _progressive_failure_candidate_passes(
        stage, candidate, origin, actor_kind=actor_kind
    ):
        return True
    if not completed_safe:
        return False
    if _capability_meets_criteria(stage, candidate):
        return True
    origin_behavior_deficits = _behavior_capability_deficits(stage, origin)
    candidate_behavior_deficits = _behavior_capability_deficits(
        stage, candidate
    )
    origin_behavior_total = sum(origin_behavior_deficits.values())
    candidate_behavior_total = sum(candidate_behavior_deficits.values())
    behavior_improved = bool(
        origin_behavior_total > 0.0
        and candidate_behavior_total
        <= origin_behavior_total
        - max(
            CANDIDATE_MIN_BEHAVIOR_DEFICIT_IMPROVEMENT,
            0.05 * origin_behavior_total,
        )
        and all(
            candidate_behavior_deficits.get(name, 0.0)
            <= deficit + 0.01
            for name, deficit in origin_behavior_deficits.items()
        )
        and candidate["max_gap"] <= origin["max_gap"] + 0.25
    )
    if behavior_improved:
        return True
    required_gap_improvement = max(0.25, 0.005 * origin["max_gap"])
    return bool(
        origin["max_gap"] > criteria["max_gap"]
        and candidate["max_gap"]
        <= origin["max_gap"] - required_gap_improvement
    )


def _progressive_failure_candidate_passes(
    stage: int,
    candidate: Mapping[str, float],
    origin: Mapping[str, float],
    actor_kind: str = "generic",
) -> bool:
    """Allow a safe failed baseline to improve without one-shot completion."""

    required = (
        "leader_station",
        "active_count",
        "connected_pair_count",
    )
    if any(name not in candidate or name not in origin for name in required):
        return False
    if origin.get("success", 0.0) >= 0.5:
        return False
    origin_hard_failure = origin.get("failure", 0.0) >= 0.5
    criteria = CURRICULUM_CRITERIA[
        min(stage, len(CURRICULUM_CRITERIA) - 1)
    ]
    leader_updated = actor_kind in {"leader", "joint"}
    follower_updated = actor_kind in {"follower", "joint"}
    origin_station = float(origin["leader_station"])
    remaining = max(0.0, 3450.0 - origin_station)
    required_progress = min(250.0, max(50.0, 0.10 * remaining))
    minimum_pairs = max(
        4.0,
        min(5.0, float(origin["connected_pair_count"]) - 1.0),
    )
    progress_shield_limit = min(
        0.55,
        max(
            1.5 * criteria["shield"],
            float(origin["shield"]) + 0.30,
        ),
    )
    max_gap_limit = min(
        REPLAY_MAX_GAP,
        max(
            1.5 * criteria["max_gap"],
            float(origin["max_gap"]) + 2.0,
        ),
    )
    common_safe = bool(
        candidate.get("success", 0.0) < 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("connected_pair_count", 0.0) >= minimum_pairs
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("max_gap", float("inf")) <= max_gap_limit
        and candidate.get("recovery", float("inf")) <= 0.60
        and candidate.get("string", float("inf"))
        <= max(criteria["string"], float(origin["string"]) + 0.50)
    )
    progress_improved = bool(
        origin_hard_failure
        and candidate.get("leader_station", 0.0)
        >= origin_station + required_progress
        and candidate.get("shield", float("inf"))
        <= progress_shield_limit
    )
    formation_improved = bool(
        origin_hard_failure
        and candidate.get("failure", 0.0) >= 0.5
        and candidate.get("leader_station", 0.0) >= origin_station - 50.0
        and candidate.get("connected_pair_count", 0.0)
        > origin.get("connected_pair_count", 0.0)
        and candidate.get("max_gap", float("inf"))
        <= origin.get("max_gap", float("inf")) - 1.0
        and candidate.get("shield", float("inf"))
        <= min(0.55, origin.get("shield", 0.0) + 0.02)
    )
    truncated_gap_improved = bool(
        not origin_hard_failure
        and candidate.get("failure", 0.0) < 0.5
        and candidate.get("leader_station", 0.0) >= origin_station - 50.0
        and candidate.get("connected_pair_count", 0.0)
        >= origin.get("connected_pair_count", 0.0)
        and candidate.get("max_gap", float("inf"))
        <= origin.get("max_gap", float("inf"))
        - max(2.0, 0.10 * origin.get("max_gap", 0.0))
        and candidate.get("shield", float("inf"))
        <= min(0.55, origin.get("shield", 0.0) + 0.05)
    )
    # A follower update can substantially repair a severe background-traffic
    # split while the unchanged leader still reaches its independent stall
    # terminal. Do not discard that recovery signal, but require a large,
    # physically safe formation improvement and bound the progress tradeoff.
    follower_formation_recovered = bool(
        follower_updated
        and not origin_hard_failure
        and candidate.get("failure", 0.0) >= 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("leader_station", 0.0) >= origin_station - 250.0
        and candidate.get("connected_pair_count", 0.0)
        >= max(4.0, origin.get("connected_pair_count", 0.0) + 1.0)
        and candidate.get("max_gap", float("inf"))
        <= min(REPLAY_MAX_GAP, 0.80 * origin.get("max_gap", 0.0))
        and candidate.get("recovery", float("inf"))
        <= min(0.60, origin.get("recovery", 0.0) - 0.10)
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("shield", float("inf"))
        <= min(
            0.55,
            max(criteria["shield"], origin.get("shield", 0.0) + 0.15),
        )
        and candidate.get("string", float("inf"))
        <= max(criteria["string"], origin.get("string", 0.0) + 0.30)
    )
    follower_major_recovery = bool(
        follower_updated
        and not origin_hard_failure
        and candidate.get("failure", 0.0) < 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("connected_pair_count", 0.0) >= len(FOLLOWER_IDS)
        and candidate.get("leader_station", 0.0) >= origin_station - 750.0
        and candidate.get("max_gap", float("inf"))
        <= 0.30 * origin.get("max_gap", 0.0)
        and candidate.get("recovery", float("inf"))
        <= min(0.60, origin.get("recovery", 0.0) + 0.15)
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("shield", float("inf")) <= criteria["shield"]
        and candidate.get("string", float("inf")) <= criteria["string"]
    )
    follower_gap_recovered = bool(
        follower_updated
        and origin_hard_failure
        and candidate.get("failure", 0.0) < 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("connected_pair_count", 0.0)
        >= max(3.0, origin.get("connected_pair_count", 0.0))
        and candidate.get("leader_station", 0.0) >= origin_station + 100.0
        and candidate.get("max_gap", float("inf"))
        <= min(
            0.60 * origin.get("max_gap", 0.0),
            3.0 * criteria["max_gap"],
        )
        and candidate.get("recovery", float("inf"))
        <= origin.get("recovery", 0.0) - 0.05
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("shield", float("inf"))
        <= min(0.55, origin.get("shield", 0.0))
        and candidate.get("string", float("inf"))
        <= min(
            criteria["string"] + 0.10,
            origin.get("string", 0.0) + 0.40,
        )
    )
    leader_stall_recovery_improved = bool(
        leader_updated
        and origin_hard_failure
        and candidate.get("failure", 0.0) >= 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("connected_pair_count", 0.0) >= len(FOLLOWER_IDS)
        and candidate.get("leader_station", 0.0) >= origin_station + 20.0
        and candidate.get("max_gap", float("inf"))
        <= max(criteria["max_gap"], origin.get("max_gap", 0.0) + 0.50)
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("shield", float("inf"))
        <= min(0.65, origin.get("shield", 0.0) - 0.02)
        and candidate.get("recovery", float("inf"))
        <= max(criteria["recovery"], origin.get("recovery", 0.0) + 0.02)
        and candidate.get("string", float("inf"))
        <= max(criteria["string"], origin.get("string", 0.0) + 0.10)
        and candidate.get("v2i_score", 0.0)
        >= origin.get("v2i_score", 0.0) - 0.01
    )
    leader_truncated_progress_improved = bool(
        leader_updated
        and not origin_hard_failure
        and candidate.get("failure", 0.0) < 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("connected_pair_count", 0.0)
        >= origin.get("connected_pair_count", 0.0)
        and candidate.get("leader_station", 0.0)
        >= origin_station + required_progress
        and candidate.get("max_gap", float("inf"))
        <= min(
            criteria["max_gap"],
            origin.get("max_gap", float("inf")) + 0.25,
        )
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("shield", float("inf"))
        <= min(0.55, origin.get("shield", 0.0))
        and candidate.get("recovery", float("inf"))
        <= min(0.60, origin.get("recovery", 0.0) + 0.04)
        and candidate.get("string", float("inf"))
        <= max(criteria["string"], origin.get("string", 0.0) + 0.15)
        and candidate.get("v2i_score", 0.0)
        >= min(
            origin.get("v2i_score", 0.0) + 0.05,
            CURRICULUM_BEHAVIOR_CRITERIA[stage].get(
                "v2i_score_min", float("inf")
            ),
        )
    )
    follower_severe_truncated_gap_recovered = bool(
        follower_updated
        and not origin_hard_failure
        and candidate.get("failure", 0.0) < 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("connected_pair_count", 0.0)
        >= origin.get("connected_pair_count", 0.0)
        and origin.get("max_gap", 0.0) > 5.0 * criteria["max_gap"]
        and candidate.get("leader_station", 0.0) >= origin_station - 250.0
        and candidate.get("max_gap", float("inf"))
        <= 0.85 * origin.get("max_gap", 0.0)
        and candidate.get("recovery", float("inf"))
        <= origin.get("recovery", 0.0) - 0.02
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("shield", float("inf"))
        <= min(0.55, origin.get("shield", 0.0) + 0.02)
        and candidate.get("string", float("inf"))
        <= max(criteria["string"], origin.get("string", 0.0) + 0.10)
    )
    follower_cohesive_stall_gap_improved = bool(
        follower_updated
        and origin_hard_failure
        and candidate.get("failure", 0.0) >= 0.5
        and candidate.get("collision", 0.0) == 0.0
        and candidate.get("active_count", 0.0) >= len(PLATOON_IDS)
        and candidate.get("connected_pair_count", 0.0) >= len(FOLLOWER_IDS)
        and origin.get("connected_pair_count", 0.0) >= len(FOLLOWER_IDS)
        and candidate.get("leader_station", 0.0) >= origin_station - 50.0
        and candidate.get("max_gap", float("inf"))
        <= origin.get("max_gap", 0.0)
        - max(1.0, 0.05 * origin.get("max_gap", 0.0))
        and candidate.get("recovery", float("inf"))
        <= origin.get("recovery", 0.0)
        and candidate.get("min_gap", 0.0)
        >= max(2.0, 0.8 * criteria["min_gap"])
        and candidate.get("shield", float("inf"))
        <= min(0.50, origin.get("shield", 0.0) + 0.03)
        and candidate.get("string", float("inf"))
        <= min(criteria["string"] + 0.10, origin.get("string", 0.0) + 0.15)
    )
    return (
        common_safe
        and (
            progress_improved
            or formation_improved
            or truncated_gap_improved
            or follower_formation_recovered
            or follower_major_recovery
            or leader_stall_recovery_improved
            or leader_truncated_progress_improved
            or follower_severe_truncated_gap_recovered
            or follower_cohesive_stall_gap_improved
        )
    ) or follower_gap_recovered


def _next_candidate_origin(
    episode_capability: Mapping[str, float],
    candidate_validation: Mapping[str, Any] | None,
    candidate_rejected: bool,
) -> dict[str, float]:
    """Keep the accepted-policy baseline when chaining an update after rollback."""

    if candidate_rejected and candidate_validation is not None:
        return copy.deepcopy(candidate_validation["origin_capability"])
    return copy.deepcopy(dict(episode_capability))


def _advance_actor_cooldown(
    cooldown_episodes: int,
    replay_eligible: bool,
    rolled_back: bool,
) -> int:
    """Count safe frozen rollouts, not only already-successful rollouts."""

    if replay_eligible and cooldown_episodes > 0 and not rolled_back:
        return cooldown_episodes - 1
    return cooldown_episodes


def _actor_update_window_open(
    follower_rolled_back: bool,
    leader_rolled_back: bool,
) -> bool:
    return not follower_rolled_back and not leader_rolled_back


def _stage_capability_passes(
    stage: int,
    stage_steps: int,
    records: list[dict[str, float]],
) -> tuple[bool, dict[str, float]]:
    summary = _capability_summary(records)
    if stage >= len(CURRICULUM_CRITERIA):
        return False, summary
    if (
        stage_steps < CURRICULUM_MIN_STAGE_STEPS[stage]
        or len(records) < CURRICULUM_WINDOW
    ):
        return False, summary
    passed = _capability_meets_criteria(stage, summary)
    return passed, summary


def _new_episode() -> dict[str, Any]:
    return {
        "leader_observations": {key: [] for key in LEADER_KEYS},
        "leader_actions": [],
        "rewards": [],
        "follower_rewards": [],
        "continues": [],
        "follower_states": [],
        "follower_actions": [],
        "follower_behavior_actions": [],
        "follower_requested_actions": [],
        "follower_next_states": [],
    }


def _append_transition(
    episode: dict[str, Any],
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: Mapping[str, Mapping[str, Any]],
    leader_reward: float,
    follower_reward: Mapping[str, float] | float,
    continued: bool,
    states: np.ndarray | None,
    next_states: np.ndarray | None,
    action_infos: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    def applied_acceleration(agent: str) -> np.ndarray:
        info = action_infos.get(agent, {}) if action_infos is not None else {}
        value = info.get("applied_acceleration", actions[agent]["accel"])
        return np.asarray(value, np.float32).reshape(1)

    def requested_acceleration(agent: str) -> np.ndarray:
        return np.asarray(actions[agent]["accel"], np.float32).reshape(1)

    for key in LEADER_KEYS:
        episode["leader_observations"][key].append(
            np.asarray(observations["leader"][key], np.float32).copy()
        )
    episode["leader_actions"].append(applied_acceleration("leader"))
    episode["rewards"].append(float(leader_reward))
    if isinstance(follower_reward, Mapping):
        follower_reward_vector = np.asarray(
            [follower_reward[agent] for agent in FOLLOWER_IDS], np.float32
        )
    else:
        follower_reward_vector = np.full(
            len(FOLLOWER_IDS), float(follower_reward), dtype=np.float32
        )
    episode["follower_rewards"].append(follower_reward_vector)
    episode["continues"].append(float(continued))
    if states is not None and next_states is not None:
        episode["follower_states"].append(states)
        episode["follower_actions"].append(
            np.stack([applied_acceleration(agent) for agent in FOLLOWER_IDS])
        )
        episode["follower_behavior_actions"].append(
            np.stack([applied_acceleration(agent) for agent in FOLLOWER_IDS])
        )
        episode["follower_requested_actions"].append(
            np.stack([requested_acceleration(agent) for agent in FOLLOWER_IDS])
        )
        episode["follower_next_states"].append(next_states)


def _attach_follower_predictions(
    env: PlatoonParallelEnv, policy_states: Tensor
) -> None:
    env.set_follower_control_predictions(
        PlatoonController.follower_control_predictions(policy_states)
    )


def _apply_correlated_follower_exploration(
    actions: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    common_std: float = 0.12,
    independent_std: float = 0.0,
    episode_step: int = 0,
) -> None:
    common_noise = float(rng.normal(0.0, common_std))
    for vehicle_index, vehicle_id in enumerate(PLATOON_IDS):
        base = float(np.asarray(actions[vehicle_id]["accel"]).reshape(-1)[0])
        independent_noise = 0.0
        if vehicle_id in FOLLOWER_IDS and independent_std > 0.0:
            # Use the discrete derivative of a periodic velocity probe. Its
            # velocity and displacement integrals both return to zero every
            # cycle, unlike white acceleration noise, which creates an
            # artificial random-walk separation over a long episode.
            period_steps = 40 + 4 * (vehicle_index - 1)
            phase = 2.0 * math.pi * (episode_step % period_steps) / period_steps
            next_phase = 2.0 * math.pi * (
                (episode_step + 1) % period_steps
            ) / period_steps
            normalization = 2.0 * math.sin(math.pi / period_steps)
            sign = -1.0 if vehicle_index % 2 == 0 else 1.0
            independent_noise = sign * independent_std * (
                math.sin(next_phase) - math.sin(phase)
            ) / normalization
        actions[vehicle_id]["accel"] = np.asarray(
            [np.clip(base + common_noise + independent_noise, -3.0, 1.2)],
            dtype=np.float32,
        )


def _behavior_cloning_weight(
    config: FollowerRLConfig,
    total_steps: int,
    actor_warmup_steps: int,
    policy_update_interval: int = 8,
) -> float:
    if config.behavior_cloning_decay_steps <= 0:
        return float(config.behavior_cloning_final_weight)
    elapsed = max(0, total_steps - actor_warmup_steps)
    actor_updates = elapsed // max(1, policy_update_interval)
    return float(
        max(
            config.behavior_cloning_final_weight,
            config.behavior_cloning_weight * (0.999**actor_updates),
        )
    )


def evaluate_controller(
    controller: PlatoonController,
    config: RLConfig,
    step: int,
    curriculum_stage: int,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    rewards: list[float] = []
    successes = 0
    collisions = 0
    failures = 0
    capabilities: list[dict[str, float]] = []
    for episode_index in range(config.evaluation_episodes):
        env = PlatoonParallelEnv(
            calibration_path=config.calibration_path,
            wave_model_path=config.wave_model_path,
            max_time=config.episode_max_time,
        )
        evaluation_rng = np.random.default_rng(
            config.seed + 20_000 + curriculum_stage * 1_000 + episode_index
        )
        options, _ = _curriculum_options(
            curriculum_stage, evaluation_rng, config, step
        )
        observations, _ = env.reset(
            seed=config.seed + 10_000 + episode_index,
            options=options,
        )
        controller.reset()
        total_reward = 0.0
        count = 0
        final_info: dict[str, Any] = {}
        pair_gaps: list[float] = []
        safety_interventions: list[float] = []
        recovery_ratios: list[float] = []
        reward_component_sums: dict[str, float] = {}
        background_active_peak = 0
        cutin_states_seen: set[str] = set()
        observed_delay = 0.0
        applied_accelerations: dict[str, list[float]] = {
            agent: [] for agent in PLATOON_IDS
        }
        try:
            while env.agents:
                with torch.no_grad():
                    policy_states = controller.follower_policy_states(
                        observations
                    )[0]
                _attach_follower_predictions(env, policy_states)
                actions = controller.act(
                    observations,
                    deterministic=True,
                    policy_states=policy_states,
                )
                observations, step_rewards, terminations, truncations, infos = env.step(actions)
                controller.record_applied_actions(infos)
                leader_info = infos["leader"]
                for name, value in leader_info["reward_components"].items():
                    reward_component_sums[name] = (
                        reward_component_sums.get(name, 0.0) + float(value)
                    )
                background_active_peak = max(
                    background_active_peak,
                    int(leader_info.get("background_active_count", 0)),
                )
                cutin_states_seen.add(str(leader_info.get("cutin_state", "idle")))
                observed_delay = max(
                    observed_delay,
                    *(
                        max(
                            float(infos[agent].get("communication_delay", 0.0)),
                            float(
                                infos[agent].get(
                                    "predecessor_communication_delay", 0.0
                                )
                            ),
                        )
                        for agent in FOLLOWER_IDS
                    ),
                )
                active = set(env.connection.vehicle.getIDList())
                for follower in FOLLOWER_IDS:
                    if follower not in active:
                        continue
                    physical_front = env.connection.vehicle.getLeader(
                        follower, 500.0
                    )
                    if physical_front and physical_front[0] == PREDECESSOR[follower]:
                        gap = float(
                            physical_front[1]
                            + env.connection.vehicle.getMinGap(follower)
                        )
                    else:
                        gap = env._platoon_gap(follower, active)
                    if gap >= 0.0:
                        pair_gaps.append(gap)
                    safety_interventions.append(
                        float(
                            infos[follower].get(
                                "collision_avoidance_intervened", False
                            )
                            or infos[follower].get(
                                "predictive_safety_intervened", False
                            )
                        )
                    )
                for agent in PLATOON_IDS:
                    if agent in infos and "applied_acceleration" in infos[agent]:
                        applied_accelerations[agent].append(
                            float(infos[agent]["applied_acceleration"])
                        )
                recovery_ratios.append(
                    float(
                        infos["leader"]["reward_components"].get(
                            "diagnostic_recovery_active_ratio", 0.0
                        )
                    )
                )
                total_reward += step_rewards["leader"]
                count += 1
                final_info = leader_info
                if progress_callback is not None and count % 100 == 0:
                    progress_callback(
                        {
                            "evaluation/active": 1.0,
                            "evaluation/live_episode": float(episode_index + 1),
                            "evaluation/live_episode_steps": float(count),
                            "evaluation/live_min_gap": float(
                                min(pair_gaps, default=0.0)
                            ),
                            "evaluation/live_max_gap": float(
                                max(pair_gaps, default=0.0)
                            ),
                            "evaluation/live_background_peak": float(
                                background_active_peak
                            ),
                        }
                    )
                if any(terminations.values()) or any(truncations.values()):
                    break
        finally:
            env.close()
        rewards.append(total_reward / max(count, 1))
        successes += int(final_info.get("success", False))
        collisions += int(final_info.get("collision", False))
        failures += int(final_info.get("failure", False))
        leader_rms = float(
            np.sqrt(np.mean(np.square(applied_accelerations["leader"])))
        )
        follower_rms = max(
            (
                float(np.sqrt(np.mean(np.square(values))))
                for agent, values in applied_accelerations.items()
                if agent != "leader" and values
            ),
            default=0.0,
        )
        capabilities.append(
            {
                "success": float(final_info.get("success", False)),
                "collision": float(final_info.get("collision", False)),
                "failure": float(final_info.get("failure", False)),
                "min_gap": float(min(pair_gaps, default=0.0)),
                "max_gap": float(max(pair_gaps, default=500.0)),
                "recovery": float(np.mean(recovery_ratios or [0.0])),
                "shield": float(np.mean(safety_interventions or [0.0])),
                "string": follower_rms / max(leader_rms, 0.1),
                **_behavior_capability_fields(
                    options,
                    reward_component_sums,
                    count,
                    final_info,
                    background_active_peak,
                    cutin_states_seen,
                    observed_delay,
                ),
            }
        )
    capability = _capability_summary(capabilities)
    capability_pass = float(
        _capability_meets_criteria(curriculum_stage, capability)
    )
    return {
        "evaluation/step": float(step),
        "evaluation/reward_per_step": float(np.mean(rewards)),
        "evaluation/success_rate": successes / config.evaluation_episodes,
        "evaluation/collision_rate": collisions / config.evaluation_episodes,
        "evaluation/failure_rate": failures / config.evaluation_episodes,
        "evaluation/curriculum_stage": float(curriculum_stage),
        "evaluation/min_gap": capability["min_gap"],
        "evaluation/max_gap": capability["max_gap"],
        "evaluation/recovery": capability["recovery"],
        "evaluation/shield": capability["shield"],
        "evaluation/string": capability["string"],
        "evaluation/v2i_score": capability["v2i_score"],
        "evaluation/curve_overspeed": capability["curve_overspeed"],
        "evaluation/curve_wasted_acceleration": capability[
            "curve_wasted_acceleration"
        ],
        "evaluation/leader_jerk": capability["leader_jerk"],
        "evaluation/background_exposure": capability[
            "background_exposure"
        ],
        "evaluation/cutin_exposure": capability["cutin_exposure"],
        "evaluation/cutin_resolution": capability["cutin_resolution"],
        "evaluation/delay_exposure": capability["delay_exposure"],
        "evaluation/observed_delay": capability["observed_delay"],
        "evaluation/capability_pass": capability_pass,
    }


def _save_training_state(
    controller: PlatoonController,
    trainer: PlatoonTrainer,
    replay: EpisodeReplay,
    config: RLConfig,
    total_steps: int,
    episodes: int,
    best_score: float,
    stale_evaluations: int,
    exploration_rng: np.random.Generator,
    curriculum_stage: int,
    recent_successes: list[float],
    stage_started_step: int,
    recent_capabilities: list[dict[str, float]],
    accepted_follower_actor_state: Mapping[str, Tensor] | None = None,
    accepted_follower_optimizer_state: Mapping[str, Any] | None = None,
    accepted_leader_actor_state: Mapping[str, Tensor] | None = None,
    accepted_leader_critic_state: Mapping[str, Tensor] | None = None,
    accepted_leader_optimizer_state: Mapping[str, Any] | None = None,
    follower_actor_cooldown_episodes: int = 0,
    leader_actor_cooldown_episodes: int = 0,
) -> Path:
    output = Path(config.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "rl_latest.pt"
    temporary = checkpoint_path.with_suffix(".tmp")
    model_state = controller.state_dict()
    optimizer_state = {
        key: value.state_dict() for key, value in trainer.optimizers.items()
    }
    # Checkpoints must contain the last policy that completed a full episode,
    # never an unvalidated between-episode Actor candidate.
    if accepted_follower_actor_state is not None:
        follower_state = dict(model_state["follower"])
        for key, value in accepted_follower_actor_state.items():
            follower_state[f"actor.{key}"] = value.detach().clone()
        model_state = dict(model_state)
        model_state["follower"] = follower_state
    if accepted_follower_optimizer_state is not None:
        optimizer_state["follower_actor"] = copy.deepcopy(
            accepted_follower_optimizer_state
        )
    if (
        accepted_leader_actor_state is not None
        and accepted_leader_critic_state is not None
    ):
        leader_state = dict(model_state["leader"])
        for key, value in accepted_leader_actor_state.items():
            leader_state[f"actor.{key}"] = value.detach().clone()
        for key, value in accepted_leader_critic_state.items():
            leader_state[f"critic.{key}"] = value.detach().clone()
        model_state = dict(model_state)
        model_state["leader"] = leader_state
    if accepted_leader_optimizer_state is not None:
        optimizer_state["leader_actor"] = copy.deepcopy(
            accepted_leader_optimizer_state
        )
    torch.save(
        {
            "format_version": 11,
            "candidate_guard_version": 3,
            "follower_actor_objective_version": (
                FOLLOWER_ACTOR_OBJECTIVE_VERSION
            ),
            "follower_state_schema_version": FOLLOWER_STATE_SCHEMA_VERSION,
            "terminal_success_schema_version": TERMINAL_SUCCESS_SCHEMA_VERSION,
            "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
            "actor_cooldown_schema_version": ACTOR_COOLDOWN_SCHEMA_VERSION,
            "control_architecture": CONTROL_ARCHITECTURE,
            "config": controller.config.to_dict(),
            "rl_config": asdict(config),
            "models": model_state,
            "optimizers": optimizer_state,
            "total_steps": total_steps,
            "episodes": episodes,
            "best_score": best_score,
            "stale_evaluations": stale_evaluations,
            "curriculum_stage": curriculum_stage,
            "recent_successes": recent_successes,
            "stage_started_step": stage_started_step,
            "recent_capabilities": recent_capabilities,
            "follower_actor_cooldown_episodes": (
                follower_actor_cooldown_episodes
            ),
            "leader_actor_cooldown_episodes": (
                leader_actor_cooldown_episodes
            ),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "exploration_random_state": exploration_rng.bit_generator.state,
        },
        temporary,
    )
    temporary.replace(checkpoint_path)
    replay_path = output / "replay_latest.pt"
    replay_temporary = replay_path.with_suffix(".tmp")
    torch.save(replay.state_dict(), replay_temporary)
    replay_temporary.replace(replay_path)
    return checkpoint_path


def train_rl(
    config: RLConfig,
    predictor_directory: str | Path = "artifacts/checkpoints",
    resume: str | Path | None = None,
    reset_replay: bool = False,
) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    checkpoint = (
        torch.load(resume, map_location="cpu", weights_only=False)
        if resume is not None
        else None
    )
    migrating_control_architecture = bool(
        checkpoint is not None
        and checkpoint.get("control_architecture") != CONTROL_ARCHITECTURE
    )
    migrating_follower_state_schema = bool(
        checkpoint is not None
        and int(checkpoint.get("follower_state_schema_version", 1))
        < FOLLOWER_STATE_SCHEMA_VERSION
    )
    migrating_terminal_success_schema = bool(
        checkpoint is not None
        and int(checkpoint.get("terminal_success_schema_version", 0))
        < TERMINAL_SUCCESS_SCHEMA_VERSION
    )
    migrating_capability_schema = bool(
        checkpoint is not None
        and int(checkpoint.get("capability_schema_version", 0))
        < CAPABILITY_SCHEMA_VERSION
    )
    migrating_actor_cooldown_schema = bool(
        checkpoint is not None
        and int(checkpoint.get("actor_cooldown_schema_version", 1))
        < ACTOR_COOLDOWN_SCHEMA_VERSION
    )
    system = (
        SystemConfig.from_dict(checkpoint["config"])
        if checkpoint is not None
        else SystemConfig()
    )
    if migrating_control_architecture:
        system.follower_rl = SystemConfig().follower_rl
    system.training.device = config.device
    system.training.batch_size = config.batch_size
    controller = PlatoonController(system)
    _load_predictors(controller, predictor_directory)
    trainer = PlatoonTrainer(controller)
    replay = EpisodeReplay(config.replay_capacity, config.seed)
    total_steps = 0
    episodes = 0
    best_score = -float("inf")
    stale_evaluations = 0
    curriculum_stage = 0
    recent_successes: list[float] = []
    stage_started_step = 0
    recent_capabilities: list[dict[str, float]] = []
    next_evaluation = config.evaluation_interval
    next_checkpoint = config.checkpoint_interval
    if checkpoint is not None:
        if migrating_control_architecture:
            print(
                json.dumps(
                    {
                        "migration": "reinitialized_gated_pbrs_v6_control_stack",
                        "source_checkpoint": str(resume),
                        "training_steps_reset": True,
                        "retained": "offline_delay_trajectory_and_front_predictors",
                    }
                ),
                flush=True,
            )
        else:
            controller.load_state_dict(checkpoint["models"])
            for key, state in checkpoint.get("optimizers", {}).items():
                if key in trainer.optimizers:
                    try:
                        trainer.optimizers[key].load_state_dict(state)
                    except ValueError:
                        pass
            # Optimizer state loading restores the historical parameter-group
            # learning rate.  Keep Actor candidates on the smaller trust-step
            # rate even when resuming checkpoints created before that rule.
            actor_learning_rates = {
                "follower_actor": system.training.learning_rate * 0.01,
                "leader_actor": system.training.learning_rate * 0.1,
            }
            for actor_optimizer_name, actor_learning_rate in (
                actor_learning_rates.items()
            ):
                for parameter_group in trainer.optimizers[
                    actor_optimizer_name
                ].param_groups:
                    parameter_group["lr"] = actor_learning_rate
            checkpoint_follower_objective_version = int(
                checkpoint.get("follower_actor_objective_version", 1)
            )
            if (
                checkpoint_follower_objective_version
                < FOLLOWER_ACTOR_OBJECTIVE_VERSION
            ):
                trainer.optimizers["follower_actor"].state.clear()
                print(
                    json.dumps(
                        {
                            "migration": "reset_follower_actor_optimizer",
                            "from_objective_version": (
                                checkpoint_follower_objective_version
                            ),
                            "to_objective_version": (
                                FOLLOWER_ACTOR_OBJECTIVE_VERSION
                            ),
                            "retained": "follower_actor_weights",
                        }
                    ),
                    flush=True,
                )
            total_steps = int(checkpoint["total_steps"])
            episodes = int(checkpoint["episodes"])
            best_score = float(checkpoint["best_score"])
            stale_evaluations = int(checkpoint["stale_evaluations"])
            curriculum_stage = int(checkpoint.get("curriculum_stage", 0))
            recent_successes = list(checkpoint.get("recent_successes", []))
            stage_started_step = int(
                checkpoint.get("stage_started_step", total_steps)
            )
            recent_capabilities = list(
                checkpoint.get("recent_capabilities", [])
            )
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.random.set_rng_state(checkpoint["torch_random_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_random_state"):
                torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
            rng.bit_generator.state = checkpoint["exploration_random_state"]
            if (
                not reset_replay
                and not migrating_follower_state_schema
                and not migrating_terminal_success_schema
            ):
                replay.load_state_dict(
                    torch.load(
                        Path(resume).with_name("replay_latest.pt"),
                        weights_only=False,
                    )
                )
            else:
                stage_started_step = total_steps
                recent_successes = []
                recent_capabilities = []
                if migrating_follower_state_schema:
                    print(
                        json.dumps(
                            {
                                "migration": "reset_incompatible_follower_replay",
                                "to_state_schema_version": (
                                    FOLLOWER_STATE_SCHEMA_VERSION
                                ),
                                "retained": (
                                    "accepted_models_and_curriculum_stage"
                                ),
                            }
                        ),
                        flush=True,
                    )
                if migrating_terminal_success_schema:
                    print(
                        json.dumps(
                            {
                                "migration": "reset_invalid_terminal_success_replay",
                                "to_terminal_success_schema_version": (
                                    TERMINAL_SUCCESS_SCHEMA_VERSION
                                ),
                                "success_requires": (
                                    "all_six_trucks_five_connected_gaps_"
                                    "leader_destination_no_intrusion"
                                ),
                                "retained": (
                                    "accepted_models_and_curriculum_stage"
                                ),
                            }
                        ),
                        flush=True,
                    )
            if migrating_capability_schema:
                recent_successes = []
                recent_capabilities = []
                print(
                    json.dumps(
                        {
                            "migration": "reset_incomplete_capability_window",
                            "to_capability_schema_version": (
                                CAPABILITY_SCHEMA_VERSION
                            ),
                            "new_gates": (
                                "terminal_progress_and_background_capabilities"
                            ),
                            "retained": (
                                "accepted_models_replay_stage_and_stage_steps"
                            ),
                        }
                    ),
                    flush=True,
                )
            if migrating_actor_cooldown_schema:
                print(
                    json.dumps(
                        {
                            "migration": "reset_legacy_actor_cooldown",
                            "to_actor_cooldown_schema_version": (
                                ACTOR_COOLDOWN_SCHEMA_VERSION
                            ),
                            "reason": (
                                "legacy cooldown only counted successful rollouts"
                            ),
                            "retained": "accepted_models_replay_and_curriculum",
                        }
                    ),
                    flush=True,
                )
            next_evaluation = (
                (total_steps // config.evaluation_interval) + 1
            ) * config.evaluation_interval
            next_checkpoint = (
                (total_steps // config.checkpoint_interval) + 1
            ) * config.checkpoint_interval
    logger = MetricLogger(Path(config.checkpoint_dir) / "logs")
    latest_checkpoint = Path(config.checkpoint_dir) / "rl_latest.pt"
    legacy_follower_cooldown = 0
    legacy_leader_cooldown = 0
    if checkpoint is not None:
        legacy_capabilities = list(recent_capabilities)
        rejected_indices = [
            index
            for index, record in enumerate(legacy_capabilities)
            if float(record.get("failure", 0.0)) >= 0.5
            or float(record.get("collision", 0.0)) >= 0.5
        ]
        if rejected_indices:
            episodes_since_rejection = (
                len(legacy_capabilities) - 1 - rejected_indices[-1]
            )
            legacy_follower_cooldown = max(
                0,
                ACTOR_ROLLBACK_COOLDOWN_EPISODES
                - episodes_since_rejection,
            )
            legacy_leader_cooldown = max(
                0,
                ACTOR_ROLLBACK_COOLDOWN_EPISODES
                - episodes_since_rejection,
            )
    accepted_follower_actor_state = copy.deepcopy(
        controller.follower.actor.state_dict()
    )
    accepted_follower_optimizer_state = copy.deepcopy(
        trainer.optimizers["follower_actor"].state_dict()
    )
    follower_actor_candidate_pending = False
    follower_candidate_validation: dict[str, Any] | None = None
    candidate_guard_version = int(
        checkpoint.get("candidate_guard_version", 1)
        if checkpoint is not None
        else 2
    )
    follower_actor_cooldown_episodes = (
        0
        if migrating_actor_cooldown_schema
        else min(
            ACTOR_ROLLBACK_COOLDOWN_EPISODES,
            int(
                checkpoint.get(
                    "follower_actor_cooldown_episodes",
                    legacy_follower_cooldown,
                )
                if checkpoint is not None and candidate_guard_version >= 2
                else 0
            ),
        )
    )
    accepted_leader_actor_state = copy.deepcopy(
        controller.leader.actor.state_dict()
    )
    accepted_leader_critic_state = copy.deepcopy(
        controller.leader.critic.state_dict()
    )
    accepted_leader_optimizer_state = copy.deepcopy(
        trainer.optimizers["leader_actor"].state_dict()
    )
    leader_actor_candidate_pending = False
    leader_candidate_validation: dict[str, Any] | None = None
    leader_actor_cooldown_episodes = (
        0
        if migrating_actor_cooldown_schema
        else min(
            ACTOR_ROLLBACK_COOLDOWN_EPISODES,
            int(
                checkpoint.get(
                    "leader_actor_cooldown_episodes",
                    legacy_leader_cooldown,
                )
                if checkpoint is not None and candidate_guard_version >= 2
                else 0
            ),
        )
    )
    foundation_best_max_gap = min(
        (
            float(record["max_gap"])
            for record in recent_capabilities
            if float(record.get("success", 0.0)) >= 0.5
        ),
        default=float("inf"),
    )

    while total_steps < config.max_sumo_steps:
        env = PlatoonParallelEnv(
            calibration_path=config.calibration_path,
            wave_model_path=config.wave_model_path,
            max_time=config.episode_max_time,
        )
        episode_curriculum_stage = curriculum_stage
        (
            follower_actor_learning_rate,
            leader_actor_learning_rate,
        ) = _set_actor_learning_rates(
            trainer,
            controller.config.training.learning_rate,
            episode_curriculum_stage,
        )
        # Scenario difficulty must not depend on how many random exploration
        # actions the previous episode happened to consume.  A per-episode RNG
        # makes curriculum coverage reproducible while `rng` remains dedicated
        # to action exploration and replay sampling.
        candidate_validation = (
            follower_candidate_validation
            if follower_actor_candidate_pending
            else leader_candidate_validation
            if leader_actor_candidate_pending
            else None
        )
        if candidate_validation is None:
            scenario_rng = np.random.default_rng(
                config.seed + 50_000 + episodes
            )
            scenario_options, _ = _curriculum_options(
                episode_curriculum_stage,
                scenario_rng,
                config,
                total_steps,
            )
            episode_reset_seed = config.seed + episodes
        else:
            scenario_options = copy.deepcopy(
                candidate_validation["scenario_options"]
            )
            episode_reset_seed = int(candidate_validation["reset_seed"])
        observations, _ = env.reset(
            seed=episode_reset_seed, options=scenario_options
        )
        controller.reset()
        episode = _new_episode()
        episode_reward = 0.0
        episode_follower_reward = 0.0
        reward_component_sums: dict[str, float] = {}
        leader_commands: list[float] = []
        leader_restart_commands: list[float] = []
        follower_commands: list[float] = []
        leader_applied_commands: list[float] = []
        leader_restart_applied_commands: list[float] = []
        follower_applied_commands: list[float] = []
        follower_safety_interventions: list[float] = []
        pair_gap_samples: list[float] = []
        background_active_peak = 0
        cutin_states_seen: set[str] = set()
        observed_delay = 0.0
        applied_accelerations: dict[str, list[float]] = {
            agent: [] for agent in ("leader", *FOLLOWER_IDS)
        }
        final_info: dict[str, Any] = {}
        train_metrics: dict[str, float] = {}
        episode_control_step = 0
        pre_episode_capability = _capability_summary(recent_capabilities)
        episode_stage_quality_ready = bool(
            len(recent_capabilities) >= CURRICULUM_WINDOW
            and _capability_meets_criteria(
                episode_curriculum_stage,
                pre_episode_capability,
            )
        )
        episode_stage_finalization = bool(
            total_steps - stage_started_step
            >= 0.75
            * CURRICULUM_MIN_STAGE_STEPS[
                min(
                    episode_curriculum_stage,
                    len(CURRICULUM_MIN_STAGE_STEPS) - 1,
                )
            ]
            and episode_stage_quality_ready
        )
        episode_policy_validation = bool(
            follower_actor_candidate_pending
            or follower_actor_cooldown_episodes > 0
            or leader_actor_candidate_pending
            or leader_actor_cooldown_episodes > 0
            or episode_stage_finalization
        )
        front_vehicle_count = env._front_vehicle_count
        demand_profile = env._demand_profile_id
        with torch.no_grad():
            follower_state_tensor = controller.follower_policy_states(observations)[0]
        _attach_follower_predictions(env, follower_state_tensor)
        follower_states = follower_state_tensor.cpu().numpy()
        try:
            while env.agents and total_steps < config.max_sumo_steps:
                # Each policy starts exploration only when its actor is allowed
                # to learn; otherwise an untrained frozen actor corrupts replay.
                actions = controller.act(
                    observations,
                    # Exploration is injected below as one coherent platoon
                    # disturbance. Independent leader noise creates an
                    # artificial string-instability training signal.
                    leader_deterministic=True,
                    follower_deterministic=True,
                    policy_states=follower_state_tensor,
                )
                # controller.act may inject a same-tick zero-delay leader
                # broadcast into the final history row. Replay must store that
                # exact causal state rather than its one-tick-old copy.
                follower_states = follower_state_tensor.cpu().numpy()
                # Collect coherent non-zero actions for critic/world-model
                # identification before either actor is allowed to update.
                if (
                    total_steps >= config.warmup_steps
                    and not episode_policy_validation
                    and episode_curriculum_stage < 4
                    and _follower_actor_curriculum_enabled(
                        episode_curriculum_stage
                    )
                ):
                    follower_actor_active = (
                        replay.steps + len(episode["rewards"])
                        >= config.actor_warmup_steps
                    )
                    _apply_correlated_follower_exploration(
                        actions,
                        rng,
                        common_std=(0.03 if follower_actor_active else 0.12),
                        independent_std=(
                            0.008 if follower_actor_active else 0.0
                        ),
                        episode_step=episode_control_step,
                    )
                _attach_follower_predictions(env, follower_state_tensor)
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                controller.record_applied_actions(infos)
                final_info = infos["leader"]
                for name, value in final_info["reward_components"].items():
                    reward_component_sums[name] = reward_component_sums.get(name, 0.0) + float(value)
                background_active_peak = max(
                    background_active_peak,
                    int(final_info.get("background_active_count", 0)),
                )
                cutin_states_seen.add(
                    str(final_info.get("cutin_state", "idle"))
                )
                observed_delay = max(
                    observed_delay,
                    *(
                        max(
                            float(infos[agent].get("communication_delay", 0.0)),
                            float(
                                infos[agent].get(
                                    "predecessor_communication_delay", 0.0
                                )
                            ),
                        )
                        for agent in FOLLOWER_IDS
                    ),
                )
                leader_commands.append(
                    float(np.asarray(actions["leader"]["accel"]).reshape(-1)[0])
                )
                follower_commands.extend(
                    float(np.asarray(actions[agent]["accel"]).reshape(-1)[0])
                    for agent in FOLLOWER_IDS
                )
                leader_applied_commands.append(
                    float(infos["leader"]["applied_acceleration"])
                )
                restart_observation = next_observations.get(
                    "leader", observations["leader"]
                )
                if (
                    final_info.get("leader_restart_opportunity", False)
                    and float(restart_observation["ego"][1]) < 5.0
                ):
                    leader_restart_commands.append(
                        float(
                            np.asarray(actions["leader"]["accel"]).reshape(-1)[0]
                        )
                    )
                    leader_restart_applied_commands.append(
                        float(infos["leader"]["applied_acceleration"])
                    )
                follower_applied_commands.extend(
                    float(infos[agent]["applied_acceleration"])
                    for agent in FOLLOWER_IDS
                )
                follower_safety_interventions.extend(
                    float(
                        infos[agent].get("collision_avoidance_intervened", False)
                        or infos[agent].get("predictive_safety_intervened", False)
                    )
                    for agent in FOLLOWER_IDS
                )
                active_vehicles = set(env.connection.vehicle.getIDList())
                for agent in FOLLOWER_IDS:
                    if agent not in active_vehicles:
                        continue
                    physical_front = env.connection.vehicle.getLeader(
                        agent, 500.0
                    )
                    if (
                        physical_front
                        and physical_front[0] == PREDECESSOR[agent]
                    ):
                        pair_gap = float(
                            physical_front[1]
                            + env.connection.vehicle.getMinGap(agent)
                        )
                    else:
                        pair_gap = env._platoon_gap(
                            agent, active_vehicles
                        )
                    if pair_gap >= 0.0:
                        pair_gap_samples.append(pair_gap)
                for agent in applied_accelerations:
                    applied_accelerations[agent].append(
                        float(infos[agent]["applied_acceleration"])
                    )
                continued = bool(next_observations)
                next_follower_states = follower_states.copy()
                next_follower_state_tensor = follower_state_tensor
                if continued:
                    with torch.no_grad():
                        next_follower_state_tensor = (
                            controller.follower_policy_states(next_observations)[0]
                        )
                        next_follower_states = (
                            next_follower_state_tensor.cpu().numpy()
                        )
                _append_transition(
                    episode,
                    observations,
                    actions,
                    rewards["leader"],
                    {agent: rewards[agent] for agent in FOLLOWER_IDS},
                    continued,
                    follower_states,
                    next_follower_states,
                    action_infos=infos,
                )
                episode_reward += rewards["leader"]
                episode_follower_reward += float(
                    np.mean([rewards[agent] for agent in FOLLOWER_IDS])
                )
                total_steps += 1
                episode_control_step += 1
                available_training_steps = (
                    replay.steps + len(episode["rewards"])
                )
                if (
                    not episode_policy_validation
                    and
                    available_training_steps >= config.warmup_steps
                    and total_steps % config.update_every == 0
                ):
                    train_metrics.update(
                        trainer.train_followers(
                            *replay.sample_followers(
                                config.batch_size, extra_episode=episode
                            ),
                            # Keep the policy fixed throughout an episode.
                            # Actor candidates are updated only after a full
                            # successful rollout and validated next episode.
                            update_actor=False,
                            behavior_weight=_behavior_cloning_weight(
                                controller.config.follower_rl,
                                total_steps,
                                config.actor_warmup_steps,
                                config.policy_update_interval,
                            )
                            if episode_curriculum_stage > 0
                            else controller.config.follower_rl.behavior_cloning_weight,
                        )
                    )
                if (
                    not episode_policy_validation
                    and
                    available_training_steps >= config.leader_warmup_steps
                    and total_steps % config.leader_update_interval == 0
                    and replay.can_sample(
                        config.sequence_length,
                        config.batch_size,
                        extra_episode=episode,
                    )
                ):
                    train_metrics.update(
                        trainer.train_leader(
                            *replay.sample_leader(
                                config.batch_size,
                                config.sequence_length,
                                extra_episode=episode,
                            ),
                            # Keep the leader policy fixed within an episode;
                            # actor candidates are produced between episodes.
                            update_actor=False,
                            behavior_weight=0.5
                            * _behavior_cloning_weight(
                                controller.config.follower_rl,
                                total_steps,
                                config.leader_actor_warmup_steps,
                                config.leader_update_interval,
                            ),
                        )
                    )
                observations = next_observations
                follower_states = next_follower_states
                follower_state_tensor = next_follower_state_tensor
                if any(terminations.values()) or any(truncations.values()):
                    break
        finally:
            env.close()
        episodes += 1
        recent_successes.append(float(final_info.get("success", False)))
        recent_successes = recent_successes[-20:]
        leader_rms = float(
            np.sqrt(np.mean(np.square(applied_accelerations["leader"])))
        )
        follower_rms = max(
            (
                float(np.sqrt(np.mean(np.square(values))))
                for agent, values in applied_accelerations.items()
                if agent != "leader" and values
            ),
            default=0.0,
        )
        string_amplification = follower_rms / max(leader_rms, 0.1)
        episode_capability = {
            "success": float(final_info.get("success", False)),
            "collision": float(final_info.get("collision", False)),
            "failure": float(final_info.get("failure", False)),
            "active_count": float(
                final_info.get("platoon_active_count", 0)
            ),
            "connected_pair_count": float(
                final_info.get("platoon_connected_pair_count", 0)
            ),
            "leader_station": float(final_info.get("leader_station", 0.0)),
            "min_gap": float(min(pair_gap_samples, default=0.0)),
            "max_gap": float(max(pair_gap_samples, default=500.0)),
            "recovery": float(
                reward_component_sums.get(
                    "diagnostic_recovery_active_ratio", 0.0
                )
                / max(len(episode["rewards"]), 1)
            ),
            "shield": float(np.mean(follower_safety_interventions)),
            "string": string_amplification,
            **_behavior_capability_fields(
                scenario_options,
                reward_component_sums,
                len(episode["rewards"]),
                final_info,
                background_active_peak,
                cutin_states_seen,
                observed_delay,
            ),
        }
        episode_criteria = CURRICULUM_CRITERIA[
            min(episode_curriculum_stage, len(CURRICULUM_CRITERIA) - 1)
        ]
        foundation_candidate_within_drift_limit = bool(
            episode_curriculum_stage != 0
            or not follower_actor_candidate_pending
            or not math.isfinite(foundation_best_max_gap)
            or episode_capability["max_gap"]
            <= foundation_best_max_gap + FOUNDATION_MAX_GAP_DRIFT
        )
        episode_accepted = bool(
            episode_capability["success"] >= 0.5
            and episode_capability["collision"] == 0.0
            and episode_capability["failure"] == 0.0
            and episode_capability["min_gap"]
            >= max(2.0, 0.8 * episode_criteria["min_gap"])
            and episode_capability["max_gap"]
            <= 1.25 * episode_criteria["max_gap"]
            and episode_capability["shield"]
            <= min(0.60, 1.5 * episode_criteria["shield"])
            and foundation_candidate_within_drift_limit
        )
        episode_replay_eligible = bool(
            episode_capability["collision"] == 0.0
            and episode_capability["min_gap"] >= 2.0
            and episode_capability["max_gap"] <= REPLAY_MAX_GAP
            and episode_capability["shield"] <= 0.60
        )
        leader_restart_failure_eligible = (
            _leader_restart_failure_episode_eligible(
                episode_curriculum_stage,
                episode_capability,
                bool(leader_restart_commands),
            )
        )
        episode_capability_meets_criteria = _capability_meets_criteria(
            episode_curriculum_stage,
            episode_capability,
        )
        follower_candidate_accepted = bool(
            follower_actor_candidate_pending
            and follower_candidate_validation is not None
            and _candidate_validation_passes(
                episode_curriculum_stage,
                episode_capability,
                follower_candidate_validation["origin_capability"],
                actor_kind="follower",
            )
        )
        leader_candidate_accepted = bool(
            leader_actor_candidate_pending
            and leader_candidate_validation is not None
            and _candidate_validation_passes(
                episode_curriculum_stage,
                episode_capability,
                leader_candidate_validation["origin_capability"],
                actor_kind="leader",
            )
        )
        follower_candidate_progress_accepted = bool(
            follower_actor_candidate_pending
            and follower_candidate_validation is not None
            and _progressive_failure_candidate_passes(
                episode_curriculum_stage,
                episode_capability,
                follower_candidate_validation["origin_capability"],
                actor_kind="follower",
            )
        )
        leader_candidate_progress_accepted = bool(
            leader_actor_candidate_pending
            and leader_candidate_validation is not None
            and _progressive_failure_candidate_passes(
                episode_curriculum_stage,
                episode_capability,
                leader_candidate_validation["origin_capability"],
                actor_kind="leader",
            )
        )
        actor_rolled_back = False
        leader_actor_rolled_back = False
        if (
            follower_candidate_accepted
            or (not follower_actor_candidate_pending and episode_accepted)
        ):
            # The candidate has now survived a complete fixed-policy rollout.
            accepted_follower_actor_state = copy.deepcopy(
                controller.follower.actor.state_dict()
            )
            accepted_follower_optimizer_state = copy.deepcopy(
                trainer.optimizers["follower_actor"].state_dict()
            )
            follower_actor_candidate_pending = False
            follower_candidate_validation = None
            if episode_curriculum_stage == 0:
                foundation_best_max_gap = min(
                    foundation_best_max_gap,
                    episode_capability["max_gap"],
                )
        elif follower_actor_candidate_pending:
            controller.follower.actor.load_state_dict(
                accepted_follower_actor_state
            )
            trainer.optimizers["follower_actor"].load_state_dict(
                accepted_follower_optimizer_state
            )
            follower_actor_candidate_pending = False
            follower_candidate_validation = None
            actor_rolled_back = True
            follower_actor_cooldown_episodes = (
                0
                if episode_replay_eligible
                else ACTOR_ROLLBACK_COOLDOWN_EPISODES
            )
        if (
            leader_candidate_accepted
            or (not leader_actor_candidate_pending and episode_accepted)
        ):
            accepted_leader_actor_state = copy.deepcopy(
                controller.leader.actor.state_dict()
            )
            accepted_leader_critic_state = copy.deepcopy(
                controller.leader.critic.state_dict()
            )
            accepted_leader_optimizer_state = copy.deepcopy(
                trainer.optimizers["leader_actor"].state_dict()
            )
            leader_actor_candidate_pending = False
            leader_candidate_validation = None
        elif leader_actor_candidate_pending:
            controller.leader.actor.load_state_dict(
                accepted_leader_actor_state
            )
            controller.leader.critic.load_state_dict(
                accepted_leader_critic_state
            )
            trainer.optimizers["leader_actor"].load_state_dict(
                accepted_leader_optimizer_state
            )
            leader_actor_candidate_pending = False
            leader_candidate_validation = None
            leader_actor_rolled_back = True
            leader_actor_cooldown_episodes = (
                0
                if episode_replay_eligible
                else ACTOR_ROLLBACK_COOLDOWN_EPISODES
            )
        if episode_replay_eligible:
            replay.add(episode)
            follower_actor_cooldown_episodes = _advance_actor_cooldown(
                follower_actor_cooldown_episodes,
                episode_replay_eligible,
                actor_rolled_back,
            )
            leader_actor_cooldown_episodes = _advance_actor_cooldown(
                leader_actor_cooldown_episodes,
                episode_replay_eligible,
                leader_actor_rolled_back,
            )
        next_candidate_origin = _next_candidate_origin(
            episode_capability,
            candidate_validation,
            actor_rolled_back or leader_actor_rolled_back,
        )
        recent_capabilities.append(episode_capability)
        recent_capabilities = recent_capabilities[-CURRICULUM_WINDOW:]
        stage_steps = total_steps - stage_started_step
        stage_advanced, capability_summary = _stage_capability_passes(
            curriculum_stage, stage_steps, recent_capabilities
        )
        recent_success_rate = capability_summary.get("success", 0.0)
        final_stage_candidate = bool(
            stage_advanced
            and curriculum_stage == len(CURRICULUM_NAMES) - 1
        )
        if stage_advanced and not final_stage_candidate:
            curriculum_stage += 1
            stage_started_step = total_steps
            recent_successes = []
            recent_capabilities = []

        actor_update_batches = 0
        follower_actor_update_enabled = (
            _follower_actor_curriculum_enabled(
                episode_curriculum_stage
            )
        )
        follower_actor_update_turn = _follower_actor_update_turn(
            episode_curriculum_stage,
            episodes,
            follower_actor_update_enabled,
            episode_capability,
        )
        if (
            episode_replay_eligible
            and not episode_capability_meets_criteria
            and follower_actor_update_turn
            and not final_stage_candidate
            and not stage_advanced
            and not episode_stage_finalization
            and follower_actor_cooldown_episodes == 0
            and _actor_update_window_open(
                actor_rolled_back, leader_actor_rolled_back
            )
            and total_steps < next_evaluation
            and total_steps >= config.actor_warmup_steps
            and replay.steps >= config.actor_warmup_steps
        ):
            actor_stage = min(
                episode_curriculum_stage,
                len(FOLLOWER_ACTOR_RL_WEIGHTS) - 1,
            )
            behavior_weight = (
                _behavior_cloning_weight(
                    controller.config.follower_rl,
                    total_steps,
                    config.actor_warmup_steps,
                    config.policy_update_interval,
                )
                if actor_stage > 0
                else controller.config.follower_rl.behavior_cloning_weight
            )
            for _ in range(
                FOLLOWER_ACTOR_UPDATES_PER_EPISODE[actor_stage]
            ):
                follower_batch = replay.sample_followers(
                    config.batch_size,
                    extra_episode=episode,
                )
                train_metrics.update(
                    trainer.train_follower_actor(
                        follower_batch[0],
                        follower_batch[5],
                        follower_batch[6],
                        behavior_weight=behavior_weight,
                        rl_weight=FOLLOWER_ACTOR_RL_WEIGHTS[actor_stage],
                        stability_weight=FOLLOWER_ACTOR_STABILITY_WEIGHTS[
                            actor_stage
                        ],
                    )
                )
                actor_update_batches += 1
            follower_actor_candidate_pending = True
            follower_candidate_validation = {
                "scenario_options": copy.deepcopy(scenario_options),
                "reset_seed": episode_reset_seed,
                "origin_capability": copy.deepcopy(next_candidate_origin),
            }
        leader_actor_update_batches = 0
        leader_restart_actor_update_batches = 0
        if (
            (episode_replay_eligible or leader_restart_failure_eligible)
            and not episode_capability_meets_criteria
            and episode_curriculum_stage >= 1
            and (not follower_actor_update_turn or leader_restart_failure_eligible)
            and not final_stage_candidate
            and not stage_advanced
            and not episode_stage_finalization
            and leader_actor_cooldown_episodes == 0
            and _actor_update_window_open(
                actor_rolled_back, leader_actor_rolled_back
            )
            and total_steps < next_evaluation
            and total_steps >= config.leader_actor_warmup_steps
            and replay.can_sample(
                config.sequence_length,
                config.batch_size,
            )
        ):
            leader_stage = min(
                episode_curriculum_stage,
                len(LEADER_ACTOR_UPDATES_PER_EPISODE) - 1,
            )
            leader_behavior_weight = (
                0.0
                if episode_capability["success"] < 0.5
                else 0.5
                * _behavior_cloning_weight(
                    controller.config.follower_rl,
                    total_steps,
                    config.leader_actor_warmup_steps,
                    config.leader_update_interval,
                )
            )
            if leader_restart_failure_eligible:
                for _ in range(
                    LEADER_RESTART_UPDATES_PER_EPISODE[leader_stage]
                ):
                    try:
                        restart_batch = replay.sample_leader_restart_episode(
                            episode,
                            config.batch_size,
                            config.sequence_length,
                        )
                    except ValueError:
                        break
                    train_metrics.update(
                        trainer.train_leader_restart_actor(
                            restart_batch[0], restart_batch[1]
                        )
                    )
                    leader_restart_actor_update_batches += 1
            for _ in range(
                LEADER_ACTOR_UPDATES_PER_EPISODE[leader_stage]
            ):
                train_metrics.update(
                    trainer.train_leader(
                        *replay.sample_leader(
                            config.batch_size,
                            config.sequence_length,
                            extra_episode=(
                                episode if episode_replay_eligible else None
                            ),
                        ),
                        update_actor=True,
                        behavior_weight=leader_behavior_weight,
                        restart_weight=LEADER_RESTART_AUX_WEIGHTS[
                            leader_stage
                        ],
                    )
                )
            leader_actor_candidate_pending = True
            leader_candidate_validation = {
                "scenario_options": copy.deepcopy(scenario_options),
                "reset_seed": episode_reset_seed,
                "origin_capability": copy.deepcopy(next_candidate_origin),
            }
            leader_actor_update_batches = LEADER_ACTOR_UPDATES_PER_EPISODE[
                leader_stage
            ]
        episode_metrics = {
            "training/episode_reward": episode_reward,
            "training/reward_per_step": episode_reward / max(len(episode["rewards"]), 1),
            "training/follower_episode_reward": episode_follower_reward,
            "training/follower_reward_per_step": episode_follower_reward
            / max(len(episode["follower_rewards"]), 1),
            "training/episode_steps": float(len(episode["rewards"])),
            "training/replay_steps": float(replay.steps),
            "training/success": float(final_info.get("success", False)),
            "training/collision": float(final_info.get("collision", False)),
            "training/collision_platoon": float(
                final_info.get("collision_involves_platoon", False)
            ),
            "training/collision_ramp": float(
                final_info.get("collision_involves_ramp", False)
            ),
            "training/collision_cutin": float(
                final_info.get("collision_involves_cutin", False)
            ),
            "training/failure": float(final_info.get("failure", False)),
            "training/terminal_platoon_active_count": float(
                final_info.get("platoon_active_count", 0)
            ),
            "training/terminal_connected_pair_count": float(
                final_info.get("platoon_connected_pair_count", 0)
            ),
            "training/terminal_leader_station": float(
                final_info.get("leader_station", 0.0)
            ),
            "training/front_vehicle_count": float(front_vehicle_count),
            "training/demand_profile": float(demand_profile),
            "training/curriculum_stage": float(episode_curriculum_stage),
            "training/curriculum_stage_steps": float(stage_steps),
            "training/curriculum_advanced": float(stage_advanced),
            "training/curriculum_success_rate": recent_success_rate,
            # Per-episode values are intentionally logged separately from the
            # rolling curriculum summary.  The watchdog must not count one
            # rejected candidate repeatedly while it remains in the window.
            "training/episode_min_gap": episode_capability["min_gap"],
            "training/episode_max_gap": episode_capability["max_gap"],
            "training/episode_recovery": episode_capability["recovery"],
            "training/episode_shield": episode_capability["shield"],
            "training/episode_string": episode_capability["string"],
            "training/episode_v2i_score": episode_capability["v2i_score"],
            "training/episode_curve_overspeed": episode_capability[
                "curve_overspeed"
            ],
            "training/episode_curve_wasted_acceleration": episode_capability[
                "curve_wasted_acceleration"
            ],
            "training/episode_leader_jerk": episode_capability[
                "leader_jerk"
            ],
            "training/episode_background_exposure": episode_capability[
                "background_exposure"
            ],
            "training/episode_cutin_exposure": episode_capability[
                "cutin_exposure"
            ],
            "training/episode_cutin_resolution": episode_capability[
                "cutin_resolution"
            ],
            "training/episode_delay_exposure": episode_capability[
                "delay_exposure"
            ],
            "training/episode_observed_delay": episode_capability[
                "observed_delay"
            ],
            "training/capability_min_gap": capability_summary.get(
                "min_gap", 0.0
            ),
            "training/capability_max_gap": capability_summary.get(
                "max_gap", 0.0
            ),
            "training/capability_recovery": capability_summary.get(
                "recovery", 0.0
            ),
            "training/capability_shield": capability_summary.get(
                "shield", 0.0
            ),
            "training/capability_string": capability_summary.get(
                "string", 0.0
            ),
            "training/capability_v2i_score": capability_summary.get(
                "v2i_score", 0.0
            ),
            "training/capability_curve_overspeed": capability_summary.get(
                "curve_overspeed", 0.0
            ),
            "training/capability_curve_wasted_acceleration": (
                capability_summary.get("curve_wasted_acceleration", 0.0)
            ),
            "training/capability_background_exposure": (
                capability_summary.get("background_exposure", 0.0)
            ),
            "training/capability_cutin_exposure": capability_summary.get(
                "cutin_exposure", 0.0
            ),
            "training/capability_cutin_resolution": capability_summary.get(
                "cutin_resolution", 0.0
            ),
            "training/capability_delay_exposure": capability_summary.get(
                "delay_exposure", 0.0
            ),
            "training/capability_observed_delay": capability_summary.get(
                "observed_delay", 0.0
            ),
            "training/v2i_enabled": float(
                scenario_options.get("v2i_enabled", True)
            ),
            "training/cutin_enabled": float(
                scenario_options.get("cutin_enabled", True)
            ),
            "training/wave_enabled": float(
                scenario_options.get("wave_mode") == "learned"
            ),
            "training/delay_probability": float(
                scenario_options.get("communication_delay_probability", 0.0)
            ),
            "training/leader_action_mean": float(np.mean(leader_commands)),
            "training/leader_action_saturation": float(
                np.mean(
                    (np.asarray(leader_commands) <= -2.95)
                    | (np.asarray(leader_commands) >= 1.15)
                )
            ),
            "training/leader_restart_action_mean": float(
                np.mean(leader_restart_commands)
                if leader_restart_commands
                else 0.0
            ),
            "training/leader_restart_applied_action_mean": float(
                np.mean(leader_restart_applied_commands)
                if leader_restart_applied_commands
                else 0.0
            ),
            "training/leader_restart_sample_count": float(
                len(leader_restart_commands)
            ),
            "training/follower_action_mean": float(np.mean(follower_commands)),
            "training/leader_applied_action_mean": float(
                np.mean(leader_applied_commands)
            ),
            "training/follower_applied_action_mean": float(
                np.mean(follower_applied_commands)
            ),
            "training/follower_safety_intervention_rate": float(
                np.mean(follower_safety_interventions)
            ),
            "training/episode_accepted": float(episode_accepted),
            "training/candidate_validation_replay": float(
                candidate_validation is not None
            ),
            "training/follower_candidate_accepted": float(
                follower_candidate_accepted
            ),
            "training/leader_candidate_accepted": float(
                leader_candidate_accepted
            ),
            "training/follower_candidate_progress_accepted": float(
                follower_candidate_progress_accepted
            ),
            "training/leader_candidate_progress_accepted": float(
                leader_candidate_progress_accepted
            ),
            "training/episode_replay_eligible": float(
                episode_replay_eligible
            ),
            "training/leader_restart_failure_eligible": float(
                leader_restart_failure_eligible
            ),
            "training/episode_capability_pass": float(
                episode_capability_meets_criteria
            ),
            "training/policy_validation_rollout": float(
                episode_policy_validation
            ),
            "training/stage_finalization_rollout": float(
                episode_stage_finalization
            ),
            "training/stage_quality_ready": float(
                episode_stage_quality_ready
            ),
            "training/foundation_drift_limit": float(
                foundation_best_max_gap + FOUNDATION_MAX_GAP_DRIFT
                if math.isfinite(foundation_best_max_gap)
                else episode_criteria["max_gap"]
            ),
            "training/foundation_drift_guard_passed": float(
                foundation_candidate_within_drift_limit
            ),
            "training/follower_actor_candidate_pending": float(
                follower_actor_candidate_pending
            ),
            "training/follower_actor_rolled_back": float(
                actor_rolled_back
            ),
            "training/follower_actor_update_batches": float(
                actor_update_batches
            ),
            "training/actor_update_target_follower": float(
                follower_actor_update_turn
            ),
            "training/follower_actor_curriculum_enabled": float(
                follower_actor_update_enabled
            ),
            "training/follower_actor_cooldown_episodes": float(
                follower_actor_cooldown_episodes
            ),
            "training/leader_actor_candidate_pending": float(
                leader_actor_candidate_pending
            ),
            "training/leader_actor_rolled_back": float(
                leader_actor_rolled_back
            ),
            "training/leader_actor_update_batches": float(
                leader_actor_update_batches
            ),
            "training/leader_restart_actor_update_batches": float(
                leader_restart_actor_update_batches
            ),
            "training/leader_actor_cooldown_episodes": float(
                leader_actor_cooldown_episodes
            ),
            "training/behavior_cloning_weight": _behavior_cloning_weight(
                controller.config.follower_rl,
                total_steps,
                config.actor_warmup_steps,
                config.policy_update_interval,
            )
            if episode_curriculum_stage > 0
            else controller.config.follower_rl.behavior_cloning_weight,
            "training/follower_actor_learning_rate": float(
                follower_actor_learning_rate
            ),
            "training/leader_actor_learning_rate": float(
                leader_actor_learning_rate
            ),
            "training/follower_lane_change_rate": 0.0,
            "training/follower_controller_rl": 1.0,
            **{
                f"reward/{name}": total / max(len(episode["rewards"]), 1)
                for name, total in reward_component_sums.items()
            },
            **train_metrics,
        }
        logger.write(total_steps, episode_metrics)
        if final_info.get("collision_details"):
            print(
                json.dumps(
                    {
                        "event": "training_collision",
                        "steps": total_steps,
                        "episode": episodes,
                        "reset_seed": episode_reset_seed,
                        "scenario_options": scenario_options,
                        "collision_ids": final_info.get("collision_ids", []),
                        "collision_details": final_info["collision_details"],
                        "collision_involves_platoon": final_info.get(
                            "collision_involves_platoon", False
                        ),
                        "collision_involves_ramp": final_info.get(
                            "collision_involves_ramp", False
                        ),
                        "collision_involves_cutin": final_info.get(
                            "collision_involves_cutin", False
                        ),
                    }
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "steps": total_steps,
                    "episode": episodes,
                    "curriculum_name": CURRICULUM_NAMES[
                        episode_curriculum_stage
                    ],
                    "next_curriculum_name": (
                        "complete_pending_evaluation"
                        if final_stage_candidate
                        else CURRICULUM_NAMES[curriculum_stage]
                    ),
                    **episode_metrics,
                }
            ),
            flush=True,
        )

        final_evaluation_performed = False
        if final_stage_candidate:
            final_evaluation_performed = True
            evaluation = evaluate_controller(
                controller, config, total_steps, curriculum_stage
            )
            logger.write(total_steps, evaluation)
            print(
                json.dumps(
                    {
                        "steps": total_steps,
                        "curriculum_completion_evaluation": True,
                        **evaluation,
                    }
                ),
                flush=True,
            )
            if evaluation["evaluation/capability_pass"] >= 0.5:
                latest_checkpoint = _save_training_state(
                    controller,
                    trainer,
                    replay,
                    config,
                    total_steps,
                    episodes,
                    best_score,
                    stale_evaluations,
                    rng,
                    curriculum_stage,
                    recent_successes,
                    stage_started_step,
                    recent_capabilities,
                    accepted_follower_actor_state,
                    accepted_follower_optimizer_state,
                    accepted_leader_actor_state,
                    accepted_leader_critic_state,
                    accepted_leader_optimizer_state,
                    follower_actor_cooldown_episodes,
                    leader_actor_cooldown_episodes,
                )
                complete_path = (
                    Path(config.checkpoint_dir) / "rl_complete.pt"
                )
                complete_temporary = complete_path.with_suffix(".tmp")
                torch.save(
                    {
                        "config": controller.config.to_dict(),
                        "models": controller.state_dict(),
                        "metrics": evaluation,
                        "step": total_steps,
                        "curriculum_stage": curriculum_stage,
                        "curriculum_complete": True,
                    },
                    complete_temporary,
                )
                complete_temporary.replace(complete_path)
                completion_marker = (
                    Path(config.checkpoint_dir) / "training_complete.json"
                )
                completion_marker.write_text(
                    json.dumps(
                        {
                            "complete": True,
                            "step": total_steps,
                            "episode": episodes,
                            "stage": CURRICULUM_NAMES[curriculum_stage],
                            "evaluation": evaluation,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.close()
                return complete_path
            # Require a fresh eight-episode window before trying the final
            # completion evaluation again.
            recent_successes = []
            recent_capabilities = []

        # Persist accepted models and replay before a potentially long frozen
        # evaluation. The watchdog may deliberately stop a degraded run while
        # evaluation is in progress; delaying this save would discard the
        # trajectories that triggered the diagnosis.
        if total_steps >= next_checkpoint:
            latest_checkpoint = _save_training_state(
                controller,
                trainer,
                replay,
                config,
                total_steps,
                episodes,
                best_score,
                stale_evaluations,
                rng,
                curriculum_stage,
                recent_successes,
                stage_started_step,
                recent_capabilities,
                accepted_follower_actor_state,
                accepted_follower_optimizer_state,
                accepted_leader_actor_state,
                accepted_leader_critic_state,
                accepted_leader_optimizer_state,
                follower_actor_cooldown_episodes,
                leader_actor_cooldown_episodes,
            )
            next_checkpoint += config.checkpoint_interval

        if (
            not final_evaluation_performed
            and (
                total_steps >= next_evaluation
                or total_steps >= config.max_sumo_steps
            )
        ):
            evaluation = evaluate_controller(
                controller, config, total_steps, curriculum_stage
            )
            logger.write(total_steps, evaluation)
            criteria = CURRICULUM_CRITERIA[
                min(curriculum_stage, len(CURRICULUM_CRITERIA) - 1)
            ]
            score = (
                evaluation["evaluation/reward_per_step"]
                + 10.0 * evaluation["evaluation/success_rate"]
                - max(
                    0.0,
                    evaluation["evaluation/max_gap"] - criteria["max_gap"],
                )
                - 5.0 * evaluation["evaluation/shield"]
                - max(0.0, evaluation["evaluation/string"] - criteria["string"])
            )
            if (
                evaluation["evaluation/capability_pass"] >= 0.5
                and score > best_score + 0.01
            ):
                best_score = score
                stale_evaluations = 0
                best_path = Path(config.checkpoint_dir) / "rl_best.pt"
                best_temporary = best_path.with_suffix(".tmp")
                torch.save(
                    {"config": controller.config.to_dict(), "models": controller.state_dict(), "metrics": evaluation, "step": total_steps},
                    best_temporary,
                )
                best_temporary.replace(best_path)
            else:
                stale_evaluations = (
                    stale_evaluations + 1
                    if evaluation["evaluation/capability_pass"] >= 0.5
                    else 0
                )
            evaluation_curriculum_advanced = False
            if (
                evaluation["evaluation/capability_pass"] >= 0.5
                and curriculum_stage < len(CURRICULUM_NAMES) - 1
                and total_steps - stage_started_step
                >= CURRICULUM_MIN_STAGE_STEPS[curriculum_stage]
            ):
                # A frozen multi-episode evaluation is stronger evidence than
                # a training window that may still contain rejected candidates
                # or exploration rollouts.  Promote immediately to avoid
                # over-training a policy that has already passed the stage.
                curriculum_stage += 1
                stage_started_step = total_steps
                recent_successes = []
                recent_capabilities = []
                evaluation_curriculum_advanced = True
            latest_checkpoint = _save_training_state(
                controller,
                trainer,
                replay,
                config,
                total_steps,
                episodes,
                best_score,
                stale_evaluations,
                rng,
                curriculum_stage,
                recent_successes,
                stage_started_step,
                recent_capabilities,
                accepted_follower_actor_state,
                accepted_follower_optimizer_state,
                accepted_leader_actor_state,
                accepted_leader_critic_state,
                accepted_leader_optimizer_state,
                follower_actor_cooldown_episodes,
                leader_actor_cooldown_episodes,
            )
            print(
                json.dumps(
                    {
                        "steps": total_steps,
                        **evaluation,
                        "evaluation/curriculum_advanced": float(
                            evaluation_curriculum_advanced
                        ),
                        "evaluation/next_curriculum_stage": float(
                            curriculum_stage
                        ),
                        "stale_evaluations": stale_evaluations,
                    }
                ),
                flush=True,
            )
            next_evaluation += config.evaluation_interval
            next_checkpoint = max(next_checkpoint, next_evaluation - config.evaluation_interval + config.checkpoint_interval)
            # Curriculum completion, not a stale-score heuristic, is the only
            # successful stop condition. The external supervisor may still
            # interrupt a crashed or unhealthy run for diagnosis and resume.
    logger.close()
    return latest_checkpoint

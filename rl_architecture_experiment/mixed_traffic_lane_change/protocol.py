"""Frozen protocol for the mixed-traffic lane-change comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("graph", "cacc_lc", "ddpg_p", "ppo_p", "sac_p", "nc_kg_sac_p")
LEARNED_METHODS = tuple(method for method in METHODS if method != "cacc_lc")
DISPLAY_NAMES = {
    "graph": "LCG-Dreamer",
    "cacc_lc": "CACC+LC",
    "ddpg_p": "PA-DDPG",
    "ppo_p": "PA-PPO",
    "sac_p": "PA-SAC",
    "nc_kg_sac_p": "PA-KG-SAC",
}


@dataclass(frozen=True)
class MixedLaneProtocol:
    protocol_version: int = 5
    seed: int = 73_100
    imitation_episodes: int = 2
    training_episodes: int = 16
    evaluation_episodes: int = 20
    max_time: float = 100.0
    initial_position_offset_m: float = 250.0
    obstacle_stations: tuple[float, float, float] = (700.0, 1050.0, 1400.0)
    obstacle_lanes: tuple[int, int, int] = (1, 2, 1)
    target_lanes: tuple[int, int, int] = (2, 1, 0)
    goal_station: float = 1560.0
    obstacle_speed_min_mps: float = 7.0
    obstacle_speed_max_mps: float = 10.0
    evaluation_delay_probability: float = 0.15
    learning_rate: float = 3.0e-4
    imitation_epochs: int = 20
    ppo_epochs: int = 4
    minibatch_size: int = 256
    replay_capacity: int = 20_000
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.005
    max_grad_norm: float = 2.0
    imitation_prior_initial_weight: float = 1.0
    imitation_prior_decay_episodes: float = 8.0
    selection_interval_episodes: int = 4
    selection_seed_offsets: tuple[int, int, int] = (50_000, 50_001, 50_003)
    lane_change_gap_wait_grace_s: float = 30.0
    graph_checkpoint: str = (
        "experiments/downstream_wave_refinement/selected/"
        "graph_wave_selected_validated_v3.pt"
    )
    predictive_checkpoint_root: str = (
        "artifacts/predictive_comparison_v2/training"
    )
    output_root: str = "artifacts/mixed_traffic_lane_change_v5"

    def base_checkpoint(self, method: str) -> Path | None:
        if method == "graph":
            return Path(self.graph_checkpoint)
        if method in ("ddpg_p", "ppo_p", "sac_p", "nc_kg_sac_p"):
            # Predictor inputs/assets are equated, while each baseline retains
            # the longitudinal actor learned by its own algorithm.
            return Path(self.predictive_checkpoint_root) / method / "complete.pt"
        if method == "cacc_lc":
            return None
        raise ValueError(f"unknown mixed-lane method: {method}")

    def lateral_checkpoint(self, method: str) -> Path | None:
        if method == "cacc_lc":
            return None
        if method not in LEARNED_METHODS:
            raise ValueError(f"unknown mixed-lane method: {method}")
        return Path(self.output_root) / "training" / method / "best.pt"

    def scenario_options(
        self, episode_seed: int, training: bool
    ) -> dict[str, Any]:
        rng = np.random.default_rng(episode_seed)
        common_shift = float(rng.uniform(-10.0, 10.0)) if training else 0.0
        station_jitter = (
            rng.uniform(-7.0, 7.0, size=3)
            if training
            else np.zeros(3)
        )
        stations = tuple(
            float(base + common_shift + jitter)
            for base, jitter in zip(self.obstacle_stations, station_jitter)
        )
        intensity = float(rng.uniform(0.75, 1.25))
        front_gaps = tuple(
            float(np.clip(rng.uniform(31.0, 47.0) / intensity, 24.0, 58.0))
            for _ in range(3)
        )
        rear_gaps = tuple(
            float(np.clip(rng.uniform(19.0, 32.0) / intensity, 14.0, 43.0))
            for _ in range(3)
        )
        front_speeds = tuple(float(rng.uniform(21.5, 24.5)) for _ in range(3))
        rear_speeds = tuple(float(rng.uniform(25.0, 27.5)) for _ in range(3))
        return {
            "cooperative_lane_change_enabled": True,
            "lane_change_traffic_enabled": True,
            "lane_change_obstacle_stations": stations,
            "lane_change_obstacle_lanes": self.obstacle_lanes,
            "lane_change_target_lanes": self.target_lanes,
            "lane_change_obstacle_speed_mps": float(
                rng.uniform(
                    self.obstacle_speed_min_mps,
                    self.obstacle_speed_max_mps,
                )
            ),
            "lane_change_goal_station": self.goal_station + common_shift,
            "lane_change_gap_wait_grace_s": self.lane_change_gap_wait_grace_s,
            "lane_change_traffic_front_gaps_m": front_gaps,
            "lane_change_traffic_rear_gaps_m": rear_gaps,
            "lane_change_traffic_front_speeds_mps": front_speeds,
            "lane_change_traffic_rear_speeds_mps": rear_speeds,
            "initial_position_offset_m": self.initial_position_offset_m,
            "cutin_enabled": False,
            "v2i_enabled": True,
            "wave_mode": "off",
            "front_vehicle_count": 0,
            "background_initial_scale": 0.0,
            "background_demand_scale": 0.0,
            "communication_delay_probability": (
                float(rng.uniform(0.05, 0.15))
                if training
                else self.evaluation_delay_probability
            ),
            "communication_delay_min": 0.1,
            "communication_delay_max": 0.2,
            "speed_reference": 25.0,
            "mixed_lane_intensity": intensity,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "obstacle_stations",
            "obstacle_lanes",
            "target_lanes",
            "selection_seed_offsets",
        ):
            payload[key] = list(payload[key])
        canonical = json.dumps(payload, sort_keys=True)
        payload["fingerprint"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        payload["methods"] = list(METHODS)
        payload["reward_is_method_invariant"] = True
        payload["base_controllers_frozen"] = True
        payload["predictor_assets_equated_across_learned_baselines"] = True
        payload["learned_methods_share_predictive_longitudinal_foundation"] = False
        payload["controller_checkpoints_are_algorithm_specific"] = True
        payload["cacc_lateral_controller"] = (
            "independent per-vehicle gap/TTC rule; no team planner"
        )
        payload["learned_lateral_parameters_shared_across_methods"] = False
        payload["deployment_selection"] = (
            "deterministic fixed low/mid/high-density validation at episodes "
            "0, 4, 8, 12, and 16"
        )
        payload["cooperative_entry_safety"] = {
            "graph": "atomic team first-entry gate plus dynamic gap/TTC acceptance",
            "learned_baselines": "individual dynamic gap/TTC acceptance",
            "cacc_lc": "sequential individual dynamic gap/TTC acceptance",
        }
        payload["representative_trace_rule"] = (
            "predeclared maximum mixed-lane traffic intensity; ties use the "
            "lowest paired episode index; selection is independent of outcomes"
        )
        return payload

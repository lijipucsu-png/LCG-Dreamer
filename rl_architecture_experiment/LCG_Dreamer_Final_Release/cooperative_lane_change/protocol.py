"""Immutable protocol for focused transfer and matched lane-change tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = ("graph", "cacc", "ddpg_p", "ppo_p", "sac_p", "nc_kg_sac_p")
TRAINABLE_METHODS = METHODS
DISPLAY_NAMES = {
    "graph": "LCG-Dreamer",
    "cacc": "CACC",
    "ddpg_p": "Pred-DDPG",
    "ppo_p": "Pred-PPO",
    "sac_p": "Pred-SAC",
    "nc_kg_sac_p": "KG-SAC",
}


@dataclass(frozen=True)
class LaneChangeProtocol:
    protocol_version: int = 1
    seed: int = 46_200
    training_episodes: int = 8
    imitation_episodes: int = 1
    evaluation_episodes: int = 20
    max_time: float = 90.0
    initial_position_offset_m: float = 250.0
    obstacle_stations: tuple[float, float, float] = (680.0, 1010.0, 1320.0)
    obstacle_lanes: tuple[int, int, int] = (1, 2, 1)
    target_lanes: tuple[int, int, int] = (2, 1, 0)
    goal_station: float = 1500.0
    obstacle_speed_min_mps: float = 7.0
    obstacle_speed_max_mps: float = 10.0
    communication_delay_probability: float = 0.15
    learning_rate: float = 3.0e-4
    imitation_epochs: int = 20
    ppo_epochs: int = 4
    minibatch_size: int = 512
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.005
    max_grad_norm: float = 1.0
    graph_checkpoint: str = (
        "experiments/downstream_wave_refinement/selected/"
        "graph_wave_selected_validated_v3.pt"
    )
    checkpoint_root: str = "artifacts/predictive_comparison_v2/training"
    output_root: str = "artifacts/cooperative_lane_change"

    def base_checkpoint(self, method: str) -> Path | None:
        if method == "graph":
            return Path(self.graph_checkpoint)
        if method == "cacc":
            return None
        if method not in TRAINABLE_METHODS:
            raise ValueError(f"unknown lane-change method: {method}")
        return Path(self.checkpoint_root) / method / "complete.pt"

    def adapter_checkpoint(self, method: str) -> Path:
        return Path(self.output_root) / "training" / method / "best.pt"

    def scenario_options(
        self, episode_seed: int, training: bool
    ) -> dict[str, Any]:
        import numpy as np

        rng = np.random.default_rng(episode_seed)
        common_shift = float(rng.uniform(-12.0, 12.0)) if training else 0.0
        station_jitter = (
            rng.uniform(-8.0, 8.0, size=3) if training else np.zeros(3)
        )
        stations = tuple(
            float(base + common_shift + jitter)
            for base, jitter in zip(self.obstacle_stations, station_jitter)
        )
        obstacle_speed = float(
            rng.uniform(
                self.obstacle_speed_min_mps,
                self.obstacle_speed_max_mps,
            )
        )
        return {
            "cooperative_lane_change_enabled": True,
            "lane_change_obstacle_stations": stations,
            "lane_change_obstacle_lanes": self.obstacle_lanes,
            "lane_change_target_lanes": self.target_lanes,
            "lane_change_obstacle_speed_mps": obstacle_speed,
            "lane_change_goal_station": self.goal_station + common_shift,
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
                else self.communication_delay_probability
            ),
            "communication_delay_min": 0.1,
            "communication_delay_max": 0.2,
            "speed_reference": 25.0,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("obstacle_stations", "obstacle_lanes", "target_lanes"):
            payload[key] = list(payload[key])
        canonical = json.dumps(payload, sort_keys=True)
        payload["fingerprint"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        payload["methods"] = list(METHODS)
        payload["reward_is_method_invariant"] = True
        payload["base_controllers_frozen"] = True
        return payload

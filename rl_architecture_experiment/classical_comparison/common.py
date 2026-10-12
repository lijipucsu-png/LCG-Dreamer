"""Frozen observation, action, reward, and scenario protocol for all baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from platoon_control.graph_agents import GraphObservationAdapter
from platoon_control.models.delay import CPIDLMDelayCompensator
from platoon_control.graph_rl import (
    _shape_coordinated_restart_rewards,
    _shape_expanding_formation_rewards,
)
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv


STATE_KEYS = (
    "node_features",
    "road_profile",
    "v2i_flow",
    "front_entities",
    "front_mask",
    "neighbor_entities",
    "neighbor_mask",
    "role",
    "adjacency",
    "edge_ages",
    "physical_target",
    "follower_mask",
)

# Predictive baselines were trained before the optional downstream-wave term
# was added to ``_shared_reward``. That term is identically zero unless its new
# training option is enabled, so this reviewed historical digest is reward-
# equivalent in all comparison scenarios. Unknown implementations fail closed.
LEGACY_EQUIVALENT_REWARD_FINGERPRINTS = frozenset(
    {"40489a82ad010cb90b8937e7fd518b64e7880290b7a448976e4c0caab90c4c37"}
)


@dataclass(frozen=True)
class ComparisonProtocol:
    """Values that must remain identical across DDPG, PPO, and SAC."""

    protocol_version: int = 1
    training_seed: int = 43
    training_steps: int = 301_144
    evaluation_seed: int = 30_000
    evaluation_episodes: int = 20
    episode_max_time: float = 240.0
    action_min: float = -3.0
    action_max: float = 1.2
    discount: float = 0.99
    reward_scale: float = 0.1
    replay_capacity: int = 30_000
    batch_size: int = 256
    checkpoint_interval: int = 25_000
    # Median stage-transition steps from completed Graph seeds 42, 43, and 44.
    curriculum_boundaries: tuple[int, ...] = (
        0,
        36_793,
        57_488,
        70_955,
        83_490,
        141_935,
        190_678,
        223_117,
    )
    calibration_path: str = "artifacts/calibration/calibrated_parameters.json"
    wave_model_path: str = "artifacts/checkpoints/wave_best.pt"
    reward_predictor_path: str = "artifacts/checkpoints/cpidlm_best.pt"

    def stage(self, step: int) -> int:
        return max(
            index
            for index, boundary in enumerate(self.curriculum_boundaries)
            if step >= boundary
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_keys"] = list(STATE_KEYS)
        payload["team_reward_aggregation"] = "mean_of_six_agent_rewards"
        payload["reward_fingerprint"] = reward_fingerprint()
        payload["controller_specific_reward_tuning"] = False
        return payload


def reward_fingerprint() -> str:
    """Hash the exact reward and shared shaping source used by every method."""

    sources = (
        inspect.getsource(PlatoonParallelEnv._shared_reward),
        inspect.getsource(_shape_coordinated_restart_rewards),
        inspect.getsource(_shape_expanding_formation_rewards),
    )
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()


def reward_fingerprint_is_compatible(fingerprint: str) -> bool:
    """Accept only the current reward or a reviewed, semantically equal digest."""

    return fingerprint == reward_fingerprint() or (
        fingerprint in LEGACY_EQUIVALENT_REWARD_FINGERPRINTS
    )


class FrozenRewardPredictor:
    """The same frozen CPIDLM reward reference used during Graph training."""

    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model = CPIDLMDelayCompensator(**checkpoint["model_arguments"])
        self.model.load_state_dict(checkpoint["model"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def attach(
        self,
        env: PlatoonParallelEnv,
        observations: Mapping[str, Mapping[str, np.ndarray]],
    ) -> None:
        v2v = torch.stack(
            [
                torch.as_tensor(
                    observations[agent]["v2v_history"], dtype=torch.float32
                )
                for agent in FOLLOWER_IDS
            ]
        ).to(self.device)
        local = torch.stack(
            [
                torch.as_tensor(
                    observations[agent]["local_predecessor"],
                    dtype=torch.float32,
                )
                for agent in FOLLOWER_IDS
            ]
        ).to(self.device)
        future = self.model(v2v, local).future_mean
        predictions = {
            agent: (
                float(future[index, -1, 1].cpu()),
                float(future[index, :, 2].mean().cpu()),
            )
            for index, agent in enumerate(FOLLOWER_IDS)
        }
        env.set_follower_control_predictions(predictions)


class StateAdapter:
    """Flatten exactly the observable Graph inputs plus prior applied actions."""

    def __init__(self) -> None:
        self.adapter = GraphObservationAdapter(teachers=None)
        self.dimension: int | None = None

    def __call__(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        previous_applied: np.ndarray,
    ) -> np.ndarray:
        compact = self.adapter(observations)
        flattened = [
            np.asarray(compact[key], dtype=np.float32).reshape(-1)
            for key in STATE_KEYS
        ]
        flattened.append(np.asarray(previous_applied, np.float32).reshape(-1))
        state = np.concatenate(flattened).astype(np.float32, copy=False)
        if self.dimension is None:
            self.dimension = int(state.size)
        elif state.size != self.dimension:
            raise RuntimeError(
                f"state dimension changed from {self.dimension} to {state.size}"
            )
        if not np.isfinite(state).all():
            raise FloatingPointError("non-finite value in shared baseline state")
        return state


class RunningNormalizer:
    def __init__(self, dimension: int, clip: float = 10.0) -> None:
        self.mean = np.zeros(dimension, np.float64)
        self.variance = np.ones(dimension, np.float64)
        self.count = 1e-4
        self.clip = float(clip)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, np.float64).reshape(-1, self.mean.size)
        batch_mean = batch.mean(0)
        batch_variance = batch.var(0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        moment_a = self.variance * self.count
        moment_b = batch_variance * batch_count
        combined = (
            moment_a
            + moment_b
            + np.square(delta) * self.count * batch_count / total
        )
        self.variance = np.maximum(combined / total, 1e-6)
        self.count = total

    def normalize(self, values: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(values, np.float32) - self.mean) / np.sqrt(
            self.variance + 1e-8
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
            "clip": self.clip,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], np.float64)
        self.variance = np.asarray(state["variance"], np.float64)
        self.count = float(state["count"])
        self.clip = float(state["clip"])


class ActionCodec:
    """Zero-centered map between normalized and physical accelerations."""

    def __init__(self, minimum: float = -3.0, maximum: float = 1.2) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def to_physical(self, normalized: np.ndarray) -> np.ndarray:
        unit = np.clip(np.asarray(normalized, np.float32), -1.0, 1.0)
        return np.where(unit >= 0.0, unit * self.maximum, unit * -self.minimum)

    def to_normalized(self, physical: np.ndarray) -> np.ndarray:
        action = np.asarray(physical, np.float32)
        return np.where(
            action >= 0.0,
            action / self.maximum,
            action / -self.minimum,
        ).clip(-1.0, 1.0)

    def action_dict(self, normalized: np.ndarray) -> dict[str, dict[str, Any]]:
        physical = self.to_physical(normalized)
        return {
            agent: {
                "accel": np.asarray([physical[index]], np.float32),
                "lane_change": 0,
            }
            for index, agent in enumerate(PLATOON_IDS)
        }


def shared_step_rewards(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: Mapping[str, Mapping[str, Any]],
    environment_rewards: Mapping[str, float],
    infos: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and Graph-equivalent shaped six-agent reward vectors."""

    shaped = {agent: float(environment_rewards[agent]) for agent in PLATOON_IDS}
    _shape_coordinated_restart_rewards(observations, actions, shaped, infos)
    _shape_expanding_formation_rewards(observations, actions, shaped)
    raw_vector = np.asarray(
        [environment_rewards[agent] for agent in PLATOON_IDS], np.float32
    )
    shaped_vector = np.asarray([shaped[agent] for agent in PLATOON_IDS], np.float32)
    return raw_vector, shaped_vector

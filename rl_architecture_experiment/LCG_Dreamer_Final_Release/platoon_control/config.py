"""Typed configuration shared by training, inference, and checkpoint metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class ModelConfig:
    history_steps: int = 50
    neighbor_history_steps: int = 20
    prediction_horizon: int = 30
    max_front_vehicles: int = 8
    max_neighbors: int = 12
    trajectory_modes: int = 3
    hidden_dim: int = 128
    stochastic_dim: int = 16
    stochastic_classes: int = 16
    ensemble_size: int = 5


@dataclass
class PlannerConfig:
    horizon: int = 20
    population: int = 512
    elites: int = 64
    iterations: int = 5
    momentum: float = 0.15
    min_std: float = 0.08
    accel_min: float = -3.0
    accel_max: float = 1.2


@dataclass
class FollowerRLConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    target_tau: float = 0.005
    entropy_coefficient: float = 0.05
    reward_scale: float = 0.1
    conservative_weight: float = 0.5
    behavior_cloning_weight: float = 10.0
    behavior_cloning_final_weight: float = 0.1
    behavior_cloning_decay_steps: int = 50_000
    residual_accel_scale: float = 1.5
    accel_min: float = -3.0
    accel_max: float = 1.2


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 100.0
    device: str = "auto"
    mixed_precision: bool = True
    checkpoint_dir: str = "artifacts/checkpoints"
    log_dir: str = "artifacts/logs"


@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    follower_rl: FollowerRLConfig = field(default_factory=FollowerRLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SystemConfig":
        def compatible(config_type: type, section: str) -> dict[str, Any]:
            allowed = {item.name for item in fields(config_type)}
            return {
                key: value
                for key, value in values.get(section, {}).items()
                if key in allowed
            }

        return cls(
            model=ModelConfig(**compatible(ModelConfig, "model")),
            follower_rl=FollowerRLConfig(
                **compatible(FollowerRLConfig, "follower_rl")
            ),
            training=TrainingConfig(**compatible(TrainingConfig, "training")),
        )

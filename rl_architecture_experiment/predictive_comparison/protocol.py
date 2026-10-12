"""Immutable protocol for prediction-equated controller comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from classical_comparison.common import ComparisonProtocol, reward_fingerprint


PREDICTIVE_METHODS = ("ddpg_p", "ppo_p", "sac_p", "nc_kg_sac_p")
ALL_METHODS = ("graph", "cacc", *PREDICTIVE_METHODS)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictiveComparisonProtocol(ComparisonProtocol):
    """One reward, one predictor, one scenario schedule, and one test split."""

    protocol_version: int = 2
    training_steps: int = 301_144
    checkpoint_interval: int = 10_000
    training_stage: int = 7
    total_interaction_budget: int = 301_144
    graph_checkpoint: str = "artifacts/graph_full_seed43/graph_complete.pt"
    predictor_directory: str = "artifacts/checkpoints"
    prediction_schema: str = "graph_rssm_stgnn_top_cutin_v1"
    nc_wave_speed_mps: float = 5.0
    nc_local_time_headway_s: float = 0.6
    nc_local_standstill_gap_m: float = 5.0
    nc_fusion_weight: float = 0.5
    nc_action_history_steps: int = 3
    source_ddpg: str = (
        "artifacts/classical_comparison/assisted/formal_v4/ddpg/"
        "complete_early_stopped.pt"
    )
    source_ppo: str = (
        "artifacts/classical_comparison/assisted/formal_v5/ppo/"
        "complete_early_stopped.pt"
    )
    source_sac: str = (
        "artifacts/classical_comparison/assisted/formal_v6/sac/"
        "stage7_candidate_4of5.pt"
    )

    def stage(self, step: int) -> int:
        del step
        return self.training_stage

    def source_checkpoint(self, method: str) -> Path:
        sources = {
            "ddpg_p": self.source_ddpg,
            "ppo_p": self.source_ppo,
            "sac_p": self.source_sac,
            "nc_kg_sac_p": self.source_sac,
        }
        try:
            return Path(sources[method])
        except KeyError as error:
            raise ValueError(f"unknown predictive method: {method}") from error

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["curriculum_boundaries"] = list(self.curriculum_boundaries)
        payload.update(
            {
                "methods": list(ALL_METHODS),
                "prediction_enabled_methods": list(PREDICTIVE_METHODS),
                "cacc_prediction_module": None,
                "team_reward_aggregation": "mean_of_six_agent_rewards",
                "reward_fingerprint": reward_fingerprint(),
                "controller_specific_reward_tuning": False,
                "training_scenario_stage": self.training_stage,
                "future_ground_truth_used_by_policy": False,
                "predictor_frozen_during_policy_training": True,
                "prediction_checkpoint_sha256": file_sha256(self.graph_checkpoint),
                "trajectory_checkpoint_sha256": file_sha256(
                    Path(self.predictor_directory) / "trajectory_best.pt"
                ),
            }
        )
        canonical = json.dumps(payload, sort_keys=True, default=str)
        payload["protocol_fingerprint"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return payload

"""Paired evaluation for Graph, CACC, and prediction-equated baselines."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from classical_comparison.common import (
    ActionCodec,
    FrozenRewardPredictor,
    reward_fingerprint_is_compatible,
)
from classical_comparison.evaluate import (
    CaccPolicy,
    GraphPolicy,
    _episode,
    write_evaluation_rows,
)
from platoon_sumo.env import PLATOON_IDS

from .algorithms import create_predictive_algorithm
from .prediction import SharedPredictiveEncoder
from .protocol import ALL_METHODS, PREDICTIVE_METHODS, PredictiveComparisonProtocol
from .state import NCKnowledgeStateAdapter, PredictiveStateAdapter


def _protocol_from_checkpoint(
    checkpoint: dict[str, Any],
) -> PredictiveComparisonProtocol:
    allowed = {field.name for field in fields(PredictiveComparisonProtocol)}
    values = {
        key: value for key, value in checkpoint["protocol"].items() if key in allowed
    }
    if "curriculum_boundaries" in values:
        values["curriculum_boundaries"] = tuple(values["curriculum_boundaries"])
    return PredictiveComparisonProtocol(**values)


class PredictivePolicy:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.name = str(checkpoint["algorithm"])
        if self.name not in PREDICTIVE_METHODS:
            raise ValueError(f"not a predictive comparison checkpoint: {self.name}")
        if not reward_fingerprint_is_compatible(checkpoint["reward_fingerprint"]):
            raise ValueError("predictive checkpoint reward fingerprint is stale")
        self.protocol = _protocol_from_checkpoint(checkpoint)
        predictor = SharedPredictiveEncoder(
            self.protocol.graph_checkpoint,
            self.protocol.predictor_directory,
            device,
        )
        self.adapter: PredictiveStateAdapter
        if self.name == "nc_kg_sac_p":
            self.adapter = NCKnowledgeStateAdapter(predictor, self.protocol)
        else:
            self.adapter = PredictiveStateAdapter(predictor)
        self.codec = ActionCodec(
            self.protocol.action_min, self.protocol.action_max
        )
        self.previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
        state_dim = int(checkpoint["state_dim"])
        self.algorithm = create_predictive_algorithm(
            self.name,
            state_dim,
            self.protocol,
            device,
            self.protocol.training_seed,
        )
        self.algorithm.load_state_dict(checkpoint["algorithm_state"])
        self.total_steps = int(checkpoint["total_steps"])
        self._last_actions: dict[str, dict[str, Any]] | None = None

    def reset(self) -> None:
        self.previous_applied.fill(0.0)
        self.adapter.reset()
        self._last_actions = None

    def act(self, observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
        state = self.adapter(observations, self.previous_applied)
        sample = self.algorithm.act(
            state, deterministic=True, total_steps=self.total_steps
        )
        self._last_actions = self.codec.action_dict(sample.action)
        return self._last_actions

    def record_applied(self, infos: dict[str, dict[str, Any]]) -> None:
        if self._last_actions is None:
            raise RuntimeError("record_applied called before act")
        self.adapter.record(infos, self._last_actions)
        self.previous_applied = np.asarray(
            [
                infos[agent].get("applied_acceleration", 0.0)
                for agent in PLATOON_IDS
            ],
            np.float32,
        )


def evaluate(
    controller: str,
    checkpoint: Path,
    output: Path,
    protocol: PredictiveComparisonProtocol,
    device: str,
    episode_start: int = 0,
) -> Path:
    if controller == "graph":
        policy: Any = GraphPolicy(checkpoint, device)
    elif controller == "cacc":
        policy = CaccPolicy()
    else:
        policy = PredictivePolicy(checkpoint, device)
        protocol = replace(
            policy.protocol,
            evaluation_episodes=protocol.evaluation_episodes,
            evaluation_seed=protocol.evaluation_seed,
            training_steps=policy.protocol.total_interaction_budget,
        )
    if policy.name != controller:
        raise ValueError(f"loaded {policy.name}, expected {controller}")
    reward_predictor = FrozenRewardPredictor(
        protocol.reward_predictor_path, device
    )
    rows = []
    for episode_index in range(
        episode_start, episode_start + protocol.evaluation_episodes
    ):
        row = _episode(policy, protocol, episode_index, reward_predictor)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return write_evaluation_rows(
        controller, checkpoint, output, protocol, rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller", choices=ALL_METHODS)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=30_000)
    arguments = parser.parse_args()
    protocol = replace(
        PredictiveComparisonProtocol(),
        evaluation_episodes=arguments.episodes,
        evaluation_seed=arguments.evaluation_seed,
    )
    path = evaluate(
        arguments.controller,
        arguments.checkpoint,
        arguments.output,
        protocol,
        arguments.device,
        arguments.episode_start,
    )
    print(json.dumps({"summary": str(path), "complete": True}))


if __name__ == "__main__":
    main()

"""Validate a completed graph controller on the full Stage-8 scenario set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from platoon_control.graph_rl import GraphRLConfig, load_graph_controller
from platoon_control.rl import evaluate_controller


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    checkpoint = torch.load(
        arguments.checkpoint, map_location="cpu", weights_only=False
    )
    config = GraphRLConfig(**checkpoint["graph_rl_config"])
    evaluation = config.evaluation_config()
    evaluation.evaluation_episodes = arguments.episodes
    controller = load_graph_controller(arguments.checkpoint, arguments.device)

    def progress(metrics: dict[str, float]) -> None:
        print(json.dumps(metrics, sort_keys=True), flush=True)

    metrics = evaluate_controller(
        controller,
        evaluation,
        int(checkpoint["total_steps"]),
        curriculum_stage=7,
        progress_callback=progress,
    )
    passed = bool(
        metrics["evaluation/capability_pass"] >= 1.0
        and metrics["evaluation/collision_rate"] == 0.0
        and metrics["evaluation/failure_rate"] == 0.0
        and metrics["evaluation/success_rate"] >= 0.8
        and metrics["evaluation/background_exposure"] >= 1.0
        and metrics["evaluation/cutin_exposure"] >= 1.0
        and metrics["evaluation/delay_exposure"] >= 1.0
    )
    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": _sha256(arguments.checkpoint),
        "seed": config.seed,
        "training_steps": int(checkpoint["total_steps"]),
        "training_episodes": int(checkpoint["episodes"]),
        "curriculum_stage": 7,
        "evaluation_episodes": arguments.episodes,
        "scenario_features": [
            "background_traffic",
            "learned_leader_oscillation",
            "communication_delay",
            "cut_in",
            "v2i_curve_profile",
        ],
        "metrics": metrics,
        "passed": passed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Recover durable evaluation rows from interrupted stdout journals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from platoon_control.evaluation import (
    EvaluationConfig,
    VARIANTS,
    seed_evaluation_resume,
)


def _records(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and {
                "variant",
                "seed",
                "front_vehicle_count",
            }.issubset(record):
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    parser.add_argument("--total-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        default="artifacts/calibration/calibrated_parameters.json",
    )
    parser.add_argument(
        "--wave-model", default="artifacts/checkpoints/wave_best.pt"
    )
    arguments = parser.parse_args()
    config = EvaluationConfig(
        seed=arguments.seed,
        total_episodes=arguments.total_episodes,
        max_steps=arguments.max_steps,
        output_dir=str(arguments.output),
        calibration_path=arguments.calibration,
        wave_model_path=arguments.wave_model,
        device="cuda",
    )
    recovered = seed_evaluation_resume(
        arguments.checkpoint,
        config,
        tuple(arguments.variants),
        _records(arguments.log),
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "recovered_rows": recovered,
                "expected_rows": arguments.total_episodes
                * len(arguments.variants),
            }
        )
    )


if __name__ == "__main__":
    main()

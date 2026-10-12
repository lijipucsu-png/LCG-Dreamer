"""Merge deterministic evaluation shards after strict scenario validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .common import ComparisonProtocol
from .evaluate import _scenario, write_evaluation_rows


def _log_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "controller" in payload and "episode_index" in payload:
            rows.append(payload)
    return rows


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted.append(
            {
                key: value
                if key in ("controller", "scenario_hash")
                else int(float(value))
                if key
                in (
                    "episode_index",
                    "reset_seed",
                    "steps",
                    "success",
                    "collision",
                    "failure",
                    "background_peak_count",
                    "cutin_exposed",
                    "cutin_resolved",
                )
                else float(value)
                for key, value in row.items()
            }
        )
    return converted


def merge_shards(
    controller: str,
    checkpoint: Path,
    logs: list[Path],
    csv_shards: list[Path],
    output: Path,
    episodes: int,
    evaluation_seed: int,
) -> Path:
    protocol = ComparisonProtocol(
        evaluation_episodes=episodes,
        evaluation_seed=evaluation_seed,
    )
    by_episode: dict[int, dict[str, Any]] = {}
    source_rows = [row for path in logs for row in _log_rows(path)]
    source_rows.extend(
        row for path in csv_shards for row in _csv_rows(path)
    )
    for row in source_rows:
        index = int(row["episode_index"])
        if index in by_episode and row != by_episode[index]:
            raise ValueError(f"conflicting duplicate episode {index}")
        by_episode[index] = row
    expected = set(range(episodes))
    if set(by_episode) != expected:
        missing = sorted(expected - set(by_episode))
        unexpected = sorted(set(by_episode) - expected)
        raise ValueError(
            f"incomplete shards: missing={missing}, unexpected={unexpected}"
        )
    rows = [by_episode[index] for index in range(episodes)]
    for index, row in enumerate(rows):
        reset_seed, _, scenario_hash = _scenario(protocol, index)
        if row["controller"] != controller:
            raise ValueError(f"episode {index} controller mismatch")
        if int(row["reset_seed"]) != reset_seed:
            raise ValueError(f"episode {index} reset seed mismatch")
        if row["scenario_hash"] != scenario_hash:
            raise ValueError(f"episode {index} scenario hash mismatch")
    return write_evaluation_rows(
        controller, checkpoint, output, protocol, rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller", choices=("graph", "cacc", "ddpg", "ppo", "sac"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--csv", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--evaluation-seed", type=int, default=30_000)
    arguments = parser.parse_args()
    summary = merge_shards(
        arguments.controller,
        arguments.checkpoint,
        arguments.log,
        arguments.csv,
        arguments.output,
        arguments.episodes,
        arguments.evaluation_seed,
    )
    print(json.dumps({"summary": str(summary), "complete": True}))


if __name__ == "__main__":
    main()

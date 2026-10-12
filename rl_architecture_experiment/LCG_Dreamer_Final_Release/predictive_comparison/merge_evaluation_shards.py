"""Merge deterministic evaluation shards into one validated 20-episode result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from classical_comparison.evaluate import _scenario, write_evaluation_rows

from .evaluate import _protocol_from_checkpoint


def _log_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "episode_index" in payload:
            rows.append(payload)
    return rows


def _csv_rows(path: Path, template: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in source.items():
                expected = template[key]
                if isinstance(expected, int):
                    row[key] = int(float(value))
                elif isinstance(expected, float):
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def merge(
    controller: str,
    checkpoint_path: Path,
    initial_log: Path,
    shard_directories: list[Path],
    output: Path,
) -> Path:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    protocol = _protocol_from_checkpoint(checkpoint)
    rows = _log_rows(initial_log)
    if not rows:
        raise RuntimeError(f"no episode rows in {initial_log}")
    template = rows[0]
    for directory in shard_directories:
        rows.extend(_csv_rows(directory / "episodes.csv", template))
    rows.sort(key=lambda item: int(item["episode_index"]))
    indices = [int(item["episode_index"]) for item in rows]
    if indices != list(range(20)):
        raise RuntimeError(f"expected episode indices 0--19, found {indices}")
    for row in rows:
        episode_index = int(row["episode_index"])
        reset_seed, _, scenario_hash = _scenario(protocol, episode_index)
        if int(row["reset_seed"]) != reset_seed:
            raise RuntimeError(f"reset seed mismatch at episode {episode_index}")
        if row["scenario_hash"] != scenario_hash:
            raise RuntimeError(f"scenario hash mismatch at episode {episode_index}")
        if row["controller"] != controller:
            raise RuntimeError(f"controller mismatch at episode {episode_index}")
    return write_evaluation_rows(
        controller, checkpoint_path, output, protocol, rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--initial-log", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = merge(
        arguments.controller,
        arguments.checkpoint,
        arguments.initial_log,
        arguments.shard,
        arguments.output,
    )
    print(json.dumps({"summary": str(summary.resolve()), "complete": True}))


if __name__ == "__main__":
    main()

"""Paired Stage-7 evaluation for all completed budgeted ablation models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from classical_comparison.common import ComparisonProtocol, FrozenRewardPredictor
from classical_comparison.evaluate import GraphPolicy, _episode

from .protocol import PROTOCOL, ROOT, VARIANTS, variant_by_key


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run(
    output_root: Path,
    device: str,
    variant_keys: tuple[str, ...] | None = None,
    training_seeds: tuple[int, ...] | None = None,
) -> Path:
    protocol = ComparisonProtocol(
        evaluation_seed=PROTOCOL.evaluation_seed,
        evaluation_episodes=PROTOCOL.evaluation_cases_per_training_seed,
        training_steps=301_144,
    )
    predictor = FrozenRewardPredictor(protocol.reward_predictor_path, "cpu")
    selected_variants = (
        tuple(variant_by_key(key) for key in variant_keys)
        if variant_keys is not None
        else VARIANTS
    )
    selected_seeds = training_seeds or PROTOCOL.training_seeds
    combined_path = output_root / "evaluation_episodes.csv"
    combined: list[dict[str, Any]] = []
    last_path = combined_path
    for seed in selected_seeds:
        for variant in selected_variants:
            run_root = output_root / variant.key / f"seed_{seed}"
            checkpoint = run_root / "budgeted_complete.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            path = run_root / "evaluation_episodes.csv"
            last_path = path
            rows = read_rows(path)
            completed = {int(row["episode_index"]) for row in rows}
            policy = GraphPolicy(checkpoint, device)
            for episode_index in range(PROTOCOL.evaluation_cases_per_training_seed):
                if episode_index in completed:
                    continue
                row = _episode(policy, protocol, episode_index, predictor)
                row.update(
                    variant=variant.key,
                    variant_label=variant.label,
                    training_seed=seed,
                )
                rows.append(row)
                rows.sort(key=lambda item: int(item["episode_index"]))
                write_rows(path, rows)
                print(json.dumps(row, sort_keys=True), flush=True)
            if len(rows) != PROTOCOL.evaluation_cases_per_training_seed:
                raise RuntimeError(f"incomplete paired evaluation: {path}")
            combined.extend(rows)
    if variant_keys is None and training_seeds is None:
        write_rows(combined_path, combined)
        return combined_path
    return last_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=[item.key for item in VARIANTS])
    parser.add_argument("--training-seed", type=int)
    arguments = parser.parse_args()
    print(
        run(
            arguments.output_root,
            arguments.device,
            (arguments.variant,) if arguments.variant else None,
            (arguments.training_seed,) if arguments.training_seed is not None else None,
        )
    )


if __name__ == "__main__":
    main()

"""Run one provenance-checked evaluation suite from the graph study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.run_graph_study import EXPERIMENT_ROOT, evaluate_one


def _experiment_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else EXPERIMENT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--total-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30_000)
    arguments = parser.parse_args()
    result = evaluate_one(
        sys.executable,
        _experiment_path(arguments.checkpoint),
        _experiment_path(arguments.output),
        tuple(arguments.variants),
        arguments.total_episodes,
        arguments.seed,
    )
    print(result)


if __name__ == "__main__":
    main()

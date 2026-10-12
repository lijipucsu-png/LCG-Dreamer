"""Parallel, resumable evaluation launcher and merger."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

from .protocol import PROTOCOL, ROOT, VARIANTS


def evaluate_one(output_root: Path, variant: str, seed: int, device: str) -> Path:
    run_root = output_root / variant / f"seed_{seed}"
    log = run_root / "evaluation_stdout.log"
    command = [
        sys.executable,
        "-m",
        "dreamerv3_ablation.evaluate",
        "--output-root",
        str(output_root),
        "--device",
        device,
        "--variant",
        variant,
        "--training-seed",
        str(seed),
    ]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if result.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"evaluation failed for {variant}/seed_{seed}:\n{tail}")
    return run_root / "evaluation_episodes.csv"


def merge(paths: list[Path], output: Path) -> Path:
    rows = []
    fieldnames = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = fieldnames or reader.fieldnames
            rows.extend(reader)
    expected = (
        len(VARIANTS)
        * len(PROTOCOL.training_seeds)
        * PROTOCOL.evaluation_cases_per_training_seed
    )
    if len(rows) != expected:
        raise RuntimeError(f"evaluation row count {len(rows)} != {expected}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return output


def run(output_root: Path, device: str, workers: int) -> Path:
    jobs = [(item.key, seed) for seed in PROTOCOL.training_seeds for item in VARIANTS]
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_one, output_root, variant, seed, device): (variant, seed)
            for variant, seed in jobs
        }
        for future in as_completed(futures):
            paths.append(future.result())
    return merge(paths, output_root / "evaluation_episodes.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v3",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=3)
    arguments = parser.parse_args()
    print(run(arguments.output_root, arguments.device, arguments.workers))


if __name__ == "__main__":
    main()

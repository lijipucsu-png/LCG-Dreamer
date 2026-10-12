"""Promote the validated aligned graph foundation into curriculum Stage 1."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def promote(
    current_path: Path,
    source_path: Path,
    validation_path: Path,
    backup_directory: Path,
) -> dict[str, Any]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passed") or not all(validation.get("gates", {}).values()):
        raise RuntimeError(f"foundation validation did not pass: {validation_path}")
    current = torch.load(current_path, map_location="cpu", weights_only=False)
    source = torch.load(source_path, map_location="cpu", weights_only=False)

    backup_directory.mkdir(parents=True, exist_ok=True)
    backup = backup_directory / f"{current_path.parent.name}_{current_path.name}"
    shutil.copy2(current_path, backup)
    current["models"] = source["models"]
    for optimizer in current["trainer"]["optimizers"].values():
        optimizer["state"] = {}
    current["trainer"]["actor_updates"] = 5_000
    current["curriculum_stage"] = 1
    current["stage_steps"] = 0
    current["capability_records"] = []
    current["foundation_passed"] = True
    current["validated_foundation_promotion"] = {
        "source": str(source_path),
        "validation": str(validation_path),
        "gates": validation["gates"],
        "model_alignment_restored": True,
        "replay_preserved": True,
        "total_steps_preserved": int(current["total_steps"]),
    }
    torch.save(current, current_path)
    return {
        "checkpoint": str(current_path),
        "backup": str(backup),
        **current["validated_foundation_promotion"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--backup-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    arguments = parser.parse_args()
    report = []
    for seed in arguments.seeds:
        report.append(
            promote(
                arguments.current_root / f"graph_full_seed{seed}" / "graph_latest.pt",
                arguments.source_root / f"graph_full_seed{seed}" / "graph_latest.pt",
                arguments.current_root
                / f"graph_full_seed{seed}"
                / "foundation_warmstart_v14.json",
                arguments.backup_directory,
            )
        )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

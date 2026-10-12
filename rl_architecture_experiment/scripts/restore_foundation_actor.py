"""Restore a validated foundation actor while preserving the learned world model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


POLICY_PREFIXES = ("actor.", "value_head.")


def restore(current_path: Path, source_path: Path, backup_directory: Path) -> dict[str, Any]:
    current = torch.load(current_path, map_location="cpu", weights_only=False)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    current_model = current["models"]["graph_dreamer"]
    source_model = source["models"]["graph_dreamer"]
    policy_keys = [
        key for key in current_model if key.startswith(POLICY_PREFIXES)
    ]
    if not policy_keys or any(key not in source_model for key in policy_keys):
        raise RuntimeError(f"incompatible foundation policy checkpoint: {source_path}")

    backup_directory.mkdir(parents=True, exist_ok=True)
    backup = backup_directory / f"{current_path.parent.name}_{current_path.name}"
    shutil.copy2(current_path, backup)
    for key in policy_keys:
        current_model[key] = source_model[key]
    for optimizer_name in ("actor", "critic"):
        current["trainer"]["optimizers"][optimizer_name]["state"] = {}
    current["trainer"]["actor_updates"] = 0
    current["foundation_passed"] = False
    current["foundation_actor_restore"] = {
        "source": str(source_path),
        "policy_key_count": len(policy_keys),
        "world_model_preserved": True,
        "optimizer_states_reset": ["actor", "critic"],
    }
    torch.save(current, current_path)
    return {
        "checkpoint": str(current_path),
        "backup": str(backup),
        **current["foundation_actor_restore"],
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
            restore(
                arguments.current_root / f"graph_full_seed{seed}" / "graph_latest.pt",
                arguments.source_root / f"graph_full_seed{seed}" / "graph_latest.pt",
                arguments.backup_directory,
            )
        )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Remove legacy Stage-0 A3 episodes collected with conflicting cruise intent."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def legacy_a3_episode(episode: dict[str, Any]) -> bool:
    requested = np.asarray(episode["requested_actions"], np.float32)
    applied = np.asarray(episode["applied_actions"], np.float32)
    behavior_valid = np.asarray(episode.get("behavior_valid", []), np.float32)
    return bool(
        len(requested) >= 1_550
        and float(np.mean(np.abs(requested - applied))) > 0.15
        and behavior_valid.size > 0
        and float(behavior_valid.mean()) > 0.99
    )


def sanitize(path: Path, backup_directory: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    episodes = list(payload["episodes"])
    removed = [episode for episode in episodes if legacy_a3_episode(episode)]
    retained = [episode for episode in episodes if not legacy_a3_episode(episode)]
    if len(removed) != 2:
        raise RuntimeError(
            f"expected exactly two legacy A3 episodes in {path}, found {len(removed)}"
        )

    backup_directory.mkdir(parents=True, exist_ok=True)
    backup = backup_directory / f"{path.parent.name}_{path.name}"
    shutil.copy2(path, backup)
    payload["episodes"] = retained
    payload["steps"] = sum(len(episode["requested_actions"]) for episode in retained)
    torch.save(payload, path)
    return {
        "path": str(path),
        "backup": str(backup),
        "removed_episode_lengths": [
            len(episode["requested_actions"]) for episode in removed
        ],
        "retained_episodes": len(retained),
        "retained_steps": int(payload["steps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--backup-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = [sanitize(path, arguments.backup_directory) for path in arguments.paths]
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

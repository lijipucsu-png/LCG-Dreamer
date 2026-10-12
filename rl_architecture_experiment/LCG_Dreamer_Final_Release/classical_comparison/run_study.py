"""Run and monitor the complete classical-baseline comparison study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ALGORITHMS = ("ddpg", "ppo", "sac")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Child:
    def __init__(self, name: str, command: list[str], log_root: Path) -> None:
        self.name = name
        log_root.mkdir(parents=True, exist_ok=True)
        self.stdout_path = log_root / f"{name}.stdout.log"
        self.stderr_path = log_root / f"{name}.stderr.log"
        self.stdout = self.stdout_path.open("a", encoding="utf-8")
        self.stderr = self.stderr_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdout=self.stdout,
            stderr=self.stderr,
            cwd=Path(__file__).resolve().parents[1],
        )

    def close(self) -> None:
        self.stdout.close()
        self.stderr.close()


def _monitor_training(root: Path, children: dict[str, Child]) -> None:
    snapshot = {
        name: _read_json(root / "training" / name / "monitor.json")
        for name in ALGORITHMS
    }
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "training",
        "controllers": snapshot,
    }
    _write_json(root / "study_monitor.json", payload)
    concise = {
        name: {
            "status": record.get("status", "starting"),
            "step": record.get("total_steps", 0),
            "stage": record.get("curriculum_stage", 0),
            "gap": [record.get("minimum_gap_m"), record.get("maximum_gap_m")],
            "speed_spread": record.get("maximum_speed_spread_mps"),
            "collision": record.get("collision"),
            "background_stuck_s": record.get("background_max_stuck_seconds"),
        }
        for name, record in snapshot.items()
    }
    print(json.dumps({"phase": "training", "status": concise}), flush=True)


def _train(root: Path, steps: int, device: str, poll_seconds: int) -> None:
    children: dict[str, Child] = {}
    attempts = {name: 0 for name in ALGORITHMS}
    log_root = root / "logs"

    def launch(name: str) -> None:
        output = root / "training" / name
        if (output / "complete.pt").exists():
            return
        attempts[name] += 1
        command = [
            sys.executable,
            "-m",
            "classical_comparison.train",
            name,
            "--output",
            str(output),
            "--steps",
            str(steps),
            "--device",
            device,
        ]
        latest = output / "latest.pt"
        if latest.exists():
            command.extend(("--resume", str(latest)))
        children[name] = Child(
            f"train_{name}_attempt{attempts[name]}", command, log_root
        )

    for algorithm in ALGORITHMS:
        launch(algorithm)
    while children:
        time.sleep(poll_seconds)
        _monitor_training(root, children)
        for name, child in list(children.items()):
            return_code = child.process.poll()
            if return_code is None:
                continue
            child.close()
            del children[name]
            complete = root / "training" / name / "complete.pt"
            if return_code == 0 and complete.exists():
                continue
            if attempts[name] >= 3:
                raise RuntimeError(
                    f"{name} training failed after {attempts[name]} attempts; "
                    f"see {child.stderr_path}"
                )
            launch(name)


def _evaluate(
    root: Path,
    graph_checkpoint: Path,
    episodes: int,
    device: str,
    poll_seconds: int,
) -> None:
    checkpoints = {
        "graph": graph_checkpoint,
        "cacc": Path("unused-cacc-checkpoint"),
        **{
            name: root / "training" / name / "complete.pt"
            for name in ALGORITHMS
        },
    }
    children: dict[str, Child] = {}
    for controller, checkpoint in checkpoints.items():
        output = root / "evaluation" / controller
        if (output / "summary.json").exists():
            continue
        command = [
            sys.executable,
            "-m",
            "classical_comparison.evaluate",
            controller,
            str(checkpoint),
            "--output",
            str(output),
            "--episodes",
            str(episodes),
            "--device",
            device,
        ]
        children[controller] = Child(
            f"evaluate_{controller}", command, root / "logs"
        )
    while children:
        time.sleep(poll_seconds)
        status: dict[str, Any] = {}
        for name, child in children.items():
            stdout = child.stdout_path
            completed_rows = 0
            if stdout.exists():
                completed_rows = sum(
                    1
                    for line in stdout.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if line.startswith("{") and '"controller"' in line
                )
            status[name] = {
                "episodes_complete": completed_rows,
                "running": child.process.poll() is None,
            }
        _write_json(
            root / "study_monitor.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "phase": "evaluation",
                "controllers": status,
            },
        )
        print(json.dumps({"phase": "evaluation", "status": status}), flush=True)
        for name, child in list(children.items()):
            return_code = child.process.poll()
            if return_code is None:
                continue
            child.close()
            del children[name]
            if return_code != 0:
                raise RuntimeError(
                    f"{name} evaluation failed; see {child.stderr_path}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/classical_comparison")
    )
    parser.add_argument(
        "--graph-checkpoint",
        type=Path,
        default=Path("artifacts/graph_full_seed43/graph_complete.pt"),
    )
    parser.add_argument("--steps", type=int, default=301_144)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=60)
    arguments = parser.parse_args()
    arguments.root.mkdir(parents=True, exist_ok=True)
    _write_json(
        arguments.root / "study_manifest.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "training_steps": arguments.steps,
            "evaluation_episodes": arguments.episodes,
            "graph_checkpoint": str(arguments.graph_checkpoint.resolve()),
            "algorithms": list(ALGORITHMS),
            "reward_tuning": False,
        },
    )
    _train(arguments.root, arguments.steps, arguments.device, arguments.poll_seconds)
    _evaluate(
        arguments.root,
        arguments.graph_checkpoint,
        arguments.episodes,
        arguments.device,
        arguments.poll_seconds,
    )
    report_path = arguments.root / "comparison_report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "classical_comparison.report",
            "--root",
            str(arguments.root / "evaluation"),
            "--output",
            str(report_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    manifest = _read_json(arguments.root / "study_manifest.json")
    manifest.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "report": str(report_path.resolve()),
        }
    )
    _write_json(arguments.root / "study_manifest.json", manifest)
    _write_json(
        arguments.root / "study_monitor.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": "complete",
            "report": str(report_path.resolve()),
        },
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

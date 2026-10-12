"""Run, monitor, resume, evaluate, and report the predictive comparison."""

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

from .protocol import ALL_METHODS, PREDICTIVE_METHODS, PredictiveComparisonProtocol


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


def _snapshot(root: Path, phase: str) -> dict[str, Any]:
    controllers = {
        name: _read_json(root / "training" / name / "monitor.json")
        for name in PREDICTIVE_METHODS
    }
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "controllers": controllers,
        "safety_watch": {
            name: {
                "step": item.get("total_steps", 0),
                "stage": item.get("curriculum_stage", 7),
                "gap_m": [item.get("minimum_gap_m"), item.get("maximum_gap_m")],
                "speed_spread_mps": item.get("maximum_speed_spread_mps"),
                "collision": item.get("collision"),
                "failure": item.get("failure"),
                "background_stuck_s": item.get("background_max_stuck_seconds"),
                "background_gridlock_s": item.get(
                    "background_max_gridlock_seconds"
                ),
                "prediction_weight_rms": item.get(
                    "prediction_input_weight_rms"
                ),
            }
            for name, item in controllers.items()
        },
    }


def _train(root: Path, total_budget: int, device: str, poll_seconds: int) -> None:
    children: dict[str, Child] = {}
    attempts = {name: 0 for name in PREDICTIVE_METHODS}

    def launch(name: str) -> None:
        output = root / "training" / name
        if (output / "complete.pt").exists():
            return
        attempts[name] += 1
        command = [
            sys.executable,
            "-m",
            "predictive_comparison.train",
            name,
            "--output",
            str(output),
            "--total-budget",
            str(total_budget),
            "--device",
            device,
        ]
        latest = output / "latest.pt"
        if latest.exists():
            command.extend(("--resume", str(latest)))
        children[name] = Child(
            f"train_{name}_attempt{attempts[name]}", command, root / "logs"
        )

    for method in PREDICTIVE_METHODS:
        launch(method)
    while children:
        time.sleep(poll_seconds)
        snapshot = _snapshot(root, "training")
        _write_json(root / "study_monitor.json", snapshot)
        print(json.dumps(snapshot["safety_watch"], sort_keys=True), flush=True)
        for name, child in list(children.items()):
            code = child.process.poll()
            if code is None:
                continue
            child.close()
            del children[name]
            complete = root / "training" / name / "complete.pt"
            if code == 0 and complete.exists():
                continue
            if attempts[name] >= 3:
                raise RuntimeError(
                    f"{name} failed after {attempts[name]} attempts; "
                    f"see {child.stderr_path}"
                )
            launch(name)


def _evaluate(
    root: Path,
    graph_checkpoint: Path,
    episodes: int,
    device: str,
) -> None:
    checkpoints = {
        "graph": graph_checkpoint,
        "cacc": Path("unused-cacc-checkpoint"),
        **{
            name: root / "training" / name / "complete.pt"
            for name in PREDICTIVE_METHODS
        },
    }
    for name in ALL_METHODS:
        output = root / "evaluation" / name
        if (output / "summary.json").exists():
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "predictive_comparison.evaluate",
                name,
                str(checkpoints[name]),
                "--output",
                str(output),
                "--episodes",
                str(episodes),
                "--device",
                device,
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        _write_json(
            root / "study_monitor.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "phase": "evaluation",
                "completed": name,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/predictive_comparison_v2"),
    )
    parser.add_argument("--steps", type=int, default=301_144)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=60)
    arguments = parser.parse_args()
    protocol = PredictiveComparisonProtocol(
        training_steps=arguments.steps,
        total_interaction_budget=arguments.steps,
    )
    arguments.root.mkdir(parents=True, exist_ok=True)
    _write_json(
        arguments.root / "study_manifest.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "total_interaction_budget_per_learned_method": arguments.steps,
            "evaluation_episodes": arguments.episodes,
            "protocol": protocol.to_dict(),
        },
    )
    _train(arguments.root, arguments.steps, arguments.device, arguments.poll_seconds)
    _evaluate(
        arguments.root,
        Path(protocol.graph_checkpoint),
        arguments.episodes,
        arguments.device,
    )
    report = arguments.root / "comparison_report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "predictive_comparison.report",
            "--root",
            str(arguments.root / "evaluation"),
            "--output",
            str(report),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    manifest = _read_json(arguments.root / "study_manifest.json")
    manifest.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "report": str(report.resolve()),
        }
    )
    _write_json(arguments.root / "study_manifest.json", manifest)
    _write_json(
        arguments.root / "study_monitor.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": "complete",
            "report": str(report.resolve()),
        },
    )


if __name__ == "__main__":
    main()

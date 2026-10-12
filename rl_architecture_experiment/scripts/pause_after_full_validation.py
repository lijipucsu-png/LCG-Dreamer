"""Pause the active graph study after all full-model validations pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import psutil


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT_ROOT / "artifacts"
SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _validated_run(seed: int) -> dict[str, object] | None:
    output = ARTIFACTS / f"graph_full_seed{seed}"
    complete = output / "graph_complete.pt"
    foundation_path = output / "foundation_validation.json"
    auxiliary_path = output / "graph_auxiliary.json"
    if not all(path.exists() for path in (complete, foundation_path, auxiliary_path)):
        return None
    checkpoint_time = complete.stat().st_mtime
    if min(foundation_path.stat().st_mtime, auxiliary_path.stat().st_mtime) < checkpoint_time:
        return None
    foundation = _read_json(foundation_path)
    auxiliary = _read_json(auxiliary_path)
    if foundation is None or auxiliary is None:
        return None
    if not bool(foundation.get("passed")) or not bool(auxiliary.get("passed")):
        return None
    return {
        "checkpoint": str(complete),
        "foundation_passed": True,
        "auxiliary_passed": True,
    }


def _stop_ablation_children(process: psutil.Process) -> None:
    for child in process.children(recursive=True):
        try:
            command = " ".join(child.cmdline())
            if "train-graph-rl" in command and "--variant full" not in command:
                child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orchestrator_pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    arguments = parser.parse_args()

    process = psutil.Process(arguments.orchestrator_pid)
    command = " ".join(process.cmdline())
    if "scripts.run_graph_study" not in command:
        raise RuntimeError(
            f"PID {arguments.orchestrator_pid} is not the graph-study orchestrator"
        )

    while process.is_running():
        validations = {
            f"graph_full_seed{seed}": _validated_run(seed) for seed in SEEDS
        }
        if all(result is not None for result in validations.values()):
            _stop_ablation_children(process)
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)

            manifest = {
                "paused_at": datetime.now(timezone.utc).isoformat(),
                "reason": "awaiting_user_validation",
                "validation_results": validations,
                "all_full_model_gates_passed": True,
            }
            manifest_path = ARTIFACTS / "full_training_validation_complete.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            state = {
                "pid": arguments.orchestrator_pid,
                "status": "paused",
                "phase": "awaiting_user_validation",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "manifest": str(manifest_path),
            }
            (ARTIFACTS / "study_orchestrator.json").write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
            print(json.dumps(manifest), flush=True)
            return
        time.sleep(max(arguments.poll_seconds, 0.1))

    raise RuntimeError("graph-study orchestrator exited before the pause gate")


if __name__ == "__main__":
    main()

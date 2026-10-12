"""Restart the graph-study orchestrator with a wider evaluation pool.

The running orchestrator owns the monitored ablation-training subprocesses, so
it must remain alive until both final checkpoints have been atomically written.
This one-shot handoff suspends it before it can launch evaluation work, waits
for the completed training children to exit, and then starts the same study
command with the requested evaluation-worker count.
"""

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

import psutil


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "artifacts" / "evaluation_handoff_state.json"
CHECKPOINTS = (
    ROOT
    / "artifacts"
    / "graph_no_semantic_communication_seed42"
    / "graph_complete.pt",
    ROOT
    / "artifacts"
    / "graph_no_auxiliary_distillation_seed42"
    / "graph_complete.pt",
)
REPLAYS = tuple(path.with_name("graph_replay_latest.pt") for path in CHECKPOINTS)


def _write_state(status: str, phase: str, **details: Any) -> None:
    payload = {
        "pid": os.getpid(),
        "status": status,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def _artifacts_ready() -> bool:
    paths = (*CHECKPOINTS, *REPLAYS)
    return all(path.is_file() and path.stat().st_size > 1_000_000 for path in paths)


def _process_running(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _wait_for_training_exit(pids: tuple[int, ...], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while any(_process_running(pid) for pid in pids):
        if time.monotonic() >= deadline:
            active = [pid for pid in pids if _process_running(pid)]
            raise TimeoutError(f"training processes did not exit: {active}")
        time.sleep(1.0)


def _restart_command(python: str, evaluation_workers: int) -> list[str]:
    return [
        python,
        "-m",
        "scripts.run_graph_study",
        "--skip-wait",
        "--skip-runtime-sync",
        "--full-training-workers",
        "1",
        "--ablation-training-workers",
        "2",
        "--evaluation-workers",
        str(evaluation_workers),
        "--episodes-per-count",
        "20",
    ]


def run(
    orchestrator_pid: int,
    training_pids: tuple[int, ...],
    python: str,
    evaluation_workers: int,
    poll_seconds: float,
    exit_timeout: float,
) -> None:
    _write_state(
        "waiting",
        "waiting_for_complete_ablation_artifacts",
        orchestrator_pid=orchestrator_pid,
        training_pids=list(training_pids),
    )
    while not _artifacts_ready():
        if not _process_running(orchestrator_pid):
            raise RuntimeError("orchestrator exited before final artifacts were ready")
        time.sleep(poll_seconds)

    orchestrator = psutil.Process(orchestrator_pid)
    suspended = False
    terminated = False
    try:
        orchestrator.suspend()
        suspended = True
        _write_state(
            "running",
            "orchestrator_suspended",
            orchestrator_pid=orchestrator_pid,
        )
        _wait_for_training_exit(training_pids, exit_timeout)
        if not _artifacts_ready():
            raise RuntimeError("final artifacts became incomplete during handoff")

        command = _restart_command(python, evaluation_workers)
        log_root = ROOT / "artifacts" / "study_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        stdout_path = log_root / "study_orchestrator_evaluation3.stdout.log"
        stderr_path = log_root / "study_orchestrator_evaluation3.stderr.log"
        orchestrator.terminate()
        terminated = True
        try:
            orchestrator.wait(timeout=20.0)
        except psutil.TimeoutExpired:
            orchestrator.kill()
            orchestrator.wait(timeout=20.0)

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            replacement = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                creationflags=creation_flags,
            )
        _write_state(
            "complete",
            "replacement_started",
            previous_orchestrator_pid=orchestrator_pid,
            replacement_pid=replacement.pid,
            evaluation_workers=evaluation_workers,
            command=command,
            stdout=str(stdout_path.resolve()),
            stderr=str(stderr_path.resolve()),
        )
    finally:
        if suspended and not terminated and _process_running(orchestrator_pid):
            try:
                psutil.Process(orchestrator_pid).resume()
            except psutil.Error:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrator-pid", type=int, required=True)
    parser.add_argument("--training-pids", type=int, nargs="+", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--evaluation-workers", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--exit-timeout", type=float, default=300.0)
    arguments = parser.parse_args()
    try:
        run(
            arguments.orchestrator_pid,
            tuple(arguments.training_pids),
            arguments.python,
            arguments.evaluation_workers,
            arguments.poll_seconds,
            arguments.exit_timeout,
        )
    except Exception as error:
        _write_state("failed", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()

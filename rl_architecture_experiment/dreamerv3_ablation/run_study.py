"""Resumable monitored runner for the formal short-budget ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .protocol import PROTOCOL, ROOT, VARIANTS


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def finite_metrics(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"missing training metrics: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        for key, value in row.items():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                raise RuntimeError(f"non-finite {key} at {path}:{line_number}")


def gpu_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def run_process(
    command: list[str],
    log_path: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        last_snapshot = 0.0
        while process.poll() is None:
            now = time.monotonic()
            if now - last_snapshot >= 20.0:
                state.update(
                    pid=process.pid,
                    gpu=gpu_snapshot(),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                atomic_json(state_path, state)
                last_snapshot = now
            time.sleep(2.0)
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(
            f"subprocess failed with code {process.returncode}: {' '.join(command)}\n{tail}"
        )


def valid_manifest(
    path: Path,
    variant: str,
    seed: int,
    final: bool,
    protocol_hash: str,
) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_world = (
        PROTOCOL.final_world_updates if final else PROTOCOL.offline_world_updates
    )
    expected_actor = (
        PROTOCOL.final_actor_updates if final else PROTOCOL.offline_actor_updates
    )
    key = payload.get("variant")
    if isinstance(key, dict):
        key = key.get("key")
    return bool(
        key == variant
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("world_updates", -1)) == expected_world
        and int(payload.get("actor_updates", -1)) == expected_actor
        and int(payload.get("critic_updates", -1)) == expected_actor
        and payload.get("protocol_sha256") == protocol_hash
        and not payload.get("reward_changed", True)
    )


def run(output_root: Path, device: str) -> Path:
    PROTOCOL.validate()
    protocol_path = PROTOCOL.write(output_root / "protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = protocol["protocol_sha256"]
    state_path = output_root / "study_state.json"
    completed: list[str] = []
    state: dict[str, Any] = {
        "study": PROTOCOL.study,
        "status": "running",
        "protocol_sha256": protocol_hash,
        "total_runs": len(VARIANTS) * len(PROTOCOL.training_seeds),
        "completed": completed,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(state_path, state)
    try:
        for seed in PROTOCOL.training_seeds:
            for variant in VARIANTS:
                run_name = f"{variant.key}/seed_{seed}"
                output = output_root / variant.key / f"seed_{seed}"
                warmup_manifest = output / "manifest.json"
                warmup = output / "warmup_complete.pt"
                state.update(current=run_name, phase="warmup", completed=completed)
                atomic_json(state_path, state)
                if not (
                    warmup.exists()
                    and valid_manifest(
                        warmup_manifest, variant.key, seed, False, protocol_hash
                    )
                ):
                    run_process(
                        [
                            sys.executable,
                            "-m",
                            "dreamerv3_ablation.train_budgeted",
                            variant.key,
                            "--seed",
                            str(seed),
                            "--output-root",
                            str(output_root),
                            "--device",
                            device,
                        ],
                        output / "warmup_stdout.log",
                        state_path,
                        state,
                    )
                if not valid_manifest(
                    warmup_manifest, variant.key, seed, False, protocol_hash
                ):
                    raise RuntimeError(
                        f"warm-up manifest failed the frozen protocol audit: {run_name}"
                    )
                finite_metrics(output / "training_metrics.jsonl")

                final_manifest = output / "budgeted_manifest.json"
                final_checkpoint = output / "budgeted_complete.pt"
                state.update(current=run_name, phase="online_adaptation")
                atomic_json(state_path, state)
                if not (
                    final_checkpoint.exists()
                    and valid_manifest(
                        final_manifest, variant.key, seed, True, protocol_hash
                    )
                ):
                    run_process(
                        [
                            sys.executable,
                            "-m",
                            "dreamerv3_ablation.online_adapt",
                            variant.key,
                            "--seed",
                            str(seed),
                            "--warmup-checkpoint",
                            str(warmup),
                            "--output-root",
                            str(output_root),
                            "--device",
                            device,
                        ],
                        output / "online_stdout.log",
                        state_path,
                        state,
                    )
                if not valid_manifest(
                    final_manifest, variant.key, seed, True, protocol_hash
                ):
                    raise RuntimeError(
                        f"final manifest failed the frozen protocol audit: {run_name}"
                    )
                finite_metrics(output / "online_metrics.jsonl")
                completed.append(run_name)
                state.update(completed=completed, completed_count=len(completed))
                atomic_json(state_path, state)
        state.update(
            status="complete",
            phase="complete",
            completed=completed,
            completed_count=len(completed),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(state_path, state)
        return state_path
    except Exception as error:
        state.update(
            status="failed",
            error=str(error),
            failed_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v1",
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    print(run(arguments.output_root, arguments.device))


if __name__ == "__main__":
    main()

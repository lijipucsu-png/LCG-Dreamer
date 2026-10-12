"""Wait for the baseline GPU job, then execute the isolated graph study."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT.parent
SOURCE_RUN = SOURCE_ROOT / "artifacts" / "rl_capability_curriculum_gpu_v8d_formal_supervised"
STUDY_STATE = EXPERIMENT_ROOT / "artifacts" / "study_orchestrator.json"
TRAINING_STALE_TIMEOUT_SECONDS = 12.0 * 60.0
MAX_TRAINING_ATTEMPTS = 20


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def active_baseline_processes() -> list[int]:
    path = SOURCE_RUN / "active_training_pids.json"
    if not path.exists():
        return []
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        pid
        for pid in (int(state.get("train", 0)), int(state.get("watchdog", 0)))
        if pid_running(pid)
    ]


def write_study_state(status: str, phase: str, **details: object) -> None:
    STUDY_STATE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "pid": os.getpid(),
        "status": status,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    temporary = STUDY_STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STUDY_STATE)


def claim_study() -> None:
    if STUDY_STATE.exists():
        try:
            previous = json.loads(STUDY_STATE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        previous_pid = int(previous.get("pid", 0))
        if previous.get("status") == "running" and pid_running(previous_pid):
            raise RuntimeError(
                f"graph study is already running under pid {previous_pid}"
            )
    write_study_state("running", "initializing")


def wait_for_gpu_handoff(poll_seconds: int, quiet_seconds: int) -> None:
    quiet_since: float | None = None
    while True:
        if (SOURCE_RUN / "training_complete.json").exists():
            print("baseline completion marker detected", flush=True)
            return
        running = active_baseline_processes()
        if running:
            quiet_since = None
            print(f"baseline still active: pids={running}", flush=True)
        else:
            quiet_since = quiet_since or time.monotonic()
            quiet_for = time.monotonic() - quiet_since
            print(
                f"baseline processes absent for {quiet_for:.0f}/{quiet_seconds}s",
                flush=True,
            )
            if quiet_for >= quiet_seconds:
                return
        time.sleep(poll_seconds)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def snapshot_baseline() -> Path:
    destination = EXPERIMENT_ROOT / "artifacts" / "baseline_final"
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, object]] = {}
    for name in ("rl_latest.pt", "rl_best.pt", "rl_complete.pt", "training_complete.json"):
        source = SOURCE_RUN / name
        if not source.exists():
            continue
        target = destination / name
        shutil.copy2(source, target)
        copied[name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    latest = destination / "rl_latest.pt"
    if not latest.exists():
        raise RuntimeError("baseline directory does not contain rl_latest.pt")
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE_RUN),
        "completed": (destination / "training_complete.json").exists(),
        "checkpoint": {
            key: checkpoint.get(key)
            for key in (
                "format_version",
                "control_architecture",
                "total_steps",
                "episodes",
                "curriculum_stage",
                "best_score",
            )
        },
        "files": copied,
    }
    (destination / "baseline_final_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return destination


def synchronize_baseline_runtime() -> Path:
    """Bring final non-graph runtime changes into the isolated comparison tree."""

    relative_files = [
        Path("platoon_sumo/env.py"),
        Path("platoon_control/rl.py"),
        Path("platoon_control/models/dreamer.py"),
        Path("platoon_control/models/follower_rl.py"),
    ]
    scenario_source = SOURCE_ROOT / "platoon_sumo" / "scenario"
    relative_files.extend(
        path.relative_to(SOURCE_ROOT)
        for path in scenario_source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    files: dict[str, dict[str, str]] = {}
    for relative in relative_files:
        source = SOURCE_ROOT / relative
        destination = EXPERIMENT_ROOT / relative
        if not source.exists():
            raise RuntimeError(f"baseline runtime file disappeared: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files[str(relative)] = {
            "source_sha256": sha256(source),
            "experiment_sha256": sha256(destination),
        }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "freeze the final baseline environment, curriculum helpers, and "
            "legacy checkpoint runtime before architecture-only comparison"
        ),
        "excluded_experiment_modules": [
            "platoon_control/graph_agents.py",
            "platoon_control/graph_rl.py",
            "platoon_control/graph_validation.py",
            "platoon_control/graph_analysis.py",
            "platoon_control/models/graph_dreamer.py",
            "platoon_control/training.py",
            "platoon_control/evaluation.py",
            "platoon_control/cli.py",
        ],
        "files": files,
    }
    output = EXPERIMENT_ROOT / "artifacts" / "baseline_runtime_final.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def baseline_checkpoint(directory: Path) -> Path:
    complete = directory / "rl_complete.pt"
    return complete if complete.exists() else directory / "rl_latest.pt"


def run(
    command: list[str],
    log_name: str,
    allowed_return_codes: tuple[int, ...] = (0,),
) -> None:
    log_directory = EXPERIMENT_ROOT / "artifacts" / "study_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    print("running:", subprocess.list2cmdline(command), flush=True)
    with (log_directory / f"{log_name}.stdout.log").open("a", encoding="utf-8") as stdout, (
        log_directory / f"{log_name}.stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        result = subprocess.run(
            command,
            cwd=EXPERIMENT_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode not in allowed_return_codes:
        raise RuntimeError(f"command failed ({result.returncode}): {log_name}")


class TrainingMonitorError(RuntimeError):
    pass


def _tail_metric_records(path: Path, limit: int = 200) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 512 * 1024))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    records: list[dict[str, object]] = []
    for line in lines[-limit:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _new_metric_records(
    path: Path, offset: int
) -> tuple[list[dict[str, object]], int]:
    if not path.exists():
        return [], 0
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read()
    last_newline = payload.rfind(b"\n")
    if last_newline < 0:
        return [], offset
    complete_payload = payload[: last_newline + 1]
    new_offset = offset + last_newline + 1
    records: list[dict[str, object]] = []
    for line in complete_payload.decode("utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records, new_offset


def _gpu_snapshot() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _training_findings(
    records: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    fatal: list[str] = []
    warnings: list[str] = []
    for record in records:
        for name, value in record.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                fatal.append(f"non-finite metric: {name}={value}")
    episodes = [
        record for record in records if "training/episode" in record
    ][-3:]
    if any(
        float(record.get("training/background_collision_events", 0.0)) > 0.0
        for record in episodes
    ):
        fatal.append("background vehicle collision or overlap detected")
    def is_network_gridlock(record: dict[str, object]) -> bool:
        gridlock_seconds = float(
            record.get("training/background_max_gridlock_seconds", 0.0)
        )
        stuck = float(record.get("training/background_stuck_30s_count", 0.0))
        peak = max(
            float(record.get("training/background_peak_count", 0.0)), 1.0
        )
        completed_merges = float(
            record.get("training/ramp_merge_completed_events", 0.0)
        )
        # A metered on-ramp intentionally holds a small queue. Treat it as a
        # network failure only when congestion is globally persistent, or a
        # material fraction of traffic is immobile with no merge throughput.
        stalled_network = stuck >= max(20.0, 0.20 * peak)
        return gridlock_seconds >= 60.0 or (
            stalled_network and completed_merges <= 0.0
        )

    severe_gridlock = [record for record in episodes if is_network_gridlock(record)]
    if len(severe_gridlock) >= 2:
        fatal.append("background flow gridlock repeated in consecutive episodes")
    if episodes:
        latest = episodes[-1]
        step = int(latest.get("step", 0))
        if any(
            float(record.get("training/background_stuck_30s_count", 0.0))
            > 0.0
            for record in episodes
        ):
            warnings.append("background vehicles queued longer than 30 seconds")
        if any(
            float(record.get("training/ramp_meter_closed_events", 0.0)) > 0.0
            for record in episodes
        ):
            warnings.append("ramp meter activated to prevent spillback")
        if any(
            float(record.get("training/leader_stall_events", 0.0)) > 0.0
            for record in episodes
        ):
            warnings.append("leader stalled despite a clear path")
        if any(
            float(record.get("training/restart_detected", 0.0)) > 0.5
            and float(record.get("training/restart_spread_seconds", 0.0)) > 0.8
            for record in episodes
        ):
            warnings.append("platoon launch propagated sequentially after a stop")
        if any(
            float(record.get("training/restart_detected", 0.0)) > 0.5
            and float(record.get("training/restart_response_seconds", 0.0)) > 0.3
            for record in episodes
        ):
            warnings.append(
                "platoon restart response remained delayed after the front moved"
            )
        if any(
            float(record.get("training/restart_detected", 0.0)) > 0.5
            and float(record.get("training/restart_maximum_speed_spread", 0.0))
            > 0.5
            for record in episodes
        ):
            warnings.append("platoon speed consensus oscillated after restart")
        eligible = [
            record
            for record in episodes
            if float(record.get("training/foundation_probe_active", 0.0)) < 0.5
        ]
        if step >= 30_000 and len(eligible) >= 3:
            if all(
                float(record.get("training/failure", 0.0)) >= 0.5
                and float(record.get("training/max_gap", 0.0)) > 150.0
                for record in eligible[-3:]
            ):
                fatal.append("policy diverged in three non-probe episodes")
            if all(
                float(record.get("training/shield", 0.0)) > 0.95
                for record in eligible[-3:]
            ):
                fatal.append("safety shield saturated in three episodes")
    return sorted(set(fatal)), sorted(set(warnings))


def _latest_metric_value(
    records: list[dict[str, object]], key: str
) -> object | None:
    """Return the newest value for a sparse metric stream."""

    for record in reversed(records):
        if key in record:
            return record[key]
    return None


def _write_monitor_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_monitored_training(
    command: list[str], log_name: str, output: Path, attempt: int
) -> int:
    log_directory = EXPERIMENT_ROOT / "artifacts" / "study_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "logs" / "metrics.jsonl"
    monitor_path = output / "training_monitor.json"
    print("running monitored:", subprocess.list2cmdline(command), flush=True)
    with (log_directory / f"{log_name}.stdout.log").open(
        "a", encoding="utf-8"
    ) as stdout, (log_directory / f"{log_name}.stderr.log").open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=EXPERIMENT_ROOT,
            stdout=stdout,
            stderr=stderr,
        )
        started = time.monotonic()
        metric_offset = metrics_path.stat().st_size if metrics_path.exists() else 0
        recent_records: list[dict[str, object]] = []
        previous_records = _tail_metric_records(metrics_path, 5_000)
        last_curriculum_stage = _latest_metric_value(
            previous_records, "training/curriculum_stage"
        )
        while True:
            return_code = process.poll()
            new_records, metric_offset = _new_metric_records(
                metrics_path, metric_offset
            )
            recent_records.extend(new_records)
            # Updates are logged every few simulator steps, while one SUMO
            # episode can span thousands of steps. Keep enough history for
            # cross-episode safety findings instead of letting loss records
            # evict the latest episode diagnostics.
            recent_records = recent_records[-5_000:]
            fatal, warnings = _training_findings(recent_records)
            latest = (
                recent_records[-1]
                if recent_records
                else (previous_records[-1] if previous_records else {})
            )
            current_curriculum_stage = _latest_metric_value(
                recent_records, "training/curriculum_stage"
            )
            if current_curriculum_stage is not None:
                last_curriculum_stage = current_curriculum_stage
            stale_seconds = (
                min(
                    time.monotonic() - started,
                    time.time() - metrics_path.stat().st_mtime,
                )
                if metrics_path.exists()
                else time.monotonic() - started
            )
            if stale_seconds >= TRAINING_STALE_TIMEOUT_SECONDS:
                fatal.append(
                    "training metrics have not advanced for "
                    f"{TRAINING_STALE_TIMEOUT_SECONDS / 60.0:g} minutes"
                )
            state = {
                "status": "failed" if fatal else (
                    "finished" if return_code is not None else "running"
                ),
                "pid": process.pid,
                "attempt": attempt,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "latest_step": int(latest.get("step", 0)),
                "curriculum_stage": last_curriculum_stage,
                "live_min_gap": _latest_metric_value(
                    recent_records or previous_records,
                    "training/live_min_gap",
                ),
                "live_max_gap": _latest_metric_value(
                    recent_records or previous_records,
                    "training/live_max_gap",
                ),
                "live_current_min_gap": _latest_metric_value(
                    recent_records or previous_records,
                    "training/live_current_min_gap",
                ),
                "live_current_max_gap": _latest_metric_value(
                    recent_records or previous_records,
                    "training/live_current_max_gap",
                ),
                "live_speed_spread": _latest_metric_value(
                    recent_records or previous_records,
                    "training/live_speed_spread",
                ),
                "live_background_gridlock_seconds": _latest_metric_value(
                    recent_records or previous_records,
                    "training/live_background_gridlock_seconds",
                ),
                "stale_seconds": stale_seconds,
                "gpu_used_free_utilization": _gpu_snapshot(),
                "warnings": warnings,
                "fatal_findings": fatal,
            }
            _write_monitor_state(monitor_path, state)
            if fatal and return_code is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TrainingMonitorError("; ".join(fatal))
            if return_code is not None:
                return return_code
            time.sleep(30)


def train_one(
    python: str,
    seed: int,
    output: Path,
    variant: str,
    max_steps: int,
    device: str = "cuda",
) -> Path:
    complete = output / "graph_complete.pt"
    if complete.exists():
        return complete
    latest = output / "graph_latest.pt"
    for attempt in range(1, MAX_TRAINING_ATTEMPTS + 1):
        command = [
            python,
            "-m",
            "platoon_control",
            "train-graph-rl",
            "--seed",
            str(seed),
            "--max-steps",
            str(max_steps),
            "--device",
            device,
            "--variant",
            variant,
            "--checkpoint-interval",
            "5000",
            "--world-pretrain-updates",
            "500",
            "--actor-warmup-steps",
            "20000",
            "--output",
            str(output.relative_to(EXPERIMENT_ROOT)),
        ]
        if latest.exists():
            command.extend(("--resume", str(latest.relative_to(EXPERIMENT_ROOT))))
        try:
            return_code = run_monitored_training(
                command, f"train_{output.name}_attempt{attempt}", output, attempt
            )
        except TrainingMonitorError as error:
            return_code = -1
            print(
                f"monitor stopped {output.name} attempt {attempt}: {error}",
                flush=True,
            )
        if return_code == 0:
            break
        if not latest.exists() or attempt == MAX_TRAINING_ATTEMPTS:
            raise RuntimeError(
                f"training process failed ({return_code}): {output.name}"
            )
        print(
            f"retrying {output.name} from {latest} after exit {return_code}",
            flush=True,
        )
        time.sleep(15)
    return complete if complete.exists() else latest


def validate_one(
    python: str, checkpoint: Path, output: Path
) -> dict[str, bool]:
    run(
        [
            python,
            "-m",
            "platoon_control",
            "validate-graph-foundation",
            str(checkpoint.relative_to(EXPERIMENT_ROOT)),
            "--device",
            "cuda",
            "--output",
            str((output / "foundation_validation.json").relative_to(EXPERIMENT_ROOT)),
        ],
        f"validate_{output.name}",
        (0, 2),
    )
    run(
        [
            python,
            "-m",
            "platoon_control",
            "evaluate-graph-auxiliary",
            str(checkpoint.relative_to(EXPERIMENT_ROOT)),
            str((output / "graph_replay_latest.pt").relative_to(EXPERIMENT_ROOT)),
            "--device",
            "cuda",
            "--output",
            str((output / "graph_auxiliary.json").relative_to(EXPERIMENT_ROOT)),
        ],
        f"auxiliary_{output.name}",
    )
    foundation = json.loads(
        (output / "foundation_validation.json").read_text(encoding="utf-8")
    )
    auxiliary = json.loads(
        (output / "graph_auxiliary.json").read_text(encoding="utf-8")
    )
    return {
        "training_complete": checkpoint.name == "graph_complete.pt",
        "foundation_passed": bool(foundation.get("passed", False)),
        "auxiliary_passed": bool(auxiliary.get("passed", False)),
    }


def evaluate_one(
    python: str,
    checkpoint: Path,
    output: Path,
    variants: tuple[str, ...],
    episodes_per_count: int,
    seed: int,
    startup_delay: float = 0.0,
) -> Path:
    csv_path = output / "episodes.csv"
    summary = output / "summary.json"
    provenance_path = output / "provenance.json"
    expected = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "variants": list(variants),
        "total_episodes": episodes_per_count,
        "seed": seed,
    }
    if csv_path.exists() and summary.exists() and provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
        if (
            all(provenance.get(key) == value for key, value in expected.items())
            and provenance.get("episodes_sha256") == sha256(csv_path)
            and provenance.get("summary_sha256") == sha256(summary)
        ):
            return csv_path
    if startup_delay > 0.0:
        time.sleep(startup_delay)
    command = [
        python,
        "-m",
        "platoon_control",
        "evaluate",
        str(checkpoint.relative_to(EXPERIMENT_ROOT)),
        "--variants",
        *variants,
        "--episodes-per-count",
        "1",
        "--total-episodes",
        str(episodes_per_count),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--output",
        str(output.relative_to(EXPERIMENT_ROOT)),
    ]
    log_name = f"evaluate_{output.parent.name}_{output.name}"
    last_error: RuntimeError | None = None
    for attempt in range(1, 4):
        try:
            run(command, f"{log_name}_attempt{attempt}")
            last_error = None
            break
        except RuntimeError as error:
            last_error = error
            if attempt < 3:
                time.sleep(10.0 * attempt)
    if last_error is not None:
        raise last_error
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = episodes_per_count * len(variants)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"evaluation row count mismatch for {output.name}: "
            f"{len(rows)} != {expected_rows}"
        )
    if {row["variant"] for row in rows} != set(variants):
        raise RuntimeError(f"evaluation variants mismatch for {output.name}")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    if set(summary_payload.get("variants", {})) != set(variants):
        raise RuntimeError(f"evaluation summary variants mismatch for {output.name}")
    provenance = {
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes_sha256": sha256(csv_path),
        "summary_sha256": sha256(summary),
        "row_count": len(rows),
    }
    temporary = provenance_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    temporary.replace(provenance_path)
    return csv_path


def compare_one(
    python: str,
    baseline_csv: Path,
    graph_csv: Path,
    output: Path,
    baseline_variant: str = "full",
    graph_variant: str = "full",
) -> Path:
    run(
        [
            python,
            "-m",
            "platoon_control",
            "compare-graph-evaluations",
            str(baseline_csv.relative_to(EXPERIMENT_ROOT)),
            str(graph_csv.relative_to(EXPERIMENT_ROOT)),
            "--baseline-variant",
            baseline_variant,
            "--graph-variant",
            graph_variant,
            "--output",
            str(output.relative_to(EXPERIMENT_ROOT)),
        ],
        f"compare_{output.stem}",
    )
    return output


def aggregate_training_seeds(reports: list[Path], output: Path) -> Path:
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    metrics = loaded[0]["results"] if loaded else {}
    aggregate: dict[str, dict[str, float]] = {}
    for metric in metrics:
        differences = np.asarray(
            [report["results"][metric]["paired_difference"] for report in loaded],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "training_seed_mean_paired_difference": float(differences.mean()),
            "training_seed_std_paired_difference": float(differences.std()),
            "minimum_paired_difference": float(differences.min()),
            "maximum_paired_difference": float(differences.max()),
        }
    report = {
        "training_seed_reports": [str(path) for path in reports],
        "training_seed_count": len(loaded),
        "all_model_comparison_gates_passed": bool(loaded)
        and all(report["passed"] for report in loaded),
        "metrics": aggregate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def gpu_gate(python: str) -> Path:
    output = EXPERIMENT_ROOT / "artifacts" / "study_logs" / "graph_gpu_benchmark.json"
    run(
        [
            python,
            "-m",
            "scripts.benchmark_graph_gpu",
            "--device",
            "cuda",
            "--batch-size",
            "16",
            "--sequence-length",
            "50",
            "--horizon",
            "30",
            "--max-memory-mib",
            "2000",
            "--output",
            str(output.relative_to(EXPERIMENT_ROOT)),
        ],
        "graph_gpu_benchmark",
    )
    return output


def regression_gate(python: str) -> None:
    run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "--basetemp=artifacts/study_logs/pytest_handoff",
        ],
        "handoff_regression",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=r"D:\sofat\anaconda3\envs\pytorch_gpu_env\python.exe",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--quiet-seconds", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--evaluation-seed", type=int, default=30_000)
    parser.add_argument("--episodes-per-count", type=int, default=20)
    parser.add_argument("--full-training-workers", type=int, default=3)
    parser.add_argument("--ablation-training-workers", type=int, default=2)
    parser.add_argument("--evaluation-workers", type=int, default=2)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--skip-runtime-sync", action="store_true")
    parser.add_argument("--skip-handoff-regression", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--pause-after-full-validation",
        action="store_true",
        help="Stop after all full seeds pass foundation and auxiliary validation.",
    )
    arguments = parser.parse_args()
    claim_study()
    if not arguments.skip_wait:
        write_study_state("running", "waiting_for_baseline_gpu")
        wait_for_gpu_handoff(arguments.poll_seconds, arguments.quiet_seconds)
    write_study_state("running", "snapshotting_baseline")
    baseline = snapshot_baseline()
    print(f"baseline snapshot: {baseline}", flush=True)
    if arguments.skip_runtime_sync:
        runtime_manifest = EXPERIMENT_ROOT / "artifacts" / "baseline_runtime_final.json"
    else:
        write_study_state("running", "synchronizing_final_runtime")
        runtime_manifest = synchronize_baseline_runtime()
    if not arguments.skip_handoff_regression:
        write_study_state("running", "handoff_regression")
        regression_gate(arguments.python)
    write_study_state("running", "gpu_gate", baseline=str(baseline))
    benchmark = gpu_gate(arguments.python)
    if not 1 <= arguments.full_training_workers <= 3:
        raise ValueError("full-training-workers must be in [1, 3]")
    if not 1 <= arguments.ablation_training_workers <= 2:
        raise ValueError("ablation-training-workers must be in [1, 2]")
    if not 1 <= arguments.evaluation_workers <= 3:
        raise ValueError("evaluation-workers must be in [1, 3]")

    completed_by_seed: dict[int, Path] = {}
    validation_results: dict[str, dict[str, bool]] = {}
    full_seeds = (42, 43, 44)
    full_devices = {seed: "cuda" for seed in full_seeds}
    write_study_state(
        "running",
        "training_full_seeds",
        seeds=list(full_seeds),
        workers=arguments.full_training_workers,
    )
    with ThreadPoolExecutor(max_workers=arguments.full_training_workers) as executor:
        full_futures = {
            executor.submit(
                train_one,
                arguments.python,
                seed,
                EXPERIMENT_ROOT / "artifacts" / f"graph_full_seed{seed}",
                "full",
                arguments.max_steps,
                full_devices[seed],
            ): seed
            for seed in full_seeds
        }
        for future in as_completed(full_futures):
            seed = full_futures[future]
            completed_by_seed[seed] = future.result()

    completed = [completed_by_seed[seed] for seed in full_seeds]
    for seed, checkpoint in zip(full_seeds, completed):
        output = checkpoint.parent
        write_study_state("running", f"validating_full_seed{seed}")
        validation_results[output.name] = validate_one(
            arguments.python, checkpoint, output
        )

    if arguments.pause_after_full_validation:
        validation_manifest = {
            "paused_at": datetime.now(timezone.utc).isoformat(),
            "completed_graph_runs": [str(path) for path in completed],
            "validation_results": validation_results,
            "all_full_model_gates_passed": all(
                all(result.values()) for result in validation_results.values()
            ),
        }
        manifest_path = (
            EXPERIMENT_ROOT
            / "artifacts"
            / "full_training_validation_complete.json"
        )
        manifest_path.write_text(
            json.dumps(validation_manifest, indent=2), encoding="utf-8"
        )
        write_study_state(
            "paused",
            "awaiting_user_validation",
            manifest=str(manifest_path),
        )
        print(json.dumps(validation_manifest), flush=True)
        return

    ablation_variants = (
        "no_semantic_communication",
        "no_auxiliary_distillation",
    )
    ablations_by_variant: dict[str, Path] = {}
    write_study_state(
        "running",
        "training_ablations",
        variants=list(ablation_variants),
        workers=arguments.ablation_training_workers,
    )
    with ThreadPoolExecutor(
        max_workers=arguments.ablation_training_workers
    ) as executor:
        ablation_futures = {
            executor.submit(
                train_one,
                arguments.python,
                42,
                EXPERIMENT_ROOT / "artifacts" / f"graph_{variant}_seed42",
                variant,
                arguments.max_steps,
            ): variant
            for variant in ablation_variants
        }
        for future in as_completed(ablation_futures):
            variant = ablation_futures[future]
            ablations_by_variant[variant] = future.result()
    ablations = [ablations_by_variant[variant] for variant in ablation_variants]

    comparisons: list[Path] = []
    baseline_comparisons: list[Path] = []
    evaluation_files: list[Path] = []
    aggregate_report: Path | None = None
    if not arguments.skip_evaluation:
        write_study_state("running", "paired_paper_evaluation")
        evaluation_root = EXPERIMENT_ROOT / "artifacts" / "paper_evaluation"
        evaluation_specs: list[
            tuple[str, Path, Path, tuple[str, ...]]
        ] = [
            (
                "baseline",
                baseline_checkpoint(baseline),
                evaluation_root / "baseline",
                ("idm", "cacc", "full"),
            )
        ]
        evaluation_specs.extend(
            (
                checkpoint.parent.name,
                checkpoint,
                evaluation_root / checkpoint.parent.name,
                (
                    "full",
                    "no_latent_extrapolation",
                    "no_differentiable_cbf",
                    "float_semantic_code",
                ),
            )
            for checkpoint in completed
        )
        evaluation_specs.extend(
            (
                checkpoint.parent.name,
                checkpoint,
                evaluation_root / checkpoint.parent.name,
                ("full",),
            )
            for checkpoint in ablations
        )
        evaluated: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=arguments.evaluation_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_one,
                    arguments.python,
                    checkpoint,
                    output,
                    variants,
                    arguments.episodes_per_count,
                    arguments.evaluation_seed,
                    float(index % arguments.evaluation_workers) * 5.0,
                ): name
                for index, (name, checkpoint, output, variants) in enumerate(
                    evaluation_specs
                )
            }
            for future in as_completed(futures):
                evaluated[futures[future]] = future.result()
        evaluation_files = [evaluated[name] for name, *_ in evaluation_specs]
        baseline_csv = evaluated["baseline"]
        full_csvs = [evaluated[checkpoint.parent.name] for checkpoint in completed]
        if len(completed) != len(full_csvs):
            raise RuntimeError(
                "completed checkpoints and evaluation files have different lengths"
            )
        for checkpoint, graph_csv in zip(completed, full_csvs):
            run_name = checkpoint.parent.name
            model_comparison = compare_one(
                arguments.python,
                baseline_csv,
                graph_csv,
                evaluation_root / f"baseline_vs_{run_name}.json",
            )
            comparisons.append(model_comparison)
            baseline_comparisons.append(model_comparison)
            for variant in (
                "no_latent_extrapolation",
                "no_differentiable_cbf",
                "float_semantic_code",
            ):
                comparisons.append(
                    compare_one(
                        arguments.python,
                        graph_csv,
                        graph_csv,
                        evaluation_root / f"{run_name}_{variant}.json",
                        "full",
                        variant,
                    )
                )
        for checkpoint in ablations:
            run_name = checkpoint.parent.name
            ablation_csv = evaluated[run_name]
            comparisons.append(
                compare_one(
                    arguments.python,
                    full_csvs[0],
                    ablation_csv,
                    evaluation_root / f"graph_full_seed42_vs_{run_name}.json",
                )
            )
        aggregate_report = aggregate_training_seeds(
            baseline_comparisons,
            evaluation_root / "baseline_vs_graph_training_seed_aggregate.json",
        )
    manifest = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "baseline": str(baseline),
        "baseline_runtime_manifest": str(runtime_manifest),
        "gpu_benchmark": str(benchmark),
        "completed_graph_runs": [str(path) for path in completed],
        "completed_retrained_ablations": [str(path) for path in ablations],
        "validation_results": validation_results,
        "all_full_model_gates_passed": all(
            all(result.values()) for result in validation_results.values()
        ),
        "evaluations": [str(path) for path in evaluation_files],
        "comparisons": [str(path) for path in comparisons],
        "training_seed_aggregate": (
            str(aggregate_report) if aggregate_report is not None else None
        ),
    }
    manifest_path = EXPERIMENT_ROOT / "artifacts" / "graph_study_complete.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_study_state("complete", "complete", manifest=str(manifest_path))
    print(
        json.dumps(manifest),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            state = json.loads(STUDY_STATE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if int(state.get("pid", 0)) == os.getpid():
            write_study_state("failed", "failed", error=repr(error))
        raise

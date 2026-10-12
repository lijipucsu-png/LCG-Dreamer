"""Paired baseline and ablation evaluation for the complete platoon controller."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions

from .training import load_controller


VARIANTS = (
    "idm",
    "cacc",
    "full",
    "no_v2i",
    "no_front_prediction",
    "no_delay_compensation",
    "no_cutin_prediction",
    "no_learned_wave",
    "no_semantic_communication",
    "no_latent_extrapolation",
    "no_differentiable_cbf",
    "float_semantic_code",
)

RESUME_SCHEMA_VERSION = 1
RESUME_ROWS_NAME = "episodes.partial.jsonl"
RESUME_METADATA_NAME = "episodes.partial.meta.json"


def _idm_actions(observations: dict) -> dict:
    actions: dict[str, dict[str, Any]] = {}
    for agent, observation in observations.items():
        speed = float(observation["ego"][1])
        if agent == "leader":
            mask = observation["front_mask"] > 0.5
            latest = observation["front_history"][:, -1]
            ahead = np.flatnonzero(mask & (latest[:, 0] > 0.0))
            if len(ahead):
                nearest = int(ahead[np.argmin(latest[ahead, 0])])
                gap = max(0.5, float(latest[nearest, 0] - latest[nearest, 4]))
                leader_speed = float(latest[nearest, 2])
            else:
                gap, leader_speed = 1e4, speed
        else:
            gap = max(0.5, float(observation["local_predecessor"][4]))
            leader_speed = speed + float(observation["local_predecessor"][1])
        desired_gap = 2.0 + max(
            0.0,
            speed * 1.2
            + speed * (speed - leader_speed) / (2.0 * np.sqrt(1.2 * 2.5)),
        )
        acceleration = 1.2 * (1.0 - (speed / 25.0) ** 4 - (desired_gap / gap) ** 2)
        actions[agent] = {
            "accel": np.asarray([np.clip(acceleration, -3.0, 1.2)], np.float32),
            "lane_change": 0,
        }
    return actions


@dataclass
class EvaluationConfig:
    seed: int = 2026
    episodes_per_front_count: int = 5
    total_episodes: int | None = None
    max_steps: int = 2400
    output_dir: str = "artifacts/evaluation"
    calibration_path: str = "artifacts/calibration/calibrated_parameters.json"
    wave_model_path: str = "artifacts/checkpoints/wave_best.pt"
    device: str = "auto"


def _evaluation_cases(config: EvaluationConfig) -> list[tuple[int, int]]:
    """Return paired (seed, front-count) cases for every controller variant."""

    if config.total_episodes is not None:
        if config.total_episodes <= 0:
            raise ValueError("total_episodes must be positive")
        return [
            (
                config.seed + (index % 9) * 1000 + index // 9,
                index % 9,
            )
            for index in range(config.total_episodes)
        ]
    if config.episodes_per_front_count <= 0:
        raise ValueError("episodes_per_front_count must be positive")
    return [
        (config.seed + front_count * 1000 + replicate, front_count)
        for front_count in range(9)
        for replicate in range(config.episodes_per_front_count)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_identity(path_value: str | Path) -> dict[str, str | None]:
    path = Path(path_value).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _resume_signature(
    checkpoint: str | Path,
    config: EvaluationConfig,
    variants: tuple[str, ...],
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "checkpoint": _input_identity(checkpoint_path),
        "variants": list(variants),
        "cases": [list(case) for case in _evaluation_cases(config)],
        "max_steps": config.max_steps,
        "calibration": _input_identity(config.calibration_path),
        "wave_model": _input_identity(config.wave_model_path),
    }


def _expected_row_keys(
    config: EvaluationConfig, variants: tuple[str, ...]
) -> list[tuple[int, int, str]]:
    return [
        (seed, front_count, variant)
        for seed, front_count in _evaluation_cases(config)
        for variant in variants
    ]


def _row_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["seed"]),
        int(row["front_vehicle_count"]),
        str(row["variant"]),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_resume_rows(
    path: Path,
    rows_by_key: dict[tuple[int, int, str], dict[str, Any]],
    expected_keys: list[tuple[int, int, str]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for key in expected_keys:
            row = rows_by_key.get(key)
            if row is not None:
                stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_resume_rows(
    path: Path,
    expected_keys: list[tuple[int, int, str]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    expected = set(expected_keys)
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise RuntimeError(
                f"corrupt evaluation resume record at {path}:{index + 1}"
            )
        if not isinstance(row, dict):
            raise RuntimeError(
                f"invalid evaluation resume record at {path}:{index + 1}"
            )
        try:
            key = _row_key(row)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"invalid evaluation resume key at {path}:{index + 1}"
            ) from error
        if key not in expected:
            raise RuntimeError(
                f"evaluation resume record is outside the requested protocol: {key}"
            )
        rows[key] = row
    return rows


def seed_evaluation_resume(
    checkpoint: str | Path,
    config: EvaluationConfig,
    variants: tuple[str, ...],
    records: list[dict[str, Any]],
) -> int:
    """Seed a resumable evaluation from durable per-episode log records."""

    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown evaluation variants: {sorted(unknown)}")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / RESUME_METADATA_NAME
    rows_path = output / RESUME_ROWS_NAME
    signature = _resume_signature(checkpoint, config, variants)
    if metadata_path.exists():
        existing_signature = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_signature != signature:
            raise RuntimeError(
                f"evaluation resume metadata does not match the requested protocol: "
                f"{metadata_path}"
            )
    elif rows_path.exists():
        raise RuntimeError(f"evaluation resume rows have no metadata: {rows_path}")
    expected_keys = _expected_row_keys(config, variants)
    expected = set(expected_keys)
    rows_by_key = _read_resume_rows(rows_path, expected_keys)
    for row in records:
        try:
            key = _row_key(row)
        except (KeyError, TypeError, ValueError):
            continue
        if key in expected:
            rows_by_key[key] = row
    _atomic_json(metadata_path, signature)
    _write_resume_rows(rows_path, rows_by_key, expected_keys)
    return len(rows_by_key)


def _load_evaluation_resume(
    checkpoint: str | Path,
    config: EvaluationConfig,
    variants: tuple[str, ...],
) -> tuple[
    Path,
    list[tuple[int, int, str]],
    dict[tuple[int, int, str], dict[str, Any]],
]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / RESUME_METADATA_NAME
    rows_path = output / RESUME_ROWS_NAME
    signature = _resume_signature(checkpoint, config, variants)
    if metadata_path.exists():
        existing_signature = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_signature != signature:
            raise RuntimeError(
                f"evaluation resume metadata does not match the requested protocol: "
                f"{metadata_path}"
            )
    elif rows_path.exists():
        raise RuntimeError(f"evaluation resume rows have no metadata: {rows_path}")
    else:
        _atomic_json(metadata_path, signature)
    expected_keys = _expected_row_keys(config, variants)
    rows_by_key = _read_resume_rows(rows_path, expected_keys)
    # Canonicalize after a power interruption, removing duplicate or torn tail rows.
    _write_resume_rows(rows_path, rows_by_key, expected_keys)
    return rows_path, expected_keys, rows_by_key


def _ablated_observations(observations: dict, variant: str) -> dict:
    if variant not in ("no_v2i", "no_front_prediction"):
        return observations
    output = copy.deepcopy(observations)
    if variant == "no_v2i":
        output["leader"]["v2i_flow"].fill(0.0)
        output["leader"]["road_profile"].fill(0.0)
    if variant == "no_front_prediction":
        output["leader"]["front_history"].fill(0.0)
        output["leader"]["front_mask"].fill(0.0)
    return output


def _episode(
    controller: Any,
    config: EvaluationConfig,
    variant: str,
    seed: int,
    front_count: int,
) -> dict[str, Any]:
    env = PlatoonParallelEnv(
        calibration_path=config.calibration_path,
        wave_model_path=config.wave_model_path,
    )
    wave_mode = "off" if variant == "no_learned_wave" else "learned"
    observations, _ = env.reset(
        seed=seed,
        options={"front_vehicle_count": front_count, "wave_mode": wave_mode},
    )
    if controller is not None:
        controller.reset()
    rewards = 0.0
    fuel = 0.0
    speeds: dict[str, list[float]] = {agent: [] for agent in PLATOON_IDS}
    accelerations: dict[str, list[float]] = {agent: [] for agent in PLATOON_IDS}
    gaps: list[float] = []
    final_info: dict[str, Any] = {}
    shield_interventions: list[float] = []
    inference_times_ms: list[float] = []
    try:
        for step_index in range(config.max_steps):
            if not env.agents:
                break
            if variant == "cacc":
                actions = baseline_actions(observations)
            elif variant == "idm":
                actions = _idm_actions(observations)
            else:
                model_observations = _ablated_observations(observations, variant)
                flags = frozenset(
                    name
                    for name in (
                        "delay_compensation",
                        "cutin_prediction",
                        "semantic_communication",
                        "latent_extrapolation",
                        "differentiable_cbf",
                        "float_semantic_code",
                    )
                    if variant == f"no_{name}"
                )
                if variant == "float_semantic_code":
                    flags = frozenset((*flags, "float_semantic_code"))
                inference_started = time.perf_counter()
                policy_states = controller.follower_policy_states(
                    model_observations, flags
                )[0]
                env.set_follower_control_predictions(
                    controller.follower_control_predictions(policy_states)
                )
                actions = controller.act(
                    model_observations,
                    deterministic=True,
                    ablations=flags,
                    policy_states=policy_states,
                )
                inference_times_ms.append(
                    (time.perf_counter() - inference_started) * 1000.0
                )
            observations, step_rewards, terminations, truncations, infos = env.step(actions)
            if controller is not None and variant not in ("cacc", "idm"):
                controller.record_applied_actions(infos)
            rewards += step_rewards["leader"]
            final_info = infos["leader"]
            shield_interventions.extend(
                float(infos[agent].get("safety_intervened", False))
                for agent in PLATOON_IDS
            )
            fuel += final_info["reward_components"]["fuel_mg_per_second"] * env.step_length
            active = set(env.connection.vehicle.getIDList())
            for agent in PLATOON_IDS:
                if agent in active:
                    speeds[agent].append(float(env.connection.vehicle.getSpeed(agent)))
                    accelerations[agent].append(float(env.connection.vehicle.getAcceleration(agent)))
                    leader = env.connection.vehicle.getLeader(agent, 100.0)
                    if leader:
                        gaps.append(float(leader[1]))
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        event_log = env.event_log
        env.close()
    leader_rms = float(np.sqrt(np.mean(np.square(accelerations["leader"])))) if accelerations["leader"] else 0.0
    follower_rms = [
        float(np.sqrt(np.mean(np.square(accelerations[agent]))))
        for agent in FOLLOWER_IDS
        if accelerations[agent]
    ]
    all_speeds = np.concatenate([np.asarray(values) for values in speeds.values() if values])
    all_accelerations = np.concatenate(
        [np.asarray(values) for values in accelerations.values() if values]
    )
    jerk_groups = [
        np.diff(values) / 0.1 for values in accelerations.values() if len(values) > 1
    ]
    jerks = np.concatenate(jerk_groups) if jerk_groups else np.zeros(1)
    event_names = {event["event"] for event in event_log}
    return {
        "variant": variant,
        "seed": seed,
        "front_vehicle_count": front_count,
        "steps": len(speeds["leader"]),
        "reward": rewards,
        "success": int(final_info.get("success", False)),
        "collision": int(final_info.get("collision", False)),
        "failure": int(final_info.get("failure", False)),
        "travel_time_s": float(final_info.get("time", config.max_steps * 0.1)),
        "mean_speed_mps": float(all_speeds.mean()) if len(all_speeds) else 0.0,
        "speed_std_mps": float(all_speeds.std()) if len(all_speeds) else 0.0,
        "acceleration_rms_mps2": float(np.sqrt(np.mean(np.square(all_accelerations)))) if len(all_accelerations) else 0.0,
        "jerk_rms_mps3": float(np.sqrt(np.mean(np.square(jerks)))) if len(jerks) else 0.0,
        "minimum_gap_m": float(np.min(gaps)) if gaps else -1.0,
        "fuel_mg": fuel,
        "string_amplification": float(max(follower_rms, default=0.0) / max(leader_rms, 1e-3)),
        "cutin_success": int("cutin_success" in event_names),
        "wave_triggered": int("wave_brake" in event_names),
        "shield_intervention_rate": float(
            np.mean(shield_interventions or [0.0])
        ),
        "inference_p95_ms": float(
            np.percentile(inference_times_ms, 95)
            if inference_times_ms
            else 0.0
        ),
        "semantic_payload_bytes": (
            18
            if variant not in ("cacc", "idm")
            and getattr(controller, "architecture_id", "")
            == "ma_latent_graph_dreamer_v1"
            else 0
        ),
    }


def evaluate(
    checkpoint: str | Path,
    config: EvaluationConfig,
    variants: tuple[str, ...] = VARIANTS,
) -> Path:
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown evaluation variants: {sorted(unknown)}")
    rows_path, expected_keys, rows_by_key = _load_evaluation_resume(
        checkpoint, config, variants
    )
    if rows_by_key:
        print(
            json.dumps(
                {
                    "event": "evaluation_resumed",
                    "completed_rows": len(rows_by_key),
                    "expected_rows": len(expected_keys),
                }
            ),
            flush=True,
        )
    controller = load_controller(checkpoint, config.device)
    with rows_path.open("a", encoding="utf-8", newline="\n") as resume_stream:
        for episode_seed, front_count, variant in expected_keys:
            key = (episode_seed, front_count, variant)
            if key in rows_by_key:
                continue
            row = _episode(controller, config, variant, episode_seed, front_count)
            if _row_key(row) != key:
                raise RuntimeError(
                    f"evaluation returned an unexpected row key: {_row_key(row)} != {key}"
                )
            rows_by_key[key] = row
            resume_stream.write(json.dumps(row) + "\n")
            resume_stream.flush()
            os.fsync(resume_stream.fileno())
            print(json.dumps(row), flush=True)
    rows = [rows_by_key[key] for key in expected_keys]
    output = Path(config.output_dir)
    csv_path = output / "episodes.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    csv_temporary.replace(csv_path)
    summary: dict[str, Any] = {"config": asdict(config), "variants": {}}
    metrics = [
        key for key, value in rows[0].items() if key not in ("variant", "seed") and isinstance(value, (int, float))
    ]
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary["variants"][variant] = {
            metric: {"mean": float(np.mean([row[metric] for row in selected])), "std": float(np.std([row[metric] for row in selected]))}
            for metric in metrics
        }
    summary_path = output / "summary.json"
    _atomic_json(summary_path, summary)
    return summary_path

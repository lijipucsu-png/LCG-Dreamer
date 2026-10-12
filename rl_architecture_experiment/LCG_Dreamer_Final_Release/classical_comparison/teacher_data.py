"""Collect one shared CACC demonstration set for assisted classical-RL training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from platoon_control.rl import RLConfig, _curriculum_options
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions

from .common import ActionCodec, ComparisonProtocol, StateAdapter, reward_fingerprint


DATASET_VERSION = 1


def _save_dataset(
    path: Path,
    stages: tuple[int, ...],
    successes_per_stage: int,
    kept_states: list[np.ndarray],
    kept_actions: list[np.ndarray],
    kept_stages: list[np.ndarray],
    episode_offsets: list[int],
    records: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    states = np.concatenate(kept_states)
    actions = np.concatenate(kept_actions)
    stage_labels = np.concatenate(kept_stages)
    metadata = {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_controller": "deterministic project CACC",
        "teacher_used_at_evaluation": False,
        "stages": list(stages),
        "successes_per_stage": successes_per_stage,
        "complete": complete,
        "transitions": int(len(states)),
        "episodes": len(episode_offsets) - 1,
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "reward_fingerprint": reward_fingerprint(),
        "records": records,
    }
    payload = {
        "metadata": metadata,
        "states": torch.from_numpy(states).to(torch.float16),
        "actions": torch.from_numpy(actions).to(torch.float16),
        "stages": torch.from_numpy(stage_labels),
        "episode_offsets": torch.as_tensor(episode_offsets, dtype=torch.int64),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _restore_partial(
    path: Path,
    stages: tuple[int, ...],
    successes_per_stage: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[dict[str, Any]],
]:
    if not path.exists():
        return [], [], [], [0], []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata.get("reward_fingerprint") != reward_fingerprint():
        raise ValueError("partial teacher data reward fingerprint is stale")
    if tuple(metadata.get("stages", ())) != stages:
        raise ValueError("partial teacher data stage list does not match")
    if int(metadata.get("successes_per_stage", -1)) != successes_per_stage:
        raise ValueError("partial teacher data target count does not match")
    states = payload["states"].float().numpy()
    actions = payload["actions"].float().numpy()
    labels = payload["stages"].numpy()
    offsets = payload["episode_offsets"].tolist()
    kept_states = [states[start:end] for start, end in zip(offsets, offsets[1:])]
    kept_actions = [actions[start:end] for start, end in zip(offsets, offsets[1:])]
    kept_stages = [labels[start:end] for start, end in zip(offsets, offsets[1:])]
    return kept_states, kept_actions, kept_stages, offsets, list(metadata["records"])


def _options(
    protocol: ComparisonProtocol, stage: int, attempt: int
) -> tuple[int, dict[str, Any], str]:
    scenario_seed = protocol.training_seed + 60_000 + stage * 1_000 + attempt
    rng = np.random.default_rng(scenario_seed)
    config = RLConfig(
        seed=protocol.training_seed,
        episode_max_time=protocol.episode_max_time,
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
    )
    options, _ = _curriculum_options(stage, rng, config, 0)
    scenario_hash = hashlib.sha256(
        json.dumps(options, sort_keys=True, default=float).encode()
    ).hexdigest()
    return scenario_seed, options, scenario_hash


def _event_reason(events: Iterable[dict[str, Any]], final: dict[str, Any]) -> str:
    names = {str(event.get("event", "")) for event in events}
    if final.get("success"):
        return "success"
    if final.get("collision"):
        return "collision"
    if "leader_stalled_with_clear_path" in names:
        return "leader_stalled_with_clear_path"
    if final.get("failure"):
        return "vehicle_missing"
    return "truncated"


def collect_teacher_data(
    output: Path,
    stages: tuple[int, ...],
    successes_per_stage: int,
    maximum_attempts_per_stage: int,
    protocol: ComparisonProtocol,
) -> Path:
    """Store only collision-free, endpoint-success CACC trajectories."""

    output.parent.mkdir(parents=True, exist_ok=True)
    env = PlatoonParallelEnv(
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
        max_time=protocol.episode_max_time,
    )
    adapter = StateAdapter()
    codec = ActionCodec(protocol.action_min, protocol.action_max)
    partial = output.with_suffix(output.suffix + ".partial")
    progress = output.with_suffix(output.suffix + ".progress.jsonl")
    (
        kept_states,
        kept_actions,
        kept_stages,
        episode_offsets,
        records,
    ) = _restore_partial(partial, stages, successes_per_stage)
    try:
        for stage in stages:
            stage_records = [row for row in records if row["stage"] == stage]
            stage_successes = sum(int(row["kept"]) for row in stage_records)
            if stage_successes >= successes_per_stage:
                continue
            first_attempt = (
                max((int(row["attempt"]) for row in stage_records), default=-1) + 1
            )
            for attempt in range(first_attempt, maximum_attempts_per_stage):
                reset_seed, options, scenario_hash = _options(
                    protocol, stage, attempt
                )
                observations, _ = env.reset(seed=reset_seed, options=options)
                previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
                episode_states: list[np.ndarray] = []
                episode_actions: list[np.ndarray] = []
                minimum_gap = float("inf")
                maximum_gap = 0.0
                maximum_speed_spread = 0.0
                shield_count = 0
                final_info: dict[str, Any] = {}
                while env.agents:
                    state = adapter(observations, previous_applied)
                    actions = baseline_actions(observations)
                    physical = np.asarray(
                        [float(actions[agent]["accel"][0]) for agent in PLATOON_IDS],
                        np.float32,
                    )
                    episode_states.append(state.copy())
                    episode_actions.append(codec.to_normalized(physical))
                    observations, _, terminations, truncations, infos = env.step(
                        actions
                    )
                    final_info = infos["leader"]
                    previous_applied = np.asarray(
                        [
                            float(infos[agent].get("applied_acceleration", 0.0))
                            for agent in PLATOON_IDS
                        ],
                        np.float32,
                    )
                    active = set(env.connection.vehicle.getIDList())
                    gaps = [
                        float(env._platoon_gap(follower, active))
                        for follower in FOLLOWER_IDS
                        if follower in active
                    ]
                    if gaps:
                        minimum_gap = min(minimum_gap, min(gaps))
                        maximum_gap = max(maximum_gap, max(gaps))
                    speeds = [
                        float(env.connection.vehicle.getSpeed(agent))
                        for agent in PLATOON_IDS
                        if agent in active
                    ]
                    if speeds:
                        maximum_speed_spread = max(
                            maximum_speed_spread, max(speeds) - min(speeds)
                        )
                    shield_count += sum(
                        int(bool(infos[agent].get("safety_intervened", False)))
                        for agent in PLATOON_IDS
                        if agent in infos
                    )
                    if any(terminations.values()) or any(truncations.values()):
                        break
                success = bool(final_info.get("success", False))
                record = {
                    "stage": stage,
                    "attempt": attempt,
                    "reset_seed": reset_seed,
                    "scenario_hash": scenario_hash,
                    "steps": len(episode_states),
                    "success": success,
                    "collision": bool(final_info.get("collision", False)),
                    "failure": bool(final_info.get("failure", False)),
                    "termination_reason": _event_reason(env.event_log, final_info),
                    "minimum_gap_m": minimum_gap,
                    "maximum_gap_m": maximum_gap,
                    "maximum_speed_spread_mps": maximum_speed_spread,
                    "shield_rate": shield_count
                    / max(len(episode_states) * len(PLATOON_IDS), 1),
                    "kept": success,
                }
                records.append(record)
                with progress.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if success:
                    states = np.stack(episode_states).astype(np.float32)
                    actions_array = np.stack(episode_actions).astype(np.float32)
                    kept_states.append(states)
                    kept_actions.append(actions_array)
                    kept_stages.append(
                        np.full(len(states), stage, dtype=np.uint8)
                    )
                    episode_offsets.append(episode_offsets[-1] + len(states))
                    stage_successes += 1
                    _save_dataset(
                        partial,
                        stages,
                        successes_per_stage,
                        kept_states,
                        kept_actions,
                        kept_stages,
                        episode_offsets,
                        records,
                        complete=False,
                    )
                    if stage_successes >= successes_per_stage:
                        break
            if stage_successes < successes_per_stage:
                raise RuntimeError(
                    f"CACC produced {stage_successes}/{successes_per_stage} "
                    f"successful demonstrations at stage {stage}"
                )
    finally:
        env.close()

    metadata = _save_dataset(
        output,
        stages,
        successes_per_stage,
        kept_states,
        kept_actions,
        kept_stages,
        episode_offsets,
        records,
        complete=True,
    )
    print(
        json.dumps(
            {
                "dataset": str(output.resolve()),
                "transitions": metadata["transitions"],
                "complete": True,
            }
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/classical_comparison/assisted/cacc_teacher.pt"),
    )
    parser.add_argument("--stages", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--successes-per-stage", type=int, default=2)
    parser.add_argument("--maximum-attempts-per-stage", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    arguments = parser.parse_args()
    stages = tuple(int(value) for value in arguments.stages.split(","))
    if not stages or any(stage < 0 or stage > 7 for stage in stages):
        raise ValueError(f"invalid stages: {stages}")
    protocol = ComparisonProtocol(training_seed=arguments.seed)
    collect_teacher_data(
        arguments.output,
        stages,
        arguments.successes_per_stage,
        arguments.maximum_attempts_per_stage,
        protocol,
    )


if __name__ == "__main__":
    main()

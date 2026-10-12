"""Paired full-scenario evaluation for Graph, DDPG, PPO, and SAC."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from platoon_control.graph_rl import load_graph_controller
from platoon_control.rl import RLConfig, _curriculum_options
from platoon_sumo.run_demo import baseline_actions
from platoon_sumo.env import (
    FOLLOWER_IDS,
    PLATOON_IDS,
    REJOIN_GAP_TOLERANCE,
    TARGET_GAP,
    PlatoonParallelEnv,
)

from .algorithms import ClassicalAlgorithm, create_algorithm
from .common import (
    ActionCodec,
    ComparisonProtocol,
    FrozenRewardPredictor,
    StateAdapter,
    reward_fingerprint,
    shared_step_rewards,
)


def _protocol_from_checkpoint(checkpoint: dict[str, Any]) -> ComparisonProtocol:
    allowed = {field.name for field in fields(ComparisonProtocol)}
    values = {
        key: value
        for key, value in checkpoint["protocol"].items()
        if key in allowed
    }
    if "curriculum_boundaries" in values:
        values["curriculum_boundaries"] = tuple(values["curriculum_boundaries"])
    return ComparisonProtocol(**values)


class ClassicalPolicy:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.name = str(checkpoint["algorithm"])
        self.protocol = _protocol_from_checkpoint(checkpoint)
        if checkpoint["reward_fingerprint"] != reward_fingerprint():
            raise ValueError("classical checkpoint reward fingerprint is stale")
        self.algorithm: ClassicalAlgorithm = create_algorithm(
            self.name,
            int(checkpoint["state_dim"]),
            self.protocol,
            device,
            int(self.protocol.training_seed),
        )
        self.algorithm.load_state_dict(checkpoint["algorithm_state"])
        self.adapter = StateAdapter()
        self.codec = ActionCodec(
            self.protocol.action_min, self.protocol.action_max
        )
        self.previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
        self.total_steps = int(checkpoint["total_steps"])

    def reset(self) -> None:
        self.previous_applied.fill(0.0)

    def act(self, observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
        state = self.adapter(observations, self.previous_applied)
        sample = self.algorithm.act(
            state, deterministic=True, total_steps=self.total_steps
        )
        return self.codec.action_dict(sample.action)

    def record_applied(self, infos: dict[str, dict[str, Any]]) -> None:
        self.previous_applied = np.asarray(
            [
                infos[agent].get("applied_acceleration", 0.0)
                for agent in PLATOON_IDS
            ],
            np.float32,
        )


class GraphPolicy:
    name = "graph"

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        self.controller = load_graph_controller(checkpoint_path, device)

    def reset(self) -> None:
        self.controller.reset()

    def act(self, observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return self.controller.act(observations, deterministic=True)

    def record_applied(self, infos: dict[str, dict[str, Any]]) -> None:
        self.controller.record_applied_actions(infos)


class CaccPolicy:
    """Deterministic project CACC baseline under the shared evaluation protocol."""

    name = "cacc"

    def reset(self) -> None:
        return None

    def act(self, observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return baseline_actions(observations)

    def record_applied(self, infos: dict[str, dict[str, Any]]) -> None:
        return None


def _scenario(
    protocol: ComparisonProtocol, episode_index: int
) -> tuple[int, dict[str, Any], str]:
    reset_seed = protocol.evaluation_seed + 10_000 + episode_index
    rng = np.random.default_rng(
        protocol.evaluation_seed + 7_000 + episode_index
    )
    config = RLConfig(
        seed=protocol.evaluation_seed,
        episode_max_time=protocol.episode_max_time,
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
    )
    options, _ = _curriculum_options(7, rng, config, protocol.training_steps)
    serialized = json.dumps(options, sort_keys=True, default=float)
    return reset_seed, options, hashlib.sha256(serialized.encode()).hexdigest()


def _episode(
    policy: ClassicalPolicy | GraphPolicy | CaccPolicy,
    protocol: ComparisonProtocol,
    episode_index: int,
    predictor: FrozenRewardPredictor,
) -> dict[str, Any]:
    reset_seed, options, scenario_hash = _scenario(protocol, episode_index)
    env = PlatoonParallelEnv(
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
        max_time=protocol.episode_max_time,
    )
    observations, _ = env.reset(seed=reset_seed, options=options)
    policy.reset()
    raw_reward_sum = 0.0
    shaped_reward_sum = 0.0
    gaps: list[float] = []
    gap_errors: list[float] = []
    speed_spreads: list[float] = []
    applied: dict[str, list[float]] = {agent: [] for agent in PLATOON_IDS}
    shield: list[float] = []
    disconnected_steps = 0
    background_peak = 0
    background_slow_runs: dict[str, int] = {}
    maximum_background_slow_run = 0
    background_gridlock_run = 0
    maximum_background_gridlock_run = 0
    cutin_states: set[str] = set()
    maximum_delay = 0.0
    fuel_mg = 0.0
    inference_times: list[float] = []
    final_info: dict[str, Any] = {}
    episode_steps = 0
    try:
        while env.agents:
            predictor.attach(env, observations)
            started = time.perf_counter()
            actions = policy.act(observations)
            inference_times.append((time.perf_counter() - started) * 1_000.0)
            old_observations = observations
            observations, rewards, terminations, truncations, infos = env.step(actions)
            policy.record_applied(infos)
            raw_vector, shaped_vector = shared_step_rewards(
                old_observations, actions, rewards, infos
            )
            raw_reward_sum += float(raw_vector.mean())
            shaped_reward_sum += float(shaped_vector.mean())
            episode_steps += 1
            final_info = infos["leader"]
            fuel_mg += (
                float(final_info["reward_components"].get("fuel_mg_per_second", 0.0))
                * env.step_length
            )
            active = set(env.connection.vehicle.getIDList())
            step_gaps = [
                env._platoon_gap(follower, active)
                for follower in FOLLOWER_IDS
                if follower in active
            ]
            step_gaps = [gap for gap in step_gaps if gap >= 0.0]
            gaps.extend(step_gaps)
            gap_errors.extend(abs(gap - TARGET_GAP) for gap in step_gaps)
            if step_gaps and any(
                gap > TARGET_GAP + REJOIN_GAP_TOLERANCE for gap in step_gaps
            ):
                disconnected_steps += 1
            speeds = [
                float(env.connection.vehicle.getSpeed(agent))
                for agent in PLATOON_IDS
                if agent in active
            ]
            if speeds:
                speed_spreads.append(max(speeds) - min(speeds))
            for agent in PLATOON_IDS:
                if agent in infos:
                    applied[agent].append(
                        float(infos[agent].get("applied_acceleration", 0.0))
                    )
                    shield.append(float(infos[agent].get("safety_intervened", False)))
            background = sorted(
                vehicle for vehicle in active if vehicle.startswith("background_")
            )
            background_peak = max(background_peak, len(background))
            background_speeds = [
                float(env.connection.vehicle.getSpeed(vehicle))
                for vehicle in background
            ]
            for vehicle, speed in zip(background, background_speeds):
                run = (
                    background_slow_runs.get(vehicle, 0) + 1
                    if speed < 0.2
                    else 0
                )
                background_slow_runs[vehicle] = run
                maximum_background_slow_run = max(maximum_background_slow_run, run)
            gridlocked = (
                len(background_speeds) >= 8
                and float(np.mean(background_speeds)) < 0.5
            )
            background_gridlock_run = (
                background_gridlock_run + 1 if gridlocked else 0
            )
            maximum_background_gridlock_run = max(
                maximum_background_gridlock_run, background_gridlock_run
            )
            cutin_states.add(str(final_info.get("cutin_state", "idle")))
            for follower in FOLLOWER_IDS:
                maximum_delay = max(
                    maximum_delay,
                    float(infos[follower].get("communication_delay", 0.0)),
                    float(
                        infos[follower].get(
                            "predecessor_communication_delay", 0.0
                        )
                    ),
                )
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        events = env.event_log
        leader_station = float(final_info.get("leader_station", env._station("leader")))
        env.close()
    acceleration_rms = {
        agent: float(np.sqrt(np.mean(np.square(values)))) if values else 0.0
        for agent, values in applied.items()
    }
    leader_rms = max(acceleration_rms["leader"], 1e-3)
    follower_rms = max(acceleration_rms[agent] for agent in FOLLOWER_IDS)
    jerks = np.concatenate(
        [
            np.diff(np.asarray(values, np.float32)) / 0.1
            for values in applied.values()
            if len(values) > 1
        ]
        or [np.zeros(1, np.float32)]
    )
    event_names = {event["event"] for event in events}
    cutin_resolved = bool(
        {"cutin_blocked", "cutin_success", "cutin_aborted"} & event_names
    )
    return {
        "controller": policy.name,
        "episode_index": episode_index,
        "reset_seed": reset_seed,
        "scenario_hash": scenario_hash,
        "steps": episode_steps,
        "success": int(final_info.get("success", False)),
        "collision": int(final_info.get("collision", False)),
        "failure": int(final_info.get("failure", False)),
        "raw_team_reward_per_step": raw_reward_sum / max(episode_steps, 1),
        "shared_training_reward_per_step": shaped_reward_sum / max(episode_steps, 1),
        "leader_station_m": leader_station,
        "minimum_gap_m": float(min(gaps, default=-1.0)),
        "maximum_gap_m": float(max(gaps, default=-1.0)),
        "mean_absolute_gap_error_m": float(np.mean(gap_errors or [0.0])),
        "maximum_speed_spread_mps": float(max(speed_spreads, default=0.0)),
        "mean_speed_spread_mps": float(np.mean(speed_spreads or [0.0])),
        "string_amplification": follower_rms / leader_rms,
        "jerk_rms_mps3": float(np.sqrt(np.mean(np.square(jerks)))),
        "shield_intervention_rate": float(np.mean(shield or [0.0])),
        "disconnected_seconds": disconnected_steps * 0.1,
        "fuel_mg": fuel_mg,
        "background_peak_count": background_peak,
        "background_max_stuck_seconds": maximum_background_slow_run * 0.1,
        "background_max_gridlock_seconds": maximum_background_gridlock_run * 0.1,
        "maximum_observed_delay_seconds": maximum_delay,
        "cutin_exposed": int(any(state != "idle" for state in cutin_states)),
        "cutin_resolved": int(cutin_resolved),
        "inference_p95_ms": float(np.percentile(inference_times, 95)),
    }


def evaluate(
    controller: str,
    checkpoint: Path,
    output: Path,
    protocol: ComparisonProtocol,
    device: str,
    episode_start: int = 0,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    policy: ClassicalPolicy | GraphPolicy | CaccPolicy
    if controller == "graph":
        policy = GraphPolicy(checkpoint, device)
    elif controller == "cacc":
        policy = CaccPolicy()
    else:
        policy = ClassicalPolicy(checkpoint, device)
    if policy.name != controller:
        raise ValueError(f"loaded {policy.name}, expected {controller}")
    predictor = FrozenRewardPredictor(protocol.reward_predictor_path, device)
    rows: list[dict[str, Any]] = []
    for episode_index in range(
        episode_start, episode_start + protocol.evaluation_episodes
    ):
        row = _episode(policy, protocol, episode_index, predictor)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return write_evaluation_rows(
        controller, checkpoint, output, protocol, rows
    )


def write_evaluation_rows(
    controller: str,
    checkpoint: Path,
    output: Path,
    protocol: ComparisonProtocol,
    rows: list[dict[str, Any]],
) -> Path:
    """Write one validated set of episode rows and its numeric summary."""

    if not rows:
        raise ValueError("evaluation requires at least one episode row")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
        and key not in ("episode_index", "reset_seed")
    ]
    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "controller": controller,
        "checkpoint": str(checkpoint.resolve()),
        "protocol": protocol.to_dict(),
        "reward_fingerprint": reward_fingerprint(),
        "episodes": len(rows),
        "scenario_hashes": [row["scenario_hash"] for row in rows],
        "metrics": {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows])),
            }
            for metric in numeric
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "controller", choices=("graph", "cacc", "ddpg", "ppo", "sac")
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=30_000)
    arguments = parser.parse_args()
    protocol = replace(
        ComparisonProtocol(),
        evaluation_episodes=arguments.episodes,
        evaluation_seed=arguments.evaluation_seed,
    )
    path = evaluate(
        arguments.controller,
        arguments.checkpoint,
        arguments.output,
        protocol,
        arguments.device,
        arguments.episode_start,
    )
    print(json.dumps({"summary": str(path), "complete": True}))


if __name__ == "__main__":
    main()

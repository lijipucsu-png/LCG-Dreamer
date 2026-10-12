"""Paired mixed-traffic repeated lane-change evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv

from .algorithms import load_native_algorithm
from .policy import (
    CaccLcPolicy,
    GraphMixedLanePolicy,
    NativeMixedLanePolicy,
    create_base_policy,
    graph_adapter,
)
from .protocol import METHODS, MixedLaneProtocol


def _load_policy(method: str, protocol: MixedLaneProtocol, device: str) -> Any:
    base = create_base_policy(method, protocol, device)
    if method == "cacc_lc":
        return CaccLcPolicy(base)
    checkpoint_path = protocol.lateral_checkpoint(method)
    assert checkpoint_path is not None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["method"] != method:
        raise ValueError(f"lateral checkpoint is {payload['method']}, not {method}")
    if payload["protocol"]["fingerprint"] != protocol.to_dict()["fingerprint"]:
        raise ValueError("mixed-lane protocol changed since training")
    if method == "graph":
        model = graph_adapter(device)
        model.load_state_dict(payload["lateral_state"])
        model.eval()
        return GraphMixedLanePolicy(base, model, device, deterministic=True)
    algorithm = load_native_algorithm(
        method, protocol, device, payload["lateral_state"]
    )
    return NativeMixedLanePolicy(base, algorithm, deterministic=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _episode(
    method: str,
    policy: Any,
    protocol: MixedLaneProtocol,
    episode_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = protocol.seed + 100_000 + episode_index
    options = protocol.scenario_options(seed, training=False)
    env = PlatoonParallelEnv(max_time=protocol.max_time)
    observations, _ = env.reset(seed=seed, options=options)
    policy.reset()
    traces: list[dict[str, Any]] = []
    minimum_gap = float("inf")
    minimum_same_lane_gap = float("inf")
    minimum_same_lane_ttc = float("inf")
    maximum_gap = 0.0
    maximum_speed_difference = 0.0
    squared_jerks: list[float] = []
    previous_acceleration: np.ndarray | None = None
    fuel_kg = 0.0
    platoon_speed_integral = 0.0
    rejected_commands = 0
    first_applied_lane_command: dict[tuple[int, str], float] = {}
    final_info: dict[str, Any] = {}
    try:
        while env.agents:
            state = np.asarray(
                observations["leader"]["platoon_state"], np.float32
            )
            gaps = state[1:, 4]
            valid_gaps = gaps[gaps > 0.0]
            if len(valid_gaps):
                minimum_gap = min(minimum_gap, float(np.min(valid_gaps)))
                maximum_gap = max(maximum_gap, float(np.max(valid_gaps)))
            speed = np.asarray(
                [observations[agent]["ego"][1] for agent in PLATOON_IDS],
                np.float64,
            )
            acceleration = np.asarray(
                [observations[agent]["ego"][2] for agent in PLATOON_IDS],
                np.float64,
            )
            if previous_acceleration is not None:
                jerk = (acceleration - previous_acceleration) / env.step_length
                squared_jerks.extend(np.square(jerk).tolist())
            previous_acceleration = acceleration
            maximum_speed_difference = max(
                maximum_speed_difference, float(np.ptp(speed))
            )
            lanes = np.rint(state[:, 3]).astype(np.int64)
            step_same_lane_gaps: list[float] = []
            step_same_lane_ttcs: list[float] = []
            for follower_index in range(1, len(PLATOON_IDS)):
                if lanes[follower_index] != lanes[follower_index - 1]:
                    continue
                pair_gap = max(0.0, float(state[follower_index, 4]))
                step_same_lane_gaps.append(pair_gap)
                closing_speed = max(
                    0.0, float(speed[follower_index] - speed[follower_index - 1])
                )
                if closing_speed > 1.0e-3:
                    step_same_lane_ttcs.append(pair_gap / closing_speed)
            if step_same_lane_gaps:
                minimum_same_lane_gap = min(
                    minimum_same_lane_gap, min(step_same_lane_gaps)
                )
            if step_same_lane_ttcs:
                minimum_same_lane_ttc = min(
                    minimum_same_lane_ttc, min(step_same_lane_ttcs)
                )
            platoon_speed_integral += float(np.mean(speed)) * env.step_length
            active = set(env.connection.vehicle.getIDList())
            social = [
                vehicle_id
                for vehicle_id in env._lane_change_traffic_ids
                if vehicle_id in active
            ]
            social_speeds = [
                float(env.connection.vehicle.getSpeed(vehicle_id))
                for vehicle_id in social
            ]
            social_accelerations = [
                float(env.connection.vehicle.getAcceleration(vehicle_id))
                for vehicle_id in social
            ]
            row: dict[str, Any] = {
                "method": method,
                "episode": episode_index,
                "time_s": float(env.sim_time),
                "event_index": int(
                    observations["leader"]["lane_mission"][1] * 3.0 + 1e-6
                ),
                "event_active": int(
                    observations["leader"]["lane_mission"][8] > 0.5
                ),
                "formation_length_m": float(-state[-1, 0] + 12.0),
                "mean_gap_m": float(np.mean(valid_gaps)) if len(valid_gaps) else np.nan,
                "speed_difference_mps": float(np.ptp(speed)),
                "same_lane_minimum_gap_m": float(
                    min(step_same_lane_gaps) if step_same_lane_gaps else np.nan
                ),
                "same_lane_minimum_ttc_s": float(
                    min(step_same_lane_ttcs) if step_same_lane_ttcs else np.nan
                ),
                "social_vehicle_count": len(social),
                "social_mean_speed_mps": float(np.mean(social_speeds)) if social_speeds else np.nan,
                "social_min_acceleration_mps2": float(np.min(social_accelerations)) if social_accelerations else np.nan,
            }
            for index, agent in enumerate(PLATOON_IDS):
                ego = np.asarray(observations[agent]["ego"], np.float32)
                row[f"{agent}_station_m"] = float(ego[0])
                row[f"{agent}_speed_mps"] = float(ego[1])
                row[f"{agent}_acceleration_mps2"] = float(ego[2])
                row[f"{agent}_lane"] = int(round(float(ego[4])))
            for gate_index in range(1, 4):
                for role in ("front", "front_queue", "rear", "rear_queue"):
                    vehicle_id = f"lane_change_traffic_{gate_index}_{role}"
                    prefix = f"traffic_{gate_index}_{role}"
                    if vehicle_id in active:
                        row[f"{prefix}_station_m"] = float(
                            env._station(vehicle_id)
                        )
                        row[f"{prefix}_speed_mps"] = float(
                            env.connection.vehicle.getSpeed(vehicle_id)
                        )
                        row[f"{prefix}_acceleration_mps2"] = float(
                            env.connection.vehicle.getAcceleration(vehicle_id)
                        )
                        row[f"{prefix}_lane"] = int(
                            env.connection.vehicle.getLaneIndex(vehicle_id)
                        )
                    else:
                        row[f"{prefix}_station_m"] = np.nan
                        row[f"{prefix}_speed_mps"] = np.nan
                        row[f"{prefix}_acceleration_mps2"] = np.nan
                        row[f"{prefix}_lane"] = np.nan
            traces.append(row)
            actions = policy.act(observations)
            command_event = int(
                observations["leader"]["lane_mission"][1] * 3.0 + 1e-6
            )
            observations, _, terminations, truncations, infos = env.step(actions)
            policy.record_applied(infos)
            final_info = infos["leader"]
            for agent in PLATOON_IDS:
                if (
                    int(actions[agent].get("lane_change", 0)) != 0
                    and not bool(infos[agent].get("lane_change_rejected", False))
                ):
                    first_applied_lane_command.setdefault(
                        (command_event, agent), float(final_info["time"])
                    )
            rejected_commands += sum(
                int(bool(infos[agent].get("lane_change_rejected", False)))
                for agent in PLATOON_IDS
            )
            fuel_kg += (
                float(
                    final_info["reward_components"].get(
                        "fuel_mg_per_second", 0.0
                    )
                )
                * env.step_length
                / 1.0e6
            )
            if any(terminations.values()) or any(truncations.values()):
                break
        completions = [
            item
            for item in env.event_log
            if item.get("event") == "cooperative_lane_change_completed"
        ]
        event_durations = [float(item["duration_s"]) for item in completions]
        execution_durations = [
            float(item["execution_duration_s"]) for item in completions
        ]
        command_spreads = []
        for event_index in range(3):
            command_times = [
                first_applied_lane_command[(event_index, agent)]
                for agent in PLATOON_IDS
                if (event_index, agent) in first_applied_lane_command
            ]
            if len(command_times) == len(PLATOON_IDS):
                command_spreads.append(max(command_times) - min(command_times))
            else:
                # Failure-aware censoring: a missing truck command must not be
                # reported as perfect zero-spread coordination.
                command_spreads.append(protocol.max_time)
        result = {
            "method": method,
            "episode": episode_index,
            "seed": seed,
            "traffic_intensity": float(options["mixed_lane_intensity"]),
            "success": int(bool(final_info.get("success", False))),
            "collision": int(bool(final_info.get("collision", False))),
            "failure": int(bool(final_info.get("failure", False))),
            "maneuvers_completed": int(
                final_info.get("lane_change_maneuvers_completed", 0)
            ),
            "completion_ratio": float(
                final_info.get("lane_change_maneuvers_completed", 0) / 3.0
            ),
            "travel_time_s": float(final_info.get("time", env.sim_time)),
            "maneuver_time_s": float(sum(event_durations)),
            "lane_execution_time_s": float(sum(execution_durations)),
            "gap_acceptance_wait_s": float(
                max(0.0, sum(event_durations) - sum(execution_durations))
            ),
            "minimum_gap_m": float(
                minimum_gap if np.isfinite(minimum_gap) else 0.0
            ),
            "minimum_same_lane_gap_m": float(
                minimum_same_lane_gap
                if np.isfinite(minimum_same_lane_gap)
                else 500.0
            ),
            "minimum_same_lane_ttc_s": float(
                minimum_same_lane_ttc
                if np.isfinite(minimum_same_lane_ttc)
                else 100.0
            ),
            "maximum_gap_m": float(maximum_gap),
            "maximum_speed_difference_mps": float(maximum_speed_difference),
            "jerk_rms_mps3": float(
                np.sqrt(np.mean(squared_jerks)) if squared_jerks else 0.0
            ),
            "mean_lane_command_spread_s": float(
                np.mean(command_spreads) if command_spreads else 0.0
            ),
            "maximum_lane_command_spread_s": float(
                max(command_spreads) if command_spreads else 0.0
            ),
            "fuel_kg": float(fuel_kg),
            "platoon_average_speed_mps": float(
                platoon_speed_integral / max(float(final_info.get("time", env.sim_time)), env.step_length)
            ),
            "traffic_delay_s": float(
                final_info.get("lane_change_traffic_delay_s", 0.0)
            ),
            "traffic_braking_integral": float(
                final_info.get("lane_change_traffic_braking_integral", 0.0)
            ),
            "traffic_hard_brake_events": int(
                final_info.get("lane_change_traffic_hard_brake_events", 0)
            ),
            "traffic_max_deceleration_mps2": float(
                final_info.get("lane_change_traffic_max_deceleration", 0.0)
            ),
            "traffic_spawned_count": int(
                final_info.get("lane_change_traffic_spawned_count", 0)
            ),
            "rejected_lane_commands": int(rejected_commands),
            "communication_delay_events": int(
                final_info.get("communication_delay_events", 0)
            ),
            "leader_station_m": float(final_info.get("leader_station", 0.0)),
        }
        numeric = np.asarray(
            [
                value
                for value in result.values()
                if isinstance(value, (float, int))
            ],
            np.float64,
        )
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"non-finite evaluation result: {result}")
        return result, traces
    finally:
        env.close()


def evaluate(
    method: str,
    protocol: MixedLaneProtocol,
    device: str,
    resume: bool = False,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"unknown mixed-lane method: {method}")
    policy = _load_policy(method, protocol, device)
    output = Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv"
    rows: list[dict[str, Any]] = (
        _read_csv(output) if resume and output.exists() else []
    )
    completed_episodes = [int(row["episode"]) for row in rows]
    if completed_episodes != list(range(len(rows))):
        raise ValueError(
            f"cannot resume non-contiguous {method} episodes: {completed_episodes}"
        )
    if len(rows) > protocol.evaluation_episodes:
        raise ValueError(f"{method} has too many saved evaluation rows")
    trace_root = Path(protocol.output_root) / "evaluation" / "traces"
    for episode in range(len(rows), protocol.evaluation_episodes):
        result, trace = _episode(method, policy, protocol, episode)
        rows.append(result)
        _write_csv(trace_root / f"{method}_episode_{episode:03d}.csv", trace)
        _write_csv(output, rows)
        print(json.dumps(result, sort_keys=True), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    protocol = MixedLaneProtocol()
    if arguments.episodes is not None:
        protocol = replace(protocol, evaluation_episodes=arguments.episodes)
    output = evaluate(
        arguments.method,
        protocol,
        arguments.device,
        resume=arguments.resume,
    )
    print(json.dumps({"complete": True, "output": str(output)}))


if __name__ == "__main__":
    main()

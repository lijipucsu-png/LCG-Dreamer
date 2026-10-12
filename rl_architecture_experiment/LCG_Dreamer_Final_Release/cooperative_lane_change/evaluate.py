"""Matched 20-scenario evaluation for cooperative multi-lane changes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv

from .policy import CooperativeLanePolicy, create_base_policy, load_adapter
from .protocol import METHODS, LaneChangeProtocol


def _episode(
    method: str,
    policy: CooperativeLanePolicy,
    protocol: LaneChangeProtocol,
    episode_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = protocol.seed + 100_000 + episode_index
    options = protocol.scenario_options(seed, training=False)
    env = PlatoonParallelEnv(max_time=protocol.max_time)
    observations, _ = env.reset(seed=seed, options=options)
    policy.reset()
    trace_rows: list[dict[str, Any]] = []
    minimum_gap = float("inf")
    maximum_gap = 0.0
    maximum_speed_difference = 0.0
    squared_jerks: list[float] = []
    speeds_all: list[float] = []
    fuel_kg = 0.0
    previous_acceleration = np.zeros(len(PLATOON_IDS), np.float64)
    final_info: dict[str, Any] = {}
    try:
        while env.agents:
            time_s = float(observations["leader"]["ego"][0] * 0.0 + env.sim_time)
            platoon_state = np.asarray(
                observations["leader"]["platoon_state"], np.float32
            )
            gaps = platoon_state[1:, 4]
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
            jerk = (acceleration - previous_acceleration) / env.step_length
            previous_acceleration = acceleration
            squared_jerks.extend(jerk[1:] ** 2)
            speeds_all.extend(speed.tolist())
            maximum_speed_difference = max(
                maximum_speed_difference, float(np.ptp(speed))
            )
            row: dict[str, Any] = {
                "method": method,
                "episode": episode_index,
                "time_s": time_s,
                "event_index": int(
                    observations["leader"]["lane_mission"][1] * 3.0 + 1e-6
                ),
                "event_active": int(
                    observations["leader"]["lane_mission"][8] > 0.5
                ),
                "formation_length_m": float(
                    -observations["leader"]["platoon_state"][-1, 0] + 12.0
                ),
                "mean_gap_m": float(np.mean(valid_gaps)) if len(valid_gaps) else np.nan,
                "speed_difference_mps": float(np.ptp(speed)),
            }
            for index, agent in enumerate(PLATOON_IDS):
                ego = np.asarray(observations[agent]["ego"], np.float32)
                row[f"{agent}_station_m"] = float(ego[0])
                row[f"{agent}_speed_mps"] = float(ego[1])
                row[f"{agent}_acceleration_mps2"] = float(ego[2])
                row[f"{agent}_lane"] = int(round(float(ego[4])))
            trace_rows.append(row)
            actions = policy.act(observations)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            policy.record_applied(infos)
            final_info = infos["leader"]
            fuel_kg += (
                float(
                    final_info["reward_components"].get(
                        "fuel_mg_per_second", 0.0
                    )
                )
                * env.step_length
                / 1.0e6
            )
        event_durations = [
            float(item["duration_s"])
            for item in env.event_log
            if item.get("event") == "cooperative_lane_change_completed"
        ]
        row = {
            "method": method,
            "episode": episode_index,
            "seed": seed,
            "obstacle_speed_mps": float(options["lane_change_obstacle_speed_mps"]),
            "success": int(bool(final_info.get("success", False))),
            "collision": int(bool(final_info.get("collision", False))),
            "maneuvers_completed": int(
                final_info.get("lane_change_maneuvers_completed", 0)
            ),
            "completion_ratio": float(
                final_info.get("lane_change_maneuvers_completed", 0) / 3.0
            ),
            "travel_time_s": float(final_info.get("time", env.sim_time)),
            "maneuver_time_s": float(sum(event_durations)),
            "minimum_gap_m": float(minimum_gap if np.isfinite(minimum_gap) else 0.0),
            "maximum_gap_m": float(maximum_gap),
            "maximum_speed_difference_mps": float(maximum_speed_difference),
            "jerk_rms_mps3": float(np.sqrt(np.mean(squared_jerks))),
            "average_speed_mps": float(np.mean(speeds_all)),
            "fuel_kg": float(fuel_kg),
            "dual_lane_occupancy_s": float(
                final_info.get("lane_change_dual_lane_time", 0.0)
            ),
            "lane_execution_time_s": float(
                final_info.get("lane_change_execution_time", 0.0)
            ),
            "lane_commands": int(
                final_info.get("lane_change_total_commands", 0)
            ),
            "leader_station_m": float(final_info.get("leader_station", 0.0)),
        }
        return row, trace_rows
    finally:
        env.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    method: str, protocol: LaneChangeProtocol, device: str
) -> Path:
    base = create_base_policy(method, protocol, device)
    adapter = load_adapter(protocol.adapter_checkpoint(method), device)
    adapter.eval()
    policy = CooperativeLanePolicy(
        base, adapter, device, deterministic=True, expert=False
    )
    rows: list[dict[str, Any]] = []
    trace_directory = Path(protocol.output_root) / "evaluation" / "traces"
    for episode_index in range(protocol.evaluation_episodes):
        row, trace = _episode(method, policy, protocol, episode_index)
        rows.append(row)
        _write_csv(
            trace_directory / f"{method}_episode_{episode_index:03d}.csv",
            trace,
        )
        print(json.dumps(row, sort_keys=True), flush=True)
    output = Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv"
    _write_csv(output, rows)
    summary = {
        "method": method,
        "episodes": len(rows),
        "success_rate": float(np.mean([item["success"] for item in rows])),
        "collision_rate": float(np.mean([item["collision"] for item in rows])),
        "mean_completion_ratio": float(
            np.mean([item["completion_ratio"] for item in rows])
        ),
        "mean_travel_time_s": float(
            np.mean([item["travel_time_s"] for item in rows])
        ),
        "mean_maneuver_time_s": float(
            np.mean([item["maneuver_time_s"] for item in rows])
        ),
        "mean_jerk_rms_mps3": float(
            np.mean([item["jerk_rms_mps3"] for item in rows])
        ),
        "mean_average_speed_mps": float(
            np.mean([item["average_speed_mps"] for item in rows])
        ),
        "mean_fuel_kg": float(np.mean([item["fuel_kg"] for item in rows])),
        "mean_lane_execution_time_s": float(
            np.mean([item["lane_execution_time_s"] for item in rows])
        ),
        "mean_maximum_speed_difference_mps": float(
            np.mean([item["maximum_speed_difference_mps"] for item in rows])
        ),
    }
    summary_path = output.with_name(f"{method}_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int)
    arguments = parser.parse_args()
    protocol = LaneChangeProtocol()
    if arguments.episodes is not None:
        protocol = replace(protocol, evaluation_episodes=arguments.episodes)
    output = evaluate(arguments.method, protocol, arguments.device)
    print(json.dumps({"complete": True, "output": str(output)}))


if __name__ == "__main__":
    main()

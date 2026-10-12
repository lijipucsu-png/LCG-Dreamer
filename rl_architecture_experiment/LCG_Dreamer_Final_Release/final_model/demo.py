"""Headless or SUMO-GUI demonstration of the released controller."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mixed_traffic_lane_change.protocol import MixedLaneProtocol
from platoon_control.rl import RLConfig, _curriculum_options
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv

from .runtime import ReleasePaths, load_final_policy


def _scenario_options(name: str, seed: int, paths: ReleasePaths) -> tuple[dict[str, Any], float]:
    if name == "lane-change":
        protocol = MixedLaneProtocol()
        return protocol.scenario_options(seed, training=False), protocol.max_time
    if name == "following":
        return {
            "cutin_enabled": False,
            "v2i_enabled": False,
            "wave_mode": "off",
            "front_vehicle_count": 1,
            "background_initial_scale": 0.0,
            "background_demand_scale": 0.0,
            "communication_delay_probability": 0.0,
            "speed_reference": 25.0,
        }, 60.0
    config = RLConfig(
        seed=seed,
        calibration_path=str(paths.calibration),
        wave_model_path=str(paths.wave_model),
        episode_max_time=240.0,
    )
    options, _ = _curriculum_options(
        7, np.random.default_rng(seed + 7_000), config, 301_144
    )
    return options, config.episode_max_time


def _write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(
    scenario: str,
    seed: int,
    device: str,
    gui: bool = False,
    max_steps: int | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    paths = ReleasePaths.discover()
    options, scenario_max_time = _scenario_options(scenario, seed, paths)
    policy = load_final_policy(device=device, enable_lane_change=True)
    env = PlatoonParallelEnv(
        render_mode="human" if gui else None,
        max_time=scenario_max_time,
        calibration_path=paths.calibration,
        wave_model_path=paths.wave_model,
        gui_auto_start=gui,
    )
    traces: list[dict[str, Any]] = []
    minimum_gap = float("inf")
    maximum_gap = 0.0
    maximum_speed_difference = 0.0
    squared_jerks: list[float] = []
    previous_acceleration: np.ndarray | None = None
    shield_events = 0
    final_info: dict[str, Any] = {}
    step = 0
    completed_naturally = False
    trace_vehicle_ids = [
        f"lane_change_obstacle_{index}" for index in range(1, 4)
    ] + [
        f"lane_change_traffic_{event}_{role}"
        for event in range(1, 4)
        for role in ("front", "front_queue", "rear", "rear_queue")
    ]
    try:
        observations, _ = env.reset(seed=seed, options=options)
        policy.reset()
        while env.agents:
            state = np.asarray(observations["leader"]["platoon_state"], np.float32)
            gaps = state[1:, 4]
            valid_gaps = gaps[gaps > 0.0]
            if len(valid_gaps):
                minimum_gap = min(minimum_gap, float(np.min(valid_gaps)))
                maximum_gap = max(maximum_gap, float(np.max(valid_gaps)))
            speeds = np.asarray(
                [observations[agent]["ego"][1] for agent in PLATOON_IDS],
                np.float64,
            )
            accelerations = np.asarray(
                [observations[agent]["ego"][2] for agent in PLATOON_IDS],
                np.float64,
            )
            if previous_acceleration is not None:
                squared_jerks.extend(
                    np.square((accelerations - previous_acceleration) / env.step_length).tolist()
                )
            previous_acceleration = accelerations
            maximum_speed_difference = max(
                maximum_speed_difference, float(np.ptp(speeds))
            )
            row: dict[str, Any] = {
                "time_s": float(env.sim_time),
                "mean_gap_m": float(np.mean(valid_gaps)) if len(valid_gaps) else np.nan,
                "speed_difference_mps": float(np.ptp(speeds)),
                "event_index": int(
                    observations["leader"]["lane_mission"][1] * 3.0 + 1e-6
                ),
                "event_active": int(
                    observations["leader"]["lane_mission"][8] > 0.5
                ),
                "target_lane": int(
                    round(float(observations["leader"]["lane_mission"][2] * 2.0))
                ),
                "obstacle_distance_m": float(
                    observations["leader"]["lane_mission"][4] * 200.0
                ),
            }
            for index, agent in enumerate(PLATOON_IDS):
                mission = np.asarray(observations[agent]["lane_mission"], np.float32)
                row[f"{agent}_station_m"] = float(
                    observations[agent]["ego"][0]
                )
                row[f"{agent}_speed_mps"] = float(speeds[index])
                row[f"{agent}_acceleration_mps2"] = float(accelerations[index])
                row[f"{agent}_lane"] = int(round(float(state[index, 3])))
                row[f"{agent}_target_lane"] = int(round(float(mission[2] * 2.0)))
                if index > 0:
                    row[f"{agent}_local_gap_m"] = float(state[index, 4])
            actions = policy.act(observations)
            lane_trace = getattr(policy.policy, "last_trace", None)
            target_gaps = (
                np.asarray(lane_trace.target_gaps, np.float32)
                if lane_trace is not None
                else np.full(5, np.nan, np.float32)
            )
            for index, agent in enumerate(PLATOON_IDS):
                row[f"{agent}_requested_lane_change"] = int(
                    actions[agent].get("lane_change", 0)
                )
                row[f"{agent}_target_gap_m"] = float(
                    target_gaps[index - 1] if index > 0 else np.nan
                )
            observations, _, terminations, truncations, infos = env.step(actions)
            policy.record_applied(infos)
            final_info = dict(infos["leader"])
            row["event_index_after"] = int(
                final_info.get("lane_change_event_index", 0)
            )
            row["event_active_after"] = int(
                bool(final_info.get("lane_change_event_active", False))
            )
            row["maneuvers_completed"] = int(
                final_info.get("lane_change_maneuvers_completed", 0)
            )
            for agent in PLATOON_IDS:
                agent_info = infos[agent]
                row[f"{agent}_lane_change_applied"] = int(
                    bool(agent_info.get("lane_change_applied", False))
                )
                row[f"{agent}_lane_change_rejected"] = int(
                    bool(agent_info.get("lane_change_rejected", False))
                )
                row[f"{agent}_accepted_lane_change"] = int(
                    actions[agent].get("lane_change", 0)
                    if agent_info.get("lane_change_applied", False)
                    else 0
                )
                for metric in (
                    "target_front_gap",
                    "target_rear_gap",
                    "target_front_ttc",
                    "target_rear_ttc",
                ):
                    value = agent_info.get(metric, np.nan)
                    row[f"{agent}_{metric}"] = float(value)
            active_ids = set(env.connection.vehicle.getIDList())
            for vehicle_id in trace_vehicle_ids:
                prefix = vehicle_id.replace("-", "_")
                if vehicle_id in active_ids:
                    row[f"{prefix}_station_m"] = float(env._station(vehicle_id))
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
            shield_events += sum(
                int(bool(infos[agent].get("shield_triggered", False)))
                for agent in PLATOON_IDS
            )
            step += 1
            if any(terminations.values()) or any(truncations.values()):
                completed_naturally = True
                break
            if max_steps is not None and step >= max_steps:
                break
    finally:
        env.close()
    if trace_path is not None:
        _write_trace(trace_path, traces)
    return {
        "scenario": scenario,
        "seed": seed,
        "device": str(policy.device),
        "steps": step,
        "sim_time_s": float(final_info.get("time", env.sim_time)),
        "completed_naturally": completed_naturally,
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "failure": bool(final_info.get("failure", False)),
        "lane_change_maneuvers_completed": int(
            final_info.get("lane_change_maneuvers_completed", 0)
        ),
        "minimum_gap_m": float(minimum_gap if np.isfinite(minimum_gap) else 0.0),
        "maximum_gap_m": maximum_gap,
        "maximum_speed_difference_mps": maximum_speed_difference,
        "jerk_rms_mps3": float(
            np.sqrt(np.mean(squared_jerks)) if squared_jerks else 0.0
        ),
        "shield_events": shield_events,
        "communication_delay_events": int(
            final_info.get("communication_delay_events", 0)
        ),
        "trace": str(trace_path) if trace_path is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=("following", "full", "lane-change"), default="full"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=173_100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--no-trace", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in range(arguments.episodes):
        trace = None
        if not arguments.no_trace:
            trace = output / f"{arguments.scenario}_episode_{episode:03d}.csv"
        result = run_episode(
            arguments.scenario,
            arguments.seed + episode,
            arguments.device,
            gui=arguments.gui,
            max_steps=arguments.max_steps,
            trace_path=trace,
        )
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = output / "summary.json"
    summary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": str(summary.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

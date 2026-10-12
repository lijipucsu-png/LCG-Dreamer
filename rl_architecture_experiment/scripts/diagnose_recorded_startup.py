"""Trace startup semantics in the exact scenario used for paired recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from platoon_control.graph_rl import GraphRLConfig, load_graph_controller
from platoon_control.rl import _attach_follower_predictions
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv
from scripts.record_paired_final_scenario import build_final_scenario


def _longest_clear_stall(samples: list[dict[str, Any]]) -> dict[str, Any]:
    longest: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for sample in samples:
        clear_to_restart = bool(
            sample["leader_speed"] <= 0.3
            and sample["front_speed"] >= 1.0
            and sample["front_gap"] >= 5.0
        )
        if clear_to_restart:
            current.append(sample)
            if len(current) > len(longest):
                longest = current.copy()
        else:
            current = []
    if not longest:
        return {"duration_seconds": 0.0, "samples": 0}
    return {
        "duration_seconds": 0.1 * len(longest),
        "samples": len(longest),
        "start_time": longest[0]["time"],
        "end_time": longest[-1]["time"],
        "start_front_gap": longest[0]["front_gap"],
        "end_front_gap": longest[-1]["front_gap"],
        "maximum_front_gap": max(item["front_gap"] for item in longest),
        "mean_leader_raw": float(
            np.mean([item["raw_actions"][0] for item in longest])
        ),
        "mean_leader_safe": float(
            np.mean([item["safe_actions"][0] for item in longest])
        ),
        "mean_leader_applied": float(
            np.mean([item["applied_actions"][0] for item in longest])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument("--training-step", type=int, default=301_144)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    options, reset_seed, scenario_rng_seed = build_final_scenario(
        arguments.seed,
        arguments.episode,
        arguments.stage,
        arguments.training_step,
    )
    checkpoint = torch.load(
        arguments.checkpoint, map_location="cpu", weights_only=False
    )
    config = GraphRLConfig(**checkpoint["graph_rl_config"])
    controller = load_graph_controller(arguments.checkpoint, arguments.device)
    env = PlatoonParallelEnv(
        calibration_path=config.calibration_path,
        wave_model_path=config.wave_model_path,
        max_time=arguments.max_time,
    )
    observations, _ = env.reset(seed=reset_seed, options=options)
    controller.reset()
    samples: list[dict[str, Any]] = []
    try:
        while env.agents:
            active = set(env.connection.vehicle.getIDList())
            speeds = [
                float(env.connection.vehicle.getSpeed(agent))
                for agent in PLATOON_IDS
            ]
            front = env.connection.vehicle.getLeader("leader", 500.0)
            if front and front[0] in active:
                front_id = str(front[0])
                front_gap = float(front[1])
                front_speed = float(env.connection.vehicle.getSpeed(front_id))
                front_acceleration = float(
                    env.connection.vehicle.getAcceleration(front_id)
                )
                front_road = str(env.connection.vehicle.getRoadID(front_id))
                front_lane = int(env.connection.vehicle.getLaneIndex(front_id))
                front_station = float(env._station(front_id))
                front_is_ramp_origin = front_id in env._ramp_origin_ids
            else:
                front_id = ""
                front_gap = 500.0
                front_speed = 0.0
                front_acceleration = 0.0
                front_road = ""
                front_lane = -1
                front_station = 0.0
                front_is_ramp_origin = False

            with torch.no_grad():
                policy_states = controller.follower_policy_states(observations)[0]
            _attach_follower_predictions(env, policy_states)
            actions = controller.act(
                observations, deterministic=True, policy_states=policy_states
            )
            raw, safe = controller.requested_action_triplet()
            compact = controller.replay_observation()
            observations, _, terminations, truncations, infos = env.step(actions)
            applied = [
                float(infos[agent]["applied_acceleration"])
                for agent in PLATOON_IDS
            ]
            controller.record_applied_actions(infos)
            samples.append(
                {
                    "time": float(env.sim_time),
                    "front_id": front_id,
                    "front_gap": front_gap,
                    "front_speed": front_speed,
                    "front_acceleration": front_acceleration,
                    "front_road": front_road,
                    "front_lane": front_lane,
                    "front_station": front_station,
                    "front_is_ramp_origin": front_is_ramp_origin,
                    "leader_speed": speeds[0],
                    "speeds": speeds,
                    "physical_gaps": np.asarray(
                        compact["physical_target"][:, 2], dtype=np.float32
                    ).tolist(),
                    "physical_relative_speeds": np.asarray(
                        compact["physical_target"][:, 3], dtype=np.float32
                    ).tolist(),
                    "physical_front_accelerations": np.asarray(
                        compact["physical_target"][:, 4], dtype=np.float32
                    ).tolist(),
                    "raw_actions": raw[:, 0].tolist(),
                    "safe_actions": safe[:, 0].tolist(),
                    "applied_actions": applied,
                    "leader_restart_opportunity": bool(
                        infos["leader"].get("leader_restart_opportunity", False)
                    ),
                    "leader_stall_duration": float(
                        infos["leader"].get("leader_stall_duration", 0.0)
                    ),
                    "shield_triggered": bool(
                        infos["leader"].get("shield_triggered", False)
                    ),
                }
            )
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        events = list(env.event_log)
        env.close()

    report = {
        "checkpoint": str(arguments.checkpoint),
        "seed": arguments.seed,
        "episode": arguments.episode + 1,
        "scenario_rng_seed": scenario_rng_seed,
        "reset_seed": reset_seed,
        "scenario_options": options,
        "longest_clear_stall": _longest_clear_stall(samples),
        "minimum_leader_speed": min(item["leader_speed"] for item in samples),
        "maximum_front_gap_while_leader_below_0_3": max(
            (
                item["front_gap"]
                for item in samples
                if item["leader_speed"] <= 0.3
            ),
            default=0.0,
        ),
        "events": events,
        "samples": samples,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(arguments.output),
        "longest_clear_stall": report["longest_clear_stall"],
    }))


if __name__ == "__main__":
    main()

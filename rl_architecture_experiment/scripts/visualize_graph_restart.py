"""Visualize and measure foundation or front-traffic restart behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from platoon_control.rl import _attach_follower_predictions
from platoon_control.training import load_controller
from platoon_control.graph_rl import (
    _apply_cohesive_warmup_control,
    _apply_correlated_follower_exploration,
    _apply_foundation_probe,
    _apply_synchronized_braking_warmup_control,
    _apply_synchronized_restart_warmup_control,
)
from platoon_control.graph_validation import _restart_timing_metrics
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions


def run(
    gui: bool,
    seed: int,
    steps: int,
    trace_output: Path | None = None,
    checkpoint: Path | None = None,
    device: str = "cpu",
    foundation_case: str = "A3",
    exploration_common_std: float = 0.08,
    exploration_independent_std: float = 0.01,
) -> dict[str, float]:
    env = PlatoonParallelEnv(
        render_mode="human" if gui else None,
        max_time=steps * 0.1 + 1.0,
    )
    observations, _ = env.reset(
        seed=seed,
        options={
            "front_vehicle_count": (
                2 if foundation_case == "stage1_front" else
                1 if foundation_case == "A3" else 0
            ),
            "cutin_enabled": False,
            "wave_mode": "off",
            "v2i_enabled": False,
            "communication_delay_probability": 0.0,
            "background_initial_scale": 0.0,
            "background_demand_scale": 0.0,
            "speed_reference": 25.0,
        },
    )
    speeds = {agent: [] for agent in PLATOON_IDS}
    maximum_gaps: list[float] = []
    front_speeds: list[float] = []
    minimum_gap = 500.0
    disconnect_step = -1
    rejoin_step = -1
    collision = False
    failure = False
    final_leader_station = 0.0
    trace: list[dict[str, object]] = []
    controller = load_controller(checkpoint, device) if checkpoint is not None else None
    collection_rng = np.random.default_rng(seed + 1_000)
    if controller is not None:
        controller.reset()
    try:
        for step in range(steps):
            pre_action_physical = {
                agent: {
                    "speed": float(observations[agent]["ego"][1]),
                    "acceleration": float(observations[agent]["ego"][2]),
                    "gap": float(observations[agent]["local_front"][4]),
                    "relative_speed": float(
                        observations[agent]["local_front"][1]
                    ),
                    "front_acceleration": float(
                        observations[agent]["local_front"][2]
                    ),
                }
                for agent in PLATOON_IDS
            }
            if controller is None:
                actions = baseline_actions(observations)
                if foundation_case == "stage0_train":
                    _apply_correlated_follower_exploration(
                        actions,
                        collection_rng,
                        common_std=exploration_common_std,
                        independent_std=exploration_independent_std,
                        episode_step=step,
                    )
                _apply_cohesive_warmup_control(observations, actions)
                _apply_synchronized_braking_warmup_control(observations, actions)
                _apply_synchronized_restart_warmup_control(observations, actions)
                raw_actions = None
                safe_actions = None
            else:
                policy_states, _ = controller.follower_policy_states(observations)
                _attach_follower_predictions(env, policy_states)
                actions = controller.act(observations, deterministic=True)
                raw_actions, safe_actions = controller.requested_action_triplet()
            if foundation_case in ("A3", "A4"):
                _apply_foundation_probe(foundation_case, env, step, actions)
            observations, _, terminations, truncations, infos = env.step(actions)
            if controller is not None:
                controller.record_applied_actions(infos)
            active = set(env.connection.vehicle.getIDList())
            gaps = [
                env._platoon_gap(follower, active)
                for follower in FOLLOWER_IDS
                if follower in active
            ]
            valid_gaps = [gap for gap in gaps if gap >= 0.0]
            maximum_gaps.append(max(valid_gaps, default=0.0))
            maximum_gap = max(valid_gaps, default=0.0)
            if maximum_gap > 15.0 and disconnect_step < 0:
                disconnect_step = step
            if (
                disconnect_step >= 0
                and maximum_gap <= 12.5
                and rejoin_step < 0
            ):
                rejoin_step = step
            minimum_gap = min(minimum_gap, min(valid_gaps, default=minimum_gap))
            front = env._front_ids[0] if env._front_ids else None
            front_speed = (
                float(env.connection.vehicle.getSpeed(front))
                if front is not None and front in active
                else 30.0
            )
            front_speeds.append(front_speed)
            for agent in PLATOON_IDS:
                if agent in active:
                    speeds[agent].append(
                        float(env.connection.vehicle.getSpeed(agent))
                    )
            if trace_output is not None:
                trace.append(
                    {
                        "step": step,
                        "sim_time": float(env.sim_time),
                        "leader_station": float(env._station("leader")),
                        "leader_restart_opportunity": bool(
                            env._leader_restart_opportunity
                        ),
                        "leader_stall_duration": float(env._leader_stall_duration),
                        "pre_action_physical": pre_action_physical,
                        "front_speed": front_speed,
                        "speeds": {
                            agent: float(env.connection.vehicle.getSpeed(agent))
                            for agent in PLATOON_IDS
                            if agent in active
                        },
                        "gaps": {
                            follower: float(env._platoon_gap(follower, active))
                            for follower in FOLLOWER_IDS
                            if follower in active
                        },
                        "requested": {
                            agent: float(infos[agent]["requested_acceleration"])
                            for agent in PLATOON_IDS
                        },
                        "applied": {
                            agent: float(infos[agent]["applied_acceleration"])
                            for agent in PLATOON_IDS
                        },
                        "actor_raw": (
                            {
                                agent: float(raw_actions[index, 0])
                                for index, agent in enumerate(PLATOON_IDS)
                            }
                            if raw_actions is not None
                            else None
                        ),
                        "actor_safe": (
                            {
                                agent: float(safe_actions[index, 0])
                                for index, agent in enumerate(PLATOON_IDS)
                            }
                            if safe_actions is not None
                            else None
                        ),
                        "predictive_shield": {
                            agent: bool(
                                infos[agent]["predictive_safety_intervened"]
                            )
                            for agent in PLATOON_IDS
                        },
                        "collision_shield": {
                            agent: bool(
                                infos[agent]["collision_avoidance_intervened"]
                            )
                            for agent in PLATOON_IDS
                        },
                        "shield_triggered": {
                            agent: bool(infos[agent].get("shield_triggered", False))
                            for agent in PLATOON_IDS
                        },
                        "success": bool(infos["leader"].get("success", False)),
                        "failure": bool(infos["leader"].get("failure", False)),
                    }
                )
            collision = collision or bool(infos["leader"].get("collision", False))
            failure = failure or bool(infos["leader"].get("failure", False))
            final_leader_station = float(env._station("leader"))
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        env.close()
    if trace_output is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output.write_text(json.dumps(trace), encoding="utf-8")
    return {
        "collision": float(collision),
        "minimum_gap": float(minimum_gap),
        "maximum_gap": float(max(maximum_gaps, default=500.0)),
        "rejoin_seconds": (
            float((rejoin_step - disconnect_step) * 0.1)
            if rejoin_step >= 0 and disconnect_step >= 0
            else 1e9
        ),
        "failure": float(failure),
        "final_leader_station": final_leader_station,
        **_restart_timing_metrics(speeds, maximum_gaps, front_speeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exploration-common-std", type=float, default=0.08)
    parser.add_argument("--exploration-independent-std", type=float, default=0.01)
    parser.add_argument(
        "--case",
        choices=("A3", "A4", "stage0", "stage0_train", "stage1_front"),
        default="A3",
    )
    arguments = parser.parse_args()
    metrics = run(
        arguments.gui,
        arguments.seed,
        arguments.steps,
        arguments.trace_output,
        arguments.checkpoint,
        arguments.device,
        arguments.case,
        arguments.exploration_common_std,
        arguments.exploration_independent_std,
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

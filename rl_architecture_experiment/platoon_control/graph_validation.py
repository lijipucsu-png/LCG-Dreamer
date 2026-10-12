"""Deterministic A1-A4 gates for the graph platoon controller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv

from .graph_agents import GraphPlatoonController
from .rl import _attach_follower_predictions


ActionOverride = Callable[
    [PlatoonParallelEnv, int, dict[str, dict[str, Any]]], None
]


def _restart_timing_metrics(
    speeds: dict[str, list[float]],
    maximum_gaps: list[float],
    front_speeds: list[float] | None = None,
) -> dict[str, float]:
    """Measure launch propagation after a sustained all-vehicle stop."""

    if not all(speeds.get(agent) for agent in PLATOON_IDS):
        return {
            "restart_detected": 0.0,
            "restart_spread_seconds": 0.0,
            "restart_completion_seconds": 0.0,
            "restart_response_seconds": 0.0,
            "restart_maximum_speed_spread": 0.0,
            "restart_maximum_gap": 0.0,
        }
    length = min(len(speeds[agent]) for agent in PLATOON_IDS)
    matrix = np.stack(
        [np.asarray(speeds[agent][:length], np.float32) for agent in PLATOON_IDS]
    )
    all_stopped = np.all(matrix <= 0.1, axis=0)
    stop_anchor = -1
    stop_start = -1
    run = 0
    for index, stopped in enumerate(all_stopped):
        run = run + 1 if stopped else 0
        if run >= 5:
            if stop_start < 0:
                stop_start = index - run + 1
            stop_anchor = index
        elif stop_anchor >= 0:
            break
    if stop_anchor < 0:
        return {
            "restart_detected": 0.0,
            "restart_spread_seconds": 0.0,
            "restart_completion_seconds": 0.0,
            "restart_response_seconds": 0.0,
            "restart_maximum_speed_spread": 0.0,
            "restart_maximum_gap": 0.0,
        }
    launch_steps: list[int] = []
    for row in matrix:
        moving = np.flatnonzero(row[stop_anchor + 1 :] > 0.1)
        if moving.size == 0:
            return {
                "restart_detected": 1.0,
                "restart_spread_seconds": 1e9,
                "restart_completion_seconds": 1e9,
                "restart_response_seconds": 1e9,
                "restart_maximum_speed_spread": 1e9,
                "restart_maximum_gap": float(
                    max(maximum_gaps[stop_anchor + 1 :], default=500.0)
                ),
            }
        launch_steps.append(stop_anchor + 1 + int(moving[0]))
    first = min(launch_steps)
    last = max(launch_steps)
    # Restart synchronization is a bounded transient, not the entire
    # remainder of a traffic episode.  Once every vehicle reaches 3 m/s the
    # launch has completed; later cut-ins and braking waves belong to their
    # own capability metrics.  Retain a five-second fallback for launches that
    # never reach the completion speed.
    all_at_restart_speed = np.flatnonzero(
        np.all(matrix[:, last:] >= 3.0, axis=0)
    )
    restart_window_end = (
        last + int(all_at_restart_speed[0])
        if all_at_restart_speed.size
        else min(length - 1, max(last, first + 50))
    )
    response_seconds = 0.0
    if front_speeds:
        front = np.asarray(front_speeds[:length], np.float32)
        stopped_run = 0
        front_was_stopped = False
        restart_step = -1
        for index in range(stop_start, length):
            if front[index] <= 0.1:
                stopped_run += 1
                front_was_stopped = front_was_stopped or stopped_run >= 5
            elif front_was_stopped:
                restart_step = index
                break
            else:
                stopped_run = 0
        response_seconds = (
            max(0.0, (first - restart_step) * 0.1)
            if restart_step >= 0
            # A signal or road-profile stop can restart the platoon without
            # the observed front vehicle ever stopping. There is no front
            # restart response to score in that case.
            else 0.0
        )
    return {
        "restart_detected": 1.0,
        "restart_spread_seconds": (last - first) * 0.1,
        "restart_completion_seconds": (last - launch_steps[0]) * 0.1,
        "restart_response_seconds": response_seconds,
        "restart_maximum_speed_spread": float(
            np.ptp(
                matrix[:, first : restart_window_end + 1], axis=0
            ).max(initial=0.0)
        ),
        "restart_maximum_gap": float(
            max(
                maximum_gaps[first : restart_window_end + 1],
                default=0.0,
            )
        ),
    }


def _run_case(
    controller: GraphPlatoonController,
    seed: int,
    steps: int,
    front_count: int,
    calibration_path: str,
    wave_model_path: str,
    override: ActionOverride | None = None,
) -> dict[str, float]:
    env = PlatoonParallelEnv(
        calibration_path=calibration_path,
        wave_model_path=wave_model_path,
        max_time=steps * 0.1 + 1.0,
    )
    observations, _ = env.reset(
        seed=seed,
        options={
            "front_vehicle_count": front_count,
            "cutin_enabled": False,
            "wave_mode": "off",
            "v2i_enabled": False,
            "communication_delay_probability": 0.0,
            "background_initial_scale": 0.0,
            "background_demand_scale": 0.0,
            "speed_reference": 25.0,
        },
    )
    controller.reset()
    gaps: list[float] = []
    speeds: dict[str, list[float]] = {agent: [] for agent in PLATOON_IDS}
    accelerations: dict[str, list[float]] = {
        agent: [] for agent in PLATOON_IDS
    }
    shields: list[float] = []
    hard_shields: list[float] = []
    final_info: dict[str, Any] = {}
    rejoin_step = -1
    disconnect_step = -1
    maximum_gaps_by_step: list[float] = []
    front_speeds: list[float] = []
    try:
        for step in range(steps):
            policy_states, _ = controller.follower_policy_states(observations)
            _attach_follower_predictions(env, policy_states)
            actions = controller.act(observations, deterministic=True)
            policy_accelerations = {
                agent: float(np.asarray(actions[agent]["accel"]).reshape(-1)[0])
                for agent in PLATOON_IDS
            }
            if override is not None:
                override(env, step, actions)
            action_overridden = any(
                abs(
                    float(np.asarray(actions[agent]["accel"]).reshape(-1)[0])
                    - policy_accelerations[agent]
                )
                > 1e-5
                for agent in PLATOON_IDS
            )
            observations, _, terminations, truncations, infos = env.step(actions)
            controller.record_applied_actions(
                infos, actions if action_overridden else None
            )
            final_info = infos["leader"]
            active = set(env.connection.vehicle.getIDList())
            current_gaps = [
                env._platoon_gap(follower, active)
                for follower in FOLLOWER_IDS
                if follower in active
            ]
            gaps.extend(gap for gap in current_gaps if gap >= 0.0)
            maximum_gaps_by_step.append(
                max((gap for gap in current_gaps if gap >= 0.0), default=0.0)
            )
            leader_front = env.connection.vehicle.getLeader("leader", 500.0)
            front_speeds.append(
                float(env.connection.vehicle.getSpeed(leader_front[0]))
                if leader_front and leader_front[0] in active
                else 30.0
            )
            maximum_current_gap = max(current_gaps, default=0.0)
            if maximum_current_gap > 15.0 and disconnect_step < 0:
                disconnect_step = step
            if (
                disconnect_step >= 0
                and maximum_current_gap <= 12.5
                and rejoin_step < 0
            ):
                rejoin_step = step
            for agent in PLATOON_IDS:
                if agent in active:
                    speeds[agent].append(float(env.connection.vehicle.getSpeed(agent)))
                    accelerations[agent].append(
                        float(env.connection.vehicle.getAcceleration(agent))
                    )
                shields.append(float(infos[agent].get("safety_intervened", False)))
                hard_shields.append(
                    float(
                        infos[agent].get("collision_avoidance_intervened", False)
                        or infos[agent].get("predictive_safety_intervened", False)
                    )
                )
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        env.close()
    leader_rms = float(np.sqrt(np.mean(np.square(accelerations["leader"]))))
    follower_rms = max(
        (
            float(np.sqrt(np.mean(np.square(accelerations[agent]))))
            for agent in FOLLOWER_IDS
            if accelerations[agent]
        ),
        default=0.0,
    )
    all_speeds = np.concatenate(
        [np.asarray(values) for values in speeds.values() if values]
    )
    common_length = min((len(values) for values in speeds.values()), default=0)
    speed_matrix = np.stack(
        [np.asarray(speeds[agent][:common_length]) for agent in PLATOON_IDS]
    )
    steady_window = speed_matrix[:, -min(200, common_length) :]
    steady_speed_rmse = float(
        np.sqrt(np.mean(np.square(steady_window - 25.0)))
    )
    steady_shield_samples = min(200 * len(PLATOON_IDS), len(shields))
    steady_shield_rate = float(
        np.mean(shields[-steady_shield_samples:] or [0.0])
    )
    steady_hard_shield_rate = float(
        np.mean(hard_shields[-steady_shield_samples:] or [0.0])
    )
    cruise_error = np.max(np.abs(speed_matrix - 25.0), axis=0)
    time_to_cruise = 1e9
    for index in range(max(0, common_length - 19)):
        if np.all(cruise_error[index : index + 20] <= 0.5):
            time_to_cruise = index * 0.1
            break
    return {
        "collision": float(final_info.get("collision", False)),
        "failure": float(final_info.get("failure", False)),
        "minimum_gap": float(min(gaps, default=0.0)),
        "maximum_gap": float(max(gaps, default=500.0)),
        "shield_rate": float(np.mean(shields or [0.0])),
        "steady_shield_rate": steady_shield_rate,
        "hard_shield_rate": float(np.mean(hard_shields or [0.0])),
        "steady_hard_shield_rate": steady_hard_shield_rate,
        "speed_rmse_25": float(np.sqrt(np.mean(np.square(all_speeds - 25.0)))),
        "steady_speed_rmse_25": steady_speed_rmse,
        "final_mean_speed": float(speed_matrix[:, -1].mean()),
        "time_to_cruise_seconds": float(time_to_cruise),
        "string_amplification": follower_rms / max(leader_rms, 0.1),
        "rejoin_seconds": (
            max(0.0, (rejoin_step - disconnect_step) * 0.1)
            if rejoin_step >= 0 and disconnect_step >= 0
            else 1e9
        ),
        **_restart_timing_metrics(
            speeds, maximum_gaps_by_step, front_speeds
        ),
    }


def evaluate_foundation_gates(
    controller: GraphPlatoonController,
    calibration_path: str,
    wave_model_path: str,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    a1 = _run_case(
        controller, seed + 1, 600, 0, calibration_path, wave_model_path
    )

    def synchronized_brake(
        env: PlatoonParallelEnv,
        step: int,
        actions: dict[str, dict[str, Any]],
    ) -> None:
        del env
        if 50 <= step < 80:
            actions["leader"]["accel"] = np.asarray([-2.0], np.float32)
        elif 80 <= step < 120:
            actions["leader"]["accel"] = np.asarray([1.0], np.float32)

    a2 = _run_case(
        controller,
        seed + 2,
        400,
        0,
        calibration_path,
        wave_model_path,
        synchronized_brake,
    )

    def front_stop_go(
        env: PlatoonParallelEnv,
        step: int,
        actions: dict[str, dict[str, Any]],
    ) -> None:
        del actions
        if not env._front_ids:
            return
        front = env._front_ids[0]
        if front not in env.connection.vehicle.getIDList():
            return
        if step < 50:
            target_speed = 25.0
        elif step < 100:
            target_speed = max(0.0, 25.0 - 0.5 * (step - 50))
        elif step < 150:
            target_speed = 0.0
        elif step < 200:
            target_speed = min(25.0, 0.5 * (step - 150))
        else:
            target_speed = 25.0
        env.connection.vehicle.setSpeed(front, target_speed)

    a3 = _run_case(
        controller,
        seed + 3,
        500,
        1,
        calibration_path,
        wave_model_path,
        front_stop_go,
    )

    def force_disconnect(
        env: PlatoonParallelEnv,
        step: int,
        actions: dict[str, dict[str, Any]],
    ) -> None:
        del env
        if 50 <= step < 80:
            actions["follower_3"]["accel"] = np.asarray([-2.0], np.float32)

    a4 = _run_case(
        controller,
        seed + 4,
        500,
        0,
        calibration_path,
        wave_model_path,
        force_disconnect,
    )
    passed = {
        "A1": bool(
            a1["collision"] == 0.0
            and a1["failure"] == 0.0
            and a1["steady_speed_rmse_25"] <= 0.5
            and a1["time_to_cruise_seconds"] <= 40.0
            and a1["minimum_gap"] >= 5.0
            and a1["maximum_gap"] <= 15.0
            and a1["steady_hard_shield_rate"] <= 0.02
        ),
        "A2": bool(
            a2["collision"] == 0.0
            and a2["minimum_gap"] >= 5.0
            and a2["string_amplification"] <= 1.3
        ),
        "A3": bool(
            a3["collision"] == 0.0
            and a3["failure"] == 0.0
            and a3["restart_detected"] == 1.0
            and a3["restart_spread_seconds"] <= 0.8
            and a3["restart_response_seconds"] <= 0.3
            and a3["restart_maximum_speed_spread"] <= 0.5
            and a3["restart_maximum_gap"] <= 15.0
            and a3["string_amplification"] <= 1.5
        ),
        "A4": bool(
            a4["collision"] == 0.0
            and (
                a4["maximum_gap"] <= 15.0
                or a4["rejoin_seconds"] <= 15.0
            )
        ),
    }
    report: dict[str, Any] = {
        "passed": all(passed.values()),
        "gates": passed,
        "metrics": {"A1": a1, "A2": a2, "A3": a3, "A4": a4},
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

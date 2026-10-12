"""Reproduce a curriculum scenario and capture stopped-team action semantics."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from platoon_control.graph_rl import GraphRLConfig, load_graph_controller
from platoon_control.rl import _attach_follower_predictions, _curriculum_options
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    checkpoint = torch.load(
        arguments.checkpoint, map_location="cpu", weights_only=False
    )
    config = GraphRLConfig(**checkpoint["graph_rl_config"])
    controller = load_graph_controller(arguments.checkpoint, arguments.device)
    scenario_rng = np.random.default_rng(
        config.seed + 50_000 + arguments.episode_index
    )
    options, _ = _curriculum_options(
        arguments.stage,
        scenario_rng,
        config.evaluation_config(),
        checkpoint["total_steps"],
    )
    reset_seed = config.seed + arguments.episode_index
    env = PlatoonParallelEnv(
        calibration_path=config.calibration_path,
        wave_model_path=config.wave_model_path,
        max_time=config.episode_max_time,
    )
    observations, _ = env.reset(seed=reset_seed, options=options)
    controller.reset()
    trace: list[dict[str, object]] = []
    restart_trace: list[dict[str, object]] = []
    formation_trace: list[dict[str, object]] = []
    terminal_trace: deque[dict[str, object]] = deque(maxlen=180)
    stopped_run = 0
    restart_capture_active = False
    final_info: dict[str, object] = {}
    terminal_active_ids: list[str] = []
    minimum_pair_gap = float("inf")
    maximum_pair_gap = 0.0
    maximum_gap_sample: dict[str, object] = {}
    try:
        while env.agents:
            with torch.no_grad():
                policy_states = controller.follower_policy_states(observations)[0]
            _attach_follower_predictions(env, policy_states)
            actions = controller.act(
                observations, deterministic=True, policy_states=policy_states
            )
            raw, safe = controller.requested_action_triplet()
            compact = controller.replay_observation()
            physical = np.asarray(compact["physical_target"], np.float32)
            speeds = np.asarray(
                [env.connection.vehicle.getSpeed(agent) for agent in PLATOON_IDS],
                np.float32,
            )
            stopped_run = stopped_run + 1 if np.all(speeds <= 0.3) else 0
            restart_capture_active = restart_capture_active or stopped_run >= 5
            actual_front = env.connection.vehicle.getLeader("leader", 500.0)
            sample = {
                "time": env.sim_time,
                "speeds": speeds.tolist(),
                "physical_gaps": physical[:, 2].tolist(),
                "relative_speeds": physical[:, 3].tolist(),
                "front_accelerations": physical[:, 4].tolist(),
                "advisories": physical[:, 6].tolist(),
                "raw_actions": raw[:, 0].tolist(),
                "safe_actions": safe[:, 0].tolist(),
                "previous_applied_actions": (
                    controller._previous_applied[0, :, 0]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "previous_intentions": (
                    controller._previous_intention[0, :, 0]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "recovery_latch": (
                    controller._formation_recovery_latch[0]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "actual_front": (
                    [
                        actual_front[0],
                        float(actual_front[1]),
                        float(env.connection.vehicle.getSpeed(actual_front[0])),
                    ]
                    if actual_front
                    else None
                ),
            }
            pair_gaps = physical[1:, 2]
            predicted_pair_gaps = pair_gaps + 1.5 * np.maximum(
                physical[1:, 3], 0.0
            )
            sample["predicted_pair_gaps_1p5s"] = predicted_pair_gaps.tolist()
            valid_pair_gaps = pair_gaps[(pair_gaps > 0.0) & (pair_gaps < 500.0)]
            if valid_pair_gaps.size:
                minimum_pair_gap = min(
                    minimum_pair_gap, float(valid_pair_gaps.min())
                )
                step_maximum_gap = float(valid_pair_gaps.max())
                if step_maximum_gap > maximum_pair_gap:
                    maximum_pair_gap = step_maximum_gap
                    maximum_gap_sample = dict(sample)
            if (
                len(formation_trace) < 3_000
                and (
                    np.any(pair_gaps > 12.0)
                    or np.any(predicted_pair_gaps > 13.0)
                )
            ):
                formation_trace.append(sample)
            terminal_trace.append(sample)
            if stopped_run >= 5 and len(trace) < 80:
                trace.append(sample)
            if restart_capture_active and len(restart_trace) < 3_000:
                restart_trace.append(sample)
            observations, _, terminations, truncations, infos = env.step(actions)
            controller.record_applied_actions(infos)
            final_info = dict(infos["leader"])
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        events = list(env.event_log)
        if env.connection is not None:
            try:
                terminal_active_ids = list(
                    env.connection.vehicle.getIDList()
                )
            except Exception:
                terminal_active_ids = []
        env.close()

    payload = {
        "checkpoint": str(arguments.checkpoint),
        "episode_index": arguments.episode_index,
        "curriculum_stage": arguments.stage,
        "reset_seed": reset_seed,
        "options": options,
        "final": {
            "success": bool(final_info.get("success", False)),
            "failure": bool(final_info.get("failure", False)),
            "collision": bool(final_info.get("collision", False)),
            "leader_station": float(final_info.get("leader_station", 0.0)),
            "active_count": int(final_info.get("active_count", 0)),
            "connected_pair_count": int(
                final_info.get("connected_pair_count", 0)
            ),
        },
        "leader_stall_events": [
            event for event in events if event.get("event") == "leader_stalled_with_clear_path"
        ],
        "stopped_trace": trace,
        "restart_trace": restart_trace,
        "formation_trace": formation_trace,
        "terminal_trace": list(terminal_trace),
        "minimum_pair_gap": (
            minimum_pair_gap if np.isfinite(minimum_pair_gap) else 0.0
        ),
        "maximum_pair_gap": maximum_pair_gap,
        "maximum_gap_sample": maximum_gap_sample,
        "terminal_active_ids": terminal_active_ids,
        "recent_events": events[-100:],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["final"]))
    print(f"stopped_trace={len(trace)} stall_events={len(payload['leader_stall_events'])}")


if __name__ == "__main__":
    main()

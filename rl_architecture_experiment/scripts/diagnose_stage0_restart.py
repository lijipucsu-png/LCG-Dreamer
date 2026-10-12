"""Reproduce the first Stage-0 collection episode with a detailed trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from platoon_control.graph_rl import (
    GraphRLConfig,
    _apply_cohesive_warmup_control,
    _apply_synchronized_braking_warmup_control,
    _apply_synchronized_restart_warmup_control,
)
from platoon_control.graph_validation import _restart_timing_metrics
from platoon_control.rl import _apply_correlated_follower_exploration, _curriculum_options
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions


def run(
    seed: int,
    steps: int,
    action_seed: int | None = None,
    common_std: float = 0.035,
    independent_std: float = 0.004,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = GraphRLConfig(seed=seed)
    rng = np.random.default_rng(
        seed + 1_000 if action_seed is None else action_seed
    )
    options, _ = _curriculum_options(0, rng, config.evaluation_config(), 0)
    env = PlatoonParallelEnv(max_time=config.episode_max_time)
    observations, _ = env.reset(seed=seed, options=options)
    speeds = {agent: [] for agent in PLATOON_IDS}
    maximum_gaps: list[float] = []
    trace: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    try:
        for step in range(steps):
            if not env.agents:
                break
            actions = baseline_actions(observations)
            _apply_correlated_follower_exploration(
                actions,
                rng,
                common_std=common_std,
                independent_std=independent_std,
                episode_step=step,
            )
            cohesive = _apply_cohesive_warmup_control(observations, actions)
            synchronized_braking = _apply_synchronized_braking_warmup_control(
                observations, actions
            )
            synchronized_restart = _apply_synchronized_restart_warmup_control(
                observations, actions
            )
            next_observations, _, terminations, truncations, infos = env.step(actions)
            final_info = dict(infos["leader"])
            active = set(env.connection.vehicle.getIDList())
            gaps = {
                follower: float(env._platoon_gap(follower, active))
                for follower in FOLLOWER_IDS
                if follower in active
            }
            maximum_gaps.append(max(gaps.values(), default=0.0))
            vehicle_speeds = {
                agent: float(env.connection.vehicle.getSpeed(agent))
                for agent in PLATOON_IDS
                if agent in active
            }
            for agent, speed in vehicle_speeds.items():
                speeds[agent].append(speed)
            trace.append(
                {
                    "step": step,
                    "time": float(final_info.get("time", 0.0)),
                    "station": {
                        agent: float(env._station(agent))
                        for agent in PLATOON_IDS
                        if agent in active
                    },
                    "speeds": vehicle_speeds,
                    "gaps": gaps,
                    "requested": {
                        agent: float(infos[agent]["requested_acceleration"])
                        for agent in PLATOON_IDS
                    },
                    "applied": {
                        agent: float(infos[agent]["applied_acceleration"])
                        for agent in PLATOON_IDS
                    },
                    "predictive_shield": {
                        agent: bool(infos[agent]["predictive_safety_intervened"])
                        for agent in PLATOON_IDS
                    },
                    "collision_shield": {
                        agent: bool(infos[agent]["collision_avoidance_intervened"])
                        for agent in PLATOON_IDS
                    },
                    "formation_guard": bool(
                        infos["leader"]["formation_guard_intervened"]
                    ),
                    "leader_restart_opportunity": bool(
                        infos["leader"].get("leader_restart_opportunity", False)
                    ),
                    "cohesive_control": cohesive,
                    "synchronized_braking": synchronized_braking,
                    "synchronized_restart": synchronized_restart,
                }
            )
            if any(terminations.values()) or any(truncations.values()):
                break
            observations = next_observations
    finally:
        events = list(env.event_log)
        env.close()
    metrics = {
        "steps": len(trace),
        "success": float(final_info.get("success", False)),
        "collision": float(final_info.get("collision", False)),
        "failure": float(final_info.get("failure", False)),
        "leader_station": float(final_info.get("route_station", 0.0)),
        "minimum_gap": float(
            min((gap for row in trace for gap in row["gaps"].values()), default=500.0)
        ),
        "maximum_gap": float(max(maximum_gaps, default=500.0)),
        **_restart_timing_metrics(speeds, maximum_gaps),
        "events": events,
        "options": options,
    }
    return metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--common-std", type=float, default=0.035)
    parser.add_argument("--independent-std", type=float, default=0.004)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    arguments = parser.parse_args()
    metrics, trace = run(
        arguments.seed,
        arguments.steps,
        arguments.action_seed,
        arguments.common_std,
        arguments.independent_std,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    arguments.trace_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.trace_output.write_text(json.dumps(trace), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

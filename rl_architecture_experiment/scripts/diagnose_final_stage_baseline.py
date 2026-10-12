"""Compare the deterministic CACC baseline on graph-training eval scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from platoon_control.graph_rl import GraphRLConfig
from platoon_control.rl import _curriculum_options
from platoon_sumo.env import FOLLOWER_IDS, PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions


def run_episode(seed: int, episode: int, stage: int, step: int) -> dict[str, object]:
    config = GraphRLConfig(seed=seed).evaluation_config()
    scenario_rng = np.random.default_rng(
        seed + 20_000 + stage * 1_000 + episode
    )
    options, _ = _curriculum_options(stage, scenario_rng, config, step)
    env = PlatoonParallelEnv(
        calibration_path=config.calibration_path,
        wave_model_path=config.wave_model_path,
        max_time=config.episode_max_time,
    )
    observations, _ = env.reset(
        seed=seed + 10_000 + episode,
        options=options,
    )
    minimum_gap = float("inf")
    maximum_gap = 0.0
    background_peak = 0
    final_info: dict[str, object] = {}
    steps = 0
    try:
        while env.agents:
            actions = baseline_actions(observations)
            observations, _, terminations, truncations, infos = env.step(actions)
            final_info = infos["leader"]
            active = set(env.connection.vehicle.getIDList())
            for follower in FOLLOWER_IDS:
                if follower not in active:
                    continue
                gap = float(env._platoon_gap(follower, active))
                if gap >= 0.0:
                    minimum_gap = min(minimum_gap, gap)
                    maximum_gap = max(maximum_gap, gap)
            background_peak = max(
                background_peak,
                int(final_info.get("background_active_count", 0)),
            )
            steps += 1
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        env.close()
    return {
        "episode": episode + 1,
        "reset_seed": seed + 10_000 + episode,
        "front_vehicle_count": options["front_vehicle_count"],
        "steps": steps,
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "failure": bool(final_info.get("failure", False)),
        "leader_station": float(final_info.get("leader_station", 0.0)),
        "connected_pair_count": int(
            final_info.get("platoon_connected_pair_count", 0)
        ),
        "minimum_gap": 0.0 if minimum_gap == float("inf") else minimum_gap,
        "maximum_gap": maximum_gap,
        "background_peak": background_peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument("--step", type=int, default=250_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/final_stage_baseline_seed42.json"),
    )
    arguments = parser.parse_args()
    episodes = [
        run_episode(arguments.seed, episode, arguments.stage, arguments.step)
        for episode in range(arguments.episodes)
    ]
    payload = {
        "controller": "cacc_baseline",
        "seed": arguments.seed,
        "stage": arguments.stage,
        "step": arguments.step,
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "episodes": episodes,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

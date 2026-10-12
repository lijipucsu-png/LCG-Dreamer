"""Run the SUMO platoon scenario with a deterministic CACC baseline."""

from __future__ import annotations

import argparse

import numpy as np

from .env import (
    CUTIN_TARGET_GAP,
    MAX_ACCELERATION,
    MAX_BRAKING,
    PLATOON_IDS,
    PREDECESSOR,
    TARGET_GAP,
    PlatoonParallelEnv,
)


def baseline_actions(observations: dict[str, dict[str, np.ndarray]]) -> dict:
    """Simple speed controller for the leader and CACC controller for followers."""

    actions = {}
    row_index = {vehicle_id: index for index, vehicle_id in enumerate(PLATOON_IDS)}
    for agent, observation in observations.items():
        state = observation["platoon_state"]
        own_row = row_index[agent]
        own_speed = float(state[own_row, 1])
        if agent == "leader":
            acceleration = 0.6 * (25.0 - own_speed)
        else:
            predecessor_row = row_index[PREDECESSOR[agent]]
            predecessor_speed = float(state[predecessor_row, 1])
            gap = float(state[own_row, 4])
            desired_gap = TARGET_GAP
            if agent == "follower_3":
                history = observation["neighbor_history"]
                mask = observation["neighbor_mask"] > 0.5
                latest = history[:, -1]
                ramp_cutin_nearby = np.any(
                    mask & (latest[:, 6] > 0.5) & (np.abs(latest[:, 0]) < 80.0)
                )
                if ramp_cutin_nearby:
                    desired_gap = CUTIN_TARGET_GAP
            acceleration = 0.25 * (gap - desired_gap) + 0.7 * (
                predecessor_speed - own_speed
            )
        actions[agent] = {
            "accel": np.asarray(
                [np.clip(acceleration, MAX_BRAKING, MAX_ACCELERATION)],
                dtype=np.float32,
            ),
            "lane_change": 0,
        }
    return actions


def run(gui: bool, seed: int, max_steps: int) -> None:
    env = PlatoonParallelEnv(render_mode="human" if gui else None)
    observations, _ = env.reset(seed=seed)
    seen_events = 0
    try:
        for step_index in range(max_steps):
            if not env.agents:
                break
            observations, rewards, terminations, truncations, infos = env.step(
                baseline_actions(observations)
            )
            for event in env.event_log[seen_events:]:
                print(f"event: {event}")
            seen_events = len(env.event_log)
            if step_index % 100 == 0 and infos:
                leader_info = infos["leader"]
                print(
                    f"t={leader_info['time']:.1f}s "
                    f"station={leader_info['route_station']:.1f}m "
                    f"reward={rewards['leader']:.3f} "
                    f"cutin={leader_info['cutin_state']}"
                )
            if any(terminations.values()) or any(truncations.values()):
                final = infos["leader"]
                print(
                    "episode finished:",
                    {
                        "success": final["success"],
                        "collision": final["collision"],
                        "failure": final["failure"],
                        "time": final["time"],
                    },
                )
                break
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Run with sumo-gui")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1800)
    arguments = parser.parse_args()
    run(arguments.gui, arguments.seed, arguments.max_steps)


if __name__ == "__main__":
    main()

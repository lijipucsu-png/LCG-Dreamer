"""Trace graph-actor actions through the differentiable and SUMO safety layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from platoon_control.rl import _attach_follower_predictions
from platoon_control.training import load_controller
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv


def run(checkpoint: Path, seed: int, steps: int, device: str) -> dict[str, object]:
    controller = load_controller(checkpoint, device)
    env = PlatoonParallelEnv(max_time=steps * 0.1 + 1.0)
    observations, _ = env.reset(
        seed=seed,
        options={
            "front_vehicle_count": 0,
            "cutin_enabled": False,
            "wave_mode": "off",
            "v2i_enabled": False,
            "communication_delay_probability": 0.0,
            "background_initial_scale": 0.0,
            "background_demand_scale": 0.0,
            "speed_reference": 25.0,
        },
    )
    trace: list[dict[str, object]] = []
    controller.reset()
    try:
        for step in range(steps):
            states, _ = controller.follower_policy_states(observations)
            _attach_follower_predictions(env, states)
            actions = controller.act(observations, deterministic=True)
            raw, safe = controller.requested_action_triplet()
            assert controller._last_policy is not None
            mean = (
                controller._last_policy.mean_acceleration[0]
                .detach()
                .cpu()
                .numpy()
            )
            next_observations, _, terminations, truncations, infos = env.step(actions)
            controller.record_applied_actions(infos)
            active = set(env.connection.vehicle.getIDList())
            trace.append(
                {
                    "step": step,
                    "raw": raw[:, 0].tolist(),
                    "mean": mean[:, 0].tolist(),
                    "safe": safe[:, 0].tolist(),
                    "applied": [
                        float(infos[agent]["applied_acceleration"])
                        for agent in PLATOON_IDS
                    ],
                    "speed": [
                        float(env.connection.vehicle.getSpeed(agent))
                        if agent in active
                        else float("nan")
                        for agent in PLATOON_IDS
                    ],
                    "shielded": [
                        bool(infos[agent].get("safety_intervened", False))
                        for agent in PLATOON_IDS
                    ],
                    "predictive_shield": [
                        bool(infos[agent].get("predictive_safety_intervened", False))
                        for agent in PLATOON_IDS
                    ],
                    "collision_shield": [
                        bool(infos[agent].get("collision_avoidance_intervened", False))
                        for agent in PLATOON_IDS
                    ],
                    "formation_guard": [
                        bool(infos[agent].get("formation_guard_intervened", False))
                        for agent in PLATOON_IDS
                    ],
                }
            )
            if any(terminations.values()) or any(truncations.values()):
                break
            observations = next_observations
    finally:
        env.close()

    arrays = {
        key: np.asarray([row[key] for row in trace], np.float64)
        for key in ("raw", "mean", "safe", "applied", "speed", "shielded")
    }
    summary = {
        "steps": len(trace),
        "agents": {
            agent: {
                "raw_mean": float(arrays["raw"][:, index].mean()),
                "raw_min": float(arrays["raw"][:, index].min()),
                "raw_max": float(arrays["raw"][:, index].max()),
                "safe_mean": float(arrays["safe"][:, index].mean()),
                "applied_mean": float(arrays["applied"][:, index].mean()),
                "final_speed": float(arrays["speed"][-1, index]),
                "maximum_speed": float(arrays["speed"][:, index].max()),
                "shield_rate": float(arrays["shielded"][:, index].mean()),
            }
            for index, agent in enumerate(PLATOON_IDS)
        },
    }
    return {"summary": summary, "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.checkpoint, arguments.seed, arguments.steps, arguments.device)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()

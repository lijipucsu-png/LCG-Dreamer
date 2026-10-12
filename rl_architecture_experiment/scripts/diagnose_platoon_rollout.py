from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from platoon_control.rl import EpisodeReplay, FOLLOWER_ACTOR_OBJECTIVE_VERSION
from platoon_control.models import (
    FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX,
    FOLLOWER_RECOVERY_GAP_INDEX,
    FOLLOWER_RECOVERY_ROLE_INDEX,
)
from platoon_control.models.follower_rl import FOLLOWER_LOCAL_RELATIVE_SPEED_INDEX
from platoon_control.training import PlatoonTrainer, load_controller
from platoon_sumo.env import (
    FOLLOWER_IDS,
    PLATOON_IDS,
    PREDECESSOR,
    PlatoonParallelEnv,
)


def vehicle_state(env: PlatoonParallelEnv, vehicle_id: str) -> dict[str, float]:
    return {
        "station": float(env._station(vehicle_id)),
        "speed": float(env.connection.vehicle.getSpeed(vehicle_id)),
        "acceleration": float(env.connection.vehicle.getAcceleration(vehicle_id)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--front-count", type=int, default=2)
    parser.add_argument("--wave-mode", choices=("off", "learned"), default="off")
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--actor-updates", type=int, default=0)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-6)
    parser.add_argument("--actor-rl-weight", type=float, default=0.05)
    parser.add_argument("--actor-stability-weight", type=float, default=150.0)
    parser.add_argument("--behavior-weight", type=float, default=0.4)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    controller = load_controller(arguments.checkpoint, arguments.device)
    update_metrics: dict[str, float] = {}
    if arguments.actor_updates > 0:
        checkpoint = torch.load(
            arguments.checkpoint, map_location="cpu", weights_only=False
        )
        trainer = PlatoonTrainer(controller)
        for name, state in checkpoint.get("optimizers", {}).items():
            if name in trainer.optimizers:
                if (
                    name == "follower_actor"
                    and int(
                        checkpoint.get("follower_actor_objective_version", 1)
                    )
                    < FOLLOWER_ACTOR_OBJECTIVE_VERSION
                ):
                    continue
                try:
                    trainer.optimizers[name].load_state_dict(state)
                except ValueError:
                    pass
        for group in trainer.optimizers["follower_actor"].param_groups:
            group["lr"] = arguments.actor_learning_rate
        replay = EpisodeReplay(capacity_steps=30_000, seed=42)
        replay.load_state_dict(
            torch.load(
                str(arguments.checkpoint).replace(
                    "rl_latest.pt", "replay_latest.pt"
                ),
                map_location="cpu",
                weights_only=False,
            )
        )
        for _ in range(arguments.actor_updates):
            batch = replay.sample_followers(32)
            update_metrics = trainer.train_follower_actor(
                batch[0],
                batch[5],
                batch[6],
                behavior_weight=arguments.behavior_weight,
                rl_weight=arguments.actor_rl_weight,
                stability_weight=arguments.actor_stability_weight,
            )
    env = PlatoonParallelEnv(max_time=240.0)
    observations, _ = env.reset(
        seed=arguments.seed,
        options={
            "front_vehicle_count": arguments.front_count,
            "cutin_enabled": False,
            "wave_mode": arguments.wave_mode,
            "v2i_enabled": False,
            "communication_delay_probability": 0.0,
            "background_initial_scale": 0.0,
            "background_demand_scale": 0.0,
        },
    )
    controller.reset()
    rows: list[dict[str, Any]] = []
    final_infos: dict[str, Any] = {}
    final_terminations: dict[str, bool] = {}
    final_truncations: dict[str, bool] = {}
    final_active: set[str] = set()
    try:
        for step in range(arguments.max_steps):
            policy_states = controller.follower_policy_states(observations)[0]
            env.set_follower_control_predictions(
                controller.follower_control_predictions(policy_states)
            )
            actions = controller.act(
                observations,
                policy_states=policy_states,
                leader_deterministic=True,
                follower_deterministic=True,
            )
            next_observations, _, terminations, truncations, infos = env.step(actions)
            final_infos = infos
            final_terminations = terminations
            final_truncations = truncations
            controller.record_applied_actions(infos)
            active = set(env.connection.vehicle.getIDList())
            final_active = active
            if all(vehicle_id in active for vehicle_id in PLATOON_IDS):
                current = policy_states[0, -1].detach().cpu().numpy()
                physical_front = env.connection.vehicle.getLeader("follower_1", 500.0)
                row = {
                    "step": step,
                    "time": float(env.sim_time),
                    "gap": float(env._platoon_gap("follower_1", active)),
                    "pair_gaps": {
                        follower: float(env._platoon_gap(follower, active))
                        for follower in FOLLOWER_IDS
                        if PREDECESSOR[follower] in active
                    },
                    "leader": vehicle_state(env, "leader"),
                    "follower_1": vehicle_state(env, "follower_1"),
                    "leader_requested": float(np.asarray(actions["leader"]["accel"]).item()),
                    "leader_applied": float(infos["leader"]["applied_acceleration"]),
                    "follower_requested": float(
                        np.asarray(actions["follower_1"]["accel"]).item()
                    ),
                    "follower_applied": float(
                        infos["follower_1"]["applied_acceleration"]
                    ),
                    "state_gap": float(current[FOLLOWER_RECOVERY_GAP_INDEX]),
                    "state_relative_speed": float(
                        current[FOLLOWER_LOCAL_RELATIVE_SPEED_INDEX]
                    ),
                    "state_recovery_role": float(
                        current[FOLLOWER_RECOVERY_ROLE_INDEX]
                    ),
                    "state_leader_plan_acceleration": float(
                        current[FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX]
                    ),
                    "physical_front": (
                        physical_front[0] if physical_front else None
                    ),
                    "physical_front_gap": (
                        float(physical_front[1]) if physical_front else None
                    ),
                }
                rows.append(row)
            observations = next_observations
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        events = env.event_log
        env.close()

    if not rows:
        raise RuntimeError("No complete platoon states were observed")
    selected: list[dict[str, Any]] = []
    selected_steps: set[int] = set()

    def select(row: dict[str, Any]) -> None:
        if row["step"] not in selected_steps:
            selected.append(row)
            selected_steps.add(row["step"])

    for threshold in (15.0, 22.0, 30.0, 60.0, 80.0):
        crossing = next((row for row in rows if row["gap"] >= threshold), None)
        if crossing is not None:
            select(crossing)
    recovery = next(
        (row for row in rows if row["state_recovery_role"] >= 0.5), None
    )
    if recovery is not None:
        select(recovery)
    stopped = next(
        (
            row
            for row in rows
            if row["gap"] > 15.0
            and row["follower_1"]["speed"] < 1.0
            and row["leader"]["speed"] > 5.0
        ),
        None,
    )
    if stopped is not None:
        select(stopped)
    maximum = max(rows, key=lambda row: row["gap"])
    select(maximum)
    for offset in (-20, -10, -1, 1, 10, 20):
        index = maximum["step"] + offset
        nearby = next((row for row in rows if row["step"] == index), None)
        if nearby is not None:
            select(nearby)
    selected.sort(key=lambda row: row["step"])
    print(
        json.dumps(
            {
                "steps": len(rows),
                "actor_updates": arguments.actor_updates,
                "update_metrics": update_metrics,
                "max_gap": maximum["gap"],
                "final_gap": rows[-1]["gap"],
                "final_pair_gaps": rows[-1]["pair_gaps"],
                "final_active_platoon": sorted(final_active.intersection(PLATOON_IDS)),
                "final_success": bool(final_infos.get("leader", {}).get("success", False)),
                "final_failure": bool(final_infos.get("leader", {}).get("failure", False)),
                "final_collision": bool(final_infos.get("leader", {}).get("collision", False)),
                "final_terminated": any(final_terminations.values()),
                "final_truncated": any(final_truncations.values()),
                "selected": selected,
                "events": [
                    event
                    for event in events
                    if event.get("event")
                    in {
                        "subleader_activated",
                        "platoon_rejoined",
                        "leader_stalled_with_clear_path",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

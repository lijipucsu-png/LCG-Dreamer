"""Paired baseline and ablation evaluation for the complete platoon controller."""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions

from .training import load_controller


VARIANTS = (
    "idm",
    "cacc",
    "full",
    "no_v2i",
    "no_front_prediction",
    "no_delay_compensation",
    "no_cutin_prediction",
    "no_learned_wave",
    "no_semantic_communication",
    "no_latent_extrapolation",
    "no_differentiable_cbf",
    "float_semantic_code",
)


def _idm_actions(observations: dict) -> dict:
    actions: dict[str, dict[str, Any]] = {}
    for agent, observation in observations.items():
        speed = float(observation["ego"][1])
        if agent == "leader":
            mask = observation["front_mask"] > 0.5
            latest = observation["front_history"][:, -1]
            ahead = np.flatnonzero(mask & (latest[:, 0] > 0.0))
            if len(ahead):
                nearest = int(ahead[np.argmin(latest[ahead, 0])])
                gap = max(0.5, float(latest[nearest, 0] - latest[nearest, 4]))
                leader_speed = float(latest[nearest, 2])
            else:
                gap, leader_speed = 1e4, speed
        else:
            gap = max(0.5, float(observation["local_predecessor"][4]))
            leader_speed = speed + float(observation["local_predecessor"][1])
        desired_gap = 2.0 + max(
            0.0,
            speed * 1.2
            + speed * (speed - leader_speed) / (2.0 * np.sqrt(1.2 * 2.5)),
        )
        acceleration = 1.2 * (1.0 - (speed / 25.0) ** 4 - (desired_gap / gap) ** 2)
        actions[agent] = {
            "accel": np.asarray([np.clip(acceleration, -3.0, 1.2)], np.float32),
            "lane_change": 0,
        }
    return actions


@dataclass
class EvaluationConfig:
    seed: int = 2026
    episodes_per_front_count: int = 5
    max_steps: int = 2400
    output_dir: str = "artifacts/evaluation"
    calibration_path: str = "artifacts/calibration/calibrated_parameters.json"
    wave_model_path: str = "artifacts/checkpoints/wave_best.pt"
    device: str = "auto"


def _ablated_observations(observations: dict, variant: str) -> dict:
    if variant not in ("no_v2i", "no_front_prediction"):
        return observations
    output = copy.deepcopy(observations)
    if variant == "no_v2i":
        output["leader"]["v2i_flow"].fill(0.0)
        output["leader"]["road_profile"].fill(0.0)
    if variant == "no_front_prediction":
        output["leader"]["front_history"].fill(0.0)
        output["leader"]["front_mask"].fill(0.0)
    return output


def _episode(
    controller: Any,
    config: EvaluationConfig,
    variant: str,
    seed: int,
    front_count: int,
) -> dict[str, Any]:
    env = PlatoonParallelEnv(
        calibration_path=config.calibration_path,
        wave_model_path=config.wave_model_path,
    )
    wave_mode = "off" if variant == "no_learned_wave" else "learned"
    observations, _ = env.reset(
        seed=seed,
        options={"front_vehicle_count": front_count, "wave_mode": wave_mode},
    )
    if controller is not None:
        controller.reset()
    rewards = 0.0
    fuel = 0.0
    speeds: dict[str, list[float]] = {agent: [] for agent in PLATOON_IDS}
    accelerations: dict[str, list[float]] = {agent: [] for agent in PLATOON_IDS}
    gaps: list[float] = []
    final_info: dict[str, Any] = {}
    shield_interventions: list[float] = []
    inference_times_ms: list[float] = []
    try:
        for step_index in range(config.max_steps):
            if not env.agents:
                break
            if variant == "cacc":
                actions = baseline_actions(observations)
            elif variant == "idm":
                actions = _idm_actions(observations)
            else:
                model_observations = _ablated_observations(observations, variant)
                flags = frozenset(
                    name
                    for name in (
                        "delay_compensation",
                        "cutin_prediction",
                        "semantic_communication",
                        "latent_extrapolation",
                        "differentiable_cbf",
                        "float_semantic_code",
                    )
                    if variant == f"no_{name}"
                )
                if variant == "float_semantic_code":
                    flags = frozenset((*flags, "float_semantic_code"))
                inference_started = time.perf_counter()
                policy_states = controller.follower_policy_states(
                    model_observations, flags
                )[0]
                env.set_follower_control_predictions(
                    controller.follower_control_predictions(policy_states)
                )
                actions = controller.act(
                    model_observations,
                    deterministic=True,
                    ablations=flags,
                    policy_states=policy_states,
                )
                inference_times_ms.append(
                    (time.perf_counter() - inference_started) * 1000.0
                )
            observations, step_rewards, terminations, truncations, infos = env.step(actions)
            if controller is not None and variant not in ("cacc", "idm"):
                controller.record_applied_actions(infos)
            rewards += step_rewards["leader"]
            final_info = infos["leader"]
            shield_interventions.extend(
                float(infos[agent].get("safety_intervened", False))
                for agent in PLATOON_IDS
            )
            fuel += final_info["reward_components"]["fuel_mg_per_second"] * env.step_length
            active = set(env.connection.vehicle.getIDList())
            for agent in PLATOON_IDS:
                if agent in active:
                    speeds[agent].append(float(env.connection.vehicle.getSpeed(agent)))
                    accelerations[agent].append(float(env.connection.vehicle.getAcceleration(agent)))
                    leader = env.connection.vehicle.getLeader(agent, 100.0)
                    if leader:
                        gaps.append(float(leader[1]))
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        event_log = env.event_log
        env.close()
    leader_rms = float(np.sqrt(np.mean(np.square(accelerations["leader"])))) if accelerations["leader"] else 0.0
    follower_rms = [
        float(np.sqrt(np.mean(np.square(accelerations[agent]))))
        for agent in FOLLOWER_IDS
        if accelerations[agent]
    ]
    all_speeds = np.concatenate([np.asarray(values) for values in speeds.values() if values])
    all_accelerations = np.concatenate(
        [np.asarray(values) for values in accelerations.values() if values]
    )
    jerk_groups = [
        np.diff(values) / 0.1 for values in accelerations.values() if len(values) > 1
    ]
    jerks = np.concatenate(jerk_groups) if jerk_groups else np.zeros(1)
    event_names = {event["event"] for event in event_log}
    return {
        "variant": variant,
        "seed": seed,
        "front_vehicle_count": front_count,
        "steps": len(speeds["leader"]),
        "reward": rewards,
        "success": int(final_info.get("success", False)),
        "collision": int(final_info.get("collision", False)),
        "failure": int(final_info.get("failure", False)),
        "travel_time_s": float(final_info.get("time", config.max_steps * 0.1)),
        "mean_speed_mps": float(all_speeds.mean()) if len(all_speeds) else 0.0,
        "speed_std_mps": float(all_speeds.std()) if len(all_speeds) else 0.0,
        "acceleration_rms_mps2": float(np.sqrt(np.mean(np.square(all_accelerations)))) if len(all_accelerations) else 0.0,
        "jerk_rms_mps3": float(np.sqrt(np.mean(np.square(jerks)))) if len(jerks) else 0.0,
        "minimum_gap_m": float(np.min(gaps)) if gaps else -1.0,
        "fuel_mg": fuel,
        "string_amplification": float(max(follower_rms, default=0.0) / max(leader_rms, 1e-3)),
        "cutin_success": int("cutin_success" in event_names),
        "wave_triggered": int("wave_brake" in event_names),
        "shield_intervention_rate": float(
            np.mean(shield_interventions or [0.0])
        ),
        "inference_p95_ms": float(
            np.percentile(inference_times_ms, 95)
            if inference_times_ms
            else 0.0
        ),
        "semantic_payload_bytes": (
            18
            if variant not in ("cacc", "idm")
            and getattr(controller, "architecture_id", "")
            == "ma_latent_graph_dreamer_v1"
            else 0
        ),
    }


def evaluate(
    checkpoint: str | Path,
    config: EvaluationConfig,
    variants: tuple[str, ...] = VARIANTS,
) -> Path:
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown evaluation variants: {sorted(unknown)}")
    controller = load_controller(checkpoint, config.device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for front_count in range(9):
        for replicate in range(config.episodes_per_front_count):
            episode_seed = config.seed + front_count * 1000 + replicate
            for variant in variants:
                row = _episode(controller, config, variant, episode_seed, front_count)
                rows.append(row)
                print(json.dumps(row))
    csv_path = output / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {"config": asdict(config), "variants": {}}
    metrics = [
        key for key, value in rows[0].items() if key not in ("variant", "seed") and isinstance(value, (int, float))
    ]
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary["variants"][variant] = {
            metric: {"mean": float(np.mean([row[metric] for row in selected])), "std": float(np.std([row[metric] for row in selected]))}
            for metric in metrics
        }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path

"""Common-leader fixed-horizon benchmark for platoon follower controllers.

The benchmark follows the comparison design of Wang et al. (2025): every
controller receives the same measured leader excitation for the same duration.
It complements, rather than replaces, the full-route survival benchmark.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from platoon_control.ngsim import NGSIMWaveDataset
from platoon_sumo.env import (
    FOLLOWER_IDS,
    MAX_ACCELERATION,
    MAX_BRAKING,
    PLATOON_IDS,
    PLATOON_LENGTH,
    REJOIN_GAP_TOLERANCE,
    TARGET_GAP,
    PlatoonParallelEnv,
)

from .common import ComparisonProtocol, FrozenRewardPredictor, reward_fingerprint
from .evaluate import CaccPolicy, ClassicalPolicy, GraphPolicy


CONTROLLERS = ("graph", "cacc", "ddpg", "ppo", "sac")
STEP_LENGTH = 0.1
DEFAULT_HORIZON_STEPS = 120
PROFILE_VERSION = 1


class CommonLeaderEnv(PlatoonParallelEnv):
    """Evaluation-only environment with follower-dependent leader gating disabled."""

    @staticmethod
    def _formation_guard_acceleration(
        gap: float,
        leader_speed: float,
        follower_speed: float,
        previous_acceleration: float,
        step_length: float,
    ) -> None:
        del gap, leader_speed, follower_speed, previous_acceleration, step_length
        return None


@dataclass(frozen=True)
class LeaderProfile:
    profile_id: int
    vehicle_id: int
    start_time_ms: int
    source_speed_mps: tuple[float, ...]
    source_acceleration_mps2: tuple[float, ...]
    target_speed_mps: tuple[float, ...]

    @property
    def steps(self) -> int:
        return len(self.target_speed_mps) - 1


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _profile_payload(profile: LeaderProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "vehicle_id": profile.vehicle_id,
        "start_time_ms": profile.start_time_ms,
        "source_speed_mps": list(profile.source_speed_mps),
        "source_acceleration_mps2": list(profile.source_acceleration_mps2),
        "target_speed_mps": list(profile.target_speed_mps),
    }


def build_profile_manifest(
    ngsim_dir: Path,
    output: Path,
    episodes: int,
    horizon_steps: int,
    seed: int,
) -> Path:
    """Select informative test-split braking waves before any controller is run."""

    dataset = NGSIMWaveDataset(ngsim_dir, split="test")
    candidates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    required = horizon_steps + 1
    for vehicle_id, timestamp in dataset.events:
        track = dataset.store.track(vehicle_id)
        offset = int(np.searchsorted(track["time_ms"], timestamp))
        end = offset + required
        if end > len(track["time_ms"]):
            continue
        times = track["time_ms"][offset:end]
        if len(times) != required or not np.all(np.diff(times) == 100):
            continue
        speed = np.asarray(track["speed"][offset:end], np.float64)
        acceleration = np.asarray(track["acceleration"][offset:end], np.float64)
        if not (np.isfinite(speed).all() and np.isfinite(acceleration).all()):
            continue
        if float(np.ptp(speed)) < 2.0 or float(np.min(acceleration)) > -0.75:
            continue
        candidates.append((int(vehicle_id), int(timestamp), speed, acceleration))
    if len(candidates) < episodes:
        raise RuntimeError(
            f"only {len(candidates)} NGSIM wave windows satisfy the fixed protocol"
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=episodes, replace=False)
    profiles: list[LeaderProfile] = []
    for profile_id, candidate_index in enumerate(selected):
        vehicle_id, timestamp, speed, acceleration = candidates[int(candidate_index)]
        # Preserve the measured speed variation while matching the simulated
        # platoon's 25 m/s initial state. Only an affine offset is applied.
        target = np.clip(25.0 + speed - speed[0], 8.0, 25.0)
        profiles.append(
            LeaderProfile(
                profile_id=profile_id,
                vehicle_id=vehicle_id,
                start_time_ms=timestamp,
                source_speed_mps=tuple(float(value) for value in speed),
                source_acceleration_mps2=tuple(float(value) for value in acceleration),
                target_speed_mps=tuple(float(value) for value in target),
            )
        )

    ngsim_manifest = json.loads(
        (ngsim_dir / "manifest.json").read_text(encoding="utf-8")
    )
    profile_values = [_profile_payload(profile) for profile in profiles]
    payload = {
        "profile_version": PROFILE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "split": "test",
            "source": "NGSIM braking-wave events",
            "seed": seed,
            "episodes": episodes,
            "horizon_steps": horizon_steps,
            "step_length_seconds": STEP_LENGTH,
            "minimum_speed_range_mps": 2.0,
            "minimum_braking_mps2": -0.75,
            "controller_blind_selection": True,
        },
        "ngsim_source_sha256": ngsim_manifest["source_sha256"],
        "profiles": profile_values,
        "profiles_sha256": _canonical_hash(profile_values),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_profiles(path: Path) -> tuple[list[LeaderProfile], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile_version") != PROFILE_VERSION:
        raise ValueError(f"unsupported profile version in {path}")
    if payload["profiles_sha256"] != _canonical_hash(payload["profiles"]):
        raise ValueError(f"profile hash mismatch in {path}")
    allowed = {field.name for field in fields(LeaderProfile)}
    profiles = []
    for item in payload["profiles"]:
        values = {key: value for key, value in item.items() if key in allowed}
        for key in (
            "source_speed_mps",
            "source_acceleration_mps2",
            "target_speed_mps",
        ):
            values[key] = tuple(values[key])
        profiles.append(LeaderProfile(**values))
    return profiles, payload


class LeaderTracker:
    def __init__(self, profile: LeaderProfile) -> None:
        self.profile = profile

    def action(self, step: int, speed: float) -> dict[str, Any]:
        index = min(step, self.profile.steps - 1)
        target_now = self.profile.target_speed_mps[index]
        target_next = self.profile.target_speed_mps[index + 1]
        feedforward = (target_next - target_now) / STEP_LENGTH
        requested = feedforward + 0.8 * (target_now - speed)
        return {
            "accel": np.asarray(
                [np.clip(requested, MAX_BRAKING, MAX_ACCELERATION)], np.float32
            ),
            "lane_change": 0,
        }


def _scenario_options(delay_probability: float) -> dict[str, Any]:
    return {
        "front_vehicle_count": 0,
        "cutin_enabled": False,
        "wave_mode": "off",
        "v2i_enabled": False,
        "speed_reference": 25.0,
        "communication_delay_probability": float(delay_probability),
        "communication_delay_min": 0.1,
        "communication_delay_max": 0.2 if delay_probability > 0.0 else 0.1,
        "background_initial_scale": 0.0,
        "background_demand_scale": 0.0,
    }


def _failure_reason(events: list[dict[str, Any]], final: Mapping[str, Any]) -> str:
    names = [str(event.get("event", "")) for event in events]
    if final.get("collision"):
        return "collision"
    if "leader_stalled_with_clear_path" in names:
        return "leader_stalled_with_clear_path"
    if final.get("failure"):
        return "vehicle_missing"
    if final.get("success"):
        return "route_success"
    return "fixed_horizon_completed"


def _metric_rows(
    controller: str,
    profile: LeaderProfile,
    trace_rows: list[dict[str, Any]],
    final: Mapping[str, Any],
    events: list[dict[str, Any]],
    reward_sum: float,
    inference_ms: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps = len(trace_rows)
    leader_acceleration = np.asarray(
        [row["leader_acceleration_mps2"] for row in trace_rows], np.float64
    )
    leader_norm = max(float(np.linalg.norm(leader_acceleration)), 1e-6)
    follower_rows: list[dict[str, Any]] = []
    for follower in FOLLOWER_IDS:
        acceleration = np.asarray(
            [row[f"{follower}_acceleration_mps2"] for row in trace_rows],
            np.float64,
        )
        speed = np.asarray(
            [row[f"{follower}_speed_mps"] for row in trace_rows], np.float64
        )
        gap = np.asarray(
            [row[f"{follower}_gap_m"] for row in trace_rows], np.float64
        )
        jerk = np.diff(acceleration) / STEP_LENGTH
        flow_variation = float(
            np.sum(np.abs(np.diff(speed)))
            / max(np.sum(gap[1:] + PLATOON_LENGTH), 1e-6)
        )
        follower_rows.append(
            {
                "controller": controller,
                "profile_id": profile.profile_id,
                "vehicle": follower,
                "acceleration_damping_ratio": float(
                    np.linalg.norm(acceleration) / leader_norm
                ),
                "acceleration_mse_m2ps4": float(np.mean(np.square(acceleration))),
                "jerk_rms_mps3": float(
                    np.sqrt(np.mean(np.square(jerk))) if jerk.size else 0.0
                ),
                "flow_variation_index": flow_variation,
                "gap_mae_m": float(np.mean(np.abs(gap - TARGET_GAP))),
                "minimum_gap_m": float(np.min(gap)),
                "maximum_gap_m": float(np.max(gap)),
            }
        )

    target = np.asarray(
        [row["target_leader_speed_mps"] for row in trace_rows], np.float64
    )
    realized = np.asarray(
        [row["leader_speed_mps"] for row in trace_rows], np.float64
    )
    fleet_jerk = []
    for agent in PLATOON_IDS:
        values = np.asarray(
            [row[f"{agent}_acceleration_mps2"] for row in trace_rows], np.float64
        )
        if values.size > 1:
            fleet_jerk.append(np.diff(values) / STEP_LENGTH)
    gap_values = np.concatenate(
        [
            np.asarray([row[f"{agent}_gap_m"] for row in trace_rows], np.float64)
            for agent in FOLLOWER_IDS
        ]
    )
    speeds = np.asarray(
        [
            [row[f"{agent}_speed_mps"] for agent in PLATOON_IDS]
            for row in trace_rows
        ],
        np.float64,
    )
    leader_distance_m = float(np.sum(speeds[:, 0]) * STEP_LENGTH)
    tail_distance_m = float(np.sum(speeds[:, -1]) * STEP_LENGTH)
    fleet_distance_km = float(np.sum(speeds) * STEP_LENGTH / 1000.0)
    formation_connected = np.asarray(
        [
            all(
                row[f"{agent}_gap_m"]
                <= TARGET_GAP + REJOIN_GAP_TOLERANCE
                for agent in FOLLOWER_IDS
            )
            for row in trace_rows
        ],
        np.float64,
    )
    energy_kwh = float(
        np.sum([row["fleet_tractive_power_w"] for row in trace_rows])
        * STEP_LENGTH
        / 3.6e6
    )
    fuel_g = float(
        np.sum([row["fleet_fuel_mg_per_s"] for row in trace_rows])
        * STEP_LENGTH
        / 1000.0
    )
    mean_follower = lambda key: float(np.mean([row[key] for row in follower_rows]))
    episode = {
        "controller": controller,
        "profile_id": profile.profile_id,
        "profile_vehicle_id": profile.vehicle_id,
        "profile_start_time_ms": profile.start_time_ms,
        "steps": steps,
        "duration_s": steps * STEP_LENGTH,
        "horizon_completed": int(steps == profile.steps and not final.get("collision", False)),
        "collision": int(bool(final.get("collision", False))),
        "environment_failure": int(bool(final.get("failure", False))),
        "failure_reason": _failure_reason(events, final),
        "leader_tracking_rmse_mps": float(
            np.sqrt(np.mean(np.square(realized - target)))
        ),
        "mean_acceleration_damping_ratio": mean_follower(
            "acceleration_damping_ratio"
        ),
        "tail_acceleration_damping_ratio": follower_rows[-1][
            "acceleration_damping_ratio"
        ],
        "acceleration_mse_m2ps4": mean_follower("acceleration_mse_m2ps4"),
        "jerk_rms_mps3": float(
            np.sqrt(np.mean(np.square(np.concatenate(fleet_jerk))))
            if fleet_jerk
            else 0.0
        ),
        "tractive_energy_kwh": energy_kwh,
        "fuel_g": fuel_g,
        "specific_energy_kwh_per_vehicle_km": energy_kwh
        / max(fleet_distance_km, 1e-6),
        "fuel_g_per_vehicle_km": fuel_g / max(fleet_distance_km, 1e-6),
        "average_speed_mps": float(np.mean(speeds)),
        "tail_mobility_ratio": tail_distance_m / max(leader_distance_m, 1e-6),
        "formation_connected_ratio": float(np.mean(formation_connected)),
        "formation_service_valid": int(
            float(np.mean(formation_connected)) >= 0.95
            and tail_distance_m / max(leader_distance_m, 1e-6) >= 0.90
        ),
        "flow_variation_index": mean_follower("flow_variation_index"),
        "gap_mae_m": float(np.mean(np.abs(gap_values - TARGET_GAP))),
        "minimum_gap_m": float(np.min(gap_values)),
        "maximum_gap_m": float(np.max(gap_values)),
        "maximum_speed_spread_mps": float(np.max(np.ptp(speeds, axis=1))),
        "shield_intervention_rate": float(
            np.mean(
                [
                    row[f"{agent}_shield_intervened"]
                    for row in trace_rows
                    for agent in PLATOON_IDS
                ]
            )
        ),
        "disconnected_seconds": float(
            sum(
                any(
                    row[f"{agent}_gap_m"]
                    > TARGET_GAP + REJOIN_GAP_TOLERANCE
                    for agent in FOLLOWER_IDS
                )
                for row in trace_rows
            )
            * STEP_LENGTH
        ),
        "environment_reward_per_step": reward_sum / max(steps, 1),
        "inference_p95_ms": float(np.percentile(inference_ms, 95)),
    }
    return episode, follower_rows


def _run_episode(
    policy: ClassicalPolicy | GraphPolicy | CaccPolicy,
    profile: LeaderProfile,
    protocol: ComparisonProtocol,
    predictor: FrozenRewardPredictor,
    delay_probability: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reset_seed = protocol.evaluation_seed + 50_000 + profile.profile_id
    env = CommonLeaderEnv(
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
        max_time=profile.steps * STEP_LENGTH + 2.0,
    )
    observations, _ = env.reset(
        seed=reset_seed, options=_scenario_options(delay_probability)
    )
    policy.reset()
    tracker = LeaderTracker(profile)
    trace_rows: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    reward_sum = 0.0
    final: Mapping[str, Any] = {}
    try:
        for step in range(profile.steps):
            if not env.agents:
                break
            predictor.attach(env, observations)
            started = time.perf_counter()
            actions = policy.act(observations)
            inference_ms.append((time.perf_counter() - started) * 1000.0)
            active = set(env.connection.vehicle.getIDList())
            leader_speed = float(env.connection.vehicle.getSpeed("leader"))
            actions["leader"] = tracker.action(step, leader_speed)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            policy.record_applied(infos)
            reward_sum += float(np.mean(list(rewards.values())))
            final = infos["leader"]
            active = set(env.connection.vehicle.getIDList())
            if not all(agent in active for agent in PLATOON_IDS):
                break
            row: dict[str, Any] = {
                "controller": policy.name,
                "profile_id": profile.profile_id,
                "step": step,
                "time_s": (step + 1) * STEP_LENGTH,
                "target_leader_speed_mps": profile.target_speed_mps[step + 1],
                "fleet_fuel_mg_per_s": float(
                    final["reward_components"].get("fuel_mg_per_second", 0.0)
                ),
            }
            fleet_power = 0.0
            for agent in PLATOON_IDS:
                speed = float(env.connection.vehicle.getSpeed(agent))
                acceleration = float(
                    infos[agent].get(
                        "applied_acceleration",
                        env.connection.vehicle.getAcceleration(agent),
                    )
                )
                row[f"{agent}_speed_mps"] = speed
                row[f"{agent}_acceleration_mps2"] = acceleration
                row[f"{agent}_shield_intervened"] = int(
                    bool(infos[agent].get("safety_intervened", False))
                )
                fleet_power += env._tractive_power(speed, acceleration)
            for follower in FOLLOWER_IDS:
                row[f"{follower}_gap_m"] = float(env._platoon_gap(follower, active))
            row["fleet_tractive_power_w"] = fleet_power
            trace_rows.append(row)
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        events = list(env.event_log)
        env.close()
    episode, follower_rows = _metric_rows(
        policy.name,
        profile,
        trace_rows,
        final,
        events,
        reward_sum,
        inference_ms,
    )
    return episode, follower_rows, trace_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_fixed_horizon(
    controller: str,
    checkpoint: Path,
    output: Path,
    profile_manifest: Path,
    protocol: ComparisonProtocol,
    device: str,
    delay_probability: float,
) -> Path:
    profiles, manifest = load_profiles(profile_manifest)
    policy: ClassicalPolicy | GraphPolicy | CaccPolicy
    if controller == "graph":
        policy = GraphPolicy(checkpoint, device)
    elif controller == "cacc":
        policy = CaccPolicy()
    else:
        policy = ClassicalPolicy(checkpoint, device)
    predictor = FrozenRewardPredictor(protocol.reward_predictor_path, device)
    episode_rows: list[dict[str, Any]] = []
    follower_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for profile in profiles:
        episode, followers, traces = _run_episode(
            policy, profile, protocol, predictor, delay_probability
        )
        episode_rows.append(episode)
        follower_rows.extend(followers)
        trace_rows.extend(traces)
        print(json.dumps(episode, sort_keys=True), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episode_rows)
    _write_csv(output / "per_vehicle_metrics.csv", follower_rows)
    _write_csv(output / "traces.csv", trace_rows)
    numeric = [
        key
        for key, value in episode_rows[0].items()
        if isinstance(value, (int, float))
        and key
        not in {
            "profile_id",
            "profile_vehicle_id",
            "profile_start_time_ms",
        }
    ]
    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "NGSIM common-leader fixed-horizon follower response",
        "controller": controller,
        "checkpoint": str(checkpoint.resolve()),
        "episodes": len(episode_rows),
        "profile_manifest": str(profile_manifest.resolve()),
        "profiles_sha256": manifest["profiles_sha256"],
        "reward_fingerprint": reward_fingerprint(),
        "controller_specific_reward_tuning": False,
        "leader_override_only_during_evaluation": True,
        "delay_probability": delay_probability,
        "metrics": {
            key: {
                "mean": float(np.mean([row[key] for row in episode_rows])),
                "std": float(np.std([row[key] for row in episode_rows])),
            }
            for key in numeric
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller", choices=CONTROLLERS)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        default=Path("artifacts/classical_comparison/fixed_horizon/profiles.json"),
    )
    parser.add_argument("--ngsim", type=Path, default=Path("artifacts/ngsim"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--evaluation-seed", type=int, default=30_000)
    parser.add_argument("--delay-probability", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rebuild-profiles", action="store_true")
    arguments = parser.parse_args()
    if not arguments.profile_manifest.exists() or arguments.rebuild_profiles:
        build_profile_manifest(
            arguments.ngsim,
            arguments.profile_manifest,
            arguments.episodes,
            arguments.horizon_steps,
            arguments.evaluation_seed,
        )
    profiles, _ = load_profiles(arguments.profile_manifest)
    if len(profiles) != arguments.episodes:
        raise ValueError(
            f"manifest contains {len(profiles)} profiles, expected {arguments.episodes}"
        )
    protocol = replace(
        ComparisonProtocol(),
        evaluation_seed=arguments.evaluation_seed,
        evaluation_episodes=arguments.episodes,
    )
    path = evaluate_fixed_horizon(
        arguments.controller,
        arguments.checkpoint,
        arguments.output,
        arguments.profile_manifest,
        protocol,
        arguments.device,
        arguments.delay_probability,
    )
    print(json.dumps({"summary": str(path.resolve()), "complete": True}))


if __name__ == "__main__":
    main()

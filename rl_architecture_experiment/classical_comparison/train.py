"""Train one classical baseline under the immutable comparison protocol."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from platoon_control.rl import RLConfig, _curriculum_options
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv

from .algorithms import ClassicalAlgorithm, create_algorithm
from .common import (
    ActionCodec,
    ComparisonProtocol,
    FrozenRewardPredictor,
    StateAdapter,
    reward_fingerprint,
    shared_step_rewards,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    algorithm: ClassicalAlgorithm,
    name: str,
    protocol: ComparisonProtocol,
    total_steps: int,
    episode: int,
    include_replay: bool = True,
) -> None:
    payload = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": name,
        "state_dim": algorithm.state_dim,
        "action_dim": len(PLATOON_IDS),
        "protocol": protocol.to_dict(),
        "reward_fingerprint": reward_fingerprint(),
        "total_steps": int(total_steps),
        "episode": int(episode),
        "algorithm_state": algorithm.state_dict(include_replay=include_replay),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    algorithm: ClassicalAlgorithm,
    name: str,
    protocol: ComparisonProtocol,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["algorithm"] != name:
        raise ValueError(f"checkpoint algorithm is {checkpoint['algorithm']}, not {name}")
    if checkpoint["reward_fingerprint"] != reward_fingerprint():
        raise ValueError("reward source changed since this baseline checkpoint")
    saved_protocol = checkpoint["protocol"]
    current_protocol = protocol.to_dict()
    for key in (
        "training_seed",
        "action_min",
        "action_max",
        "discount",
        "reward_scale",
        "curriculum_boundaries",
        "team_reward_aggregation",
    ):
        if saved_protocol[key] != current_protocol[key]:
            raise ValueError(f"comparison protocol mismatch for {key}")
    algorithm.load_state_dict(checkpoint["algorithm_state"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    return int(checkpoint["total_steps"]), int(checkpoint["episode"])


def _scenario_options(
    protocol: ComparisonProtocol,
    episode: int,
    total_steps: int,
) -> tuple[int, dict[str, Any]]:
    stage = protocol.stage(total_steps)
    rng = np.random.default_rng(protocol.training_seed + 17_000 + episode)
    config = RLConfig(
        seed=protocol.training_seed,
        episode_max_time=protocol.episode_max_time,
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
    )
    options, _ = _curriculum_options(stage, rng, config, total_steps)
    return stage, options


class EpisodeMonitor:
    def __init__(self) -> None:
        self.steps = 0
        self.raw_reward = 0.0
        self.shaped_reward = 0.0
        self.minimum_gap = float("inf")
        self.maximum_gap = 0.0
        self.maximum_speed_spread = 0.0
        self.shield_count = 0
        self.background_peak = 0
        self.background_slow_runs: dict[str, int] = {}
        self.maximum_background_slow_run = 0
        self.gridlock_run = 0
        self.maximum_gridlock_run = 0
        self.maximum_delay = 0.0
        self.cutin_states: set[str] = set()
        self.final_info: dict[str, Any] = {}

    def observe(
        self,
        env: PlatoonParallelEnv,
        raw_reward: float,
        shaped_reward: float,
        infos: dict[str, dict[str, Any]],
    ) -> None:
        self.steps += 1
        self.raw_reward += raw_reward
        self.shaped_reward += shaped_reward
        self.final_info = infos["leader"]
        active = set(env.connection.vehicle.getIDList())
        gaps = [
            env._platoon_gap(follower, active)
            for follower in FOLLOWER_IDS
            if follower in active
        ]
        gaps = [gap for gap in gaps if gap >= 0.0]
        if gaps:
            self.minimum_gap = min(self.minimum_gap, min(gaps))
            self.maximum_gap = max(self.maximum_gap, max(gaps))
        speeds = [
            float(env.connection.vehicle.getSpeed(agent))
            for agent in PLATOON_IDS
            if agent in active
        ]
        if speeds:
            self.maximum_speed_spread = max(
                self.maximum_speed_spread, max(speeds) - min(speeds)
            )
        self.shield_count += sum(
            int(infos[agent].get("safety_intervened", False))
            for agent in PLATOON_IDS
            if agent in infos
        )
        background = sorted(
            vehicle for vehicle in active if vehicle.startswith("background_")
        )
        self.background_peak = max(self.background_peak, len(background))
        background_speeds = [
            float(env.connection.vehicle.getSpeed(vehicle)) for vehicle in background
        ]
        for vehicle, speed in zip(background, background_speeds):
            run = self.background_slow_runs.get(vehicle, 0) + 1 if speed < 0.2 else 0
            self.background_slow_runs[vehicle] = run
            self.maximum_background_slow_run = max(
                self.maximum_background_slow_run, run
            )
        gridlocked = len(background_speeds) >= 8 and np.mean(background_speeds) < 0.5
        self.gridlock_run = self.gridlock_run + 1 if gridlocked else 0
        self.maximum_gridlock_run = max(self.maximum_gridlock_run, self.gridlock_run)
        self.cutin_states.add(str(infos["leader"].get("cutin_state", "idle")))
        for follower in FOLLOWER_IDS:
            if follower in infos:
                self.maximum_delay = max(
                    self.maximum_delay,
                    float(infos[follower].get("communication_delay", 0.0)),
                    float(
                        infos[follower].get(
                            "predecessor_communication_delay", 0.0
                        )
                    ),
                )

    def result(
        self,
        algorithm: str,
        episode: int,
        total_steps: int,
        stage: int,
        elapsed: float,
        update_metrics: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": algorithm,
            "episode": episode,
            "total_steps": total_steps,
            "curriculum_stage": stage,
            "episode_steps": self.steps,
            "steps_per_second": self.steps / max(elapsed, 1e-6),
            "raw_team_reward_per_step": self.raw_reward / max(self.steps, 1),
            "shared_training_reward_per_step": self.shaped_reward
            / max(self.steps, 1),
            "minimum_gap_m": (
                self.minimum_gap if np.isfinite(self.minimum_gap) else -1.0
            ),
            "maximum_gap_m": self.maximum_gap,
            "maximum_speed_spread_mps": self.maximum_speed_spread,
            "shield_rate": self.shield_count / max(self.steps * len(PLATOON_IDS), 1),
            "background_peak_count": self.background_peak,
            "background_max_stuck_seconds": self.maximum_background_slow_run * 0.1,
            "background_max_gridlock_seconds": self.maximum_gridlock_run * 0.1,
            "maximum_observed_delay_seconds": self.maximum_delay,
            "cutin_states": sorted(self.cutin_states),
            "leader_stalled_failure": bool(
                self.final_info.get("leader_stalled_failure", False)
            ),
            "leader_stall_duration_seconds": float(
                self.final_info.get("leader_stall_duration", 0.0)
            ),
            "success": bool(self.final_info.get("success", False)),
            "collision": bool(self.final_info.get("collision", False)),
            "failure": bool(self.final_info.get("failure", False)),
            **update_metrics,
        }


def train(
    name: str,
    output: Path,
    protocol: ComparisonProtocol,
    device: str,
    resume: Path | None = None,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "protocol.json", protocol.to_dict())
    random.seed(protocol.training_seed)
    np.random.seed(protocol.training_seed)
    torch.manual_seed(protocol.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol.training_seed)
    torch.set_num_threads(2)

    env = PlatoonParallelEnv(
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
        max_time=protocol.episode_max_time,
    )
    predictor = FrozenRewardPredictor(protocol.reward_predictor_path, "cpu")
    adapter = StateAdapter()
    codec = ActionCodec(protocol.action_min, protocol.action_max)
    total_steps = 0
    episode = 0
    stage, options = _scenario_options(protocol, episode, total_steps)
    observations, _ = env.reset(seed=protocol.training_seed + episode, options=options)
    previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
    initial_state = adapter(observations, previous_applied)
    algorithm = create_algorithm(
        name, initial_state.size, protocol, device, protocol.training_seed
    )
    if resume is not None and resume.exists():
        total_steps, episode = _load_checkpoint(
            resume, algorithm, name, protocol
        )
        env.close()
        stage, options = _scenario_options(protocol, episode, total_steps)
        observations, _ = env.reset(
            seed=protocol.training_seed + episode, options=options
        )
        previous_applied.fill(0.0)
        initial_state = adapter(observations, previous_applied)

    metrics_path = output / "metrics.jsonl"
    latest_path = output / "latest.pt"
    next_checkpoint = (
        (total_steps // protocol.checkpoint_interval) + 1
    ) * protocol.checkpoint_interval
    last_update_metrics: dict[str, float] = {}
    try:
        while total_steps < protocol.training_steps:
            stage, options = _scenario_options(protocol, episode, total_steps)
            if episode > 0 or total_steps > 0:
                observations, _ = env.reset(
                    seed=protocol.training_seed + episode, options=options
                )
                previous_applied.fill(0.0)
            monitor = EpisodeMonitor()
            episode_started = time.perf_counter()
            while env.agents and total_steps < protocol.training_steps:
                predictor.attach(env, observations)
                state = adapter(observations, previous_applied)
                sample = algorithm.act(state, deterministic=False, total_steps=total_steps)
                actions = codec.action_dict(sample.action)
                old_observations = observations
                (
                    next_observations,
                    environment_rewards,
                    terminations,
                    truncations,
                    infos,
                ) = env.step(actions)
                raw_vector, shaped_vector = shared_step_rewards(
                    old_observations, actions, environment_rewards, infos
                )
                raw_team_reward = float(raw_vector.mean())
                shaped_team_reward = float(shaped_vector.mean())
                done = bool(
                    any(terminations.values()) or any(truncations.values())
                )
                next_applied = np.asarray(
                    [
                        infos[agent].get(
                            "applied_acceleration",
                            float(actions[agent]["accel"][0]),
                        )
                        for agent in PLATOON_IDS
                    ],
                    np.float32,
                )
                next_state = (
                    adapter(next_observations, next_applied)
                    if next_observations
                    else np.zeros_like(state)
                )
                algorithm.record(
                    state,
                    sample,
                    shaped_team_reward,
                    next_state,
                    done,
                )
                total_steps += 1
                update_metrics = algorithm.update_if_ready(total_steps)
                if update_metrics:
                    if not all(np.isfinite(value) for value in update_metrics.values()):
                        raise FloatingPointError(
                            f"non-finite {name} update at step {total_steps}: {update_metrics}"
                        )
                    last_update_metrics = update_metrics
                monitor.observe(
                    env, raw_team_reward, shaped_team_reward, infos
                )
                previous_applied = next_applied
                observations = next_observations
                if total_steps >= next_checkpoint:
                    _save_checkpoint(
                        latest_path,
                        algorithm,
                        name,
                        protocol,
                        total_steps,
                        episode,
                    )
                    next_checkpoint += protocol.checkpoint_interval
                if done:
                    break
            elapsed = time.perf_counter() - episode_started
            episode += 1
            record = monitor.result(
                name,
                episode,
                total_steps,
                stage,
                elapsed,
                last_update_metrics,
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            _write_json(
                output / "monitor.json",
                {
                    "status": "training",
                    "pid": os.getpid(),
                    "reward_fingerprint": reward_fingerprint(),
                    **record,
                },
            )
            print(json.dumps(record, sort_keys=True), flush=True)
        final_update = algorithm.update_if_ready(total_steps, force=True)
        if final_update:
            last_update_metrics = final_update
        complete_path = output / "complete.pt"
        _save_checkpoint(
            complete_path,
            algorithm,
            name,
            protocol,
            total_steps,
            episode,
        )
        _write_json(
            output / "monitor.json",
            {
                "status": "complete",
                "pid": os.getpid(),
                "algorithm": name,
                "total_steps": total_steps,
                "episode": episode,
                "curriculum_stage": protocol.stage(total_steps),
                "reward_fingerprint": reward_fingerprint(),
                **last_update_metrics,
            },
        )
        return complete_path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("ddpg", "ppo", "sac"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=301_144)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--checkpoint-interval", type=int, default=25_000)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    protocol = replace(
        ComparisonProtocol(),
        training_seed=arguments.seed,
        training_steps=arguments.steps,
        checkpoint_interval=arguments.checkpoint_interval,
    )
    checkpoint = train(
        arguments.algorithm,
        arguments.output,
        protocol,
        arguments.device,
        arguments.resume,
    )
    print(json.dumps({"checkpoint": str(checkpoint), "complete": True}))


if __name__ == "__main__":
    main()

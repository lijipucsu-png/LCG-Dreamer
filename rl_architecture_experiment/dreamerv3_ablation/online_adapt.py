"""Exact-budget Stage-7 online adaptation for one warmed ablation model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from classical_comparison.common import shared_step_rewards
from platoon_control.agents import PLATOON_IDS
from platoon_control.graph_agents import GraphPlatoonController
from platoon_control.graph_rl import (
    GraphDreamerTrainer,
    GraphEpisodeReplay,
    GraphRLConfig,
    _append_graph_transition,
    _new_graph_episode,
)
from platoon_control.models.graph_dreamer import GraphDreamerConfig
from platoon_control.rl import RLConfig, _curriculum_options
from platoon_sumo.env import PlatoonParallelEnv

from .protocol import PROTOCOL, ROOT, variant_by_key
from .train_budgeted import atomic_torch_save, sha256


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def stage7_options(seed: int, episode: int, step: int) -> tuple[int, dict[str, Any]]:
    reset_seed = seed * 100_000 + 60_000 + episode
    rng = np.random.default_rng(seed * 100_000 + 50_000 + episode)
    config = RLConfig(
        seed=seed,
        calibration_path="artifacts/calibration/calibrated_parameters.json",
        wave_model_path="artifacts/checkpoints/wave_best.pt",
        episode_max_time=240.0,
    )
    options, _ = _curriculum_options(7, rng, config, step)
    return reset_seed, options


def load_training_stack(
    checkpoint: Path, device: str
) -> tuple[dict[str, Any], GraphPlatoonController, GraphDreamerTrainer, GraphRLConfig]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = GraphDreamerConfig(**state["graph_model_config"])
    config_values = dict(state["graph_rl_config"])
    config_values["device"] = device
    config = GraphRLConfig(**config_values)
    controller = GraphPlatoonController(
        model_config, device=device, predictor_directory=None, variant="full"
    )
    controller.load_state_dict(state["models"])
    trainer = GraphDreamerTrainer(controller, config)
    trainer.load_state_dict(state["trainer"])
    return state, controller, trainer, config


def _background_health(env: PlatoonParallelEnv) -> tuple[int, float, bool]:
    active = set(env.connection.vehicle.getIDList())
    background = sorted(
        vehicle for vehicle in active if vehicle.startswith("background_")
    )
    speeds = [float(env.connection.vehicle.getSpeed(vehicle)) for vehicle in background]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    gridlocked = len(speeds) >= 8 and mean_speed < 0.5
    return len(background), mean_speed, gridlocked


def run(
    variant_key: str,
    seed: int,
    warmup_checkpoint: Path,
    output_root: Path,
    device: str,
    online_steps: int,
    target_world_updates: int = PROTOCOL.final_world_updates,
    target_actor_updates: int = PROTOCOL.final_actor_updates,
    expected_warmup_world_updates: int = PROTOCOL.offline_world_updates,
    expected_warmup_actor_updates: int = PROTOCOL.offline_actor_updates,
) -> Path:
    PROTOCOL.validate()
    variant = variant_by_key(variant_key)
    state, controller, trainer, config = load_training_stack(
        warmup_checkpoint, device
    )
    ablation = state.get("ablation", {})
    if ablation.get("variant") != variant_key:
        raise RuntimeError(
            f"warm-up variant mismatch: {ablation.get('variant')} != {variant_key}"
        )
    if trainer.world_updates != expected_warmup_world_updates:
        raise RuntimeError("warm-up world update count does not match protocol")
    if trainer.actor_updates != expected_warmup_actor_updates:
        raise RuntimeError("warm-up actor update count does not match protocol")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    replay_path = ROOT / PROTOCOL.replay_path
    replay_state = torch.load(replay_path, map_location="cpu", weights_only=False)
    replay = GraphEpisodeReplay(PROTOCOL.replay_transitions, seed)
    replay.load_state_dict(replay_state)
    replay.rng = np.random.default_rng(seed)

    output = output_root / variant.key / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "online_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    target_world_delta = target_world_updates - trainer.world_updates
    target_actor_delta = target_actor_updates - trainer.actor_updates
    if target_world_delta < 0 or target_actor_delta < 0:
        raise ValueError("final update target cannot be below warm-up count")
    if online_steps <= 0:
        raise ValueError("online_steps must be positive")
    total_steps = 0
    episode_index = 0
    online_world_updates = 0
    online_actor_updates = 0
    started = time.perf_counter()

    while total_steps < online_steps:
        reset_seed, options = stage7_options(seed, episode_index, total_steps)
        env = PlatoonParallelEnv(
            calibration_path=config.calibration_path,
            wave_model_path=config.wave_model_path,
            max_time=config.episode_max_time,
        )
        observations, _ = env.reset(seed=reset_seed, options=options)
        controller.reset()
        initial_compact = controller.adapter(observations)
        episode = _new_graph_episode(initial_compact)
        episode_steps = 0
        min_gap = float("inf")
        max_gap = 0.0
        maximum_background_stuck_run = 0
        background_stuck_run = 0
        maximum_gridlock_run = 0
        gridlock_run = 0
        collision = False
        success = False
        shaped_reward_sum = 0.0
        final_info: dict[str, Any] = {}
        try:
            while env.agents and total_steps < online_steps:
                previous_observations = observations
                actions = controller.act(observations, deterministic=False)
                raw_actions, _ = controller.requested_action_triplet()
                observations, rewards, terminations, truncations, infos = env.step(actions)
                controller.record_applied_actions(infos, actions)
                _, shaped = shared_step_rewards(
                    previous_observations, actions, rewards, infos
                )
                shaped_rewards = {
                    agent: float(shaped[index])
                    for index, agent in enumerate(PLATOON_IDS)
                }
                shaped_reward_sum += float(np.mean(shaped))
                total_steps += 1
                episode_steps += 1
                final_info = dict(infos["leader"])
                collision = collision or bool(final_info.get("collision", False))
                success = success or bool(final_info.get("success", False))
                active = set(env.connection.vehicle.getIDList())
                gaps = [
                    env._platoon_gap(agent, active)
                    for agent in PLATOON_IDS[1:]
                    if agent in active
                ]
                gaps = [gap for gap in gaps if gap >= 0.0]
                if gaps:
                    min_gap = min(min_gap, min(gaps))
                    max_gap = max(max_gap, max(gaps))
                background_count, background_speed, gridlocked = _background_health(env)
                background_stuck_run = (
                    background_stuck_run + 1
                    if background_count >= 8 and background_speed < 0.2
                    else 0
                )
                gridlock_run = gridlock_run + 1 if gridlocked else 0
                maximum_background_stuck_run = max(
                    maximum_background_stuck_run, background_stuck_run
                )
                maximum_gridlock_run = max(maximum_gridlock_run, gridlock_run)
                continued = not (
                    any(terminations.values())
                    or any(truncations.values())
                    or total_steps >= online_steps
                )
                missing_agents = [
                    agent for agent in PLATOON_IDS if agent not in observations
                ]
                if missing_agents:
                    if continued:
                        raise RuntimeError(
                            "non-terminal graph observation is missing controlled "
                            f"agents: {missing_agents}"
                        )
                    # PettingZoo may remove vehicles from the observation mapping
                    # on the terminal step. The continuation mask excludes this
                    # placeholder from latent bootstrapping.
                    next_compact = controller.adapter(previous_observations)
                else:
                    next_compact = controller.adapter(observations)
                _append_graph_transition(
                    episode,
                    next_compact,
                    actions,
                    raw_actions,
                    shaped_rewards,
                    continued,
                    infos,
                    behavior_supervised=False,
                )

                world_due = int(total_steps * target_world_delta / online_steps)
                while online_world_updates < world_due:
                    batch = replay.sample(config.batch_size, config.sequence_length)
                    update_metrics = trainer.train_world(
                        batch, PROTOCOL.communication_dropout
                    )
                    online_world_updates += 1
                    if online_world_updates % 25 == 0:
                        write_jsonl(
                            metrics_path,
                            {
                                "phase": "online_world",
                                "step": total_steps,
                                "update": trainer.world_updates,
                                **update_metrics,
                            },
                        )
                actor_due = int(total_steps * target_actor_delta / online_steps)
                while online_actor_updates < actor_due:
                    batch = replay.sample(config.batch_size, config.sequence_length)
                    update_metrics = trainer.train_actor_critic(
                        batch,
                        PROTOCOL.imagination_horizon,
                        PROTOCOL.communication_dropout,
                    )
                    online_actor_updates += 1
                    if online_actor_updates % 25 == 0:
                        write_jsonl(
                            metrics_path,
                            {
                                "phase": "online_actor_critic",
                                "step": total_steps,
                                "update": trainer.actor_updates,
                                **update_metrics,
                            },
                        )
                if any(terminations.values()) or any(truncations.values()):
                    break
        finally:
            env.close()
        replay.add(episode)
        episode_record = {
            "phase": "online_episode",
            "variant": variant.key,
            "seed": seed,
            "episode": episode_index,
            "reset_seed": reset_seed,
            "episode_steps": episode_steps,
            "total_steps": total_steps,
            "success": int(success),
            "collision": int(collision),
            "shared_reward_per_step": shaped_reward_sum / max(episode_steps, 1),
            "minimum_gap_m": float(min_gap if np.isfinite(min_gap) else -1.0),
            "maximum_gap_m": float(max_gap),
            "background_max_stuck_s": maximum_background_stuck_run * env.step_length,
            "background_max_gridlock_s": maximum_gridlock_run * env.step_length,
            "leader_station_m": float(final_info.get("leader_station", 0.0)),
            "world_updates": trainer.world_updates,
            "actor_updates": trainer.actor_updates,
            "critic_updates": trainer.critic_updates,
            "elapsed_s": time.perf_counter() - started,
        }
        write_jsonl(metrics_path, episode_record)
        print(json.dumps(episode_record), flush=True)
        episode_index += 1

    expected = (
        target_world_updates,
        target_actor_updates,
        target_actor_updates,
    )
    actual = (trainer.world_updates, trainer.actor_updates, trainer.critic_updates)
    if actual != expected:
        raise RuntimeError(f"final optimizer budget mismatch: {actual} != {expected}")
    checkpoint = output / "budgeted_complete.pt"
    payload = {
        **state,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graph_model_config": controller.checkpoint_config(),
        "graph_rl_config": asdict(config),
        "models": controller.state_dict(),
        "trainer": trainer.state_dict(),
        "total_steps": online_steps,
        "episodes": episode_index,
        "curriculum_stage": 7,
        "stage_steps": online_steps,
        "pretrain_updates_completed": PROTOCOL.offline_world_updates,
        "complete": True,
        "ablation": {
            **ablation,
            "phase": "budgeted_complete",
            "online_sumo_steps": online_steps,
        },
    }
    atomic_torch_save(payload, checkpoint)
    manifest = {
        "study": PROTOCOL.study,
        "variant": variant.key,
        "seed": seed,
        "phase": "budgeted_complete",
        "online_sumo_steps": online_steps,
        "episodes": episode_index,
        "world_updates": trainer.world_updates,
        "actor_updates": trainer.actor_updates,
        "critic_updates": trainer.critic_updates,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "protocol_sha256": ablation["protocol_sha256"],
        "shared_replay_sha256": ablation["shared_replay_sha256"],
        "reward_changed": False,
        "elapsed_s": time.perf_counter() - started,
    }
    (output / "budgeted_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--warmup-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--online-steps", type=int, default=PROTOCOL.online_sumo_steps)
    parser.add_argument(
        "--target-world-updates", type=int, default=PROTOCOL.final_world_updates
    )
    parser.add_argument(
        "--target-actor-updates", type=int, default=PROTOCOL.final_actor_updates
    )
    parser.add_argument(
        "--expected-warmup-world-updates",
        type=int,
        default=PROTOCOL.offline_world_updates,
    )
    parser.add_argument(
        "--expected-warmup-actor-updates",
        type=int,
        default=PROTOCOL.offline_actor_updates,
    )
    arguments = parser.parse_args()
    print(
        run(
            arguments.variant,
            arguments.seed,
            arguments.warmup_checkpoint,
            arguments.output_root,
            arguments.device,
            arguments.online_steps,
            arguments.target_world_updates,
            arguments.target_actor_updates,
            arguments.expected_warmup_world_updates,
            arguments.expected_warmup_actor_updates,
        )
    )


if __name__ == "__main__":
    main()

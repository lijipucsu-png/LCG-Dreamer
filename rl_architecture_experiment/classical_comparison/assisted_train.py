"""Capability-gated DDPG/PPO/SAC training with CACC actor initialization.

The reward, observation, action, and simulator protocol remain unchanged. CACC
is used only as an offline actor-initialization dataset and is absent from
validation and deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from platoon_control.rl import RLConfig, _curriculum_options
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions

from .algorithms import ClassicalAlgorithm, DDPG, PPO, SAC, create_algorithm
from .common import (
    ActionCodec,
    ComparisonProtocol,
    FrozenRewardPredictor,
    StateAdapter,
    reward_fingerprint,
    shared_step_rewards,
)
from .train import EpisodeMonitor, _load_checkpoint, _save_checkpoint, _write_json


@dataclass(frozen=True)
class AssistedConfig:
    pretrain_epochs: int = 30
    pretrain_batch_size: int = 512
    pretrain_validation_fraction: float = 0.1
    pretrain_target_mae: float = 0.025
    bc_decay_steps: int = 400_000
    bc_batch_size: int = 256
    bc_offpolicy_interval: int = 2
    validation_interval_steps: int = 20_000
    validation_episodes: int = 3
    final_validation_episodes: int = 5
    final_stage_minimum_steps: int = 100_000
    maximum_stage: int = 7
    dagger_mixture: tuple[float, ...] = (0.75, 0.35, 0.0, 0.0)
    dagger_epochs_per_round: int = 5
    dagger_tail_steps: int = 0
    dagger_tail_repeats: int = 0
    resume_actor_warmup_steps: int = 0
    reset_sac_value_on_resume: bool = False


MINIMUM_STAGE_STEPS = (0, 10_000, 15_000, 15_000, 25_000, 30_000, 30_000, 100_000)
FINAL_UNASSISTED_STAGE = 7
PLATEAU_PROMOTION_MAXIMUM_GAP_M = 40.0


def _assistance_coefficient(
    total_steps: int, stage: int, config: AssistedConfig
) -> float:
    """Return zero throughout the final capability stage for pure RL adaptation."""

    if stage >= FINAL_UNASSISTED_STAGE:
        return 0.0
    return max(0.0, 1.0 - total_steps / max(config.bc_decay_steps, 1))


def _apply_plateau_promotion(
    validation: Mapping[str, Any], stage: int, stage_steps: int
) -> dict[str, Any]:
    """Advance a saturated pre-final curriculum without hiding strict metrics."""

    result = dict(validation)
    result["strict_passed"] = bool(result.get("passed", False))
    minimum_steps = MINIMUM_STAGE_STEPS[stage]
    plateau_eligible = (
        5 <= stage < FINAL_UNASSISTED_STAGE
        and stage_steps >= 2 * minimum_steps
        and float(result.get("success_rate", 0.0)) == 1.0
        and float(result.get("collision_rate", 1.0)) == 0.0
        and float(result.get("failure_rate", 1.0)) == 0.0
        and float(result.get("maximum_background_gridlock_seconds", 1.0)) == 0.0
        and float(result.get("maximum_gap_m", float("inf")))
        <= PLATEAU_PROMOTION_MAXIMUM_GAP_M
    )
    if not result["strict_passed"] and plateau_eligible:
        result["passed"] = True
        result["promotion_reason"] = "functional_plateau"
    else:
        result["promotion_reason"] = (
            "strict_gate" if result["strict_passed"] else "none"
        )
    return result


def _set_stage_assistance(
    algorithm: ClassicalAlgorithm, stage: int
) -> None:
    """Disable the CACC policy anchor before final-stage optimization."""

    if stage >= FINAL_UNASSISTED_STAGE:
        algorithm.anchor_coefficient = 0.0


def _set_stage_optimization(
    algorithm: ClassicalAlgorithm, stage: int
) -> None:
    """Use conservative policy updates after the final distribution shift."""

    if stage < FINAL_UNASSISTED_STAGE:
        return
    if isinstance(algorithm, DDPG):
        algorithm.policy_delay = 4
        algorithm.target_policy_coefficient = 0.0
        algorithm.reference_policy_coefficient = 5.0
        algorithm.normalize_actor_q = True
        for group in algorithm.actor_optimizer.param_groups:
            group["lr"] = 5e-6
    elif isinstance(algorithm, SAC):
        algorithm.policy_delay = 4
        algorithm.reference_policy_coefficient = 5.0
        algorithm.behavior_noise_scale = 0.10
        algorithm.target_entropy = -float(len(PLATOON_IDS))
        algorithm.maximum_alpha = 0.03
        algorithm.log_alpha.data.clamp_(max=math.log(algorithm.maximum_alpha))
        algorithm.normalize_actor_q = True
        for group in algorithm.actor_optimizer.param_groups:
            group["lr"] = 5e-6


def _configure_algorithm(algorithm: ClassicalAlgorithm) -> None:
    """Apply algorithm-specific optimization settings, never reward settings."""

    algorithm.normalizer_frozen = True
    if isinstance(algorithm, DDPG):
        algorithm.actor_learning_starts = 25_000
        algorithm.policy_delay = 2
        algorithm.policy_objective_weight = 0.02
        algorithm.anchor_coefficient = 2.0
        algorithm.exploration_initial = 0.10
        algorithm.exploration_final = 0.02
        algorithm.exploration_decay_steps = 300_000
        for group in algorithm.actor_optimizer.param_groups:
            group["lr"] = 5e-5
    elif isinstance(algorithm, PPO):
        algorithm.actor_learning_starts = 50_000
        algorithm.anchor_coefficient = 1.0
        algorithm.epochs = 2
        algorithm.clip_ratio = 0.05
        algorithm.entropy_coefficient = 0.0
        algorithm.target_kl = 0.01
        assert algorithm.actor.log_std is not None
        algorithm.actor.log_std.data.fill_(-2.0)
        for group in algorithm.optimizer.param_groups:
            group["lr"] = 1e-5
    elif isinstance(algorithm, SAC):
        algorithm.actor_learning_starts = 25_000
        algorithm.policy_delay = 2
        algorithm.policy_objective_weight = 0.02
        algorithm.anchor_coefficient = 2.0
        algorithm.conservative_weight = 0.1
        algorithm.conservative_samples = 4
        algorithm.normalize_actor_q = True
        algorithm.target_entropy = -3.0
        algorithm.maximum_alpha = 0.20
        algorithm.behavior_noise_scale = 0.45
        algorithm.log_alpha.data.fill_(math.log(0.10))
    _set_optimizer_learning_rates(algorithm)


def _set_optimizer_learning_rates(algorithm: ClassicalAlgorithm) -> None:
    """Reapply settings that optimizer state loading otherwise overwrites."""

    if isinstance(algorithm, DDPG):
        for group in algorithm.actor_optimizer.param_groups:
            group["lr"] = 5e-5
        for group in algorithm.critic_optimizer.param_groups:
            group["lr"] = 3e-4
        for group in algorithm.critic2_optimizer.param_groups:
            group["lr"] = 3e-4
    elif isinstance(algorithm, PPO):
        for group in algorithm.optimizer.param_groups:
            group["lr"] = 1e-5
    elif isinstance(algorithm, SAC):
        for group in algorithm.actor_optimizer.param_groups:
            group["lr"] = 5e-5
        for group in algorithm.critic_optimizer.param_groups:
            group["lr"] = 5e-5


def _prepare_resumed_algorithm(
    algorithm: ClassicalAlgorithm,
    total_steps: int,
    config: AssistedConfig,
) -> None:
    """Apply explicit post-checkpoint adaptation settings."""

    _set_optimizer_learning_rates(algorithm)
    if isinstance(algorithm, DDPG):
        algorithm.reset_policy_reference()
    if isinstance(algorithm, SAC):
        algorithm.reset_policy_reference()
    if isinstance(algorithm, SAC) and config.reset_sac_value_on_resume:
        algorithm.reset_value_learning()
    if config.resume_actor_warmup_steps > 0 and hasattr(
        algorithm, "actor_learning_starts"
    ):
        algorithm.actor_learning_starts = max(
            int(algorithm.actor_learning_starts),
            total_steps + config.resume_actor_warmup_steps,
        )


def _actor_prediction(algorithm: ClassicalAlgorithm, state: Tensor) -> Tensor:
    if isinstance(algorithm, DDPG):
        return algorithm.actor(state)
    if isinstance(algorithm, (PPO, SAC)):
        return algorithm.actor.deterministic(state)
    raise TypeError(type(algorithm))


def _actor_optimizer(algorithm: ClassicalAlgorithm) -> torch.optim.Optimizer:
    if isinstance(algorithm, DDPG):
        return algorithm.actor_optimizer
    if isinstance(algorithm, SAC):
        return algorithm.actor_optimizer
    if isinstance(algorithm, PPO):
        return algorithm.optimizer
    raise TypeError(type(algorithm))


def _clone_loss(
    algorithm: ClassicalAlgorithm,
    states: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    prediction = _actor_prediction(algorithm, states)
    loss = F.smooth_l1_loss(prediction, targets)
    if isinstance(algorithm, SAC):
        distribution = algorithm.actor.distribution(states)
        log_std = distribution.scale.log()
        loss = loss + 0.01 * F.mse_loss(
            log_std, torch.full_like(log_std, -2.0)
        )
    return loss, prediction


def _load_teacher(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["reward_fingerprint"] != reward_fingerprint():
        raise ValueError("teacher data reward fingerprint is stale")
    if metadata.get("teacher_used_at_evaluation", True):
        raise ValueError("teacher dataset does not declare evaluation isolation")
    return payload


def _fit_normalizer(algorithm: ClassicalAlgorithm, states: Tensor) -> None:
    algorithm.normalizer_frozen = False
    for start in range(0, len(states), 2_048):
        algorithm.normalizer.update(states[start : start + 2_048].float().numpy())
    algorithm.normalizer_frozen = True


def pretrain_actor(
    algorithm: ClassicalAlgorithm,
    teacher: Mapping[str, Any],
    config: AssistedConfig,
    seed: int,
) -> dict[str, float]:
    states: Tensor = teacher["states"]
    actions: Tensor = teacher["actions"]
    _fit_normalizer(algorithm, states)
    count = len(states)
    rng = np.random.default_rng(seed + 81_000)
    order = rng.permutation(count)
    validation_count = max(1, int(count * config.pretrain_validation_fraction))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    optimizer = _actor_optimizer(algorithm)
    metrics: dict[str, float] = {}
    for epoch in range(config.pretrain_epochs):
        rng.shuffle(training_indices)
        losses = []
        for start in range(0, len(training_indices), config.pretrain_batch_size):
            index = training_indices[start : start + config.pretrain_batch_size]
            state_np = algorithm.normalizer.normalize(states[index].float().numpy())
            state = torch.as_tensor(
                state_np, dtype=torch.float32, device=algorithm.device
            )
            target = actions[index].float().to(algorithm.device)
            loss, _ = _clone_loss(algorithm, state, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            algorithm._clip(algorithm.actor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            validation_np = algorithm.normalizer.normalize(
                states[validation_indices].float().numpy()
            )
            validation_state = torch.as_tensor(
                validation_np, dtype=torch.float32, device=algorithm.device
            )
            validation_target = actions[validation_indices].float().to(
                algorithm.device
            )
            validation_loss, prediction = _clone_loss(
                algorithm, validation_state, validation_target
            )
            validation_mae = float(
                (prediction - validation_target).abs().mean().detach()
            )
        metrics = {
            "pretrain/epoch": float(epoch + 1),
            "pretrain/loss": float(np.mean(losses)),
            "pretrain/validation_loss": float(validation_loss.detach()),
            "pretrain/validation_mae": validation_mae,
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if validation_mae <= config.pretrain_target_mae and epoch >= 4:
            break
    if isinstance(algorithm, DDPG):
        algorithm.actor_target.load_state_dict(algorithm.actor.state_dict())
    return metrics


def _train_clone_epochs(
    algorithm: ClassicalAlgorithm,
    states: Tensor,
    actions: Tensor,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    optimizer = _actor_optimizer(algorithm)
    order = np.arange(len(states))
    metrics: dict[str, float] = {}
    for _ in range(epochs):
        rng.shuffle(order)
        losses = []
        errors = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            normalized = algorithm.normalizer.normalize(
                states[index].float().numpy()
            )
            state = torch.as_tensor(
                normalized, dtype=torch.float32, device=algorithm.device
            )
            target = actions[index].float().to(algorithm.device)
            loss, prediction = _clone_loss(algorithm, state, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            algorithm._clip(algorithm.actor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            errors.append(float((prediction - target).abs().mean().detach()))
        metrics = {
            "dagger/loss": float(np.mean(losses)),
            "dagger/mae": float(np.mean(errors)),
        }
    if isinstance(algorithm, DDPG):
        algorithm.actor_target.load_state_dict(algorithm.actor.state_dict())
    return metrics


def dagger_refine(
    algorithm: ClassicalAlgorithm,
    teacher: dict[str, Any],
    protocol: ComparisonProtocol,
    config: AssistedConfig,
    stage: int = 0,
    episode_start: int = 10_000,
) -> list[dict[str, Any]]:
    """Aggregate teacher labels on states visited by the initialized actor."""

    env = PlatoonParallelEnv(
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
        max_time=protocol.episode_max_time,
    )
    adapter = StateAdapter()
    codec = ActionCodec(protocol.action_min, protocol.action_max)
    rng = np.random.default_rng(protocol.training_seed + 83_000 + stage)
    records: list[dict[str, Any]] = []
    try:
        for round_index, teacher_fraction in enumerate(config.dagger_mixture):
            reset_seed, options = _scenario_options(
                protocol,
                stage,
                episode_start + round_index,
                validation=True,
            )
            observations, _ = env.reset(seed=reset_seed, options=options)
            previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
            collected_states: list[np.ndarray] = []
            collected_actions: list[np.ndarray] = []
            final: dict[str, Any] = {}
            maximum_gap = 0.0
            while env.agents:
                state = adapter(observations, previous_applied)
                teacher_action_dict = baseline_actions(observations)
                teacher_physical = np.asarray(
                    [
                        float(teacher_action_dict[agent]["accel"][0])
                        for agent in PLATOON_IDS
                    ],
                    np.float32,
                )
                collected_states.append(state.copy())
                collected_actions.append(codec.to_normalized(teacher_physical))
                student = algorithm.act(
                    state, deterministic=True, total_steps=0
                ).action
                student_physical = codec.to_physical(student)
                mixed_physical = (
                    teacher_fraction * teacher_physical
                    + (1.0 - teacher_fraction) * student_physical
                )
                actions = codec.action_dict(codec.to_normalized(mixed_physical))
                observations, _, terminations, truncations, infos = env.step(actions)
                final = infos["leader"]
                previous_applied = np.asarray(
                    [
                        float(infos[agent].get("applied_acceleration", 0.0))
                        for agent in PLATOON_IDS
                    ],
                    np.float32,
                )
                active = set(env.connection.vehicle.getIDList())
                gaps = [
                    float(env._platoon_gap(follower, active))
                    for follower in FOLLOWER_IDS
                    if follower in active
                ]
                if gaps:
                    maximum_gap = max(maximum_gap, max(gaps))
                if any(terminations.values()) or any(truncations.values()):
                    break
            new_states = torch.from_numpy(np.stack(collected_states)).to(torch.float16)
            new_actions = torch.from_numpy(np.stack(collected_actions)).to(
                torch.float16
            )
            teacher["states"] = torch.cat((teacher["states"], new_states))
            teacher["actions"] = torch.cat((teacher["actions"], new_actions))
            teacher["stages"] = torch.cat(
                (
                    teacher["stages"],
                    torch.full(
                        (len(new_states),), stage, dtype=teacher["stages"].dtype
                    ),
                )
            )
            tail_count = min(config.dagger_tail_steps, len(new_states))
            if tail_count > 0 and config.dagger_tail_repeats > 0:
                repeated_states = new_states[-tail_count:].repeat(
                    (config.dagger_tail_repeats, 1)
                )
                repeated_actions = new_actions[-tail_count:].repeat(
                    (config.dagger_tail_repeats, 1)
                )
                teacher["states"] = torch.cat(
                    (teacher["states"], repeated_states)
                )
                teacher["actions"] = torch.cat(
                    (teacher["actions"], repeated_actions)
                )
                teacher["stages"] = torch.cat(
                    (
                        teacher["stages"],
                        torch.full(
                            (len(repeated_states),),
                            stage,
                            dtype=teacher["stages"].dtype,
                        ),
                    )
                )
            clone_metrics = _train_clone_epochs(
                algorithm,
                teacher["states"],
                teacher["actions"],
                config.dagger_epochs_per_round,
                config.pretrain_batch_size,
                rng,
            )
            record = {
                "dagger/round": round_index + 1,
                "dagger/reset_seed": reset_seed,
                "dagger/teacher_fraction": teacher_fraction,
                "dagger/new_states": len(new_states),
                "dagger/tail_steps": tail_count,
                "dagger/tail_repeats": config.dagger_tail_repeats,
                "dagger/aggregate_states": len(teacher["states"]),
                "dagger/rollout_success": int(bool(final.get("success", False))),
                "dagger/rollout_collision": int(bool(final.get("collision", False))),
                "dagger/rollout_failure": int(bool(final.get("failure", False))),
                "dagger/maximum_gap_m": maximum_gap,
                **clone_metrics,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        env.close()
    return records


def behavior_clone_update(
    algorithm: ClassicalAlgorithm,
    teacher: Mapping[str, Any],
    coefficient: float,
    batch_size: int,
    rng: np.random.Generator,
    maximum_stage: int,
) -> dict[str, float]:
    if coefficient <= 0.0:
        return {}
    stages = teacher["stages"].numpy()
    eligible = np.flatnonzero(stages <= maximum_stage)
    index = rng.choice(eligible, size=min(batch_size, len(eligible)), replace=True)
    states = teacher["states"][index].float().numpy()
    normalized = algorithm.normalizer.normalize(states)
    state = torch.as_tensor(
        normalized, dtype=torch.float32, device=algorithm.device
    )
    target = teacher["actions"][index].float().to(algorithm.device)
    loss, prediction = _clone_loss(algorithm, state, target)
    weighted_loss = coefficient * loss
    optimizer = _actor_optimizer(algorithm)
    optimizer.zero_grad(set_to_none=True)
    weighted_loss.backward()
    algorithm._clip(algorithm.actor.parameters(), 1.0)
    optimizer.step()
    if isinstance(algorithm, DDPG):
        algorithm._soft_update(algorithm.actor_target, algorithm.actor, algorithm.tau)
    return {
        "bc/coefficient": coefficient,
        "bc/loss": float(loss.detach()),
        "bc/mae": float((prediction - target).abs().mean().detach()),
    }


def _scenario_options(
    protocol: ComparisonProtocol,
    stage: int,
    episode: int,
    validation: bool,
) -> tuple[int, dict[str, Any]]:
    offset = 90_000 if validation else 70_000
    seed = protocol.training_seed + offset + stage * 1_000 + episode
    rng = np.random.default_rng(seed)
    config = RLConfig(
        seed=protocol.training_seed,
        episode_max_time=protocol.episode_max_time,
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
    )
    options, _ = _curriculum_options(stage, rng, config, 0)
    return seed, options


def validate_capability(
    algorithm: ClassicalAlgorithm,
    protocol: ComparisonProtocol,
    stage: int,
    episodes: int,
    episode_start: int = 0,
) -> dict[str, Any]:
    env = PlatoonParallelEnv(
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
        max_time=protocol.episode_max_time,
    )
    codec = ActionCodec(protocol.action_min, protocol.action_max)
    adapter = StateAdapter()
    rows: list[dict[str, Any]] = []
    try:
        for index in range(episode_start, episode_start + episodes):
            reset_seed, options = _scenario_options(
                protocol, stage, index, validation=True
            )
            observations, _ = env.reset(seed=reset_seed, options=options)
            previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
            maximum_gap = 0.0
            maximum_speed_spread = 0.0
            gridlock_run = 0
            maximum_gridlock_run = 0
            final: dict[str, Any] = {}
            steps = 0
            while env.agents:
                state = adapter(observations, previous_applied)
                sample = algorithm.act(state, deterministic=True, total_steps=0)
                actions = codec.action_dict(sample.action)
                observations, _, terminations, truncations, infos = env.step(actions)
                final = infos["leader"]
                previous_applied = np.asarray(
                    [
                        float(infos[agent].get("applied_acceleration", 0.0))
                        for agent in PLATOON_IDS
                    ],
                    np.float32,
                )
                active = set(env.connection.vehicle.getIDList())
                gaps = [
                    float(env._platoon_gap(follower, active))
                    for follower in FOLLOWER_IDS
                    if follower in active
                ]
                if gaps:
                    maximum_gap = max(maximum_gap, max(gaps))
                speeds = [
                    float(env.connection.vehicle.getSpeed(agent))
                    for agent in PLATOON_IDS
                    if agent in active
                ]
                if speeds:
                    maximum_speed_spread = max(
                        maximum_speed_spread, max(speeds) - min(speeds)
                    )
                background_speeds = [
                    float(env.connection.vehicle.getSpeed(vehicle))
                    for vehicle in active
                    if vehicle.startswith("background_")
                ]
                gridlocked = (
                    len(background_speeds) >= 8
                    and float(np.mean(background_speeds)) < 0.5
                )
                gridlock_run = gridlock_run + 1 if gridlocked else 0
                maximum_gridlock_run = max(maximum_gridlock_run, gridlock_run)
                steps += 1
                if any(terminations.values()) or any(truncations.values()):
                    break
            rows.append(
                {
                    "episode": index,
                    "reset_seed": reset_seed,
                    "steps": steps,
                    "success": int(bool(final.get("success", False))),
                    "collision": int(bool(final.get("collision", False))),
                    "failure": int(bool(final.get("failure", False))),
                    "leader_stalled_failure": int(
                        bool(final.get("leader_stalled_failure", False))
                    ),
                    "leader_stall_duration_seconds": float(
                        final.get("leader_stall_duration", 0.0)
                    ),
                    "platoon_intrusion_count": int(
                        final.get("platoon_intrusion_count", 0)
                    ),
                    "collision_ids": list(final.get("collision_ids", [])),
                    "active_platoon_ids": sorted(
                        vehicle for vehicle in active if vehicle in PLATOON_IDS
                    ),
                    "missing_platoon_ids": sorted(set(PLATOON_IDS) - active),
                    "final_pair_gaps_m": [
                        float(env._platoon_gap(follower, active))
                        for follower in FOLLOWER_IDS
                        if follower in active
                    ],
                    "leader_station_m": float(final.get("leader_station", 0.0)),
                    "maximum_gap_m": maximum_gap,
                    "maximum_speed_spread_mps": maximum_speed_spread,
                    "background_gridlock_seconds": maximum_gridlock_run * 0.1,
                }
            )
    finally:
        env.close()
    success_rate = float(np.mean([row["success"] for row in rows]))
    collision_rate = float(np.mean([row["collision"] for row in rows]))
    failure_rate = float(np.mean([row["failure"] for row in rows]))
    maximum_gap = float(max(row["maximum_gap_m"] for row in rows))
    maximum_gridlock = float(
        max(row["background_gridlock_seconds"] for row in rows)
    )
    required_success = 1.0 if episodes <= 3 else 0.8
    passed = bool(
        success_rate >= required_success
        and collision_rate == 0.0
        and failure_rate == 0.0
        and maximum_gridlock == 0.0
        and maximum_gap <= (18.0 if stage <= 3 else 35.0)
    )
    return {
        "stage": stage,
        "episodes": episodes,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "failure_rate": failure_rate,
        "maximum_gap_m": maximum_gap,
        "maximum_speed_spread_mps": float(
            max(row["maximum_speed_spread_mps"] for row in rows)
        ),
        "maximum_background_gridlock_seconds": maximum_gridlock,
        "passed": passed,
        "rows": rows,
    }


def _save_assisted_metadata(
    output: Path,
    name: str,
    stage: int,
    total_steps: int,
    stage_steps: int,
    validation: Mapping[str, Any],
    pretraining: Mapping[str, float],
    config: AssistedConfig,
) -> None:
    _write_json(
        output,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": name,
            "curriculum_stage": stage,
            "total_steps": total_steps,
            "stage_steps": stage_steps,
            "validation": dict(validation),
            "pretraining": dict(pretraining),
            "assisted_config": config.__dict__,
            "teacher_runtime_dependency": False,
            "reward_fingerprint": reward_fingerprint(),
        },
    )


def train_assisted(
    name: str,
    output: Path,
    teacher_path: Path,
    protocol: ComparisonProtocol,
    device: str,
    config: AssistedConfig,
    pretrain_only: bool = False,
    resume: Path | None = None,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    random.seed(protocol.training_seed)
    np.random.seed(protocol.training_seed)
    torch.manual_seed(protocol.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol.training_seed)
    torch.set_num_threads(2)
    teacher = _load_teacher(teacher_path)
    state_dim = int(teacher["metadata"]["state_dim"])
    algorithm = create_algorithm(
        name, state_dim, protocol, device, protocol.training_seed
    )
    _configure_algorithm(algorithm)
    if resume is None:
        pretraining = pretrain_actor(
            algorithm, teacher, config, protocol.training_seed
        )
        dagger_records = dagger_refine(
            algorithm, teacher, protocol, config, stage=0
        )
        if dagger_records:
            pretraining.update(
                {
                    "dagger/rounds": float(len(dagger_records)),
                    "dagger/final_mae": float(dagger_records[-1]["dagger/mae"]),
                    "dagger/final_rollout_success": float(
                        dagger_records[-1]["dagger/rollout_success"]
                    ),
                }
            )
        if isinstance(algorithm, (DDPG, PPO, SAC)):
            algorithm.anchor_actor.load_state_dict(algorithm.actor.state_dict())
        if isinstance(algorithm, PPO):
            for group in algorithm.optimizer.param_groups:
                group["lr"] = 1e-5
        if isinstance(algorithm, DDPG):
            algorithm.actor_target.load_state_dict(algorithm.actor.state_dict())
        pretrained_path = output / "pretrained.pt"
        _save_checkpoint(
            pretrained_path,
            algorithm,
            name,
            protocol,
            total_steps=0,
            episode=0,
            include_replay=False,
        )
        validation = validate_capability(
            algorithm,
            protocol,
            stage=0,
            episodes=config.validation_episodes,
        )
        print(json.dumps({"validation": validation}, sort_keys=True), flush=True)
        _save_assisted_metadata(
            output / "monitor.json",
            name,
            0,
            0,
            0,
            validation,
            pretraining,
            config,
        )
        if pretrain_only:
            return pretrained_path
        total_steps = 0
        episode = 0
        stage = 0
        stage_steps = 0
    else:
        if pretrain_only:
            raise ValueError("--resume cannot be combined with --pretrain-only")
        total_steps, episode = _load_checkpoint(
            resume, algorithm, name, protocol
        )
        _prepare_resumed_algorithm(algorithm, total_steps, config)
        monitor_path = resume.parent / "monitor.json"
        if not monitor_path.exists():
            raise FileNotFoundError(
                f"resume requires curriculum metadata at {monitor_path}"
            )
        monitor_state = json.loads(monitor_path.read_text(encoding="utf-8"))
        if monitor_state.get("algorithm") != name:
            raise ValueError("resume monitor algorithm does not match")
        if monitor_state.get("reward_fingerprint") != reward_fingerprint():
            raise ValueError("resume monitor reward fingerprint is stale")
        if int(monitor_state.get("total_steps", -1)) != total_steps:
            raise ValueError("resume checkpoint and monitor step counts differ")
        stage = int(monitor_state["curriculum_stage"])
        stage_steps = int(monitor_state["stage_steps"])
        validation = _apply_plateau_promotion(
            monitor_state["validation"], stage, stage_steps
        )
        pretraining = dict(monitor_state["pretraining"])
        print(
            json.dumps(
                {
                    "resume": str(resume.resolve()),
                    "total_steps": total_steps,
                    "episode": episode,
                    "stage": stage,
                    "stage_steps": stage_steps,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    env = PlatoonParallelEnv(
        calibration_path=protocol.calibration_path,
        wave_model_path=protocol.wave_model_path,
        max_time=protocol.episode_max_time,
    )
    predictor = FrozenRewardPredictor(protocol.reward_predictor_path, "cpu")
    adapter = StateAdapter()
    codec = ActionCodec(protocol.action_min, protocol.action_max)
    rng = np.random.default_rng(protocol.training_seed + 82_000)
    next_validation = stage_steps + config.validation_interval_steps
    last_update: dict[str, float] = {}
    metrics_path = output / "metrics.jsonl"

    if validation["passed"] and stage < config.maximum_stage:
        _save_checkpoint(
            output / f"best_stage_{stage}.pt",
            algorithm,
            name,
            protocol,
            total_steps,
            episode,
            include_replay=False,
        )
        stage += 1
        stage_steps = 0
        next_validation = config.validation_interval_steps
    elif validation["passed"] and config.maximum_stage == 0:
        complete = output / "complete.pt"
        _save_checkpoint(
            complete,
            algorithm,
            name,
            protocol,
            total_steps,
            episode,
            include_replay=False,
        )
        return complete

    try:
        while total_steps < protocol.training_steps:
            _set_stage_assistance(algorithm, stage)
            _set_stage_optimization(algorithm, stage)
            reset_seed, options = _scenario_options(
                protocol, stage, episode, validation=False
            )
            observations, _ = env.reset(seed=reset_seed, options=options)
            previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
            monitor = EpisodeMonitor()
            started = time.perf_counter()
            while env.agents and total_steps < protocol.training_steps:
                predictor.attach(env, observations)
                state = adapter(observations, previous_applied)
                sample = algorithm.act(
                    state, deterministic=False, total_steps=total_steps
                )
                actions = codec.action_dict(sample.action)
                old_observations = observations
                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
                raw_vector, shaped_vector = shared_step_rewards(
                    old_observations, actions, rewards, infos
                )
                raw_reward = float(raw_vector.mean())
                shaped_reward = float(shaped_vector.mean())
                done = bool(any(terminations.values()) or any(truncations.values()))
                next_applied = np.asarray(
                    [
                        float(
                            infos[agent].get(
                                "applied_acceleration", actions[agent]["accel"][0]
                            )
                        )
                        for agent in PLATOON_IDS
                    ],
                    np.float32,
                )
                next_state = (
                    adapter(observations, next_applied)
                    if observations
                    else np.zeros_like(state)
                )
                algorithm.record(state, sample, shaped_reward, next_state, done)
                total_steps += 1
                stage_steps += 1
                update = algorithm.update_if_ready(total_steps)
                if update:
                    last_update.update(update)
                coefficient = _assistance_coefficient(total_steps, stage, config)
                should_clone = (
                    bool(update) if isinstance(algorithm, PPO) else total_steps % config.bc_offpolicy_interval == 0
                )
                if should_clone and coefficient > 0.0:
                    last_update.update(
                        behavior_clone_update(
                            algorithm,
                            teacher,
                            coefficient,
                            config.bc_batch_size,
                            rng,
                            stage,
                        )
                    )
                if not all(np.isfinite(value) for value in last_update.values()):
                    raise FloatingPointError(
                        f"non-finite assisted update at step {total_steps}: {last_update}"
                    )
                monitor.observe(env, raw_reward, shaped_reward, infos)
                if monitor.steps % 250 == 0:
                    _write_json(
                        output / "heartbeat.json",
                        {
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "algorithm": name,
                            "episode": episode + 1,
                            "curriculum_stage": stage,
                            "total_steps": total_steps,
                            "stage_steps": stage_steps,
                            "episode_steps": monitor.steps,
                            "minimum_gap_m": (
                                monitor.minimum_gap
                                if np.isfinite(monitor.minimum_gap)
                                else -1.0
                            ),
                            "maximum_gap_m": monitor.maximum_gap,
                            "maximum_speed_spread_mps": monitor.maximum_speed_spread,
                            "background_peak_count": monitor.background_peak,
                            "background_max_gridlock_seconds": (
                                monitor.maximum_gridlock_run * env.step_length
                            ),
                            "collision": bool(
                                monitor.final_info.get("collision", False)
                            ),
                            "failure": bool(
                                monitor.final_info.get("failure", False)
                            ),
                            "leader_stalled_failure": bool(
                                monitor.final_info.get(
                                    "leader_stalled_failure", False
                                )
                            ),
                            "leader_stall_duration_seconds": float(
                                monitor.final_info.get("leader_stall_duration", 0.0)
                            ),
                        },
                    )
                previous_applied = next_applied
                if done:
                    break
            episode += 1
            record = monitor.result(
                name,
                episode,
                total_steps,
                stage,
                time.perf_counter() - started,
                last_update,
            )
            record["stage_steps"] = stage_steps
            record["bc_coefficient"] = _assistance_coefficient(
                total_steps, stage, config
            )
            record["pure_rl"] = stage >= FINAL_UNASSISTED_STAGE
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

            minimum = (
                config.final_stage_minimum_steps
                if stage == config.maximum_stage
                else MINIMUM_STAGE_STEPS[stage]
            )
            validation_due = stage_steps >= max(minimum, next_validation)
            if validation_due:
                validation_episodes = (
                    config.final_validation_episodes
                    if stage == config.maximum_stage
                    else config.validation_episodes
                )
                validation = validate_capability(
                    algorithm, protocol, stage, validation_episodes
                )
                validation = _apply_plateau_promotion(
                    validation, stage, stage_steps
                )
                print(
                    json.dumps({"validation": validation}, sort_keys=True),
                    flush=True,
                )
                _save_assisted_metadata(
                    output / "monitor.json",
                    name,
                    stage,
                    total_steps,
                    stage_steps,
                    validation,
                    pretraining,
                    config,
                )
                _save_checkpoint(
                    output / "latest.pt",
                    algorithm,
                    name,
                    protocol,
                    total_steps,
                    episode,
                    include_replay=True,
                )
                next_validation = stage_steps + config.validation_interval_steps
                if validation["passed"]:
                    _save_checkpoint(
                        output / f"best_stage_{stage}.pt",
                        algorithm,
                        name,
                        protocol,
                        total_steps,
                        episode,
                        include_replay=False,
                    )
                    if stage < config.maximum_stage:
                        stage += 1
                        stage_steps = 0
                        next_validation = config.validation_interval_steps
                    elif stage_steps >= config.final_stage_minimum_steps:
                        break
        complete_path = output / "complete.pt"
        _save_checkpoint(
            complete_path,
            algorithm,
            name,
            protocol,
            total_steps,
            episode,
            include_replay=False,
        )
        _save_assisted_metadata(
            output / "monitor.json",
            name,
            stage,
            total_steps,
            stage_steps,
            validation,
            pretraining,
            config,
        )
        return complete_path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("ddpg", "ppo", "sac"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--validation-episodes", type=int, default=3)
    parser.add_argument("--validation-interval", type=int, default=20_000)
    parser.add_argument("--final-stage-minimum", type=int, default=100_000)
    parser.add_argument("--maximum-stage", type=int, default=7)
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-actor-warmup", type=int, default=0)
    parser.add_argument("--reset-sac-value", action="store_true")
    arguments = parser.parse_args()
    protocol = replace(
        ComparisonProtocol(),
        training_seed=arguments.seed,
        training_steps=arguments.steps,
        replay_capacity=50_000,
        checkpoint_interval=50_000,
    )
    config = AssistedConfig(
        pretrain_epochs=arguments.pretrain_epochs,
        validation_episodes=arguments.validation_episodes,
        validation_interval_steps=arguments.validation_interval,
        final_stage_minimum_steps=arguments.final_stage_minimum,
        maximum_stage=arguments.maximum_stage,
        dagger_mixture=AssistedConfig().dagger_mixture[: arguments.dagger_rounds],
        resume_actor_warmup_steps=arguments.resume_actor_warmup,
        reset_sac_value_on_resume=arguments.reset_sac_value,
    )
    checkpoint = train_assisted(
        arguments.algorithm,
        arguments.output,
        arguments.teacher,
        protocol,
        arguments.device,
        config,
        arguments.pretrain_only,
        arguments.resume,
    )
    print(json.dumps({"checkpoint": str(checkpoint.resolve()), "complete": True}))


if __name__ == "__main__":
    main()

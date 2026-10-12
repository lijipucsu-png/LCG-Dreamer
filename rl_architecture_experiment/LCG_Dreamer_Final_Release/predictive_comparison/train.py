"""Full-scenario predictive fine-tuning under the immutable shared reward."""

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

from classical_comparison.common import ActionCodec, reward_fingerprint, shared_step_rewards
from classical_comparison.train import (
    EpisodeMonitor,
    _load_checkpoint,
    _save_checkpoint,
    _scenario_options,
    _write_json,
)
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv

from .algorithms import create_predictive_algorithm, initialize_from_classical_checkpoint
from .prediction import PREDICTION_LAYOUT, SharedPredictiveEncoder
from .protocol import PREDICTIVE_METHODS, PredictiveComparisonProtocol, file_sha256
from .state import NCKnowledgeStateAdapter, PredictiveStateAdapter


def _adapter(
    method: str,
    protocol: PredictiveComparisonProtocol,
    device: str,
) -> PredictiveStateAdapter:
    predictor = SharedPredictiveEncoder(
        protocol.graph_checkpoint, protocol.predictor_directory, device
    )
    if method == "nc_kg_sac_p":
        return NCKnowledgeStateAdapter(predictor, protocol)
    return PredictiveStateAdapter(predictor)


def _prediction_input_norm(
    algorithm: Any, base_dimension: int, prediction_dimension: int
) -> float:
    weights: list[np.ndarray] = []
    actors = (
        list(algorithm.actor.experts)
        if hasattr(algorithm.actor, "experts")
        else [algorithm.actor]
    )
    for actor in actors:
        matrix = actor.network[0].weight.detach().cpu().numpy()
        weights.append(matrix[:, base_dimension : base_dimension + prediction_dimension])
    return float(np.sqrt(np.mean(np.square(np.concatenate(weights, axis=0)))))


def train(
    method: str,
    output: Path,
    protocol: PredictiveComparisonProtocol,
    device: str,
    resume: Path | None = None,
) -> Path:
    if method not in PREDICTIVE_METHODS:
        raise ValueError(method)
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
    adapter = _adapter(method, protocol, device)
    codec = ActionCodec(protocol.action_min, protocol.action_max)
    total_steps = 0
    episode = 0
    stage, options = _scenario_options(protocol, episode, total_steps)
    observations, _ = env.reset(seed=protocol.training_seed + episode, options=options)
    previous_applied = np.zeros(len(PLATOON_IDS), np.float32)
    adapter.reset()
    initial_state = adapter(observations, previous_applied)
    algorithm = create_predictive_algorithm(
        method, initial_state.size, protocol, device, protocol.training_seed
    )
    migration: dict[str, Any]
    if resume is not None and resume.exists():
        total_steps, episode = _load_checkpoint(
            resume, algorithm, method, protocol
        )
        existing_migration = {}
        try:
            existing_migration = json.loads(
                (output / "migration.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            pass
        migration = {
            **existing_migration,
            "resumed_from": str(resume.resolve()),
            "resume_sha256": file_sha256(resume),
        }
        source_total_steps = int(existing_migration.get("source_total_steps", 0))
    else:
        source_path = protocol.source_checkpoint(method)
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if source["reward_fingerprint"] != reward_fingerprint():
            raise ValueError("source policy reward fingerprint is stale")
        migration = initialize_from_classical_checkpoint(algorithm, source)
        source_total_steps = int(source["total_steps"])
        migration.update(
            {
                "source_checkpoint": str(source_path.resolve()),
                "source_checkpoint_sha256": file_sha256(source_path),
                "source_total_steps": source_total_steps,
                "migrated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    migration.update(
        {
            "method": method,
            "prediction_schema": protocol.prediction_schema,
            "prediction_layout": PREDICTION_LAYOUT.to_dict(),
            "reward_fingerprint": reward_fingerprint(),
            "adaptation_steps": protocol.training_steps,
            "total_interaction_budget": protocol.total_interaction_budget,
            "expected_total_interaction_steps": (
                source_total_steps + protocol.training_steps
            ),
            "interaction_budget_matched": (
                source_total_steps + protocol.training_steps
                == protocol.total_interaction_budget
            ),
        }
    )
    _write_json(output / "migration.json", migration)

    stage, options = _scenario_options(protocol, episode, total_steps)
    observations, _ = env.reset(
        seed=protocol.training_seed + episode, options=options
    )
    previous_applied.fill(0.0)
    adapter.reset()
    adapter(observations, previous_applied)

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
                adapter.reset()
            monitor = EpisodeMonitor()
            episode_started = time.perf_counter()
            while env.agents and total_steps < protocol.training_steps:
                state = adapter(observations, previous_applied)
                adapter.predictor.attach_reward_prediction(env)
                sample = algorithm.act(
                    state, deterministic=False, total_steps=total_steps
                )
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
                done = bool(any(terminations.values()) or any(truncations.values()))
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
                adapter.record(infos, actions)
                next_state = (
                    adapter(next_observations, next_applied)
                    if next_observations
                    else np.zeros_like(state)
                )
                algorithm.record(
                    state,
                    sample,
                    float(shaped_vector.mean()),
                    next_state,
                    done,
                )
                total_steps += 1
                update_metrics = algorithm.update_if_ready(total_steps)
                if update_metrics:
                    if not all(np.isfinite(value) for value in update_metrics.values()):
                        raise FloatingPointError(
                            f"non-finite {method} update at {total_steps}: {update_metrics}"
                        )
                    last_update_metrics = update_metrics
                monitor.observe(
                    env,
                    float(raw_vector.mean()),
                    float(shaped_vector.mean()),
                    infos,
                )
                if total_steps % 250 == 0:
                    live = monitor.result(
                        method,
                        episode + 1,
                        total_steps,
                        stage,
                        time.perf_counter() - episode_started,
                        last_update_metrics,
                    )
                    live["prediction_input_weight_rms"] = _prediction_input_norm(
                        algorithm,
                        int(adapter.base_dimension or 0),
                        PREDICTION_LAYOUT.dimension,
                    )
                    live["source_total_steps"] = source_total_steps
                    live["total_interaction_steps"] = (
                        source_total_steps + total_steps
                    )
                    live["estimated_training_seconds_remaining"] = (
                        (protocol.training_steps - total_steps)
                        / max(float(live["steps_per_second"]), 1e-6)
                    )
                    _write_json(
                        output / "monitor.json",
                        {
                            "status": "training",
                            "pid": os.getpid(),
                            "reward_fingerprint": reward_fingerprint(),
                            "prediction_schema": protocol.prediction_schema,
                            **live,
                        },
                    )
                previous_applied = next_applied
                observations = next_observations
                if total_steps >= next_checkpoint:
                    _save_checkpoint(
                        latest_path,
                        algorithm,
                        method,
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
                method,
                episode,
                total_steps,
                stage,
                elapsed,
                last_update_metrics,
            )
            record["prediction_input_weight_rms"] = _prediction_input_norm(
                algorithm,
                int(adapter.base_dimension or 0),
                PREDICTION_LAYOUT.dimension,
            )
            record["source_total_steps"] = source_total_steps
            record["total_interaction_steps"] = source_total_steps + total_steps
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            _write_json(
                output / "monitor.json",
                {
                    "status": "training",
                    "pid": os.getpid(),
                    "reward_fingerprint": reward_fingerprint(),
                    "prediction_schema": protocol.prediction_schema,
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
            method,
            protocol,
            total_steps,
            episode,
        )
        _write_json(
            output / "monitor.json",
            {
                "status": "complete",
                "pid": os.getpid(),
                "algorithm": method,
                "total_steps": total_steps,
                "episode": episode,
                "curriculum_stage": stage,
                "reward_fingerprint": reward_fingerprint(),
                "prediction_schema": protocol.prediction_schema,
                "source_total_steps": source_total_steps,
                "total_interaction_steps": source_total_steps + total_steps,
                "prediction_input_weight_rms": _prediction_input_norm(
                    algorithm,
                    int(adapter.base_dimension or 0),
                    PREDICTION_LAYOUT.dimension,
                ),
                **last_update_metrics,
            },
        )
        return complete_path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=PREDICTIVE_METHODS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--total-budget", type=int, default=301_144)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    base_protocol = PredictiveComparisonProtocol(
        training_seed=arguments.seed,
        total_interaction_budget=arguments.total_budget,
    )
    source = torch.load(
        base_protocol.source_checkpoint(arguments.method),
        map_location="cpu",
        weights_only=False,
    )
    adaptation_steps = (
        arguments.steps
        if arguments.steps is not None
        else arguments.total_budget - int(source["total_steps"])
    )
    if adaptation_steps <= 0:
        raise ValueError("total interaction budget must exceed source steps")
    protocol = replace(
        base_protocol,
        training_seed=arguments.seed,
        training_steps=adaptation_steps,
        checkpoint_interval=arguments.checkpoint_interval,
    )
    checkpoint = train(
        arguments.method,
        arguments.output,
        protocol,
        arguments.device,
        arguments.resume,
    )
    print(json.dumps({"checkpoint": str(checkpoint), "complete": True}))


if __name__ == "__main__":
    main()

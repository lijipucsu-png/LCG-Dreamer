"""Refine a graph actor from a frozen world model and an existing replay."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from platoon_control.graph_agents import GraphPlatoonController
from platoon_control.graph_rl import (
    GRAPH_ACTOR_OBJECTIVE_VERSION,
    GraphDreamerTrainer,
    GraphEpisodeReplay,
    GraphRLConfig,
    _atomic_torch_save,
)
from platoon_control.models.graph_dreamer import GraphDreamerConfig


def run(
    checkpoint_path: Path,
    replay_path: Path,
    output_path: Path,
    metrics_path: Path,
    updates: int,
    report_every: int,
    save_every: int,
    device: str,
    reset_optimizer: bool,
    actor_learning_rate: float | None,
    imagination_weight: float | None,
    foundation_event_fraction: float,
    observed_prior_weight: float | None,
    behavior_initial_weight: float | None,
    physical_residual_only: bool,
    horizon: int,
    seed: int | None,
) -> None:
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(json.dumps({"phase": "checkpoint_loaded", "seconds": time.perf_counter() - started}), flush=True)
    replay_state = torch.load(replay_path, map_location="cpu", weights_only=False)
    print(json.dumps({"phase": "replay_loaded", "seconds": time.perf_counter() - started}), flush=True)
    config = GraphRLConfig(**checkpoint["graph_rl_config"])
    config.device = device
    if behavior_initial_weight is not None:
        config.behavior_initial_weight = behavior_initial_weight
    if actor_learning_rate is not None:
        config.actor_learning_rate = actor_learning_rate
    if observed_prior_weight is not None:
        config.observed_prior_weight = observed_prior_weight
    effective_imagination_weight = (
        config.imagination_weight
        if imagination_weight is None
        else imagination_weight
    )
    refinement_seed = config.seed if seed is None else seed
    random.seed(refinement_seed)
    np.random.seed(refinement_seed)
    torch.manual_seed(refinement_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(refinement_seed)
    model_config = GraphDreamerConfig(**checkpoint["graph_model_config"])
    controller = GraphPlatoonController(
        model_config,
        device,
        variant=config.variant,
    )
    controller.load_state_dict(checkpoint["models"])
    trainer = GraphDreamerTrainer(controller, config)
    trainer.load_state_dict(checkpoint["trainer"])
    if reset_optimizer:
        trainer.optimizers["actor"].state.clear()
        trainer.optimizers["critic"].state.clear()
    if physical_residual_only:
        for name, parameter in controller.model.actor.named_parameters():
            parameter.requires_grad_(name.startswith("physical_trunk."))
    if actor_learning_rate is not None:
        for name in ("actor", "critic"):
            for group in trainer.optimizers[name].param_groups:
                group["lr"] = actor_learning_rate
    replay = GraphEpisodeReplay(config.replay_capacity, config.seed)
    replay.load_state_dict(replay_state)
    # The online checkpoints contain different replay RNG positions. Resetting
    # them here gives paired structural variants the same sampling protocol.
    replay.rng = np.random.default_rng(refinement_seed)
    print(json.dumps({"phase": "runtime_ready", "seconds": time.perf_counter() - started}), flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float]] = []
    for update in range(1, updates + 1):
        batch = replay.sample(
            config.batch_size,
            config.sequence_length,
            foundation_event_fraction,
        )
        update_started = time.perf_counter()
        metrics = trainer.train_actor_critic(
            batch, horizon, 0.0, effective_imagination_weight
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"non-finite actor metric at refinement update {update}")
        record = {
            "refinement/update": float(update),
            "refinement/update_seconds": time.perf_counter() - update_started,
            **metrics,
        }
        records.append(record)
        if update == 1 or update % report_every == 0 or update == updates:
            print(json.dumps(record), flush=True)
        if update % save_every == 0 or update == updates:
            refined = copy.deepcopy(checkpoint)
            refined.update(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "models": controller.state_dict(),
                    "trainer": trainer.state_dict(),
                    "graph_model_config": controller.checkpoint_config(),
                    "graph_rl_config": asdict(config),
                    "actor_objective_version": GRAPH_ACTOR_OBJECTIVE_VERSION,
                    "complete": False,
                    "actor_refinement": {
                        "source": str(checkpoint_path.resolve()),
                        "source_replay": str(replay_path.resolve()),
                        "updates": update,
                        "actor_updates_total": trainer.actor_updates,
                        "optimizer_reset": reset_optimizer,
                        "actor_learning_rate": config.actor_learning_rate,
                        "imagination_weight": effective_imagination_weight,
                        "foundation_event_fraction": foundation_event_fraction,
                        "observed_prior_weight": config.observed_prior_weight,
                        "behavior_initial_weight": config.behavior_initial_weight,
                        "physical_residual_only": physical_residual_only,
                        "horizon": horizon,
                        "seed": refinement_seed,
                        "reward_changed": False,
                        "world_model_updated": False,
                    },
                }
            )
            intermediate = output_path.with_name(
                f"{output_path.stem}_u{update:05d}{output_path.suffix}"
            )
            _atomic_torch_save(refined, intermediate)
            if update == updates:
                _atomic_torch_save(refined, output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--actor-learning-rate", type=float)
    parser.add_argument("--imagination-weight", type=float)
    parser.add_argument("--foundation-event-fraction", type=float, default=0.0)
    parser.add_argument("--observed-prior-weight", type=float)
    parser.add_argument("--behavior-initial-weight", type=float)
    parser.add_argument("--physical-residual-only", action="store_true")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    run(
        arguments.checkpoint,
        arguments.replay,
        arguments.output,
        arguments.metrics,
        arguments.updates,
        arguments.report_every,
        arguments.save_every,
        arguments.device,
        arguments.reset_optimizer,
        arguments.actor_learning_rate,
        arguments.imagination_weight,
        arguments.foundation_event_fraction,
        arguments.observed_prior_weight,
        arguments.behavior_initial_weight,
        arguments.physical_residual_only,
        arguments.horizon,
        arguments.seed,
    )


if __name__ == "__main__":
    main()

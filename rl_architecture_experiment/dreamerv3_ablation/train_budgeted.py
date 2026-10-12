"""Run the fixed-update warm-up for one DreamerV3 ablation variant."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from platoon_control.graph_agents import GraphPlatoonController
from platoon_control.graph_rl import GraphDreamerTrainer, GraphEpisodeReplay, GraphRLConfig

from .protocol import PROTOCOL, ROOT, VARIANTS, variant_by_key


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def parameter_counts(trainer: GraphDreamerTrainer) -> dict[str, int]:
    model = trainer.controller.model
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "world": sum(parameter.numel() for parameter in trainer.world_parameters),
        "actor": sum(parameter.numel() for parameter in trainer.actor_parameters),
        "critic": sum(parameter.numel() for parameter in trainer.critic_parameters),
    }


def checkpoint_payload(
    controller: GraphPlatoonController,
    trainer: GraphDreamerTrainer,
    config: GraphRLConfig,
    variant_key: str,
    replay_path: Path,
    replay_hash: str,
    protocol_hash: str,
    complete: bool,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "control_architecture": controller.architecture_id,
        "graph_model_config": controller.checkpoint_config(),
        "graph_rl_config": asdict(config),
        "models": controller.state_dict(),
        "trainer": trainer.state_dict(),
        "total_steps": 0,
        "episodes": 0,
        "curriculum_stage": 7,
        "stage_steps": 0,
        "capability_records": [],
        "foundation_passed": True,
        "pretrain_updates_completed": trainer.world_updates,
        "actor_objective_version": 11,
        "environment_dynamics_schema_version": 2,
        "restart_metric_schema_version": 2,
        "complete": complete,
        "ablation": {
            "study": PROTOCOL.study,
            "variant": variant_key,
            "phase": "shared_replay_warmup",
            "protocol_sha256": protocol_hash,
            "shared_replay": str(replay_path.resolve()),
            "shared_replay_sha256": replay_hash,
        },
    }


def run(
    variant_key: str,
    seed: int,
    output_root: Path,
    device: str,
    world_updates: int,
    actor_updates: int,
    batch_size: int,
    sequence_length: int,
) -> Path:
    PROTOCOL.validate()
    variant = variant_by_key(variant_key)
    if world_updates < 0 or actor_updates < 0:
        raise ValueError("update counts must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    output = output_root / variant.key / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = PROTOCOL.write(output_root / "protocol.json")
    protocol_manifest = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = str(protocol_manifest["protocol_sha256"])
    replay_path = ROOT / PROTOCOL.replay_path
    replay_hash = sha256(replay_path)
    replay_state = torch.load(replay_path, map_location="cpu", weights_only=False)
    replay = GraphEpisodeReplay(PROTOCOL.replay_transitions, seed)
    replay.load_state_dict(replay_state)
    # Every architecture sees the same minibatch sequence for a given seed.
    replay.rng = np.random.default_rng(seed)

    config = GraphRLConfig(
        seed=seed,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
        checkpoint_dir=str(output),
        variant="full",
        behavior_initial_weight=3.0,
        behavior_decay_updates=max(actor_updates, 1),
        behavior_cutoff_updates=max(actor_updates, 1),
    )
    controller = GraphPlatoonController(
        variant.model_config(),
        device=device,
        predictor_directory=None,
        variant="full",
    )
    trainer = GraphDreamerTrainer(controller, config)
    counts = parameter_counts(trainer)
    metrics_path = output / "training_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    started = time.perf_counter()

    for update in range(world_updates):
        batch = replay.sample(batch_size, sequence_length)
        metrics = trainer.train_world(batch, PROTOCOL.communication_dropout)
        if update == 0 or (update + 1) % 25 == 0 or update + 1 == world_updates:
            append_jsonl(
                metrics_path,
                {
                    "phase": "world",
                    "update": update + 1,
                    "elapsed_s": time.perf_counter() - started,
                    **metrics,
                },
            )
            print(
                json.dumps(
                    {
                        "variant": variant.key,
                        "seed": seed,
                        "phase": "world",
                        "update": update + 1,
                        "loss": metrics["graph/world_optimized_loss"],
                    }
                ),
                flush=True,
            )

    for update in range(actor_updates):
        batch = replay.sample(batch_size, sequence_length)
        metrics = trainer.train_actor_critic(
            batch,
            PROTOCOL.imagination_horizon,
            PROTOCOL.communication_dropout,
        )
        if update == 0 or (update + 1) % 25 == 0 or update + 1 == actor_updates:
            append_jsonl(
                metrics_path,
                {
                    "phase": "actor_critic",
                    "update": update + 1,
                    "elapsed_s": time.perf_counter() - started,
                    **metrics,
                },
            )
            print(
                json.dumps(
                    {
                        "variant": variant.key,
                        "seed": seed,
                        "phase": "actor_critic",
                        "update": update + 1,
                        "actor_loss": metrics["graph/actor_loss"],
                        "critic_loss": metrics["graph/critic_loss"],
                    }
                ),
                flush=True,
            )

    if (
        trainer.world_updates != world_updates
        or trainer.actor_updates != actor_updates
        or trainer.critic_updates != actor_updates
    ):
        raise RuntimeError(
            "optimizer budget mismatch: "
            f"world={trainer.world_updates}, actor={trainer.actor_updates}, "
            f"critic={trainer.critic_updates}"
        )
    checkpoint = output / "warmup_complete.pt"
    atomic_torch_save(
        checkpoint_payload(
            controller,
            trainer,
            config,
            variant.key,
            replay_path,
            replay_hash,
            protocol_hash,
            True,
        ),
        checkpoint,
    )
    manifest = {
        "study": PROTOCOL.study,
        "variant": asdict(variant),
        "seed": seed,
        "phase": "shared_replay_warmup",
        "world_updates": trainer.world_updates,
        "actor_updates": trainer.actor_updates,
        "critic_updates": trainer.critic_updates,
        "parameters": counts,
        "elapsed_s": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "shared_replay": str(replay_path.resolve()),
        "shared_replay_sha256": replay_hash,
        "protocol_sha256": protocol_hash,
        "reward_changed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=[item.key for item in VARIANTS])
    parser.add_argument("--seed", type=int, default=PROTOCOL.debug_seed)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-updates", type=int, default=PROTOCOL.offline_world_updates)
    parser.add_argument("--actor-updates", type=int, default=PROTOCOL.offline_actor_updates)
    parser.add_argument("--batch-size", type=int, default=PROTOCOL.batch_size)
    parser.add_argument("--sequence-length", type=int, default=PROTOCOL.sequence_length)
    arguments = parser.parse_args()
    print(
        run(
            arguments.variant,
            arguments.seed,
            arguments.output_root,
            arguments.device,
            arguments.world_updates,
            arguments.actor_updates,
            arguments.batch_size,
            arguments.sequence_length,
        )
    )


if __name__ == "__main__":
    main()

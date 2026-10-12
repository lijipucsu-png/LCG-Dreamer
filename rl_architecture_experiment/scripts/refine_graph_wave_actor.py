"""Refine only the Graph actor on wave events already present in replay."""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from platoon_control.graph_agents import GraphPlatoonController
from platoon_control.graph_rl import (
    GRAPH_ACTOR_OBJECTIVE_VERSION,
    GraphDreamerTrainer,
    GraphEpisodeReplay,
    GraphRLConfig,
)
from platoon_control.models.graph_dreamer import GraphDreamerConfig


def _dataclass_values(cls: type[Any], values: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


def _atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    arguments = parser.parse_args()
    if arguments.updates <= 0 or arguments.save_every <= 0:
        raise ValueError("updates and save-every must be positive")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    checkpoint = torch.load(
        arguments.checkpoint, map_location="cpu", weights_only=False
    )
    model_config = GraphDreamerConfig(
        **_dataclass_values(
            GraphDreamerConfig, dict(checkpoint["graph_model_config"])
        )
    )
    rl_config = GraphRLConfig(
        **_dataclass_values(
            GraphRLConfig, dict(checkpoint.get("graph_rl_config", {}))
        )
    )
    rl_config = replace(
        rl_config,
        device=arguments.device,
        checkpoint_dir=str(arguments.output),
    )
    controller = GraphPlatoonController(
        model_config,
        arguments.device,
        predictor_directory=rl_config.predictor_directory,
        variant=rl_config.variant,
    )
    controller.load_state_dict(checkpoint["models"])
    trainer = GraphDreamerTrainer(controller, rl_config)
    trainer.load_state_dict(checkpoint["trainer"])

    replay_path = arguments.replay or arguments.checkpoint.with_name(
        "graph_replay_latest.pt"
    )
    replay_state = torch.load(replay_path, map_location="cpu", weights_only=False)
    replay = GraphEpisodeReplay(rl_config.replay_capacity, arguments.seed)
    replay.load_state_dict(replay_state)
    event_counts = {
        name: len(events)
        for name, events in replay._foundation_event_candidates().items()
    }
    if event_counts.get("wave", 0) < rl_config.batch_size:
        raise RuntimeError(
            f"insufficient replay wave events: {event_counts.get('wave', 0)}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output / "wave_actor_refinement.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event_counts": event_counts}, sort_keys=True) + "\n")
        for update in range(1, arguments.updates + 1):
            batch = replay.sample(
                rl_config.batch_size,
                rl_config.sequence_length,
                foundation_event_fraction=1.0,
                event_name="wave",
            )
            metrics = trainer.train_actor_critic(
                batch,
                horizon=30,
                communication_dropout=0.0,
                imagination_weight=0.0,
            )
            record = {"update": update, **metrics}
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            if update % arguments.save_every == 0 or update == arguments.updates:
                payload = copy.deepcopy(checkpoint)
                payload.update(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "graph_model_config": controller.checkpoint_config(),
                        "graph_rl_config": {
                            **dict(checkpoint.get("graph_rl_config", {})),
                            "checkpoint_dir": str(arguments.output),
                        },
                        "models": controller.state_dict(),
                        "trainer": trainer.state_dict(),
                        "actor_objective_version": GRAPH_ACTOR_OBJECTIVE_VERSION,
                        "complete": False,
                        "wave_actor_refinement": {
                            "source_checkpoint": str(arguments.checkpoint.resolve()),
                            "source_replay": str(replay_path.resolve()),
                            "updates": update,
                            "event_counts": event_counts,
                            "reward_changed": False,
                            "world_model_updated": False,
                            "critic_updated": False,
                        },
                    }
                )
                _atomic_save(
                    payload,
                    arguments.output / f"graph_wave_actor_{update:04d}.pt",
                )
            if update == 1 or update % 10 == 0:
                print(json.dumps(record, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "updates": arguments.updates,
                "event_counts": event_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

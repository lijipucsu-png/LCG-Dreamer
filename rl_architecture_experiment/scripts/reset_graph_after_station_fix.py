"""Warm-start graph models while discarding discontinuous-coordinate training state."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from platoon_control.graph_rl import (
    GRAPH_ACTOR_OBJECTIVE_VERSION,
    GRAPH_ENVIRONMENT_DYNAMICS_SCHEMA_VERSION,
    GraphEpisodeReplay,
    _atomic_torch_save,
)


def reset_checkpoint(source: Path, output: Path, seed: int) -> dict[str, object]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("control_architecture") != "ma_latent_graph_dreamer_v1":
        raise ValueError("source is not an MA-Latent-Graph-Dreamer checkpoint")

    trainer = copy.deepcopy(checkpoint["trainer"])
    for optimizer in trainer["optimizers"].values():
        optimizer["state"] = {}
    trainer["actor_updates"] = 0

    config = copy.deepcopy(checkpoint["graph_rl_config"])
    config["seed"] = int(seed)
    config["checkpoint_dir"] = str(output)
    replay = GraphEpisodeReplay(int(config["replay_capacity"]), seed)

    migrated = copy.deepcopy(checkpoint)
    migrated.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trainer": trainer,
            "graph_rl_config": config,
            "total_steps": 0,
            "episodes": 0,
            "curriculum_stage": 0,
            "stage_steps": 0,
            "capability_records": [],
            "foundation_passed": False,
            "pretrain_updates_completed": 0,
            "actor_objective_version": GRAPH_ACTOR_OBJECTIVE_VERSION,
            "environment_dynamics_schema_version": (
                GRAPH_ENVIRONMENT_DYNAMICS_SCHEMA_VERSION
            ),
            "complete": False,
            "station_continuity_migration": {
                "source": str(source.resolve()),
                "source_total_steps": int(checkpoint["total_steps"]),
                "source_curriculum_stage": int(checkpoint["curriculum_stage"]),
                "models_preserved": True,
                "optimizer_states_reset": True,
                "actor_update_count_reset": True,
                "replay_reset": True,
                "curriculum_reset": True,
                "reason": (
                    "SUMO internal junction lanes were absent from the prior "
                    "route coordinate, producing false 3 m and 8 m gap jumps"
                ),
            },
        }
    )

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "graph_latest.pt"
    replay_path = output / "graph_replay_latest.pt"
    _atomic_torch_save(migrated, checkpoint_path)
    _atomic_torch_save(replay.state_dict(), replay_path)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "replay": str(replay_path),
        **migrated["station_continuity_migration"],
    }
    (output / "station_continuity_migration.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            reset_checkpoint(arguments.source, arguments.output, arguments.seed)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

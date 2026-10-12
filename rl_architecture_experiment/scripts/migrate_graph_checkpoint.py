"""Merge a validated graph-actor candidate into a resumable checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from platoon_control.graph_rl import (
    GRAPH_ACTOR_OBJECTIVE_VERSION,
    GraphRLConfig,
    _atomic_torch_save,
    load_graph_controller,
)


def migrate(
    base_path: Path,
    candidate_path: Path,
    foundation_report_path: Path,
    output_path: Path,
) -> dict[str, object]:
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    candidate = torch.load(
        candidate_path, map_location="cpu", weights_only=False
    )
    foundation = json.loads(foundation_report_path.read_text(encoding="utf-8"))

    architecture = "ma_latent_graph_dreamer_v1"
    if base.get("control_architecture") != architecture:
        raise ValueError("base checkpoint is not an MA-Latent-Graph-Dreamer")
    if candidate.get("control_architecture") != architecture:
        raise ValueError("candidate checkpoint has an incompatible architecture")
    if candidate.get("actor_objective_version") != GRAPH_ACTOR_OBJECTIVE_VERSION:
        raise ValueError("candidate does not use the current actor objective")
    if not foundation.get("passed", False):
        raise ValueError("foundation report did not pass all gates")

    online_config = GraphRLConfig(**base["graph_rl_config"])
    trainer = deepcopy(candidate["trainer"])
    trainer["optimizers"]["world"] = deepcopy(
        base["trainer"]["optimizers"]["world"]
    )
    for optimizer_name in ("actor", "critic"):
        optimizer = trainer["optimizers"][optimizer_name]
        optimizer["state"] = {}
        for group in optimizer["param_groups"]:
            group["lr"] = online_config.actor_learning_rate

    migrated = dict(base)
    migrated.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": candidate["models"],
            "trainer": trainer,
            "graph_model_config": candidate["graph_model_config"],
            "graph_rl_config": asdict(online_config),
            "foundation_passed": True,
            "actor_objective_version": GRAPH_ACTOR_OBJECTIVE_VERSION,
            "actor_refinement": candidate.get("actor_refinement"),
            "foundation_validation": foundation,
            "checkpoint_migration": {
                "base": str(base_path),
                "candidate": str(candidate_path),
                "foundation_report": str(foundation_report_path),
                "actor_optimizer_reset": True,
                "critic_optimizer_reset": True,
                "world_optimizer_preserved": True,
            },
        }
    )
    _atomic_torch_save(migrated, output_path)
    controller = load_graph_controller(output_path, device="cpu")
    summary = {
        "output": str(output_path),
        "architecture": controller.architecture_id,
        "total_steps": int(migrated["total_steps"]),
        "episodes": int(migrated["episodes"]),
        "curriculum_stage": int(migrated["curriculum_stage"]),
        "foundation_passed": bool(migrated["foundation_passed"]),
        "actor_objective_version": int(
            migrated["actor_objective_version"]
        ),
        "online_actor_learning_rate": online_config.actor_learning_rate,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("foundation_report", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            migrate(
                arguments.base,
                arguments.candidate,
                arguments.foundation_report,
                arguments.output,
            )
        )
    )


if __name__ == "__main__":
    main()

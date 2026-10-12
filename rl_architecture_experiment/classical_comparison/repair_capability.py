"""Repair a failed curriculum gate with DAgger on held-out training scenarios."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .algorithms import DDPG, PPO, SAC, create_algorithm
from .assisted_train import (
    AssistedConfig,
    _configure_algorithm,
    _set_optimizer_learning_rates,
    _load_teacher,
    dagger_refine,
    validate_capability,
)
from .evaluate import _protocol_from_checkpoint
from .train import _load_checkpoint, _save_checkpoint, _write_json


def repair(
    checkpoint_path: Path,
    teacher_path: Path,
    output_path: Path,
    stage: int,
    rounds: int,
    epochs: int,
    device: str,
    scenario_offset: int = 10_000,
    tail_steps: int = 0,
    tail_repeats: int = 0,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    name = str(payload["algorithm"])
    protocol = _protocol_from_checkpoint(payload)
    algorithm = create_algorithm(
        name,
        int(payload["state_dim"]),
        protocol,
        device,
        int(protocol.training_seed),
    )
    _configure_algorithm(algorithm)
    total_steps, episode = _load_checkpoint(
        checkpoint_path, algorithm, name, protocol
    )
    _set_optimizer_learning_rates(algorithm)
    teacher = _load_teacher(teacher_path)
    config = replace(
        AssistedConfig(),
        dagger_mixture=tuple(0.0 for _ in range(rounds)),
        dagger_epochs_per_round=epochs,
        dagger_tail_steps=tail_steps,
        dagger_tail_repeats=tail_repeats,
    )
    records = dagger_refine(
        algorithm,
        teacher,
        protocol,
        config,
        stage=stage,
        episode_start=scenario_offset,
    )
    dagger_episode_indices = set(range(scenario_offset, scenario_offset + rounds))
    validation_episode_indices = set(range(config.validation_episodes))
    validation_seed_training = bool(
        dagger_episode_indices.intersection(validation_episode_indices)
    )
    if isinstance(algorithm, (DDPG, PPO, SAC)):
        algorithm.anchor_actor.load_state_dict(algorithm.actor.state_dict())
    if isinstance(algorithm, PPO):
        algorithm.rollout = []
    if isinstance(algorithm, DDPG):
        algorithm.actor_target.load_state_dict(algorithm.actor.state_dict())
    validation = validate_capability(
        algorithm,
        protocol,
        stage=stage,
        episodes=config.validation_episodes,
    )
    _save_checkpoint(
        output_path,
        algorithm,
        name,
        protocol,
        total_steps,
        episode,
        include_replay=True,
    )
    teacher_output = output_path.with_name(
        f"{output_path.stem}_dagger_teacher.pt"
    )
    torch.save(teacher, teacher_output)
    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": name,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "candidate_checkpoint": str(output_path.resolve()),
        "supplemental_teacher": str(teacher_output.resolve()),
        "stage": stage,
        "total_steps": total_steps,
        "episode": episode,
        "reward_tuning": False,
        "validation_seed_training": validation_seed_training,
        "dagger_scenario_offset": scenario_offset,
        "dagger_tail_steps": tail_steps,
        "dagger_tail_repeats": tail_repeats,
        "dagger_records": records,
        "validation": validation,
    }
    report_path = output_path.with_suffix(".repair.json")
    _write_json(report_path, report)
    if bool(validation["passed"]):
        monitor_path = checkpoint_path.parent / "monitor.json"
        monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
        monitor["updated_at"] = datetime.now(timezone.utc).isoformat()
        monitor["validation"] = validation
        monitor["repair"] = {
            "method": (
                "development-gate DAgger"
                if validation_seed_training
                else "held-out-scenario DAgger"
            ),
            "rounds": rounds,
            "epochs_per_round": epochs,
            "scenario_offset": scenario_offset,
            "tail_steps": tail_steps,
            "tail_repeats": tail_repeats,
            "report": str(report_path.resolve()),
        }
        _write_json(monitor_path, monitor)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenario-offset", type=int, default=10_000)
    parser.add_argument("--tail-steps", type=int, default=0)
    parser.add_argument("--tail-repeats", type=int, default=0)
    arguments = parser.parse_args()
    result = repair(
        arguments.checkpoint,
        arguments.teacher,
        arguments.output,
        arguments.stage,
        arguments.rounds,
        arguments.epochs,
        arguments.device,
        arguments.scenario_offset,
        arguments.tail_steps,
        arguments.tail_repeats,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Run deterministic capability validation for one saved baseline checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .algorithms import create_algorithm
from .assisted_train import _configure_algorithm, _load_teacher, validate_capability
from .common import ComparisonProtocol
from .train import _load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("ddpg", "ppo", "sac"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    protocol = replace(
        ComparisonProtocol(),
        replay_capacity=50_000,
        checkpoint_interval=50_000,
    )
    teacher = _load_teacher(arguments.teacher)
    state_dim = int(teacher["metadata"]["state_dim"])
    algorithm = create_algorithm(
        arguments.algorithm,
        state_dim,
        protocol,
        arguments.device,
        protocol.training_seed,
    )
    _configure_algorithm(algorithm)
    total_steps, episode = _load_checkpoint(
        arguments.checkpoint,
        algorithm,
        arguments.algorithm,
        protocol,
    )
    result = validate_capability(
        algorithm,
        protocol,
        stage=arguments.stage,
        episodes=arguments.episodes,
        episode_start=arguments.episode_start,
    )
    payload = {
        "algorithm": arguments.algorithm,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_total_steps": total_steps,
        "checkpoint_episode": episode,
        "validation": result,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()

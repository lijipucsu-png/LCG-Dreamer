"""Measure graph actor objective movement on one fixed replay batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from platoon_control.graph_agents import GraphPlatoonController
from platoon_control.graph_rl import GraphDreamerTrainer, GraphEpisodeReplay, GraphRLConfig
from platoon_control.models.graph_dreamer import GraphDreamerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--actor-learning-rate", type=float)
    parser.add_argument("--imagination-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    config = GraphRLConfig(**checkpoint["graph_rl_config"])
    config.device = arguments.device
    controller = GraphPlatoonController(
        GraphDreamerConfig(**checkpoint["graph_model_config"]),
        arguments.device,
        variant=config.variant,
    )
    controller.load_state_dict(checkpoint["models"])
    trainer = GraphDreamerTrainer(controller, config)
    trainer.load_state_dict(checkpoint["trainer"])
    if arguments.reset_optimizer:
        trainer.optimizers["actor"].state.clear()
        trainer.optimizers["critic"].state.clear()
    if arguments.actor_learning_rate is not None:
        config.actor_learning_rate = arguments.actor_learning_rate
        for name in ("actor", "critic"):
            for group in trainer.optimizers[name].param_groups:
                group["lr"] = arguments.actor_learning_rate
    replay = GraphEpisodeReplay(config.replay_capacity, config.seed)
    replay.load_state_dict(
        torch.load(arguments.replay, map_location="cpu", weights_only=False)
    )
    batch = replay.sample(config.batch_size, config.sequence_length)

    def evaluate() -> dict[str, float]:
        model = controller.model
        model.train()
        device_batch = trainer._variant_batch(batch)
        torch.manual_seed(arguments.seed)
        with torch.no_grad():
            states, _, _ = model.observe_sequence(device_batch, 0.0)
        observations = device_batch["observations"]
        losses = model.imagination_loss(
            states[-1],
            observations["adjacency"][:, -1],
            observations["edge_ages"][:, -1],
            15,
            0.0,
        )
        return {name: float(value.detach()) for name, value in losses.items()}

    records: list[dict[str, object]] = []
    before = evaluate()
    for step in range(1, arguments.steps + 1):
        torch.manual_seed(arguments.seed)
        training = trainer.train_actor_critic(
            batch, 15, 0.0, arguments.imagination_weight
        )
        after = evaluate()
        records.append(
            {
                "step": step,
                "before": before,
                "after": after,
                "training": training,
                "actor_learning_rate": config.actor_learning_rate,
                "imagination_weight": arguments.imagination_weight,
            }
        )
        before = after
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()

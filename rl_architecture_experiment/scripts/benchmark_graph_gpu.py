"""Run the production graph-training shapes before committing a long GPU job."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from platoon_control.graph_agents import GraphObservationAdapter, GraphPlatoonController
from platoon_control.graph_rl import GraphDreamerTrainer, GraphRLConfig
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv


def _sample_observations() -> dict[str, dict[str, np.ndarray]]:
    env = PlatoonParallelEnv()
    try:
        return {
            agent: env.observation_space(agent).sample()
            for agent in PLATOON_IDS
        }
    finally:
        env.close()


def _batch(batch_size: int, sequence_length: int) -> dict[str, object]:
    compact = GraphObservationAdapter()(_sample_observations())
    observations = {}
    for key, value in compact.items():
        dtype = torch.int64 if key == "role" else torch.float32
        tensor = torch.as_tensor(value, dtype=dtype)
        observations[key] = tensor.view(1, 1, *tensor.shape).expand(
            batch_size, sequence_length + 1, *tensor.shape
        ).clone()
    action_shape = (batch_size, sequence_length, len(PLATOON_IDS), 1)
    node_shape = (batch_size, sequence_length, len(PLATOON_IDS))
    return {
        "observations": observations,
        "applied_actions": torch.zeros(action_shape),
        "requested_actions": torch.zeros(action_shape),
        "raw_actions": torch.zeros(action_shape),
        "rewards": torch.zeros(node_shape),
        "continues": torch.ones(batch_size, sequence_length),
        "shielded": torch.zeros(node_shape),
        "behavior_valid": torch.ones(node_shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high"), default="highest"
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-memory-mib", type=float, default=10_500.0)
    parser.add_argument(
        "--output", default="artifacts/study_logs/graph_gpu_benchmark.json"
    )
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.unlink(missing_ok=True)
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.set_float32_matmul_precision(arguments.matmul_precision)
    torch.manual_seed(1701)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1701)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    config = GraphRLConfig(
        batch_size=arguments.batch_size,
        sequence_length=arguments.sequence_length,
        device=arguments.device,
        variant="full",
    )
    controller = GraphPlatoonController(device=arguments.device)
    trainer = GraphDreamerTrainer(controller, config)
    batch = _batch(arguments.batch_size, arguments.sequence_length)
    world_times: list[float] = []
    actor_times: list[float] = []
    for _ in range(max(1, arguments.iterations)):
        started = time.perf_counter()
        world_metrics = trainer.train_world(batch, communication_dropout=0.15)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        world_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        actor_metrics = trainer.train_actor_critic(
            batch, arguments.horizon, communication_dropout=0.15
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        actor_times.append(time.perf_counter() - started)
    world_seconds = statistics.median(world_times)
    actor_seconds = statistics.median(actor_times)
    if torch.cuda.is_available():
        peak_memory_mib = torch.cuda.max_memory_allocated() / (1024.0**2)
    else:
        peak_memory_mib = 0.0
    metrics = {**world_metrics, **actor_metrics}
    finite = all(math.isfinite(value) for value in metrics.values())
    passed = finite and (
        not arguments.device.startswith("cuda")
        or peak_memory_mib <= arguments.max_memory_mib
    )
    report = {
        "device": str(controller.device),
        "batch_size": arguments.batch_size,
        "sequence_length": arguments.sequence_length,
        "imagination_horizon": arguments.horizon,
        "matmul_precision": arguments.matmul_precision,
        "iterations": max(1, arguments.iterations),
        "parameters": sum(parameter.numel() for parameter in controller.model.parameters()),
        "world_update_seconds": world_seconds,
        "actor_critic_update_seconds": actor_seconds,
        "world_update_times": world_times,
        "actor_critic_update_times": actor_times,
        "peak_memory_mib": peak_memory_mib,
        "maximum_memory_mib": arguments.max_memory_mib,
        "all_metrics_finite": finite,
        "passed": passed,
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

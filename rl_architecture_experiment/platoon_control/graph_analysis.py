"""Auxiliary-model and paired statistical reports for graph experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .graph_rl import GraphEpisodeReplay, load_graph_controller


@torch.no_grad()
def evaluate_graph_auxiliary(
    checkpoint_path: str | Path,
    replay_path: str | Path,
    output_path: str | Path,
    device: str = "auto",
    batches: int = 20,
    batch_size: int = 16,
    sequence_length: int = 50,
) -> Path:
    controller = load_graph_controller(checkpoint_path, device)
    replay_state = torch.load(replay_path, map_location="cpu", weights_only=False)
    replay = GraphEpisodeReplay(int(replay_state["capacity_steps"]), seed=71)
    replay.load_state_dict(replay_state)
    if not replay.can_sample(sequence_length, batch_size):
        raise RuntimeError("replay does not contain enough held-out graph sequences")
    metrics: dict[str, list[float]] = {
        "world_loss": [],
        "physical_normalized_rmse": [],
        "delay_teacher_rmse": [],
        "delay_teacher_normalized_rmse": [],
        "delay_position_rmse_m": [],
        "delay_speed_rmse_mps": [],
        "delay_acceleration_rmse_mps2": [],
        "cutin_teacher_brier": [],
        "front_teacher_mae": [],
    }
    model = controller.model
    model.eval()
    for _ in range(batches):
        batch = replay.sample(batch_size, sequence_length)
        device_batch = {
            key: (
                {name: value.to(controller.device) for name, value in item.items()}
                if isinstance(item, dict)
                else item.to(controller.device)
            )
            for key, item in batch.items()
        }
        losses, state, context = model.world_model_loss(device_batch, 0.0)
        observations = device_batch["observations"]
        latest = {key: value[:, -1] for key, value in observations.items()}
        features = model._features(state, context)
        physical = model.physical_head(features)
        physical_target = latest["physical_target"] / model.physical_scale
        normalized_delay = model.delay_head(features).view(
            batch_size, 6, model.config.prediction_horizon, 3
        )
        delay = normalized_delay * model.delay_scale
        follower = latest["follower_mask"].bool()
        delay_target = latest["delay_target"]
        delay_error = delay[follower] - delay_target[follower]
        normalized_delay_error = (
            normalized_delay[follower]
            - delay_target[follower] / model.delay_scale
        )
        cutin = model.cutin_head(features).squeeze(-1).sigmoid()
        front = model.front_head(features)
        metrics["world_loss"].append(float(losses["loss"]))
        metrics["physical_normalized_rmse"].append(
            float(torch.sqrt((physical - physical_target).square().mean()))
        )
        metrics["delay_teacher_rmse"].append(
            float(torch.sqrt(delay_error.square().mean()))
        )
        metrics["delay_teacher_normalized_rmse"].append(
            float(torch.sqrt(normalized_delay_error.square().mean()))
        )
        metrics["delay_position_rmse_m"].append(
            float(torch.sqrt(delay_error[..., 0].square().mean()))
        )
        metrics["delay_speed_rmse_mps"].append(
            float(torch.sqrt(delay_error[..., 1].square().mean()))
        )
        metrics["delay_acceleration_rmse_mps2"].append(
            float(torch.sqrt(delay_error[..., 2].square().mean()))
        )
        metrics["cutin_teacher_brier"].append(
            float((cutin - latest["cutin_target"]).square().mean())
        )
        metrics["front_teacher_mae"].append(
            float((front - latest["front_target"]).abs().mean())
        )
    summary = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for name, values in metrics.items()
    }
    gates = {
        "physical": summary["physical_normalized_rmse"]["mean"] <= 0.10,
        "cpidlm_teacher_loaded": controller.adapter.teachers is not None,
        "cpidlm_distillation": (
            summary["delay_teacher_normalized_rmse"]["mean"] <= 0.10
            and summary["delay_speed_rmse_mps"]["mean"] <= 5.0
            and summary["delay_acceleration_rmse_mps2"]["mean"] <= 1.0
        ),
        "cutin_distillation": summary["cutin_teacher_brier"]["mean"] <= 0.05,
        "front_distillation": summary["front_teacher_mae"]["mean"] <= 0.10,
    }
    report = {
        "checkpoint": str(checkpoint_path),
        "replay": str(replay_path),
        "batches": batches,
        "summary": summary,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def _read_variant(path: str | Path, variant: str) -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            if raw["variant"] != variant:
                continue
            key = (int(raw["seed"]), int(raw["front_vehicle_count"]))
            rows[key] = {
                name: float(value)
                for name, value in raw.items()
                if name not in ("variant", "seed") and value not in (None, "")
            }
    return rows


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(-1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _paired_permutation_p(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(10_000, len(values)))
    permuted = np.abs((signs * values).mean(-1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))


def compare_paired_evaluations(
    baseline_csv: str | Path,
    graph_csv: str | Path,
    output_path: str | Path,
    baseline_variant: str = "full",
    graph_variant: str = "full",
    seed: int = 42,
) -> Path:
    baseline = _read_variant(baseline_csv, baseline_variant)
    graph = _read_variant(graph_csv, graph_variant)
    keys = sorted(set(baseline) & set(graph))
    if not keys:
        raise ValueError("evaluation files do not contain paired scenario rows")
    metrics = (
        "success",
        "collision",
        "failure",
        "travel_time_s",
        "minimum_gap_m",
        "fuel_mg",
        "string_amplification",
        "shield_intervention_rate",
        "inference_p95_ms",
    )
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    p_values: list[tuple[str, float]] = []
    for metric in metrics:
        differences = np.asarray(
            [graph[key][metric] - baseline[key][metric] for key in keys],
            np.float64,
        )
        lower, upper = _bootstrap_interval(differences, rng)
        p_value = _paired_permutation_p(differences, rng)
        results[metric] = {
            "baseline_mean": float(np.mean([baseline[key][metric] for key in keys])),
            "graph_mean": float(np.mean([graph[key][metric] for key in keys])),
            "paired_difference": float(differences.mean()),
            "ci95": [lower, upper],
            "p_value": p_value,
        }
        p_values.append((metric, p_value))
    ordered = sorted(p_values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (metric, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[metric] = running
    for metric, value in adjusted.items():
        results[metric]["holm_p_value"] = value
    baseline_shield = results["shield_intervention_rate"]["baseline_mean"]
    graph_shield = results["shield_intervention_rate"]["graph_mean"]
    shield_reduction = (
        (baseline_shield - graph_shield) / baseline_shield
        if baseline_shield > 1e-8
        else 0.0
    )
    improved_efficiency = any(
        results[metric]["paired_difference"] < 0.0
        and results[metric]["holm_p_value"] < 0.05
        for metric in ("fuel_mg", "string_amplification")
    )
    gates = {
        "success_noninferior": results["success"]["ci95"][0] >= -0.02,
        "collision_noninferior": results["collision"]["ci95"][1] <= 0.0,
        "failure_noninferior": results["failure"]["ci95"][1] <= 0.0,
        "shield_reduction_30_percent": shield_reduction >= 0.30,
        "efficiency_or_string_improved": improved_efficiency,
        "inference_p95_under_50ms": results["inference_p95_ms"]["graph_mean"] < 50.0,
    }
    report = {
        "baseline_csv": str(baseline_csv),
        "graph_csv": str(graph_csv),
        "paired_rows": len(keys),
        "results": results,
        "shield_relative_reduction": shield_reduction,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output

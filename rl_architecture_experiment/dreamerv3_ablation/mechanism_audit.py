"""Held-out representation tests that explain each ablation increment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from platoon_control.graph_rl import GraphEpisodeReplay, load_graph_controller
from platoon_control.models.graph_dreamer import NODE_COUNT

from .protocol import PROTOCOL, ROOT, VARIANTS


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            {subkey: tensor.clone() for subkey, tensor in value.items()}
            if isinstance(value, Mapping)
            else value.clone()
        )
        for key, value in batch.items()
    }


def prediction_metrics(model, batch: Mapping[str, Any], active_nodes: int = NODE_COUNT):
    states, _, contexts = model.observe_sequence(batch, 0.0)
    observations = batch["observations"]
    delay_errors: list[Tensor] = []
    physical_errors: list[Tensor] = []
    brier: list[Tensor] = []
    for step in range(batch["applied_actions"].shape[1]):
        features = model._features(states[step + 1], contexts[step + 1])
        active = slice(0, active_nodes)
        delay = model.delay_head(features).view(
            features.shape[0], NODE_COUNT, model.config.prediction_horizon, 3
        ) * model.delay_scale
        delay_target = observations["delay_target"][:, step + 1]
        delay_errors.append(
            ((delay[:, active] - delay_target[:, active]) / model.delay_scale)
            .square()
            .mean()
        )
        physical = model.physical_head(features) * model.physical_scale
        physical_target = observations["physical_target"][:, step + 1]
        physical_errors.append(
            ((physical[:, active] - physical_target[:, active]) / model.physical_scale)
            .square()
            .mean()
        )
        probability = model.cutin_head(features).squeeze(-1).sigmoid()
        target = observations["cutin_target"][:, step + 1]
        brier.append((probability[:, active] - target[:, active]).square().mean())
    return {
        "delay_scaled_rmse": float(torch.stack(delay_errors).mean().sqrt()),
        "physical_scaled_rmse": float(torch.stack(physical_errors).mean().sqrt()),
        "cutin_brier": float(torch.stack(brier).mean()),
    }


def permute_observation(observation: Mapping[str, Tensor], permutation: Tensor):
    result: dict[str, Tensor] = {}
    for key, value in observation.items():
        if key in {"adjacency", "edge_ages"}:
            result[key] = value[:, permutation][:, :, permutation]
        elif value.ndim >= 2 and value.shape[1] == NODE_COUNT:
            result[key] = value[:, permutation]
        else:
            result[key] = value
    return result


@torch.no_grad()
def equivariance_error(model, batch: Mapping[str, Any]) -> float:
    observations = batch["observations"]
    observation = {key: value[:, 0] for key, value in observations.items()}
    permutation = torch.tensor([0, 3, 1, 5, 2, 4], device=batch["applied_actions"].device)
    permuted = permute_observation(observation, permutation)
    state = model.initial(observation["role"].shape[0], permutation.device)
    action = torch.zeros(
        observation["role"].shape[0], NODE_COUNT, 1, device=permutation.device
    )
    first, _, first_context = model.observe_step(
        state,
        action,
        observation,
        observation["adjacency"],
        observation["edge_ages"],
    )
    second, _, second_context = model.observe_step(
        state,
        action[:, permutation],
        permuted,
        permuted["adjacency"],
        permuted["edge_ages"],
    )
    first_feature = torch.cat((first.feature, first_context), -1)[:, permutation]
    second_feature = torch.cat((second.feature, second_context), -1)
    scale = first_feature.square().mean().sqrt().clamp_min(1e-6)
    return float((first_feature - second_feature).square().mean().sqrt() / scale)


def apply_edge_dropout(batch: dict[str, Any], rate: float, seed: int) -> None:
    adjacency = batch["observations"]["adjacency"]
    generator = torch.Generator(device=adjacency.device).manual_seed(seed)
    drop = torch.rand(adjacency[..., 0].shape, generator=generator, device=adjacency.device) < rate
    adjacency[..., 0] = adjacency[..., 0] * (~drop).to(adjacency.dtype)


def apply_team_size(batch: dict[str, Any], size: int) -> None:
    observations = batch["observations"]
    for key, value in observations.items():
        if key in {"adjacency", "edge_ages"}:
            value[:, :, size:] = 0.0
            value[:, :, :, size:] = 0.0
        elif value.ndim >= 3 and value.shape[2] == NODE_COUNT:
            value[:, :, size:] = 0.0
    for key in ("applied_actions", "requested_actions", "raw_actions"):
        batch[key][:, :, size:] = 0.0


def apply_packet_age(batch: dict[str, Any], age: float) -> None:
    ages = batch["observations"]["edge_ages"]
    adjacency = batch["observations"]["adjacency"]
    ages[..., 1] = adjacency[..., 1] * age


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def run(output_root: Path, device_name: str) -> Path:
    replay_state = torch.load(
        ROOT / PROTOCOL.replay_path, map_location="cpu", weights_only=False
    )
    replay = GraphEpisodeReplay(PROTOCOL.replay_transitions, 58_003)
    replay.load_state_dict(replay_state)
    replay.rng = np.random.default_rng(58_003)
    heldout = replay.sample(8, PROTOCOL.sequence_length)
    rows: list[dict[str, Any]] = []
    for seed in PROTOCOL.training_seeds:
        for variant in VARIANTS:
            checkpoint = (
                output_root / variant.key / f"seed_{seed}" / "budgeted_complete.pt"
            )
            controller = load_graph_controller(checkpoint, device_name)
            model = controller.model.eval()
            batch = to_device(clone_batch(heldout), controller.device)
            base = prediction_metrics(model, batch)
            for metric, value in base.items():
                rows.append(
                    {
                        "variant": variant.key,
                        "training_seed": seed,
                        "test": "heldout_prediction",
                        "condition": "nominal",
                        "metric": metric,
                        "value": value,
                    }
                )
            rows.append(
                {
                    "variant": variant.key,
                    "training_seed": seed,
                    "test": "permutation",
                    "condition": "followers_permuted",
                    "metric": "equivariance_nrmse",
                    "value": equivariance_error(model, batch),
                }
            )
            for rate in (0.0, 0.2, 0.4, 0.6):
                changed = to_device(clone_batch(heldout), controller.device)
                apply_edge_dropout(changed, rate, 71_000 + seed)
                value = prediction_metrics(model, changed)["physical_scaled_rmse"]
                rows.append(
                    {
                        "variant": variant.key,
                        "training_seed": seed,
                        "test": "physical_edge_dropout",
                        "condition": f"{rate:.1f}",
                        "metric": "physical_scaled_rmse",
                        "value": value,
                    }
                )
            for size in (3, 4, 5, 6):
                changed = to_device(clone_batch(heldout), controller.device)
                apply_team_size(changed, size)
                value = prediction_metrics(model, changed, size)[
                    "physical_scaled_rmse"
                ]
                rows.append(
                    {
                        "variant": variant.key,
                        "training_seed": seed,
                        "test": "team_size",
                        "condition": str(size),
                        "metric": "physical_scaled_rmse",
                        "value": value,
                    }
                )
            for age in (0.0, 0.1, 0.2):
                changed = to_device(clone_batch(heldout), controller.device)
                apply_packet_age(changed, age)
                value = prediction_metrics(model, changed)["delay_scaled_rmse"]
                rows.append(
                    {
                        "variant": variant.key,
                        "training_seed": seed,
                        "test": "packet_age",
                        "condition": f"{age:.1f}",
                        "metric": "delay_scaled_rmse",
                        "value": value,
                    }
                )
    path = output_root / "mechanism_audit.csv"
    write_rows(path, rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v1",
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    print(run(arguments.output_root, arguments.device))


if __name__ == "__main__":
    main()

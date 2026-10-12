"""Leakage-safe supervised training for the three predictive control modules."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from .data import MatDelayDataset
from .models import (
    CPIDLMDelayCompensator,
    ConditionalWaveCVAE,
    DreamerV3Leader,
    STGNNTrajectoryPredictor,
)
from .ngsim import NGSIMTrajectoryDataset, NGSIMWaveDataset


@dataclass
class SupervisedConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 12
    hidden_dim: int = 128
    device: str = "auto"
    limit: int | None = None
    output_dir: str = "artifacts/checkpoints"


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name if name != "auto" else "cpu")


def _loader(dataset: Dataset, config: SupervisedConfig, shuffle: bool) -> DataLoader:
    if config.limit is not None and len(dataset) > config.limit:
        dataset = Subset(dataset, range(config.limit))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _fit(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    loss_function: Callable[[nn.Module, Mapping[str, Tensor], bool], dict[str, Tensor]],
    config: SupervisedConfig,
    model_arguments: dict,
) -> dict[str, object]:
    device = _device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_epoch = -1
    remaining_patience = config.patience
    history: list[dict[str, float]] = []
    checkpoint_path = Path(config.output_dir) / f"{name}_best.pt"

    def run(loader: DataLoader, training: bool) -> dict[str, float]:
        model.train(training)
        totals: dict[str, float] = {}
        samples = 0
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                losses = loss_function(model, batch, not training)
                if training:
                    losses["loss"].backward()
                    clip_grad_norm_(model.parameters(), 20.0)
                    optimizer.step()
            batch_size = int(next(iter(batch.values())).shape[0])
            samples += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
        return {key: value / max(samples, 1) for key, value in totals.items()}

    for epoch in range(config.epochs):
        train_metrics = run(train_loader, True)
        validation_metrics = run(validation_loader, False)
        record = {
            "epoch": float(epoch),
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"validation/{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(json.dumps({"model": name, **record}), flush=True)
        selection_value = validation_metrics.get("selection", validation_metrics["loss"])
        if selection_value < best_loss - 1e-6:
            best_loss = selection_value
            best_epoch = epoch
            remaining_patience = config.patience
            _atomic_save(
                {
                    "format_version": 1,
                    "model_name": name,
                    "model_arguments": model_arguments,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run(test_loader, False)
    return {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "test": test_metrics,
        "history": history,
    }


def _delay_loss(model: nn.Module, batch: Mapping[str, Tensor], evaluation: bool) -> dict[str, Tensor]:
    assert isinstance(model, CPIDLMDelayCompensator)
    prediction = model(batch["v2v_history"], batch["local_predecessor"])
    losses = model.loss(prediction, batch["current_target"], batch["future_target"])
    current_rmse = (prediction.current_state - batch["current_target"]).square().mean().sqrt()
    future_rmse = (prediction.future_mean - batch["future_target"]).square().mean().sqrt()
    return {**losses, "current_rmse": current_rmse, "future_rmse": future_rmse}


def _trajectory_loss(model: nn.Module, batch: Mapping[str, Tensor], evaluation: bool) -> dict[str, Tensor]:
    assert isinstance(model, STGNNTrajectoryPredictor)
    prediction = model(batch["history"], batch["mask"])
    losses = model.loss(
        prediction, batch["future"], batch["mask"], batch["cutin"], batch["lane_change"]
    )
    selected = prediction.mode_logits.argmax(-1)
    chosen = prediction.trajectories.gather(
        2,
        selected[..., None, None, None].expand(
            -1, -1, 1, prediction.trajectories.shape[-2], prediction.trajectories.shape[-1]
        ),
    ).squeeze(2)
    displacement = (chosen[..., :2] - batch["future"][..., :2]).square().sum(-1).sqrt()
    mask = batch["mask"]
    ade = (displacement.mean(-1) * mask).sum() / mask.sum().clamp_min(1.0)
    fde = (displacement[..., -1] * mask).sum() / mask.sum().clamp_min(1.0)
    lane_accuracy = (
        (prediction.lane_change_logits.argmax(-1) == batch["lane_change"]).float() * mask
    ).sum() / mask.sum().clamp_min(1.0)
    lane_prediction = prediction.lane_change_logits.argmax(-1)
    class_f1 = []
    for label in range(3):
        true_positive = (((lane_prediction == label) & (batch["lane_change"] == label)).float() * mask).sum()
        false_positive = (((lane_prediction == label) & (batch["lane_change"] != label)).float() * mask).sum()
        false_negative = (((lane_prediction != label) & (batch["lane_change"] == label)).float() * mask).sum()
        class_f1.append(2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative).clamp_min(1.0))
    lane_macro_f1 = torch.stack(class_f1).mean()
    return {
        **losses,
        "ade_m": ade,
        "fde_m": fde,
        "lane_accuracy": lane_accuracy,
        "lane_macro_f1": lane_macro_f1,
        "selection": ade,
    }


def _front_loss(model: nn.Module, batch: Mapping[str, Tensor], evaluation: bool) -> dict[str, Tensor]:
    assert isinstance(model, DreamerV3Leader)
    history = batch["front_history"]
    ego = torch.zeros(history.shape[0], 8, device=history.device)
    ego[:, 1] = history[:, 0, -1, 2]
    losses = model.front_prediction_loss(
        history, ego, batch["front_mask"], batch["front_future"]
    )
    prediction = model.front_predictor(history, ego)
    selected = prediction.mode_logits.argmax(-1)
    chosen = prediction.trajectories.gather(
        2,
        selected[..., None, None, None].expand(
            -1, -1, 1, prediction.trajectories.shape[-2], 2
        ),
    ).squeeze(2)
    displacement = (chosen - batch["front_future"]).square().sum(-1).sqrt()
    mask = batch["front_mask"]
    ade = (displacement.mean(-1) * mask).sum() / mask.sum().clamp_min(1.0)
    fde = (displacement[..., -1] * mask).sum() / mask.sum().clamp_min(1.0)
    return {**losses, "ade_m": ade, "fde_m": fde, "selection": ade}


def _wave_loss(model: nn.Module, batch: Mapping[str, Tensor], evaluation: bool) -> dict[str, Tensor]:
    assert isinstance(model, ConditionalWaveCVAE)
    prediction = model(batch["condition"], batch["target_acceleration"])
    losses = model.loss(prediction, batch["target_acceleration"])
    prior = prediction.prior_profile
    rmse = (prior - batch["target_acceleration"]).square().mean().sqrt()
    speed_delta = (prior.sum(-1) * 0.1).mean()
    return {
        **losses,
        "prior_rmse": rmse,
        "mean_speed_delta": speed_delta,
        "selection": rmse,
    }


def train_supervised(
    config: SupervisedConfig,
    models: tuple[str, ...] = ("cpidlm", "trajectory", "leader_front", "wave"),
    mat_path: str | Path = "data_30.mat",
    ngsim_dir: str | Path = "artifacts/ngsim",
) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    results: dict[str, object] = {"config": asdict(config), "models": {}}

    if "cpidlm" in models:
        arguments = {"hidden_dim": config.hidden_dim, "horizon": 30}
        model = CPIDLMDelayCompensator(**arguments)
        result = _fit(
            "cpidlm", model,
            _loader(MatDelayDataset(mat_path, split="train"), config, True),
            _loader(MatDelayDataset(mat_path, split="validation"), config, False),
            _loader(MatDelayDataset(mat_path, split="test"), config, False),
            _delay_loss, config, arguments,
        )
        results["models"]["cpidlm"] = result

    trajectory_datasets: dict[str, NGSIMTrajectoryDataset] = {}
    if "trajectory" in models or "leader_front" in models:
        trajectory_datasets = {
            split: NGSIMTrajectoryDataset(ngsim_dir, split=split)
            for split in ("train", "validation", "test")
        }
    if "trajectory" in models:
        arguments = {"hidden_dim": config.hidden_dim, "horizon": 30, "modes": 3}
        model = STGNNTrajectoryPredictor(**arguments)
        result = _fit(
            "trajectory", model,
            _loader(trajectory_datasets["train"], config, True),
            _loader(trajectory_datasets["validation"], config, False),
            _loader(trajectory_datasets["test"], config, False),
            _trajectory_loss, config, arguments,
        )
        results["models"]["trajectory"] = result

    if "leader_front" in models:
        arguments = {
            "hidden_dim": config.hidden_dim,
            "stochastic_dim": 16,
            "stochastic_classes": 16,
            "trajectory_modes": 3,
            "prediction_horizon": 30,
        }
        model = DreamerV3Leader(**arguments)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.front_predictor.parameters():
            parameter.requires_grad_(True)
        result = _fit(
            "leader_front", model,
            _loader(trajectory_datasets["train"], config, True),
            _loader(trajectory_datasets["validation"], config, False),
            _loader(trajectory_datasets["test"], config, False),
            _front_loss, config, arguments,
        )
        results["models"]["leader_front"] = result

    if "wave" in models:
        arguments = {"hidden_dim": config.hidden_dim, "latent_dim": 12, "horizon": 60}
        model = ConditionalWaveCVAE(**arguments)
        result = _fit(
            "wave", model,
            _loader(NGSIMWaveDataset(ngsim_dir, split="train"), config, True),
            _loader(NGSIMWaveDataset(ngsim_dir, split="validation"), config, False),
            _loader(NGSIMWaveDataset(ngsim_dir, split="test"), config, False),
            _wave_loss, config, arguments,
        )
        results["models"]["wave"] = result

    report_path = Path(config.output_dir) / "supervised_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous.setdefault("models", {}).update(results["models"])
        previous.setdefault("runs", []).append(asdict(config))
        results = previous
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return report_path


def evaluate_supervised(
    checkpoint_dir: str | Path = "artifacts/checkpoints",
    mat_path: str | Path = "data_30.mat",
    ngsim_dir: str | Path = "artifacts/ngsim",
    device_name: str = "auto",
    batch_size: int = 512,
) -> Path:
    """Evaluate frozen best checkpoints and compare them with physical baselines."""

    device = _device(device_name)
    directory = Path(checkpoint_dir)
    specifications = {
        "cpidlm": (
            CPIDLMDelayCompensator,
            MatDelayDataset(mat_path, split="test"),
            _delay_loss,
        ),
        "trajectory": (
            STGNNTrajectoryPredictor,
            NGSIMTrajectoryDataset(ngsim_dir, split="test"),
            _trajectory_loss,
        ),
        "leader_front": (
            DreamerV3Leader,
            NGSIMTrajectoryDataset(ngsim_dir, split="test"),
            _front_loss,
        ),
        "wave": (
            ConditionalWaveCVAE,
            NGSIMWaveDataset(ngsim_dir, split="test"),
            _wave_loss,
        ),
    }
    output: dict[str, Any] = {"models": {}, "passed": True}
    for name, (model_class, dataset, loss_function) in specifications.items():
        checkpoint = torch.load(directory / f"{name}_best.pt", map_location=device, weights_only=False)
        model = model_class(**checkpoint["model_arguments"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        totals: dict[str, float] = {}
        baseline_squares = 0.0
        baseline_count = 0
        baseline_fde_sum = 0.0
        mask_count = 0.0
        model_squares = 0.0
        model_future_squares = 0.0
        target_speed_delta: list[float] = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for raw_batch in loader:
                batch = _to_device(raw_batch, device)
                metrics = loss_function(model, batch, True)
                count = int(next(iter(batch.values())).shape[0])
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * count
                if name == "cpidlm":
                    prediction = model(
                        batch["v2v_history"], batch["local_predecessor"]
                    )
                    delayed = batch["v2v_history"][:, -1, 4:8]
                    age = delayed[:, 3]
                    current = delayed[:, :3].clone()
                    current[:, 0] += delayed[:, 1] * age + 0.5 * delayed[:, 2] * age.square()
                    current[:, 1] += delayed[:, 2] * age
                    seconds = torch.arange(1, 31, device=device)[None] * 0.1
                    future = current[:, None].expand(-1, 30, -1).clone()
                    future[:, :, 0] = current[:, 0, None] + current[:, 1, None] * seconds + 0.5 * current[:, 2, None] * seconds.square()
                    future[:, :, 1] = current[:, 1, None] + current[:, 2, None] * seconds
                    baseline_squares += float((current - batch["current_target"]).square().sum())
                    model_squares += float(
                        (prediction.current_state - batch["current_target"]).square().sum()
                    )
                    baseline_count += int(current.numel())
                    baseline_fde_sum += float((future - batch["future_target"]).square().sum())
                    model_future_squares += float(
                        (prediction.future_mean - batch["future_target"]).square().sum()
                    )
                    mask_count += float(future.numel())
                elif name in ("trajectory", "leader_front"):
                    prefix = "front_" if name == "leader_front" else ""
                    history = batch[f"{prefix}history"]
                    future = batch[f"{prefix}future"]
                    mask = batch[f"{prefix}mask"]
                    seconds = torch.arange(1, 31, device=device)[None, None] * 0.1
                    base = future.new_zeros(*future.shape[:-1], 2)
                    base[..., 0] = history[:, :, -1, 0, None] + history[:, :, -1, 2, None] * seconds
                    base[..., 1] = history[:, :, -1, 1, None]
                    displacement = (base - future[..., :2]).square().sum(-1).sqrt()
                    baseline_squares += float((displacement.mean(-1) * mask).sum())
                    baseline_fde_sum += float((displacement[..., -1] * mask).sum())
                    mask_count += float(mask.sum())
                else:
                    target = batch["target_acceleration"]
                    baseline_squares += float(target.square().sum())
                    baseline_count += int(target.numel())
                    target_speed_delta.extend((target.sum(-1) * 0.1).cpu().tolist())
        metrics = {key: value / len(dataset) for key, value in totals.items()}
        if name == "cpidlm":
            metrics["current_rmse"] = float(np.sqrt(model_squares / baseline_count))
            metrics["future_rmse"] = float(np.sqrt(model_future_squares / mask_count))
            baseline = {
                "current_rmse": float(np.sqrt(baseline_squares / baseline_count)),
                "future_rmse": float(np.sqrt(baseline_fde_sum / mask_count)),
            }
            passed = metrics["current_rmse"] <= baseline["current_rmse"] and metrics["future_rmse"] < baseline["future_rmse"]
        elif name in ("trajectory", "leader_front"):
            baseline = {
                "ade_m": baseline_squares / max(mask_count, 1.0),
                "fde_m": baseline_fde_sum / max(mask_count, 1.0),
            }
            passed = metrics["ade_m"] < baseline["ade_m"] and metrics["fde_m"] < baseline["fde_m"]
        else:
            baseline = {
                "zero_acceleration_rmse": float(np.sqrt(baseline_squares / baseline_count)),
                "target_mean_speed_delta": float(np.mean(target_speed_delta)),
            }
            passed = metrics["prior_rmse"] < baseline["zero_acceleration_rmse"] and metrics["mean_speed_delta"] < -1.0
        output["models"][name] = {"metrics": metrics, "baseline": baseline, "passed": bool(passed)}
        output["passed"] = bool(output["passed"] and passed)
    path = directory / "predictor_test_report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return path

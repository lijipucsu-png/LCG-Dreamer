"""Adapters for the supplied CPIDLM data, shockwave events, and SUMO replay."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset


FEET_TO_METERS = 0.3048


class MatDelayDataset(Dataset[dict[str, Tensor]]):
    """Adapt `data_30.mat` to delayed V2V plus current-radar supervision."""

    def __init__(
        self,
        path: str | Path = "data_30.mat",
        limit: int | None = None,
        split: str | None = None,
        guard_samples: int = 50,
    ) -> None:
        source = loadmat(path, variable_names=("train_data", "lable_data", "label_data"))
        history = source["train_data"]
        labels = source.get("lable_data", source.get("label_data"))
        if labels is None:
            raise KeyError("MAT file must contain lable_data or label_data")
        count = len(history)
        boundaries = {
            "train": (0, int(0.60 * count) - guard_samples),
            "validation": (int(0.60 * count) + guard_samples, int(0.80 * count) - guard_samples),
            "test": (int(0.80 * count) + guard_samples, count),
        }
        start, stop = boundaries.get(split, (0, count))
        if limit is not None:
            stop = min(stop, start + limit)
        self.indices = np.arange(start, stop, dtype=np.int64)
        self.history = np.asarray(history, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        source_index = int(self.indices[index])
        raw = self.history[source_index] * FEET_TO_METERS
        future = self.labels[source_index] * FEET_TO_METERS
        delay_steps = 1 + source_index % 2
        delayed = np.concatenate(
            (np.repeat(raw[:1], delay_steps, axis=0), raw[:-delay_steps]), axis=0
        )
        ages = np.full((len(raw),), delay_steps * 0.1, dtype=np.float32)
        v2v = np.stack(
            (
                delayed[:, 7], delayed[:, 5], delayed[:, 6], ages,
                delayed[:, 7], delayed[:, 5], delayed[:, 6], ages,
            ),
            axis=-1,
        ).astype(np.float32)
        relative_position = raw[-1, 7] - raw[-1, 4]
        local = np.asarray(
            [relative_position, raw[-1, 5] - raw[-1, 0], raw[-1, 6], 0.0, raw[-1, 1], 1.0],
            dtype=np.float32,
        )
        current = raw[-1, [7, 5, 6]].astype(np.float32)
        future_speed = future[:, 4]
        future_acceleration = np.gradient(future_speed, 0.1).astype(np.float32)
        future_target = np.stack((future[:, 5], future_speed, future_acceleration), axis=-1)
        return {
            "v2v_history": torch.from_numpy(v2v),
            "local_predecessor": torch.from_numpy(local),
            "current_target": torch.from_numpy(current),
            "future_target": torch.from_numpy(future_target.astype(np.float32)),
        }


class ShockwaveTrajectoryDataset(Dataset[dict[str, Tensor]]):
    """Convert real longitudinal shockwave events to masked graph trajectories.

    The supplied data has no lane coordinate or cut-in annotation.  Those channels
    are deliberately masked/zero here and are trained from SUMO merge replays.
    """

    def __init__(
        self,
        path: str | Path = "ying/processed_data-0.8/dataset_all_events.pkl",
        history_steps: int = 20,
        horizon: int = 30,
        max_neighbors: int = 12,
        limit: int | None = None,
    ) -> None:
        with Path(path).open("rb") as stream:
            events = pickle.load(stream)
        self.events = events if limit is None else events[:limit]
        self.history_steps = history_steps
        self.horizon = horizon
        self.max_neighbors = max_neighbors

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        trajectories = self.events[index]["trajs"][: self.max_neighbors]
        usable = [item for item in trajectories if len(item) >= self.history_steps + self.horizon]
        if not usable:
            return self.__getitem__((index + 1) % len(self))
        output_history = np.zeros((self.max_neighbors, self.history_steps, 8), np.float32)
        output_future = np.zeros((self.max_neighbors, self.horizon, 4), np.float32)
        mask = np.zeros(self.max_neighbors, np.float32)
        cutin = np.zeros(self.max_neighbors, np.float32)
        common_length = min(len(item) for item in usable)
        start = max(0, (common_length - self.history_steps - self.horizon) // 2)
        anchor = float(usable[-1].iloc[start + self.history_steps - 1]["pos"])
        for slot, frame in enumerate(usable):
            history = frame.iloc[start : start + self.history_steps]
            future = frame.iloc[
                start + self.history_steps : start + self.history_steps + self.horizon
            ]
            output_history[slot, :, 0] = history["pos"].to_numpy(np.float32) - anchor
            output_history[slot, :, 2] = history["vel"].to_numpy(np.float32)
            output_history[slot, :, 3] = history["acc"].to_numpy(np.float32)
            output_history[slot, :, 4] = history["length"].to_numpy(np.float32)
            output_history[slot, :, 7] = 1.0
            output_future[slot, :, 0] = future["pos"].to_numpy(np.float32) - anchor
            output_future[slot, :, 2] = future["vel"].to_numpy(np.float32)
            output_future[slot, :, 3] = future["acc"].to_numpy(np.float32)
            mask[slot] = 1.0
        return {
            "history": torch.from_numpy(output_history),
            "mask": torch.from_numpy(mask),
            "future": torch.from_numpy(output_future),
            "cutin": torch.from_numpy(cutin),
        }


@dataclass
class ReplayTransition:
    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_observation: dict[str, np.ndarray]
    terminal: bool


class ReplayWriter:
    """Write complete episodes with metadata using PyTorch's portable archive."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, episode: list[dict[str, Any]], seed: int, name: str) -> Path:
        path = self.directory / f"{name}_seed{seed}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"seed": seed, "steps": episode, "format_version": 1}, temporary)
        temporary.replace(path)
        return path


class SumoCutinReplayDataset(Dataset[dict[str, Tensor]]):
    """Build ST-GNN samples from SUMO ramp observations saved by ``collect-sumo``."""

    def __init__(self, paths: list[str | Path]) -> None:
        self.samples: list[tuple[dict[str, np.ndarray], str]] = []
        for path in paths:
            replay = torch.load(path, map_location="cpu", weights_only=False)
            for step in replay["steps"]:
                cutin_state = step["infos"].get("leader", {}).get("cutin_state", "idle")
                for agent, observation in step["observations"].items():
                    if agent != "leader":
                        self.samples.append((observation, cutin_state))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        observation, cutin_state = self.samples[index]
        history = np.asarray(observation["neighbor_history"], dtype=np.float32)
        mask = np.asarray(observation["neighbor_mask"], dtype=np.float32)
        latest = history[:, -1]
        horizon = 30
        time = np.arange(1, horizon + 1, dtype=np.float32)[None] * 0.1
        relative_speed = latest[:, 2:3] - float(observation["ego"][1])
        future = np.zeros((history.shape[0], horizon, 4), dtype=np.float32)
        future[:, :, 0] = latest[:, 0:1] + relative_speed * time
        ramp_vehicle = history[:, :, 6].max(axis=1) > 0.5
        active_merge = cutin_state in ("armed", "aligning")
        cutin = (ramp_vehicle & active_merge).astype(np.float32) * mask
        lateral_target = np.where(cutin > 0.0, 0.0, latest[:, 1])[:, None]
        blend = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[None]
        future[:, :, 1] = latest[:, 1:2] * (1.0 - blend) + lateral_target * blend
        future[:, :, 2] = latest[:, 2:3]
        future[:, :, 3] = latest[:, 3:4]
        return {
            "history": torch.from_numpy(history),
            "mask": torch.from_numpy(mask),
            "future": torch.from_numpy(future),
            "cutin": torch.from_numpy(cutin),
        }


def stack_observation_sequence(
    sequence: list[dict[str, np.ndarray]], device: torch.device
) -> dict[str, Tensor]:
    return {
        key: torch.as_tensor(
            np.stack([observation[key] for observation in sequence]),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        for key in sequence[0]
    }

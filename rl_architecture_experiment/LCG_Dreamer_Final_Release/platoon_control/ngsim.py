"""NGSIM preprocessing and leakage-safe trajectory/wave datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from torch import Tensor
from torch.utils.data import Dataset


FEET_TO_METERS = 0.3048
STEP_SECONDS = 0.1
REQUIRED_COLUMNS = (
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "v_Length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TimeSplit:
    start_ms: int
    end_ms: int


def _split_boundaries(start_ms: int, end_ms: int) -> dict[str, TimeSplit]:
    duration = end_ms - start_ms
    train_end = int(start_ms + 0.60 * duration)
    validation_start = train_end + 5_000
    validation_end = int(start_ms + 0.80 * duration)
    test_start = validation_end + 5_000
    return {
        "train": TimeSplit(start_ms, train_end),
        "validation": TimeSplit(validation_start, validation_end),
        "test": TimeSplit(test_start, end_ms),
    }


def _smooth_track(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = frame["Local_X"].to_numpy(np.float64) * FEET_TO_METERS
    y = frame["Local_Y"].to_numpy(np.float64) * FEET_TO_METERS
    length = len(frame)
    window = min(11, length if length % 2 else length - 1)
    if window >= 5:
        x = savgol_filter(x, window, 3 if window >= 7 else 2, mode="interp")
        y = savgol_filter(y, window, 3 if window >= 7 else 2, mode="interp")
    speed = np.gradient(y, STEP_SECONDS)
    acceleration = np.gradient(speed, STEP_SECONDS)
    return (
        x.astype(np.float32),
        y.astype(np.float32),
        np.clip(speed, 0.0, 45.0).astype(np.float32),
        np.clip(acceleration, -10.0, 8.0).astype(np.float32),
    )


def prepare_ngsim(
    source: str | Path = "NGSIM/trajectories-0500-0515.csv",
    output_dir: str | Path = "artifacts/ngsim",
) -> Path:
    """Validate, smooth, convert, and cache the supplied NGSIM CSV."""

    source = Path(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, usecols=list(REQUIRED_COLUMNS))
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"NGSIM source is missing columns: {sorted(missing)}")
    frame = frame.sort_values(["Vehicle_ID", "Global_Time"], kind="stable").reset_index(drop=True)
    duplicate = frame.duplicated(("Vehicle_ID", "Global_Time")).sum()
    if duplicate:
        raise ValueError(f"NGSIM source contains {duplicate} duplicate vehicle timestamps")
    time_steps = frame.groupby("Vehicle_ID", sort=False)["Global_Time"].diff().dropna()
    non_contiguous = int((time_steps != 100).sum())
    if non_contiguous:
        raise ValueError(f"NGSIM source contains {non_contiguous} non-100 ms track steps")

    x = np.empty(len(frame), np.float32)
    y = np.empty(len(frame), np.float32)
    speed = np.empty(len(frame), np.float32)
    acceleration = np.empty(len(frame), np.float32)
    for indices in frame.groupby("Vehicle_ID", sort=False).indices.values():
        track = frame.iloc[indices]
        track_x, track_y, track_speed, track_acceleration = _smooth_track(track)
        x[indices], y[indices] = track_x, track_y
        speed[indices], acceleration[indices] = track_speed, track_acceleration

    cache_path = output / "clean_tracks.npz"
    np.savez_compressed(
        cache_path,
        vehicle_id=frame["Vehicle_ID"].to_numpy(np.int32),
        frame_id=frame["Frame_ID"].to_numpy(np.int32),
        time_ms=frame["Global_Time"].to_numpy(np.int64),
        lateral=x,
        longitudinal=y,
        speed=speed,
        acceleration=acceleration,
        length=(frame["v_Length"].to_numpy(np.float32) * FEET_TO_METERS),
        width=(frame["v_Width"].to_numpy(np.float32) * FEET_TO_METERS),
        vehicle_class=frame["v_Class"].to_numpy(np.int8),
        lane=frame["Lane_ID"].to_numpy(np.int8),
        preceding=frame["Preceding"].to_numpy(np.int32),
        following=frame["Following"].to_numpy(np.int32),
        space_headway=(frame["Space_Headway"].to_numpy(np.float32) * FEET_TO_METERS),
        time_headway=frame["Time_Headway"].to_numpy(np.float32),
    )
    start_ms = int(frame["Global_Time"].min())
    end_ms = int(frame["Global_Time"].max())
    splits = _split_boundaries(start_ms, end_ms)
    previous_lane = frame.groupby("Vehicle_ID", sort=False)["Lane_ID"].shift()
    lane_changes = int((previous_lane.notna() & (previous_lane != frame["Lane_ID"])).sum())
    unique_class_counts = (
        frame.groupby("v_Class")["Vehicle_ID"].nunique().astype(int).to_dict()
    )
    vehicle_total = sum(unique_class_counts.values())
    manifest = {
        "format_version": 1,
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "rows": len(frame),
        "vehicles": int(frame["Vehicle_ID"].nunique()),
        "duration_seconds": (end_ms - start_ms) / 1000.0,
        "sample_period_seconds": STEP_SECONDS,
        "lane_changes": lane_changes,
        "vehicle_class_counts": {str(key): value for key, value in unique_class_counts.items()},
        "vehicle_class_probabilities": {
            str(key): value / vehicle_total for key, value in unique_class_counts.items()
        },
        "splits": {
            key: {"start_ms": value.start_ms, "end_ms": value.end_ms}
            for key, value in splits.items()
        },
        "units": "SI",
        "smoothing": {"method": "savgol", "window_steps": 11, "polynomial": 3},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class NGSIMStore:
    def __init__(self, directory: str | Path = "artifacts/ngsim") -> None:
        directory = Path(directory)
        with (directory / "manifest.json").open("r", encoding="utf-8") as stream:
            self.manifest: dict[str, Any] = json.load(stream)
        archive = np.load(directory / "clean_tracks.npz")
        self.arrays = {key: archive[key] for key in archive.files}
        vehicle_ids = self.arrays["vehicle_id"]
        unique_vehicles, starts, counts = np.unique(
            vehicle_ids, return_index=True, return_counts=True
        )
        self.track_slices = {
            int(vehicle): slice(int(start), int(start + count))
            for vehicle, start, count in zip(unique_vehicles, starts, counts)
        }
        time_order = np.argsort(self.arrays["time_ms"], kind="stable")
        ordered_times = self.arrays["time_ms"][time_order]
        unique_times, time_starts, time_counts = np.unique(
            ordered_times, return_index=True, return_counts=True
        )
        self.time_rows = {
            int(timestamp): time_order[int(start) : int(start + count)]
            for timestamp, start, count in zip(unique_times, time_starts, time_counts)
        }

    def split(self, name: str) -> TimeSplit:
        values = self.manifest["splits"][name]
        return TimeSplit(int(values["start_ms"]), int(values["end_ms"]))

    def track(self, vehicle_id: int) -> dict[str, np.ndarray]:
        selection = self.track_slices[vehicle_id]
        return {key: value[selection] for key, value in self.arrays.items()}


@lru_cache(maxsize=4)
def load_ngsim_store(directory: str = "artifacts/ngsim") -> NGSIMStore:
    return NGSIMStore(directory)


class NGSIMTrajectoryDataset(Dataset[dict[str, Tensor]]):
    """Agent-centric NGSIM scenes with multi-modal trajectory and lane labels."""

    def __init__(
        self,
        directory: str | Path = "artifacts/ngsim",
        split: str = "train",
        history_steps: int = 20,
        horizon: int = 30,
        max_neighbors: int = 12,
        stride_steps: int = 10,
    ) -> None:
        self.store = load_ngsim_store(str(directory))
        self.history_steps = history_steps
        self.horizon = horizon
        self.max_neighbors = max_neighbors
        split_range = self.store.split(split)
        anchors: list[tuple[int, int]] = []
        for vehicle_id, selection in self.store.track_slices.items():
            times = self.store.arrays["time_ms"][selection]
            for offset in range(history_steps - 1, len(times) - horizon, stride_steps):
                timestamp = int(times[offset])
                if (
                    timestamp - (history_steps - 1) * 100 >= split_range.start_ms
                    and timestamp + horizon * 100 <= split_range.end_ms
                ):
                    anchors.append((vehicle_id, timestamp))
        self.anchors = anchors

    def __len__(self) -> int:
        return len(self.anchors)

    def _window(self, vehicle_id: int, timestamp: int) -> tuple[dict[str, np.ndarray], int] | None:
        track = self.store.track(vehicle_id)
        offset = int(np.searchsorted(track["time_ms"], timestamp))
        if offset >= len(track["time_ms"]) or int(track["time_ms"][offset]) != timestamp:
            return None
        if offset < self.history_steps - 1 or offset + self.horizon >= len(track["time_ms"]):
            return None
        expected_start = timestamp - (self.history_steps - 1) * 100
        expected_end = timestamp + self.horizon * 100
        if int(track["time_ms"][offset - self.history_steps + 1]) != expected_start:
            return None
        if int(track["time_ms"][offset + self.horizon]) != expected_end:
            return None
        return track, offset

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        ego_id, timestamp = self.anchors[index]
        ego_window = self._window(ego_id, timestamp)
        if ego_window is None:
            raise RuntimeError("Invalid ego window in prevalidated NGSIM anchor")
        ego, ego_offset = ego_window
        ego_x = float(ego["lateral"][ego_offset])
        ego_y = float(ego["longitudinal"][ego_offset])
        candidates = self.store.time_rows[timestamp]
        relative_x = self.store.arrays["lateral"][candidates] - ego_x
        relative_y = self.store.arrays["longitudinal"][candidates] - ego_y
        distance = np.hypot(relative_y, 3.0 * relative_x)
        candidate_order = np.argsort(distance)

        history = np.zeros((self.max_neighbors, self.history_steps, 8), np.float32)
        future = np.zeros((self.max_neighbors, self.horizon, 4), np.float32)
        mask = np.zeros(self.max_neighbors, np.float32)
        lane_change = np.zeros(self.max_neighbors, np.int64)
        cutin = np.zeros(self.max_neighbors, np.float32)
        selected = 0
        for candidate_index in candidate_order:
            vehicle_id = int(self.store.arrays["vehicle_id"][candidates[candidate_index]])
            if vehicle_id == ego_id:
                continue
            neighbor_window = self._window(vehicle_id, timestamp)
            if neighbor_window is None:
                continue
            neighbor, offset = neighbor_window
            history_slice = slice(offset - self.history_steps + 1, offset + 1)
            future_slice = slice(offset + 1, offset + self.horizon + 1)
            current_lane = int(neighbor["lane"][offset])
            final_lane = int(neighbor["lane"][offset + self.horizon])
            lane_delta = final_lane - current_lane
            label = 1 if lane_delta < 0 else 2 if lane_delta > 0 else 0
            ego_lane = int(ego["lane"][ego_offset])

            history[selected, :, 0] = neighbor["longitudinal"][history_slice] - ego_y
            history[selected, :, 1] = neighbor["lateral"][history_slice] - ego_x
            history[selected, :, 2] = neighbor["speed"][history_slice]
            history[selected, :, 3] = neighbor["acceleration"][history_slice]
            history[selected, :, 4] = neighbor["length"][history_slice]
            history[selected, :, 5] = neighbor["lane"][history_slice] - ego_lane
            history[selected, :, 6] = neighbor["vehicle_class"][history_slice] / 3.0
            history[selected, :, 7] = 1.0
            future[selected, :, 0] = neighbor["longitudinal"][future_slice] - ego_y
            future[selected, :, 1] = neighbor["lateral"][future_slice] - ego_x
            future[selected, :, 2] = neighbor["speed"][future_slice]
            future[selected, :, 3] = neighbor["acceleration"][future_slice]
            mask[selected] = 1.0
            lane_change[selected] = label
            cutin[selected] = float(current_lane != ego_lane and final_lane == ego_lane)
            selected += 1
            if selected == self.max_neighbors:
                break

        latest_y = history[:, -1, 0]
        front_indices = np.flatnonzero((mask > 0.0) & (latest_y > 0.0))
        front_indices = front_indices[np.argsort(latest_y[front_indices])][:8]
        front_history = np.zeros((8, self.history_steps, 6), np.float32)
        front_future = np.zeros((8, self.horizon, 2), np.float32)
        front_mask = np.zeros(8, np.float32)
        for slot, neighbor_index in enumerate(front_indices):
            front_history[slot] = history[neighbor_index, :, :6]
            front_future[slot] = future[neighbor_index, :, :2]
            front_mask[slot] = 1.0
        return {
            "history": torch.from_numpy(history),
            "mask": torch.from_numpy(mask),
            "future": torch.from_numpy(future),
            "lane_change": torch.from_numpy(lane_change),
            "cutin": torch.from_numpy(cutin),
            "front_history": torch.from_numpy(front_history),
            "front_mask": torch.from_numpy(front_mask),
            "front_future": torch.from_numpy(front_future),
        }


class NGSIMWaveDataset(Dataset[dict[str, Tensor]]):
    """Conditional braking-wave snippets extracted from smoothed NGSIM tracks."""

    def __init__(
        self,
        directory: str | Path = "artifacts/ngsim",
        split: str = "train",
        history_steps: int = 20,
        horizon: int = 60,
    ) -> None:
        self.store = load_ngsim_store(str(directory))
        self.history_steps = history_steps
        self.horizon = horizon
        split_range = self.store.split(split)
        events: list[tuple[int, int]] = []
        for vehicle_id in self.store.track_slices:
            track = self.store.track(vehicle_id)
            speed = track["speed"]
            acceleration = track["acceleration"]
            last_event = -horizon
            for offset in range(history_steps - 1, len(speed) - horizon):
                timestamp = int(track["time_ms"][offset])
                if timestamp - (history_steps - 1) * 100 < split_range.start_ms:
                    continue
                if timestamp + horizon * 100 > split_range.end_ms:
                    continue
                future_3s = speed[offset : offset + 31]
                if (
                    speed[offset] - float(future_3s.min()) >= 3.0
                    and float(acceleration[offset : offset + 31].min()) <= -1.0
                    and offset - last_event >= horizon
                ):
                    events.append((vehicle_id, timestamp))
                    last_event = offset
        self.events = events

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        vehicle_id, timestamp = self.events[index]
        track = self.store.track(vehicle_id)
        offset = int(np.searchsorted(track["time_ms"], timestamp))
        history_slice = slice(offset - self.history_steps + 1, offset + 1)
        target_slice = slice(offset + 1, offset + self.horizon + 1)
        condition = np.zeros((self.history_steps, 8), np.float32)
        condition[:, 0] = track["speed"][history_slice]
        condition[:, 1] = track["acceleration"][history_slice]
        condition[:, 6] = track["vehicle_class"][history_slice] / 3.0

        for step, row_offset in enumerate(range(offset - self.history_steps + 1, offset + 1)):
            time_ms = int(track["time_ms"][row_offset])
            preceding_id = int(track["preceding"][row_offset])
            if preceding_id in self.store.track_slices:
                leader = self.store.track(preceding_id)
                leader_offset = int(np.searchsorted(leader["time_ms"], time_ms))
                if leader_offset < len(leader["time_ms"]) and int(leader["time_ms"][leader_offset]) == time_ms:
                    condition[step, 2] = max(
                        0.0,
                        float(leader["longitudinal"][leader_offset] - track["longitudinal"][row_offset] - leader["length"][leader_offset]),
                    )
                    condition[step, 3] = float(leader["speed"][leader_offset] - track["speed"][row_offset])
        local_rows = self.store.time_rows[timestamp]
        center = float(track["longitudinal"][offset])
        local_distance = np.abs(self.store.arrays["longitudinal"][local_rows] - center)
        local_speeds = self.store.arrays["speed"][local_rows][local_distance <= 100.0]
        condition[:, 4] = float(local_speeds.mean()) if len(local_speeds) else 0.0
        condition[:, 5] = float(local_speeds.std()) if len(local_speeds) else 0.0
        condition[:, 7] = min(len(local_speeds) / 60.0, 1.0)
        target = track["acceleration"][target_slice].astype(np.float32)
        return {
            "condition": torch.from_numpy(condition),
            "target_acceleration": torch.from_numpy(target),
        }


def batch_metrics(values: Iterator[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else 0.0,
        "std": float(array.std()) if len(array) else 0.0,
        "p05": float(np.quantile(array, 0.05)) if len(array) else 0.0,
        "p50": float(np.quantile(array, 0.50)) if len(array) else 0.0,
        "p95": float(np.quantile(array, 0.95)) if len(array) else 0.0,
    }

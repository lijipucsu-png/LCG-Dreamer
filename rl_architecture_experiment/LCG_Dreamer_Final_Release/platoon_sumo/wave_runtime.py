"""Runtime bridge from the trained NGSIM wave model to TraCI."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


class LearnedWavePolicy:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        import torch

        from platoon_control.models.wave import ConditionalWaveCVAE

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Learned wave checkpoint not found: {path}. Run train-supervised --models wave."
            )
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        arguments = checkpoint.get("model_arguments", {})
        self.model = ConditionalWaveCVAE(**arguments).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.device = torch.device(device)

    def condition(
        self,
        connection: Any,
        target: str,
        history: deque[np.ndarray],
        station_lookup: dict[str, float],
    ) -> np.ndarray:
        rows = list(history)[-20:]
        if not rows:
            raise RuntimeError(f"No history is available for wave source {target}")
        if len(rows) < 20:
            rows = [rows[0]] * (20 - len(rows)) + rows
        condition = np.zeros((20, 8), dtype=np.float32)
        condition[:, 0] = [row[2] for row in rows]
        condition[:, 1] = [row[3] for row in rows]
        leader = connection.vehicle.getLeader(target, 150.0)
        if leader:
            condition[:, 2] = float(leader[1])
            condition[:, 3] = float(
                connection.vehicle.getSpeed(leader[0]) - connection.vehicle.getSpeed(target)
            )
        active = connection.vehicle.getIDList()
        center = station_lookup.get(target, 0.0)
        local_speeds = [
            connection.vehicle.getSpeed(vehicle_id)
            for vehicle_id in active
            if abs(station_lookup.get(vehicle_id, -10_000.0) - center) <= 100.0
        ]
        if local_speeds:
            condition[:, 4] = float(np.mean(local_speeds))
            condition[:, 5] = float(np.std(local_speeds))
        vehicle_class = connection.vehicle.getVehicleClass(target)
        condition[:, 6] = {"motorcycle": 1.0 / 3.0, "passenger": 2.0 / 3.0, "truck": 1.0}.get(
            vehicle_class, 2.0 / 3.0
        )
        condition[:, 7] = min(len(local_speeds) / 60.0, 1.0)
        return condition

    def sample(self, condition: np.ndarray, seed: int) -> np.ndarray:
        import torch

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        tensor = torch.as_tensor(condition, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            profile = self.model.sample(tensor, generator=generator)[0]
        return profile.cpu().numpy().astype(np.float32)

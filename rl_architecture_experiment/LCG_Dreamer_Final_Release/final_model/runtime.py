"""Release-relative loading for the final longitudinal and lateral policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mixed_traffic_lane_change.policy import GraphMixedLanePolicy, graph_adapter
from platoon_control.graph_agents import GraphPlatoonController
from platoon_control.models.graph_dreamer import GraphDreamerConfig


@dataclass(frozen=True)
class ReleasePaths:
    """All runtime assets, resolved independently of the working directory."""

    root: Path
    longitudinal: Path
    lateral: Path
    predictors: Path
    calibration: Path
    wave_model: Path

    @classmethod
    def discover(cls, root: str | Path | None = None) -> "ReleasePaths":
        release_root = (
            Path(root).resolve()
            if root is not None
            else Path(__file__).resolve().parents[1]
        )
        return cls(
            root=release_root,
            longitudinal=release_root / "weights" / "graph_longitudinal_final.pt",
            lateral=release_root / "weights" / "graph_lane_change_final.pt",
            predictors=release_root / "weights" / "predictors",
            calibration=(
                release_root
                / "weights"
                / "calibration"
                / "calibrated_parameters.json"
            ),
            wave_model=release_root / "weights" / "predictors" / "wave_best.pt",
        )

    def required_files(self, include_lane_change: bool = True) -> tuple[Path, ...]:
        files = (
            self.longitudinal,
            self.predictors / "cpidlm_best.pt",
            self.predictors / "trajectory_best.pt",
            self.predictors / "leader_front_best.pt",
            self.wave_model,
            self.calibration,
            self.calibration.parent / "ngsim_vtypes.add.xml",
        )
        return files + ((self.lateral,) if include_lane_change else ())

    def validate(self, include_lane_change: bool = True) -> None:
        missing = [str(path) for path in self.required_files(include_lane_change) if not path.is_file()]
        if missing:
            raise FileNotFoundError("release assets are missing: " + ", ".join(missing))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _GraphBasePolicy:
    """Adapter from the graph controller to the lateral policy interface."""

    name = "graph"

    def __init__(self, controller: GraphPlatoonController) -> None:
        self.controller = controller

    def reset(self) -> None:
        self.controller.reset()

    def act(self, observations: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
        return self.controller.act(observations, deterministic=True)

    def record_applied(self, infos: Mapping[str, Mapping[str, Any]]) -> None:
        self.controller.record_applied_actions(infos)


class FinalPlatoonPolicy:
    """Unified deployment policy exposing one action dictionary for all six trucks."""

    def __init__(
        self,
        paths: ReleasePaths | None = None,
        device: str = "auto",
        enable_lane_change: bool = True,
    ) -> None:
        self.paths = paths or ReleasePaths.discover()
        self.paths.validate(enable_lane_change)
        checkpoint = torch.load(
            self.paths.longitudinal, map_location="cpu", weights_only=False
        )
        if checkpoint.get("control_architecture") != "ma_latent_graph_dreamer_v1":
            raise ValueError("the longitudinal checkpoint is not LCG-Dreamer")
        config = GraphDreamerConfig(**checkpoint["graph_model_config"])
        variant = checkpoint.get("graph_rl_config", {}).get("variant", "full")
        controller = GraphPlatoonController(
            config,
            device=device,
            predictor_directory=self.paths.predictors,
            variant=variant,
        )
        controller.load_state_dict(checkpoint["models"])
        controller.eval()
        self.controller = controller
        self.enable_lane_change = bool(enable_lane_change)
        base = _GraphBasePolicy(controller)
        if self.enable_lane_change:
            lateral = torch.load(
                self.paths.lateral, map_location="cpu", weights_only=False
            )
            if lateral.get("method") != "graph":
                raise ValueError("the lateral checkpoint is not the Graph policy")
            expected_base = lateral.get("base_checkpoint_sha256")
            actual_base = _sha256(self.paths.longitudinal)
            if expected_base and expected_base != actual_base:
                raise ValueError(
                    "longitudinal and lateral checkpoints were not selected together"
                )
            adapter = graph_adapter(str(controller.device))
            adapter.load_state_dict(lateral["lateral_state"])
            adapter.eval()
            self.policy: Any = GraphMixedLanePolicy(
                base,
                adapter,
                str(controller.device),
                deterministic=True,
            )
        else:
            self.policy = base

    @property
    def device(self) -> torch.device:
        return self.controller.device

    def reset(self) -> None:
        self.policy.reset()

    def act(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> dict[str, dict[str, Any]]:
        """Return acceleration and lane-change commands for all six trucks."""

        return self.policy.act(observations)

    def record_applied(
        self, infos: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Feed SUMO-applied accelerations back into the recurrent world model."""

        self.policy.record_applied(infos)


def load_final_policy(
    device: str = "auto",
    enable_lane_change: bool = True,
    root: str | Path | None = None,
) -> FinalPlatoonPolicy:
    """Load the final policy with release-relative assets."""

    return FinalPlatoonPolicy(
        ReleasePaths.discover(root),
        device=device,
        enable_lane_change=enable_lane_change,
    )

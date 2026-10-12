"""Frozen prediction backbone shared by every learned comparison controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from platoon_control.graph_agents import (
    GraphObservationAdapter,
    GraphTeacherBundle,
)
from platoon_control.graph_rl import load_graph_controller
from platoon_control.models.graph_dreamer import NODE_COUNT
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PlatoonParallelEnv


@dataclass(frozen=True)
class PredictionLayout:
    physical: int = NODE_COUNT * 7
    predecessor_trajectory: int = NODE_COUNT * 30 * 3
    cutin_probability: int = NODE_COUNT
    surrounding_summary: int = NODE_COUNT * 4
    selected_cutin_trajectory: int = NODE_COUNT * 30 * 4
    selected_cutin_probability: int = NODE_COUNT
    selected_lane_probability: int = NODE_COUNT * 3
    selected_mode_probability: int = NODE_COUNT
    selected_validity: int = NODE_COUNT

    @property
    def dimension(self) -> int:
        return sum(self.__dict__.values())

    def to_dict(self) -> dict[str, int]:
        return {**self.__dict__, "dimension": self.dimension}


PREDICTION_LAYOUT = PredictionLayout()


class SharedPredictiveEncoder:
    """Online-only Graph-RSSM and ST-GNN predictions with frozen weights.

    The encoder never reads simulator future state. The Graph-RSSM consumes the
    same current graph observation, previous requested action, and previous
    applied action as the proposed controller. The ST-GNN consumes only the
    observed neighbor-history tensor and exposes the most likely cut-in track.
    """

    def __init__(
        self,
        graph_checkpoint: str | Path,
        predictor_directory: str | Path,
        device: str = "cuda",
    ) -> None:
        self.controller = load_graph_controller(graph_checkpoint, device)
        # Teacher targets must not be injected into the Graph-RSSM observation.
        self.controller.adapter = GraphObservationAdapter(teachers=None)
        self.device = self.controller.device
        self.teachers = GraphTeacherBundle(predictor_directory, self.device)
        for parameter in self.controller.model.parameters():
            parameter.requires_grad_(False)
        self.layout = PREDICTION_LAYOUT
        self._identity: int | None = None
        self._vector: np.ndarray | None = None
        self._delay_prediction: Tensor | None = None

    def reset(self) -> None:
        self.controller.reset()
        self._identity = None
        self._vector = None
        self._delay_prediction = None

    @staticmethod
    def _stack(
        observations: Mapping[str, Mapping[str, np.ndarray]],
        key: str,
        device: torch.device,
    ) -> Tensor:
        return torch.stack(
            [
                torch.as_tensor(
                    observations[agent][key], dtype=torch.float32, device=device
                )
                for agent in PLATOON_IDS
            ]
        )

    @torch.no_grad()
    def __call__(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> np.ndarray:
        identity = id(observations)
        if self._identity == identity and self._vector is not None:
            return self._vector

        _, context = self.controller._prepare(observations)
        model = self.controller.model
        features = model._features(self.controller._state, context)
        physical = model.decoded_physical(self.controller._state, context)
        physical_normalized = physical / model.physical_scale
        delay = model.delay_head(features).view(
            1, NODE_COUNT, model.config.prediction_horizon, 3
        )
        cutin = model.cutin_head(features).sigmoid().squeeze(-1)
        surrounding = model.front_head(features)

        histories = self._stack(observations, "neighbor_history", self.device)
        masks = self._stack(observations, "neighbor_mask", self.device)
        prediction = self.teachers.trajectory(histories, masks)
        cutin_probability = prediction.cutin_logits.sigmoid() * masks
        selected_neighbor = cutin_probability.argmax(-1)
        batch_index = torch.arange(NODE_COUNT, device=self.device)
        selected_modes = prediction.mode_logits[batch_index, selected_neighbor].argmax(-1)
        selected_trajectory = prediction.trajectories[
            batch_index, selected_neighbor, selected_modes
        ]
        selected_cutin = cutin_probability[batch_index, selected_neighbor]
        selected_lane = prediction.lane_change_logits[
            batch_index, selected_neighbor
        ].softmax(-1)
        selected_mode = prediction.mode_logits[
            batch_index, selected_neighbor
        ].softmax(-1).amax(-1)
        selected_valid = masks[batch_index, selected_neighbor]
        selected_scale = selected_trajectory.new_tensor([150.0, 15.0, 35.0, 6.0])

        follower_v2v = torch.stack(
            [
                torch.as_tensor(
                    observations[agent]["v2v_history"],
                    dtype=torch.float32,
                    device=self.device,
                )
                for agent in FOLLOWER_IDS
            ]
        )
        follower_local = torch.stack(
            [
                torch.as_tensor(
                    observations[agent]["local_predecessor"],
                    dtype=torch.float32,
                    device=self.device,
                )
                for agent in FOLLOWER_IDS
            ]
        )
        self._delay_prediction = self.teachers.delay(
            follower_v2v, follower_local
        ).future_mean

        tensors = (
            physical_normalized,
            delay,
            cutin,
            surrounding,
            selected_trajectory / selected_scale,
            selected_cutin,
            selected_lane,
            selected_mode,
            selected_valid,
        )
        vector = torch.cat([item.reshape(-1) for item in tensors]).float()
        if vector.numel() != self.layout.dimension:
            raise RuntimeError(
                f"prediction dimension {vector.numel()} != {self.layout.dimension}"
            )
        if not torch.isfinite(vector).all():
            raise FloatingPointError("non-finite shared prediction feature")
        self._identity = identity
        self._vector = vector.cpu().numpy().astype(np.float32, copy=False)
        return self._vector

    def attach_reward_prediction(self, env: PlatoonParallelEnv) -> None:
        if self._delay_prediction is None:
            raise RuntimeError("prediction must be evaluated before reward attachment")
        future = self._delay_prediction
        predictions = {
            agent: (
                float(future[index, -1, 1].detach().cpu()),
                float(future[index, :, 2].mean().detach().cpu()),
            )
            for index, agent in enumerate(FOLLOWER_IDS)
        }
        env.set_follower_control_predictions(predictions)

    def record_applied(
        self,
        infos: Mapping[str, Mapping[str, Any]],
        requested_actions: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.controller.record_applied_actions(infos, requested_actions)
        self._identity = None
        self._vector = None
        self._delay_prediction = None

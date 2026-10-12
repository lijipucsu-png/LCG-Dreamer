"""Prediction-equated state adapters, including the NC knowledge module."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from classical_comparison.common import StateAdapter
from platoon_control.graph_agents import GraphObservationAdapter
from platoon_sumo.env import PLATOON_IDS, PLATOON_LENGTH

from .prediction import SharedPredictiveEncoder
from .protocol import PredictiveComparisonProtocol


class PredictiveStateAdapter:
    """Append a common frozen prediction vector to the legacy shared state."""

    def __init__(self, predictor: SharedPredictiveEncoder) -> None:
        self.base = StateAdapter()
        self.predictor = predictor
        self.dimension: int | None = None
        self.base_dimension: int | None = None
        self._shared_dimension: int | None = None

    def reset(self) -> None:
        self.predictor.reset()

    def __call__(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        previous_applied: np.ndarray,
    ) -> np.ndarray:
        base = self.base(observations, previous_applied)
        prediction = self.predictor(observations)
        state = np.concatenate((base, prediction)).astype(np.float32, copy=False)
        self.base_dimension = int(base.size)
        if self._shared_dimension is None:
            self._shared_dimension = int(state.size)
        elif self._shared_dimension != state.size:
            raise RuntimeError(
                "predictive state dimension changed from "
                f"{self._shared_dimension} to {state.size}"
            )
        self.dimension = int(state.size)
        if not np.isfinite(state).all():
            raise FloatingPointError("non-finite predictive state")
        return state

    def record(
        self,
        infos: Mapping[str, Mapping[str, Any]],
        actions: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.predictor.record_applied(infos, actions)


class NCKnowledgeStateAdapter(PredictiveStateAdapter):
    """Wang et al. knowledge-guided state adapted to the six-truck graph task.

    The source paper's CV-HVs-CV aggregate is mapped to each upstream segment
    between the platoon leader and a controlled follower. This preserves its
    kinematic-wave/Newell features without pretending that controlled trucks
    are unobservable human-driven vehicles in this environment.
    """

    KNOWLEDGE_FEATURES_PER_NODE = 8
    DELAY_MODES = 3

    def __init__(
        self,
        predictor: SharedPredictiveEncoder,
        protocol: PredictiveComparisonProtocol,
    ) -> None:
        super().__init__(predictor)
        self.protocol = protocol
        self.compact = GraphObservationAdapter(teachers=None)
        self.action_history: deque[np.ndarray] = deque(
            maxlen=protocol.nc_action_history_steps
        )
        self.delay_gate_dimension = len(PLATOON_IDS) * self.DELAY_MODES
        self.reset()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "action_history"):
            self.action_history.clear()
            for _ in range(self.protocol.nc_action_history_steps):
                self.action_history.append(np.zeros(len(PLATOON_IDS), np.float32))

    def _knowledge(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> np.ndarray:
        physical = self.compact(observations)["physical_target"]
        values = np.zeros(
            (len(PLATOON_IDS), self.KNOWLEDGE_FEATURES_PER_NODE), np.float32
        )
        wave_speed = self.protocol.nc_wave_speed_mps
        local_h = self.protocol.nc_local_time_headway_s
        local_l = self.protocol.nc_local_standstill_gap_m
        fusion = self.protocol.nc_fusion_weight
        for index, agent in enumerate(PLATOON_IDS[1:], start=1):
            ego_speed = float(physical[index, 0])
            local_gap = float(physical[index, 2])
            local_relative_speed = float(physical[index, 3])
            platoon = np.asarray(observations[agent]["platoon_state"], np.float32)
            leader_front_distance = max(0.0, float(platoon[0, 0]))
            aggregate_gap = max(
                0.0, leader_front_distance - index * float(PLATOON_LENGTH)
            )
            leader_speed = float(platoon[0, 1])
            predecessor_speed = max(0.0, ego_speed + local_relative_speed)
            aggregate_h = aggregate_gap / max(wave_speed + predecessor_speed, 1e-3)
            aggregate_l = wave_speed * aggregate_h
            local_desired = local_h * ego_speed + local_l
            total_desired = (
                (local_h + aggregate_h) * ego_speed + local_l + aggregate_l
            )
            local_error = local_gap - local_desired
            aggregate_error = aggregate_gap - total_desired
            aggregate_relative_speed = leader_speed - ego_speed
            fused_gap = fusion * local_error + (1.0 - fusion) * aggregate_error
            fused_speed = (
                fusion * local_relative_speed
                + (1.0 - fusion) * aggregate_relative_speed
            )
            values[index] = np.asarray(
                [
                    local_error / 50.0,
                    local_relative_speed / 30.0,
                    aggregate_error / 250.0,
                    aggregate_relative_speed / 30.0,
                    aggregate_h / 10.0,
                    aggregate_l / 50.0,
                    fused_gap / 100.0,
                    fused_speed / 30.0,
                ],
                np.float32,
            )
        return values.reshape(-1)

    @staticmethod
    def _delay_modes(
        observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> np.ndarray:
        gates = np.zeros((len(PLATOON_IDS), 3), np.float32)
        gates[0, 0] = 1.0
        for index, agent in enumerate(PLATOON_IDS[1:], start=1):
            communication = np.asarray(
                observations[agent]["communication"], np.float32
            )
            packet_age = max(float(communication[1]), float(communication[3]))
            mode = int(np.clip(np.rint(packet_age / 0.1), 0, 2))
            gates[index, mode] = 1.0
        return gates.reshape(-1)

    def __call__(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        previous_applied: np.ndarray,
    ) -> np.ndarray:
        shared = super().__call__(observations, previous_applied)
        history = np.stack(tuple(self.action_history)).reshape(-1) / 3.0
        delay_modes = self._delay_modes(observations)
        state = np.concatenate(
            (shared, self._knowledge(observations), history, delay_modes)
        ).astype(np.float32, copy=False)
        self.dimension = int(state.size)
        if not np.isfinite(state).all():
            raise FloatingPointError("non-finite NC knowledge state")
        return state

    def record(
        self,
        infos: Mapping[str, Mapping[str, Any]],
        actions: Mapping[str, Mapping[str, Any]],
    ) -> None:
        requested = np.asarray(
            [
                float(np.asarray(actions[agent]["accel"]).reshape(-1)[0])
                for agent in PLATOON_IDS
            ],
            np.float32,
        )
        self.action_history.append(requested)
        super().record(infos, actions)

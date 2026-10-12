"""Distinct Graph, native learned, and CACC+LC lateral controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from classical_comparison.evaluate import CaccPolicy, GraphPolicy
from cooperative_lane_change.model import (
    CooperativeLaneActorCritic,
    LanePolicySample,
    gap_raw_to_target,
    gap_target_to_raw,
)
from platoon_sumo.env import (
    LANE_CHANGE_MIN_FRONT_HEADWAY,
    LANE_CHANGE_MIN_REAR_HEADWAY,
    LANE_CHANGE_REAR_LOOKAHEAD,
    MIN_SAFE_GAP,
    PLATOON_IDS,
)
from predictive_comparison.evaluate import PredictivePolicy

from .algorithms import (
    ACTION_DIM,
    NODE_FEATURE_DIM,
    NativeActionSample,
    NativeLaneAlgorithm,
)
from .protocol import MixedLaneProtocol


@dataclass
class MixedLaneTrace:
    features: Tensor
    state: np.ndarray
    change_mask: Tensor
    gap_active: Tensor
    change: Tensor
    gap_raw: Tensor
    action_unit: np.ndarray
    log_prob: float
    value: float
    target_gaps: np.ndarray


def create_base_policy(
    method: str, protocol: MixedLaneProtocol, device: str
) -> Any:
    checkpoint = protocol.base_checkpoint(method)
    if method == "graph":
        assert checkpoint is not None
        return GraphPolicy(checkpoint, device)
    if method == "cacc_lc":
        return CaccPolicy()
    if method in ("ddpg_p", "ppo_p", "sac_p", "nc_kg_sac_p"):
        assert checkpoint is not None
        policy = PredictivePolicy(checkpoint, device)
        if policy.name != method:
            raise ValueError(
                f"base checkpoint is {policy.name}, expected {method}"
            )
        return policy
    raise ValueError(f"unknown mixed-lane method: {method}")


def _target_lane_context(
    observation: Mapping[str, np.ndarray], mission: np.ndarray
) -> np.ndarray:
    target_lane = int(round(float(mission[2] * 2.0)))
    ego_speed = float(observation["ego"][1])
    tracks = np.asarray(observation["neighbor_history"], np.float32)[:, -1]
    mask = np.asarray(observation["neighbor_mask"], np.float32) > 0.5
    valid = tracks[mask]
    valid = valid[(valid[:, 7] > 0.5) & (np.rint(valid[:, 5]).astype(int) == target_lane)]
    front = valid[valid[:, 0] >= 0.0]
    rear = valid[valid[:, 0] < 0.0]
    if len(front):
        front_row = front[int(np.argmin(front[:, 0]))]
        front_gap = max(0.0, float(front_row[0] - front_row[4]))
        front_closing = max(0.0, ego_speed - float(front_row[2]))
        front_ttc = front_gap / max(front_closing, 1e-3) if front_closing > 0.0 else 20.0
        front_relative_speed = float(front_row[2]) - ego_speed
    else:
        front_ttc = 20.0
        front_relative_speed = 0.0
    if len(rear):
        rear_row = rear[int(np.argmax(rear[:, 0]))]
        rear_gap = max(0.0, float(-rear_row[0] - 12.0))
        rear_closing = max(0.0, float(rear_row[2]) - ego_speed)
        rear_ttc = rear_gap / max(rear_closing, 1e-3) if rear_closing > 0.0 else 20.0
        rear_relative_speed = float(rear_row[2]) - ego_speed
    else:
        rear_ttc = 20.0
        rear_relative_speed = 0.0
    return np.asarray(
        [
            np.clip(front_relative_speed / 15.0, -2.0, 2.0),
            np.clip(rear_relative_speed / 15.0, -2.0, 2.0),
            np.clip(front_ttc / 10.0, 0.0, 2.0),
            np.clip(rear_ttc / 10.0, 0.0, 2.0),
            np.clip(len(valid) / 8.0, 0.0, 1.5),
            np.clip((mission[6] + mission[7]) * 0.5, 0.0, 1.0),
        ],
        np.float32,
    )


def common_lane_features(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    base_actions: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, agent in enumerate(PLATOON_IDS):
        observation = observations[agent]
        mission = np.asarray(observation["lane_mission"], np.float32)
        ego = np.asarray(observation["ego"], np.float32)
        predecessor = np.asarray(
            observation["local_predecessor"], np.float32
        ).copy()
        predecessor /= np.asarray(
            [100.0, 30.0, 3.0, 2.0, 50.0, 1.0], np.float32
        )
        communication = np.clip(
            np.asarray(observation["communication"], np.float32) / 0.2,
            0.0,
            2.0,
        )
        base_acceleration = float(
            np.asarray(base_actions[agent]["accel"], np.float32).reshape(-1)[0]
        )
        row = np.concatenate(
            (
                mission,
                np.asarray(
                    [ego[1] / 30.0, ego[2] / 3.0, ego[3] / 10.0, ego[4] / 2.0],
                    np.float32,
                ),
                predecessor,
                communication,
                np.asarray([base_acceleration / 3.0, index / 5.0], np.float32),
                _target_lane_context(observation, mission),
            )
        ).astype(np.float32, copy=False)
        rows.append(row)
    result = np.stack(rows)
    if result.shape != (len(PLATOON_IDS), NODE_FEATURE_DIM):
        raise RuntimeError(f"unexpected mixed lane feature shape: {result.shape}")
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite mixed lane feature")
    return result


def lane_action_masks(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    *,
    atomic_team_entry: bool = True,
) -> tuple[np.ndarray, bool]:
    missions = np.stack(
        [
            np.asarray(observations[agent]["lane_mission"], np.float32)
            for agent in PLATOON_IDS
        ]
    )
    event_active = bool(missions[0, 8] > 0.5)
    platoon = np.asarray(observations["leader"]["platoon_state"], np.float32)
    gaps = platoon[1:, 4]
    valid_gaps = gaps[gaps > 0.0]
    mean_gap = float(np.mean(valid_gaps)) if len(valid_gaps) else 12.0
    obstacle_distance = float(missions[0, 4])
    elapsed = float(missions[0, 9])
    shape_ready = mean_gap <= 11.2 or obstacle_distance <= 0.80 or elapsed >= 0.25
    direction_pending = np.abs(missions[:, 3]) > 0.5
    contexts = np.stack(
        [
            _target_lane_context(observations[agent], missions[index])
            for index, agent in enumerate(PLATOON_IDS)
        ]
    )
    clearance_safe = np.asarray(
        [
            _individual_lane_clearance_safe(
                observations, index, missions[index], contexts[index]
            )
            for index in range(len(PLATOON_IDS))
        ],
        dtype=bool,
    )
    # A unified learned controller must not strand the tail by inserting only
    # part of the formation into a marginal gap.  Before the first truck has
    # moved, the target gap is accepted atomically; after entry begins, any
    # remaining truck is released by the same per-vehicle safety criterion.
    if (
        atomic_team_entry
        and bool(np.all(direction_pending))
        and not bool(np.all(clearance_safe))
    ):
        clearance_safe.fill(False)
    mask = event_active & shape_ready & direction_pending & clearance_safe
    return mask.astype(bool), event_active


def _individual_lane_clearance_safe(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    index: int,
    mission: np.ndarray,
    context: np.ndarray | None = None,
) -> bool:
    """Mirror the environment's dynamic gap/TTC acceptance from observations."""

    if context is None:
        context = _target_lane_context(observations[PLATOON_IDS[index]], mission)
    target_lane = int(round(float(mission[2] * 2.0)))
    ego_speed = max(
        0.0, float(observations[PLATOON_IDS[index]]["ego"][1])
    )
    front_is_platoon = bool(
        index > 0
        and int(
            round(float(observations[PLATOON_IDS[index - 1]]["ego"][4]))
        )
        == target_lane
    )
    rear_is_platoon = bool(
        index + 1 < len(PLATOON_IDS)
        and int(
            round(float(observations[PLATOON_IDS[index + 1]]["ego"][4]))
        )
        == target_lane
    )
    front_required = MIN_SAFE_GAP
    if not front_is_platoon:
        front_required = max(
            front_required, LANE_CHANGE_MIN_FRONT_HEADWAY * ego_speed
        )
    rear_relative_speed = float(context[1]) * 15.0
    rear_speed = max(0.0, ego_speed + rear_relative_speed)
    rear_required = max(
        MIN_SAFE_GAP,
        MIN_SAFE_GAP
        + LANE_CHANGE_REAR_LOOKAHEAD * max(0.0, rear_relative_speed),
    )
    if not rear_is_platoon:
        rear_required = max(
            rear_required, LANE_CHANGE_MIN_REAR_HEADWAY * rear_speed
        )
    return bool(
        float(mission[6]) * 200.0 >= front_required
        and float(mission[7]) * 200.0 >= rear_required
        and float(context[2]) * 10.0 >= 1.5
        and float(context[3]) * 10.0 >= 1.5
    )


def _apply_shape_control(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    actions: dict[str, dict[str, Any]],
    target_gaps: np.ndarray,
) -> None:
    for pair_index, follower in enumerate(PLATOON_IDS[1:]):
        local = np.asarray(observations[follower]["local_predecessor"], np.float32)
        gap = float(local[4])
        relative_speed = float(local[1])
        correction = np.clip(
            0.18 * (gap - float(target_gaps[pair_index]))
            + 0.55 * relative_speed,
            -0.8,
            0.8,
        )
        base_acceleration = float(np.asarray(actions[follower]["accel"])[0])
        actions[follower]["accel"] = np.asarray(
            [np.clip(base_acceleration + correction, -3.0, 1.2)], np.float32
        )
    gaps = np.asarray(observations["leader"]["platoon_state"], np.float32)[1:, 4]
    valid = gaps[gaps > 0.0]
    correction = 0.0
    if len(valid) and float(np.mean(valid)) > float(np.mean(target_gaps)) + 0.8:
        correction = -0.35
    if len(valid) and float(np.max(valid)) > 18.0:
        correction = -0.8
    leader = float(np.asarray(actions["leader"]["accel"])[0])
    actions["leader"]["accel"] = np.asarray(
        [np.clip(leader + correction, -3.0, 1.2)], np.float32
    )


class GraphMixedLanePolicy:
    def __init__(
        self,
        base_policy: Any,
        adapter: CooperativeLaneActorCritic,
        device: str,
        deterministic: bool,
        expert: bool = False,
    ) -> None:
        self.base = base_policy
        self.adapter = adapter
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.expert = bool(expert)
        self.last_trace: MixedLaneTrace | None = None
        self._cooldown = np.zeros(6, np.int64)

    def reset(self) -> None:
        self.base.reset()
        self.last_trace = None
        self._cooldown.fill(0)

    def act(self, observations: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
        base_actions = self.base.act(observations)
        feature_array = common_lane_features(observations, base_actions)
        mask_array, active_value = lane_action_masks(
            observations, atomic_team_entry=True
        )
        self._cooldown = np.maximum(self._cooldown - 1, 0)
        mask_array &= self._cooldown == 0
        features = torch.as_tensor(feature_array, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(mask_array, dtype=torch.bool, device=self.device)
        active = torch.as_tensor(active_value, dtype=torch.bool, device=self.device)
        if self.expert:
            with torch.no_grad():
                _, _, value = self.adapter(features)
            change = mask.float()
            target = torch.full((5,), 10.0 if active_value else 11.8, device=self.device)
            gap_raw = gap_target_to_raw(target)
            with torch.no_grad():
                log_prob, _, _ = self.adapter.evaluate_actions(
                    features.unsqueeze(0), mask.unsqueeze(0), active.unsqueeze(0),
                    change.unsqueeze(0), gap_raw.unsqueeze(0)
                )
            sample = LanePolicySample(
                change=change.unsqueeze(0), gap_raw=gap_raw.unsqueeze(0),
                log_prob=log_prob, value=value, entropy=torch.zeros(1, device=self.device),
                change_logits=torch.zeros(1, 6, device=self.device), gap_mean=gap_raw.unsqueeze(0)
            )
        else:
            with torch.set_grad_enabled(not self.deterministic):
                sample = self.adapter.sample(features, mask, active, self.deterministic)
        target_gaps = gap_raw_to_target(sample.gap_raw[0]).detach().cpu().numpy()
        actions = {
            agent: {"accel": np.asarray(base_actions[agent]["accel"], np.float32).copy(), "lane_change": 0}
            for agent in PLATOON_IDS
        }
        if active_value:
            _apply_shape_control(observations, actions, target_gaps)
        missions = {agent: np.asarray(observations[agent]["lane_mission"], np.float32) for agent in PLATOON_IDS}
        for index, agent in enumerate(PLATOON_IDS):
            if sample.change[0, index].item() > 0.5:
                direction = int(np.sign(missions[agent][3]))
                actions[agent]["lane_change"] = 1 if direction > 0 else 2
                self._cooldown[index] = 15
        action_unit = np.concatenate(
            (
                np.where(sample.change[0].detach().cpu().numpy() > 0.5, 0.9, -0.9),
                np.clip((target_gaps - 10.75) / 1.25, -1.0, 1.0),
            )
        ).astype(np.float32)
        self.last_trace = MixedLaneTrace(
            features=features.detach().cpu(), state=feature_array.reshape(-1).copy(),
            change_mask=mask.detach().cpu(), gap_active=active.detach().cpu(),
            change=sample.change[0].detach().cpu(), gap_raw=sample.gap_raw[0].detach().cpu(),
            action_unit=action_unit, log_prob=float(sample.log_prob[0].detach().cpu()),
            value=float(sample.value[0].detach().cpu()), target_gaps=target_gaps.copy()
        )
        return actions

    def record_applied(self, infos: Mapping[str, Mapping[str, Any]]) -> None:
        self.base.record_applied(infos)


class NativeMixedLanePolicy:
    def __init__(
        self,
        base_policy: Any,
        algorithm: NativeLaneAlgorithm,
        deterministic: bool,
        expert: bool = False,
    ) -> None:
        self.base = base_policy
        self.algorithm = algorithm
        self.deterministic = bool(deterministic)
        self.expert = bool(expert)
        self.last_trace: MixedLaneTrace | None = None
        self._cooldown = np.zeros(6, np.int64)

    def reset(self) -> None:
        self.base.reset()
        self.last_trace = None
        self._cooldown.fill(0)

    def act(self, observations: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
        base_actions = self.base.act(observations)
        features = common_lane_features(observations, base_actions)
        state = features.reshape(-1).astype(np.float32, copy=False)
        intent_mask, active_value = lane_action_masks(
            observations, atomic_team_entry=False
        )
        self._cooldown = np.maximum(self._cooldown - 1, 0)
        execution_mask = intent_mask & (self._cooldown == 0)
        if self.expert:
            action_unit = np.concatenate(
                # Supervise the persistent semantic intent. Command repetition
                # is an execution concern handled by the cooldown below and is
                # deliberately excluded from the learned state-action mapping.
                (np.where(intent_mask, 0.9, -0.9), np.full(5, -0.6))
            ).astype(np.float32)
            sample = NativeActionSample(action_unit)
        else:
            sample = self.algorithm.act(state, self.deterministic)
            action_unit = sample.action
        target_gaps = np.clip(10.75 + 1.25 * action_unit[6:], 9.5, 12.0)
        change = (action_unit[:6] > 0.0) & execution_mask
        actions = {
            agent: {"accel": np.asarray(base_actions[agent]["accel"], np.float32).copy(), "lane_change": 0}
            for agent in PLATOON_IDS
        }
        if active_value:
            _apply_shape_control(observations, actions, target_gaps)
        for index, agent in enumerate(PLATOON_IDS):
            if change[index]:
                direction = int(np.sign(observations[agent]["lane_mission"][3]))
                actions[agent]["lane_change"] = 1 if direction > 0 else 2
                self._cooldown[index] = 15
        self.last_trace = MixedLaneTrace(
            features=torch.as_tensor(features), state=state.copy(),
            change_mask=torch.as_tensor(execution_mask), gap_active=torch.as_tensor(active_value),
            change=torch.as_tensor(change.astype(np.float32)),
            gap_raw=torch.as_tensor(action_unit[6:].copy()), action_unit=action_unit.copy(),
            log_prob=sample.log_prob, value=sample.value, target_gaps=target_gaps.copy()
        )
        return actions

    def record_applied(self, infos: Mapping[str, Mapping[str, Any]]) -> None:
        self.base.record_applied(infos)


class CaccLcPolicy:
    """CACC longitudinal control plus independent bottom-layer lane changes."""

    def __init__(self, base_policy: Any) -> None:
        self.base = base_policy
        self.last_trace = None
        self._cooldown = np.zeros(6, np.int64)

    def reset(self) -> None:
        self.base.reset()
        self._cooldown.fill(0)

    def act(self, observations: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
        actions = self.base.act(observations)
        self._cooldown = np.maximum(self._cooldown - 1, 0)
        for index, agent in enumerate(PLATOON_IDS):
            actions[agent]["lane_change"] = 0
            mission = np.asarray(observations[agent]["lane_mission"], np.float32)
            if self._cooldown[index] > 0 or mission[8] <= 0.5 or abs(mission[3]) < 0.5:
                continue
            clearance_safe = _individual_lane_clearance_safe(
                observations,
                index,
                mission,
            )
            if not clearance_safe:
                continue
            if agent == "leader":
                local_trigger = mission[4] * 200.0 <= 135.0
            else:
                predecessor = PLATOON_IDS[index - 1]
                predecessor_mission = np.asarray(
                    observations[predecessor]["lane_mission"], np.float32
                )
                predecessor_in_target = abs(
                    predecessor_mission[5] - mission[2]
                ) <= 0.05
                local_front = np.asarray(observations[agent]["local_front"], np.float32)
                local_trigger = bool(
                    predecessor_in_target
                    and (local_front[4] <= 45.0 or mission[4] * 200.0 <= 80.0)
                )
            if local_trigger:
                actions[agent]["lane_change"] = 1 if mission[3] > 0.0 else 2
                self._cooldown[index] = 15
        return actions

    def record_applied(self, infos: Mapping[str, Mapping[str, Any]]) -> None:
        self.base.record_applied(infos)


def graph_adapter(device: str) -> CooperativeLaneActorCritic:
    return CooperativeLaneActorCritic(feature_dim=NODE_FEATURE_DIM).to(device)


def load_graph_adapter(path: Path, device: str) -> CooperativeLaneActorCritic:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = graph_adapter(device)
    model.load_state_dict(payload["lateral_state"])
    return model

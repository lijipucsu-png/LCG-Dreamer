"""Hybrid policy that overlays learned lateral/shape actions on frozen control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from classical_comparison.evaluate import CaccPolicy, GraphPolicy
from platoon_sumo.env import PLATOON_IDS
from predictive_comparison.evaluate import PredictivePolicy

from .model import (
    NODE_FEATURE_DIM,
    CooperativeLaneActorCritic,
    LanePolicySample,
    gap_raw_to_target,
    gap_target_to_raw,
)
from .protocol import LaneChangeProtocol


@dataclass
class LaneActionTrace:
    features: Tensor
    change_mask: Tensor
    gap_active: Tensor
    change: Tensor
    gap_raw: Tensor
    log_prob: Tensor
    value: Tensor
    target_gaps: np.ndarray


def create_base_policy(
    method: str, protocol: LaneChangeProtocol, device: str
) -> Any:
    checkpoint = protocol.base_checkpoint(method)
    if method == "graph":
        assert checkpoint is not None
        return GraphPolicy(checkpoint, device)
    if method == "cacc":
        return CaccPolicy()
    assert checkpoint is not None
    return PredictivePolicy(checkpoint, device)


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
            )
        ).astype(np.float32, copy=False)
        rows.append(row)
    result = np.stack(rows)
    if result.shape != (len(PLATOON_IDS), NODE_FEATURE_DIM):
        raise RuntimeError(f"unexpected lane feature shape: {result.shape}")
    return result


def lane_action_masks(
    observations: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, bool]:
    missions = np.stack(
        [np.asarray(observations[agent]["lane_mission"], np.float32) for agent in PLATOON_IDS]
    )
    event_active = bool(missions[0, 8] > 0.5)
    leader_platoon = np.asarray(
        observations["leader"]["platoon_state"], np.float32
    )
    gaps = leader_platoon[1:, 4]
    valid_gaps = gaps[gaps > 0.0]
    mean_gap = float(np.mean(valid_gaps)) if len(valid_gaps) else 12.0
    obstacle_distance = float(missions[0, 4])
    elapsed = float(missions[0, 9])
    shape_ready = mean_gap <= 10.8 or obstacle_distance <= 0.85 or elapsed >= 0.30
    direction_pending = np.abs(missions[:, 3]) > 0.5
    clearance_safe = (
        missions[:, 6] * 200.0 >= 5.0
    ) & (missions[:, 7] * 200.0 >= 5.0)
    mask = (
        event_active & shape_ready & direction_pending & clearance_safe
    )
    return mask.astype(bool), event_active


class CooperativeLanePolicy:
    """Transfer wrapper; only ``adapter`` is optimized during focused training."""

    def __init__(
        self,
        base_policy: Any,
        adapter: CooperativeLaneActorCritic,
        device: str | torch.device,
        deterministic: bool = False,
        expert: bool = False,
    ) -> None:
        self.base = base_policy
        self.adapter = adapter
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.expert = bool(expert)
        self.last_trace: LaneActionTrace | None = None
        self.last_actions: dict[str, dict[str, Any]] | None = None
        self._command_cooldown = np.zeros(len(PLATOON_IDS), np.int64)

    @property
    def name(self) -> str:
        return str(self.base.name)

    def reset(self) -> None:
        self.base.reset()
        self.last_trace = None
        self.last_actions = None
        self._command_cooldown.fill(0)

    def act(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> dict[str, dict[str, Any]]:
        base_actions = self.base.act(observations)
        feature_array = common_lane_features(observations, base_actions)
        change_mask_array, gap_active_value = lane_action_masks(observations)
        self._command_cooldown = np.maximum(self._command_cooldown - 1, 0)
        change_mask_array &= self._command_cooldown == 0
        features = torch.as_tensor(
            feature_array, dtype=torch.float32, device=self.device
        )
        change_mask = torch.as_tensor(
            change_mask_array, dtype=torch.bool, device=self.device
        )
        gap_active = torch.as_tensor(
            gap_active_value, dtype=torch.bool, device=self.device
        )
        if self.expert:
            with torch.no_grad():
                _, _, value = self.adapter(features)
            change = change_mask.float()
            target = torch.full(
                (5,), 10.0 if gap_active_value else 11.5, device=self.device
            )
            gap_raw = gap_target_to_raw(target)
            with torch.no_grad():
                log_prob, _, _ = self.adapter.evaluate_actions(
                    features.unsqueeze(0),
                    change_mask.unsqueeze(0),
                    gap_active.unsqueeze(0),
                    change.unsqueeze(0),
                    gap_raw.unsqueeze(0),
                )
            sample = LanePolicySample(
                change=change.unsqueeze(0),
                gap_raw=gap_raw.unsqueeze(0),
                log_prob=log_prob,
                value=value,
                entropy=torch.zeros(1, device=self.device),
                change_logits=torch.zeros(1, 6, device=self.device),
                gap_mean=gap_raw.unsqueeze(0),
            )
        else:
            with torch.set_grad_enabled(not self.deterministic):
                sample = self.adapter.sample(
                    features,
                    change_mask,
                    gap_active,
                    self.deterministic,
                )
        target_gaps = (
            gap_raw_to_target(sample.gap_raw[0]).detach().cpu().numpy()
        )
        actions = {
            agent: {
                "accel": np.asarray(base_actions[agent]["accel"], np.float32).copy(),
                "lane_change": 0,
            }
            for agent in PLATOON_IDS
        }
        if gap_active_value:
            for pair_index, follower in enumerate(PLATOON_IDS[1:]):
                local = np.asarray(
                    observations[follower]["local_predecessor"], np.float32
                )
                gap = float(local[4])
                relative_speed = float(local[1])
                correction = np.clip(
                    0.18 * (gap - float(target_gaps[pair_index]))
                    + 0.55 * relative_speed,
                    -0.8,
                    0.8,
                )
                base_acceleration = float(
                    np.asarray(actions[follower]["accel"]).reshape(-1)[0]
                )
                actions[follower]["accel"] = np.asarray(
                    [np.clip(base_acceleration + correction, -3.0, 1.2)],
                    np.float32,
                )
            leader_gaps = np.asarray(
                observations["leader"]["platoon_state"], np.float32
            )[1:, 4]
            valid = leader_gaps[leader_gaps > 0.0]
            leader_correction = 0.0
            if len(valid) and float(np.mean(valid)) > float(np.mean(target_gaps)) + 0.8:
                leader_correction = -0.35
            if len(valid) and float(np.max(valid)) > 18.0:
                leader_correction = -0.8
            leader_base = float(
                np.asarray(actions["leader"]["accel"]).reshape(-1)[0]
            )
            actions["leader"]["accel"] = np.asarray(
                [np.clip(leader_base + leader_correction, -3.0, 1.2)],
                np.float32,
            )
        missions = {
            agent: np.asarray(observations[agent]["lane_mission"], np.float32)
            for agent in PLATOON_IDS
        }
        for index, agent in enumerate(PLATOON_IDS):
            if bool(sample.change[0, index].item() > 0.5):
                direction = int(np.sign(missions[agent][3]))
                actions[agent]["lane_change"] = 1 if direction > 0 else 2
                self._command_cooldown[index] = 15
        self.last_trace = LaneActionTrace(
            features=features.detach().cpu(),
            change_mask=change_mask.detach().cpu(),
            gap_active=gap_active.detach().cpu(),
            change=sample.change[0].detach().cpu(),
            gap_raw=sample.gap_raw[0].detach().cpu(),
            log_prob=sample.log_prob[0].detach().cpu(),
            value=sample.value[0].detach().cpu(),
            target_gaps=target_gaps.copy(),
        )
        self.last_actions = actions
        return actions

    def record_applied(self, infos: Mapping[str, Mapping[str, Any]]) -> None:
        self.base.record_applied(infos)


def load_adapter(
    checkpoint: str | Path,
    device: str | torch.device,
) -> CooperativeLaneActorCritic:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CooperativeLaneActorCritic()
    model.load_state_dict(payload["adapter_state"])
    return model.to(device)

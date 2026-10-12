"""Leader/follower inference wrappers and complete six-truck controller."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from platoon_sumo.env import (
    CUTIN_TARGET_GAP,
    FOLLOWER_IDS,
    PLATOON_LENGTH,
    PLATOON_IDS,
    STEP_LENGTH,
    TARGET_GAP,
)

from .config import SystemConfig
from .models import (
    CPIDLMDelayCompensator,
    DreamerV3Leader,
    FOLLOWER_MEMORY_STEPS,
    FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX,
    FOLLOWER_PREDICTED_ACCELERATION_INDEX,
    FOLLOWER_PREDICTED_SPEED_INDEX,
    MultiAgentFollowerSAC,
    STGNNTrajectoryPredictor,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def tensorize(observation: Mapping[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


class FollowerStateBuilder(nn.Module):
    """Convert predictions and causal observations into follower policy states."""

    def forward(
        self,
        observations: Mapping[str, Tensor],
        reconstructed: Tensor,
        cutin_probability: Tensor,
        predicted_neighbor_trajectories: Tensor,
    ) -> Tensor:
        ego = observations["ego"]
        local_predecessor = observations["local_predecessor"]
        local_front = observations["local_front"]
        platoon = observations["platoon_state"]
        road = observations["road_profile"]
        communication = observations["communication"]
        recovery = observations["recovery"]
        leader_plan = observations["leader_plan"]
        front_history = observations["front_history"]
        front_mask = observations["front_mask"].bool()
        v2i_flow = observations["v2i_flow"]
        nearest_mode = predicted_neighbor_trajectories[:, :, 0, 0]
        neighbor_mask = observations["neighbor_mask"]
        distances = nearest_mode[..., 0].abs().masked_fill(~neighbor_mask.bool(), 1e4)
        nearest_index = distances.argmin(-1)
        batch_index = torch.arange(ego.shape[0], device=ego.device)
        nearest = nearest_mode[batch_index, nearest_index]
        cutin = (cutin_probability * neighbor_mask).amax(-1)
        desired_gap = TARGET_GAP - cutin.clamp(0.0, 1.0) * (
            TARGET_GAP - CUTIN_TARGET_GAP
        )
        leader_speed = platoon[:, 0, 1]
        leader_acceleration = platoon[:, 0, 2]
        planned_speeds = leader_plan[:, 1:]
        latest_front = front_history[:, :, -1]
        ahead_mask = front_mask & (latest_front[..., 0] > 0.0)
        front_count = ahead_mask.float().sum(-1)
        front_min_gap = latest_front[..., 0].masked_fill(~ahead_mask, 800.0).amin(-1)
        front_min_gap = torch.where(
            ahead_mask.any(-1), front_min_gap, torch.zeros_like(front_min_gap)
        )
        front_mean_speed = (
            (latest_front[..., 2] * ahead_mask).sum(-1)
            / ahead_mask.float().sum(-1).clamp_min(1.0)
        )
        flow_mask = v2i_flow[..., 0] > 0.0
        v2i_mean_speed = (
            (v2i_flow[..., 0] * flow_mask).sum(-1)
            / flow_mask.float().sum(-1).clamp_min(1.0)
        )
        v2i_mean_deviation = (
            (v2i_flow[..., 2] * flow_mask).sum(-1)
            / flow_mask.float().sum(-1).clamp_min(1.0)
        )
        return torch.stack(
            (
                ego[:, 0],
                ego[:, 1],
                ego[:, 2],
                local_front[:, 4],
                reconstructed[:, 1],
                local_predecessor[:, 2],
                desired_gap,
                road[:, 0, 1],
                road[:, 0, 2],
                cutin,
                nearest[:, 0],
                nearest[:, 2],
                leader_speed,
                leader_acceleration,
                communication[:, 1],
                ego[:, 7],
                recovery[:, 0],
                recovery[:, 1],
                recovery[:, 2],
                recovery[:, 3],
                recovery[:, 4],
                recovery[:, 5],
                ego[:, 5],
                leader_plan[:, 0],
                planned_speeds.mean(-1),
                planned_speeds.amin(-1),
                planned_speeds[:, -1],
                road[..., 2].amin(-1),
                road[..., 1].abs().amax(-1),
                front_count,
                front_min_gap,
                front_mean_speed,
                v2i_mean_speed,
                v2i_mean_deviation,
            ),
            dim=-1,
        )


class PlatoonController:
    """DreamerV3 leader plus shared CTDE multi-agent RL follower policy."""

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or SystemConfig()
        model = self.config.model
        self.device = resolve_device(self.config.training.device)
        self.leader = DreamerV3Leader(
            hidden_dim=model.hidden_dim,
            stochastic_dim=model.stochastic_dim,
            stochastic_classes=model.stochastic_classes,
            trajectory_modes=model.trajectory_modes,
            prediction_horizon=model.prediction_horizon,
        ).to(self.device)
        self.delay = CPIDLMDelayCompensator(
            hidden_dim=model.hidden_dim, horizon=model.prediction_horizon
        ).to(self.device)
        self.trajectory = STGNNTrajectoryPredictor(
            hidden_dim=model.hidden_dim,
            horizon=model.prediction_horizon,
            modes=model.trajectory_modes,
        ).to(self.device)
        self.state_builder = FollowerStateBuilder().to(self.device)
        self.follower = MultiAgentFollowerSAC(self.config.follower_rl).to(self.device)
        self._leader_state = self.leader.initial(1, self.device)
        self._leader_action = torch.zeros(1, 1, device=self.device)
        self._follower_policy_history: dict[str, deque[Tensor]] = {}
        self._reset_follower_history()

    def _reset_follower_history(self) -> None:
        self._follower_policy_history = {
            agent: deque(maxlen=FOLLOWER_MEMORY_STEPS) for agent in FOLLOWER_IDS
        }

    def reset(self) -> None:
        self._leader_state = self.leader.initial(1, self.device)
        self._leader_action.zero_()
        self._reset_follower_history()

    def record_applied_actions(
        self, infos: Mapping[str, Mapping[str, object]]
    ) -> None:
        """Keep recurrent dynamics aligned with the action the plant received."""

        applied = infos.get("leader", {}).get("applied_acceleration")
        if applied is not None:
            self._leader_action.fill_(float(applied))

    @staticmethod
    def follower_control_predictions(
        policy_states: Tensor,
    ) -> dict[str, tuple[float, float]]:
        current = (
            policy_states[:, -1]
            if policy_states.ndim == 3
            else policy_states
        )
        values = current.detach().cpu()
        return {
            follower: (
                float(values[index, FOLLOWER_PREDICTED_SPEED_INDEX]),
                float(values[index, FOLLOWER_PREDICTED_ACCELERATION_INDEX]),
            )
            for index, follower in enumerate(FOLLOWER_IDS)
        }

    @torch.no_grad()
    def _follower_features(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        ablations: frozenset[str],
    ) -> tuple[Tensor, Tensor, object]:
        follower_batch = {
            key: torch.stack(
                [torch.as_tensor(observations[agent][key], dtype=torch.float32) for agent in FOLLOWER_IDS]
            ).to(self.device)
            for key in observations[FOLLOWER_IDS[0]]
        }
        delay_prediction = self.delay(
            follower_batch["v2v_history"], follower_batch["local_predecessor"]
        )
        human_prediction = self.trajectory(
            follower_batch["neighbor_history"], follower_batch["neighbor_mask"]
        )
        reconstructed = delay_prediction.current_state
        delay_future = delay_prediction.future_mean
        cutin_probability = human_prediction.cutin_logits.sigmoid()
        if "delay_compensation" in ablations:
            reconstructed = follower_batch["v2v_history"][:, -1, 4:7]
            delay_future = reconstructed[:, None].expand(
                -1, self.config.model.prediction_horizon, -1
            )
        if "cutin_prediction" in ablations:
            cutin_probability = torch.zeros_like(cutin_probability)
        selected_mode = human_prediction.mode_logits.argmax(-1)
        selected_trajectories = human_prediction.trajectories.gather(
            2,
            selected_mode[..., None, None, None].expand(
                -1,
                -1,
                1,
                human_prediction.trajectories.shape[-2],
                human_prediction.trajectories.shape[-1],
            ),
        )
        states = self.state_builder(
            follower_batch,
            reconstructed,
            cutin_probability,
            selected_trajectories,
        )
        selected = selected_trajectories.squeeze(2)
        mask = follower_batch["neighbor_mask"].bool()
        current_distance = selected[..., 0, 0].abs().masked_fill(~mask, 1e4)
        nearest_index = current_distance.argmin(-1)
        batch_index = torch.arange(selected.shape[0], device=self.device)
        nearest_future = selected[batch_index, nearest_index]
        has_neighbor = mask.any(-1)
        nearest_future_distance = nearest_future[..., 0].abs().amin(-1)
        nearest_future_distance = torch.where(
            has_neighbor,
            nearest_future_distance,
            torch.zeros_like(nearest_future_distance),
        )
        nearest_lateral_end = torch.where(
            has_neighbor,
            nearest_future[..., -1, 1],
            torch.zeros_like(nearest_future[..., -1, 1]),
        )
        local_front = follower_batch["local_front"]
        closing_speed = (-local_front[:, 1]).clamp_min(0.0)
        front_ttc = torch.where(
            (local_front[:, 5] > 0.5) & (closing_speed > 1e-3),
            local_front[:, 4].clamp_min(0.0) / closing_speed,
            torch.full_like(closing_speed, 10.0),
        ).clamp(0.0, 10.0)

        horizon_seconds = self.config.model.prediction_horizon * STEP_LENGTH
        ego = follower_batch["ego"]
        ego_future_station = (
            ego[:, 0]
            + ego[:, 1] * horizon_seconds
            + 0.5 * ego[:, 2] * horizon_seconds**2
        )
        predicted_gap_end = (
            delay_future[:, -1, 0] - ego_future_station - PLATOON_LENGTH
        )
        future_summary = torch.stack(
            (
                predicted_gap_end,
                delay_future[:, -1, 1],
                delay_future[..., 2].mean(-1),
                delay_future[..., 2].amin(-1),
                nearest_future_distance,
                nearest_lateral_end,
                local_front[:, 1],
                front_ttc,
            ),
            dim=-1,
        )
        return states, torch.cat((states, future_summary), dim=-1), human_prediction

    @torch.no_grad()
    def follower_states(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        ablations: frozenset[str] = frozenset(),
    ) -> tuple[Tensor, object]:
        states, _, human_prediction = self._follower_features(
            observations, ablations
        )
        return states, human_prediction

    @torch.no_grad()
    def follower_policy_states(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        ablations: frozenset[str] = frozenset(),
        update_history: bool = True,
    ) -> tuple[Tensor, object]:
        _, current_states, human_prediction = self._follower_features(
            observations, ablations
        )
        if update_history:
            for index, agent in enumerate(FOLLOWER_IDS):
                history = self._follower_policy_history[agent]
                current = current_states[index].detach()
                if not history:
                    history.extend(current.clone() for _ in range(FOLLOWER_MEMORY_STEPS))
                else:
                    history.append(current)
        sequences = torch.stack(
            [
                torch.stack(tuple(self._follower_policy_history[agent]))
                for agent in FOLLOWER_IDS
            ]
        )
        return sequences, human_prediction

    @torch.no_grad()
    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        deterministic: bool = True,
        leader_deterministic: bool | None = None,
        follower_deterministic: bool | None = None,
        ablations: frozenset[str] = frozenset(),
        policy_states: Tensor | None = None,
    ) -> dict[str, dict[str, np.ndarray | int]]:
        self.eval()
        leader_deterministic = (
            deterministic if leader_deterministic is None else leader_deterministic
        )
        follower_deterministic = (
            deterministic if follower_deterministic is None else follower_deterministic
        )
        leader_observation = tensorize(observations["leader"], self.device)
        leader_action, self._leader_state, _ = self.leader.infer(
            leader_observation,
            self._leader_state,
            self._leader_action,
            deterministic=leader_deterministic,
        )
        self._leader_action = leader_action
        states = policy_states
        if states is None:
            states, _ = self.follower_policy_states(observations, ablations)
        # A zero-delay packet is available in the same decision tick. Inject
        # the just-computed leader action into the causal state/history so all
        # followers can respond synchronously. Delayed packets retain their
        # FIFO value and must be compensated by the learned residual policy.
        for index, agent in enumerate(FOLLOWER_IDS):
            communication = observations[agent]["communication"]
            if float(communication[0]) > 1e-6:
                continue
            current_leader_action = float(leader_action[0, 0])
            if states.ndim == 3:
                states[index, -1, FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX] = (
                    current_leader_action
                )
            else:
                states[index, FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX] = (
                    current_leader_action
                )
            history = self._follower_policy_history.get(agent)
            if history:
                history[-1][FOLLOWER_LEADER_PLAN_ACCELERATION_INDEX] = (
                    current_leader_action
                )
        follower_actions = self.follower.act(
            states, deterministic=follower_deterministic
        )
        result = {
            "leader": {
                "accel": leader_action[0].cpu().numpy().astype(np.float32),
                "lane_change": 0,
            }
        }
        for index, agent in enumerate(FOLLOWER_IDS):
            result[agent] = {
                "accel": follower_actions.acceleration[index]
                .cpu()
                .numpy()
                .astype(np.float32),
                "lane_change": 0,
            }
        return result

    def eval(self) -> None:
        self.leader.eval()
        self.delay.eval()
        self.trajectory.eval()
        self.follower.eval()

    def state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {
            "leader": self.leader.state_dict(),
            "delay": self.delay.state_dict(),
            "trajectory": self.trajectory.state_dict(),
            "follower": self.follower.state_dict(),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Mapping[str, Tensor]],
        require_follower: bool = True,
    ) -> None:
        self.leader.load_state_dict(state["leader"])
        self.delay.load_state_dict(state["delay"])
        self.trajectory.load_state_dict(state["trajectory"])
        if "follower" in state:
            self.follower.load_state_dict(state["follower"])
        elif require_follower:
            raise RuntimeError(
                "Checkpoint predates the MARL follower policy and cannot be used "
                "for inference; migrate it through train-rl first"
            )

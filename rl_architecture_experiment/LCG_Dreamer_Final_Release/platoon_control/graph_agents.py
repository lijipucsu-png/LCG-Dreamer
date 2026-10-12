"""Inference wrapper and compact observation adapter for graph Dreamer."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS, PREDECESSOR

from .agents import PlatoonController, resolve_device
from .models.delay import CPIDLMDelayCompensator
from .models.dreamer import DreamerV3Leader
from .models.follower_rl import (
    FOLLOWER_POLICY_STATE_DIM,
    FOLLOWER_PREDICTED_ACCELERATION_INDEX,
    FOLLOWER_PREDICTED_SPEED_INDEX,
)
from .models.graph_dreamer import (
    FORMATION_BREAK_GAP,
    FRONT_COUNT,
    NEIGHBOR_COUNT,
    NODE_COUNT,
    GraphDreamerConfig,
    GraphPolicyOutput,
    GraphRSSMState,
    MALatentGraphDreamer,
)
from .models.trajectory import STGNNTrajectoryPredictor


AGENT_INDEX = {agent: index for index, agent in enumerate(PLATOON_IDS)}


class GraphTeacherBundle:
    """Frozen predictors used only to create distillation targets."""

    def __init__(self, directory: str | Path, device: torch.device) -> None:
        directory = Path(directory)
        self.device = device
        self.delay = self._load(
            directory / "cpidlm_best.pt", CPIDLMDelayCompensator
        )
        self.trajectory = self._load(
            directory / "trajectory_best.pt", STGNNTrajectoryPredictor
        )
        self.leader_front = self._load(
            directory / "leader_front_best.pt", DreamerV3Leader
        )
        for model in (self.delay, self.trajectory, self.leader_front):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def _load(self, path: Path, model_type: type) -> torch.nn.Module:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = model_type(**checkpoint["model_arguments"])
        strict = checkpoint.get("model_name") != "leader_front"
        model.load_state_dict(checkpoint["model"], strict=strict)
        return model.to(self.device)

    @torch.no_grad()
    def targets(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        follower_v2v = torch.stack(
            [
                torch.as_tensor(observations[agent]["v2v_history"], dtype=torch.float32)
                for agent in FOLLOWER_IDS
            ]
        ).to(self.device)
        follower_local = torch.stack(
            [
                torch.as_tensor(
                    observations[agent]["local_predecessor"], dtype=torch.float32
                )
                for agent in FOLLOWER_IDS
            ]
        ).to(self.device)
        delay_prediction = self.delay(follower_v2v, follower_local)
        delay = torch.zeros(
            NODE_COUNT,
            delay_prediction.future_mean.shape[-2],
            3,
            device=self.device,
        )
        delay[1:] = delay_prediction.future_mean

        neighbor_history = torch.stack(
            [
                torch.as_tensor(
                    observations[agent]["neighbor_history"], dtype=torch.float32
                )
                for agent in PLATOON_IDS
            ]
        ).to(self.device)
        neighbor_mask = torch.stack(
            [
                torch.as_tensor(
                    observations[agent]["neighbor_mask"], dtype=torch.float32
                )
                for agent in PLATOON_IDS
            ]
        ).to(self.device)
        human = self.trajectory(neighbor_history, neighbor_mask)
        cutin = (human.cutin_logits.sigmoid() * neighbor_mask).amax(-1)
        selected_mode = human.mode_logits.argmax(-1)
        selected = human.trajectories.gather(
            2,
            selected_mode[..., None, None, None].expand(
                -1, -1, 1, human.trajectories.shape[-2], human.trajectories.shape[-1]
            ),
        ).squeeze(2)
        end = selected[..., -1, :]
        human_summary = torch.stack(
            (
                end[..., 0].masked_fill(neighbor_mask <= 0.0, 500.0).amin(-1)
                / 500.0,
                (end[..., 0] * neighbor_mask).sum(-1)
                / neighbor_mask.sum(-1).clamp_min(1.0)
                / 500.0,
                (end[..., 1].abs() * neighbor_mask).amax(-1) / 10.0,
                cutin,
            ),
            -1,
        )

        leader_observation = observations["leader"]
        leader_history = torch.as_tensor(
            leader_observation["front_history"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        leader_ego = torch.as_tensor(
            leader_observation["ego"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        leader_mask = torch.as_tensor(
            leader_observation["front_mask"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        front = self.leader_front.front_predictor(leader_history, leader_ego)
        modes = front.mode_logits.argmax(-1)
        chosen = front.trajectories.gather(
            2,
            modes[..., None, None, None].expand(
                -1, -1, 1, front.trajectories.shape[-2], 2
            ),
        ).squeeze(2)
        final = chosen[..., -1, :]
        valid = leader_mask > 0.5
        leader_summary = torch.stack(
            (
                final[..., 0].masked_fill(~valid, 800.0).amin(-1) / 800.0,
                (final[..., 0] * leader_mask).sum(-1)
                / leader_mask.sum(-1).clamp_min(1.0)
                / 800.0,
                (final[..., 1].abs() * leader_mask).amax(-1) / 10.0,
                leader_mask.mean(-1),
            ),
            -1,
        ).squeeze(0)
        human_summary[0] = leader_summary
        return (
            delay.cpu().numpy().astype(np.float32),
            cutin.cpu().numpy().astype(np.float32),
            human_summary.cpu().numpy().astype(np.float32),
        )


class GraphObservationAdapter:
    def __init__(self, teachers: GraphTeacherBundle | None = None) -> None:
        self.teachers = teachers

    @staticmethod
    def _physical_target(observation: Mapping[str, np.ndarray]) -> np.ndarray:
        ego = np.asarray(observation["ego"], np.float32)
        local_front = np.asarray(observation["local_front"], np.float32)
        road = np.asarray(observation["road_profile"], np.float32)
        speed = float(ego[1])
        signal = np.asarray(
            observation.get("signal", np.zeros(4, np.float32)), np.float32
        )
        signal_mask = np.asarray(
            observation.get("signal_mask", np.zeros(1, np.float32)), np.float32
        )
        leader_signal = signal_mask.size > 0 and signal_mask[0] > 0.5
        if leader_signal:
            # The leader's signal is produced from TraCI getLeader and is the
            # lane-certified safety measurement. front_history intentionally
            # contains nearby traffic from other lanes and must not become a
            # hard longitudinal CBF constraint.
            normalized_gap = float(signal[2])
            if normalized_gap < 1.0 - 1e-4:
                gap = max(0.0, 300.0 * normalized_gap + 2.0)
                relative_speed = 30.0 * float(signal[3]) - speed
                front_acceleration = 0.0
            else:
                gap = 0.0
                relative_speed = 0.0
                front_acceleration = 0.0
        else:
            gap = float(local_front[4]) if local_front[5] > 0.5 else 0.0
            relative_speed = (
                float(local_front[1]) if local_front[5] > 0.5 else 0.0
            )
            front_acceleration = (
                float(local_front[2]) if local_front[5] > 0.5 else 0.0
            )
        closing = max(0.0, -relative_speed)
        ttc = min(10.0, gap / closing) if gap > 0.0 and closing > 1e-3 else 10.0
        positive_advisory = road[:, 2][road[:, 2] > 0.0]
        advisory = float(positive_advisory.min()) if positive_advisory.size else 40.0
        return np.asarray(
            [speed, ego[2], gap, relative_speed, front_acceleration, ttc, advisory],
            dtype=np.float32,
        )

    @staticmethod
    def _edges(
        observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        adjacency = np.zeros((NODE_COUNT, NODE_COUNT, 2), np.float32)
        ages = np.zeros_like(adjacency)
        for receiver, follower in enumerate(FOLLOWER_IDS, start=1):
            predecessor = AGENT_INDEX[PREDECESSOR[follower]]
            adjacency[receiver, predecessor, 0] = 1.0
            communication = np.asarray(observations[follower]["communication"])
            # The environment exposes leader-to-follower packet age; use the
            # same measured channel age for the reciprocal V2V broadcast.
            adjacency[0, receiver, 1] = 1.0
            ages[0, receiver, 1] = float(communication[1])
            adjacency[receiver, 0, 1] = 1.0
            adjacency[receiver, predecessor, 1] = 1.0
            ages[receiver, 0, 1] = float(communication[1])
            ages[receiver, predecessor, 1] = float(communication[3])
        return adjacency, ages

    def __call__(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        node_features: list[np.ndarray] = []
        roads: list[np.ndarray] = []
        flows: list[np.ndarray] = []
        front_entities: list[np.ndarray] = []
        front_masks: list[np.ndarray] = []
        neighbor_entities: list[np.ndarray] = []
        neighbor_masks: list[np.ndarray] = []
        physical: list[np.ndarray] = []
        for agent in PLATOON_IDS:
            observation = observations[agent]
            node_features.append(
                np.concatenate(
                    (
                        observation["ego"],
                        observation["local_predecessor"],
                        observation["local_front"],
                        observation["communication"],
                        observation["recovery"],
                        observation["leader_plan"],
                        observation["signal"],
                        observation["signal_mask"],
                    )
                ).astype(np.float32)
            )
            roads.append(np.asarray(observation["road_profile"], np.float32))
            flows.append(np.asarray(observation["v2i_flow"], np.float32))
            front_entities.append(
                np.asarray(observation["front_history"], np.float32)[:, -1]
            )
            front_masks.append(np.asarray(observation["front_mask"], np.float32))
            neighbor_entities.append(
                np.asarray(observation["neighbor_history"], np.float32)[:, -1]
            )
            neighbor_masks.append(
                np.asarray(observation["neighbor_mask"], np.float32)
            )
            physical.append(self._physical_target(observation))
        adjacency, ages = self._edges(observations)
        delay = np.zeros((NODE_COUNT, 30, 3), np.float32)
        cutin = np.zeros(NODE_COUNT, np.float32)
        front = np.zeros((NODE_COUNT, 4), np.float32)
        if self.teachers is not None:
            delay, cutin, front = self.teachers.targets(observations)
        return {
            "node_features": np.stack(node_features),
            "road_profile": np.stack(roads),
            "v2i_flow": np.stack(flows),
            "front_entities": np.stack(front_entities),
            "front_mask": np.stack(front_masks),
            "neighbor_entities": np.stack(neighbor_entities),
            "neighbor_mask": np.stack(neighbor_masks),
            "role": np.asarray([0, 1, 1, 1, 1, 1], np.int64),
            "adjacency": adjacency,
            "edge_ages": ages,
            "physical_target": np.stack(physical),
            "delay_target": delay,
            "cutin_target": cutin,
            "front_target": front,
            "follower_mask": np.asarray([0, 1, 1, 1, 1, 1], np.float32),
        }


def tensorize_graph_observation(
    observation: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, Tensor]:
    return {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


class GraphPlatoonController:
    """Six-vehicle controller backed by a unified latent graph world model."""

    def __init__(
        self,
        model_config: GraphDreamerConfig | None = None,
        device: str = "auto",
        predictor_directory: str | Path | None = None,
        variant: str = "full",
    ) -> None:
        self.model_config = model_config or GraphDreamerConfig()
        self.device = resolve_device(device)
        self.model = MALatentGraphDreamer(self.model_config).to(self.device)
        self.variant = variant
        # Frozen teachers only produce replay labels. Keeping these small
        # predictors on CPU prevents multiple SUMO workers from starving each
        # other's latency-sensitive graph-policy CUDA contexts on Windows.
        teacher_device = (
            torch.device("cpu") if self.device.type == "cuda" else self.device
        )
        teachers = (
            GraphTeacherBundle(predictor_directory, teacher_device)
            if predictor_directory is not None
            and variant != "no_auxiliary_distillation"
            else None
        )
        self.adapter = GraphObservationAdapter(teachers)
        self._state: GraphRSSMState
        self._previous_applied: Tensor
        self._previous_intention: Tensor
        self._prepared_identity: int | None = None
        self._prepared_compact: dict[str, np.ndarray] | None = None
        self._prepared_observation: dict[str, Tensor] | None = None
        self._prepared_context: Tensor | None = None
        self._last_policy: GraphPolicyOutput | None = None
        self._runtime_ablations: frozenset[str] = frozenset()
        self.use_teacher_reward_predictions = teachers is not None
        self.reset()

    @property
    def architecture_id(self) -> str:
        return self.model.architecture_id

    def reset(self) -> None:
        self._state = self.model.initial(1, self.device)
        self._previous_applied = torch.zeros(1, NODE_COUNT, 1, device=self.device)
        self._previous_intention = torch.zeros(
            1, NODE_COUNT, 1, device=self.device
        )
        self._formation_recovery_latch = torch.zeros(
            1, NODE_COUNT - 1, dtype=torch.bool, device=self.device
        )
        self._prepared_identity = None
        self._prepared_compact = None
        self._prepared_observation = None
        self._prepared_context = None
        self._last_policy = None
        self._runtime_ablations = frozenset()

    @torch.no_grad()
    def _prepare(
        self, observations: Mapping[str, Mapping[str, np.ndarray]]
    ) -> tuple[dict[str, Tensor], Tensor]:
        identity = id(observations)
        if self._prepared_identity == identity:
            assert self._prepared_observation is not None
            assert self._prepared_context is not None
            return self._prepared_observation, self._prepared_context
        compact = self.adapter(observations)
        tensor = tensorize_graph_observation(compact, self.device)
        adjacency = tensor["adjacency"]
        if (
            self.variant == "no_semantic_communication"
            or "semantic_communication" in self._runtime_ablations
        ):
            adjacency = adjacency.clone()
            adjacency[..., 1] = 0.0
        state = self._state
        if "latent_extrapolation" in self._runtime_ablations:
            state = replace(state, edge_belief=torch.zeros_like(state.edge_belief))
        self.model.channel.quantization_enabled = (
            "float_semantic_code" not in self._runtime_ablations
        )
        self._state, _, context = self.model.observe_step(
            state,
            self._previous_applied,
            tensor,
            adjacency,
            tensor["edge_ages"],
            0.0,
            self._previous_intention,
        )
        self._prepared_identity = identity
        self._prepared_compact = compact
        self._prepared_observation = tensor
        self._prepared_context = context
        return tensor, context

    @torch.no_grad()
    def follower_policy_states(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        ablations: frozenset[str] = frozenset(),
        update_history: bool = True,
    ) -> tuple[Tensor, None]:
        del update_history
        self._runtime_ablations = ablations
        _, context = self._prepare(observations)
        features = self.model._features(self._state, context)
        if self.use_teacher_reward_predictions and self.adapter.teachers is not None:
            delay = self._prepared_observation["delay_target"]
        else:
            delay = self.model.delay_head(features).view(
                1, NODE_COUNT, self.model_config.prediction_horizon, 3
            ) * self.model.delay_scale
        states = torch.zeros(
            len(FOLLOWER_IDS), 1, FOLLOWER_POLICY_STATE_DIM, device=self.device
        )
        states[:, 0, FOLLOWER_PREDICTED_SPEED_INDEX] = delay[0, 1:, -1, 1]
        states[:, 0, FOLLOWER_PREDICTED_ACCELERATION_INDEX] = delay[
            0, 1:, :, 2
        ].mean(-1)
        return states, None

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
        del leader_deterministic, follower_deterministic, policy_states
        self._runtime_ablations = ablations
        tensor, context = self._prepare(observations)
        physical = tensor["physical_target"]
        pair_gap = physical[:, 1:, 2]
        valid_gap = (pair_gap > 0.0) & (pair_gap < 500.0)
        self._formation_recovery_latch = (
            self._formation_recovery_latch
            | (pair_gap > FORMATION_BREAK_GAP)
        ) & valid_gap & (pair_gap > 12.5)
        output = self.model.policy(
            self._state,
            context,
            physical,
            self._previous_applied,
            deterministic,
            self._previous_intention,
            self._formation_recovery_latch,
        )
        self._last_policy = output
        selected = (
            output.raw_acceleration.clamp(
                self.model_config.action_min, self.model_config.action_max
            )
            if "differentiable_cbf" in self._runtime_ablations
            else output.safe_acceleration
        )
        accelerations = selected[0].cpu().numpy()
        return {
            agent: {
                "accel": accelerations[index].astype(np.float32),
                "lane_change": 0,
            }
            for index, agent in enumerate(PLATOON_IDS)
        }

    def record_applied_actions(
        self,
        infos: Mapping[str, Mapping[str, object]],
        requested_actions: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        values = []
        intentions = []
        for index, agent in enumerate(PLATOON_IDS):
            fallback = (
                float(self._last_policy.safe_acceleration[0, index, 0])
                if self._last_policy is not None
                else 0.0
            )
            values.append(
                float(infos.get(agent, {}).get("applied_acceleration", fallback))
            )
            if requested_actions is not None:
                intention = float(
                    np.asarray(requested_actions[agent]["accel"]).reshape(-1)[0]
                )
            elif self._last_policy is not None:
                intention = float(
                    self._last_policy.raw_acceleration[0, index, 0]
                )
            else:
                intention = fallback
            intentions.append(intention)
        self._previous_applied = torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        ).view(1, NODE_COUNT, 1)
        self._previous_intention = torch.as_tensor(
            intentions, dtype=torch.float32, device=self.device
        ).view(1, NODE_COUNT, 1)
        self._prepared_identity = None
        self._prepared_compact = None
        self._prepared_observation = None
        self._prepared_context = None

    def replay_observation(self) -> dict[str, np.ndarray]:
        """Return the prepared CPU observation without a GPU round trip."""

        if self._prepared_compact is None:
            raise RuntimeError("no graph observation has been prepared")
        return {
            key: np.asarray(value).copy()
            for key, value in self._prepared_compact.items()
        }

    def requested_action_triplet(self) -> tuple[np.ndarray, np.ndarray]:
        if self._last_policy is None:
            zeros = np.zeros((NODE_COUNT, 1), np.float32)
            return zeros, zeros
        return (
            self._last_policy.raw_acceleration[0].detach().cpu().numpy(),
            self._last_policy.safe_acceleration[0].detach().cpu().numpy(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {"graph_dreamer": self.model.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        incompatible = self.model.load_state_dict(
            state["graph_dreamer"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("actor.physical_trunk.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                "incompatible graph checkpoint: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def eval(self) -> None:
        self.model.eval()

    def checkpoint_config(self) -> dict[str, Any]:
        return asdict(self.model_config)

    @staticmethod
    def follower_control_predictions(
        policy_states: Tensor,
    ) -> dict[str, tuple[float, float]]:
        return PlatoonController.follower_control_predictions(policy_states)

from __future__ import annotations

import numpy as np

from mixed_traffic_lane_change.policy import CaccLcPolicy, lane_action_masks
from mixed_traffic_lane_change.protocol import (
    DISPLAY_NAMES,
    LEARNED_METHODS,
    METHODS,
    MixedLaneProtocol,
)
from platoon_sumo.env import PLATOON_IDS


class _BasePolicy:
    def reset(self) -> None:
        return None

    def act(self, observations):
        return {
            agent: {"accel": np.zeros(1, np.float32), "lane_change": 0}
            for agent in observations
        }

    def record_applied(self, infos) -> None:
        return None


def _observation(current_lanes: list[int]) -> dict[str, dict[str, np.ndarray]]:
    observations = {}
    for index, agent in enumerate(PLATOON_IDS):
        mission = np.zeros(10, np.float32)
        mission[2] = 1.0  # Target lane 2 after denormalization.
        mission[3] = float(current_lanes[index] != 2)
        mission[4] = 0.5
        mission[5] = current_lanes[index] / 2.0
        mission[6:8] = 0.5
        mission[8] = 1.0
        observations[agent] = {
            "ego": np.asarray([0.0, 25.0, 0.0, 12.0, current_lanes[index]], np.float32),
            "lane_mission": mission,
            "neighbor_history": np.zeros((8, 8, 8), np.float32),
            "neighbor_mask": np.zeros(8, np.float32),
            "local_front": np.asarray([40.0, 0.0, 25.0, 0.0, 40.0], np.float32),
        }
    platoon_state = np.zeros((len(PLATOON_IDS), 8), np.float32)
    platoon_state[:, 3] = np.asarray(current_lanes, np.float32)
    platoon_state[1:, 4] = 10.0
    observations["leader"]["platoon_state"] = platoon_state
    return observations


def test_protocol_uses_mixed_traffic_without_generic_demand() -> None:
    protocol = MixedLaneProtocol()
    options = protocol.scenario_options(protocol.seed, training=False)
    assert options["lane_change_traffic_enabled"] is True
    assert options["background_initial_scale"] == 0.0
    assert options["background_demand_scale"] == 0.0
    assert options["communication_delay_probability"] == 0.15
    assert protocol.lateral_checkpoint("cacc_lc") is None
    assert set(LEARNED_METHODS) == set(METHODS) - {"cacc_lc"}
    assert DISPLAY_NAMES["cacc_lc"] == "CACC+LC"


def test_cacc_lc_uses_sequential_local_triggers() -> None:
    policy = CaccLcPolicy(_BasePolicy())
    observations = _observation([1, 1, 1, 1, 1, 1])
    actions = policy.act(observations)
    assert actions["leader"]["lane_change"] == 1
    assert all(actions[agent]["lane_change"] == 0 for agent in PLATOON_IDS[1:])

    policy._cooldown.fill(0)
    observations = _observation([2, 1, 1, 1, 1, 1])
    actions = policy.act(observations)
    assert actions["follower_1"]["lane_change"] == 1
    assert all(actions[agent]["lane_change"] == 0 for agent in PLATOON_IDS[2:])


def test_protocol_metadata_freezes_fairness_contract() -> None:
    protocol = MixedLaneProtocol()
    metadata = protocol.to_dict()
    assert metadata["reward_is_method_invariant"] is True
    assert metadata["base_controllers_frozen"] is True
    assert metadata["learned_methods_share_predictive_longitudinal_foundation"] is False
    assert metadata["predictor_assets_equated_across_learned_baselines"] is True
    assert metadata["controller_checkpoints_are_algorithm_specific"] is True
    assert metadata["learned_lateral_parameters_shared_across_methods"] is False
    assert "independent per-vehicle" in metadata["cacc_lateral_controller"]
    assert protocol.protocol_version == 5
    assert protocol.output_root.endswith("mixed_traffic_lane_change_v5")
    assert protocol.imitation_prior_initial_weight == 1.0
    assert protocol.imitation_prior_decay_episodes == 8.0
    assert protocol.selection_interval_episodes == 4
    assert protocol.selection_seed_offsets == (50_000, 50_001, 50_003)
    assert "low/mid/high-density" in metadata["deployment_selection"]
    assert metadata["cooperative_entry_safety"]["graph"].startswith("atomic team")
    assert "individual" in metadata["cooperative_entry_safety"]["learned_baselines"]
    assert protocol.base_checkpoint("ddpg_p").as_posix().endswith(
        "predictive_comparison_v2/training/ddpg_p/complete.pt"
    )


def test_learned_policy_requires_atomic_first_entry_clearance() -> None:
    observations = _observation([1, 1, 1, 1, 1, 1])
    observations["follower_5"]["lane_mission"][7] = 0.01
    mask, active = lane_action_masks(observations)
    assert active
    assert not mask.any()

    # Once the first truck has entered, remaining trucks use their local
    # dynamic gap/TTC acceptance rather than blocking the whole formation.
    observations = _observation([2, 1, 1, 1, 1, 1])
    observations["follower_5"]["lane_mission"][7] = 0.01
    mask, active = lane_action_masks(observations)
    assert active
    assert not bool(mask[0])
    assert mask[1:5].all()
    assert not bool(mask[5])


def test_native_baseline_uses_individual_first_entry_clearance() -> None:
    observations = _observation([1, 1, 1, 1, 1, 1])
    observations["follower_5"]["lane_mission"][7] = 0.01
    mask, active = lane_action_masks(
        observations, atomic_team_entry=False
    )
    assert active
    assert mask[:5].all()
    assert not bool(mask[5])

from __future__ import annotations

import torch

from cooperative_lane_change.model import (
    NODE_COUNT,
    NODE_FEATURE_DIM,
    CooperativeLaneActorCritic,
    gap_raw_to_target,
)
from cooperative_lane_change.protocol import LaneChangeProtocol
from platoon_sumo.env import PlatoonParallelEnv


def test_team_lane_action_is_joint_under_deterministic_inference() -> None:
    model = CooperativeLaneActorCritic()
    with torch.no_grad():
        model.change_head[-1].weight.zero_()
        model.change_head[-1].bias.fill_(2.0)
    features = torch.zeros(NODE_COUNT, NODE_FEATURE_DIM)
    mask = torch.tensor([True, True, False, True, False, True])
    sample = model.sample(
        features,
        mask,
        torch.tensor(True),
        deterministic=True,
    )
    assert torch.equal(sample.change[0].bool(), mask)
    assert sample.log_prob.shape == (1,)
    assert torch.isfinite(sample.log_prob).all()


def test_shape_action_respects_physical_gap_interval() -> None:
    raw = torch.tensor([-100.0, -2.0, 0.0, 2.0, 100.0])
    target = gap_raw_to_target(raw)
    assert float(target.min()) >= 9.5
    assert float(target.max()) <= 12.0
    assert torch.all(target[1:] >= target[:-1])


def test_focused_protocol_keeps_original_model_external() -> None:
    protocol = LaneChangeProtocol()
    options = protocol.scenario_options(protocol.seed, training=False)
    assert options["cooperative_lane_change_enabled"] is True
    assert options["lane_change_target_lanes"] == (2, 1, 0)
    assert protocol.base_checkpoint("graph") != protocol.adapter_checkpoint("graph")


def test_follower_lane_actions_are_opt_in() -> None:
    env = PlatoonParallelEnv()
    assert env.action_space("follower_1")["lane_change"].n == 1
    env._cooperative_lane_change_enabled = True
    assert env.action_space("follower_1")["lane_change"].n == 3

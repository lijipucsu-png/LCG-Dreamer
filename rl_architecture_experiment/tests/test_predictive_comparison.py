from __future__ import annotations

import numpy as np
import torch

from predictive_comparison.algorithms import DelaySwitchingActor
from predictive_comparison.prediction import PREDICTION_LAYOUT
from predictive_comparison.protocol import (
    PREDICTIVE_METHODS,
    PredictiveComparisonProtocol,
)
from predictive_comparison.state import NCKnowledgeStateAdapter
from platoon_sumo.env import PLATOON_IDS


def test_predictive_protocol_freezes_one_predictor_and_reward() -> None:
    payload = PredictiveComparisonProtocol().to_dict()
    assert payload["prediction_enabled_methods"] == list(PREDICTIVE_METHODS)
    assert payload["cacc_prediction_module"] is None
    assert payload["controller_specific_reward_tuning"] is False
    assert payload["future_ground_truth_used_by_policy"] is False
    assert len(payload["prediction_checkpoint_sha256"]) == 64
    assert len(payload["trajectory_checkpoint_sha256"]) == 64
    assert PREDICTION_LAYOUT.dimension == 1368


def test_nc_delay_gate_uses_packet_age_bins() -> None:
    observations = {
        agent: {"communication": np.zeros(4, np.float32)}
        for agent in PLATOON_IDS
    }
    ages = (0.0, 0.04, 0.09, 0.11, 0.16)
    for agent, age in zip(PLATOON_IDS[1:], ages):
        observations[agent]["communication"] = np.asarray(
            [0.0, age, 0.0, age], np.float32
        )
    gates = NCKnowledgeStateAdapter._delay_modes(observations).reshape(6, 3)
    assert np.array_equal(gates.argmax(-1), np.asarray([0, 0, 0, 1, 1, 2]))
    assert np.allclose(gates.sum(-1), 1.0)


def test_delay_switching_actor_selects_per_vehicle_expert() -> None:
    actor = DelaySwitchingActor(state_dim=24)
    with torch.no_grad():
        for mode, expert in enumerate(actor.experts):
            for parameter in expert.parameters():
                parameter.zero_()
            expert.network[-1].bias[:6].fill_(float(mode + 1))
    state = torch.zeros(1, 24)
    modes = torch.tensor([0, 1, 2, 0, 1, 2])
    state[0, -18:] = torch.nn.functional.one_hot(modes, 3).float().reshape(-1)
    distribution = actor.distribution(state)
    assert torch.allclose(
        distribution.mean,
        torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]]),
    )
    terminal = actor.distribution(torch.zeros(1, 24))
    assert torch.allclose(terminal.mean, torch.ones(1, 6))
    assert torch.all(terminal.scale > 0.0)

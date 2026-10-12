from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from platoon_control.agents import FollowerStateBuilder, PlatoonController
from platoon_control.config import FollowerRLConfig, PlannerConfig
from platoon_control.models import (
    CPIDLMDelayCompensator,
    DreamerV3Leader,
    FOLLOWER_POLICY_STATE_DIM,
    FOLLOWER_PREDICTED_ACCELERATION_INDEX,
    MBRLStructuredAdapter,
    MultiAgentFollowerSAC,
    STGNNTrajectoryPredictor,
    StructuredEnsembleDynamics,
)
from platoon_control.models.dynamics import FOLLOWER_STATE_DIM
from platoon_control.planning import CEMPlanner
from platoon_control.rl import (
    CURRICULUM_CRITERIA,
    CURRICULUM_MIN_STAGE_STEPS,
    CURRICULUM_NAMES,
    CURRICULUM_WINDOW,
    EpisodeReplay,
    FOLLOWER_ACTOR_UPDATES_PER_EPISODE,
    LEADER_KEYS,
    RLConfig,
    _append_transition,
    _apply_correlated_follower_exploration,
    _behavior_cloning_weight,
    _candidate_validation_passes,
    _curriculum_options,
    _follower_actor_curriculum_enabled,
    _new_episode,
    _next_candidate_origin,
    _stage_capability_passes,
)
from platoon_sumo.env import FOLLOWER_IDS, PLATOON_IDS


def test_delay_and_trajectory_models_have_trainable_probabilistic_outputs() -> None:
    delay = CPIDLMDelayCompensator(hidden_dim=32)
    prediction = delay(torch.randn(2, 50, 8), torch.randn(2, 6))
    losses = delay.loss(prediction, torch.randn(2, 3), torch.randn(2, 30, 3))
    losses["loss"].backward()
    assert prediction.future_mean.shape == (2, 30, 3)
    assert delay.mixer.adjacency_logits.grad is not None

    predictor = STGNNTrajectoryPredictor(hidden_dim=32)
    mask = torch.ones(2, 12)
    output = predictor(torch.randn(2, 12, 20, 8), mask)
    loss = predictor.loss(
        output, torch.randn(2, 12, 30, 4), mask, torch.zeros(2, 12)
    )["loss"]
    loss.backward()
    assert output.trajectories.shape == (2, 12, 3, 30, 4)
    assert output.cutin_logits.shape == (2, 12)


def _leader_observation(batch: int) -> dict[str, torch.Tensor]:
    return {
        "ego": torch.randn(batch, 8),
        "platoon_state": torch.randn(batch, 6, 6),
        "front_history": torch.randn(batch, 8, 50, 6),
        "front_mask": torch.ones(batch, 8),
        "v2i_flow": torch.randn(batch, 8, 4),
        "road_profile": torch.randn(batch, 20, 3),
        "signal": torch.randn(batch, 4),
        "signal_mask": torch.ones(batch, 1),
    }


def test_dreamer_inference_and_imagination_actor_critic() -> None:
    model = DreamerV3Leader(hidden_dim=32, stochastic_dim=4, stochastic_classes=4)
    scaled = model.scale_action(torch.tensor([[-1.0], [0.0], [1.0]]))
    assert torch.allclose(scaled[:, 0], torch.tensor([-3.0, 0.0, 1.2]))
    observation = _leader_observation(2)
    state = model.initial(2, "cpu")
    action, state, front = model.infer(observation, state, torch.zeros(2, 1))
    assert action.shape == (2, 1)
    assert torch.all(action >= -3.0) and torch.all(action <= 1.2)
    assert front.trajectories.shape == (2, 8, 3, 30, 2)
    imagination = model.imagination_loss(state, horizon=3)
    imagination["loss"].backward()
    assert model.actor[0].weight.grad is not None


def test_dreamer_latent_reconstructs_road_and_v2i_profiles() -> None:
    model = DreamerV3Leader(hidden_dim=32, stochastic_dim=4, stochastic_classes=4)
    batch, time = 2, 3
    flat = _leader_observation(batch * time)
    observations = {
        key: value.reshape(batch, time, *value.shape[1:])
        for key, value in flat.items()
    }
    losses = model.world_model_loss(
        observations,
        torch.zeros(batch, time, 1),
        torch.zeros(batch, time),
        torch.ones(batch, time),
    )
    losses["loss"].backward()
    assert losses["road_profile_reconstruction"] >= 0.0
    assert losses["v2i_plan_reconstruction"] >= 0.0
    assert model.road_profile_decoder[0].weight.grad is not None
    assert model.v2i_plan_decoder[0].weight.grad is not None


def test_front_predictor_does_not_use_absolute_ego_station() -> None:
    model = DreamerV3Leader(hidden_dim=32, stochastic_dim=4, stochastic_classes=4)
    history = torch.randn(2, 8, 50, 6)
    history[1] = history[0]
    ego = torch.zeros(2, 8)
    ego[:, 1] = 20.0
    ego[1, 0] = 3000.0
    prediction = model.front_predictor(history, ego)
    assert torch.allclose(
        prediction.trajectories[0], prediction.trajectories[1], atol=1e-6
    )


def test_structured_dynamics_and_batched_cem_respect_action_limits() -> None:
    dynamics = StructuredEnsembleDynamics(hidden_dim=32, ensemble_size=3)
    adapter = MBRLStructuredAdapter(dynamics, "cpu")
    state = torch.zeros(5, FOLLOWER_STATE_DIM)
    state[:, 1] = 25.0
    state[:, 3] = 20.0
    state[:, 4] = 25.0
    state[:, 6] = 20.0
    state[:, 8] = 25.0
    planner = CEMPlanner(
        adapter,
        PlannerConfig(horizon=3, population=16, elites=4, iterations=2),
    )
    result = planner.plan(state)
    assert result.action.shape == (5, 1)
    assert torch.all(result.action >= -3.0)
    assert torch.all(result.action <= 1.2)


def test_shared_longitudinal_actor_and_centralized_critics_train_without_mpc() -> None:
    config = FollowerRLConfig(hidden_dim=32)
    policy = MultiAgentFollowerSAC(config)
    states = torch.zeros(4, 5, 10, FOLLOWER_POLICY_STATE_DIM)
    states[..., 1] = 25.0
    states[..., 3] = 12.0
    states[..., 6] = 12.0
    next_states = states.clone()
    sample = policy.act(states, deterministic=False)

    assert sample.acceleration.shape == (4, 5, 1)
    assert sample.critic_action.shape == (4, 5, 1)
    assert torch.all(sample.acceleration >= -3.0)
    assert torch.all(sample.acceleration <= 1.2)

    gated_policy = MultiAgentFollowerSAC(config)
    equilibrium = states.clone()
    equilibrium[..., 6] = 12.0
    equilibrium[..., 17] = 12.0
    equilibrium[..., 5] = -0.4
    equilibrium[..., 23] = -0.4
    equilibrium[..., 40] = 0.0
    with torch.no_grad():
        gated_policy.actor.follower_mean.bias.fill_(2.0)
    equilibrium_action = gated_policy.act(
        equilibrium, deterministic=True
    ).acceleration
    assert torch.allclose(
        equilibrium_action,
        torch.full_like(equilibrium_action, -0.4),
        atol=1e-5,
    )

    disturbed = equilibrium.clone()
    disturbed[..., 17] = 20.0
    disturbed_action = gated_policy.act(
        disturbed, deterministic=True
    ).acceleration
    assert torch.all(disturbed_action > equilibrium_action)

    predicted = states.clone()
    predicted[..., 23] = -0.7
    feed_forward = policy.act(predicted, deterministic=True)
    assert torch.allclose(
        feed_forward.acceleration,
        torch.full_like(feed_forward.acceleration, -0.7),
        atol=1e-5,
    )

    predicted[..., 16] = 1.0
    predicted[..., FOLLOWER_PREDICTED_ACCELERATION_INDEX] = 0.4
    recovery_feed_forward = policy.act(predicted, deterministic=True)
    assert torch.allclose(
        recovery_feed_forward.acceleration,
        torch.full_like(recovery_feed_forward.acceleration, -0.7),
        atol=1e-5,
    )

    critic_loss, critic_metrics = policy.critic_loss(
        states,
        sample.critic_action.detach(),
        torch.ones(4, len(FOLLOWER_IDS)),
        next_states,
        torch.ones(4),
    )
    critic_loss.backward()
    assert policy.critic1.network[0].weight.grad is not None
    assert torch.isfinite(critic_metrics["conservative"])
    assert critic_metrics["conservative"] >= 0.0

    policy.zero_grad(set_to_none=True)
    deterministic_behavior = policy.actor.sample(
        states, deterministic=True
    ).acceleration.detach()
    actor_loss, actor_metrics = policy.actor_loss(
        states,
        deterministic_behavior,
        requested_behavior_actions=deterministic_behavior,
    )
    actor_loss.backward()
    assert policy.actor.acceleration_mean.weight.grad is not None
    assert actor_metrics["behavior"] == 0.0
    assert torch.isfinite(actor_metrics["policy_q_symlog"])

    shielded_actions = sample.critic_action.detach() - 1.0
    _, shielded_metrics = policy.actor_loss(
        states,
        shielded_actions,
        requested_behavior_actions=sample.critic_action.detach(),
    )
    assert shielded_metrics["behavior"] == 0.0
    assert shielded_metrics["behavior_shield_fraction"] == 1.0


def test_controller_exports_cpidlm_reward_predictions() -> None:
    states = torch.zeros(len(FOLLOWER_IDS), 10, FOLLOWER_POLICY_STATE_DIM)
    states[:, -1, 35] = torch.arange(len(FOLLOWER_IDS)) + 20.0
    states[:, -1, 36] = -1.0
    predictions = PlatoonController.follower_control_predictions(states)
    assert predictions["follower_1"] == pytest.approx((20.0, -1.0))
    assert predictions["follower_5"] == pytest.approx((24.0, -1.0))


def test_follower_training_exploration_is_correlated() -> None:
    actions = {
        "leader": {"accel": torch.zeros(1).numpy(), "lane_change": 0},
        **{
            follower: {
                "accel": torch.zeros(1).numpy(),
                "lane_change": 0,
            }
            for follower in FOLLOWER_IDS
        },
    }
    _apply_correlated_follower_exploration(
        actions, np.random.default_rng(4), independent_std=0.0
    )
    accelerations = [
        float(actions[vehicle_id]["accel"][0]) for vehicle_id in PLATOON_IDS
    ]
    assert max(accelerations) - min(accelerations) == pytest.approx(0.0)


def test_independent_exploration_probe_has_no_velocity_or_position_drift() -> None:
    velocity = 0.0
    position = 0.0
    accelerations: list[float] = []
    for step in range(40):
        actions = {
            "leader": {"accel": np.zeros(1, np.float32), "lane_change": 0},
            **{
                follower: {
                    "accel": np.zeros(1, np.float32),
                    "lane_change": 0,
                }
                for follower in FOLLOWER_IDS
            },
        }
        _apply_correlated_follower_exploration(
            actions,
            np.random.default_rng(4),
            common_std=0.0,
            independent_std=0.04,
            episode_step=step,
        )
        acceleration = float(actions["follower_1"]["accel"][0])
        accelerations.append(acceleration)
        velocity += 0.1 * acceleration
        position += 0.1 * velocity

    assert max(abs(value) for value in accelerations) <= 0.04001
    assert velocity == pytest.approx(0.0, abs=1e-7)
    assert position == pytest.approx(0.0, abs=1e-7)


def test_terminal_collision_reward_is_available_to_follower_replay() -> None:
    episode = _new_episode()
    observations = {
        "leader": {key: torch.zeros(1).numpy() for key in LEADER_KEYS}
    }
    actions = {
        "leader": {"accel": torch.zeros(1).numpy(), "lane_change": 0},
        **{
            follower: {
                "accel": torch.zeros(1).numpy(),
                "lane_change": 0,
            }
            for follower in FOLLOWER_IDS
        },
    }
    states = torch.zeros(5, FOLLOWER_POLICY_STATE_DIM).numpy()
    _append_transition(
        episode,
        observations,
        actions,
        leader_reward=-100.0,
        follower_reward=-100.0,
        continued=False,
        states=states,
        next_states=states.copy(),
    )
    replay = EpisodeReplay(capacity_steps=10, seed=7)
    replay.add(episode)
    (
        _,
        sampled_actions,
        sampled_rewards,
        _,
        sampled_continues,
        sampled_behavior,
        sampled_requested,
    ) = (
        replay.sample_followers(4)
    )
    assert sampled_actions.shape == (4, 5, 1)
    assert sampled_rewards.shape == (4, len(FOLLOWER_IDS))
    assert torch.all(sampled_rewards == -100.0)
    assert torch.all(sampled_continues == 0.0)
    assert torch.all(sampled_behavior == sampled_actions)
    assert torch.all(sampled_requested == sampled_actions)


def test_replay_separates_requested_and_applied_acceleration() -> None:
    episode = _new_episode()
    observations = {
        "leader": {key: torch.zeros(1).numpy() for key in LEADER_KEYS}
    }
    actions = {
        "leader": {"accel": torch.tensor([1.0]).numpy(), "lane_change": 0},
        **{
            follower: {
                "accel": torch.tensor([0.8]).numpy(),
                "lane_change": 0,
            }
            for follower in FOLLOWER_IDS
        },
    }
    infos = {
        "leader": {"applied_acceleration": -0.6},
        **{
            follower: {"applied_acceleration": -1.0 - 0.1 * index}
            for index, follower in enumerate(FOLLOWER_IDS)
        },
    }
    states = torch.zeros(5, 10, FOLLOWER_POLICY_STATE_DIM).numpy()
    _append_transition(
        episode,
        observations,
        actions,
        leader_reward=0.0,
        follower_reward=0.0,
        continued=True,
        states=states,
        next_states=states.copy(),
        action_infos=infos,
    )
    assert episode["leader_actions"][0][0] == pytest.approx(-0.6)
    assert episode["follower_actions"][0][:, 0].tolist() == pytest.approx(
        [-1.0, -1.1, -1.2, -1.3, -1.4]
    )
    assert episode["follower_behavior_actions"][0][:, 0].tolist() == pytest.approx(
        [-1.0, -1.1, -1.2, -1.3, -1.4]
    )
    assert episode["follower_requested_actions"][0][:, 0].tolist() == pytest.approx(
        [0.8, 0.8, 0.8, 0.8, 0.8]
    )


def test_behavior_cloning_weight_decays_toward_floor() -> None:
    config = FollowerRLConfig()
    assert _behavior_cloning_weight(config, 8_000, 8_000, 40) == pytest.approx(10.0)
    middle = _behavior_cloning_weight(config, 33_000, 8_000, 40)
    late = _behavior_cloning_weight(config, 58_000, 8_000, 40)
    assert 0.1 < late < middle < 10.0


def test_capability_curriculum_unlocks_one_major_disturbance_at_a_time() -> None:
    config = RLConfig()
    options = [
        _curriculum_options(
            stage, np.random.default_rng(100 + stage), config
        )[0]
        for stage in range(len(CURRICULUM_NAMES))
    ]
    assert options[0]["front_vehicle_count"] == 0
    assert not options[0]["v2i_enabled"]
    assert options[0]["speed_reference"] == 25.0
    assert options[1]["front_vehicle_count"] <= 2
    assert not options[1]["v2i_enabled"]
    assert options[2]["front_vehicle_count"] >= 1
    assert options[2]["wave_mode"] in ("off", "learned")
    assert options[3]["v2i_enabled"]
    assert options[4]["background_initial_scale"] > 0.0
    assert not options[4]["cutin_enabled"]
    assert options[5]["cutin_enabled"]
    assert options[5]["communication_delay_probability"] == 0.0
    assert options[6]["communication_delay_probability"] == 0.05
    assert options[6]["communication_delay_max"] == 0.1
    assert options[7]["communication_delay_probability"] == 0.15
    assert options[7]["communication_delay_max"] == 0.2


def test_follower_actor_adapts_from_wave_stage_onward() -> None:
    assert _follower_actor_curriculum_enabled(0)
    assert not _follower_actor_curriculum_enabled(1)
    assert all(
        _follower_actor_curriculum_enabled(stage)
        for stage in range(2, len(CURRICULUM_NAMES))
    )
    assert FOLLOWER_ACTOR_UPDATES_PER_EPISODE[2] >= 16
    assert FOLLOWER_ACTOR_UPDATES_PER_EPISODE[3] >= 8


def test_curriculum_gate_requires_all_capability_metrics() -> None:
    passing = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 8.0,
        "max_gap": 13.0,
        "recovery": 0.0,
        "shield": 0.0,
        "string": 1.0,
        "v2i_exposure": 1.0,
        "v2i_score": 0.95,
        "curve_overspeed": 0.0,
        "curve_wasted_acceleration": 0.0,
        "leader_jerk": 0.0,
        "background_exposure": 1.0,
        "cutin_exposure": 1.0,
        "cutin_resolution": 1.0,
        "delay_exposure": 1.0,
        "observed_delay": 0.15,
    }
    records = [passing.copy() for _ in range(8)]
    assert not _stage_capability_passes(0, 39_999, records)[0]
    assert _stage_capability_passes(0, 40_000, records)[0]
    records[-1]["min_gap"] = 4.0
    assert not _stage_capability_passes(0, 40_000, records)[0]


def test_paired_candidate_guard_accepts_only_safe_material_improvement() -> None:
    origin = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 11.0,
        "max_gap": 90.0,
        "recovery": 0.19,
        "shield": 0.03,
        "string": 1.10,
    }
    improved = dict(origin, max_gap=89.4, recovery=0.18)
    assert _candidate_validation_passes(2, improved, origin)
    assert _candidate_validation_passes(
        2,
        dict(improved, max_gap=25.5, recovery=0.46),
        dict(origin, max_gap=93.6),
    )
    assert not _candidate_validation_passes(
        2, dict(improved, max_gap=89.8), origin
    )
    assert not _candidate_validation_passes(
        2, dict(improved, collision=1.0), origin
    )
    assert not _candidate_validation_passes(
        2, dict(improved, string=2.0), origin
    )
    assert not _candidate_validation_passes(
        2, dict(improved, max_gap=35.0, recovery=0.46), origin
    )
    assert not _candidate_validation_passes(
        2, dict(improved, max_gap=25.5, recovery=0.70), origin
    )


def test_paired_candidate_guard_accepts_safe_gradual_v2i_improvement() -> None:
    origin = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 11.9,
        "max_gap": 17.49,
        "recovery": 0.041,
        "shield": 0.066,
        "string": 1.11,
        "v2i_exposure": 1.0,
        "v2i_score": 0.723,
        "curve_overspeed": 0.0,
        "curve_wasted_acceleration": 0.0,
        "leader_jerk": 0.002,
        "background_exposure": 0.0,
        "cutin_exposure": 0.0,
        "cutin_resolution": 0.0,
        "delay_exposure": 0.0,
        "observed_delay": 0.0,
    }
    improved = dict(origin, max_gap=17.06, v2i_score=0.742)
    assert _candidate_validation_passes(3, improved, origin)
    assert not _candidate_validation_passes(
        3, dict(improved, curve_overspeed=0.20), origin
    )
    assert not _candidate_validation_passes(
        3, dict(improved, max_gap=17.80), origin
    )


def test_rejected_candidate_does_not_lower_the_next_comparison_baseline() -> None:
    accepted = {"max_gap": 89.6, "shield": 0.03}
    rejected = {"max_gap": 91.7, "shield": 0.02}
    validation = {"origin_capability": accepted}
    assert _next_candidate_origin(rejected, validation, True) == accepted
    assert _next_candidate_origin(rejected, validation, False) == rejected


def test_every_curriculum_stage_has_a_completion_gate() -> None:
    assert len(CURRICULUM_NAMES) == len(CURRICULUM_CRITERIA)
    assert len(CURRICULUM_NAMES) == len(CURRICULUM_MIN_STAGE_STEPS)
    passing = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 8.0,
        "max_gap": 13.0,
        "recovery": 0.0,
        "shield": 0.0,
        "string": 1.0,
        "v2i_exposure": 1.0,
        "v2i_score": 0.95,
        "curve_overspeed": 0.0,
        "curve_wasted_acceleration": 0.0,
        "leader_jerk": 0.0,
        "background_exposure": 1.0,
        "cutin_exposure": 1.0,
        "cutin_resolution": 1.0,
        "delay_exposure": 1.0,
        "observed_delay": 0.15,
    }
    records = [passing.copy() for _ in range(CURRICULUM_WINDOW)]
    last_stage = len(CURRICULUM_NAMES) - 1
    assert _stage_capability_passes(
        last_stage,
        CURRICULUM_MIN_STAGE_STEPS[last_stage],
        records,
    )[0]


def test_later_curriculum_gates_require_real_behavior_exposure() -> None:
    passing = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 8.0,
        "max_gap": 13.0,
        "recovery": 0.0,
        "shield": 0.0,
        "string": 1.0,
        "v2i_exposure": 1.0,
        "v2i_score": 0.95,
        "curve_overspeed": 0.0,
        "curve_wasted_acceleration": 0.0,
        "leader_jerk": 0.0,
        "background_exposure": 1.0,
        "cutin_exposure": 1.0,
        "cutin_resolution": 1.0,
        "delay_exposure": 1.0,
        "observed_delay": 0.10,
    }
    records = [passing.copy() for _ in range(CURRICULUM_WINDOW)]
    assert _stage_capability_passes(
        6, CURRICULUM_MIN_STAGE_STEPS[6], records
    )[0]
    for missing_metric in (
        "v2i_exposure",
        "background_exposure",
        "cutin_exposure",
        "cutin_resolution",
        "delay_exposure",
    ):
        degraded = [row.copy() for row in records]
        for row in degraded:
            row[missing_metric] = 0.0
        assert not _stage_capability_passes(
            6, CURRICULUM_MIN_STAGE_STEPS[6], degraded
        )[0]

    out_of_range = [row.copy() for row in records]
    for row in out_of_range:
        row["observed_delay"] = 0.20
    assert not _stage_capability_passes(
        6, CURRICULUM_MIN_STAGE_STEPS[6], out_of_range
    )[0]


def test_follower_stability_loss_rejects_gap_worsening_residual() -> None:
    policy = MultiAgentFollowerSAC(FollowerRLConfig())
    states = torch.zeros(4, len(FOLLOWER_IDS), 10, FOLLOWER_POLICY_STATE_DIM)
    states[..., -1, 6] = 12.0
    states[..., -1, 17] = 20.0
    with torch.no_grad():
        policy.actor.follower_mean.bias.fill_(-0.2)
    _, worsening = policy.actor_loss(states, stability_weight=20.0)
    with torch.no_grad():
        policy.actor.follower_mean.bias.fill_(0.2)
    _, restoring = policy.actor_loss(states, stability_weight=20.0)
    assert worsening["stability_direction"] > 0.0
    assert restoring["stability_direction"] == 0.0
    assert worsening["stability"] > restoring["stability"]


def test_follower_actor_tracks_predecessor_recovery_before_gap_opens() -> None:
    policy = MultiAgentFollowerSAC(FollowerRLConfig())
    states = torch.zeros(4, len(FOLLOWER_IDS), 10, FOLLOWER_POLICY_STATE_DIM)
    states[..., -1, 5] = 0.5
    states[..., -1, 6] = 12.0
    states[..., -1, 17] = 12.0
    states[..., -1, 23] = 0.0
    states[..., -1, 40] = 0.0

    with torch.no_grad():
        policy.actor.follower_mean.bias.fill_(-0.5)
    opposing_action = policy.act(states, deterministic=True).acceleration
    _, opposing = policy.actor_loss(states, stability_weight=20.0)
    with torch.no_grad():
        policy.actor.follower_mean.bias.fill_(0.5)
    following_action = policy.act(states, deterministic=True).acceleration
    _, following = policy.actor_loss(states, stability_weight=20.0)

    assert torch.all(following_action > opposing_action)
    assert following["stability_equilibrium"] < opposing["stability_equilibrium"]


def test_follower_state_temporarily_reduces_gap_for_predicted_cutin() -> None:
    batch = 2
    neighbors = 12
    observations = {
        "ego": torch.zeros(batch, 8),
        "local_predecessor": torch.zeros(batch, 6),
        "local_front": torch.zeros(batch, 6),
        "platoon_state": torch.zeros(batch, 6, 6),
        "road_profile": torch.zeros(batch, 20, 3),
        "communication": torch.zeros(batch, 4),
        "neighbor_mask": torch.ones(batch, neighbors),
        "recovery": torch.zeros(batch, 6),
        "leader_plan": torch.zeros(batch, 11),
        "front_history": torch.zeros(batch, 8, 50, 6),
        "front_mask": torch.zeros(batch, 8),
        "v2i_flow": torch.zeros(batch, 8, 4),
    }
    cutin_probability = torch.zeros(batch, neighbors)
    cutin_probability[0, 0] = 1.0
    observations["recovery"][0, 0] = 1.0
    observations["recovery"][0, 1] = 50.0
    observations["local_predecessor"][:, 2] = 1.25
    trajectories = torch.zeros(batch, neighbors, 1, 30, 4)
    trajectories[:, :, :, 0] = 50.0
    states = FollowerStateBuilder()(
        observations,
        torch.full((batch, 3), -4.0),
        cutin_probability,
        trajectories,
    )
    assert states[0, 6] == 8.0
    assert states[1, 6] == 12.0
    assert torch.all(states[:, 5] == 1.25)
    assert states[0, 16] == 1.0
    assert states[0, 17] == 50.0

from dataclasses import replace

import numpy as np
import torch

from classical_comparison.algorithms import create_algorithm
from classical_comparison.assisted_train import (
    AssistedConfig,
    FINAL_UNASSISTED_STAGE,
    _assistance_coefficient,
    _apply_plateau_promotion,
    _configure_algorithm,
    _prepare_resumed_algorithm,
    _set_optimizer_learning_rates,
    _set_stage_assistance,
    _set_stage_optimization,
)
from classical_comparison.common import (
    ActionCodec,
    ComparisonProtocol,
    RunningNormalizer,
    reward_fingerprint,
    reward_fingerprint_is_compatible,
)
from classical_comparison.fixed_horizon import (
    LeaderProfile,
    LeaderTracker,
    _canonical_hash,
    load_profiles,
)
from classical_comparison.models import SquashedGaussianActor


def test_comparison_protocol_freezes_stage_and_reward_source() -> None:
    protocol = ComparisonProtocol()
    assert protocol.stage(0) == 0
    assert protocol.stage(223_116) == 6
    assert protocol.stage(223_117) == 7
    assert protocol.to_dict()["controller_specific_reward_tuning"] is False
    assert protocol.to_dict()["reward_fingerprint"] == reward_fingerprint()


def test_reviewed_legacy_reward_fingerprint_remains_compatible() -> None:
    assert reward_fingerprint_is_compatible(
        "40489a82ad010cb90b8937e7fd518b64e7880290b7a448976e4c0caab90c4c37"
    )
    assert not reward_fingerprint_is_compatible("unreviewed")


def test_zero_centered_action_codec_round_trip() -> None:
    codec = ActionCodec()
    normalized = np.asarray([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], np.float32)
    physical = codec.to_physical(normalized)
    np.testing.assert_allclose(
        physical, [-3.0, -1.5, 0.0, 0.3, 0.6, 1.2], atol=1e-6
    )
    np.testing.assert_allclose(codec.to_normalized(physical), normalized)


def test_running_normalizer_round_trip_state() -> None:
    normalizer = RunningNormalizer(3)
    normalizer.update(np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]]))
    restored = RunningNormalizer(3)
    restored.load_state_dict(normalizer.state_dict())
    sample = np.asarray([1.5, 3.0, 5.5], np.float32)
    np.testing.assert_allclose(restored.normalize(sample), normalizer.normalize(sample))


def test_classical_algorithms_emit_finite_bounded_actions() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=32, batch_size=4)
    state = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    for name in ("ddpg", "ppo", "sac"):
        algorithm = create_algorithm(name, state.size, protocol, "cpu", 43)
        sample = algorithm.act(state, deterministic=True, total_steps=0)
        assert sample.action.shape == (6,)
        assert np.isfinite(sample.action).all()
        assert np.max(np.abs(sample.action)) <= 1.0


def test_squashed_actor_log_prob_is_consistent_at_action_boundary() -> None:
    actor = SquashedGaussianActor(4, 6, state_dependent_std=False)
    with torch.no_grad():
        for parameter in actor.network.parameters():
            parameter.zero_()
        actor.network[-1].bias.fill_(20.0)
        assert actor.log_std is not None
        actor.log_std.fill_(-2.0)
    state = torch.zeros((3, 4))
    action, sampled_log_prob, _ = actor.sample(state)
    recomputed_log_prob, _ = actor.log_prob(state, action)
    assert torch.isfinite(sampled_log_prob).all()
    assert torch.max(action) < 1.0
    assert torch.isfinite(recomputed_log_prob).all()
    assert torch.min(sampled_log_prob) > -100.0


def test_ddpg_actor_stays_frozen_during_critic_warmup() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("ddpg", 8, protocol, "cpu", 43)
    algorithm.learning_starts = 0
    algorithm.actor_learning_starts = 100
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=True, total_steps=0)
        algorithm.record(state, sample, 0.1, state, False)
    before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    metrics = algorithm.update_if_ready(total_steps=10)
    assert metrics["train/actor_updated"] == 0.0
    for old, current in zip(before, algorithm.actor.parameters()):
        torch.testing.assert_close(old, current)


def test_ppo_actor_stays_frozen_during_value_warmup() -> None:
    protocol = replace(ComparisonProtocol(), batch_size=4)
    algorithm = create_algorithm("ppo", 8, protocol, "cpu", 43)
    algorithm.rollout_size = 4
    algorithm.epochs = 1
    algorithm.actor_learning_starts = 100
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=False, total_steps=0)
        algorithm.record(state, sample, 0.1, state, False)
    before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    metrics = algorithm.update_if_ready(total_steps=10)
    assert metrics["train/actor_updated"] == 0.0
    for old, current in zip(before, algorithm.actor.parameters()):
        torch.testing.assert_close(old, current)


def test_ppo_stores_log_prob_of_the_bounded_environment_action() -> None:
    protocol = replace(ComparisonProtocol(), batch_size=4)
    algorithm = create_algorithm("ppo", 8, protocol, "cpu", 43)
    with torch.no_grad():
        for parameter in algorithm.actor.network.parameters():
            parameter.zero_()
        algorithm.actor.network[-1].bias.fill_(20.0)
    state = np.zeros(8, np.float32)
    sample = algorithm.act(state, deterministic=False, total_steps=0)
    normalized = torch.as_tensor(
        algorithm.normalizer.normalize(state), dtype=torch.float32
    ).unsqueeze(0)
    action = torch.as_tensor(sample.action).unsqueeze(0)
    expected, _ = algorithm.actor.log_prob(normalized, action)
    assert np.isfinite(sample.auxiliary["log_prob"])
    assert float(algorithm.actor.distribution(normalized).loc.max()) == 5.0
    assert float(action.max()) < 1.0 - 1e-6
    np.testing.assert_allclose(
        sample.auxiliary["log_prob"], float(expected.squeeze()), rtol=1e-5
    )


def test_ppo_anchor_is_serialized_and_used_after_warmup() -> None:
    protocol = replace(ComparisonProtocol(), batch_size=4)
    algorithm = create_algorithm("ppo", 8, protocol, "cpu", 43)
    algorithm.rollout_size = 4
    algorithm.epochs = 1
    algorithm.actor_learning_starts = 0
    algorithm.anchor_coefficient = 2.0
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=False, total_steps=0)
        algorithm.record(state, sample, 0.1, state, False)
    metrics = algorithm.update_if_ready(total_steps=10)
    assert metrics["train/actor_updated"] == 1.0
    assert metrics["train/anchor_coefficient"] > 0.0
    assert metrics["loss/anchor"] >= 0.0
    restored = create_algorithm("ppo", 8, protocol, "cpu", 44)
    restored.load_state_dict(algorithm.state_dict())
    for expected, actual in zip(
        algorithm.anchor_actor.parameters(), restored.anchor_actor.parameters()
    ):
        torch.testing.assert_close(expected, actual)


def test_ppo_kl_stop_on_first_batch_keeps_metrics_finite() -> None:
    protocol = replace(ComparisonProtocol(), batch_size=4)
    algorithm = create_algorithm("ppo", 8, protocol, "cpu", 43)
    algorithm.rollout_size = 4
    algorithm.epochs = 1
    algorithm.actor_learning_starts = 0
    algorithm.target_kl = 1e-8
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=False, total_steps=0)
        algorithm.record(state, sample, 0.1, state, False)
    with torch.no_grad():
        algorithm.actor.network[-1].bias[:6].add_(2.0)
    metrics = algorithm.update_if_ready(total_steps=10)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["train/actor_updated"] == 0.0
    assert metrics["train/kl_stopped"] == 1.0


def test_ppo_resume_learning_rate_matches_post_dagger_rate() -> None:
    protocol = replace(ComparisonProtocol(), batch_size=4)
    algorithm = create_algorithm("ppo", 8, protocol, "cpu", 43)
    _configure_algorithm(algorithm)
    assert algorithm.optimizer.param_groups[0]["lr"] == 1e-5
    algorithm.optimizer.param_groups[0]["lr"] = 3e-5
    _set_optimizer_learning_rates(algorithm)
    assert algorithm.optimizer.param_groups[0]["lr"] == 1e-5


def test_sac_entropy_temperature_is_capped() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("sac", 8, protocol, "cpu", 43)
    algorithm.learning_starts = 0
    algorithm.actor_learning_starts = 0
    algorithm.maximum_alpha = 0.2
    algorithm.log_alpha.data.fill_(0.0)
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=False, total_steps=0)
        algorithm.record(state, sample, 0.1, state, False)
    algorithm.update_if_ready(total_steps=10)
    assert float(algorithm.log_alpha.exp()) <= 0.2 + 1e-6


def test_sac_behavior_noise_scale_can_recover_deterministic_action() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("sac", 8, protocol, "cpu", 43)
    algorithm.behavior_noise_scale = 0.0
    state = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    sampled = algorithm.act(state, deterministic=False, total_steps=0).action
    deterministic = algorithm.act(state, deterministic=True, total_steps=0).action
    np.testing.assert_allclose(sampled, deterministic, atol=1e-6)


def test_sac_uses_same_scaled_policy_distribution_for_training() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("sac", 8, protocol, "cpu", 43)
    algorithm.behavior_noise_scale = 0.45
    states = torch.zeros((4, 8))
    base = algorithm.actor.distribution(states)
    policy = algorithm._policy_distribution(states)
    torch.testing.assert_close(policy.loc, base.loc)
    torch.testing.assert_close(policy.scale, base.scale * 0.45)
    action, log_prob, deterministic = algorithm._sample_policy(states)
    recomputed = algorithm.actor._squashed_log_prob(
        policy, torch.atanh(action), action
    )
    assert action.shape == deterministic.shape == (4, 6)
    assert log_prob.shape == (4, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()
    torch.testing.assert_close(log_prob, recomputed)


def test_sac_critic_target_is_bounded_under_ood_bootstrap() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("sac", 8, protocol, "cpu", 43)
    algorithm.learning_starts = 0
    algorithm.actor_learning_starts = 100
    with torch.no_grad():
        for target in (algorithm.target1, algorithm.target2):
            for parameter in target.parameters():
                parameter.zero_()
            target.network[-1].bias.fill_(1e9)
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=False, total_steps=0)
        algorithm.record(state, sample, 0.1, state, False)
    metrics = algorithm.update_if_ready(total_steps=10)
    assert np.isfinite(metrics["loss/critic"])
    assert metrics["train/target_abs_max"] <= max(
        abs(algorithm.critic_target_min), algorithm.critic_target_max
    )
    assert metrics["train/target_clipped_fraction"] == 1.0


def test_sac_conservative_penalty_and_actor_q_normalization_are_finite() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("sac", 8, protocol, "cpu", 43)
    algorithm.learning_starts = 0
    algorithm.actor_learning_starts = 0
    algorithm.conservative_weight = 0.1
    algorithm.conservative_samples = 2
    algorithm.normalize_actor_q = True
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=False, total_steps=0)
        algorithm.record(state, sample, -0.5, state, False)
    metrics = algorithm.update_if_ready(total_steps=10)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["loss/conservative"] > 0.0
    assert metrics["train/policy_q_scale"] >= 1.0


def test_resume_reapplies_sac_optimizer_rates_without_resetting_alpha() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("sac", 8, protocol, "cpu", 43)
    _configure_algorithm(algorithm)
    algorithm.log_alpha.data.fill_(-2.0)
    for group in algorithm.critic_optimizer.param_groups:
        group["lr"] = 3e-4
    _set_optimizer_learning_rates(algorithm)
    assert algorithm.critic_optimizer.param_groups[0]["lr"] == 5e-5
    torch.testing.assert_close(algorithm.log_alpha, torch.tensor(-2.0))


def test_sac_resume_can_reset_value_state_and_warm_up_actor() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("sac", 8, protocol, "cpu", 43)
    _configure_algorithm(algorithm)
    state = np.zeros(8, np.float32)
    sample = algorithm.act(state, deterministic=True, total_steps=0)
    algorithm.record(state, sample, 0.1, state, False)
    actor_before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in algorithm.critic1.parameters()]
    config = replace(
        AssistedConfig(),
        reset_sac_value_on_resume=True,
        resume_actor_warmup_steps=10_000,
    )
    _prepare_resumed_algorithm(algorithm, total_steps=87_223, config=config)
    assert algorithm.replay.size == 0
    assert algorithm.actor_learning_starts == 97_223
    assert any(
        not torch.equal(old, current)
        for old, current in zip(critic_before, algorithm.critic1.parameters())
    )
    for old, current in zip(actor_before, algorithm.actor.parameters()):
        torch.testing.assert_close(old, current)
    for source, target in zip(
        algorithm.critic1.parameters(), algorithm.target1.parameters()
    ):
        torch.testing.assert_close(source, target)


def test_final_stage_removes_all_cacc_policy_assistance() -> None:
    config = AssistedConfig()
    assert _assistance_coefficient(10_000, 6, config) > 0.0
    assert (
        _assistance_coefficient(10_000, FINAL_UNASSISTED_STAGE, config) == 0.0
    )
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("ddpg", 8, protocol, "cpu", 43)
    algorithm.anchor_coefficient = 2.0
    _set_stage_assistance(algorithm, FINAL_UNASSISTED_STAGE)
    assert algorithm.anchor_coefficient == 0.0
    _configure_algorithm(algorithm)
    _set_stage_optimization(algorithm, FINAL_UNASSISTED_STAGE)
    assert algorithm.policy_delay == 4
    assert algorithm.target_policy_coefficient == 0.0
    assert algorithm.reference_policy_coefficient == 5.0
    assert algorithm.normalize_actor_q is True
    assert algorithm.actor_optimizer.param_groups[0]["lr"] == 5e-6


def test_plateau_promotion_is_pre_final_and_preserves_strict_result() -> None:
    validation = {
        "passed": False,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "failure_rate": 0.0,
        "maximum_background_gridlock_seconds": 0.0,
        "maximum_gap_m": 38.9,
    }
    promoted = _apply_plateau_promotion(validation, stage=6, stage_steps=60_000)
    assert promoted["passed"] is True
    assert promoted["strict_passed"] is False
    assert promoted["promotion_reason"] == "functional_plateau"
    final = _apply_plateau_promotion(validation, stage=7, stage_steps=200_000)
    assert final["passed"] is False
    assert final["promotion_reason"] == "none"


def test_ddpg_final_stage_uses_fixed_policy_trust_without_teacher_anchor() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("ddpg", 8, protocol, "cpu", 43)
    _configure_algorithm(algorithm)
    _set_stage_assistance(algorithm, FINAL_UNASSISTED_STAGE)
    _set_stage_optimization(algorithm, FINAL_UNASSISTED_STAGE)
    algorithm.learning_starts = 0
    algorithm.actor_learning_starts = 0
    algorithm.policy_delay = 1
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = algorithm.act(state, deterministic=False, total_steps=0)
        algorithm.record(state, sample, -0.1, state, False)
    metrics = algorithm.update_if_ready(total_steps=10)
    assert metrics["train/actor_updated"] == 1.0
    assert metrics["train/anchor_coefficient"] == 0.0
    assert metrics["train/target_policy_coefficient"] == 0.0
    assert metrics["train/reference_policy_coefficient"] == 5.0
    assert metrics["loss/reference_policy_trust"] >= 0.0
    assert metrics["train/policy_q_scale"] >= 1.0


def test_ddpg_uses_independent_twin_critics_and_loads_legacy_state() -> None:
    protocol = replace(ComparisonProtocol(), replay_capacity=16, batch_size=4)
    algorithm = create_algorithm("ddpg", 8, protocol, "cpu", 43)
    assert any(
        not torch.equal(first, second)
        for first, second in zip(
            algorithm.critic.parameters(), algorithm.critic2.parameters()
        )
    )
    legacy = algorithm.state_dict()
    for key in (
        "critic2",
        "critic2_target",
        "critic2_optimizer",
        "implementation_variant",
    ):
        legacy.pop(key)
    restored = create_algorithm("ddpg", 8, protocol, "cpu", 44)
    restored.load_state_dict(legacy)
    for source, target in zip(
        restored.critic2.parameters(), restored.critic2_target.parameters()
    ):
        torch.testing.assert_close(source, target)
    restored.learning_starts = 0
    restored.actor_learning_starts = 100
    state = np.zeros(8, np.float32)
    for _ in range(4):
        sample = restored.act(state, deterministic=False, total_steps=0)
        restored.record(state, sample, -0.1, state, False)
    metrics = restored.update_if_ready(total_steps=10)
    assert metrics["train/twin_critic"] == 1.0
    assert metrics["train/q_disagreement"] >= 0.0


def test_fixed_horizon_profile_hash_and_leader_action(tmp_path) -> None:
    profile_values = [
        {
            "profile_id": 0,
            "vehicle_id": 7,
            "start_time_ms": 100,
            "source_speed_mps": [10.0, 9.9, 9.7],
            "source_acceleration_mps2": [0.0, -1.0, -2.0],
            "target_speed_mps": [25.0, 24.9, 24.7],
        }
    ]
    payload = {
        "profile_version": 1,
        "profiles": profile_values,
        "profiles_sha256": _canonical_hash(profile_values),
    }
    path = tmp_path / "profiles.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    profiles, restored = load_profiles(path)
    assert restored["profiles_sha256"] == payload["profiles_sha256"]
    assert profiles == [
        LeaderProfile(
            0,
            7,
            100,
            (10.0, 9.9, 9.7),
            (0.0, -1.0, -2.0),
            (25.0, 24.9, 24.7),
        )
    ]
    action = LeaderTracker(profiles[0]).action(0, 25.0)
    assert -3.0 <= float(action["accel"][0]) <= 1.2
    assert action["lane_change"] == 0

from __future__ import annotations

from dataclasses import asdict
import csv
import json

import numpy as np
import pytest
import torch
import scripts.run_graph_study as graph_study
import platoon_control.evaluation as evaluation_module

from platoon_control.graph_agents import (
    GraphObservationAdapter,
    GraphPlatoonController,
)
from platoon_control.graph_analysis import compare_paired_evaluations
from platoon_control.evaluation import (
    EvaluationConfig,
    _evaluation_cases,
    evaluate,
    seed_evaluation_resume,
)
from platoon_control.graph_rl import (
    GraphCandidateGuard,
    GraphDreamerTrainer,
    GraphEpisodeReplay,
    GraphRLConfig,
    _apply_cohesive_warmup_control,
    _apply_synchronized_braking_warmup_control,
    _apply_synchronized_restart_warmup_control,
    _append_graph_transition,
    _collection_update_intervals,
    _collection_actor_update_due,
    _apply_foundation_probe,
    _foundation_probe,
    _final_evaluation_requires_refinement,
    _graph_stage_capability_passes,
    _migrate_restart_capability_records,
    _graph_capability_meets_criteria,
    _new_graph_episode,
    _post_episode_actor_update_count,
    _resume_final_refinement_pending,
    _role_balanced_behavior_loss,
    _shape_coordinated_restart_rewards,
    _shape_expanding_formation_rewards,
    _uses_collection_expert,
)
from platoon_control.graph_validation import _restart_timing_metrics
from platoon_control.models.graph_dreamer import (
    apply_compact_cruise_guidance,
    apply_wave_buffer_guidance,
    apply_team_motion_consensus,
    DifferentiableCBFProjector,
    GraphDreamerConfig,
    LatentGraphMessenger,
    MALatentGraphDreamer,
    NODE_COUNT,
    QuantizedSemanticChannel,
    physical_actor_priors,
)
from platoon_control.training import load_controller
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv
from scripts.run_graph_study import (
    _latest_metric_value,
    _new_metric_records,
    _training_findings,
    aggregate_training_seeds,
    evaluate_one,
)
from scripts.benchmark_graph_gpu import _batch as benchmark_batch


def test_total_evaluation_cases_are_balanced_unique_and_paired() -> None:
    cases = _evaluation_cases(EvaluationConfig(seed=2026, total_episodes=20))
    assert len(cases) == 20
    assert len(set(cases)) == 20
    counts = [front_count for _, front_count in cases]
    assert counts == [index % 9 for index in range(20)]
    assert max(counts.count(front_count) for front_count in range(9)) - min(
        counts.count(front_count) for front_count in range(9)
    ) == 1


def test_legacy_evaluation_cases_retain_per_front_count_semantics() -> None:
    cases = _evaluation_cases(
        EvaluationConfig(seed=2026, episodes_per_front_count=2)
    )
    assert len(cases) == 18
    assert all(
        sum(front_count == expected for _, front_count in cases) == 2
        for expected in range(9)
    )


def test_evaluation_resumes_only_missing_protocol_rows(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "evaluation"
    config = EvaluationConfig(
        seed=30_000,
        total_episodes=2,
        output_dir=str(output),
        calibration_path=str(tmp_path / "calibration.json"),
        wave_model_path=str(tmp_path / "wave.pt"),
        device="cpu",
    )
    variants = ("full", "no_latent_extrapolation")
    first_seed, first_count = _evaluation_cases(config)[0]
    recovered = {
        "variant": "full",
        "seed": first_seed,
        "front_vehicle_count": first_count,
        "success": 1,
    }
    assert seed_evaluation_resume(
        checkpoint, config, variants, [recovered, recovered]
    ) == 1

    calls: list[tuple[int, int, str]] = []

    def fake_episode(controller, episode_config, variant, seed, front_count):
        del controller, episode_config
        calls.append((seed, front_count, variant))
        return {
            "variant": variant,
            "seed": seed,
            "front_vehicle_count": front_count,
            "success": 1,
        }

    monkeypatch.setattr(evaluation_module, "load_controller", lambda *_: object())
    monkeypatch.setattr(evaluation_module, "_episode", fake_episode)
    assert evaluate(checkpoint, config, variants) == output / "summary.json"
    assert len(calls) == 3
    with (output / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    calls.clear()
    evaluate(checkpoint, config, variants)
    assert calls == []


def test_study_evaluation_cache_is_bound_to_checkpoint_and_protocol(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(graph_study, "EXPERIMENT_ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    output = tmp_path / "evaluation"
    calls: list[list[str]] = []

    def fake_run(command, log_name, allowed_return_codes=(0,)) -> None:
        del log_name, allowed_return_codes
        calls.append(command)
        variants_start = command.index("--variants") + 1
        variants_stop = command.index("--episodes-per-count")
        variants = command[variants_start:variants_stop]
        total = int(command[command.index("--total-episodes") + 1])
        output.mkdir(parents=True, exist_ok=True)
        with (output / "episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["variant", "seed"])
            writer.writeheader()
            for episode in range(total):
                for variant in variants:
                    writer.writerow({"variant": variant, "seed": episode})
        (output / "summary.json").write_text(
            json.dumps({"variants": {variant: {} for variant in variants}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(graph_study, "run", fake_run)
    variants = ("full", "no_latent_extrapolation")
    result = evaluate_one(
        str(tmp_path / "python.exe"),
        checkpoint,
        output,
        variants,
        3,
        30_000,
    )
    assert result == output / "episodes.csv"
    assert len(calls) == 1
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["row_count"] == 6
    assert provenance["variants"] == list(variants)

    evaluate_one(
        str(tmp_path / "python.exe"),
        checkpoint,
        output,
        variants,
        3,
        30_000,
    )
    assert len(calls) == 1

    checkpoint.write_bytes(b"checkpoint-v2")
    evaluate_one(
        str(tmp_path / "python.exe"),
        checkpoint,
        output,
        variants,
        3,
        30_000,
    )
    assert len(calls) == 2


def sampled_observations() -> dict[str, dict[str, np.ndarray]]:
    env = PlatoonParallelEnv()
    try:
        return {
            agent: env.observation_space(agent).sample()
            for agent in PLATOON_IDS
        }
    finally:
        env.close()


def test_gpu_benchmark_batch_matches_actor_training_contract() -> None:
    batch = benchmark_batch(batch_size=2, sequence_length=3)
    expected_shape = (2, 3, NODE_COUNT)
    assert batch["behavior_valid"].shape == expected_shape
    assert torch.all(batch["behavior_valid"] == 1.0)


def test_role_balanced_behavior_loss_does_not_dilute_leader_error() -> None:
    error = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    valid = torch.ones_like(error)
    total, leader, follower = _role_balanced_behavior_loss(error, valid)
    assert total.item() == pytest.approx(2.0)
    assert leader.item() == pytest.approx(4.0)
    assert follower.item() == pytest.approx(0.0)


def test_physical_target_ignores_multilane_history_for_hard_cbf() -> None:
    front_history = np.zeros((8, 20, 6), np.float32)
    front_history[0, -1] = np.asarray(
        [3.0, 0.0, 0.0, 0.0, 2.0, 0.0], np.float32
    )
    common = {
        "ego": np.asarray([0.0, 10.0, 0.0], np.float32),
        "local_front": np.zeros(6, np.float32),
        "road_profile": np.zeros((20, 3), np.float32),
        "front_history": front_history,
        "front_mask": np.asarray([1.0] + [0.0] * 7, np.float32),
    }

    leader = {
        **common,
        "signal": np.asarray([0.0, 0.0, 1.0, 0.55], np.float32),
        "signal_mask": np.ones(1, np.float32),
    }
    leader_physical = GraphObservationAdapter._physical_target(leader)
    assert leader_physical[2] == 0.0

    follower = {
        **common,
        "signal": np.zeros(4, np.float32),
        "signal_mask": np.zeros(1, np.float32),
    }
    follower_physical = GraphObservationAdapter._physical_target(follower)
    assert follower_physical[2] == 0.0

    leader["signal"] = np.asarray([0.0, 0.0, 0.02, 0.5], np.float32)
    certified = GraphObservationAdapter._physical_target(leader)
    assert certified[2] == pytest.approx(8.0)
    assert certified[3] == pytest.approx(5.0)


def test_physical_actor_priors_remove_restart_and_rejoin_dead_zones() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 25.0
    physical[..., 6] = 25.0
    physical[:, 1:, 2] = 12.0

    raw = torch.zeros(1, NODE_COUNT, 1, requires_grad=True)
    cruise = physical_actor_priors(raw, physical)
    assert cruise["cruise_mask"].item() == 1.0
    assert cruise["cruise_loss"].item() == pytest.approx(0.0)

    physical[..., 0] = 0.0
    physical[:, 0, 2] = 10.0
    physical[:, 0, 3] = 0.2
    restart = physical_actor_priors(raw, physical)
    assert restart["restart_mask"].item() == 1.0
    assert restart["restart_loss"].item() == pytest.approx(1.0)

    physical[:, 0, 3] = 0.0
    physical[:, 3, 2] = 25.0
    rejoin = physical_actor_priors(raw, physical)
    assert rejoin["rejoin_mask"].item() == 1.0
    assert rejoin["rejoin_loss"].item() > 1.0
    (restart["restart_loss"] + rejoin["rejoin_loss"]).sum().backward()
    assert raw.grad is not None
    assert torch.count_nonzero(raw.grad) > 0


def test_wave_attenuation_prior_uses_front_gap_as_disturbance_buffer() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 20.0
    physical[..., 6] = 25.0
    physical[:, 0, 2] = 30.0
    physical[:, 0, 4] = -2.0
    physical[:, 0, 5] = 10.0
    physical[:, 1:, 2] = 12.0
    raw = torch.zeros(1, NODE_COUNT, 1, requires_grad=True)
    priors = physical_actor_priors(raw, physical)
    assert priors["wave_mask"].item() == 1.0
    assert priors["wave_loss"].item() == pytest.approx(0.49)
    priors["wave_loss"].backward()
    assert raw.grad is not None
    assert torch.all(raw.grad > 0.0)

    physical[:, 0, 4] = 0.0
    physical[:, 0, 3] = -1.0
    fallback = physical_actor_priors(raw.detach(), physical)
    assert fallback["wave_mask"].item() == 1.0


def test_graph_replay_indexes_wave_events() -> None:
    replay = GraphEpisodeReplay(capacity_steps=20, seed=7)
    observations = []
    for _ in range(6):
        physical = np.zeros((NODE_COUNT, 7), np.float32)
        physical[:, 0] = 20.0
        physical[:, 2] = 12.0
        physical[0, 2] = 30.0
        physical[0, 3] = -1.0
        physical[0, 5] = 6.0
        observations.append({"physical_target": physical})
    zeros = [np.zeros((NODE_COUNT, 1), np.float32) for _ in range(5)]
    replay.episodes = [
        {
            "observations": observations,
            "applied_actions": zeros,
            "requested_actions": zeros,
            "raw_actions": zeros,
            "rewards": [np.zeros(NODE_COUNT, np.float32) for _ in range(5)],
            "continues": [np.ones(NODE_COUNT, np.float32) for _ in range(5)],
            "shielded": [np.zeros(NODE_COUNT, np.float32) for _ in range(5)],
            "behavior_valid": [np.ones(NODE_COUNT, np.float32) for _ in range(5)],
        }
    ]
    replay.steps = 5
    assert len(replay._foundation_event_candidates()["wave"]) == 5


def test_wave_buffer_guidance_is_shared_and_default_off() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[:, 1:, 2] = 12.0
    physical[:, 0, 2] = 40.0
    physical[:, 0, 3] = -2.0
    physical[:, 0, 5] = 8.0
    raw = torch.ones(1, NODE_COUNT, 1)
    assert torch.equal(
        apply_wave_buffer_guidance(raw, physical, GraphDreamerConfig()), raw
    )
    config = GraphDreamerConfig(wave_guidance_blend=1.0)
    guided = apply_wave_buffer_guidance(raw, physical, config)
    assert torch.allclose(guided, guided[:, :1].expand_as(guided))
    assert torch.all(guided < 0.0)


def test_compact_cruise_guidance_is_shared_and_yields_to_braking() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    physical[..., 6] = 25.0
    physical[:, 1:, 2] = 12.0
    raw = torch.zeros(1, NODE_COUNT, 1)
    previous = torch.zeros_like(raw)

    guided = apply_compact_cruise_guidance(raw, physical, previous)
    assert torch.all(guided > 0.0)
    assert guided[0, 0, 0].item() == pytest.approx(1.2)

    previous[:, 0] = -2.0
    positive_intention = torch.ones_like(previous)
    projected_brake = apply_compact_cruise_guidance(
        raw, physical, previous, positive_intention
    )
    assert torch.all(projected_brake > 0.0)

    braking = apply_compact_cruise_guidance(
        raw, physical, previous, previous.clone()
    )
    assert braking[0, 0, 0].item() == pytest.approx(1.2)
    assert torch.all(braking[0, 1:, 0] == -2.0)

    previous.zero_()
    raw[:, 0] = -1.0
    uncalibrated_residual = apply_compact_cruise_guidance(
        raw, physical, previous
    )
    assert torch.all(uncalibrated_residual > 0.0)

    previous.zero_()
    raw.zero_()
    physical[:, 3, 2] = 18.0
    physical[:, 3, 3] = 1.0
    broken = apply_compact_cruise_guidance(raw, physical, previous)
    assert torch.all(broken[0, :3, 0] < 0.0)
    assert torch.all(broken[0, 3:, 0] > 0.0)


def test_compact_cruise_guidance_uses_distant_stable_front_for_progress() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 15.0
    physical[:, 0, 2] = 60.0
    physical[:, 1:, 2] = 12.0
    physical[..., 6] = 25.0
    raw = torch.zeros(1, NODE_COUNT, 1)

    guided = apply_compact_cruise_guidance(
        raw, physical, torch.zeros_like(raw)
    )

    assert torch.all(guided == 1.2)


def test_clear_road_cruise_rejects_conflicting_actor_residuals() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = torch.tensor([25.2, 25.1, 24.9, 25.0, 24.8, 25.0])
    physical[:, 1:, 2] = torch.tensor([12.2, 12.4, 11.8, 12.1, 11.9])
    physical[:, 1:, 3] = physical[:, :-1, 0] - physical[:, 1:, 0]
    physical[..., 6] = 40.0
    raw = torch.tensor(
        [[[-1.0], [-0.8], [1.2], [1.0], [-0.7], [1.1]]]
    )

    guided = apply_compact_cruise_guidance(
        raw, physical, torch.zeros_like(raw)
    )

    assert guided.abs().amax() < 0.4
    assert guided[0, 0, 0] < 0.0
    assert guided[0, 4, 0] > 0.0


def test_distant_front_cruise_yields_to_risk_and_recovery_intent() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 15.0
    physical[:, 0, 2] = 60.0
    physical[:, 1:, 2] = 12.0
    physical[..., 6] = 25.0
    raw = torch.zeros(1, NODE_COUNT, 1)
    previous = torch.zeros_like(raw)

    physical[:, 0, 3] = -2.0
    closing = apply_compact_cruise_guidance(raw, physical, previous)
    assert torch.equal(closing, raw)

    physical[:, 0, 3] = 0.0
    physical[:, 0, 4] = -1.0
    front_braking = apply_compact_cruise_guidance(raw, physical, previous)
    assert torch.equal(front_braking, raw)

    physical[:, 0, 4] = 0.0
    previous[:, 0, 0] = -0.5
    braking = apply_compact_cruise_guidance(
        raw, physical, previous, previous.clone()
    )
    assert braking[0, 0, 0] == 0.0
    assert torch.all(braking[0, 1:, 0] == -0.5)

    previous.zero_()
    recovery_latch = torch.tensor([[True, False, False, False, False]])
    recovering = apply_compact_cruise_guidance(
        raw,
        physical,
        previous,
        recovery_latch=recovery_latch,
    )
    assert recovering[0, 0, 0] <= 0.0


def test_compact_cruise_guidance_synchronizes_restart_with_moving_front() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 6] = 25.0
    physical[:, 1:, 2] = 12.0
    physical[:, 0, 2] = 10.0
    raw = torch.full((1, NODE_COUNT, 1), -0.5)
    waiting = apply_compact_cruise_guidance(
        torch.full_like(raw, 0.5), physical, torch.zeros_like(raw)
    )
    assert torch.all(waiting == 0.0)

    physical[:, 0, 3] = 0.5
    guided = apply_compact_cruise_guidance(
        raw, physical, torch.zeros_like(raw)
    )
    assert torch.all(guided == 1.0)


def test_closing_recovery_holds_upstream_acceleration_until_rejoined() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 10.0
    physical[..., 6] = 25.0
    physical[:, 1:, 2] = 12.0
    physical[:, 1, 2] = 15.2
    physical[:, 1, 3] = -0.1
    raw = torch.ones(1, NODE_COUNT, 1)
    recovery_latch = torch.tensor([[True, False, False, False, False]])
    guided = apply_compact_cruise_guidance(
        raw,
        physical,
        torch.zeros_like(raw),
        recovery_latch=recovery_latch,
    )
    assert guided[0, 0, 0] <= 0.0
    assert torch.all(guided[0, 1:, 0] > 0.0)


def test_predictive_recovery_waits_before_fast_opening_link_breaks() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 10.0
    physical[..., 6] = 25.0
    physical[:, 1:, 2] = 12.0
    physical[:, 2, 2] = 9.84
    physical[:, 2, 3] = 2.30
    raw = torch.full((1, NODE_COUNT, 1), 1.2)
    previous = torch.tensor(
        [[[0.74], [0.45], [-1.95], [-6.0], [-3.0], [-3.0]]]
    )

    guided = apply_compact_cruise_guidance(raw, physical, previous)

    assert torch.all(guided[0, :2, 0] < 0.0)
    assert torch.all(guided[0, 2:, 0] > 0.0)


def test_team_motion_consensus_bounds_restart_speed_profile() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 2.0
    physical[:, 0, 2] = 20.0
    physical[:, 1:, 2] = 7.0
    acceleration = torch.tensor(
        [[[1.0], [0.8], [0.4], [0.0], [-0.4], [-0.8]]]
    )
    coordinated = apply_team_motion_consensus(acceleration, physical)
    assert coordinated[0, 0, 0].item() == pytest.approx(1.0)
    assert coordinated[0, -1, 0].item() == pytest.approx(0.52)
    assert torch.all(coordinated[0, 1:, 0] > acceleration[0, 1:, 0])

    physical[:, 3, 2] = 5.2
    physical[:, 3, 3] = -1.0
    risky = apply_team_motion_consensus(acceleration, physical)
    assert risky[0, 3, 0] <= acceleration[0, 3, 0]


def test_team_motion_consensus_yields_to_latched_formation_recovery() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 1.5
    physical[:, 0, 2] = 100.0
    physical[:, 1:, 2] = 12.0
    physical[:, 2, 2] = 13.3
    acceleration = torch.tensor(
        [[[0.0], [0.0], [0.25], [0.25], [0.25], [0.25]]]
    )
    recovery_latch = torch.tensor(
        [[False, True, False, False, False]]
    )

    ordinary = apply_team_motion_consensus(acceleration, physical)
    recovering = apply_team_motion_consensus(
        acceleration, physical, recovery_latch
    )

    assert not torch.allclose(ordinary, acceleration)
    assert torch.equal(recovering, acceleration)

    physical[:, 2, 3] = 2.0
    still_opening = apply_team_motion_consensus(
        acceleration, physical, recovery_latch
    )
    assert torch.equal(still_opening, ordinary)

    physical[:, 2, 3] = 0.2
    controlled_opening = apply_team_motion_consensus(
        acceleration, physical, recovery_latch
    )
    assert torch.equal(controlled_opening, acceleration)


def test_projector_does_not_broadcast_local_disturbance_upstream() -> None:
    model = MALatentGraphDreamer()
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 25.0
    physical[:, 1:, 2] = 12.0
    physical[..., 6] = 25.0
    raw = torch.ones(1, NODE_COUNT, 1)
    previous = torch.zeros_like(raw)
    previous[:, 3, 0] = -2.0
    safe, _ = model.projector(raw, physical, previous)
    assert safe[0, 0, 0] > -0.5


def test_rejoin_prior_holds_the_upstream_subplatoon() -> None:
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 6] = 25.0
    physical[:, 1:, 2] = 12.0
    physical[:, 3, 2] = 25.0
    physical[:, 3, 3] = 1.0
    raw = torch.full((1, NODE_COUNT, 1), 0.5, requires_grad=True)
    priors = physical_actor_priors(raw, physical)
    priors["rejoin_loss"].sum().backward()
    assert raw.grad is not None
    assert torch.all(raw.grad[0, :3, 0] > 0.0)


def tensor_observation(batch: int = 2) -> dict[str, torch.Tensor]:
    compact = GraphObservationAdapter()(sampled_observations())
    return {
        key: torch.as_tensor(value).unsqueeze(0).expand(
            batch, *value.shape
        ).clone()
        for key, value in compact.items()
    }


def test_graph_rssm_eval_transition_is_deterministic() -> None:
    model = MALatentGraphDreamer()
    model.eval()
    observation = tensor_observation(batch=1)
    initial = model.initial(1, torch.device("cpu"))
    action = torch.zeros(1, NODE_COUNT, 1)

    torch.manual_seed(11)
    first, first_prior, _ = model.observe_step(
        initial,
        action,
        observation,
        observation["adjacency"],
        observation["edge_ages"],
    )
    torch.manual_seed(97)
    second, second_prior, _ = model.observe_step(
        initial,
        action,
        observation,
        observation["adjacency"],
        observation["edge_ages"],
    )

    assert torch.equal(first.stochastic, second.stochastic)
    assert torch.equal(first_prior.stochastic, second_prior.stochastic)


def test_graph_adapter_builds_only_causal_physical_and_v2v_edges() -> None:
    observations = sampled_observations()
    for agent in PLATOON_IDS:
        observations[agent]["communication"][:] = np.asarray(
            [0.2, 0.2, 0.1, 0.1], np.float32
        )
    compact = GraphObservationAdapter()(observations)
    adjacency = compact["adjacency"]
    ages = compact["edge_ages"]
    assert adjacency.shape == (6, 6, 2)
    assert adjacency[1, 0, 0] == 1.0
    assert adjacency[1, 0, 1] == 1.0
    assert adjacency[0, 1, 0] == 0.0
    assert adjacency[0, 1, 1] == 1.0
    assert adjacency[5, 4, 0] == 1.0
    assert adjacency[5, 4, 1] == 1.0
    assert adjacency[5, 2, 1] == 0.0
    assert ages[5, 0, 1] == pytest.approx(0.2)
    assert ages[5, 4, 1] == pytest.approx(0.1)


def test_semantic_channel_is_16_dimensional_and_8_bit_quantized() -> None:
    config = GraphDreamerConfig(communication_dim=16, communication_bits=8)
    channel = QuantizedSemanticChannel(384, config).eval()
    code = channel(torch.randn(32, NODE_COUNT, 384))
    assert code.shape == (32, NODE_COUNT, 16)
    levels = (code + 1.0) * 0.5 * 255.0
    assert torch.allclose(levels, levels.round(), atol=1e-5)


def test_packet_age_selects_the_matching_semantic_history() -> None:
    messenger = LatentGraphMessenger(GraphDreamerConfig())
    history = torch.zeros(1, 3, NODE_COUNT, 16)
    history[:, 0] = 1.0
    history[:, 1] = 2.0
    history[:, 2] = 3.0
    ages = torch.zeros(1, NODE_COUNT, NODE_COUNT, 2)
    ages[:, 1, 0, 1] = 0.2
    selected = messenger._select_history(history, ages)
    assert torch.all(selected[:, 1, 0, 1] == 1.0)
    ages[:, 1, 0, 1] = 0.1
    selected = messenger._select_history(history, ages)
    assert torch.all(selected[:, 1, 0, 1] == 2.0)


def test_dropped_v2v_packet_cannot_leak_its_current_code() -> None:
    messenger = LatentGraphMessenger(GraphDreamerConfig()).train()
    history_a = torch.zeros(1, 3, NODE_COUNT, 16)
    history_b = history_a.clone()
    history_b[:, -1, 0] = 1.0
    intentions = torch.zeros(1, 3, NODE_COUNT, 1)
    adjacency = torch.zeros(1, NODE_COUNT, NODE_COUNT, 2)
    adjacency[:, 1, 0, 1] = 1.0
    ages = torch.zeros_like(adjacency)
    belief = torch.randn(1, NODE_COUNT, NODE_COUNT, 2, 128)
    context_a, belief_a = messenger(
        history_a, intentions, adjacency, ages, belief, communication_dropout=1.0
    )
    context_b, belief_b = messenger(
        history_b, intentions, adjacency, ages, belief, communication_dropout=1.0
    )
    assert torch.allclose(context_a, context_b)
    assert torch.allclose(belief_a, belief_b)


def test_semantic_intention_is_distinct_from_applied_transition_action() -> None:
    torch.manual_seed(6)
    model = MALatentGraphDreamer().eval()
    observation = tensor_observation(1)
    observation["adjacency"].zero_()
    observation["adjacency"][:, 1, 0, 1] = 1.0
    observation["edge_ages"].zero_()
    initial = model.initial(1, torch.device("cpu"))
    applied = torch.zeros(1, NODE_COUNT, 1)
    intent_a = torch.zeros_like(applied)
    intent_b = intent_a.clone()
    intent_b[:, 0] = 1.0
    with torch.no_grad():
        _, _, context_a = model.observe_step(
            initial,
            applied,
            observation,
            observation["adjacency"],
            observation["edge_ages"],
            previous_intention=intent_a,
        )
        _, _, context_b = model.observe_step(
            initial,
            applied,
            observation,
            observation["adjacency"],
            observation["edge_ages"],
            previous_intention=intent_b,
        )
    assert not torch.allclose(context_a[:, 1], context_b[:, 1])


def test_graph_rssm_uses_applied_action_for_the_next_posterior() -> None:
    torch.manual_seed(4)
    model = MALatentGraphDreamer().eval()
    observation = tensor_observation(1)
    initial = model.initial(1, torch.device("cpu"))
    zero = torch.zeros(1, NODE_COUNT, 1)
    one = torch.ones(1, NODE_COUNT, 1)
    with torch.no_grad():
        zero_state, _, _ = model.observe_step(
            initial,
            zero,
            observation,
            observation["adjacency"],
            observation["edge_ages"],
        )
        one_state, _, _ = model.observe_step(
            initial,
            one,
            observation,
            observation["adjacency"],
            observation["edge_ages"],
        )
    assert not torch.allclose(zero_state.deterministic, one_state.deterministic)


def test_differentiable_cbf_is_bounded_and_allows_emergency_braking() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 1.2, requires_grad=True)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 25.0
    physical[..., 2] = 3.0
    physical[..., 3] = -10.0
    physical[..., 4] = -3.0
    physical[..., 6] = 30.0
    safe, correction = projector(raw, physical, torch.zeros_like(raw))
    assert torch.all(safe >= config.emergency_action_min)
    assert torch.all(safe <= config.action_max)
    assert torch.all(safe < -1.0)
    (safe.mean() + correction.mean()).backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_differentiable_cbf_releases_jerk_tail_when_front_restarts() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 1.0)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 2] = 12.0
    physical[..., 3] = 0.2
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.full_like(raw, -0.7)
    safe, _ = projector(raw, physical, previous)
    assert torch.all(safe > 0.8)


def test_differentiable_cbf_releases_opening_leader_brake_as_team_intent() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.ones(1, NODE_COUNT, 1)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 6.0
    physical[:, 0, 2] = 10.0
    physical[:, 0, 3] = 3.0
    physical[:, 1:, 2] = 12.0
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.full_like(raw, -3.0)

    safe, _ = projector(raw, physical, previous)
    coordinated = apply_team_motion_consensus(safe, physical)

    assert safe[0, 0, 0] > 0.5
    assert torch.all(coordinated > 0.4)


def test_differentiable_cbf_keeps_braking_for_closing_leader_front() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.ones(1, NODE_COUNT, 1)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 10.0
    physical[..., 2] = 8.0
    physical[..., 3] = -3.0
    physical[..., 4] = -1.0
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.full_like(raw, -3.0)

    safe, _ = projector(raw, physical, previous)

    assert torch.all(safe < -1.0)


def test_differentiable_cbf_releases_compact_team_from_leader_intent() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 0.8)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 2] = 12.0
    physical[:, 0, 3] = 0.2
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.full_like(raw, -0.7)
    safe, _ = projector(raw, physical, previous)
    assert torch.all(safe > 0.6)


def test_differentiable_cbf_restart_overrides_stopped_cohesion_brake() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.ones(1, NODE_COUNT, 1)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[:, 0, 2] = 140.0
    physical[:, 0, 3] = 16.0
    physical[:, 1:, 2] = torch.tensor(
        [12.70, 13.33, 11.20, 11.70, 12.04]
    )
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.full_like(raw, -0.17)
    safe, _ = projector(raw, physical, previous)
    safe = apply_team_motion_consensus(safe, physical)
    assert torch.all(safe > 0.6)


def test_differentiable_cbf_preserves_advisory_speed_during_rejoin() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.ones(1, NODE_COUNT, 1)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 28.0
    physical[..., 2] = 50.0
    physical[..., 3] = 2.0
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.zeros_like(raw)
    rejoin_safe, _ = projector(raw, physical, previous)
    physical[..., 2] = 12.0
    cruise_safe, _ = projector(raw, physical, previous)
    assert torch.all(rejoin_safe < 0.0)
    assert torch.all(cruise_safe < 0.0)


def test_differentiable_projector_slows_only_upstream_of_expanding_break() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 1.0)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 20.0
    physical[:, 1:, 2] = 12.0
    physical[:, 3, 2] = 20.0
    physical[:, 3, 3] = 1.0
    physical[:, 3, 3] = 2.0
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.zeros_like(raw)
    safe, _ = projector(raw, physical, previous)
    assert torch.all(safe[0, :3, 0] < 0.0)
    assert torch.all(safe[0, 3:, 0] > 0.0)

    physical[:, 3, 3] = -0.2
    closing_safe, _ = projector(raw, physical, previous)
    assert torch.all(closing_safe[0, :3, 0] > 0.0)
    physical[:, 3, 2] = 12.8
    recovered_safe, _ = projector(raw, physical, previous)
    assert torch.all(recovered_safe[0, :3, 0] > 0.0)


def test_differentiable_projector_anticipates_fast_formation_opening() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 1.2)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 10.0
    physical[:, 1:, 2] = 12.0
    physical[:, 2, 2] = 9.84
    physical[:, 2, 3] = 2.30
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.tensor(
        [[[0.74], [0.45], [-1.95], [-6.0], [-3.0], [-3.0]]]
    )

    predictive_safe, _ = projector(raw, physical, previous)
    physical[:, 2, 3] = 0.0
    ordinary_safe, _ = projector(raw, physical, previous)

    assert torch.all(predictive_safe[0, :2, 0] < ordinary_safe[0, :2, 0])
    assert torch.all(predictive_safe[0, 2:, 0] == ordinary_safe[0, 2:, 0])


def test_differentiable_projector_preserves_upstream_subplatoon_cohesion() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 1.0)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 20.0
    physical[:, 0, 0] = 21.0
    physical[:, 1, 0] = 20.5
    physical[:, 1:, 2] = 12.0
    physical[:, 3, 2] = 20.0
    physical[:, 3, 3] = 1.0
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    safe, _ = projector(raw, physical, torch.zeros_like(raw))
    assert torch.all(safe[0, :3, 0] <= 0.0)
    assert safe[0, 0, 0] <= safe[0, 1, 0] <= safe[0, 2, 0]
    assert safe[0, 3, 0] > 0.0


def test_closing_recovery_keeps_free_leader_creeping() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = torch.tensor(
        [0.0, 0.12, 0.46, 0.48, 0.62, 0.72]
    )
    physical[:, 1:, 2] = torch.tensor(
        [13.08, 12.87, 7.77, 10.02, 8.11]
    )
    physical[:, 1:, 3] = torch.tensor(
        [-0.12, -0.34, -0.02, -0.14, -0.09]
    )
    physical[..., 6] = 25.0
    raw = torch.zeros(1, NODE_COUNT, 1)
    recovery_latch = torch.tensor([[True, True, False, False, False]])
    guided = apply_compact_cruise_guidance(
        raw,
        physical,
        torch.full_like(raw, -0.17),
        recovery_latch=recovery_latch,
    )
    safe, _ = projector(guided, physical, torch.full_like(raw, -0.17))
    assert guided[0, 0, 0] >= 0.8
    assert safe[0, 0, 0] > 0.0


def test_differentiable_projector_avoids_hard_shield_during_rejoin() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 0.8)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 20.0
    physical[:, 1:, 2] = 12.0
    physical[:, 3, 0] = 22.0
    physical[:, 3, 2] = 25.0
    physical[:, 3, 3] = -3.5
    physical[:, 3, 4] = -0.8
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    previous = torch.full_like(raw, 0.6)
    safe, _ = projector(raw, physical, previous)
    assert safe[0, 3, 0] < -0.5
    assert safe[0, 0, 0] > 0.0


def test_differentiable_projector_closes_subthreshold_gap_smoothly() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 0.8)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 20.0
    physical[:, 1:, 2] = 12.0
    physical[:, 3, 2] = 14.0
    physical[:, 3, 3] = 0.2
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    safe, _ = projector(raw, physical, torch.full_like(raw, 0.8))
    assert torch.all(safe[0, :3, 0] < safe[0, 3, 0])
    assert torch.all(safe[0, 3:, 0] > 0.0)


def test_differentiable_projector_damps_compact_team_speed_spread() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 1.0)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = torch.linspace(10.0, 10.5, NODE_COUNT)
    physical[:, 1:, 2] = 12.0
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    safe, _ = projector(raw, physical, torch.full_like(raw, 0.8))
    assert safe[0, -1, 0] < safe[0, 0, 0]


def test_clear_road_consensus_cannot_latch_a_braking_tail() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.full((1, NODE_COUNT, 1), 1.2)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = torch.linspace(20.2, 20.0, NODE_COUNT)
    physical[:, 1:, 2] = 12.0
    physical[..., 5] = 10.0
    physical[..., 6] = 40.0
    previous = torch.full_like(raw, -3.0)

    projected = []
    for _ in range(30):
        previous, _ = projector(raw, physical, previous)
        projected.append(previous.clone())

    trajectory = torch.stack(projected)
    assert torch.all(trajectory[1:] >= trajectory[:-1] - 1e-6)
    assert torch.all(trajectory[-1] > 0.0)


def test_differentiable_projector_enforces_nominal_speed_without_v2i() -> None:
    config = GraphDreamerConfig()
    projector = DifferentiableCBFProjector(config)
    raw = torch.ones(1, NODE_COUNT, 1)
    physical = torch.zeros(1, NODE_COUNT, 7)
    physical[..., 0] = 26.0
    physical[:, 1:, 2] = 12.0
    physical[..., 5] = 10.0
    physical[..., 6] = 40.0
    safe, _ = projector(raw, physical, torch.zeros_like(raw))
    assert safe[0, 0, 0] < 0.0
    assert torch.all(safe[0, 1:, 0] < 0.0)


def test_critic_regression_does_not_backpropagate_into_actor() -> None:
    torch.manual_seed(5)
    model = MALatentGraphDreamer()
    state = model.initial(2, torch.device("cpu"))
    adjacency = torch.ones(2, NODE_COUNT, NODE_COUNT, 2)
    ages = torch.zeros_like(adjacency)
    losses = model.imagination_loss(state, adjacency, ages, horizon=3)
    assert torch.isfinite(losses["motion_consensus"])
    losses["critic_loss"].backward()
    actor_gradients = [
        parameter.grad for parameter in model.actor.parameters()
        if parameter.grad is not None
    ]
    assert not actor_gradients or all(
        torch.count_nonzero(gradient) == 0 for gradient in actor_gradients
    )
    assert any(
        parameter.grad is not None for parameter in model.value_head.parameters()
    )


def test_graph_replay_keeps_t_plus_one_observation_and_three_action_semantics() -> None:
    observation = GraphObservationAdapter()(sampled_observations())
    episode = _new_graph_episode(observation)
    actions = {
        agent: {"accel": np.asarray([0.4], np.float32), "lane_change": 0}
        for agent in PLATOON_IDS
    }
    infos = {
        agent: {
            "applied_acceleration": -0.2,
            "safety_intervened": True,
        }
        for agent in PLATOON_IDS
    }
    rewards = {agent: 1.0 for agent in PLATOON_IDS}
    for _ in range(3):
        _append_graph_transition(
            episode,
            observation,
            actions,
            np.full((NODE_COUNT, 1), 0.8, np.float32),
            rewards,
            True,
            infos,
        )
    replay = GraphEpisodeReplay(20, 9)
    replay.add(episode)
    batch = replay.sample(1, 2)
    assert batch["observations"]["node_features"].shape[:2] == (1, 3)
    assert torch.all(batch["raw_actions"] == 0.8)
    assert torch.all(batch["requested_actions"] == 0.4)
    assert torch.all(batch["applied_actions"] == -0.2)
    assert torch.all(batch["shielded"] == 1.0)


def test_graph_replay_does_not_clone_the_online_policy_as_an_expert() -> None:
    observation = GraphObservationAdapter()(sampled_observations())
    episode = _new_graph_episode(observation)
    actions = {
        agent: {"accel": np.asarray([0.4], np.float32), "lane_change": 0}
        for agent in PLATOON_IDS
    }
    infos = {
        agent: {"applied_acceleration": 0.4, "safety_intervened": False}
        for agent in PLATOON_IDS
    }
    _append_graph_transition(
        episode,
        observation,
        actions,
        np.full((NODE_COUNT, 1), 0.4, np.float32),
        {agent: 1.0 for agent in PLATOON_IDS},
        True,
        infos,
        behavior_supervised=False,
    )
    assert np.count_nonzero(episode["behavior_valid"][0]) == 0


def test_graph_candidate_guard_accepts_then_rolls_back_the_whole_model() -> None:
    controller = GraphPlatoonController(device="cpu")
    trainer = GraphDreamerTrainer(controller, GraphRLConfig(device="cpu"))
    guard = GraphCandidateGuard(controller, trainer, rollback_cooldown_episodes=1)
    parameter = next(controller.model.actor.parameters())
    with torch.no_grad():
        parameter.add_(0.25)
    accepted_value = parameter.detach().clone()
    good = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 11.0,
        "max_gap": 13.0,
        "recovery": 0.0,
        "shield": 0.0,
        "string": 1.0,
        "active_count": 6.0,
        "connected_pair_count": 5.0,
        "leader_station": 1_000.0,
    }
    guard.begin({"front_vehicle_count": 2}, 42, good)
    assert guard.resolve(1, good) == (True, False)

    with torch.no_grad():
        parameter.add_(1.0)
    bad = {
        **good,
        "success": 0.0,
        "collision": 1.0,
        "failure": 1.0,
        "min_gap": 0.0,
        "max_gap": 500.0,
        "recovery": 1.0,
        "shield": 1.0,
        "string": 5.0,
        "active_count": 0.0,
        "connected_pair_count": 0.0,
        "leader_station": 0.0,
    }
    guard.begin({"front_vehicle_count": 2}, 42, good)
    assert guard.resolve(1, bad) == (False, True)
    assert torch.equal(parameter, accepted_value)
    assert guard.cooldown_episodes == 1

    truncated = {
        **good,
        "success": 0.0,
        "failure": 0.0,
        "leader_station": 3_249.0,
        "min_gap": 7.1,
        "max_gap": 12.01,
        "shield": 0.062,
        "string": 0.99,
        "v2i_score": 0.717,
    }
    safer_progress = {
        **truncated,
        "leader_station": 3_348.0,
        "max_gap": 12.0,
        "shield": 0.045,
        "v2i_score": 0.750,
    }
    guard.begin({"front_vehicle_count": 2}, 104, truncated)
    assert guard.resolve(4, safer_progress) == (True, False)


def test_foundation_probes_are_stage_zero_only_and_do_not_replace_normal_episodes() -> None:
    assert _foundation_probe(0, 0) is None
    assert _foundation_probe(1, 0) == "A2"
    assert _foundation_probe(2, 0) == "A3"
    assert _foundation_probe(3, 0) == "A4"
    assert _foundation_probe(4, 0) is None
    assert _foundation_probe(1, 1) is None


def test_stage_zero_stop_go_probe_contains_a_sustained_full_stop() -> None:
    commanded: list[float] = []
    speeds = {agent: 25.0 for agent in PLATOON_IDS}

    class Vehicle:
        @staticmethod
        def getIDList() -> list[str]:
            return ["front", *PLATOON_IDS]

        @staticmethod
        def getSpeed(vehicle: str) -> float:
            return speeds.get(vehicle, 0.0)

        @staticmethod
        def setSpeed(vehicle: str, speed: float) -> None:
            assert vehicle == "front"
            commanded.append(float(speed))

    env = type(
        "Env",
        (),
        {
            "_front_ids": ["front"],
            "connection": type("Connection", (), {"vehicle": Vehicle()})(),
        },
    )()
    actions = {
        agent: {"accel": np.asarray([0.0], np.float32)}
        for agent in PLATOON_IDS
    }
    assert not _apply_foundation_probe("A3", env, 49, actions)
    assert _apply_foundation_probe("A3", env, 50, actions)
    assert commanded[-1] == pytest.approx(24.7)
    assert all(
        float(actions[agent]["accel"][0]) == pytest.approx(-3.0)
        for agent in PLATOON_IDS
    )

    speeds.update({agent: 0.0 for agent in PLATOON_IDS})
    assert _apply_foundation_probe("A3", env, 134, actions)
    assert commanded[-1] == pytest.approx(0.0)
    assert all(float(actions[agent]["accel"][0]) == 0.0 for agent in PLATOON_IDS)
    assert _apply_foundation_probe("A3", env, 174, actions)
    assert commanded[-1] == pytest.approx(0.015)
    assert all(
        float(actions[agent]["accel"][0]) == pytest.approx(1.2)
        for agent in PLATOON_IDS
    )


def test_warmup_expert_does_not_restart_into_a_stopped_front_vehicle() -> None:
    platoon = np.zeros((NODE_COUNT, 6), np.float32)
    platoon[:, 5] = 1.0
    platoon[1:, 4] = 12.0
    observations = {
        "leader": {
            "platoon_state": platoon,
            "local_front": np.asarray([18.0, 0.0, 0.0, 0.0, 18.0, 1.0], np.float32),
        }
    }
    actions = {
        "leader": {"accel": np.asarray([0.0], np.float32), "lane_change": 0}
    }
    assert not _apply_cohesive_warmup_control(observations, actions)
    assert float(actions["leader"]["accel"][0]) == 0.0


def test_warmup_expert_holds_leader_for_a_disconnected_formation() -> None:
    platoon = np.zeros((NODE_COUNT, 6), np.float32)
    platoon[:, 5] = 1.0
    platoon[0, 1] = 25.0
    platoon[1:, 1] = 18.0
    platoon[1:, 4] = 24.0
    observations = {"leader": {"platoon_state": platoon}}
    actions = {
        "leader": {"accel": np.asarray([1.2], np.float32), "lane_change": 0}
    }
    assert _apply_cohesive_warmup_control(observations, actions)
    assert float(actions["leader"]["accel"][0]) < 0.0

    platoon[0, 1] = 0.0
    platoon[1:, 1] = 0.0
    platoon[1:, 4] = 12.0
    actions["leader"]["accel"] = np.asarray([0.0], np.float32)
    assert _apply_cohesive_warmup_control(observations, actions)
    assert float(actions["leader"]["accel"][0]) > 0.0


def test_warmup_expert_broadcasts_restart_intent_to_all_followers() -> None:
    platoon = np.zeros((NODE_COUNT, 6), np.float32)
    platoon[:, 5] = 1.0
    platoon[1:, 4] = 12.0
    observations = {"leader": {"platoon_state": platoon}}
    actions = {
        agent: {
            "accel": np.asarray([1.0 if agent == "leader" else 0.0], np.float32),
            "lane_change": 0,
        }
        for agent in PLATOON_IDS
    }
    assert _apply_synchronized_restart_warmup_control(observations, actions)
    assert all(float(actions[agent]["accel"][0]) == pytest.approx(1.0) for agent in PLATOON_IDS)

    platoon[-1, 4] = 35.0
    for agent in PLATOON_IDS:
        actions[agent]["accel"] = np.asarray(
            [1.0 if agent == "leader" else 0.0], np.float32
        )
    assert _apply_synchronized_restart_warmup_control(observations, actions)
    assert all(
        float(actions[agent]["accel"][0]) == pytest.approx(1.0)
        for agent in PLATOON_IDS[:-1]
    )
    assert float(actions[PLATOON_IDS[-1]]["accel"][0]) == pytest.approx(1.2)


def test_warmup_expert_broadcasts_leader_braking_to_all_followers() -> None:
    platoon = np.zeros((NODE_COUNT, 6), np.float32)
    platoon[:, 5] = 1.0
    platoon[:, 1] = 20.0
    platoon[0, 2] = -1.5
    platoon[1:, 4] = 12.0
    observations = {
        "leader": {
            "platoon_state": platoon,
            "local_front": np.asarray([20.0, -5.0, -1.5, 0.0, 20.0, 1.0], np.float32),
        }
    }
    actions = {
        agent: {"accel": np.asarray([0.5], np.float32), "lane_change": 0}
        for agent in PLATOON_IDS
    }
    assert _apply_synchronized_braking_warmup_control(observations, actions)
    assert all(
        float(actions[agent]["accel"][0]) == pytest.approx(-1.5)
        for agent in PLATOON_IDS[1:]
    )


def test_warmup_expert_does_not_broadcast_cohesion_braking_without_front() -> None:
    platoon = np.zeros((NODE_COUNT, 6), np.float32)
    platoon[:, 5] = 1.0
    platoon[:, 1] = 20.0
    platoon[0, 2] = -1.5
    platoon[1:, 4] = 18.0
    observations = {
        "leader": {
            "platoon_state": platoon,
            "local_front": np.zeros(6, np.float32),
        }
    }
    actions = {
        agent: {"accel": np.asarray([0.5], np.float32), "lane_change": 0}
        for agent in PLATOON_IDS
    }
    assert not _apply_synchronized_braking_warmup_control(observations, actions)
    assert all(
        float(actions[agent]["accel"][0]) == pytest.approx(0.5)
        for agent in PLATOON_IDS[1:]
    )


def test_graph_restart_reward_penalizes_delayed_follower_intent() -> None:
    platoon = np.zeros((NODE_COUNT, 6), np.float32)
    platoon[:, 5] = 1.0
    platoon[1:, 4] = 12.0
    observations = {"leader": {"platoon_state": platoon}}
    actions = {
        agent: {
            "accel": np.asarray([1.0 if agent == "leader" else 0.0], np.float32)
        }
        for agent in PLATOON_IDS
    }
    rewards = {agent: 0.0 for agent in PLATOON_IDS}
    active, spread = _shape_coordinated_restart_rewards(
        observations,
        actions,
        rewards,
        {"leader": {"leader_restart_opportunity": True}},
    )
    assert active
    assert spread == pytest.approx(1.0)
    assert rewards["leader"] < 0.0
    assert all(rewards[agent] < 0.0 for agent in PLATOON_IDS[1:])


def test_expanding_formation_penalizes_only_the_upstream_subplatoon() -> None:
    platoon = np.zeros((NODE_COUNT, 6), np.float32)
    platoon[:, 5] = 1.0
    platoon[:, 1] = 20.0
    platoon[2, 1] = 22.0
    platoon[3, 1] = 18.0
    platoon[1:, 4] = 12.0
    platoon[3, 4] = 20.0
    observations = {"leader": {"platoon_state": platoon}}
    actions = {
        agent: {"accel": np.asarray([0.5], np.float32)}
        for agent in PLATOON_IDS
    }
    rewards = {agent: 0.0 for agent in PLATOON_IDS}
    active, opening_speed, penalty = _shape_expanding_formation_rewards(
        observations, actions, rewards
    )
    assert active
    assert opening_speed == pytest.approx(4.0)
    assert penalty > 0.0
    assert all(rewards[agent] < 0.0 for agent in PLATOON_IDS[:3])
    assert all(rewards[agent] == 0.0 for agent in PLATOON_IDS[3:])

    platoon[2, 1] = 17.0
    closing_rewards = {agent: 0.0 for agent in PLATOON_IDS}
    active, _, _ = _shape_expanding_formation_rewards(
        observations, actions, closing_rewards
    )
    assert not active
    assert all(value == 0.0 for value in closing_rewards.values())
    platoon[3, 4] = 12.8
    recovered_rewards = {agent: 0.0 for agent in PLATOON_IDS}
    active, _, _ = _shape_expanding_formation_rewards(
        observations, actions, recovered_rewards
    )
    assert not active
    assert all(value == 0.0 for value in recovered_rewards.values())


def test_restart_timing_reports_whole_platoon_launch_spread() -> None:
    speeds = {}
    for index, agent in enumerate(PLATOON_IDS):
        speeds[agent] = [5.0] * 3 + [0.0] * (5 + index) + [0.6] * (10 - index)
    front_speeds = [5.0] * 3 + [0.0] * 5 + [0.6] * 10
    metrics = _restart_timing_metrics(speeds, [12.0] * 18, front_speeds)
    assert metrics["restart_detected"] == 1.0
    assert metrics["restart_spread_seconds"] == pytest.approx(0.5)
    assert metrics["restart_completion_seconds"] == pytest.approx(0.5)
    assert metrics["restart_response_seconds"] == pytest.approx(0.0)
    assert metrics["restart_maximum_speed_spread"] == pytest.approx(0.6)
    assert metrics["restart_maximum_gap"] == pytest.approx(12.0)


def test_restart_response_ignores_front_deceleration_before_full_stop() -> None:
    speeds = {
        agent: [5.0] * 3 + [0.0] * 12 + [0.6] * 5
        for agent in PLATOON_IDS
    }
    front_speeds = [5.0, 5.0, 5.0, 1.0, 0.5] + [0.0] * 8 + [0.6] * 7
    metrics = _restart_timing_metrics(speeds, [12.0] * 20, front_speeds)
    assert metrics["restart_response_seconds"] == pytest.approx(0.2)


def test_restart_response_is_not_scored_without_a_front_vehicle_restart() -> None:
    speeds = {
        agent: [5.0] * 3 + [0.0] * 8 + [0.6] * 7
        for agent in PLATOON_IDS
    }
    front_speeds = [5.0] * 18
    metrics = _restart_timing_metrics(speeds, [12.0] * 18, front_speeds)
    assert metrics["restart_detected"] == 1.0
    assert metrics["restart_spread_seconds"] == pytest.approx(0.0)
    assert metrics["restart_response_seconds"] == pytest.approx(0.0)


def test_restart_metrics_exclude_later_traffic_disturbances() -> None:
    shared_restart = [0.0] * 5 + [0.2, 1.0, 2.0, 3.1]
    speeds = {
        agent: shared_restart + [6.0 if index == 0 else 0.0] * 6
        for index, agent in enumerate(PLATOON_IDS)
    }
    maximum_gaps = [12.0] * len(shared_restart) + [40.0] * 6

    metrics = _restart_timing_metrics(speeds, maximum_gaps)

    assert metrics["restart_detected"] == 1.0
    assert metrics["restart_spread_seconds"] == pytest.approx(0.0)
    assert metrics["restart_maximum_speed_spread"] == pytest.approx(0.0)
    assert metrics["restart_maximum_gap"] == pytest.approx(12.0)


def test_legacy_restart_metric_migration_preserves_other_capabilities() -> None:
    legacy = {
        "success": 1.0,
        "v2i_score": 0.9,
        "restart_detected": 1.0,
        "restart_spread_seconds": 0.0,
        "restart_response_seconds": 0.0,
        "restart_maximum_speed_spread": 6.2,
    }

    migrated = _migrate_restart_capability_records([legacy], 1)[0]
    current = _migrate_restart_capability_records([legacy], 2)[0]

    assert migrated["success"] == legacy["success"]
    assert migrated["v2i_score"] == legacy["v2i_score"]
    assert migrated["restart_metric_valid"] == 0.0
    assert current["restart_metric_valid"] == 1.0


def test_graph_controller_matches_existing_evaluation_interface() -> None:
    observations = sampled_observations()
    expected_replay = GraphObservationAdapter()(observations)
    controller = GraphPlatoonController(device="cpu")
    policy_states, prediction = controller.follower_policy_states(observations)
    actions = controller.act(observations, deterministic=True)
    replay_observation = controller.replay_observation()
    assert prediction is None
    assert policy_states.shape == (5, 1, 42)
    assert set(actions) == set(PLATOON_IDS)
    assert all(
        GraphDreamerConfig().emergency_action_min
        <= float(actions[agent]["accel"][0])
        <= 1.2
        for agent in PLATOON_IDS
    )
    assert replay_observation.keys() == expected_replay.keys()
    for key in expected_replay:
        assert np.array_equal(replay_observation[key], expected_replay[key])
    replay_observation["physical_target"][0, 0] = -999.0
    assert controller.replay_observation()["physical_target"][0, 0] != -999.0


def test_graph_controller_migrates_pre_physical_actor_checkpoint() -> None:
    controller = GraphPlatoonController(device="cpu")
    state = controller.state_dict()
    state["graph_dreamer"] = {
        key: value
        for key, value in state["graph_dreamer"].items()
        if not key.startswith("actor.physical_trunk.")
    }
    GraphPlatoonController(device="cpu").load_state_dict(state)


def test_graph_checkpoint_is_auto_dispatched(tmp_path) -> None:
    controller = GraphPlatoonController(device="cpu")
    checkpoint = tmp_path / "graph.pt"
    torch.save(
        {
            "format_version": 1,
            "control_architecture": controller.architecture_id,
            "graph_model_config": asdict(controller.model_config),
            "graph_rl_config": {"variant": "full"},
            "models": controller.state_dict(),
        },
        checkpoint,
    )
    loaded = load_controller(checkpoint, "cpu")
    assert isinstance(loaded, GraphPlatoonController)


def test_paired_graph_report_matches_rows_by_seed_and_front_count(tmp_path) -> None:
    fields = [
        "variant",
        "seed",
        "front_vehicle_count",
        "success",
        "collision",
        "failure",
        "travel_time_s",
        "minimum_gap_m",
        "fuel_mg",
        "string_amplification",
        "shield_intervention_rate",
        "inference_p95_ms",
    ]
    baseline = tmp_path / "baseline.csv"
    graph = tmp_path / "graph.csv"
    for path, shield, fuel in ((baseline, 0.10, 100.0), (graph, 0.05, 90.0)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(4):
                writer.writerow(
                    {
                        "variant": "full",
                        "seed": 30000 + index,
                        "front_vehicle_count": 0,
                        "success": 1,
                        "collision": 0,
                        "failure": 0,
                        "travel_time_s": 120,
                        "minimum_gap_m": 8,
                        "fuel_mg": fuel,
                        "string_amplification": 1.0,
                        "shield_intervention_rate": shield,
                        "inference_p95_ms": 10,
                    }
                )
    output = tmp_path / "comparison.json"
    compare_paired_evaluations(baseline, graph, output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["paired_rows"] == 4
    assert report["shield_relative_reduction"] == pytest.approx(0.5)
    assert report["gates"]["success_noninferior"]


def test_training_seed_aggregate_preserves_between_seed_variation(tmp_path) -> None:
    inputs = []
    for seed, difference in ((42, -2.0), (43, -1.0), (44, 0.0)):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "passed": difference < 0.0,
                    "results": {
                        "fuel_mg": {"paired_difference": difference}
                    },
                }
            ),
            encoding="utf-8",
        )
        inputs.append(path)
    output = aggregate_training_seeds(inputs, tmp_path / "aggregate.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["training_seed_count"] == 3
    assert report["metrics"]["fuel_mg"][
        "training_seed_mean_paired_difference"
    ] == pytest.approx(-1.0)
    assert not report["all_model_comparison_gates_passed"]


def test_training_monitor_detects_simulation_and_numeric_faults(tmp_path) -> None:
    metric_path = tmp_path / "metrics.jsonl"
    metric_path.write_text(
        json.dumps(
            {
                "step": 40_000,
                "graph/world_loss": float("nan"),
                "training/episode": 10.0,
                "training/background_collision_events": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records, offset = _new_metric_records(metric_path, 0)
    fatal, warnings = _training_findings(records)
    assert offset == metric_path.stat().st_size
    assert any("non-finite" in finding for finding in fatal)
    assert any("background" in finding for finding in fatal)
    assert not warnings


def test_training_monitor_distinguishes_metered_queue_from_gridlock() -> None:
    metered = {
        "training/episode": 10.0,
        "training/background_peak_count": 180.0,
        "training/background_stuck_30s_count": 8.0,
        "training/background_max_gridlock_seconds": 0.0,
        "training/ramp_meter_closed_events": 8.0,
        "training/ramp_merge_completed_events": 9.0,
    }
    fatal, warnings = _training_findings([metered, dict(metered)])
    assert not fatal
    assert any("queued longer" in warning for warning in warnings)

    gridlocked = {
        **metered,
        "training/background_stuck_30s_count": 45.0,
        "training/ramp_merge_completed_events": 0.0,
    }
    fatal, _ = _training_findings([gridlocked, dict(gridlocked)])
    assert any("gridlock" in finding for finding in fatal)


def test_training_monitor_retains_sparse_curriculum_stage() -> None:
    records = [
        {"step": 100, "training/curriculum_stage": 2.0},
        {"step": 105, "graph/world_loss": 0.7},
    ]
    assert _latest_metric_value(records, "training/curriculum_stage") == 2.0
    assert _latest_metric_value(records, "missing") is None


def test_training_monitor_warns_on_sequential_platoon_restart() -> None:
    fatal, warnings = _training_findings(
        [
            {
                "step": 35_000,
                "training/episode": 12.0,
                "training/restart_detected": 1.0,
                "training/restart_spread_seconds": 2.4,
            }
        ]
    )
    assert not fatal
    assert any("sequentially" in warning for warning in warnings)


def test_training_monitor_warns_on_slow_platoon_restart_response() -> None:
    fatal, warnings = _training_findings(
        [
            {
                "step": 35_000,
                "training/episode": 12.0,
                "training/restart_detected": 1.0,
                "training/restart_spread_seconds": 0.0,
                "training/restart_response_seconds": 0.5,
            }
        ]
    )
    assert not fatal
    assert any("response remained delayed" in warning for warning in warnings)


def test_training_monitor_warns_on_post_restart_speed_oscillation() -> None:
    fatal, warnings = _training_findings(
        [
            {
                "step": 35_000,
                "training/episode": 12.0,
                "training/restart_detected": 1.0,
                "training/restart_maximum_speed_spread": 1.2,
            }
        ]
    )
    assert not fatal
    assert any("speed consensus oscillated" in warning for warning in warnings)


def test_training_monitor_retains_restart_warning_when_next_probe_is_clean() -> None:
    fatal, warnings = _training_findings(
        [
            {
                "step": 4_500,
                "training/episode": 3.0,
                "training/restart_detected": 1.0,
                "training/restart_maximum_speed_spread": 1.2,
            },
            {
                "step": 6_000,
                "training/episode": 4.0,
                "training/restart_detected": 0.0,
                "training/restart_maximum_speed_spread": 0.0,
            },
        ]
    )
    assert not fatal
    assert any("speed consensus oscillated" in warning for warning in warnings)


def test_collection_expert_covers_actor_training_and_foundation_gate() -> None:
    config = GraphRLConfig(
        warmup_steps=20_000,
        actor_warmup_steps=20_000,
        policy_handoff_steps=40_000,
    )
    assert config.actor_warmup_steps == config.warmup_steps
    assert _uses_collection_expert(20_000, config, 0, False)
    assert _uses_collection_expert(30_000, config, 0, False)
    assert _uses_collection_expert(39_999, config, 0, False)
    assert _uses_collection_expert(40_000, config, 0, False)
    assert not _uses_collection_expert(40_000, config, 0, True)
    assert not _uses_collection_expert(40_000, config, 1, False)
    assert _collection_update_intervals(config, False) == (5, 20)
    assert _collection_update_intervals(config, True) == (20, 40)
    assert _collection_actor_update_due(20_000, 20, config)
    assert _collection_actor_update_due(20_000, 40, config)
    assert not _collection_actor_update_due(19_999, 20, config)


def test_graph_foundation_can_gate_after_stabilized_36k_handoff() -> None:
    records = [
        {
            "success": 1.0,
            "collision": 0.0,
            "failure": 0.0,
            "min_gap": 12.0,
            "max_gap": 12.0,
            "recovery": 0.0,
            "shield": 0.0,
            "string": 1.0,
        }
        for _ in range(8)
    ]
    assert not _graph_stage_capability_passes(0, 35_999, records)[0]
    assert _graph_stage_capability_passes(0, 36_000, records)[0]
    assert GraphRLConfig().policy_handoff_steps == 36_000


def test_graph_stage_advances_early_only_after_stable_capability_window() -> None:
    passing = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 12.0,
        "max_gap": 12.5,
        "recovery": 0.0,
        "shield": 0.01,
        "string": 1.05,
    }
    stable_records = [dict(passing) for _ in range(8)]
    assert not _graph_stage_capability_passes(2, 12_499, stable_records)[0]
    assert _graph_stage_capability_passes(2, 12_500, stable_records)[0]

    unstable_records = [dict(passing) for _ in range(8)]
    unstable_records[-1]["string"] = 2.0
    assert not _graph_stage_capability_passes(2, 12_500, unstable_records)[0]
    assert _graph_stage_capability_passes(2, 25_000, unstable_records)[0]


def test_complex_stages_use_single_step_actor_candidates() -> None:
    config = GraphRLConfig(online_actor_updates_per_episode=4)
    assert _post_episode_actor_update_count(config, 3) == 4
    assert _post_episode_actor_update_count(config, 4) == 1
    assert _post_episode_actor_update_count(config, 7) == 1


def test_failed_final_evaluation_reopens_guarded_refinement() -> None:
    passing = {
        "evaluation/capability_pass": 1.0,
        "evaluation/collision_rate": 0.0,
        "evaluation/failure_rate": 0.0,
    }
    assert not _final_evaluation_requires_refinement(True, passing)
    assert not _final_evaluation_requires_refinement(
        False, {**passing, "evaluation/capability_pass": 0.0}
    )
    assert _final_evaluation_requires_refinement(
        True, {**passing, "evaluation/capability_pass": 0.0}
    )
    assert _final_evaluation_requires_refinement(
        True, {**passing, "evaluation/collision_rate": 0.2}
    )
    assert _final_evaluation_requires_refinement(
        True, {**passing, "evaluation/failure_rate": 0.2}
    )


def test_failed_final_evaluation_is_recovered_from_metric_journal(
    tmp_path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "step": 100,
                        "evaluation/capability_pass": 1.0,
                        "evaluation/collision_rate": 0.0,
                        "evaluation/failure_rate": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "step": 200,
                        "evaluation/capability_pass": 0.0,
                        "evaluation/collision_rate": 0.0,
                        "evaluation/failure_rate": 0.0,
                    }
                ),
            )
        ),
        encoding="utf-8",
    )
    assert _resume_final_refinement_pending(metrics, 200, True)
    assert not _resume_final_refinement_pending(metrics, 200, False)
    assert not _resume_final_refinement_pending(metrics, 199, True)


def test_graph_capability_rejects_restart_speed_oscillation() -> None:
    passing = {
        "success": 1.0,
        "collision": 0.0,
        "failure": 0.0,
        "min_gap": 11.0,
        "max_gap": 12.1,
        "recovery": 0.0,
        "shield": 0.01,
        "string": 1.0,
        "restart_detected": 1.0,
        "restart_response_seconds": 0.0,
        "restart_spread_seconds": 0.0,
        "restart_maximum_speed_spread": 0.4,
    }
    assert _graph_capability_meets_criteria(2, passing)
    oscillatory = {**passing, "restart_maximum_speed_spread": 0.6}
    assert not _graph_capability_meets_criteria(2, oscillatory)
    records = [dict(passing) for _ in range(7)] + [oscillatory]
    assert not _graph_stage_capability_passes(2, 25_000, records)[0]

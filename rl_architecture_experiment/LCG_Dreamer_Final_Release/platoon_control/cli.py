"""Command-line entry points for data preparation, training, and SUMO inference."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from platoon_sumo.env import PlatoonParallelEnv
from platoon_sumo.run_demo import baseline_actions

from .agents import PlatoonController
from .calibration import calibrate_sumo, calibration_gate
from .config import SystemConfig
from .data import (
    MatDelayDataset,
    ReplayWriter,
    stack_observation_sequence,
)
from .evaluation import EvaluationConfig, VARIANTS, evaluate
from .graph_rl import GraphRLConfig, train_graph_rl
from .graph_validation import evaluate_foundation_gates
from .graph_analysis import compare_paired_evaluations, evaluate_graph_auxiliary
from .models.graph_dreamer import GraphDreamerConfig
from .ngsim import NGSIMTrajectoryDataset, NGSIMWaveDataset, prepare_ngsim
from .rl import RLConfig, train_rl
from .supervised import SupervisedConfig, evaluate_supervised, train_supervised
from .training import MetricLogger, PlatoonTrainer, load_controller


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_ngsim_command(arguments: argparse.Namespace) -> None:
    path = prepare_ngsim(arguments.source, arguments.output)
    print(json.dumps({"manifest": str(path)}))


def calibrate_command(arguments: argparse.Namespace) -> None:
    path = calibrate_sumo(
        arguments.ngsim,
        arguments.output,
        trials=arguments.trials,
        seeds=arguments.seeds,
        duration=arguments.duration,
        seed=arguments.seed,
    )
    passed, report = calibration_gate(arguments.output, arguments.max_objective)
    print(json.dumps({"parameters": str(path), **report}))
    if not passed:
        raise SystemExit(2)


def train_supervised_command(arguments: argparse.Namespace) -> None:
    config = SupervisedConfig(
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        patience=arguments.patience,
        hidden_dim=arguments.hidden_dim,
        device=arguments.device,
        limit=arguments.limit,
        output_dir=arguments.output,
    )
    report = train_supervised(
        config,
        tuple(arguments.models),
        mat_path=arguments.mat,
        ngsim_dir=arguments.ngsim,
    )
    print(json.dumps({"report": str(report)}))


def evaluate_supervised_command(arguments: argparse.Namespace) -> None:
    report = evaluate_supervised(
        arguments.checkpoints,
        arguments.mat,
        arguments.ngsim,
        arguments.device,
        arguments.batch_size,
    )
    print(json.dumps({"report": str(report)}))


def train_rl_command(arguments: argparse.Namespace) -> None:
    config = RLConfig(
        seed=arguments.seed,
        max_sumo_steps=arguments.max_steps,
        warmup_steps=arguments.warmup_steps,
        actor_warmup_steps=arguments.actor_warmup_steps,
        leader_warmup_steps=arguments.leader_warmup_steps,
        leader_actor_warmup_steps=arguments.leader_actor_warmup_steps,
        replay_capacity=arguments.replay_capacity,
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
        update_every=arguments.update_every,
        policy_update_interval=arguments.policy_update_interval,
        leader_update_interval=arguments.leader_update_interval,
        evaluation_interval=arguments.evaluation_interval,
        checkpoint_interval=arguments.checkpoint_interval,
        evaluation_episodes=arguments.evaluation_episodes,
        convergence_patience=arguments.convergence_patience,
        min_steps_before_convergence=arguments.min_convergence_steps,
        device=arguments.device,
        checkpoint_dir=arguments.output,
        calibration_path=arguments.calibration,
        wave_model_path=arguments.wave_model,
        episode_max_time=arguments.episode_max_time,
        formation_curriculum_steps=arguments.formation_curriculum_steps,
        disturbance_curriculum_steps=arguments.disturbance_curriculum_steps,
        downstream_wave_training=arguments.downstream_wave_training,
        downstream_wave_training_probability=(
            arguments.downstream_wave_training_probability
        ),
    )
    checkpoint = train_rl(
        config,
        arguments.predictors,
        arguments.resume,
        reset_replay=arguments.reset_replay,
    )
    print(json.dumps({"checkpoint": str(checkpoint)}))


def train_graph_rl_command(arguments: argparse.Namespace) -> None:
    config = GraphRLConfig(
        seed=arguments.seed,
        max_sumo_steps=arguments.max_steps,
        warmup_steps=arguments.warmup_steps,
        actor_warmup_steps=arguments.actor_warmup_steps,
        world_pretrain_updates=arguments.world_pretrain_updates,
        replay_capacity=arguments.replay_capacity,
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
        update_every=arguments.update_every,
        actor_update_every=arguments.actor_update_every,
        online_world_updates_per_episode=(
            arguments.online_world_updates_per_episode
        ),
        online_actor_updates_per_episode=(
            arguments.online_actor_updates_per_episode
        ),
        candidate_rollback_cooldown_episodes=(
            arguments.candidate_rollback_cooldown_episodes
        ),
        behavior_initial_weight=arguments.behavior_initial_weight,
        behavior_decay_updates=arguments.behavior_decay_updates,
        behavior_cutoff_updates=arguments.behavior_cutoff_updates,
        actor_learning_rate=arguments.actor_learning_rate,
        foundation_imagination_weight=arguments.foundation_imagination_weight,
        imagination_weight=arguments.imagination_weight,
        observed_prior_weight=arguments.observed_prior_weight,
        foundation_event_fraction=arguments.foundation_event_fraction,
        evaluation_interval=arguments.evaluation_interval,
        checkpoint_interval=arguments.checkpoint_interval,
        evaluation_episodes=arguments.evaluation_episodes,
        min_steps_before_convergence=arguments.min_convergence_steps,
        device=arguments.device,
        checkpoint_dir=arguments.output,
        predictor_directory=arguments.predictors,
        calibration_path=arguments.calibration,
        wave_model_path=arguments.wave_model,
        episode_max_time=arguments.episode_max_time,
        downstream_wave_training=arguments.downstream_wave_training,
        downstream_wave_training_probability=(
            arguments.downstream_wave_training_probability
        ),
        variant=arguments.variant,
    )
    model_config = GraphDreamerConfig(
        communication_dim=arguments.communication_dim,
        communication_bits=arguments.communication_bits,
    )
    checkpoint = train_graph_rl(
        config,
        model_config,
        arguments.resume,
        reset_replay_on_resume=arguments.reset_replay,
        reset_curriculum_on_resume=arguments.reset_curriculum,
    )
    print(json.dumps({"checkpoint": str(checkpoint)}))


def validate_graph_command(arguments: argparse.Namespace) -> None:
    controller = load_controller(arguments.checkpoint, arguments.device)
    if not hasattr(controller, "architecture_id"):
        raise SystemExit("validate-graph-foundation requires a graph checkpoint")
    report = evaluate_foundation_gates(
        controller,
        arguments.calibration,
        arguments.wave_model,
        arguments.seed,
        arguments.output,
    )
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(2)


def evaluate_graph_auxiliary_command(arguments: argparse.Namespace) -> None:
    report = evaluate_graph_auxiliary(
        arguments.checkpoint,
        arguments.replay,
        arguments.output,
        arguments.device,
        arguments.batches,
        arguments.batch_size,
        arguments.sequence_length,
    )
    print(json.dumps({"report": str(report)}))


def compare_graph_evaluations_command(arguments: argparse.Namespace) -> None:
    report = compare_paired_evaluations(
        arguments.baseline_csv,
        arguments.graph_csv,
        arguments.output,
        arguments.baseline_variant,
        arguments.graph_variant,
        arguments.seed,
    )
    print(json.dumps({"report": str(report)}))


def evaluate_command(arguments: argparse.Namespace) -> None:
    config = EvaluationConfig(
        seed=arguments.seed,
        episodes_per_front_count=arguments.episodes_per_count,
        max_steps=arguments.max_steps,
        output_dir=arguments.output,
        calibration_path=arguments.calibration,
        wave_model_path=arguments.wave_model,
        device=arguments.device,
    )
    report = evaluate(arguments.checkpoint, config, tuple(arguments.variants))
    print(json.dumps({"report": str(report)}))


def collect_sumo(arguments: argparse.Namespace) -> None:
    env = PlatoonParallelEnv(max_time=max(2.0, arguments.steps * 0.1 + 0.2))
    observations, _ = env.reset(seed=arguments.seed)
    episode: list[dict] = []
    try:
        for _ in range(arguments.steps):
            if not env.agents:
                break
            actions = baseline_actions(observations)
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            episode.append(
                {
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "infos": infos,
                    "events": env.event_log,
                }
            )
            observations = next_observations
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        env.close()
    path = ReplayWriter(arguments.output).save(episode, arguments.seed, "sumo")
    print(json.dumps({"path": str(path), "steps": len(episode)}))


def inspect_data(arguments: argparse.Namespace) -> None:
    delay = MatDelayDataset(arguments.mat, limit=2)
    trajectory = NGSIMTrajectoryDataset(arguments.ngsim, split="train")
    wave = NGSIMWaveDataset(arguments.ngsim, split="train")
    print(
        json.dumps(
            {
                "mat_samples": len(delay),
                "mat_shapes": {key: list(value.shape) for key, value in delay[0].items()},
                "trajectory_samples": len(trajectory),
                "trajectory_shapes": {key: list(value.shape) for key, value in trajectory[0].items()},
                "wave_samples": len(wave),
                "wave_shapes": {key: list(value.shape) for key, value in wave[0].items()},
            },
            indent=2,
        )
    )


def _collect_leader_sequence(seed: int, steps: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    env = PlatoonParallelEnv(max_time=max(2.0, steps * 0.1 + 0.2))
    observations, _ = env.reset(seed=seed)
    leader_observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    try:
        for _ in range(steps):
            command = baseline_actions(observations)
            leader_observations.append(observations["leader"])
            actions.append(command["leader"]["accel"])
            observations, step_rewards, terminations, truncations, infos = env.step(command)
            rewards.append(step_rewards["leader"])
            actions[-1] = np.asarray(
                [infos["leader"]["applied_acceleration"]], dtype=np.float32
            )
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        env.close()
    device = torch.device("cpu")
    stacked = stack_observation_sequence(leader_observations, device)
    return (
        stacked,
        torch.as_tensor(np.stack(actions), dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(0),
        torch.ones(1, len(rewards), dtype=torch.float32),
    )


def smoke_train(arguments: argparse.Namespace) -> None:
    seed_everything(arguments.seed)
    config = SystemConfig()
    config.training.device = arguments.device
    config.model.hidden_dim = arguments.hidden_dim
    config.model.stochastic_dim = 8
    config.model.stochastic_classes = 8
    controller = PlatoonController(config)
    trainer = PlatoonTrainer(controller)
    logger = MetricLogger(config.training.log_dir)

    delay_loader = DataLoader(
        MatDelayDataset(arguments.mat, limit=max(8, arguments.batch_size)),
        batch_size=arguments.batch_size,
        shuffle=False,
    )
    trajectory_loader = DataLoader(
        NGSIMTrajectoryDataset(arguments.ngsim, split="train"),
        batch_size=arguments.batch_size,
        shuffle=False,
    )
    leader_batch = _collect_leader_sequence(arguments.seed, 6)
    metrics: dict[str, float] = {}
    for step in range(arguments.steps):
        metrics.update(trainer.train_delay(next(iter(delay_loader))))
        trajectory_batch = next(iter(trajectory_loader))
        metrics.update(trainer.train_trajectory(trajectory_batch))
        metrics.update(trainer.train_leader_front(trajectory_batch))
        metrics.update(trainer.train_leader(*leader_batch))
        logger.write(step, metrics)
        print(json.dumps({"step": step, **{key: round(value, 5) for key, value in metrics.items()}}))
    output = trainer.save_checkpoint(
        arguments.output,
        arguments.steps,
        metrics,
        [arguments.mat, arguments.ngsim, "SUMO:platoon_sumo/scenario"],
    )
    logger.close()
    print(json.dumps({"checkpoint": str(output), "device": str(controller.device)}))


def train_predictors(arguments: argparse.Namespace) -> None:
    config = SupervisedConfig(
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        device=arguments.device,
        limit=arguments.limit,
        output_dir=str(Path(arguments.output).parent),
    )
    train_supervised(config, mat_path=arguments.mat, ngsim_dir=arguments.ngsim)


def run_controller(arguments: argparse.Namespace) -> None:
    controller = load_controller(arguments.checkpoint, arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    total_steps = int(checkpoint.get("total_steps", 0))
    print(
        json.dumps(
            {
                "checkpoint_steps": total_steps,
                "follower_controller": "ctde_sac_longitudinal",
                "cacc_blending": False,
            }
        ),
        flush=True,
    )
    env = PlatoonParallelEnv(render_mode="human" if arguments.gui else None)
    scenario_options: dict[str, object] = {}
    if arguments.front_count is not None:
        scenario_options["front_vehicle_count"] = arguments.front_count
    if arguments.disable_cutin:
        scenario_options["cutin_enabled"] = False
    if arguments.wave_mode is not None:
        scenario_options["wave_mode"] = arguments.wave_mode
    if arguments.communication_delay_probability is not None:
        scenario_options["communication_delay_probability"] = (
            arguments.communication_delay_probability
        )
    if arguments.disable_v2i:
        scenario_options["v2i_enabled"] = False
    if arguments.speed_reference is not None:
        scenario_options["speed_reference"] = arguments.speed_reference
    if arguments.background_initial_scale is not None:
        scenario_options["background_initial_scale"] = (
            arguments.background_initial_scale
        )
    if arguments.background_demand_scale is not None:
        scenario_options["background_demand_scale"] = (
            arguments.background_demand_scale
        )
    observations, _ = env.reset(
        seed=arguments.seed,
        options=scenario_options or None,
    )
    controller.reset()
    try:
        for _ in range(arguments.max_steps):
            policy_states = controller.follower_policy_states(observations)[0]
            env.set_follower_control_predictions(
                controller.follower_control_predictions(policy_states)
            )
            actions = controller.act(observations, policy_states=policy_states)
            observations, _, terminations, truncations, infos = env.step(actions)
            controller.record_applied_actions(infos)
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-ngsim")
    prepare.add_argument("--source", default="NGSIM/trajectories-0500-0515.csv")
    prepare.add_argument("--output", default="artifacts/ngsim")
    prepare.set_defaults(function=prepare_ngsim_command)

    calibrate = subparsers.add_parser("calibrate-sumo")
    calibrate.add_argument("--ngsim", default="artifacts/ngsim")
    calibrate.add_argument("--output", default="artifacts/calibration")
    calibrate.add_argument("--trials", type=int, default=400)
    calibrate.add_argument("--seeds", type=int, default=3)
    calibrate.add_argument("--duration", type=float, default=90.0)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--max-objective", type=float, default=2.0)
    calibrate.set_defaults(function=calibrate_command)

    supervised = subparsers.add_parser("train-supervised")
    supervised.add_argument(
        "--models",
        nargs="+",
        choices=("cpidlm", "trajectory", "leader_front", "wave"),
        default=("cpidlm", "trajectory", "leader_front", "wave"),
    )
    supervised.add_argument("--epochs", type=int, default=100)
    supervised.add_argument("--batch-size", type=int, default=128)
    supervised.add_argument("--learning-rate", type=float, default=3e-4)
    supervised.add_argument("--patience", type=int, default=12)
    supervised.add_argument("--hidden-dim", type=int, default=128)
    supervised.add_argument("--limit", type=int)
    supervised.add_argument("--seed", type=int, default=42)
    supervised.add_argument("--device", default="auto")
    supervised.add_argument("--mat", default="data_30.mat")
    supervised.add_argument("--ngsim", default="artifacts/ngsim")
    supervised.add_argument("--output", default="artifacts/checkpoints")
    supervised.set_defaults(function=train_supervised_command)

    supervised_evaluation = subparsers.add_parser("evaluate-supervised")
    supervised_evaluation.add_argument("--checkpoints", default="artifacts/checkpoints")
    supervised_evaluation.add_argument("--mat", default="data_30.mat")
    supervised_evaluation.add_argument("--ngsim", default="artifacts/ngsim")
    supervised_evaluation.add_argument("--device", default="auto")
    supervised_evaluation.add_argument("--batch-size", type=int, default=512)
    supervised_evaluation.set_defaults(function=evaluate_supervised_command)

    rl = subparsers.add_parser("train-rl")
    rl.add_argument("--max-steps", type=int, default=1_000_000)
    rl.add_argument("--warmup-steps", type=int, default=5_000)
    rl.add_argument("--actor-warmup-steps", type=int, default=10_000)
    rl.add_argument("--leader-warmup-steps", type=int, default=5_000)
    rl.add_argument("--leader-actor-warmup-steps", type=int, default=30_000)
    rl.add_argument("--replay-capacity", type=int, default=30_000)
    rl.add_argument("--sequence-length", type=int, default=50)
    rl.add_argument("--batch-size", type=int, default=32)
    rl.add_argument("--update-every", type=int, default=1)
    rl.add_argument("--policy-update-interval", type=int, default=8)
    rl.add_argument("--leader-update-interval", type=int, default=20)
    rl.add_argument("--evaluation-interval", type=int, default=50_000)
    rl.add_argument("--checkpoint-interval", type=int, default=10_000)
    rl.add_argument("--evaluation-episodes", type=int, default=5)
    rl.add_argument("--convergence-patience", type=int, default=5)
    rl.add_argument("--min-convergence-steps", type=int, default=300_000)
    rl.add_argument("--seed", type=int, default=42)
    rl.add_argument("--device", default="auto")
    rl.add_argument("--predictors", default="artifacts/checkpoints")
    rl.add_argument("--calibration", default="artifacts/calibration/calibrated_parameters.json")
    rl.add_argument("--wave-model", default="artifacts/checkpoints/wave_best.pt")
    rl.add_argument("--episode-max-time", type=float, default=240.0)
    rl.add_argument("--formation-curriculum-steps", type=int, default=30_000)
    rl.add_argument("--disturbance-curriculum-steps", type=int, default=100_000)
    rl.add_argument("--downstream-wave-training", action="store_true")
    rl.add_argument(
        "--downstream-wave-training-probability", type=float, default=0.25
    )
    rl.add_argument("--output", default="artifacts/rl")
    rl.add_argument("--resume")
    rl.add_argument(
        "--reset-replay",
        action="store_true",
        help="Resume model/optimizer state but start a fresh replay buffer",
    )
    rl.set_defaults(function=train_rl_command)

    graph_rl = subparsers.add_parser("train-graph-rl")
    graph_rl.add_argument("--max-steps", type=int, default=1_000_000)
    graph_rl.add_argument("--warmup-steps", type=int, default=20_000)
    graph_rl.add_argument("--actor-warmup-steps", type=int, default=20_000)
    graph_rl.add_argument("--world-pretrain-updates", type=int, default=500)
    graph_rl.add_argument("--replay-capacity", type=int, default=30_000)
    graph_rl.add_argument("--sequence-length", type=int, default=50)
    graph_rl.add_argument("--batch-size", type=int, default=16)
    graph_rl.add_argument("--update-every", type=int, default=5)
    graph_rl.add_argument("--actor-update-every", type=int, default=20)
    graph_rl.add_argument("--online-world-updates-per-episode", type=int, default=16)
    graph_rl.add_argument("--online-actor-updates-per-episode", type=int, default=4)
    graph_rl.add_argument(
        "--candidate-rollback-cooldown-episodes", type=int, default=1
    )
    graph_rl.add_argument("--behavior-initial-weight", type=float, default=3.0)
    graph_rl.add_argument("--behavior-decay-updates", type=int, default=5_000)
    graph_rl.add_argument("--behavior-cutoff-updates", type=int, default=50_000)
    graph_rl.add_argument("--actor-learning-rate", type=float, default=3e-5)
    graph_rl.add_argument(
        "--foundation-imagination-weight", type=float, default=0.0
    )
    graph_rl.add_argument("--imagination-weight", type=float, default=0.05)
    graph_rl.add_argument("--observed-prior-weight", type=float, default=5.0)
    graph_rl.add_argument("--foundation-event-fraction", type=float, default=0.5)
    graph_rl.add_argument("--evaluation-interval", type=int, default=50_000)
    graph_rl.add_argument("--checkpoint-interval", type=int, default=10_000)
    graph_rl.add_argument("--evaluation-episodes", type=int, default=5)
    graph_rl.add_argument("--min-convergence-steps", type=int, default=300_000)
    graph_rl.add_argument("--seed", type=int, default=42)
    graph_rl.add_argument("--device", default="auto")
    graph_rl.add_argument("--predictors", default="artifacts/checkpoints")
    graph_rl.add_argument("--calibration", default="artifacts/calibration/calibrated_parameters.json")
    graph_rl.add_argument("--wave-model", default="artifacts/checkpoints/wave_best.pt")
    graph_rl.add_argument("--episode-max-time", type=float, default=240.0)
    graph_rl.add_argument("--downstream-wave-training", action="store_true")
    graph_rl.add_argument(
        "--downstream-wave-training-probability", type=float, default=0.25
    )
    graph_rl.add_argument("--communication-dim", type=int, default=16)
    graph_rl.add_argument("--communication-bits", type=int, default=8)
    graph_rl.add_argument(
        "--variant",
        choices=("full", "no_semantic_communication", "no_auxiliary_distillation"),
        default="full",
    )
    graph_rl.add_argument("--output", default="artifacts/graph_rl_seed42")
    graph_rl.add_argument("--resume")
    graph_rl.add_argument(
        "--reset-replay",
        action="store_true",
        help="Resume network/optimizer state but collect a fresh replay buffer",
    )
    graph_rl.add_argument(
        "--reset-curriculum",
        action="store_true",
        help="Resume weights but restart the scenario curriculum at stage 0",
    )
    graph_rl.set_defaults(function=train_graph_rl_command)

    graph_validation = subparsers.add_parser("validate-graph-foundation")
    graph_validation.add_argument("checkpoint")
    graph_validation.add_argument("--seed", type=int, default=42)
    graph_validation.add_argument("--device", default="auto")
    graph_validation.add_argument("--calibration", default="artifacts/calibration/calibrated_parameters.json")
    graph_validation.add_argument("--wave-model", default="artifacts/checkpoints/wave_best.pt")
    graph_validation.add_argument("--output", default="artifacts/evaluation/foundation_validation.json")
    graph_validation.set_defaults(function=validate_graph_command)

    graph_auxiliary = subparsers.add_parser("evaluate-graph-auxiliary")
    graph_auxiliary.add_argument("checkpoint")
    graph_auxiliary.add_argument("replay")
    graph_auxiliary.add_argument("--device", default="auto")
    graph_auxiliary.add_argument("--batches", type=int, default=20)
    graph_auxiliary.add_argument("--batch-size", type=int, default=16)
    graph_auxiliary.add_argument("--sequence-length", type=int, default=50)
    graph_auxiliary.add_argument("--output", default="artifacts/evaluation/graph_auxiliary.json")
    graph_auxiliary.set_defaults(function=evaluate_graph_auxiliary_command)

    graph_comparison = subparsers.add_parser("compare-graph-evaluations")
    graph_comparison.add_argument("baseline_csv")
    graph_comparison.add_argument("graph_csv")
    graph_comparison.add_argument("--baseline-variant", default="full")
    graph_comparison.add_argument("--graph-variant", default="full")
    graph_comparison.add_argument("--seed", type=int, default=42)
    graph_comparison.add_argument("--output", default="artifacts/evaluation/paired_comparison.json")
    graph_comparison.set_defaults(function=compare_graph_evaluations_command)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("checkpoint")
    evaluation.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    evaluation.add_argument("--episodes-per-count", type=int, default=5)
    evaluation.add_argument("--max-steps", type=int, default=2400)
    evaluation.add_argument("--seed", type=int, default=2026)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--calibration", default="artifacts/calibration/calibrated_parameters.json")
    evaluation.add_argument("--wave-model", default="artifacts/checkpoints/wave_best.pt")
    evaluation.add_argument("--output", default="artifacts/evaluation")
    evaluation.set_defaults(function=evaluate_command)

    inspect = subparsers.add_parser("inspect-data")
    inspect.add_argument("--mat", default="data_30.mat")
    inspect.add_argument("--ngsim", default="artifacts/ngsim")
    inspect.set_defaults(function=inspect_data)

    collect = subparsers.add_parser("collect-sumo")
    collect.add_argument("--steps", type=int, default=1800)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--output", default="artifacts/replay")
    collect.set_defaults(function=collect_sumo)

    smoke = subparsers.add_parser("smoke-train")
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument("--batch-size", type=int, default=4)
    smoke.add_argument("--hidden-dim", type=int, default=64)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--mat", default="data_30.mat")
    smoke.add_argument("--ngsim", default="artifacts/ngsim")
    smoke.add_argument("--output", default="artifacts/checkpoints/smoke.pt")
    smoke.set_defaults(function=smoke_train)

    train = subparsers.add_parser("train-predictors")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--limit", type=int)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--mat", default="data_30.mat")
    train.add_argument("--ngsim", default="artifacts/ngsim")
    train.add_argument("--output", default="artifacts/checkpoints/predictors.pt")
    train.set_defaults(function=train_predictors)

    run = subparsers.add_parser("run")
    run.add_argument("checkpoint")
    run.add_argument("--gui", action="store_true")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", default="auto")
    run.add_argument("--max-steps", type=int, default=2400)
    run.add_argument("--front-count", type=int, choices=range(9))
    run.add_argument("--disable-cutin", action="store_true")
    run.add_argument("--wave-mode", choices=("off", "periodic", "learned"))
    run.add_argument("--communication-delay-probability", type=float)
    run.add_argument("--disable-v2i", action="store_true")
    run.add_argument("--speed-reference", type=float)
    run.add_argument("--background-initial-scale", type=float)
    run.add_argument("--background-demand-scale", type=float)
    run.set_defaults(function=run_controller)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()

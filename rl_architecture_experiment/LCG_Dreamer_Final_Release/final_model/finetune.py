"""Fine-tune the final graph controller with a release-local scenario factory."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch

import platoon_control.graph_rl as graph_training
import platoon_control.rl as rl_training
from platoon_control.graph_rl import GraphRLConfig
from platoon_control.models.graph_dreamer import GraphDreamerConfig

from .runtime import ReleasePaths


ScenarioFactory = Callable[
    [int, np.random.Generator, Any, int], tuple[dict[str, Any], str]
]


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_new_scene", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import scenario file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_factory(path: Path) -> ScenarioFactory:
    module = _load_module(path)
    factory = getattr(module, "curriculum_options", None)
    if not callable(factory):
        raise AttributeError(
            f"{path} must define curriculum_options(stage, rng, config, total_steps)"
        )
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-file", default="examples/new_scene_curriculum.py"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--additional-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episode-max-time", type=float, default=240.0)
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument(
        "--keep-curriculum-progress",
        action="store_true",
        help="Keep stage 7 from the deployment checkpoint instead of restarting stage 0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate assets and the scenario factory without starting SUMO training",
    )
    arguments = parser.parse_args()
    if arguments.additional_steps <= 0:
        raise ValueError("--additional-steps must be positive")
    paths = ReleasePaths.discover()
    paths.validate(include_lane_change=False)
    scenario_file = (paths.root / arguments.scenario_file).resolve()
    factory = _load_factory(scenario_file)
    checkpoint = torch.load(
        paths.longitudinal, map_location="cpu", weights_only=False
    )
    allowed = {field.name for field in fields(GraphRLConfig)}
    stored = {
        key: value
        for key, value in checkpoint.get("graph_rl_config", {}).items()
        if key in allowed
    }
    base = GraphRLConfig(**stored)
    total_steps = int(checkpoint["total_steps"])
    config = replace(
        base,
        seed=arguments.seed,
        max_sumo_steps=total_steps + arguments.additional_steps,
        checkpoint_dir=str(Path(arguments.output).resolve()),
        predictor_directory=str(paths.predictors),
        calibration_path=str(paths.calibration),
        wave_model_path=str(paths.wave_model),
        episode_max_time=arguments.episode_max_time,
        evaluation_interval=arguments.evaluation_interval,
        checkpoint_interval=arguments.checkpoint_interval,
        device=arguments.device,
    )
    model_config = GraphDreamerConfig(**checkpoint["graph_model_config"])
    probe, probe_name = factory(
        0,
        np.random.default_rng(arguments.seed),
        config.evaluation_config(),
        total_steps,
    )
    if not isinstance(probe, dict) or not probe:
        raise TypeError("the scenario factory must return a non-empty options dictionary")
    graph_training._curriculum_options = factory
    rl_training._curriculum_options = factory
    run_metadata = {
        "base_checkpoint": str(paths.longitudinal),
        "base_total_steps": total_steps,
        "additional_steps": arguments.additional_steps,
        "scenario_file": str(scenario_file),
        "scenario_probe_name": probe_name,
        "fresh_replay": True,
        "curriculum_restarted": not arguments.keep_curriculum_progress,
    }
    output = Path(config.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "adaptation_request.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    if arguments.dry_run:
        print(json.dumps({"dry_run": True, **run_metadata}, ensure_ascii=False))
        return
    result = graph_training.train_graph_rl(
        config,
        model_config,
        paths.longitudinal,
        reset_replay_on_resume=True,
        reset_curriculum_on_resume=not arguments.keep_curriculum_progress,
    )
    print(json.dumps({"checkpoint": str(result.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

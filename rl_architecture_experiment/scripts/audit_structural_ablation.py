"""Audit and summarize the equal-budget structural graph ablation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from classical_comparison.common import reward_fingerprint
from platoon_control.graph_rl import GraphRLConfig
from platoon_control.models.graph_dreamer import GraphDreamerConfig


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "full": "graph_full_seed42",
    "no_semantic_communication": "graph_no_semantic_communication_seed42",
    "no_auxiliary_distillation": "graph_no_auxiliary_distillation_seed42",
}
METRICS = (
    "success",
    "collision",
    "failure",
    "travel_time_s",
    "mean_speed_mps",
    "jerk_rms_mps3",
    "minimum_gap_m",
    "fuel_mg",
    "string_amplification",
    "shield_intervention_rate",
    "inference_p95_ms",
)
AUXILIARY_METRICS = (
    "delay_teacher_normalized_rmse",
    "cutin_teacher_brier",
)
FULL_RUNTIME_VARIANTS = (
    "full",
    "no_latent_extrapolation",
    "no_differentiable_cbf",
    "float_semantic_code",
)
CONFIG_EXCLUSIONS = {"checkpoint_dir", "variant"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalized_config(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = asdict(GraphRLConfig())
    defaults.update(payload)
    return {
        key: value
        for key, value in defaults.items()
        if key not in CONFIG_EXCLUSIONS
    }


def _normalized_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = asdict(GraphDreamerConfig())
    defaults.update(payload)
    return defaults


def _optimizer_updates(trainer: dict[str, Any], name: str) -> int:
    optimizer = trainer.get("optimizers", {}).get(name, {})
    states = optimizer.get("state", {}).values()
    steps: list[int] = []
    for state in states:
        value = state.get("step", 0)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        steps.append(int(value))
    return max(steps, default=0)


def _read_rows(
    path: Path,
    expected_rows: int,
    expected_total_rows: int | None = None,
    expected_variants: tuple[str, ...] = ("full",),
) -> dict[tuple[int, int], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    expected_total_rows = expected_total_rows or expected_rows
    if len(raw_rows) != expected_total_rows:
        raise RuntimeError(
            f"{path} contains {len(raw_rows)} rows, expected {expected_total_rows}"
        )
    observed_variants = {str(raw.get("variant")) for raw in raw_rows}
    if observed_variants != set(expected_variants):
        raise RuntimeError(
            f"unexpected runtime variants in {path}: {sorted(observed_variants)}"
        )
    selected_rows = [raw for raw in raw_rows if raw.get("variant") == "full"]
    if len(selected_rows) != expected_rows:
        raise RuntimeError(
            f"{path} contains {len(selected_rows)} full rows, expected {expected_rows}"
        )
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for raw in selected_rows:
        key = (int(raw["seed"]), int(raw["front_vehicle_count"]))
        if key in rows:
            raise RuntimeError(f"duplicate paired case in {path}: {key}")
        rows[key] = {metric: float(raw[metric]) for metric in METRICS}
    values = np.asarray(
        [[row[metric] for metric in METRICS] for row in rows.values()],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite metric in {path}")
    return rows


def _bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(-1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _permutation_p(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(10_000, len(values)))
    permuted = np.abs((signs * values).mean(-1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (metric, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[metric] = running
    return adjusted


def _validate_provenance(
    path: Path,
    checkpoint: Path,
    episodes_path: Path,
    summary_path: Path,
    expected_total_episodes: int,
    seed: int,
    expected_variants: tuple[str, ...] = ("full",),
    expected_row_count: int | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_row_count = expected_row_count or expected_total_episodes
    expected = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "variants": list(expected_variants),
        "total_episodes": expected_total_episodes,
        "seed": seed,
        "episodes_sha256": sha256(episodes_path),
        "summary_sha256": sha256(summary_path),
        "row_count": expected_row_count,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid evaluation provenance {path}: {mismatches}")
    return payload


def _validate_auxiliary_report(
    path: Path,
    checkpoint: Path,
    replay: Path,
    minimum_batches: int = 20,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reported_checkpoint = (ROOT / Path(payload["checkpoint"])).resolve()
    reported_replay = (ROOT / Path(payload["replay"])).resolve()
    if reported_checkpoint != checkpoint.resolve():
        raise RuntimeError(
            f"auxiliary report checkpoint mismatch: "
            f"{reported_checkpoint} != {checkpoint.resolve()}"
        )
    if reported_replay != replay.resolve():
        raise RuntimeError(
            f"auxiliary report replay mismatch: "
            f"{reported_replay} != {replay.resolve()}"
        )
    if int(payload.get("batches", 0)) < minimum_batches:
        raise RuntimeError(f"insufficient auxiliary batches in {path}")
    metric_values: dict[str, dict[str, float]] = {}
    for metric in AUXILIARY_METRICS:
        metric_summary = payload.get("summary", {}).get(metric, {})
        values = np.asarray(
            [metric_summary.get("mean"), metric_summary.get("std")],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite auxiliary metric {metric} in {path}")
        metric_values[metric] = {
            "mean": float(values[0]),
            "std": float(values[1]),
        }
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "batches": int(payload["batches"]),
        "passed": bool(payload.get("passed", False)),
        "metrics": metric_values,
    }


def audit(
    evaluation_root: Path,
    output: Path,
    expected_rows: int = 20,
    seed: int = 30_000,
    minimum_steps: int = 300_000,
) -> Path:
    checkpoints: dict[str, dict[str, Any]] = {}
    rows_by_variant: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    normalized_configs: dict[str, dict[str, Any]] = {}
    normalized_model_configs: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    auxiliary_reports: dict[str, dict[str, Any]] = {}
    for variant, run_name in VARIANTS.items():
        run_root = ROOT / "artifacts" / run_name
        checkpoint = run_root / "graph_complete.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = dict(state["graph_rl_config"])
        if config.get("variant") != variant:
            raise RuntimeError(
                f"checkpoint variant mismatch for {run_name}: {config.get('variant')}"
            )
        gates = {
            "complete": bool(state.get("complete", False)),
            "minimum_steps": int(state.get("total_steps", 0)) >= minimum_steps,
            "final_stage": int(state.get("curriculum_stage", -1)) == 7,
            "foundation_passed": bool(state.get("foundation_passed", False)),
            "actor_trained": int(state.get("trainer", {}).get("actor_updates", 0)) > 0,
        }
        if not all(gates.values()):
            raise RuntimeError(f"checkpoint gates failed for {run_name}: {gates}")
        checkpoint_info = {
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
            "total_steps": int(state["total_steps"]),
            "episodes": int(state["episodes"]),
            "curriculum_stage": int(state["curriculum_stage"]),
            "actor_updates": int(state["trainer"]["actor_updates"]),
            "world_optimizer_updates": _optimizer_updates(
                state["trainer"], "world"
            ),
            "critic_optimizer_updates": _optimizer_updates(
                state["trainer"], "critic"
            ),
            "pretrain_updates_completed": int(
                state.get("pretrain_updates_completed", 0)
            ),
            "gates": gates,
        }
        checkpoints[variant] = checkpoint_info
        normalized_configs[variant] = _normalized_config(config)
        normalized_model_configs[variant] = _normalized_model_config(
            dict(state["graph_model_config"])
        )
        suite = evaluation_root / run_name
        episodes_path = suite / "episodes.csv"
        summary_path = suite / "summary.json"
        provenance_path = suite / "provenance.json"
        source_variants = (
            FULL_RUNTIME_VARIANTS if variant == "full" else ("full",)
        )
        source_row_count = expected_rows * len(source_variants)
        rows_by_variant[variant] = _read_rows(
            episodes_path,
            expected_rows,
            source_row_count,
            source_variants,
        )
        provenance[variant] = _validate_provenance(
            provenance_path,
            checkpoint,
            episodes_path,
            summary_path,
            expected_rows,
            seed,
            source_variants,
            source_row_count,
        )
        auxiliary_path = run_root / "graph_auxiliary.json"
        expected_replay = (run_root / "graph_replay_latest.pt").resolve()
        try:
            auxiliary_reports[variant] = _validate_auxiliary_report(
                auxiliary_path,
                checkpoint,
                expected_replay,
            )
        except RuntimeError as error:
            raise RuntimeError(f"invalid auxiliary report for {run_name}") from error

    reference_config = normalized_configs["full"]
    config_mismatches = {
        variant: {
            key: {"full": reference_config.get(key), "variant": config.get(key)}
            for key in sorted(set(reference_config) | set(config))
            if reference_config.get(key) != config.get(key)
        }
        for variant, config in normalized_configs.items()
        if variant != "full"
    }
    config_mismatches = {
        variant: values for variant, values in config_mismatches.items() if values
    }
    if config_mismatches:
        raise RuntimeError(f"unequal training configuration: {config_mismatches}")

    reference_model_config = normalized_model_configs["full"]
    model_config_mismatches = {
        variant: {
            key: {
                "full": reference_model_config.get(key),
                "variant": config.get(key),
            }
            for key in sorted(set(reference_model_config) | set(config))
            if reference_model_config.get(key) != config.get(key)
        }
        for variant, config in normalized_model_configs.items()
        if variant != "full"
    }
    model_config_mismatches = {
        variant: values
        for variant, values in model_config_mismatches.items()
        if values
    }
    if model_config_mismatches:
        raise RuntimeError(
            f"unequal graph-model configuration: {model_config_mismatches}"
        )

    update_fields = (
        "actor_updates",
        "world_optimizer_updates",
        "critic_optimizer_updates",
        "pretrain_updates_completed",
    )
    update_mismatches = {
        field: {
            variant: checkpoints[variant][field] for variant in VARIANTS
        }
        for field in update_fields
        if len({checkpoints[variant][field] for variant in VARIANTS}) != 1
    }
    if update_mismatches:
        raise RuntimeError(
            "unequal optimization budget; the study is not an equal-budget "
            f"ablation: {update_mismatches}"
        )

    case_sets = {variant: set(rows) for variant, rows in rows_by_variant.items()}
    reference_cases = case_sets["full"]
    if any(cases != reference_cases for cases in case_sets.values()):
        raise RuntimeError("structural ablation evaluation cases are not strictly paired")

    rng = np.random.default_rng(42)
    statistics: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for variant, rows in rows_by_variant.items():
        summary_row: dict[str, Any] = {"variant": variant}
        for metric in METRICS:
            values = np.asarray([rows[key][metric] for key in sorted(rows)])
            summary_row[f"{metric}_mean"] = float(values.mean())
            summary_row[f"{metric}_std"] = float(values.std())
        for metric in AUXILIARY_METRICS:
            metric_summary = auxiliary_reports[variant]["metrics"][metric]
            summary_row[f"{metric}_mean"] = metric_summary["mean"]
            summary_row[f"{metric}_std"] = metric_summary["std"]
        summary_rows.append(summary_row)
        if variant == "full":
            continue
        metric_results: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        for metric in METRICS:
            differences = np.asarray(
                [rows[key][metric] - rows_by_variant["full"][key][metric] for key in sorted(rows)],
                dtype=np.float64,
            )
            interval = _bootstrap_interval(differences, rng)
            p_value = _permutation_p(differences, rng)
            raw_p[metric] = p_value
            metric_results[metric] = {
                "paired_difference_ablation_minus_full": float(differences.mean()),
                "ci95": list(interval),
                "p_value": p_value,
            }
        adjusted = _holm_adjust(raw_p)
        for metric, value in adjusted.items():
            metric_results[metric]["holm_p_value"] = value
        statistics[variant] = metric_results

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name("structural_ablation_summary.csv")
    temporary_summary = summary_path.with_suffix(".csv.tmp")
    with temporary_summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    temporary_summary.replace(summary_path)

    source_files = (
        ROOT / "platoon_sumo" / "env.py",
        ROOT / "platoon_control" / "graph_rl.py",
        ROOT / "platoon_control" / "models" / "graph_dreamer.py",
        ROOT / "platoon_control" / "evaluation.py",
        ROOT / "platoon_control" / "graph_analysis.py",
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "equal_budget_retrained_structural_ablation",
        "expected_rows_per_variant": expected_rows,
        "evaluation_seed": seed,
        "minimum_training_steps": minimum_steps,
        "reward_changed": False,
        "runtime_reward_fingerprint": reward_fingerprint(),
        "strictly_paired": True,
        "control_rows_selected_runtime_variant": "full",
        "training_configuration_equal_except_variant": True,
        "graph_model_configuration_equal": True,
        "optimization_update_counts_equal": True,
        "checkpoints": checkpoints,
        "evaluation_provenance": provenance,
        "auxiliary_reports": auxiliary_reports,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "summary_csv": str(summary_path.resolve()),
        "summary_csv_sha256": sha256(summary_path),
        "statistics": statistics,
        "passed": True,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root", default="artifacts/paper_evaluation", type=Path
    )
    parser.add_argument(
        "--output",
        default="artifacts/paper_evaluation/structural_ablation_audit.json",
        type=Path,
    )
    parser.add_argument("--expected-rows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--minimum-steps", type=int, default=300_000)
    arguments = parser.parse_args()
    evaluation_root = (
        arguments.evaluation_root
        if arguments.evaluation_root.is_absolute()
        else ROOT / arguments.evaluation_root
    )
    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    print(
        audit(
            evaluation_root,
            output,
            arguments.expected_rows,
            arguments.seed,
            arguments.minimum_steps,
        )
    )


if __name__ == "__main__":
    main()

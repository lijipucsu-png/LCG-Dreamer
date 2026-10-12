import json
import csv

import pytest
import torch

from scripts.audit_structural_ablation import (
    _holm_adjust,
    _normalized_model_config,
    _optimizer_updates,
    _read_rows,
    _validate_auxiliary_report,
)


def test_holm_adjustment_preserves_ordered_monotonicity() -> None:
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.20})


def test_model_config_normalization_fills_current_defaults() -> None:
    normalized = _normalized_model_config({"hidden_dim": 64})

    assert normalized["hidden_dim"] == 64
    assert normalized["prediction_horizon"] == 30


def test_optimizer_update_accounting_accepts_tensor_steps() -> None:
    trainer = {
        "optimizers": {
            "world": {
                "state": {
                    0: {"step": torch.tensor(12.0)},
                    1: {"step": torch.tensor(9.0)},
                }
            }
        }
    }

    assert _optimizer_updates(trainer, "world") == 12
    assert _optimizer_updates(trainer, "actor") == 0


def test_auxiliary_report_is_bound_to_checkpoint_and_replay(tmp_path) -> None:
    checkpoint = tmp_path / "graph_complete.pt"
    replay = tmp_path / "graph_replay_latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    replay.write_bytes(b"replay")
    report = tmp_path / "graph_auxiliary.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "replay": str(replay),
                "batches": 20,
                "summary": {
                    "delay_teacher_normalized_rmse": {
                        "mean": 0.08,
                        "std": 0.01,
                    },
                    "cutin_teacher_brier": {"mean": 0.04, "std": 0.005},
                },
                "passed": False,
            }
        ),
        encoding="utf-8",
    )

    validated = _validate_auxiliary_report(report, checkpoint, replay)

    assert validated["batches"] == 20
    assert not validated["passed"]
    assert validated["metrics"]["cutin_teacher_brier"]["mean"] == 0.04

    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        _validate_auxiliary_report(report, tmp_path / "other.pt", replay)


def test_read_rows_selects_full_from_audited_runtime_superset(tmp_path) -> None:
    path = tmp_path / "episodes.csv"
    fieldnames = [
        "variant",
        "seed",
        "front_vehicle_count",
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
    ]
    rows = []
    for variant in ("full", "no_latent_extrapolation"):
        for seed, front_count in ((30_000, 0), (31_000, 1)):
            row = {name: 1.0 for name in fieldnames}
            row.update(
                variant=variant,
                seed=seed,
                front_vehicle_count=front_count,
            )
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected = _read_rows(
        path,
        expected_rows=2,
        expected_total_rows=4,
        expected_variants=("full", "no_latent_extrapolation"),
    )

    assert set(selected) == {(30_000, 0), (31_000, 1)}

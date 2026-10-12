"""Validate fairness and aggregate the paired cooperative lane-change study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .protocol import DISPLAY_NAMES, METHODS, LaneChangeProtocol


METRICS = (
    "success",
    "collision",
    "travel_time_s",
    "maneuver_time_s",
    "lane_execution_time_s",
    "maximum_speed_difference_mps",
    "jerk_rms_mps3",
    "minimum_gap_m",
    "maximum_gap_m",
    "average_speed_mps",
    "fuel_kg",
)


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def _values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray([float(row[metric]) for row in rows], np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {metric}")
    return values


def _paired_bootstrap(
    graph: np.ndarray, baseline: np.ndarray, seed: int
) -> dict[str, float]:
    difference = baseline - graph
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(difference), size=(20_000, len(difference)))
    means = difference[samples].mean(1)
    nonzero = difference[np.abs(difference) > 1e-12]
    wins = int(np.sum(nonzero > 0.0))
    losses = int(np.sum(nonzero < 0.0))
    count = wins + losses
    # Exact two-sided sign-test tail without a SciPy dependency.
    tail = min(wins, losses)
    p_value = min(
        1.0,
        2.0
        * sum(math.comb(count, index) for index in range(tail + 1))
        / (2.0**count),
    ) if count else 1.0
    return {
        "baseline_minus_graph_mean": float(difference.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "graph_better_pairs": wins,
        "graph_worse_pairs": losses,
        "two_sided_sign_test_p": float(p_value),
    }


def build_report(protocol: LaneChangeProtocol | None = None) -> Path:
    protocol = protocol or LaneChangeProtocol()
    evaluation = Path(protocol.output_root) / "evaluation"
    frames = {
        method: _read(evaluation / f"{method}_rows.csv")
        for method in METHODS
    }
    expected = protocol.evaluation_episodes
    seeds = [int(row["seed"]) for row in frames["graph"]]
    if len(seeds) != expected or len(set(seeds)) != expected:
        raise RuntimeError("Graph evaluation seeds are incomplete or repeated")
    for method, rows in frames.items():
        if len(rows) != expected:
            raise RuntimeError(f"{method} has {len(rows)} rows, expected {expected}")
        if [int(row["seed"]) for row in rows] != seeds:
            raise RuntimeError(f"{method} evaluation seeds are not paired")
        for metric in METRICS:
            _values(rows, metric)

    training: dict[str, Any] = {}
    budget_pairs = set()
    parameter_counts = set()
    for method in METHODS:
        checkpoint = torch.load(
            protocol.adapter_checkpoint(method),
            map_location="cpu",
            weights_only=False,
        )
        budget = (
            int(checkpoint["protocol"]["imitation_episodes"]),
            int(checkpoint["protocol"]["training_episodes"]),
        )
        budget_pairs.add(budget)
        parameter_counts.add(int(checkpoint["trained_parameters"]))
        training[method] = {
            "imitation_episodes": budget[0],
            "ppo_episodes": budget[1],
            "trained_parameters": int(checkpoint["trained_parameters"]),
            "base_controller_frozen": bool(checkpoint["base_controller_frozen"]),
            "base_checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
        }
    fairness_passed = (
        budget_pairs == {(1, 8)}
        and len(parameter_counts) == 1
        and all(item["base_controller_frozen"] for item in training.values())
    )
    if not fairness_passed:
        raise RuntimeError("lane-change training fairness gate failed")

    summary: dict[str, Any] = {}
    for method, rows in frames.items():
        summary[method] = {
            metric: {
                "mean": float(_values(rows, metric).mean()),
                "population_sd": float(_values(rows, metric).std()),
            }
            for metric in METRICS
        }
    graph_msd = _values(frames["graph"], "maximum_speed_difference_mps")
    paired = {
        method: _paired_bootstrap(
            graph_msd,
            _values(frames[method], "maximum_speed_difference_mps"),
            protocol.seed + index,
        )
        for index, method in enumerate(METHODS[1:], start=1)
    }
    report = {
        "protocol": protocol.to_dict(),
        "fairness": {
            "passed": fairness_passed,
            "paired_seeds": seeds,
            "identical_adapter_parameter_count": next(iter(parameter_counts)),
            "identical_training_budget": {"imitation_episodes": 1, "ppo_episodes": 8},
            "method_specific_reward_tuning": False,
        },
        "training": training,
        "summary": summary,
        "paired_maximum_speed_difference": paired,
    }
    output = evaluation / "comparison_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    summary_path = evaluation / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "display_name", "metric", "mean", "population_sd"])
        for method in METHODS:
            for metric in METRICS:
                item = summary[method][metric]
                writer.writerow(
                    [method, DISPLAY_NAMES[method], metric, item["mean"], item["population_sd"]]
                )
    return output


if __name__ == "__main__":
    print(json.dumps({"report": str(build_report()), "complete": True}))

"""Fairness validation and paired statistics for the mixed-lane study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .protocol import DISPLAY_NAMES, LEARNED_METHODS, METHODS, MixedLaneProtocol


METRICS = (
    "success",
    "collision",
    "completion_ratio",
    "travel_time_s",
    "maneuver_time_s",
    "gap_acceptance_wait_s",
    "platoon_average_speed_mps",
    "maximum_speed_difference_mps",
    "jerk_rms_mps3",
    "minimum_same_lane_gap_m",
    "minimum_same_lane_ttc_s",
    "fuel_kg",
    "traffic_delay_s",
    "traffic_braking_integral",
    "traffic_hard_brake_events",
    "mean_lane_command_spread_s",
)

HIGHER_IS_BETTER = {
    "success",
    "completion_ratio",
    "platoon_average_speed_mps",
    "minimum_same_lane_gap_m",
    "minimum_same_lane_ttc_s",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _values(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    values = np.asarray([float(row[metric]) for row in rows], np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite evaluation metric: {metric}")
    return values


def _paired_effect(
    graph: np.ndarray,
    baseline: np.ndarray,
    higher_is_better: bool,
    seed: int,
) -> dict[str, Any]:
    improvement = graph - baseline if higher_is_better else baseline - graph
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(
        0, len(improvement), size=(20_000, len(improvement))
    )
    bootstrap_means = improvement[sample_index].mean(axis=1)
    nonzero = improvement[np.abs(improvement) > 1.0e-12]
    wins = int(np.sum(nonzero > 0.0))
    losses = int(np.sum(nonzero < 0.0))
    count = wins + losses
    tail = min(wins, losses)
    sign_p = (
        min(
            1.0,
            2.0
            * sum(math.comb(count, index) for index in range(tail + 1))
            / (2.0**count),
        )
        if count
        else 1.0
    )
    return {
        "positive_favors_graph": True,
        "paired_mean_improvement": float(improvement.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "graph_better_pairs": wins,
        "graph_worse_pairs": losses,
        "ties": int(len(improvement) - count),
        "two_sided_sign_test_p": float(sign_p),
    }


def _training_metadata(protocol: MixedLaneProtocol) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "cacc_lc": {
            "trained": False,
            "lateral_controller": (
                "independent per-vehicle obstacle, target-gap, and TTC rule"
            ),
            "team_lateral_planner": False,
        }
    }
    fingerprints = set()
    budgets = set()
    base_hashes: dict[str, str] = {}
    predictor_signatures = set()
    for method in LEARNED_METHODS:
        selected_path = protocol.lateral_checkpoint(method)
        assert selected_path is not None
        latest_path = selected_path.with_name("latest.pt")
        selected = torch.load(
            selected_path, map_location="cpu", weights_only=False
        )
        latest = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        fingerprint = latest["protocol"]["fingerprint"]
        fingerprints.add(fingerprint)
        budget = (
            int(latest["protocol"]["imitation_episodes"]),
            int(latest["protocol"]["training_episodes"]),
        )
        budgets.add(budget)
        expected_base = protocol.base_checkpoint(method)
        assert expected_base is not None
        if Path(latest["base_checkpoint"]) != expected_base:
            raise RuntimeError(f"{method} used an unexpected base controller")
        base_hashes[method] = latest["base_checkpoint_sha256"]
        if method != "graph":
            predictive_protocol_path = expected_base.parent / "protocol.json"
            predictive_protocol = json.loads(
                predictive_protocol_path.read_text(encoding="utf-8")
            )
            predictor_signatures.add(
                (
                    predictive_protocol["graph_checkpoint"],
                    predictive_protocol["predictor_directory"],
                    predictive_protocol["prediction_schema"],
                    predictive_protocol["prediction_checkpoint_sha256"],
                    predictive_protocol["trajectory_checkpoint_sha256"],
                    predictive_protocol["reward_fingerprint"],
                )
            )
        if int(latest["episode"]) != protocol.training_episodes:
            raise RuntimeError(f"{method} training did not consume the fixed budget")
        if selected["metrics"].get("selection_timing") != (
            "fixed_three_seed_deterministic_after_update"
        ):
            raise RuntimeError(f"{method} lacks deterministic deployment selection")
        expected_selection_seeds = [
            protocol.seed + offset for offset in protocol.selection_seed_offsets
        ]
        if selected["metrics"].get("selection_seeds") != expected_selection_seeds:
            raise RuntimeError(f"{method} used different selection seeds")
        if int(selected["metrics"].get("validation_episode_count", -1)) != len(
            expected_selection_seeds
        ):
            raise RuntimeError(f"{method} has incomplete selection validation")
        selected_episode = int(selected["episode"])
        if selected_episode != 0 and (
            selected_episode % protocol.selection_interval_episodes != 0
        ):
            raise RuntimeError(f"{method} selected an unregistered candidate")
        metadata[method] = {
            "trained": True,
            "algorithm": latest["lateral_algorithm"],
            "completed_episode": int(latest["episode"]),
            "selected_episode": int(selected["episode"]),
            "selection_timing": selected["metrics"]["selection_timing"],
            "selection_seeds": selected["metrics"]["selection_seeds"],
            "selection_success_rate": float(
                selected["metrics"]["success_rate"]
            ),
            "budget": list(budget),
            "trained_parameters": int(latest["trained_parameters"]),
            "base_controller_frozen": bool(latest["base_controller_frozen"]),
            "base_checkpoint": latest["base_checkpoint"],
            "base_checkpoint_sha256": latest["base_checkpoint_sha256"],
            "lateral_policy_shared": bool(latest["lateral_policy_shared"]),
            "imitation_prior_initial_weight": float(
                latest["protocol"]["imitation_prior_initial_weight"]
            ),
            "imitation_prior_decay_episodes": float(
                latest["protocol"]["imitation_prior_decay_episodes"]
            ),
        }
    expected_fingerprint = protocol.to_dict()["fingerprint"]
    if fingerprints != {expected_fingerprint}:
        raise RuntimeError("learned methods do not share the frozen protocol")
    if budgets != {(protocol.imitation_episodes, protocol.training_episodes)}:
        raise RuntimeError("learned methods do not share the fixed budget")
    if not all(
        metadata[method]["base_controller_frozen"] for method in LEARNED_METHODS
    ):
        raise RuntimeError("at least one comparison changed its base controller")
    if any(metadata[method]["lateral_policy_shared"] for method in LEARNED_METHODS):
        raise RuntimeError("a learned baseline incorrectly shares lateral weights")
    if len(set(base_hashes.values())) != len(LEARNED_METHODS):
        raise RuntimeError("two learned methods incorrectly share a controller checkpoint")
    if len(predictor_signatures) != 1:
        raise RuntimeError("predictive baselines do not use equivalent predictor assets")
    metadata["controller_isolation"] = {
        "algorithm_specific_base_checkpoints": True,
        "distinct_base_checkpoint_hashes": base_hashes,
        "predictive_baseline_assets_equivalent": True,
        "graph_atomic_team_admission_only": True,
    }
    return metadata


def build_report(protocol: MixedLaneProtocol | None = None) -> Path:
    protocol = protocol or MixedLaneProtocol()
    evaluation = Path(protocol.output_root) / "evaluation"
    frames = {
        method: _read(evaluation / f"{method}_rows.csv") for method in METHODS
    }
    expected_seeds = [
        protocol.seed + 100_000 + index
        for index in range(protocol.evaluation_episodes)
    ]
    for method, rows in frames.items():
        seeds = [int(row["seed"]) for row in rows]
        if seeds != expected_seeds:
            raise RuntimeError(f"{method} does not use all paired evaluation seeds")
        for metric in METRICS:
            _values(rows, metric)

    training = _training_metadata(protocol)
    intensities = {
        method: _values(rows, "traffic_intensity")
        for method, rows in frames.items()
    }
    reference_intensity = intensities["graph"]
    if any(
        not np.array_equal(values, reference_intensity)
        for values in intensities.values()
    ):
        raise RuntimeError("paired scenarios use different traffic intensities")
    representative_episode = int(np.argmax(reference_intensity))
    representative_seed = expected_seeds[representative_episode]
    representative_options = protocol.scenario_options(
        representative_seed, training=False
    )
    summary = {
        method: {
            metric: {
                "mean": float(_values(rows, metric).mean()),
                "population_sd": float(_values(rows, metric).std()),
            }
            for metric in METRICS
        }
        for method, rows in frames.items()
    }
    paired = {
        metric: {
            method: _paired_effect(
                _values(frames["graph"], metric),
                _values(frames[method], metric),
                metric in HIGHER_IS_BETTER,
                protocol.seed + 1000 * metric_index + method_index,
            )
            for method_index, method in enumerate(METHODS[1:], start=1)
        }
        for metric_index, metric in enumerate(METRICS)
    }
    report = {
        "protocol": protocol.to_dict(),
        "fairness": {
            "passed": True,
            "paired_seeds": expected_seeds,
            "common_reward": True,
            "method_specific_reward_tuning": False,
            "fixed_learning_budget": [
                protocol.imitation_episodes,
                protocol.training_episodes,
            ],
            "cacc_is_independent_lc_not_team_planning": True,
            "learned_lateral_parameters_are_not_shared": True,
            "controller_checkpoints_are_algorithm_specific": True,
            "predictor_assets_are_equated_not_controllers": True,
            "graph_coordination_is_not_shared": True,
        },
        "representative_trace": {
            "selection_rule": protocol.to_dict()["representative_trace_rule"],
            "episode": representative_episode,
            "scenario_label": f"S{representative_episode + 1:02d}",
            "seed": representative_seed,
            "traffic_intensity": float(
                reference_intensity[representative_episode]
            ),
            "minimum_target_front_gap_m": float(
                min(representative_options["lane_change_traffic_front_gaps_m"])
            ),
            "minimum_target_rear_gap_m": float(
                min(representative_options["lane_change_traffic_rear_gaps_m"])
            ),
            "maximum_target_rear_speed_mps": float(
                max(
                    representative_options[
                        "lane_change_traffic_rear_speeds_mps"
                    ]
                )
            ),
            "communication_delay_probability": float(
                representative_options["communication_delay_probability"]
            ),
            "communication_delay_range_s": [
                float(representative_options["communication_delay_min"]),
                float(representative_options["communication_delay_max"]),
            ],
            "graph_success": bool(
                int(float(frames["graph"][representative_episode]["success"]))
            ),
        },
        "training": training,
        "summary": summary,
        "paired_effects": paired,
    }
    output = evaluation / "comparison_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_path = evaluation / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "display_name", "metric", "mean", "population_sd"]
        )
        for method in METHODS:
            for metric in METRICS:
                item = summary[method][metric]
                writer.writerow(
                    [
                        method,
                        DISPLAY_NAMES[method],
                        metric,
                        item["mean"],
                        item["population_sd"],
                    ]
                )
    return output


if __name__ == "__main__":
    print(json.dumps({"report": str(build_report()), "complete": True}))

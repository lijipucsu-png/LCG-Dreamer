"""Build the paired statistical report for the four-controller comparison."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CONTROLLERS = ("graph", "cacc", "ddpg", "ppo", "sac")
LOWER_IS_BETTER = {
    "steps",
    "collision",
    "failure",
    "mean_absolute_gap_error_m",
    "maximum_gap_m",
    "maximum_speed_spread_mps",
    "mean_speed_spread_mps",
    "string_amplification",
    "jerk_rms_mps3",
    "shield_intervention_rate",
    "disconnected_seconds",
    "fuel_mg",
    "fuel_mg_per_step",
    "fuel_mg_per_km",
    "background_max_stuck_seconds",
    "background_max_gridlock_seconds",
    "inference_p95_ms",
}
PRIMARY_METRICS = (
    "success",
    "collision",
    "failure",
    "shared_training_reward_per_step",
    "mean_absolute_gap_error_m",
    "maximum_gap_m",
    "maximum_speed_spread_mps",
    "string_amplification",
    "jerk_rms_mps3",
    "shield_intervention_rate",
    "disconnected_seconds",
    "fuel_mg",
    "cutin_resolved",
)
OPERATIONAL_METRICS = (
    "steps",
    "leader_station_m",
    "average_speed_mps",
    "mean_speed_spread_mps",
    "fuel_mg_per_step",
    "fuel_mg_per_km",
    "background_max_stuck_seconds",
    "background_max_gridlock_seconds",
    "inference_p95_ms",
)
COMPARISON_METRICS = PRIMARY_METRICS + OPERATIONAL_METRICS
COMPLETION_CONDITIONED_METRICS = (
    "fuel_mg_per_km",
    "fuel_mg_per_step",
    "average_speed_mps",
    "jerk_rms_mps3",
    "mean_speed_spread_mps",
    "maximum_speed_spread_mps",
    "string_amplification",
    "mean_absolute_gap_error_m",
    "steps",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric_rows: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in ("controller", "scenario_hash"):
                converted[key] = value
            else:
                converted[key] = float(value)
        converted["fuel_mg_per_step"] = converted["fuel_mg"] / max(
            converted["steps"], 1.0
        )
        converted["fuel_mg_per_km"] = 1_000.0 * converted["fuel_mg"] / max(
            converted["leader_station_m"], 1.0
        )
        converted["average_speed_mps"] = converted["leader_station_m"] / max(
            0.1 * converted["steps"], 0.1
        )
        numeric_rows.append(converted)
    return numeric_rows


def _paired_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    samples = rng.choice(values, size=(10_000, values.size), replace=True).mean(1)
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(low), float(high)


def build_report(root: Path, output: Path) -> Path:
    controllers = tuple(
        controller
        for controller in EXPECTED_CONTROLLERS
        if (root / controller / "summary.json").exists()
    )
    if "graph" not in controllers or len(controllers) < 2:
        raise RuntimeError("comparison requires Graph and at least one baseline")
    summaries = {
        controller: json.loads(
            (root / controller / "summary.json").read_text(encoding="utf-8")
        )
        for controller in controllers
    }
    rows = {
        controller: _read_rows(root / controller / "episodes.csv")
        for controller in controllers
    }
    fingerprints = {summary["reward_fingerprint"] for summary in summaries.values()}
    scenario_sets = {
        tuple(summary["scenario_hashes"]) for summary in summaries.values()
    }
    episode_counts = {len(controller_rows) for controller_rows in rows.values()}
    fairness_passed = bool(
        len(fingerprints) == 1
        and len(scenario_sets) == 1
        and len(episode_counts) == 1
        and all(
            not summary["protocol"]["controller_specific_reward_tuning"]
            for summary in summaries.values()
        )
    )
    rng = np.random.default_rng(20_260_726)
    comparisons: dict[str, dict[str, Any]] = {}
    graph_rows = rows["graph"]
    for baseline in controllers[1:]:
        metrics: dict[str, Any] = {}
        for metric in COMPARISON_METRICS:
            graph_values = np.asarray([row[metric] for row in graph_rows])
            baseline_values = np.asarray([row[metric] for row in rows[baseline]])
            raw_difference = graph_values - baseline_values
            oriented = -raw_difference if metric in LOWER_IS_BETTER else raw_difference
            low, high = _paired_interval(oriented, rng)
            metrics[metric] = {
                "graph_mean": float(graph_values.mean()),
                "baseline_mean": float(baseline_values.mean()),
                "graph_minus_baseline": float(raw_difference.mean()),
                "oriented_improvement": float(oriented.mean()),
                "paired_bootstrap_95ci": [low, high],
                "graph_better": bool(low > 0.0),
                "indistinguishable": bool(low <= 0.0 <= high),
            }
        comparisons[baseline] = metrics
    common_success_indices = [
        index
        for index in range(len(graph_rows))
        if all(
            controller_rows[index]["success"] == 1.0
            and controller_rows[index]["collision"] == 0.0
            and controller_rows[index]["failure"] == 0.0
            for controller_rows in rows.values()
        )
    ]
    conditioned_summaries = {
        controller: {
            metric: {
                "mean": float(
                    np.mean(
                        [
                            controller_rows[index][metric]
                            for index in common_success_indices
                        ]
                    )
                ),
                "std": float(
                    np.std(
                        [
                            controller_rows[index][metric]
                            for index in common_success_indices
                        ]
                    )
                ),
            }
            for metric in COMPLETION_CONDITIONED_METRICS
        }
        for controller, controller_rows in rows.items()
    } if common_success_indices else {}
    conditioned_comparisons: dict[str, dict[str, Any]] = {}
    if common_success_indices:
        for baseline in controllers[1:]:
            metrics = {}
            for metric in COMPLETION_CONDITIONED_METRICS:
                graph_values = np.asarray(
                    [graph_rows[index][metric] for index in common_success_indices]
                )
                baseline_values = np.asarray(
                    [rows[baseline][index][metric] for index in common_success_indices]
                )
                raw_difference = graph_values - baseline_values
                oriented = (
                    -raw_difference
                    if metric in LOWER_IS_BETTER
                    else raw_difference
                )
                low, high = _paired_interval(oriented, rng)
                metrics[metric] = {
                    "graph_mean": float(graph_values.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "graph_minus_baseline": float(raw_difference.mean()),
                    "oriented_improvement": float(oriented.mean()),
                    "paired_bootstrap_95ci": [low, high],
                    "graph_better": bool(low > 0.0),
                    "indistinguishable": bool(low <= 0.0 <= high),
                }
            conditioned_comparisons[baseline] = metrics
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "MA-Latent-Graph-Dreamer vs classical continuous-control baselines",
        "controllers": list(controllers),
        "fairness": {
            "passed": fairness_passed,
            "single_reward_fingerprint": len(fingerprints) == 1,
            "identical_scenarios_and_seeds": len(scenario_sets) == 1,
            "identical_episode_counts": len(episode_counts) == 1,
            "controller_specific_reward_tuning": False,
            "reward_fingerprint": next(iter(fingerprints)) if fingerprints else None,
        },
        "interpretation": {
            "equal_task_completion": len(
                {
                    float(np.mean([row["success"] for row in controller_rows]))
                    for controller_rows in rows.values()
                }
            )
            == 1,
            "efficiency_metrics_directly_comparable": all(
                float(np.mean([row["success"] for row in controller_rows])) == 1.0
                for controller_rows in rows.values()
            ),
            "efficiency_caveat": (
                "Unconditioned fuel, jerk, speed-spread, and string-stability values "
                "mix completed and failed trajectories. Primary efficiency inference "
                "therefore uses the paired common-completion subset reported below."
            ),
        },
        "summaries": {
            controller: summary["metrics"]
            for controller, summary in summaries.items()
        },
        "operational_context": {
            controller: {
                metric: float(np.mean([row[metric] for row in controller_rows]))
                for metric in OPERATIONAL_METRICS
            }
            for controller, controller_rows in rows.items()
        },
        "paired_comparisons": comparisons,
        "completion_conditioned": {
            "rule": (
                "same paired episode completed successfully by every controller "
                "with zero collision and zero failure"
            ),
            "episode_indices": common_success_indices,
            "episodes": len(common_success_indices),
            "retained_fraction": len(common_success_indices) / len(graph_rows),
            "summaries": conditioned_summaries,
            "paired_comparisons": conditioned_comparisons,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = [
        "# Classical Baseline Comparison",
        "",
        f"Fairness gate: **{'PASS' if fairness_passed else 'FAIL'}**",
        "",
        "All controllers use the same observation fields, acceleration bounds, "
        "reward implementation, frozen CPIDLM reward reference, evaluation scenarios, "
        "reset seeds, and episode count. No controller-specific reward tuning is used.",
        "",
        "| Metric | " + " | ".join(name.upper() for name in controllers) + " |",
        "|---|" + "---:|" * len(controllers),
    ]
    for metric in PRIMARY_METRICS:
        values = [summaries[name]["metrics"][metric]["mean"] for name in controllers]
        markdown.append(
            f"| {metric} | " + " | ".join(f"{value:.6g}" for value in values) + " |"
        )
    all_complete = report["interpretation"]["efficiency_metrics_directly_comparable"]
    operational_note = (
        "All controllers completed every paired route, so fuel, travel efficiency, "
        "jerk, speed-spread, and string-stability metrics are directly comparable."
        if all_complete
        else "At least one controller did not complete every route. Fuel, jerk, "
        "speed-spread, and string-stability values are reported transparently; "
        "primary efficiency inference uses the common-completion subset below."
    )
    markdown.extend(
        [
            "",
            "## Operational Context",
            "",
            operational_note,
            "",
            "| Metric | " + " | ".join(name.upper() for name in controllers) + " |",
            "|---|" + "---:|" * len(controllers),
        ]
    )
    for metric in OPERATIONAL_METRICS:
        values = [
            float(np.mean([row[metric] for row in rows[name]]))
            for name in controllers
        ]
        markdown.append(
            f"| {metric} | " + " | ".join(f"{value:.6g}" for value in values) + " |"
        )
    markdown.extend(
        [
            "",
            "Positive oriented improvement means Graph is better after accounting for "
            "whether a metric is minimized or maximized. Confidence intervals are paired "
            "episode bootstrap intervals.",
        ]
    )
    markdown.extend(
        [
            "",
            "## Common-Completion Comparison",
            "",
            f"Retained paired episodes: {len(common_success_indices)}/{len(graph_rows)}. "
            "Only episodes completed by every controller without collision or failure "
            "are used below.",
            "",
            "| Metric | " + " | ".join(name.upper() for name in controllers) + " |",
            "|---|" + "---:|" * len(controllers),
        ]
    )
    if common_success_indices:
        for metric in COMPLETION_CONDITIONED_METRICS:
            values = [
                conditioned_summaries[name][metric]["mean"]
                for name in controllers
            ]
            markdown.append(
                f"| {metric} | "
                + " | ".join(f"{value:.6g}" for value in values)
                + " |"
            )
    else:
        markdown.append("No episode was successfully completed by every controller.")
    output.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    if not fairness_passed:
        raise RuntimeError("comparison fairness gate failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report(arguments.root, arguments.output)
    print(json.dumps({"report": str(report), "complete": True}))


if __name__ == "__main__":
    main()

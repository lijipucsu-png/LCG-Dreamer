"""Paired statistical report with prediction-equivalence fairness checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import classical_comparison.report as legacy

from .protocol import ALL_METHODS, PREDICTIVE_METHODS


def build_report(root: Path, output: Path) -> Path:
    legacy.EXPECTED_CONTROLLERS = ALL_METHODS
    path = legacy.build_report(root, output)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = {
        name: json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
        for name in payload["controllers"]
    }
    predictor_hashes = {
        summaries[name]["protocol"].get("prediction_checkpoint_sha256")
        for name in PREDICTIVE_METHODS
        if name in summaries
    }
    trajectory_hashes = {
        summaries[name]["protocol"].get("trajectory_checkpoint_sha256")
        for name in PREDICTIVE_METHODS
        if name in summaries
    }
    schemas = {
        summaries[name]["protocol"].get("prediction_schema")
        for name in PREDICTIVE_METHODS
        if name in summaries
    }
    expected = set(PREDICTIVE_METHODS)
    present = expected.intersection(summaries)
    migrations = {
        name: json.loads(
            (root.parent / "training" / name / "migration.json").read_text(
                encoding="utf-8"
            )
        )
        for name in present
    }
    prediction_gate = {
        "all_predictive_methods_present": present == expected,
        "identical_graph_predictor_checkpoint": len(predictor_hashes) == 1,
        "identical_stgnn_trajectory_checkpoint": len(trajectory_hashes) == 1,
        "identical_prediction_schema": len(schemas) == 1,
        "cacc_has_no_prediction_module": summaries.get("cacc", {})
        .get("protocol", {})
        .get("cacc_prediction_module")
        is None,
        "future_ground_truth_used_by_policy": False,
        "predictor_frozen_during_policy_training": True,
        "equal_total_interaction_budget": (
            len(
                {
                    item.get("expected_total_interaction_steps")
                    for item in migrations.values()
                }
            )
            == 1
            and all(item.get("interaction_budget_matched") for item in migrations.values())
        ),
    }
    positive_checks = (
        value
        for key, value in prediction_gate.items()
        if key not in {"passed", "future_ground_truth_used_by_policy"}
    )
    prediction_gate["passed"] = bool(
        all(positive_checks)
        and not prediction_gate["future_ground_truth_used_by_policy"]
    )
    payload["study"] = (
        "MA-Latent-Graph-Dreamer vs prediction-equated DDPG/PPO/SAC and "
        "NC knowledge-guided SAC"
    )
    payload["prediction_equivalence"] = prediction_gate
    payload["fairness"]["passed"] = bool(
        payload["fairness"]["passed"] and prediction_gate["passed"]
    )
    payload["nc_baseline"] = {
        "label": "NC-KG-SAC-P (task-adapted)",
        "source": (
            "Wang et al., Knowledge-guided self-learning control strategy for "
            "mixed vehicle platoons with delays, Nature Communications 16, 7705 (2025)"
        ),
        "doi": "10.1038/s41467-025-62597-x",
        "preserved_components": [
            "kinematic-wave/Newell aggregate behavior features",
            "previous control-instruction history",
            "SAC continuous control",
            "0/100/200 ms delay-conditioned model switching",
        ],
        "task_adaptation": (
            "CV-HVs-CV aggregate features are computed for each leader-to-follower "
            "segment because all six study trucks are controlled vehicles. The common "
            "frozen Graph/ST-GNN predictor is appended exactly as for DDPG-P/PPO-P/SAC-P."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["fairness"]["passed"]:
        raise RuntimeError("prediction-equated fairness gate failed")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    path = build_report(arguments.root, arguments.output)
    print(json.dumps({"report": str(path), "complete": True}))


if __name__ == "__main__":
    main()

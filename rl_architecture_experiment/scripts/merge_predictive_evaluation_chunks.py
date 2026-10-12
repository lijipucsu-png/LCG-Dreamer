"""Merge parallel predictive-evaluation chunks with pairing validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from classical_comparison.common import reward_fingerprint_is_compatible
from classical_comparison.evaluate import write_evaluation_rows
from predictive_comparison.protocol import PredictiveComparisonProtocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--evaluation-seed", type=int, default=30_000)
    arguments = parser.parse_args()

    files = sorted(arguments.chunks.glob("chunk_*/episodes.csv"))
    if not files:
        raise FileNotFoundError(f"no evaluation chunks under {arguments.chunks}")
    combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    combined = combined.sort_values("episode_index").reset_index(drop=True)
    expected_indices = list(range(arguments.episodes))
    if combined["episode_index"].astype(int).tolist() != expected_indices:
        raise RuntimeError("evaluation chunks are incomplete, duplicated, or unordered")
    reference = pd.read_csv(arguments.reference).sort_values("episode_index")
    if reference["episode_index"].astype(int).tolist() != expected_indices:
        raise RuntimeError("reference evaluation does not cover the expected episodes")
    if combined["scenario_hash"].tolist() != reference["scenario_hash"].tolist():
        raise RuntimeError("refined Graph scenarios are not paired with the reference")
    if combined["reset_seed"].astype(int).tolist() != reference[
        "reset_seed"
    ].astype(int).tolist():
        raise RuntimeError("refined Graph reset seeds differ from the reference")

    protocol = PredictiveComparisonProtocol(
        evaluation_episodes=arguments.episodes,
        evaluation_seed=arguments.evaluation_seed,
    )
    summary = write_evaluation_rows(
        "graph",
        arguments.checkpoint,
        arguments.output,
        protocol,
        combined.to_dict(orient="records"),
    )
    reference_summary_path = arguments.reference.with_name("summary.json")
    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    training_fingerprint = reference_summary["reward_fingerprint"]
    runtime_fingerprint = payload["reward_fingerprint"]
    if not reward_fingerprint_is_compatible(training_fingerprint):
        raise RuntimeError("reference reward fingerprint is not reviewed-compatible")
    if not reward_fingerprint_is_compatible(runtime_fingerprint):
        raise RuntimeError("runtime reward fingerprint is not reviewed-compatible")
    payload["runtime_reward_fingerprint"] = runtime_fingerprint
    payload["reward_fingerprint"] = training_fingerprint
    payload["protocol"]["reward_fingerprint"] = training_fingerprint
    payload["reward_fingerprint_compatibility_reviewed"] = True
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.resolve())


if __name__ == "__main__":
    main()

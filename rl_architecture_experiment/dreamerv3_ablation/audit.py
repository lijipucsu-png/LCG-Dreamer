"""Fail-closed provenance audit for the formal DreamerV3 ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .protocol import PROTOCOL, ROOT, VARIANTS
from .train_budgeted import sha256


def run(output_root: Path) -> Path:
    protocol_path = output_root / "protocol.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    # Compare the persisted JSON representation. Dataclass tuples are serialized
    # as JSON arrays, so comparing the raw Python objects would reject an
    # otherwise byte-equivalent protocol solely on tuple-versus-list types.
    current = json.loads(json.dumps(PROTOCOL.manifest(), sort_keys=True))
    if frozen != current:
        raise RuntimeError("frozen protocol no longer matches code, reward, or source hashes")
    checkpoint_rows: list[dict[str, Any]] = []
    replay_hashes: set[str] = set()
    for seed in PROTOCOL.training_seeds:
        for variant in VARIANTS:
            root = output_root / variant.key / f"seed_{seed}"
            manifest = json.loads(
                (root / "budgeted_manifest.json").read_text(encoding="utf-8")
            )
            checkpoint = root / "budgeted_complete.pt"
            expected = {
                "variant": variant.key,
                "seed": seed,
                "world_updates": PROTOCOL.final_world_updates,
                "actor_updates": PROTOCOL.final_actor_updates,
                "critic_updates": PROTOCOL.final_critic_updates,
                "protocol_sha256": frozen["protocol_sha256"],
                "reward_changed": False,
            }
            mismatches = {
                key: {"expected": value, "actual": manifest.get(key)}
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"manifest mismatch at {root}: {mismatches}")
            if manifest["checkpoint_sha256"] != sha256(checkpoint):
                raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = state["graph_model_config"]
            expected_model = variant.model_config()
            named_flags = {
                "team_context_mode": expected_model.team_context_mode,
                "enable_semantic_v2v": expected_model.enable_semantic_v2v,
                "enable_auxiliary_losses": expected_model.enable_auxiliary_losses,
                "enable_control_priors": expected_model.enable_control_priors,
                "dreamerv3_distributional_heads": True,
            }
            if any(model.get(key) != value for key, value in named_flags.items()):
                raise RuntimeError(f"architecture flags mismatch in {checkpoint}")
            trainer = state["trainer"]
            if (
                int(trainer["world_updates"]) != PROTOCOL.final_world_updates
                or int(trainer["actor_updates"]) != PROTOCOL.final_actor_updates
                or int(trainer["critic_updates"]) != PROTOCOL.final_critic_updates
            ):
                raise RuntimeError(f"checkpoint optimizer budget mismatch: {checkpoint}")
            replay_hashes.add(manifest["shared_replay_sha256"])
            checkpoint_rows.append(
                {
                    "variant": variant.key,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "world_updates": trainer["world_updates"],
                    "actor_updates": trainer["actor_updates"],
                    "critic_updates": trainer["critic_updates"],
                }
            )
    if len(replay_hashes) != 1:
        raise RuntimeError(f"multiple replay hashes observed: {sorted(replay_hashes)}")

    evaluation = pd.read_csv(output_root / "evaluation_episodes.csv")
    expected_rows = (
        len(VARIANTS)
        * len(PROTOCOL.training_seeds)
        * PROTOCOL.evaluation_cases_per_training_seed
    )
    if len(evaluation) != expected_rows:
        raise RuntimeError(f"evaluation rows {len(evaluation)} != {expected_rows}")
    for episode_index, group in evaluation.groupby("episode_index"):
        if group.scenario_hash.nunique() != 1:
            raise RuntimeError(f"unpaired scenario hash at episode {episode_index}")
        if len(group) != len(VARIANTS) * len(PROTOCOL.training_seeds):
            raise RuntimeError(f"incomplete paired case at episode {episode_index}")
    mechanism = pd.read_csv(output_root / "mechanism_audit.csv")
    expected_pairs = {(item.key, seed) for item in VARIANTS for seed in PROTOCOL.training_seeds}
    observed_pairs = set(zip(mechanism.variant, mechanism.training_seed))
    if observed_pairs != expected_pairs:
        raise RuntimeError("mechanism audit is missing a variant/seed pair")

    report = {
        "schema_version": 1,
        "study": PROTOCOL.study,
        "passed": True,
        "full_convergence_claim": False,
        "reward_fingerprint": frozen["reward_fingerprint"],
        "source_sha256": frozen["source_sha256"],
        "runtime_support_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "dreamerv3_ablation" / "online_adapt.py",
                ROOT / "dreamerv3_ablation" / "evaluate.py",
                ROOT / "dreamerv3_ablation" / "mechanism_audit.py",
                ROOT / "dreamerv3_ablation" / "report.py",
            )
        },
        "terminal_observation_policy": (
            "reuse the pre-terminal compact observation only when a controlled "
            "agent is removed on a transition with continuation=0"
        ),
        "protocol_sha256": frozen["protocol_sha256"],
        "shared_replay_sha256": next(iter(replay_hashes)),
        "training_runs": checkpoint_rows,
        "evaluation_rows": len(evaluation),
        "mechanism_rows": len(mechanism),
        "strictly_paired": True,
        "optimizer_update_counts_equal": True,
    }
    path = output_root / "formal_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v3",
    )
    arguments = parser.parse_args()
    print(run(arguments.output_root))


if __name__ == "__main__":
    main()

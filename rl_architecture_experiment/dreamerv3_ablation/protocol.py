"""Frozen protocol for the short-budget DreamerV3 architecture ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from classical_comparison.common import reward_fingerprint
from platoon_control.models.graph_dreamer import GraphDreamerConfig


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    team_context_mode: str
    semantic_v2v: bool
    auxiliary_losses: bool
    control_priors: bool
    added_module: str

    def model_config(self) -> GraphDreamerConfig:
        return GraphDreamerConfig(
            team_context_mode=self.team_context_mode,
            enable_semantic_v2v=self.semantic_v2v,
            enable_auxiliary_losses=self.auxiliary_losses,
            enable_control_priors=self.control_priors,
            dreamerv3_distributional_heads=True,
        )


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "a0_dreamerv3",
        "DreamerV3",
        "fixed_order",
        False,
        False,
        False,
        "flat centralized team world model",
    ),
    VariantSpec(
        "a1_graph_perception",
        "DreamerV3 + graph perception",
        "graph",
        False,
        False,
        False,
        "Graph-RSSM structured perception",
    ),
    VariantSpec(
        "a2_latent_v2v",
        "+ latent V2V",
        "graph",
        True,
        False,
        False,
        "packet-age-aware latent semantic V2V",
    ),
    VariantSpec(
        "a3_aux_prediction",
        "+ predictive supervision",
        "graph",
        True,
        True,
        False,
        "CPIDLM/ST-GNN auxiliary prediction supervision",
    ),
    VariantSpec(
        "a4_full",
        "LCG-Dreamer",
        "graph",
        True,
        True,
        True,
        "differentiable safety and formation-control priors",
    ),
)


@dataclass(frozen=True)
class AblationProtocol:
    schema_version: int = 1
    study: str = "short_budget_dreamerv3_incremental_ablation"
    replay_path: str = "artifacts/graph_full_seed42/graph_replay_latest.pt"
    training_seeds: tuple[int, ...] = (17, 29, 43)
    debug_seed: int = 901
    replay_transitions: int = 30_000
    offline_world_updates: int = 200
    offline_actor_updates: int = 100
    online_sumo_steps: int = 1_000
    final_world_updates: int = 240
    final_actor_updates: int = 120
    final_critic_updates: int = 120
    sequence_length: int = 25
    batch_size: int = 8
    imagination_horizon: int = 15
    communication_dropout: float = 0.10
    evaluation_seed: int = 47_000
    evaluation_cases_per_training_seed: int = 6

    def validate(self) -> None:
        if self.final_world_updates <= self.offline_world_updates:
            raise ValueError("final world budget must exceed offline warm-up")
        if self.final_actor_updates <= self.offline_actor_updates:
            raise ValueError("final actor budget must exceed offline warm-up")
        if self.final_actor_updates != self.final_critic_updates:
            raise ValueError("actor and critic budgets must be identical")
        if len(self.training_seeds) < 3:
            raise ValueError("the formal ablation requires at least three seeds")
        _validate_incremental_ladder(VARIANTS)

    def manifest(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        source_files = (
            ROOT / "platoon_control" / "models" / "graph_dreamer.py",
            ROOT / "platoon_control" / "models" / "dreamer_v3_core.py",
            ROOT / "platoon_control" / "graph_rl.py",
            ROOT / "platoon_sumo" / "env.py",
        )
        payload.update(
            reward_fingerprint=reward_fingerprint(),
            source_sha256={
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest().upper()
                for path in source_files
            },
            variants=[asdict(item) for item in VARIANTS],
            fixed_reward=True,
            shared_replay=True,
            exact_optimizer_update_budget=True,
            full_convergence_claim=False,
        )
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
        return payload

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.manifest(), indent=2), encoding="utf-8")
        temporary.replace(path)
        return path


def _validate_incremental_ladder(variants: tuple[VariantSpec, ...]) -> None:
    if tuple(item.key[:2] for item in variants) != ("a0", "a1", "a2", "a3", "a4"):
        raise ValueError("variant keys must form the A0--A4 ladder")
    fields = ("team_context_mode", "semantic_v2v", "auxiliary_losses", "control_priors")
    for previous, current in zip(variants, variants[1:]):
        changes = [field for field in fields if getattr(previous, field) != getattr(current, field)]
        if len(changes) != 1:
            raise ValueError(
                f"{previous.key} -> {current.key} changes {changes}; exactly one module is required"
            )


def variant_by_key(key: str) -> VariantSpec:
    for item in VARIANTS:
        if item.key == key:
            return item
    raise KeyError(key)


PROTOCOL = AblationProtocol()

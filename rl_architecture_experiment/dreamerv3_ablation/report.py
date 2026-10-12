"""Statistics and publication figures for the budgeted architecture ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .protocol import PROTOCOL, ROOT, VARIANTS


COLORS = {
    "a0_dreamerv3": "#111111",
    "a1_graph_perception": "#2166AC",
    "a2_latent_v2v": "#1B9E77",
    "a3_aux_prediction": "#E69F00",
    "a4_full": "#B2182B",
}
LABELS = {item.key: item.label for item in VARIANTS}
PLOT_LABELS = {
    "a0_dreamerv3": "A0: DreamerV3",
    "a1_graph_perception": "A1: A0 + Graph-RSSM",
    "a2_latent_v2v": "A2: A1 + Latent V2V",
    "a3_aux_prediction": "A3: A2 + Predictive supervision",
    "a4_full": "A4: A3 + Control priors (LCG-Dreamer)",
}
HEATMAP_LABELS = {
    "a0_dreamerv3": "A0\nDreamerV3",
    "a1_graph_perception": "A1\nA0 + Graph-RSSM",
    "a2_latent_v2v": "A2\nA1 + Latent V2V",
    "a3_aux_prediction": "A3\nA2 + Predictive supervision",
    "a4_full": "A4\nA3 + Control priors\n(LCG-Dreamer)",
}
PRIMARY_METRICS = {
    "success": ("Success rate", 1),
    "collision": ("Collision rate", -1),
    "shared_training_reward_per_step": ("Team reward", 1),
    "mean_absolute_gap_error_m": ("Gap MAE (m)", -1),
    "maximum_speed_spread_mps": ("Maximum speed spread (m/s)", -1),
    "string_amplification": ("String amplification", -1),
    "jerk_rms_mps3": ("Jerk RMS (m/s$^3$)", -1),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def bootstrap_seed_interval(values: np.ndarray, seed: int = 73) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(10_000, len(values)), replace=True).mean(-1)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


def evaluation_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for metric in PRIMARY_METRICS:
        frame[metric] = pd.to_numeric(frame[metric])
    frame["training_seed"] = pd.to_numeric(frame["training_seed"]).astype(int)
    seed_means = frame.groupby(["variant", "training_seed"])[list(PRIMARY_METRICS)].mean()
    summary_rows: list[dict[str, Any]] = []
    for variant in LABELS:
        row: dict[str, Any] = {"variant": variant, "label": LABELS[variant]}
        values = seed_means.loc[variant]
        for metric in PRIMARY_METRICS:
            vector = values[metric].to_numpy(float)
            low, high = bootstrap_seed_interval(vector)
            row[f"{metric}_mean"] = float(vector.mean())
            row[f"{metric}_std_across_seeds"] = float(vector.std(ddof=1))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)

    indexed = frame.set_index(["variant", "training_seed", "episode_index"])
    effects: list[dict[str, Any]] = []
    for previous, current in zip(VARIANTS, VARIANTS[1:]):
        left = indexed.loc[previous.key]
        right = indexed.loc[current.key]
        if set(left.index) != set(right.index):
            raise RuntimeError(f"unpaired evaluation: {previous.key} vs {current.key}")
        for metric, (label, direction) in PRIMARY_METRICS.items():
            paired = (right[metric] - left[metric]) * direction
            by_seed = paired.groupby(level=0).mean().to_numpy(float)
            low, high = bootstrap_seed_interval(by_seed, seed=91)
            effects.append(
                {
                    "from": previous.key,
                    "to": current.key,
                    "added_module": current.added_module,
                    "metric": metric,
                    "metric_label": label,
                    "beneficial_difference": float(by_seed.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "benefit_direction": "positive_is_better",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(effects)


def load_learning_curves(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in PROTOCOL.training_seeds:
            path = output_root / variant.key / f"seed_{seed}" / "training_metrics.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("phase") != "world":
                    continue
                rows.append(
                    {
                        "variant": variant.key,
                        "seed": seed,
                        "update": int(row["update"]),
                        # Compare only the objective terms shared by every level.
                        # A3/A4 contain additional module-specific losses, so their
                        # optimized total is not on the same numerical scale.
                        "core_loss": float(row["graph/world_reward"])
                        + float(row["graph/world_continue"])
                        + float(row["graph/world_kl"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_learning(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.3))
    for variant in VARIANTS:
        selected = frame[frame.variant == variant.key]
        aggregate = selected.groupby("update").core_loss.agg(["mean", "std"]).reset_index()
        axis.plot(
            aggregate["update"],
            aggregate["mean"],
            color=COLORS[variant.key],
            label=PLOT_LABELS[variant.key],
            linewidth=3.0 if variant.key == "a4_full" else 2.0,
            linestyle="--" if variant.key == "a0_dreamerv3" else "-",
        )
        axis.errorbar(
            aggregate["update"],
            aggregate["mean"],
            yerr=aggregate["std"].fillna(0.0),
            color=COLORS[variant.key],
            linewidth=0.6,
            capsize=2,
            fmt="none",
            alpha=0.75,
        )
    axis.set_xlabel("Shared world-model gradient updates")
    axis.set_ylabel("Shared DreamerV3 core loss")
    axis.set_title("Cumulative A0-A4 architecture ladder")
    axis.set_xlim(0, PROTOCOL.offline_world_updates)
    axis.grid(False)
    axis.legend(frameon=False, ncol=2, labelcolor="black", loc="upper right")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def mechanism_series(frame: pd.DataFrame, test: str, metric: str, variants: list[str]):
    selected = frame[(frame.test == test) & (frame.metric == metric) & frame.variant.isin(variants)].copy()
    selected["value"] = pd.to_numeric(selected["value"])
    return selected.groupby(["variant", "condition"]).value.agg(["mean", "std"]).reset_index()


def plot_graph_mechanisms(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17.4, 8.8))
    variants = ["a0_dreamerv3", "a1_graph_perception"]

    data = mechanism_series(frame, "permutation", "equivariance_nrmse", variants)
    means = [float(data[data.variant == key]["mean"].iloc[0]) for key in variants]
    stds = [float(data[data.variant == key]["std"].iloc[0]) for key in variants]
    axes[0, 0].bar(
        [0, 1], means, yerr=stds, color=[COLORS[key] for key in variants], capsize=4, width=0.58
    )
    axes[0, 0].set_xticks(
        [0, 1], ["A0\nDreamerV3", "A1\nA0 + Graph-RSSM"]
    )
    axes[0, 0].set_ylabel("Permutation equivariance NRMSE")
    axes[0, 0].set_title("(a) A0 vs A1: follower-order perturbation")

    for axis, test, title, xlabel in (
        (axes[0, 1], "team_size", "(b) A0 vs A1: unseen platoon sizes", "Controlled trucks"),
        (axes[0, 2], "physical_edge_dropout", "(c) A0 vs A1: physical-edge loss", "Dropped physical edges"),
    ):
        data = mechanism_series(frame, test, "physical_scaled_rmse", variants)
        for key in variants:
            selected = data[data.variant == key].sort_values("condition", key=lambda x: x.astype(float))
            axis.errorbar(
                selected.condition.astype(float), selected["mean"], yerr=selected["std"],
                color=COLORS[key], marker="o", capsize=3, label=PLOT_LABELS[key]
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Physical-state scaled RMSE")
        axis.set_title(title)
        axis.grid(False)

    data = mechanism_series(
        frame, "packet_age", "delay_scaled_rmse", ["a1_graph_perception", "a2_latent_v2v"]
    )
    for key in ("a1_graph_perception", "a2_latent_v2v"):
        selected = data[data.variant == key].sort_values("condition", key=lambda x: x.astype(float))
        axes[1, 0].errorbar(
            selected.condition.astype(float) * 1000.0,
            selected["mean"], yerr=selected["std"], color=COLORS[key], marker="o",
            capsize=3, label=PLOT_LABELS[key]
        )
    axes[1, 0].set_xlabel("V2V packet age (ms)")
    axes[1, 0].set_ylabel("Delay-trajectory scaled RMSE")
    axes[1, 0].set_title("(d) A1 vs A2: packet-age perturbation")

    prediction_variants = ["a2_latent_v2v", "a3_aux_prediction"]
    for axis, metric, title, ylabel in (
        (
            axes[1, 1],
            "delay_scaled_rmse",
            "(e) A2 vs A3: delay-trajectory prediction",
            "Delay-trajectory scaled RMSE",
        ),
        (
            axes[1, 2],
            "cutin_brier",
            "(f) A2 vs A3: cut-in prediction",
            "Cut-in Brier score",
        ),
    ):
        data = mechanism_series(
            frame, "heldout_prediction", metric, prediction_variants
        )
        means = [
            float(data[data.variant == key]["mean"].iloc[0])
            for key in prediction_variants
        ]
        standard_deviations = [
            float(data[data.variant == key]["std"].iloc[0])
            for key in prediction_variants
        ]
        axis.bar(
            [0, 1],
            means,
            yerr=standard_deviations,
            color=[COLORS[key] for key in prediction_variants],
            capsize=4,
            width=0.58,
        )
        axis.set_xticks(
            [0, 1], ["A2\nA1 + Latent V2V", "A3\nA2 + Prediction"]
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(False)
    for axis in axes.flat:
        axis.tick_params(colors="black")
    axes[0, 2].legend(frameon=False, labelcolor="black")
    axes[1, 0].legend(frameon=False, labelcolor="black")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_incremental_heatmap(summary: pd.DataFrame, path: Path) -> None:
    metric_keys = list(PRIMARY_METRICS)
    raw = summary.set_index("variant")[[f"{key}_mean" for key in metric_keys]].copy()
    raw.columns = metric_keys
    baseline = raw.loc["a0_dreamerv3"]
    scale = raw.std(axis=0).replace(0.0, 1.0)
    direction = pd.Series({key: value[1] for key, value in PRIMARY_METRICS.items()})
    normalized = (raw - baseline) / scale * direction
    figure, axis = plt.subplots(figsize=(13.2, 5.8))
    maximum = max(1.0, float(np.nanmax(np.abs(normalized.to_numpy()))))
    image = axis.imshow(
        normalized.to_numpy(), cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto"
    )
    axis.set_yticks(
        range(len(normalized)), [HEATMAP_LABELS[key] for key in normalized.index]
    )
    axis.set_xticks(
        range(len(metric_keys)), [PRIMARY_METRICS[key][0] for key in metric_keys], rotation=28, ha="right"
    )
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            axis.text(
                column, row, f"{normalized.iloc[row, column]:.2f}", ha="center", va="center",
                fontsize=10.5, color="black"
            )
    axis.set_title(
        "Cumulative A0-A4 ablation relative to A0 DreamerV3\n"
        "Each row contains all preceding modules; positive is better"
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Standardized improvement")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(output_root: Path) -> Path:
    configure_style()
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(output_root / "evaluation_episodes.csv")
    mechanism = pd.read_csv(output_root / "mechanism_audit.csv")
    summary, effects = evaluation_statistics(evaluation)
    summary.to_csv(output_root / "ablation_summary.csv", index=False)
    effects.to_csv(output_root / "incremental_effects.csv", index=False)
    plot_learning(load_learning_curves(output_root), figures / "ablation_learning_curve.png")
    plot_graph_mechanisms(mechanism, figures / "graph_perception_mechanisms.png")
    plot_incremental_heatmap(summary, figures / "ablation_standardized_improvement.png")
    report = {
        "study": PROTOCOL.study,
        "full_convergence_claim": False,
        "training_seeds": list(PROTOCOL.training_seeds),
        "paired_cases_per_seed": PROTOCOL.evaluation_cases_per_training_seed,
        "summary_csv": str((output_root / "ablation_summary.csv").resolve()),
        "incremental_effects_csv": str((output_root / "incremental_effects.csv").resolve()),
        "excluded_endpoint_dependent_metric": (
            "fuel_mg is retained in evaluation_episodes.csv but excluded from "
            "the ablation comparison because unsuccessful controllers have "
            "shorter realized route exposure"
        ),
        "figures": [str(path.resolve()) for path in sorted(figures.glob("*.png"))],
    }
    path = output_root / "report_manifest.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "dreamerv3_ablation_v1",
    )
    arguments = parser.parse_args()
    print(run(arguments.output_root))


if __name__ == "__main__":
    main()

"""Build tables and figures for the two-layer classical-controller benchmark."""

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


CONTROLLERS = ("graph", "cacc", "ddpg", "ppo", "sac")
LABELS = {
    "graph": "MA-Latent-Graph-Dreamer",
    "cacc": "CACC",
    "ddpg": "DDPG",
    "ppo": "PPO",
    "sac": "SAC",
}
COLORS = {
    "graph": "#13795B",
    "cacc": "#D97706",
    "ddpg": "#2878B5",
    "ppo": "#C44E52",
    "sac": "#6C757D",
}
VEHICLES = ("leader", "follower_1", "follower_2", "follower_3", "follower_4", "follower_5")
FOLLOWERS = VEHICLES[1:]
LOWER_IS_BETTER = (
    "mean_acceleration_damping_ratio",
    "tail_acceleration_damping_ratio",
    "acceleration_mse_m2ps4",
    "jerk_rms_mps3",
    "flow_variation_index",
    "gap_mae_m",
    "maximum_speed_spread_mps",
    "shield_intervention_rate",
    "disconnected_seconds",
)
SUMMARY_METRICS = (
    "horizon_completed",
    "collision",
    "formation_service_valid",
    "formation_connected_ratio",
    "tail_mobility_ratio",
    "average_speed_mps",
    "mean_acceleration_damping_ratio",
    "jerk_rms_mps3",
    "gap_mae_m",
    "flow_variation_index",
    "tractive_energy_kwh",
    "fuel_g",
    "leader_tracking_rmse_mps",
)


def _load(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    episodes = {
        name: pd.read_csv(root / name / "episodes.csv") for name in CONTROLLERS
    }
    vehicles = {
        name: pd.read_csv(root / name / "per_vehicle_metrics.csv")
        for name in CONTROLLERS
    }
    traces = {
        name: pd.read_csv(root / name / "traces.csv") for name in CONTROLLERS
    }
    summaries = {
        name: json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
        for name in CONTROLLERS
    }
    profile_hashes = {value["profiles_sha256"] for value in summaries.values()}
    reward_hashes = {value["reward_fingerprint"] for value in summaries.values()}
    profile_ids = {
        tuple(frame["profile_id"].astype(int)) for frame in episodes.values()
    }
    fairness = {
        "passed": len(profile_hashes) == len(reward_hashes) == len(profile_ids) == 1,
        "identical_ngsim_profiles": len(profile_hashes) == 1,
        "identical_reward_fingerprint": len(reward_hashes) == 1,
        "identical_profile_order": len(profile_ids) == 1,
        "controller_specific_reward_tuning": False,
        "leader_override_evaluation_only": True,
        "profiles_sha256": next(iter(profile_hashes)),
        "reward_fingerprint": next(iter(reward_hashes)),
    }
    if not fairness["passed"]:
        raise RuntimeError(f"fixed-horizon fairness gate failed: {fairness}")
    return episodes, vehicles, traces, fairness


def _bootstrap(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    sampled = rng.choice(values, size=(10_000, len(values)), replace=True).mean(1)
    return [float(value) for value in np.percentile(sampled, (2.5, 97.5))]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| Metric | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for index, row in frame.iterrows():
        lines.append(
            f"| {index} | "
            + " | ".join(str(row[column]) for column in frame.columns)
            + " |"
        )
    return "\n".join(lines)


def _paired_report(episodes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    graph = episodes["graph"].sort_values("profile_id")
    rng = np.random.default_rng(20_260_727)
    result: dict[str, Any] = {}
    for baseline in CONTROLLERS[1:]:
        other = episodes[baseline].sort_values("profile_id")
        metrics: dict[str, Any] = {}
        for metric in LOWER_IS_BETTER:
            graph_values = graph[metric].to_numpy(float)
            baseline_values = other[metric].to_numpy(float)
            improvement = baseline_values - graph_values
            metrics[metric] = {
                "graph_mean": float(graph_values.mean()),
                "baseline_mean": float(baseline_values.mean()),
                "oriented_graph_improvement": float(improvement.mean()),
                "paired_bootstrap_95ci": _bootstrap(improvement, rng),
            }
        result[baseline] = metrics
    return result


def _representative_profile(episodes: dict[str, pd.DataFrame]) -> int:
    graph = episodes["graph"]
    order = np.argsort(graph["mean_acceleration_damping_ratio"].to_numpy(float))
    return int(graph.iloc[int(order[len(order) // 2])]["profile_id"])


def plot_response_grid(traces: dict[str, pd.DataFrame], profile_id: int, output: Path) -> None:
    fig, axes = plt.subplots(3, len(CONTROLLERS), figsize=(16.2, 8.4), sharex=True)
    vehicle_colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(VEHICLES)))
    for column, controller in enumerate(CONTROLLERS):
        frame = traces[controller].loc[traces[controller]["profile_id"] == profile_id]
        time_s = frame["time_s"]
        axes[0, column].plot(
            time_s,
            frame["target_leader_speed_mps"],
            color="black",
            linestyle="--",
            linewidth=1.1,
            label="NGSIM target",
        )
        for color, vehicle in zip(vehicle_colors, VEHICLES):
            axes[0, column].plot(time_s, frame[f"{vehicle}_speed_mps"], color=color, linewidth=1.0)
            axes[1, column].plot(time_s, frame[f"{vehicle}_acceleration_mps2"], color=color, linewidth=1.0)
        for color, follower in zip(vehicle_colors[1:], FOLLOWERS):
            axes[2, column].plot(time_s, frame[f"{follower}_gap_m"], color=color, linewidth=1.0)
        axes[0, column].set_title(LABELS[controller], fontsize=10, color=COLORS[controller], weight="bold")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[2, column].grid(alpha=0.2)
        axes[2, column].axhline(12.0, color="black", linestyle="--", linewidth=0.8)
        axes[2, column].set_xlabel("Time (s)")
    legend_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.1, label="NGSIM target")
    ] + [
        plt.Line2D([0], [0], color=color, linewidth=1.3, label="L" if index == 0 else f"F{index}")
        for index, color in enumerate(vehicle_colors)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=7, fontsize=8, frameon=False)
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[1, 0].set_ylabel("Acceleration (m/s2)")
    axes[2, 0].set_ylabel("Net gap (m)")
    fig.suptitle(f"Common NGSIM leader excitation, paired profile {profile_id}", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.98))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vehicle_metrics(vehicles: dict[str, pd.DataFrame], output: Path) -> None:
    metrics = (
        ("acceleration_damping_ratio", "Acceleration damping ratio"),
        ("acceleration_mse_m2ps4", "Mean squared acceleration (m2/s4)"),
        ("flow_variation_index", "Flow variation index"),
        ("gap_mae_m", "Gap MAE (m)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    x = np.arange(1, 6)
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for controller in CONTROLLERS:
            grouped = vehicles[controller].groupby("vehicle", sort=False)[metric]
            mean = grouped.mean().reindex(FOLLOWERS).to_numpy(float)
            std = grouped.std(ddof=0).reindex(FOLLOWERS).to_numpy(float)
            axis.plot(x, mean, marker="o", linewidth=1.5, color=COLORS[controller], label=LABELS[controller])
            axis.fill_between(x, mean - std, mean + std, color=COLORS[controller], alpha=0.08)
        axis.set_xticks(x, [f"F{i}" for i in x])
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_distributions(episodes: dict[str, pd.DataFrame], output: Path) -> None:
    metrics = (
        ("mean_acceleration_damping_ratio", "Damping ratio"),
        ("jerk_rms_mps3", "Jerk RMS (m/s3)"),
        ("gap_mae_m", "Gap MAE (m)"),
        ("maximum_speed_spread_mps", "Maximum speed spread (m/s)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        data = [episodes[name][metric].to_numpy(float) for name in CONTROLLERS]
        boxes = axis.boxplot(data, patch_artist=True, showfliers=False, widths=0.65)
        for patch, controller in zip(boxes["boxes"], CONTROLLERS):
            patch.set_facecolor(COLORS[controller])
            patch.set_alpha(0.65)
        axis.set_xticks(np.arange(1, 6), [name.upper() if name != "graph" else "Graph" for name in CONTROLLERS], rotation=15)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pip(episodes: dict[str, pd.DataFrame], output: Path) -> None:
    metrics = (
        ("mean_acceleration_damping_ratio", "Damping"),
        ("jerk_rms_mps3", "Jerk"),
        ("gap_mae_m", "Gap error"),
        ("flow_variation_index", "Flow variation"),
    )
    graph = episodes["graph"]
    x = np.arange(len(metrics))
    width = 0.2
    fig, axis = plt.subplots(figsize=(9.6, 4.8))
    for index, controller in enumerate(CONTROLLERS[1:]):
        values = []
        for metric, _ in metrics:
            baseline = float(episodes[controller][metric].mean())
            proposed = float(graph[metric].mean())
            values.append(100.0 * (baseline - proposed) / max(abs(baseline), 1e-9))
        service_valid = float(episodes[controller]["formation_service_valid"].mean()) >= 0.95
        axis.bar(
            x + (index - 1.5) * width,
            values,
            width,
            color=COLORS[controller],
            label=LABELS[controller],
            hatch=None if service_valid else "///",
            edgecolor="white" if not service_valid else COLORS[controller],
            linewidth=0.7,
        )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylabel("Graph performance improvement (%)")
    axis.legend(ncol=4, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.995,
        0.02,
        "Hatched bars: baseline failed the formation-service gate",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_route_survival(route_root: Path, output: Path) -> dict[str, Any]:
    stations = np.linspace(0.0, 3450.0, 140)
    route_summary: dict[str, Any] = {}
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for controller in CONTROLLERS:
        frame = pd.read_csv(route_root / controller / "episodes.csv")
        reached = frame["leader_station_m"].to_numpy(float)
        survival = np.asarray([(reached >= station).mean() for station in stations])
        axis.step(stations, survival, where="post", color=COLORS[controller], linewidth=1.8, label=LABELS[controller])
        route_summary[controller] = {
            "route_success_rate": float(frame["success"].mean()),
            "collision_rate": float(frame["collision"].mean()),
            "failure_rate": float(frame["failure"].mean()),
            "mean_terminal_station_m": float(frame["leader_station_m"].mean()),
            "mean_terminal_time_s": float(frame["steps"].mean() * 0.1),
        }
    axis.set_xlabel("Route station (m)")
    axis.set_ylabel("Fraction of episodes still operational")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return route_summary


def build_report(root: Path, route_root: Path, output: Path) -> Path:
    episodes, vehicles, traces, fairness = _load(root)
    figure_dir = output.parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    representative = _representative_profile(episodes)
    plot_response_grid(traces, representative, figure_dir / "fixed_horizon_responses.png")
    plot_vehicle_metrics(vehicles, figure_dir / "fixed_horizon_vehicle_metrics.png")
    plot_metric_distributions(episodes, figure_dir / "fixed_horizon_boxplots.png")
    plot_pip(episodes, figure_dir / "fixed_horizon_pip.png")
    route = plot_route_survival(route_root, figure_dir / "full_route_survival.png")

    summaries = {
        controller: {
            metric: {
                "mean": float(episodes[controller][metric].mean()),
                "std": float(episodes[controller][metric].std(ddof=0)),
            }
            for metric in SUMMARY_METRICS
        }
        for controller in CONTROLLERS
    }
    report = {
        "study": "two-layer full-route survival and common-leader response benchmark",
        "fairness": fairness,
        "representative_profile_id": representative,
        "fixed_horizon": {
            "summaries": summaries,
            "paired_graph_comparisons": _paired_report(episodes),
            "interpretation": {
                "directly_comparable": True,
                "same_duration_and_leader_excitation": True,
                "energy_service_gate": (
                    "Energy is interpreted as transport efficiency only when formation_service_valid=1; "
                    "low energy from stopped followers is a mobility failure, not an efficiency gain."
                ),
                "flow_variation_orientation": "lower is better",
            },
        },
        "full_route": route,
        "figures": [str(path.name) for path in sorted(figure_dir.glob("*.png"))],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    table = pd.DataFrame(
        {
            LABELS[controller]: {
                metric: f"{values[metric]['mean']:.4g} +/- {values[metric]['std']:.3g}"
                for metric in SUMMARY_METRICS
            }
            for controller, values in summaries.items()
        }
    )
    table.to_csv(output.with_name("fixed_horizon_summary_table.csv"))
    markdown = [
        "# Two-layer controller comparison",
        "",
        f"Fairness gate: **{'PASS' if fairness['passed'] else 'FAIL'}**",
        "",
        "The fixed-horizon layer uses 20 controller-blind NGSIM braking-wave profiles, "
        "the same external leader trajectory, the same duration, and paired reset seeds. "
        "The full-route layer retains early failures as survival outcomes.",
        "",
        _markdown_table(table),
        "",
        "Raw energy for a controller with stopped followers is not labelled as an energy-saving result; "
        "formation service validity and tail mobility are reported alongside it.",
    ]
    output.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    path = build_report(arguments.root, arguments.route_root, arguments.output)
    print(json.dumps({"report": str(path.resolve()), "complete": True}))


if __name__ == "__main__":
    main()

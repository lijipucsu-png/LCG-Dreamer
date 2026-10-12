"""Publication figures for mixed-traffic repeated platoon lane changes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from platoon_sumo.env import PLATOON_IDS

from .protocol import DISPLAY_NAMES, METHODS, MixedLaneProtocol


METHOD_COLORS = {
    "graph": "#C00000",
    "cacc_lc": "#222222",
    "ddpg_p": "#4C78A8",
    "ppo_p": "#59A14F",
    "sac_p": "#B279A2",
    "nc_kg_sac_p": "#E69F00",
}
VEHICLE_COLORS = (
    "#111111",
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], np.float64)


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{stem}.png", bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _decorate(axis: plt.Axes, panel: str) -> None:
    axis.text(
        0.015,
        0.97,
        panel,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        color="black",
    )
    axis.grid(True, linewidth=0.35, color="#D0D0D0")
    axis.tick_params(direction="in", top=True, right=True, colors="black")


def _representative_episode(
    protocol: MixedLaneProtocol,
) -> tuple[int, float, float, float]:
    frames = {
        method: _read(
            Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv"
        )
        for method in METHODS
    }
    reference = _numeric(frames["graph"], "traffic_intensity")
    for method in METHODS[1:]:
        candidate = _numeric(frames[method], "traffic_intensity")
        if not np.array_equal(candidate, reference):
            raise RuntimeError(
                f"{method} does not use the paired traffic intensities"
            )
    episode = int(np.argmax(reference))
    options = protocol.scenario_options(
        protocol.seed + 100_000 + episode, training=False
    )
    return (
        episode,
        float(reference[episode]),
        float(min(options["lane_change_traffic_front_gaps_m"])),
        float(min(options["lane_change_traffic_rear_gaps_m"])),
    )


def plot_trajectories(
    protocol: MixedLaneProtocol,
    output: Path,
    stem: str = "mixed_lane_trajectories",
) -> None:
    episode, intensity, front_gap, rear_gap = _representative_episode(protocol)
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 5.9), sharex=True, sharey=True)
    for panel_index, (axis, method) in enumerate(zip(axes.flat, METHODS)):
        rows = _read(
            Path(protocol.output_root)
            / "evaluation"
            / "traces"
            / f"{method}_episode_{episode:03d}.csv"
        )
        for gate_index in range(1, 4):
            for role in ("front", "front_queue", "rear", "rear_queue"):
                prefix = f"traffic_{gate_index}_{role}"
                axis.plot(
                    _numeric(rows, f"{prefix}_station_m"),
                    _numeric(rows, f"{prefix}_lane"),
                    color="#A8A8A8",
                    linewidth=0.8,
                    alpha=0.85,
                    zorder=1,
                )
        for vehicle_index, agent in enumerate(PLATOON_IDS):
            axis.plot(
                _numeric(rows, f"{agent}_station_m"),
                _numeric(rows, f"{agent}_lane"),
                color=VEHICLE_COLORS[vehicle_index],
                linewidth=2.2 if method == "graph" else 1.35,
                zorder=4,
            )
        axis.scatter(
            protocol.obstacle_stations,
            protocol.obstacle_lanes,
            marker="X",
            s=38,
            color="#B2182B",
            edgecolor="white",
            linewidth=0.45,
            zorder=6,
        )
        axis.set_title(f"({chr(97 + panel_index)}) {DISPLAY_NAMES[method]}", color="black")
        axis.set_xlim(500.0, 1510.0)
        axis.set_ylim(-0.12, 2.12)
        axis.set_yticks((0, 1, 2))
        axis.tick_params(direction="in", top=True, right=True)
        if method == "graph":
            for spine in axis.spines.values():
                spine.set_color(METHOD_COLORS["graph"])
                spine.set_linewidth(1.45)
    figure.supxlabel("Longitudinal station (m)")
    figure.supylabel("Lane index")
    figure.text(
        0.5,
        0.925,
        f"Predeclared hardest paired case: S{episode + 1:02d}, "
        f"intensity = {intensity:.3f}, minimum front/rear gate gaps = "
        f"{front_gap:.1f}/{rear_gap:.1f} m, delay = 15%",
        ha="center",
        color="black",
        fontsize=10,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=VEHICLE_COLORS[index],
            lw=1.8,
            label="Leader" if index == 0 else f"Follower {index}",
        )
        for index in range(6)
    ]
    handles.extend(
        (
            Line2D([0], [0], color="#A8A8A8", lw=1.0, label="Mixed traffic"),
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="None",
                color="#B2182B",
                label="Obstacle",
            ),
        )
    )
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    for item in legend.get_texts():
        item.set_color("black")
    figure.subplots_adjust(
        top=0.84, bottom=0.11, left=0.08, right=0.99, hspace=0.24, wspace=0.10
    )
    _save(figure, output, stem)


def plot_team_dynamics(
    protocol: MixedLaneProtocol,
    output: Path,
    stem: str = "mixed_lane_team_dynamics",
) -> None:
    episode, intensity, front_gap, rear_gap = _representative_episode(protocol)
    fields = (
        ("speed_difference_mps", "Intra-platoon speed difference (m/s)"),
        ("platoon_span_m", "Longitudinal platoon span (m)"),
        ("mean_gap_m", "Mean projected pair gap (m)"),
    )
    figure, axes_grid = plt.subplots(2, 2, figsize=(8.9, 6.1))
    axes = axes_grid.flat
    maximum_time = 0.0
    for method in METHODS:
        rows = _read(
            Path(protocol.output_root)
            / "evaluation"
            / "traces"
            / f"{method}_episode_{episode:03d}.csv"
        )
        time_s = _numeric(rows, "time_s")
        maximum_time = max(maximum_time, float(np.max(time_s)))
        for axis, (field, ylabel) in zip(axes[:3], fields):
            if field == "platoon_span_m":
                stations = np.stack(
                    [
                        _numeric(rows, f"{agent}_station_m")
                        for agent in PLATOON_IDS
                    ],
                    axis=1,
                )
                values = np.ptp(stations, axis=1) + 12.0
            else:
                values = _numeric(rows, field)
            axis.plot(
                time_s,
                values,
                color=METHOD_COLORS[method],
                linewidth=2.7 if method == "graph" else 1.25,
                alpha=1.0 if method == "graph" else 0.78,
                label=DISPLAY_NAMES[method],
                zorder=6 if method == "graph" else 2,
            )
            axis.set_ylabel(ylabel)
    axes[1].set_yscale("log")
    for index, axis in enumerate(axes[:3]):
        axis.set_xlabel("Time (s)")
        axis.set_xlim(0.0, np.ceil(maximum_time / 5.0) * 5.0)
        _decorate(axis, f"({chr(97 + index)})")
    summaries = {
        method: _read(
            Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv"
        )
        for method in METHODS
    }
    jerk_rms = [
        _numeric(summaries[method], "jerk_rms_mps3")
        for method in METHODS
    ]
    means = np.asarray([np.mean(values) for values in jerk_rms])
    ci95 = np.asarray(
        [
            1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
            for values in jerk_rms
        ]
    )
    axes[3].bar(
        np.arange(len(METHODS)),
        means,
        yerr=ci95,
        capsize=2.5,
        width=0.68,
        color=[METHOD_COLORS[method] for method in METHODS],
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "ecolor": "black"},
    )
    axes[3].set_ylabel("Jerk RMS (m/s$^3$)")
    axes[3].set_xticks(
        np.arange(len(METHODS)),
        [DISPLAY_NAMES[method] for method in METHODS],
        rotation=25,
        ha="right",
        color="black",
    )
    axes[3].set_ylim(bottom=0.0)
    _decorate(axes[3], "(d)")
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            lw=2.7 if method == "graph" else 1.35,
            label=DISPLAY_NAMES[method],
        )
        for method in METHODS
    ]
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    for item in legend.get_texts():
        item.set_color("black")
    figure.text(
        0.5,
        0.925,
        f"Predeclared hardest paired case: S{episode + 1:02d}, "
        f"intensity = {intensity:.3f}, minimum front/rear gate gaps = "
        f"{front_gap:.1f}/{rear_gap:.1f} m, delay = 15%",
        ha="center",
        color="black",
        fontsize=10,
    )
    figure.subplots_adjust(
        top=0.84,
        bottom=0.12,
        left=0.10,
        right=0.99,
        hspace=0.36,
        wspace=0.28,
    )
    _save(figure, output, stem)


def plot_overall(
    protocol: MixedLaneProtocol,
    output: Path,
    stem: str = "mixed_lane_overall_performance",
) -> None:
    specifications = (
        ("success", "Success rate (%)", 100.0),
        ("travel_time_s", "Travel time (s)", 1.0),
        ("maximum_speed_difference_mps", "Maximum speed difference (m/s)", 1.0),
        ("jerk_rms_mps3", "Jerk RMS (m/s$^3$)", 1.0),
        ("traffic_delay_s", "Mixed-traffic delay (veh-s)", 1.0),
        ("traffic_hard_brake_events", "Mixed-traffic hard-brake events", 1.0),
    )
    frames = {
        method: _read(
            Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv"
        )
        for method in METHODS
    }
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.0))
    x = np.arange(len(METHODS))
    for panel_index, (axis, (field, ylabel, scale)) in enumerate(
        zip(axes.flat, specifications)
    ):
        samples = [_numeric(frames[method], field) * scale for method in METHODS]
        means = np.asarray([np.mean(values) for values in samples])
        ci95 = np.asarray(
            [1.96 * np.std(values, ddof=1) / np.sqrt(len(values)) for values in samples]
        )
        axis.bar(
            x,
            means,
            yerr=ci95,
            capsize=2.5,
            width=0.68,
            color=[METHOD_COLORS[method] for method in METHODS],
            edgecolor="black",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "ecolor": "black"},
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            x,
            [DISPLAY_NAMES[method] for method in METHODS],
            rotation=25,
            ha="right",
            color="black",
        )
        axis.set_ylim(bottom=0.0)
        _decorate(axis, f"({chr(97 + panel_index)})")
    figure.subplots_adjust(
        top=0.98, bottom=0.15, left=0.09, right=0.99, hspace=0.42, wspace=0.34
    )
    _save(figure, output, stem)


def plot_twenty_scenarios(
    protocol: MixedLaneProtocol,
    output: Path,
    stem: str = "mixed_lane_twenty_scenarios",
) -> None:
    frames = [
        _read(Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv")
        for method in METHODS
    ]
    episodes = min(len(rows) for rows in frames)
    completion = np.asarray(
        [
            [float(row["completion_ratio"]) for row in rows[:episodes]]
            for rows in frames
        ]
    )
    figure, axis = plt.subplots(figsize=(10.2, 3.2))
    image = axis.imshow(
        completion,
        aspect="auto",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    for method_index, rows in enumerate(frames):
        for episode_index, row in enumerate(rows[:episodes]):
            success = bool(int(float(row["success"])))
            label = (
                f"{float(row['travel_time_s']):.0f}"
                if success
                else f"{int(float(row['maneuvers_completed']))}/3"
            )
            axis.text(
                episode_index,
                method_index,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if completion[method_index, episode_index] > 0.65 else "black",
            )
    axis.add_patch(
        mpl.patches.Rectangle(
            (-0.5, -0.5),
            episodes,
            1.0,
            fill=False,
            edgecolor=METHOD_COLORS["graph"],
            linewidth=1.8,
        )
    )
    axis.set_xticks(np.arange(episodes), [f"S{index + 1:02d}" for index in range(episodes)])
    axis.set_yticks(np.arange(len(METHODS)), [DISPLAY_NAMES[method] for method in METHODS])
    axis.set_xlabel("Paired mixed-traffic scenario")
    axis.set_ylabel("Controller")
    axis.tick_params(direction="out", colors="black")
    colorbar = figure.colorbar(image, ax=axis, pad=0.015, fraction=0.025)
    colorbar.set_label("Maneuver completion ratio")
    figure.text(
        0.5,
        0.01,
        "Cell text: travel time (s) for a successful trip; completed maneuvers/3 otherwise.",
        ha="center",
        color="black",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.13, right=0.94, top=0.98, bottom=0.22)
    _save(figure, output, stem)


def plot_all(protocol: MixedLaneProtocol | None = None) -> Path:
    _style()
    protocol = protocol or MixedLaneProtocol()
    output = Path(protocol.output_root) / "figures"
    plot_trajectories(protocol, output)
    plot_team_dynamics(protocol, output)
    plot_overall(protocol, output)
    plot_twenty_scenarios(protocol, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(plot_all())


if __name__ == "__main__":
    main()

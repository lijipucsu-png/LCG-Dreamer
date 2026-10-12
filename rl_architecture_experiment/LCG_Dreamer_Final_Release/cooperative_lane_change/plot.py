"""Publication figures for the matched cooperative lane-change study."""

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

from .protocol import DISPLAY_NAMES, METHODS, LaneChangeProtocol


METHOD_COLORS = {
    "graph": "#D62728",
    "cacc": "#3A3A3A",
    "ddpg_p": "#377EB8",
    "ppo_p": "#4DAF4A",
    "sac_p": "#984EA3",
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
            "lines.linewidth": 1.6,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], np.float64)


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lane_trajectories(
    protocol: LaneChangeProtocol,
    output: Path,
    stem: str = "cooperative_lane_change_trajectories",
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.0, 5.9), sharex=True, sharey=True)
    for panel, (axis, method) in enumerate(zip(axes.flat, METHODS)):
        rows = _read(
            Path(protocol.output_root)
            / "evaluation"
            / "traces"
            / f"{method}_episode_000.csv"
        )
        for vehicle_index, agent in enumerate(PLATOON_IDS):
            axis.plot(
                _numeric(rows, f"{agent}_station_m"),
                _numeric(rows, f"{agent}_lane"),
                color=VEHICLE_COLORS[vehicle_index],
                linewidth=2.1 if method == "graph" else 1.35,
                alpha=1.0 if method == "graph" else 0.88,
            )
        axis.scatter(
            protocol.obstacle_stations,
            protocol.obstacle_lanes,
            marker="X",
            s=42,
            color="#B2182B",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        axis.set_title(f"({chr(97 + panel)}) {DISPLAY_NAMES[method]}", color="black")
        axis.set_xlim(420.0, 1510.0)
        axis.set_ylim(-0.12, 2.12)
        axis.set_yticks((0, 1, 2))
        axis.grid(True, linewidth=0.35, color="#D0D0D0")
        axis.tick_params(direction="in", top=True, right=True)
        if method == "graph":
            for spine in axis.spines.values():
                spine.set_color(METHOD_COLORS[method])
                spine.set_linewidth(1.5)
    figure.supxlabel("Longitudinal station (m)")
    figure.supylabel("Lane index")
    handles = [
        Line2D([0], [0], color=VEHICLE_COLORS[index], lw=1.8, label=("Leader" if index == 0 else f"Follower {index}"))
        for index in range(6)
    ]
    handles.append(
        Line2D([0], [0], marker="X", linestyle="None", color="#B2182B", label="Lane obstacle")
    )
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    for text in legend.get_texts():
        text.set_color("black")
    figure.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.99, hspace=0.23, wspace=0.10)
    _save(figure, output, stem)


def plot_dynamics(
    protocol: LaneChangeProtocol,
    output: Path,
    stem: str = "cooperative_lane_change_dynamics",
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))
    maximum_trace_time = 0.0
    fields = (
        ("formation_length_m", "Platoon length (m)", (115.0, 140.0)),
        ("speed_difference_mps", "Intra-platoon speed difference (m/s)", (0.0, 5.0)),
        ("mean_gap_m", "Mean pair gap (m)", (8.0, 16.0)),
    )
    for method in METHODS:
        rows = _read(
            Path(protocol.output_root)
            / "evaluation"
            / "traces"
            / f"{method}_episode_000.csv"
        )
        time_s = _numeric(rows, "time_s")
        maximum_trace_time = max(maximum_trace_time, float(np.max(time_s)))
        for axis, (field, ylabel, limits) in zip(axes.flat[:3], fields):
            axis.plot(
                time_s,
                _numeric(rows, field),
                color=METHOD_COLORS[method],
                linewidth=2.6 if method == "graph" else 1.25,
                alpha=1.0 if method == "graph" else 0.8,
                label=DISPLAY_NAMES[method],
                zorder=5 if method == "graph" else 2,
            )
            axis.set_ylabel(ylabel)
            axis.set_ylim(*limits)
    summaries = {
        method: _read(
            Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv"
        )
        for method in METHODS
    }
    axis = axes.flat[3]
    x = np.arange(len(METHODS))
    synchronization = np.asarray(
        [
            np.mean(_numeric(summaries[method], "maximum_speed_difference_mps"))
            for method in METHODS
        ]
    )
    synchronization_sd = np.asarray(
        [
            np.std(_numeric(summaries[method], "maximum_speed_difference_mps"))
            for method in METHODS
        ]
    )
    axis.bar(
        x,
        synchronization,
        yerr=synchronization_sd,
        color=[METHOD_COLORS[method] for method in METHODS],
        width=0.66,
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_ylabel("Maximum speed difference (m/s)")
    axis.set_ylim(0.0, 4.0)
    axis.set_xticks(x, [DISPLAY_NAMES[method] for method in METHODS], rotation=25, ha="right", color="black")
    for panel, current in enumerate(axes.flat):
        current.text(0.02, 0.96, f"({chr(97 + panel)})", transform=current.transAxes, ha="left", va="top", fontweight="bold")
        current.grid(True, axis="y", linewidth=0.35, color="#D0D0D0")
        current.tick_params(direction="in", top=True, right=True)
    for current in axes.flat[:3]:
        current.set_xlabel("Time (s)")
        current.set_xlim(0.0, np.ceil(maximum_trace_time / 5.0) * 5.0)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], lw=2.6 if method == "graph" else 1.4, label=DISPLAY_NAMES[method])
        for method in METHODS
    ]
    legend = figure.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.01))
    for text in legend.get_texts():
        text.set_color("black")
    figure.subplots_adjust(top=0.88, bottom=0.12, left=0.11, right=0.98, hspace=0.30, wspace=0.30)
    _save(figure, output, stem)


def plot_twenty_scenarios(
    protocol: LaneChangeProtocol,
    output: Path,
    stem: str = "cooperative_lane_change_20_scenarios",
) -> None:
    rows_by_method = [
        _read(Path(protocol.output_root) / "evaluation" / f"{method}_rows.csv")
        for method in METHODS
    ]
    episodes = min(len(rows) for rows in rows_by_method)
    values = np.full((len(METHODS), episodes), 4.0)
    annotations: list[list[str]] = []
    for method_index, rows in enumerate(rows_by_method):
        labels: list[str] = []
        for episode, row in enumerate(rows[:episodes]):
            if int(float(row["collision"])):
                labels.append("C")
            elif int(float(row["success"])):
                values[method_index, episode] = float(
                    row["maximum_speed_difference_mps"]
                )
                labels.append(
                    f"{float(row['maximum_speed_difference_mps']):.1f}"
                )
            else:
                labels.append(f"{int(float(row['maneuvers_completed']))}/3")
        annotations.append(labels)
    figure, axis = plt.subplots(figsize=(10.0, 3.1))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=4.0,
    )
    for row_index in range(len(METHODS)):
        for column_index in range(episodes):
            axis.text(
                column_index,
                row_index,
                annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    axis.set_xticks(np.arange(episodes), np.arange(1, episodes + 1))
    axis.set_yticks(np.arange(len(METHODS)), [DISPLAY_NAMES[method] for method in METHODS], color="black")
    axis.set_xlabel("Matched test scenario")
    axis.set_ylabel("Control method")
    axis.tick_params(direction="out")
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label(
        "Maximum intra-platoon speed difference (m/s); C = collision; n/3 = incomplete"
    )
    _save(figure, output, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript_trc/figures/lane_change"),
    )
    arguments = parser.parse_args()
    _style()
    protocol = LaneChangeProtocol()
    plot_lane_trajectories(protocol, arguments.output)
    plot_dynamics(protocol, arguments.output)
    plot_twenty_scenarios(protocol, arguments.output)


if __name__ == "__main__":
    main()

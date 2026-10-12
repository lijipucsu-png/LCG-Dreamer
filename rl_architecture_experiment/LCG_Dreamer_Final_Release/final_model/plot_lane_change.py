"""Create an SCI-style six-panel figure from a released lane-change trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


VEHICLES = (
    "leader",
    "follower_1",
    "follower_2",
    "follower_3",
    "follower_4",
    "follower_5",
)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
TRAFFIC_COLOR = "#7A7A7A"


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 600,
        }
    )


def _event_lines(frame: pd.DataFrame) -> list[float]:
    active = frame["event_active"].to_numpy(dtype=float)
    starts = []
    was_active = False
    for time, current in zip(frame["time_s"], active):
        if current > 0.5 and not was_active:
            starts.append(float(time))
        was_active = current > 0.5
    return starts


def plot_lane_change(trace_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(trace_path, encoding="utf-8-sig")
    required = {
        "time_s",
        "event_active",
        "event_index",
        "mean_gap_m",
        "speed_difference_mps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"trace lacks publication fields: {missing}")
    _configure_style()
    time = frame["time_s"].to_numpy(float)
    event_lines = _event_lines(frame)
    figure, axes = plt.subplots(3, 2, figsize=(13.0, 12.0), constrained_layout=True)
    ax_space, ax_lane = axes[0]
    ax_gap, ax_speed = axes[1]
    ax_accel, ax_command = axes[2]

    # Road-space diagram: each truck changes lane continuously in SUMO.
    for agent, color in zip(VEHICLES, COLORS):
        ax_space.plot(
            frame[f"{agent}_station_m"],
            frame[f"{agent}_lane"],
            color=color,
            linewidth=1.9,
            label=agent.replace("_", " "),
        )
    traffic_columns = [
        column[: -len("_station_m")]
        for column in frame.columns
        if column.endswith("_station_m") and column.startswith("lane_change_")
    ]
    for prefix in sorted(set(traffic_columns)):
        lane_column = f"{prefix}_lane"
        if lane_column not in frame:
            continue
        ax_space.plot(
            frame[f"{prefix}_station_m"],
            frame[lane_column],
            color=TRAFFIC_COLOR,
            linewidth=0.65,
            alpha=0.8,
        )
    ax_space.set_title("(a) Road-space trajectories")
    ax_space.set_xlabel("Longitudinal position, $s$ (m)")
    ax_space.set_ylabel("Lane index")
    ax_space.set_yticks((0, 1, 2))
    ax_space.grid(True, linewidth=0.35, alpha=0.35)
    ax_space.legend(
        handles=[
            *[
                Line2D([0], [0], color=color, linewidth=1.9, label=agent.replace("_", " "))
                for agent, color in zip(VEHICLES, COLORS)
            ],
            Line2D([0], [0], color=TRAFFIC_COLOR, linewidth=0.9, label="Gap traffic"),
        ],
        ncol=2,
        frameon=False,
        labelcolor="black",
    )

    for agent, color in zip(VEHICLES, COLORS):
        ax_lane.step(
            time,
            frame[f"{agent}_lane"],
            where="post",
            color=color,
            linewidth=1.7,
        )
    ax_lane.set_title("(b) Lane transition synchronisation")
    ax_lane.set_xlabel("Time, $t$ (s)")
    ax_lane.set_ylabel("Lane index")
    ax_lane.set_yticks((0, 1, 2))
    ax_lane.grid(True, linewidth=0.35, alpha=0.35)
    ax_lane.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=1.7, label=agent.replace("_", " "))
            for agent, color in zip(VEHICLES, COLORS)
        ],
        ncol=2,
        frameon=False,
        labelcolor="black",
    )

    for index, agent in enumerate(VEHICLES[1:]):
        ax_gap.plot(
            time,
            frame[f"{agent}_local_gap_m"]
            if f"{agent}_local_gap_m" in frame
            else frame["mean_gap_m"],
            color=COLORS[index + 1],
            linewidth=1.4,
            label=agent.replace("_", " "),
        )
    target_columns = [f"{agent}_target_gap_m" for agent in VEHICLES[1:]]
    if all(column in frame for column in target_columns):
        target_gap = frame[target_columns].to_numpy(float)
        ax_gap.plot(
            time,
            np.nanmean(target_gap, axis=1),
            color="black",
            linestyle="-.",
            linewidth=1.1,
            label="Graph target gap",
        )
    ax_gap.axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="Minimum safe gap")
    ax_gap.axhline(10.75, color="#444444", linestyle=":", linewidth=1.0, label="Nominal target")
    ax_gap.set_title("(c) Adaptive formation spacing")
    ax_gap.set_xlabel("Time, $t$ (s)")
    ax_gap.set_ylabel("Net gap (m)")
    gap_values = np.concatenate(
        [frame[f"{agent}_local_gap_m"].to_numpy(float) for agent in VEHICLES[1:]]
    )
    ax_gap.set_ylim(0.0, max(18.0, float(np.nanmax(gap_values)) + 2.0))
    ax_gap.grid(True, linewidth=0.35, alpha=0.35)
    ax_gap.legend(ncol=2, frameon=False, labelcolor="black")

    for agent, color in zip(VEHICLES, COLORS):
        ax_speed.plot(
            time,
            frame[f"{agent}_speed_mps"],
            color=color,
            linewidth=1.5,
        )
    ax_speed.set_title("(d) Longitudinal speed during manoeuvres")
    ax_speed.set_xlabel("Time, $t$ (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid(True, linewidth=0.35, alpha=0.35)

    for agent, color in zip(VEHICLES, COLORS):
        ax_accel.plot(
            time,
            frame[f"{agent}_acceleration_mps2"],
            color=color,
            linewidth=1.35,
        )
    ax_accel.axhline(0.0, color="black", linewidth=0.7)
    ax_accel.set_title("(e) Applied acceleration")
    ax_accel.set_xlabel("Time, $t$ (s)")
    ax_accel.set_ylabel("Acceleration (m/s$^2$)")
    ax_accel.grid(True, linewidth=0.35, alpha=0.35)

    vehicle_rows = np.arange(len(VEHICLES), dtype=float)
    vehicle_markers = ("o", "s", "^", "D", "P", "X")
    for row_index, (agent, color, marker) in enumerate(
        zip(VEHICLES, COLORS, vehicle_markers)
    ):
        accepted = frame[f"{agent}_accepted_lane_change"].to_numpy(float)
        requested = frame[f"{agent}_requested_lane_change"].to_numpy(float)
        requested_mask = requested > 0.5
        accepted_mask = accepted > 0.5
        if np.any(requested_mask):
            ax_command.scatter(
                time[requested_mask],
                np.full(int(requested_mask.sum()), vehicle_rows[row_index]),
                facecolors="white",
                edgecolors=color,
                marker=marker,
                s=31,
                linewidth=0.9,
                zorder=3,
            )
        if np.any(accepted_mask):
            ax_command.scatter(
                time[accepted_mask],
                np.full(int(accepted_mask.sum()), vehicle_rows[row_index]),
                facecolors=color,
                edgecolors="black",
                marker=marker,
                s=25,
                linewidth=0.45,
                zorder=4,
            )
    ax_command.set_title("(f) Requested and accepted lane commands")
    ax_command.set_xlabel("Time, $t$ (s)")
    ax_command.set_ylabel("Vehicle")
    ax_command.set_yticks(vehicle_rows)
    ax_command.set_yticklabels([agent.replace("_", " ") for agent in VEHICLES])
    ax_command.set_ylim(-0.7, len(VEHICLES) - 0.3)
    ax_command.grid(True, linewidth=0.35, alpha=0.35)
    ax_command.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="None", markersize=5, label="Requested"),
            Line2D([0], [0], marker="o", markerfacecolor="#555555", markeredgecolor="black", linestyle="None", markersize=5, label="Accepted by SUMO"),
        ],
        frameon=False,
        labelcolor="black",
    )

    for axis in axes.flat:
        for start in event_lines:
            axis.axvline(start, color="#555555", linestyle="--", linewidth=0.7)
        axis.tick_params(width=0.8)
    figure.suptitle(
        "LCG-Dreamer cooperative lane-change execution (seed 173100)",
        fontsize=15,
        color="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    plot_lane_change(arguments.trace, arguments.output)
    print(str(arguments.output.resolve()))


if __name__ == "__main__":
    main()

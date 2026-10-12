"""Capture and compose real SUMO-GUI lane-change interaction storyboards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from mixed_traffic_lane_change.protocol import MixedLaneProtocol
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv

from .runtime import ReleasePaths, load_final_policy


FRAME_WIDTH = 1600
FRAME_HEIGHT = 900
VIEW_SCHEMA = "real world"
PLATOON_COLOR = (20, 95, 190, 255)
LEADER_COLOR = (0, 55, 135, 255)
TRAFFIC_COLOR = (245, 145, 25, 255)
OBSTACLE_COLOR = (220, 45, 35, 255)
PHASES = (
    ("approach", "Approach and gap observation"),
    ("decision", "Team entry decision"),
    ("transition", "Lateral transition"),
    ("reformed", "Formation recovered"),
)


def _wait_for_screenshot(path: Path, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 1_000:
            try:
                image = mpimg.imread(path)
                if image.size:
                    return
            except (OSError, ValueError):
                pass
        time.sleep(0.03)
    raise RuntimeError(f"SUMO screenshot was not produced: {path}")


def _close_gui_environment(env: PlatoonParallelEnv) -> None:
    connection = env.connection
    if connection is None:
        return
    process = getattr(connection, "_process", None)
    try:
        connection.close(wait=False)
    finally:
        env.connection = None
        env.connection_label = None
        env.agents = []
    if process is None or process.poll() is not None:
        return
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _apply_storyboard_colors(env: PlatoonParallelEnv) -> None:
    active = set(env.connection.vehicle.getIDList())
    for agent in PLATOON_IDS:
        if agent in active:
            env.connection.vehicle.setColor(
                agent, LEADER_COLOR if agent == "leader" else PLATOON_COLOR
            )
    for vehicle_id in env._lane_change_traffic_ids:
        if vehicle_id in active:
            env.connection.vehicle.setColor(vehicle_id, TRAFFIC_COLOR)
    for vehicle_id in env._lane_change_obstacle_ids:
        if vehicle_id in active:
            env.connection.vehicle.setColor(vehicle_id, OBSTACLE_COLOR)


def _view_boundary(env: PlatoonParallelEnv) -> tuple[float, float, float, float]:
    active = set(env.connection.vehicle.getIDList())
    platoon_positions = [
        env.connection.vehicle.getPosition(agent)
        for agent in PLATOON_IDS
        if agent in active
    ]
    if not platoon_positions:
        return (-200.0, -120.0, 200.0, 120.0)
    center_x = float(np.mean([position[0] for position in platoon_positions]))
    center_y = float(np.mean([position[1] for position in platoon_positions]))
    # Keep the six-truck formation and its nearest target-lane neighbours large
    # enough to inspect. Distant obstacles are intentionally outside this close
    # interaction view and are documented by the scenario definition instead.
    width = 350.0
    height = 118.0
    local_center_x = center_x + 55.0
    local_center_y = center_y
    return (
        local_center_x - 0.5 * width,
        local_center_y - 0.5 * height,
        local_center_x + 0.5 * width,
        local_center_y + 0.5 * height,
    )


def _event_metadata(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    event_number: int,
    phase: str,
    sim_time: float,
) -> dict[str, Any]:
    leader_mission = np.asarray(observations["leader"]["lane_mission"], np.float32)
    tail_mission = np.asarray(observations["follower_5"]["lane_mission"], np.float32)
    state = np.asarray(observations["leader"]["platoon_state"], np.float32)
    gaps = state[1:, 4]
    valid = gaps[gaps > 0.0]
    speeds = np.asarray(
        [observations[agent]["ego"][1] for agent in PLATOON_IDS], np.float32
    )
    return {
        "event": int(event_number),
        "phase": phase,
        "time_s": float(sim_time),
        "source_lane": int(round(float(observations["leader"]["ego"][4]))),
        "target_lane": int(round(float(leader_mission[2] * 2.0))),
        "target_front_clearance_m": float(leader_mission[6] * 200.0),
        "target_rear_clearance_m": float(tail_mission[7] * 200.0),
        "minimum_platoon_gap_m": float(np.min(valid)) if len(valid) else math.nan,
        "maximum_platoon_gap_m": float(np.max(valid)) if len(valid) else math.nan,
        "speed_spread_mps": float(np.ptp(speeds)),
    }


def _request_screenshot(
    env: PlatoonParallelEnv,
    view_id: str,
    path: Path,
) -> None:
    _apply_storyboard_colors(env)
    env.connection.gui.setSchema(view_id, VIEW_SCHEMA)
    env.connection.gui.trackVehicle(view_id, "follower_2")
    env.connection.gui.setBoundary(view_id, *_view_boundary(env))
    env.connection.gui.screenshot(
        view_id, str(path.resolve()), FRAME_WIDTH, FRAME_HEIGHT
    )


def capture_storyboard(
    output: Path,
    seed: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = ReleasePaths.discover()
    protocol = MixedLaneProtocol()
    options = protocol.scenario_options(seed, training=False)
    policy = load_final_policy(device=device, enable_lane_change=True)
    env = PlatoonParallelEnv(
        render_mode="human",
        max_time=protocol.max_time,
        calibration_path=paths.calibration,
        wave_model_path=paths.wave_model,
        gui_auto_start=True,
    )
    raw = output / "raw_sumo_frames"
    raw.mkdir(parents=True, exist_ok=True)
    captures: list[dict[str, Any]] = []
    captured: set[tuple[int, str]] = set()
    command_times: dict[int, float] = {}
    event_state: dict[int, dict[str, Any]] = {}
    event_context: dict[int, dict[str, Any]] = {}
    completed_pending: tuple[int, dict[str, Any]] | None = None
    approach_pending: tuple[int, float] | None = None
    previous_completed = 0
    final_info: dict[str, Any] = {}
    try:
        observations, _ = env.reset(seed=seed, options=options)
        policy.reset()
        view_id = env.connection.gui.getIDList()[0]
        while env.agents:
            mission = np.asarray(
                observations["leader"]["lane_mission"], np.float32
            )
            event_index = int(mission[1] * 3.0 + 1e-6)
            event_number = min(event_index + 1, 3)
            event_active = bool(mission[8] > 0.5)
            phase: str | None = None
            metadata_override: dict[str, Any] | None = None
            if completed_pending is not None:
                event_number, metadata_override = completed_pending
                phase = "reformed"
                completed_pending = None
            elif (
                approach_pending is not None
                and env.sim_time >= approach_pending[1] - 1e-9
            ):
                event_number = approach_pending[0]
                phase = "approach"
                approach_pending = None
            elif (
                approach_pending is None
                and event_active
                and (event_number, "approach") not in captured
            ):
                phase = "approach"

            actions = policy.act(observations)
            lane_requested = any(
                int(actions[agent].get("lane_change", 0)) != 0
                for agent in PLATOON_IDS
            )
            if lane_requested and (event_number, "decision") not in captured:
                command_times[event_number] = float(env.sim_time)
                phase = "decision"
            command_time = command_times.get(event_number)
            if (
                command_time is not None
                and env.sim_time >= command_time + 0.4 - 1e-9
                and (event_number, "transition") not in captured
            ):
                phase = "transition"

            pending_path: Path | None = None
            pending_meta: dict[str, Any] | None = None
            if phase is not None and (event_number, phase) not in captured:
                pending_meta = _event_metadata(
                    observations, event_number, phase, env.sim_time
                )
                if event_number in event_context:
                    pending_meta.update(event_context[event_number])
                if metadata_override is not None:
                    pending_meta.update(
                        {
                            key: metadata_override[key]
                            for key in (
                                "source_lane",
                                "target_lane",
                                "target_front_clearance_m",
                                "target_rear_clearance_m",
                            )
                        }
                    )
                pending_path = raw / f"event_{event_number}_{phase}.png"
                _request_screenshot(env, view_id, pending_path)

            observations, _, terminations, truncations, infos = env.step(actions)
            policy.record_applied(infos)
            final_info = dict(infos["leader"])
            if pending_path is not None and pending_meta is not None:
                _wait_for_screenshot(pending_path)
                pending_meta["image"] = str(pending_path.resolve())
                captures.append(pending_meta)
                captured.add((event_number, phase or ""))
                event_state[event_number] = pending_meta.copy()
                if phase == "approach":
                    event_context[event_number] = {
                        "source_lane": pending_meta["source_lane"],
                        "target_lane": pending_meta["target_lane"],
                    }
            completed = int(
                final_info.get("lane_change_maneuvers_completed", 0)
            )
            if completed > previous_completed:
                state = event_state.get(completed)
                if state is None:
                    state = _event_metadata(
                        observations, completed, "reformed", env.sim_time
                    )
                if completed in event_context:
                    state = {**state, **event_context[completed]}
                completed_pending = (completed, state)
                previous_completed = completed
            active_after = bool(
                final_info.get("lane_change_event_active", False)
            )
            if active_after and not event_active:
                approach_pending = (
                    int(final_info.get("lane_change_event_index", event_index))
                    + 1,
                    float(env.sim_time) + 0.2,
                )
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        _close_gui_environment(env)
    expected = {(event, phase) for event in range(1, 4) for phase, _ in PHASES}
    actual = {(int(row["event"]), str(row["phase"])) for row in captures}
    missing = sorted(expected.difference(actual))
    if missing:
        raise RuntimeError(f"storyboard phases were not captured: {missing}")
    captures.sort(
        key=lambda row: (
            int(row["event"]),
            [name for name, _ in PHASES].index(str(row["phase"])),
        )
    )
    summary = {
        "seed": seed,
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "maneuvers_completed": int(
            final_info.get("lane_change_maneuvers_completed", 0)
        ),
        "communication_delay_probability": options[
            "communication_delay_probability"
        ],
        "communication_delay_range_s": [
            options["communication_delay_min"],
            options["communication_delay_max"],
        ],
        "captured_frames": len(captures),
    }
    return captures, summary


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 11.5,
            "figure.titlesize": 15,
            "savefig.dpi": 600,
        }
    )


def _crop_interaction_view(image: np.ndarray) -> np.ndarray:
    """Crop empty terrain while retaining the platoon and closest traffic."""
    rgb = image[..., :3]
    blue = (
        (rgb[..., 2] > 0.45)
        & (rgb[..., 0] < 0.35)
        & (rgb[..., 1] < 0.65)
    )
    road = np.max(rgb, axis=2) < 0.28
    blue_y, blue_x = np.where(blue)
    road_rows = np.where(np.sum(road, axis=1) > image.shape[1] * 0.05)[0]
    if not len(blue_x) or not len(road_rows):
        return image
    x_min = max(0, int(np.min(blue_x)) - 220)
    x_max = min(image.shape[1], int(np.max(blue_x)) + 221)
    y_min = max(0, int(np.min(road_rows)) - 75)
    y_max = min(image.shape[0], int(np.max(road_rows)) + 76)
    return image[y_min:y_max, x_min:x_max]


def _draw_panel(axis: Any, snapshot: Mapping[str, Any]) -> None:
    image = _crop_interaction_view(mpimg.imread(snapshot["image"]))
    axis.imshow(image)
    axis.set_axis_off()
    phase_title = dict(PHASES)[str(snapshot["phase"])]
    axis.set_title(
        f"M{int(snapshot['event'])}: {phase_title}", color="black", pad=5
    )
    source = int(snapshot["source_lane"])
    target = int(snapshot["target_lane"])
    annotation = (
        f"$t={float(snapshot['time_s']):.1f}$ s | lane {source} $\\rightarrow$ {target}\n"
        f"target-lane clearance: front {float(snapshot['target_front_clearance_m']):.1f} m, "
        f"rear {float(snapshot['target_rear_clearance_m']):.1f} m"
    )
    axis.text(
        0.015,
        0.025,
        annotation,
        transform=axis.transAxes,
        fontsize=8.5,
        color="black",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.90, "pad": 3.0},
    )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#145FBE", markersize=8, label="LCG-Dreamer platoon"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#F59119", markersize=8, label="Interacting traffic"),
    ]


def compose_storyboards(
    captures: list[dict[str, Any]], output: Path
) -> tuple[Path, Path]:
    _configure_style()
    lookup = {
        (int(row["event"]), str(row["phase"])): row for row in captures
    }
    all_png = output / "graph_sumo_lane_change_storyboard_all.png"
    figure, axes = plt.subplots(3, 4, figsize=(16.0, 6.8), constrained_layout=True)
    for event in range(1, 4):
        for column, (phase, _) in enumerate(PHASES):
            _draw_panel(axes[event - 1, column], lookup[(event, phase)])
    figure.suptitle(
        "LCG-Dreamer interaction with mixed traffic during three consecutive lane changes",
        color="black",
    )
    figure.legend(
        handles=_legend_handles(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        labelcolor="black",
    )
    figure.savefig(all_png, bbox_inches="tight")
    figure.savefig(all_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    main_png = output / "graph_sumo_lane_change_storyboard_main.png"
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 5.8), constrained_layout=True)
    for axis, (phase, _) in zip(axes.flat, PHASES):
        _draw_panel(axis, lookup[(2, phase)])
    figure.suptitle(
        "Representative LCG-Dreamer cooperative lane change in mixed traffic",
        color="black",
    )
    figure.legend(
        handles=_legend_handles(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        labelcolor="black",
    )
    figure.savefig(main_png, bbox_inches="tight")
    figure.savefig(main_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return main_png, all_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_lane_change_sumo"))
    parser.add_argument("--seed", type=int, default=173_100)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    captures, summary = capture_storyboard(
        arguments.output, arguments.seed, arguments.device
    )
    main_png, all_png = compose_storyboards(captures, arguments.output)
    payload = {**summary, "frames": captures, "main_figure": str(main_png.resolve()), "all_figure": str(all_png.resolve())}
    manifest = arguments.output / "storyboard_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"main": str(main_png.resolve()), "all": str(all_png.resolve()), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Capture publication storyboards with continuous SUMO lateral dynamics."""

from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import shutil
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from mixed_traffic_lane_change.protocol import MixedLaneProtocol
from platoon_sumo.env import (
    CONTINUOUS_LATERAL_RESOLUTION,
    CONTINUOUS_PLATOON_LATERAL_ACCELERATION,
    CONTINUOUS_PLATOON_MAX_LATERAL_SPEED,
    PLATOON_IDS,
    PlatoonParallelEnv,
)

from .runtime import ReleasePaths, load_final_policy
from .sumo_lane_change_storyboard import (
    VIEW_SCHEMA,
    _apply_storyboard_colors,
    _close_gui_environment,
    _wait_for_screenshot,
)


CAPTURE_INTERVAL_S = 0.5
PUBLICATION_FRAME_WIDTH = 2400
PUBLICATION_FRAME_HEIGHT = 1350
PUBLICATION_VIEW_WIDTH_M = 180.0
PUBLICATION_VIEW_HEIGHT_M = PUBLICATION_VIEW_WIDTH_M * 9.0 / 16.0
MAIN_PHASES = (
    ("initiation", "Lane-change initiation"),
    ("early_transition", "Early coordinated transition"),
    ("late_transition", "Late coordinated transition"),
    ("completion", "Lane change just completed"),
)
STORYBOARD_PROGRESS_TARGETS = (0.08, 0.40, 0.80, 0.995)
STORYBOARD_PHASE_RANGES = ((0.02, 0.15), (0.25, 0.55), (0.65, 0.90), (0.95, 1.01))
VISUAL_SCENARIO_OPTIONS = {
    # Keep the complete formation inside the fixed three-lane wave segment.
    # This avoids the lane-index remapping at the ramp/merge boundary.
    "initial_leader_station_m": 2480.0,
    "lane_change_obstacle_stations": (2650.0, 3000.0, 3300.0),
    "lane_change_goal_station": 3450.0,
    # Restore calibrated NGSIM traffic at a deliberately demanding density.
    "demand_profile": 4,
    "background_initial_scale": 1.15,
    "background_demand_scale": 1.0,
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 8.6,
            "legend.fontsize": 10.0,
            "savefig.dpi": 600,
        }
    )


def _prepare_continuous_calibration(source: Path, output: Path) -> Path:
    """Create an isolated SL2015 copy of the calibrated NGSIM vehicle types."""

    destination = output / "continuous_calibration"
    destination.mkdir(parents=True, exist_ok=True)
    calibration_path = destination / source.name
    shutil.copy2(source, calibration_path)
    source_vtypes = source.parent / "ngsim_vtypes.add.xml"
    destination_vtypes = destination / source_vtypes.name
    tree = ET.parse(source_vtypes)
    for vehicle_type in tree.getroot().findall("vType"):
        vehicle_type.set("laneChangeModel", "SL2015")
        vehicle_type.set("maxSpeedLat", "1.0")
        vehicle_type.set("lcAccelLat", "0.5")
        vehicle_type.set("minGapLat", "0.3")
        vehicle_type.set("latAlignment", "center")
    tree.write(destination_vtypes, encoding="utf-8", xml_declaration=True)
    return calibration_path


def _view_boundary(env: PlatoonParallelEnv) -> tuple[float, float, float, float]:
    active = set(env.connection.vehicle.getIDList())
    positions = [
        env.connection.vehicle.getPosition(agent)
        for agent in PLATOON_IDS
        if agent in active
    ]
    if not positions:
        return (-200.0, -70.0, 200.0, 70.0)
    center_x = float(np.mean([position[0] for position in positions])) + 18.0
    center_y = float(np.mean([position[1] for position in positions]))
    width = PUBLICATION_VIEW_WIDTH_M
    height = PUBLICATION_VIEW_HEIGHT_M
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _flatten_boundary(boundary: Any) -> list[float]:
    return [
        float(boundary[0][0]),
        float(boundary[0][1]),
        float(boundary[1][0]),
        float(boundary[1][1]),
    ]


def _vehicle_state(env: PlatoonParallelEnv) -> dict[str, dict[str, Any]]:
    active = set(env.connection.vehicle.getIDList())
    relevant = list(PLATOON_IDS) + list(env._lane_change_traffic_ids)
    states: dict[str, dict[str, float | int]] = {}
    for vehicle_id in relevant:
        if vehicle_id not in active:
            continue
        x_position, y_position = env.connection.vehicle.getPosition(vehicle_id)
        lane_index = int(env.connection.vehicle.getLaneIndex(vehicle_id))
        lane_id = env.connection.vehicle.getLaneID(vehicle_id)
        lane_width = float(env.connection.lane.getWidth(lane_id))
        lateral_position = float(
            env.connection.vehicle.getLateralLanePosition(vehicle_id)
        )
        states[vehicle_id] = {
            "x": float(x_position),
            "y": float(y_position),
            "road_id": str(env.connection.vehicle.getRoadID(vehicle_id)),
            "lane_id": str(lane_id),
            "lane": lane_index,
            "lane_width": lane_width,
            "lateral_position": lateral_position,
            "lateral_coordinate": lane_index + lateral_position / lane_width,
            "lateral_speed": float(
                env.connection.vehicle.getLateralSpeed(vehicle_id)
            ),
            "speed": float(env.connection.vehicle.getSpeed(vehicle_id)),
        }
    return states


def _request_frame(
    env: PlatoonParallelEnv,
    view_id: str,
    path: Path,
) -> list[float]:
    _apply_storyboard_colors(env)
    active = set(env.connection.vehicle.getIDList())
    for vehicle_id in active:
        if vehicle_id.startswith("background_"):
            env.connection.vehicle.setColor(vehicle_id, (150, 150, 150, 255))
    env.connection.gui.setSchema(view_id, VIEW_SCHEMA)
    env.connection.gui.trackVehicle(view_id, "follower_2")
    env.connection.gui.setBoundary(view_id, *_view_boundary(env))
    boundary = _flatten_boundary(env.connection.gui.getBoundary(view_id))
    env.connection.gui.screenshot(
        view_id,
        str(path.resolve()),
        PUBLICATION_FRAME_WIDTH,
        PUBLICATION_FRAME_HEIGHT,
    )
    return boundary


def _visible_scene_metrics(
    env: PlatoonParallelEnv, boundary: list[float]
) -> dict[str, Any]:
    x_min, y_min, x_max, y_max = boundary
    visible: dict[str, dict[str, Any]] = {}
    for vehicle_id in env.connection.vehicle.getIDList():
        x_position, y_position = env.connection.vehicle.getPosition(vehicle_id)
        if x_min <= x_position <= x_max and y_min <= y_position <= y_max:
            visible[vehicle_id] = {
                "station": float(env._station(vehicle_id)),
                "road_id": str(env.connection.vehicle.getRoadID(vehicle_id)),
                "lane": int(env.connection.vehicle.getLaneIndex(vehicle_id)),
            }

    closest_pair: list[str] = []
    minimum_net_gap = float("inf")
    for vehicle_id in visible:
        leader = env.connection.vehicle.getLeader(vehicle_id, 200.0)
        if not leader or not leader[0] or leader[0] not in visible:
            continue
        gap = float(leader[1])
        if gap < minimum_net_gap:
            minimum_net_gap = gap
            closest_pair = [vehicle_id, str(leader[0])]

    non_platoon = {
        vehicle_id: state
        for vehicle_id, state in visible.items()
        if vehicle_id not in PLATOON_IDS
    }
    aligned_pairs: list[dict[str, Any]] = []
    for (left_id, left), (right_id, right) in combinations(non_platoon.items(), 2):
        if left["road_id"] != right["road_id"]:
            continue
        if abs(int(left["lane"]) - int(right["lane"])) != 1:
            continue
        longitudinal_offset = abs(float(left["station"]) - float(right["station"]))
        if longitudinal_offset < 2.5:
            aligned_pairs.append(
                {
                    "vehicles": [left_id, right_id],
                    "longitudinal_offset_m": longitudinal_offset,
                    "lanes": [int(left["lane"]), int(right["lane"])],
                }
            )
    return {
        "visible_vehicle_count": len(visible),
        "visible_background_count": sum(
            vehicle_id.startswith("background_") for vehicle_id in visible
        ),
        "visible_minimum_net_gap_m": (
            None if not np.isfinite(minimum_net_gap) else minimum_net_gap
        ),
        "visible_closest_longitudinal_pair": closest_pair,
        "visually_aligned_non_platoon_pairs": aligned_pairs,
        "visual_alignment_count": len(aligned_pairs),
    }


def _context_from_observations(
    observations: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, float | int]:
    mission = np.asarray(observations["leader"]["lane_mission"], np.float32)
    return {
        "source_lane": int(round(float(observations["leader"]["ego"][4]))),
        "target_lane": int(round(float(mission[2] * 2.0))),
        "front_clearance_m": float(mission[6] * 200.0),
        "rear_clearance_m": float(
            observations["follower_5"]["lane_mission"][7] * 200.0
        ),
    }


def capture_continuous_episode(
    output: Path,
    seed: int,
    device: str,
    capture_images: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = ReleasePaths.discover()
    protocol = MixedLaneProtocol()
    options = protocol.scenario_options(seed, training=False)
    options.update(VISUAL_SCENARIO_OPTIONS)
    policy = load_final_policy(device=device, enable_lane_change=True)
    continuous_calibration = _prepare_continuous_calibration(
        paths.calibration, output
    )
    env = PlatoonParallelEnv(
        render_mode="human" if capture_images else None,
        max_time=220.0,
        calibration_path=continuous_calibration,
        wave_model_path=paths.wave_model,
        gui_auto_start=capture_images,
        continuous_lateral_model=True,
        gui_delay_ms=0 if capture_images else 100,
    )
    raw = output / "raw_sumo_frames"
    raw.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    contexts: dict[int, dict[str, float | int]] = {}
    command_times: dict[int, float] = {}
    completion_times: dict[int, float] = {}
    next_capture_time: dict[int, float] = {}
    post_event: int | None = None
    post_until = -1.0
    previous_completed = 0
    previous_lateral_speed = {agent: 0.0 for agent in PLATOON_IDS}
    maximum_lateral_speed = 0.0
    maximum_lateral_acceleration = 0.0
    all_vehicle_collision_events: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    try:
        observations, _ = env.reset(seed=seed, options=options)
        policy.reset()
        view_id = env.connection.gui.getIDList()[0] if capture_images else None
        while env.agents:
            pending_frame_path: Path | None = None
            sim_time = float(env.sim_time)
            mission = np.asarray(
                observations["leader"]["lane_mission"], np.float32
            )
            event_index = int(mission[1] * 3.0 + 1e-6)
            event_number = min(event_index + 1, 3)
            event_active = bool(mission[8] > 0.5)
            if event_active:
                current = _context_from_observations(observations)
                if event_number not in contexts:
                    contexts[event_number] = current
                    next_capture_time[event_number] = sim_time
                else:
                    contexts[event_number].update(
                        {
                            "front_clearance_m": current["front_clearance_m"],
                            "rear_clearance_m": current["rear_clearance_m"],
                        }
                    )

            states = _vehicle_state(env)
            trajectory.append({"time_s": sim_time, "vehicles": states})
            capture_event: int | None = event_number if event_active else None
            if capture_event is None and post_event is not None and sim_time <= post_until:
                capture_event = post_event
            if (
                capture_event is not None
                and capture_event in contexts
                and capture_event in command_times
                and sim_time
                >= next_capture_time.get(capture_event, sim_time) - 1e-9
            ):
                frame_path = raw / f"event_{capture_event}_t_{sim_time:06.1f}.png"
                boundary = (
                    _request_frame(env, str(view_id), frame_path)
                    if capture_images
                    else list(_view_boundary(env))
                )
                scene_metrics = _visible_scene_metrics(env, boundary)
                pending_frame_path = frame_path if capture_images else None
                context = contexts[capture_event]
                frames.append(
                    {
                        "event": capture_event,
                        "time_s": sim_time,
                        "image": str(frame_path.resolve()) if capture_images else "",
                        "boundary": boundary,
                        "source_lane": int(context["source_lane"]),
                        "target_lane": int(context["target_lane"]),
                        "front_clearance_m": float(context["front_clearance_m"]),
                        "rear_clearance_m": float(context["rear_clearance_m"]),
                        **scene_metrics,
                        "lateral_coordinates": {
                            agent: float(states[agent]["lateral_coordinate"])
                            for agent in PLATOON_IDS
                            if agent in states
                        },
                    }
                )
                next_capture_time[capture_event] = sim_time + CAPTURE_INTERVAL_S

            actions = policy.act(observations)
            requested = any(
                int(actions[agent].get("lane_change", 0)) != 0
                for agent in PLATOON_IDS
            )
            if requested and event_number not in command_times:
                command_times[event_number] = sim_time
                next_capture_time[event_number] = sim_time + env.step_length
            observations, _, terminations, truncations, infos = env.step(actions)
            policy.record_applied(infos)
            if pending_frame_path is not None:
                _wait_for_screenshot(pending_frame_path)
            final_info = dict(infos["leader"])
            collision_ids = list(final_info.get("collision_ids", []))
            if collision_ids:
                all_vehicle_collision_events.append(
                    {
                        "time_s": float(env.sim_time),
                        "vehicles": collision_ids,
                        "details": list(final_info.get("collision_details", [])),
                    }
                )
            active = set(env.connection.vehicle.getIDList())
            for agent in PLATOON_IDS:
                if agent not in active:
                    continue
                lateral_speed = float(env.connection.vehicle.getLateralSpeed(agent))
                maximum_lateral_speed = max(
                    maximum_lateral_speed, abs(lateral_speed)
                )
                maximum_lateral_acceleration = max(
                    maximum_lateral_acceleration,
                    abs(lateral_speed - previous_lateral_speed[agent])
                    / env.step_length,
                )
                previous_lateral_speed[agent] = lateral_speed
            completed = int(
                final_info.get("lane_change_maneuvers_completed", 0)
            )
            if completed > previous_completed:
                completion_times[completed] = float(env.sim_time)
                post_event = completed
                post_until = float(env.sim_time) + 0.6
                next_capture_time[completed] = float(env.sim_time)
                previous_completed = completed
            if post_event is not None and float(env.sim_time) > post_until:
                post_event = None
            if previous_completed >= 1 and float(env.sim_time) >= (
                completion_times[1] + 0.8
            ):
                break
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        _close_gui_environment(env)

    if 1 not in command_times or 1 not in completion_times:
        raise RuntimeError(
            "continuous capture did not complete the displayed maneuver: "
            f"commands={command_times}, completions={completion_times}"
        )
    displayed_success = not all_vehicle_collision_events
    summary = {
        "seed": seed,
        "success": displayed_success,
        "scope": "one representative straight-road mixed-traffic maneuver",
        "collision": bool(final_info.get("collision", False)),
        "all_vehicle_collision_free": not all_vehicle_collision_events,
        "all_vehicle_collision_event_count": len(all_vehicle_collision_events),
        "all_vehicle_collision_events": all_vehicle_collision_events,
        "maneuvers_completed": int(
            final_info.get("lane_change_maneuvers_completed", 0)
        ),
        "simulation_end_time_s": float(final_info.get("time", trajectory[-1]["time_s"])),
        "command_times_s": {str(key): value for key, value in command_times.items()},
        "completion_times_s": {
            str(key): value for key, value in completion_times.items()
        },
        "communication_delay_probability": options[
            "communication_delay_probability"
        ],
        "communication_delay_range_s": [
            options["communication_delay_min"],
            options["communication_delay_max"],
        ],
        "communication_delay_events": int(
            final_info.get("communication_delay_events", 0)
        ),
        "continuous_lateral_model": "SL2015",
        "lateral_resolution_m": CONTINUOUS_LATERAL_RESOLUTION,
        "maximum_lateral_speed_limit_mps": CONTINUOUS_PLATOON_MAX_LATERAL_SPEED,
        "maximum_lateral_acceleration_limit_mps2": (
            CONTINUOUS_PLATOON_LATERAL_ACCELERATION
        ),
        "observed_maximum_lateral_speed_mps": maximum_lateral_speed,
        "observed_maximum_lateral_acceleration_mps2": (
            maximum_lateral_acceleration
        ),
        "visual_scenario": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in VISUAL_SCENARIO_OPTIONS.items()
        },
        "trajectory_overlay": "none",
    }
    return trajectory, frames, summary


def _nearest(items: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    return min(items, key=lambda row: abs(float(row["time_s"]) - time_s))


def _progress_frames(
    event_frames: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    command_time: float,
    completion_time: float,
) -> list[dict[str, Any]]:
    base = _nearest(trajectory, command_time)["vehicles"]
    final = _nearest(trajectory, completion_time + 0.3)["vehicles"]
    enriched: list[dict[str, Any]] = []
    for frame in event_frames:
        row = dict(frame)
        progresses = []
        for agent in PLATOON_IDS:
            if agent not in row["lateral_coordinates"] or agent not in base or agent not in final:
                continue
            # The displayed segment is straight, so global y is the physical
            # lateral coordinate. Unlike laneIndex, it cannot jump merely
            # because a ramp lane is inserted or removed.
            start = float(base[agent]["y"])
            end = float(final[agent]["y"])
            denominator = end - start
            if abs(denominator) < 1e-6:
                continue
            progresses.append(
                np.clip(
                    (
                        float(
                            _nearest(
                                trajectory, float(row["time_s"])
                            )["vehicles"][agent]["y"]
                        )
                        - start
                    )
                    / denominator,
                    0.0,
                    1.0,
                )
            )
        row["mean_progress"] = float(np.mean(progresses)) if progresses else 0.0
        row["minimum_progress"] = float(np.min(progresses)) if progresses else 0.0
        row["maximum_progress"] = float(np.max(progresses)) if progresses else 0.0
        row["delta_command_s"] = float(row["time_s"]) - command_time
        enriched.append(row)
    return enriched


def select_storyboard_frames(
    trajectory: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    selected_by_event: dict[int, list[dict[str, Any]]] = {}
    for event in range(1, 2):
        command_time = float(summary["command_times_s"][str(event)])
        completion_time = float(summary["completion_times_s"][str(event)])
        candidates = _progress_frames(
            [row for row in frames if int(row["event"]) == event],
            trajectory,
            command_time,
            completion_time,
        )
        if not candidates:
            raise RuntimeError(f"no SUMO frames captured for maneuver {event}")

        phase_candidates = [
            [
                row
                for row in candidates
                if lower <= float(row["mean_progress"]) <= upper
            ]
            for lower, upper in STORYBOARD_PHASE_RANGES
        ]
        if any(not rows for rows in phase_candidates):
            raise RuntimeError("captured frames do not cover all lane-change phases")

        def sequence_score(sequence: tuple[dict[str, Any], ...]) -> tuple[float, ...]:
            times = np.asarray([float(row["time_s"]) for row in sequence])
            progresses = np.asarray(
                [float(row["mean_progress"]) for row in sequence]
            )
            return (
                float(sum(int(row["visual_alignment_count"]) for row in sequence)),
                float(np.std(np.diff(times))),
                float(
                    np.sum(
                        np.abs(
                            progresses
                            - np.asarray(STORYBOARD_PROGRESS_TARGETS, dtype=np.float64)
                        )
                    )
                ),
            )

        selected_by_event[event] = list(
            min(product(*phase_candidates), key=sequence_score)
        )
    event = 1
    main = [dict(row) for row in selected_by_event[event]]
    for row, (phase, title) in zip(main, MAIN_PHASES):
        row["phase"] = phase
        row["phase_title"] = title
    return main, selected_by_event


def validate_physical_lane_change(
    trajectory: list[dict[str, Any]],
    main_frames: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    command_time = float(summary["command_times_s"]["1"])
    completion_time = float(summary["completion_times_s"]["1"])
    start = _nearest(trajectory, command_time)["vehicles"]
    end = _nearest(trajectory, completion_time + 0.3)["vehicles"]
    vehicle_checks: dict[str, Any] = {}
    failures: list[str] = []
    for agent in PLATOON_IDS:
        start_state = start[agent]
        end_state = end[agent]
        displacement = float(end_state["y"]) - float(start_state["y"])
        stable_three_lane_roads = {"wave", "recovery"}
        passed = (
            str(start_state["road_id"]) in stable_three_lane_roads
            and str(end_state["road_id"]) in stable_three_lane_roads
            and int(start_state["lane"]) == 1
            and int(end_state["lane"]) == 2
            and 2.75 <= abs(displacement) <= 4.5
        )
        vehicle_checks[agent] = {
            "start_road_id": str(start_state["road_id"]),
            "end_road_id": str(end_state["road_id"]),
            "start_lane_index": int(start_state["lane"]),
            "end_lane_index": int(end_state["lane"]),
            "start_physical_y_m": float(start_state["y"]),
            "end_physical_y_m": float(end_state["y"]),
            "physical_lateral_displacement_m": displacement,
            "passed": passed,
        }
        if not passed:
            failures.append(agent)
    progresses = [float(frame["mean_progress"]) for frame in main_frames]
    monotonic = all(
        right + 1e-6 >= left for left, right in zip(progresses, progresses[1:])
    )
    if not monotonic:
        failures.append("non_monotonic_frame_progress")
    phase_coverage = all(
        lower <= progress <= upper
        for progress, (lower, upper) in zip(progresses, STORYBOARD_PHASE_RANGES)
    )
    if not phase_coverage:
        failures.append("incomplete_lane_change_phase_coverage")
    if not bool(summary.get("all_vehicle_collision_free", False)):
        failures.append("all_vehicle_collision_detected")
    negative_gap_frames = [
        frame
        for frame in main_frames
        if frame.get("visible_minimum_net_gap_m") is not None
        and float(frame["visible_minimum_net_gap_m"]) < -1e-6
    ]
    if negative_gap_frames:
        failures.append("negative_visible_vehicle_net_gap")
    if failures:
        raise RuntimeError(
            "physical lane-change validation failed: " + ", ".join(failures)
        )
    return {
        "passed": True,
        "criterion": (
            "stable three-lane wave/recovery roads; laneIndex 1 to 2; "
            "per-truck physical lateral displacement in [2.75, 4.5] m; "
            "monotonic selected frames covering initiation, early transition, "
            "late transition, and immediate completion"
        ),
        "selected_frame_progress": progresses,
        "selected_frame_phase_ranges": STORYBOARD_PHASE_RANGES,
        "all_vehicle_collision_free": True,
        "selected_frame_minimum_net_gaps_m": [
            frame.get("visible_minimum_net_gap_m") for frame in main_frames
        ],
        "selected_frame_visual_alignment_counts": [
            int(frame["visual_alignment_count"]) for frame in main_frames
        ],
        "vehicles": vehicle_checks,
    }


def _crop_bounds(image: np.ndarray) -> tuple[int, int, int, int]:
    rgb = image[..., :3]
    road = np.max(rgb, axis=2) < 0.28
    road_rows = np.where(np.sum(road, axis=1) > image.shape[1] * 0.05)[0]
    if not len(road_rows):
        return (0, 0, image.shape[1], image.shape[0])
    # The lower-left SUMO scale bar can satisfy the black-pixel threshold at
    # high zoom. Keep the largest contiguous black-row band, which is the road.
    breaks = np.where(np.diff(road_rows) > 1)[0] + 1
    bands = np.split(road_rows, breaks)
    road_rows = max(bands, key=len)
    return (
        0,
        max(0, int(np.min(road_rows)) - 100),
        image.shape[1],
        min(image.shape[0], int(np.max(road_rows)) + 101),
    )


def _draw_panel(
    axis: Any,
    frame: Mapping[str, Any],
    title: str,
    panel_label: str,
) -> None:
    image = mpimg.imread(frame["image"])
    crop = _crop_bounds(image)
    cropped = image[crop[1] : crop[3], crop[0] : crop[2]]
    axis.imshow(cropped)
    axis.set_axis_off()
    axis.set_title(
        f"({panel_label}) {title}, $t={float(frame['time_s']):.1f}$ s "
        f"($\\Delta t={float(frame['delta_command_s']):+.1f}$ s)\n"
        f"{int(frame['source_lane'])}$\\rightarrow${int(frame['target_lane'])}; "
        f"$p_{{lat}}={100.0 * float(frame['mean_progress']):.0f}\\%$; "
        f"$g_F/g_R={float(frame['front_clearance_m']):.1f}/"
        f"{float(frame['rear_clearance_m']):.1f}$ m; "
        f"$N_{{vis}}={int(frame['visible_vehicle_count'])}$",
        color="black",
        pad=2,
    )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="#145FBE",
            markersize=7, label="LCG-Dreamer platoon",
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="#F59119",
            markersize=7, label="Interacting traffic",
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="#DC2D23",
            markersize=7, label="Slow obstacle vehicle",
        ),
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="#969696",
            markersize=7, label="Calibrated background traffic",
        ),
    ]


def compose_figures(
    output: Path,
    trajectory: list[dict[str, Any]],
    main_frames: list[dict[str, Any]],
    selected_by_event: Mapping[int, list[dict[str, Any]]],
) -> tuple[Path, Path]:
    _configure_style()
    main_path = output / "graph_sumo_continuous_lane_change_main.png"
    # One full-width SUMO frame per row keeps vehicle bodies and lane markings
    # legible at the final two-column manuscript width.
    figure, axes = plt.subplots(4, 1, figsize=(7.5, 6.0), constrained_layout=True)
    figure.set_constrained_layout_pads(h_pad=0.01, hspace=0.015)
    for index, (axis, frame, (_, title)) in enumerate(
        zip(axes.flat, main_frames, MAIN_PHASES)
    ):
        _draw_panel(
            axis,
            frame,
            title,
            chr(ord("a") + index),
        )
    figure.legend(
        handles=_legend_handles(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
        labelcolor="black",
        fontsize=8.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    figure.savefig(main_path, bbox_inches="tight", facecolor="white")
    figure.savefig(main_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    all_path = output / "graph_sumo_continuous_lane_change_all.png"
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 1.8), constrained_layout=True)
    titles = (
        "Lane-change initiation",
        "Early transition",
        "Late transition",
        "Immediate completion",
    )
    for event in range(1, 2):
        frames = selected_by_event[event]
        for column, (axis, frame, title) in enumerate(
            zip(axes, frames, titles)
        ):
            _draw_panel(
                axis,
                frame,
                f"M{event}: {title}",
                chr(ord("a") + column),
            )
    figure.legend(
        handles=_legend_handles(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
        labelcolor="black",
        fontsize=8.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    figure.savefig(all_path, bbox_inches="tight", facecolor="white")
    figure.savefig(all_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return main_path, all_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_lane_change_continuous"),
    )
    parser.add_argument("--seed", type=int, default=173_116)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    trajectory, frames, summary = capture_continuous_episode(
        arguments.output, arguments.seed, arguments.device
    )
    main_frames, selected_by_event = select_storyboard_frames(
        trajectory, frames, summary
    )
    summary["physical_lane_change_validation"] = validate_physical_lane_change(
        trajectory, main_frames, summary
    )
    main_path, all_path = compose_figures(
        arguments.output, trajectory, main_frames, selected_by_event
    )
    trajectory_path = arguments.output / "continuous_trajectory.json"
    trajectory_path.write_text(
        json.dumps(trajectory, separators=(",", ":")), encoding="utf-8"
    )
    manifest = {
        **summary,
        "captured_frame_count": len(frames),
        "trajectory_step_count": len(trajectory),
        "trajectory_file": str(trajectory_path.resolve()),
        "selected_main_frames": main_frames,
        "selected_supplement_frames": {
            str(key): value for key, value in selected_by_event.items()
        },
        "main_figure": str(main_path.resolve()),
        "supplement_figure": str(all_path.resolve()),
    }
    manifest_path = arguments.output / "continuous_storyboard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "main": str(main_path.resolve()),
                "supplement": str(all_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "success": summary["success"],
                "collision": summary["collision"],
                "maneuvers_completed": summary["maneuvers_completed"],
                "observed_maximum_lateral_speed_mps": summary[
                    "observed_maximum_lateral_speed_mps"
                ],
                "observed_maximum_lateral_acceleration_mps2": summary[
                    "observed_maximum_lateral_acceleration_mps2"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

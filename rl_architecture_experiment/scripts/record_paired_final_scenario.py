"""Record paired Graph-Dreamer and CACC runs in one final-stage SUMO scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from platoon_control.graph_rl import GraphRLConfig, load_graph_controller
from platoon_control.rl import _attach_follower_predictions, _curriculum_options
from platoon_sumo.env import (
    CUTIN_ID,
    FOLLOWER_IDS,
    PLATOON_IDS,
    PlatoonParallelEnv,
)
from platoon_sumo.run_demo import baseline_actions


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIDEO_FPS = 10.0
VIEW_SCHEMA = "real world"
STATUS_RED = (255, 0, 0, 255)
CUTIN_THREAT_STATES = frozenset(("armed", "aligning"))


def build_final_scenario(
    seed: int,
    episode: int,
    stage: int,
    training_step: int,
) -> tuple[dict[str, Any], int, int]:
    """Reproduce the scenario sampling used by final capability evaluation."""

    evaluation = GraphRLConfig(seed=seed).evaluation_config()
    scenario_rng_seed = seed + 20_000 + stage * 1_000 + episode
    scenario_rng = np.random.default_rng(scenario_rng_seed)
    options, _ = _curriculum_options(
        stage,
        scenario_rng,
        evaluation,
        training_step,
    )
    return options, seed + 10_000 + episode, scenario_rng_seed


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _platoon_snapshot(env: PlatoonParallelEnv) -> dict[str, Any]:
    active = set(env.connection.vehicle.getIDList())
    present = [vehicle_id for vehicle_id in PLATOON_IDS if vehicle_id in active]
    positions = {
        vehicle_id: env.connection.vehicle.getPosition(vehicle_id)
        for vehicle_id in present
    }
    speeds = {
        vehicle_id: float(env.connection.vehicle.getSpeed(vehicle_id))
        for vehicle_id in present
    }
    gaps = {
        follower: float(env._platoon_gap(follower, active))
        for follower in FOLLOWER_IDS
        if follower in active
    }
    valid_gaps = [gap for gap in gaps.values() if gap >= 0.0]
    return {
        "active": present,
        "positions": positions,
        "speeds": speeds,
        "gaps": gaps,
        "minimum_gap": min(valid_gaps, default=math.nan),
        "maximum_gap": max(valid_gaps, default=math.nan),
        "speed_spread": (
            max(speeds.values()) - min(speeds.values()) if speeds else math.nan
        ),
    }


def _target_boundary(snapshot: Mapping[str, Any]) -> tuple[float, float, float, float]:
    positions = list(snapshot["positions"].values())
    if not positions:
        return (-100.0, -100.0, 100.0, 100.0)
    xs = [float(position[0]) for position in positions]
    ys = [float(position[1]) for position in positions]
    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    # Native vehicle tracking recentres the view around the middle truck.  A
    # wider initial scale keeps the complete platoon visible and also exposes
    # enough road ahead to inspect the causal front-vehicle motion.
    width = max(max(xs) - min(xs) + 240.0, 320.0)
    height = max(max(ys) - min(ys) + 140.0, 180.0)
    aspect = FRAME_WIDTH / FRAME_HEIGHT
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    return (
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        center_x + 0.5 * width,
        center_y + 0.5 * height,
    )


def _smooth_boundary(
    previous: tuple[float, float, float, float] | None,
    target: tuple[float, float, float, float],
    alpha: float = 0.18,
) -> tuple[float, float, float, float]:
    if previous is None:
        return target
    return tuple(
        float(old + alpha * (new - old))
        for old, new in zip(previous, target)
    )


def _read_screenshot(path: Path, timeout_seconds: float = 3.0) -> np.ndarray:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        time.sleep(0.02)
    raise RuntimeError(f"SUMO screenshot was not produced: {path}")


def _annotate_frame(
    frame: np.ndarray,
    controller_name: str,
    sim_time: float,
    snapshot: Mapping[str, Any],
) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 54), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0.0, output)
    minimum_gap = float(snapshot["minimum_gap"])
    maximum_gap = float(snapshot["maximum_gap"])
    spread = float(snapshot["speed_spread"])
    gap_text = (
        f"gap {minimum_gap:5.1f}-{maximum_gap:5.1f} m"
        if math.isfinite(minimum_gap) and math.isfinite(maximum_gap)
        else "gap unavailable"
    )
    text = (
        f"{controller_name}   t={sim_time:6.1f} s   {gap_text}   "
        f"speed spread {spread:4.1f} m/s"
    )
    cv2.putText(
        output,
        text,
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def _open_writer(path: Path, frame: np.ndarray) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer: {path}")
    return writer


def _communication_delay_active(
    infos: Mapping[str, Mapping[str, Any]],
) -> bool:
    return any(
        max(
            float(infos.get(agent, {}).get("communication_delay", 0.0)),
            float(
                infos.get(agent, {}).get(
                    "predecessor_communication_delay", 0.0
                )
            ),
        )
        > 0.0
        for agent in FOLLOWER_IDS
    )


def _apply_status_colors(
    env: PlatoonParallelEnv,
    infos: Mapping[str, Mapping[str, Any]],
    original_colors: dict[str, tuple[int, int, int, int]],
) -> tuple[bool, bool]:
    active = set(env.connection.vehicle.getIDList())
    tracked = set(PLATOON_IDS).union({CUTIN_ID}).intersection(active)
    for vehicle_id in tracked:
        original_colors.setdefault(
            vehicle_id, tuple(env.connection.vehicle.getColor(vehicle_id))
        )

    delay_active = _communication_delay_active(infos)
    for vehicle_id in set(PLATOON_IDS).intersection(active):
        env.connection.vehicle.setColor(
            vehicle_id,
            STATUS_RED if delay_active else original_colors[vehicle_id],
        )

    cutin_active = (
        CUTIN_ID in active and env._cutin_state in CUTIN_THREAT_STATES
    )
    if CUTIN_ID in active:
        env.connection.vehicle.setColor(
            CUTIN_ID,
            STATUS_RED if cutin_active else original_colors[CUTIN_ID],
        )
    return delay_active, cutin_active


def _close_gui_environment(env: PlatoonParallelEnv) -> None:
    """Close TraCI without waiting indefinitely for SUMO GUI to exit."""

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


def record_controller(
    controller_kind: str,
    options: Mapping[str, Any],
    reset_seed: int,
    output_path: Path,
    event_path: Path,
    checkpoint: Path | None,
    device: str,
    max_time: float,
) -> dict[str, Any]:
    print(f"{controller_kind}: loading controller", flush=True)
    graph_controller = None
    if controller_kind == "graph":
        if checkpoint is None:
            raise ValueError("a checkpoint is required for Graph-Dreamer recording")
        graph_controller = load_graph_controller(checkpoint, device=device)
        graph_controller.reset()

    evaluation = GraphRLConfig(seed=reset_seed).evaluation_config()
    env = PlatoonParallelEnv(
        render_mode="human",
        calibration_path=evaluation.calibration_path,
        wave_model_path=evaluation.wave_model_path,
        max_time=max_time,
        gui_auto_start=True,
    )
    print(f"{controller_kind}: starting SUMO GUI reset", flush=True)
    observations, _ = env.reset(seed=reset_seed, options=dict(options))
    print(f"{controller_kind}: SUMO GUI reset complete", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    boundary: tuple[float, float, float, float] | None = None
    global_minimum_gap = math.inf
    global_maximum_gap = 0.0
    maximum_speed_spread = 0.0
    speed_spread_sum = 0.0
    disconnected_steps = 0
    shield_steps = 0
    background_peak = 0
    maximum_stuck_seconds = 0.0
    maximum_gridlock_seconds = 0.0
    maximum_observed_delay = 0.0
    communication_delay_events = 0
    delay_color_frames = 0
    cutin_color_frames = 0
    final_info: Mapping[str, Any] = {}
    latest_infos: Mapping[str, Mapping[str, Any]] = {}
    original_colors: dict[str, tuple[int, int, int, int]] = {}
    frame_count = 0
    cv2.setNumThreads(1)

    try:
        view_id = env.connection.gui.getIDList()[0]
        env.connection.gui.setSchema(view_id, VIEW_SCHEMA)
        with tempfile.TemporaryDirectory(prefix=f"sumo_{controller_kind}_frames_") as temp:
            frame_directory = Path(temp)
            while env.agents:
                delay_color_active, cutin_color_active = _apply_status_colors(
                    env, latest_infos, original_colors
                )
                if frame_count < 3:
                    print(
                        f"{controller_kind}: scheduling frame {frame_count}",
                        flush=True,
                    )
                snapshot = _platoon_snapshot(env)
                if frame_count < 3:
                    print(f"{controller_kind}: snapshot ready", flush=True)
                if boundary is None:
                    boundary = _target_boundary(snapshot)
                    env.connection.gui.setBoundary(view_id, *boundary)
                    env.connection.gui.trackVehicle(
                        view_id, PLATOON_IDS[len(PLATOON_IDS) // 2]
                    )
                    if frame_count < 3:
                        print(f"{controller_kind}: boundary set", flush=True)
                frame_path = frame_directory / f"frame_{frame_count:06d}.png"
                env.connection.gui.screenshot(
                    view_id,
                    str(frame_path),
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                )
                if frame_count < 3:
                    print(f"{controller_kind}: screenshot queued", flush=True)

                if graph_controller is None:
                    actions = baseline_actions(observations)
                else:
                    policy_states, _ = graph_controller.follower_policy_states(
                        observations
                    )
                    _attach_follower_predictions(env, policy_states)
                    actions = graph_controller.act(observations, deterministic=True)
                if frame_count < 3:
                    print(f"{controller_kind}: actions ready", flush=True)

                observations, _, terminations, truncations, infos = env.step(actions)
                latest_infos = infos
                if frame_count < 3:
                    print(
                        f"{controller_kind}: simulation step returned for frame "
                        f"{frame_count}",
                        flush=True,
                    )
                if graph_controller is not None:
                    graph_controller.record_applied_actions(infos)
                final_info = infos.get("leader", {})
                frame = _read_screenshot(frame_path)
                if frame_count < 3:
                    print(
                        f"{controller_kind}: screenshot read for frame {frame_count}",
                        flush=True,
                    )
                frame_path.unlink(missing_ok=True)
                frame = _annotate_frame(
                    frame,
                    "MA-Latent-Graph-Dreamer"
                    if controller_kind == "graph"
                    else "CACC baseline",
                    float(env.sim_time),
                    snapshot,
                )
                if writer is None:
                    writer = _open_writer(output_path, frame)
                writer.write(frame)
                frame_count += 1
                delay_color_frames += int(delay_color_active)
                cutin_color_frames += int(cutin_color_active)

                minimum_gap = float(snapshot["minimum_gap"])
                maximum_gap = float(snapshot["maximum_gap"])
                speed_spread = float(snapshot["speed_spread"])
                if math.isfinite(minimum_gap):
                    global_minimum_gap = min(global_minimum_gap, minimum_gap)
                if math.isfinite(maximum_gap):
                    global_maximum_gap = max(global_maximum_gap, maximum_gap)
                    disconnected_steps += int(maximum_gap > 15.0)
                if math.isfinite(speed_spread):
                    maximum_speed_spread = max(maximum_speed_spread, speed_spread)
                    speed_spread_sum += speed_spread
                shield_steps += int(
                    any(bool(info.get("shield_triggered", False)) for info in infos.values())
                )
                background_peak = max(
                    background_peak,
                    int(final_info.get("background_active_count", 0)),
                )
                maximum_stuck_seconds = max(
                    maximum_stuck_seconds,
                    float(final_info.get("background_max_stuck_seconds", 0.0)),
                )
                maximum_gridlock_seconds = max(
                    maximum_gridlock_seconds,
                    float(final_info.get("background_max_gridlock_seconds", 0.0)),
                )
                communication_delay_events = max(
                    communication_delay_events,
                    int(final_info.get("communication_delay_events", 0)),
                )
                maximum_observed_delay = max(
                    maximum_observed_delay,
                    max(
                        (
                            float(info.get("communication_delay", 0.0))
                            for info in infos.values()
                        ),
                        default=0.0,
                    ),
                )
                if frame_count % 100 == 0:
                    print(
                        f"{controller_kind}: frame={frame_count} "
                        f"time={env.sim_time:.1f}s min_gap={global_minimum_gap:.2f} "
                        f"max_gap={global_maximum_gap:.2f}",
                        flush=True,
                    )
                if any(terminations.values()) or any(truncations.values()):
                    break
    finally:
        if writer is not None:
            writer.release()
        events = list(env.event_log)
        _close_gui_environment(env)

    event_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    event_counts = Counter(str(event.get("event", "unknown")) for event in events)
    return {
        "controller": controller_kind,
        "video": str(output_path.resolve()),
        "events": str(event_path.resolve()),
        "frames": frame_count,
        "duration_seconds": frame_count / VIDEO_FPS,
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "failure": bool(final_info.get("failure", False)),
        "leader_station": float(final_info.get("leader_station", 0.0)),
        "connected_pair_count": int(
            final_info.get("platoon_connected_pair_count", 0)
        ),
        "minimum_gap_m": (
            0.0 if global_minimum_gap == math.inf else global_minimum_gap
        ),
        "maximum_gap_m": global_maximum_gap,
        "maximum_speed_spread_mps": maximum_speed_spread,
        "mean_speed_spread_mps": speed_spread_sum / max(frame_count, 1),
        "disconnected_seconds": disconnected_steps / VIDEO_FPS,
        "shield_seconds": shield_steps / VIDEO_FPS,
        "background_peak_count": background_peak,
        "background_max_stuck_seconds": maximum_stuck_seconds,
        "background_max_gridlock_seconds": maximum_gridlock_seconds,
        "communication_delay_events": communication_delay_events,
        "maximum_observed_delay_seconds": maximum_observed_delay,
        "communication_delay_red_seconds": delay_color_frames / VIDEO_FPS,
        "cutin_threat_red_seconds": cutin_color_frames / VIDEO_FPS,
        "final_cutin_state": str(final_info.get("cutin_state", "unknown")),
        "selected_wave_source": final_info.get("selected_wave_source"),
        "event_counts": dict(sorted(event_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_videos"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument("--training-step", type=int, default=300_000)
    parser.add_argument("--max-time", type=float, default=240.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--controllers",
        choices=("both", "graph", "cacc"),
        default="both",
    )
    arguments = parser.parse_args()

    checkpoint = arguments.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    options, reset_seed, scenario_rng_seed = build_final_scenario(
        arguments.seed,
        arguments.episode,
        arguments.stage,
        arguments.training_step,
    )
    if arguments.stage >= 7 and int(options.get("front_vehicle_count", 0)) < 1:
        raise ValueError(
            "selected final scenario has no explicit front vehicle; choose another episode"
        )

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"final_seed{arguments.seed}_episode{arguments.episode + 1}"
    results = []
    controller_kinds = (
        ("graph", "cacc")
        if arguments.controllers == "both"
        else (arguments.controllers,)
    )
    for controller_kind in controller_kinds:
        results.append(
            record_controller(
                controller_kind=controller_kind,
                options=options,
                reset_seed=reset_seed,
                output_path=output_dir / f"{prefix}_{controller_kind}.mp4",
                event_path=output_dir / f"{prefix}_{controller_kind}_events.json",
                checkpoint=checkpoint if controller_kind == "graph" else None,
                device=arguments.device,
                max_time=arguments.max_time,
            )
        )

    manifest = {
        "purpose": "paired final-stage visual comparison",
        "view_schema": VIEW_SCHEMA,
        "camera": (
            "expanded overhead view with initial team/front-road framing and "
            "midpoint-vehicle tracking"
        ),
        "visual_status_colors": {
            "platoon_during_communication_delay": "red",
            "cutin_vehicle_during_armed_or_aligning": "red",
        },
        "video_fps": VIDEO_FPS,
        "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
        "seed": arguments.seed,
        "episode": arguments.episode + 1,
        "stage": arguments.stage,
        "human_stage": arguments.stage + 1,
        "training_step": arguments.training_step,
        "scenario_rng_seed": scenario_rng_seed,
        "reset_seed": reset_seed,
        "scenario_options": options,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_digest(checkpoint),
        "checkpoint_steps": int(checkpoint_payload.get("total_steps", 0)),
        "architecture": checkpoint_payload.get("control_architecture"),
        "results": results,
    }
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""NGSIM-driven IDM, LC2013, vehicle-mix, and demand calibration."""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import traci
from scipy.optimize import least_squares
from scipy.stats import ks_2samp, wasserstein_distance

from platoon_sumo.build_network import NET_FILE, _find_sumo_binary, build_network

from .ngsim import NGSIMStore


@dataclass
class IDMParameters:
    tau: float
    min_gap: float
    accel: float
    decel: float
    emergency_decel: float
    delta: float
    action_step_length: float
    max_speed: float
    speed_factor: float
    speed_deviation: float
    length: float
    width: float


@dataclass
class VehicleTypeCalibration:
    type_id: str
    vehicle_class: int
    style: str
    probability: float
    idm: IDMParameters


STYLE_NAMES = ("cautious", "normal", "aggressive")
SUMO_CLASS = {1: "motorcycle", 2: "passenger", 3: "truck"}


def idm_acceleration(
    speed: np.ndarray,
    leader_speed: np.ndarray,
    gap: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    tau, min_gap, accel, decel, delta, desired_speed = parameters
    closing_speed = speed - leader_speed
    desired_gap = min_gap + np.maximum(
        0.0,
        speed * tau + speed * closing_speed / (2.0 * np.sqrt(accel * decel)),
    )
    unconstrained = accel * (
        1.0
        - np.power(np.clip(speed / desired_speed, 0.0, 3.0), delta)
        - np.square(desired_gap / np.maximum(gap, 0.2))
    )
    return np.clip(unconstrained, -10.0, 5.0)


def _calibration_frame(store: NGSIMStore) -> pd.DataFrame:
    arrays = store.arrays
    frame = pd.DataFrame(
        {
            "vehicle_id": arrays["vehicle_id"],
            "time_ms": arrays["time_ms"],
            "speed": arrays["speed"],
            "acceleration": arrays["acceleration"],
            "length": arrays["length"],
            "width": arrays["width"],
            "vehicle_class": arrays["vehicle_class"],
            "lane": arrays["lane"],
            "preceding": arrays["preceding"],
            "gap": arrays["space_headway"],
            "time_headway": arrays["time_headway"],
        }
    )
    previous_lane = frame.groupby("vehicle_id", sort=False)["lane"].shift()
    frame["lane_change"] = (previous_lane.notna() & (previous_lane != frame["lane"])).astype(np.int8)
    return frame


def _assign_driver_styles(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[int, str]]:
    passenger = frame[frame["vehicle_class"] == 2]
    features = passenger.groupby("vehicle_id").agg(
        headway=("time_headway", lambda values: np.median(values[(values > 0.2) & (values < 20.0)]) if np.any((values > 0.2) & (values < 20.0)) else 5.0),
        positive_acceleration=("acceleration", lambda values: np.quantile(values, 0.90)),
        braking=("acceleration", lambda values: -np.quantile(values, 0.10)),
        lane_change_rate=("lane_change", "mean"),
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = features.to_numpy(np.float64)
    scaled = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)
    rng = np.random.default_rng(seed)
    centers = scaled[rng.choice(len(scaled), 3, replace=False)].copy()
    labels = np.zeros(len(scaled), dtype=np.int64)
    for _ in range(100):
        updated = np.argmin(np.square(scaled[:, None] - centers[None]).sum(axis=2), axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(3):
            members = scaled[labels == cluster]
            centers[cluster] = members.mean(axis=0) if len(members) else scaled[rng.integers(len(scaled))]
    aggression = -scaled[:, 0] + scaled[:, 1] + scaled[:, 2] + 5.0 * scaled[:, 3]
    cluster_score = {cluster: float(aggression[labels == cluster].mean()) for cluster in range(3)}
    ordered = sorted(cluster_score, key=cluster_score.get)
    cluster_name = {cluster: STYLE_NAMES[index] for index, cluster in enumerate(ordered)}
    assignments = {
        int(vehicle_id): cluster_name[int(label)]
        for vehicle_id, label in zip(features.index.to_numpy(), labels)
    }
    frame = frame.copy()
    frame["style"] = [assignments.get(int(vehicle), "normal") for vehicle in frame["vehicle_id"]]
    return frame, assignments


def _leader_join(frame: pd.DataFrame) -> pd.DataFrame:
    leaders = frame[["vehicle_id", "time_ms", "speed"]].rename(
        columns={"vehicle_id": "preceding", "speed": "leader_speed"}
    )
    joined = frame.merge(leaders, on=("preceding", "time_ms"), how="inner")
    return joined[
        (joined["preceding"] > 0)
        & (joined["gap"] > 0.5)
        & (joined["gap"] < 150.0)
        & (joined["speed"] >= 0.0)
        & (joined["acceleration"] >= -8.0)
        & (joined["acceleration"] <= 5.0)
    ]


def _fit_group(group: pd.DataFrame, rng: np.random.Generator, max_rows: int) -> tuple[IDMParameters, dict[str, float]]:
    if len(group) > max_rows:
        group = group.iloc[rng.choice(len(group), max_rows, replace=False)]
    speed = group["speed"].to_numpy(np.float64)
    leader_speed = group["leader_speed"].to_numpy(np.float64)
    gap = group["gap"].to_numpy(np.float64)
    observed = group["acceleration"].to_numpy(np.float64)
    positive = observed[observed > 0.0]
    negative = -observed[observed < 0.0]
    empirical_accel = float(np.clip(np.quantile(positive, 0.80), 0.6, 3.5))
    empirical_decel = float(np.clip(np.quantile(negative, 0.80), 1.0, 5.0))
    initial = np.asarray(
        [1.2, 3.0, empirical_accel, empirical_decel, 4.0, max(12.0, np.quantile(speed, 0.99))]
    )
    lower = np.asarray(
        [0.4, 0.5, max(0.4, 0.65 * empirical_accel), max(0.8, 0.65 * empirical_decel), 2.0, 8.0]
    )
    upper = np.asarray(
        [3.0, 10.0, min(4.0, 1.5 * empirical_accel), min(6.0, 1.5 * empirical_decel), 6.0, 35.0]
    )

    result = least_squares(
        lambda values: idm_acceleration(speed, leader_speed, gap, values) - observed,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.8,
        max_nfev=300,
    )
    predicted = idm_acceleration(speed, leader_speed, gap, result.x)
    tau, min_gap, accel, decel, delta, desired_speed = result.x
    emergency = max(float(decel), float(np.quantile(np.maximum(-observed, 0.0), 0.99)))
    length = float(group["length"].median())
    width = float(group["width"].median())
    action_step = round(float(np.clip(tau / 3.0, 0.1, 1.0)) * 10.0) / 10.0
    parameters = IDMParameters(
        tau=float(tau),
        min_gap=float(min_gap),
        accel=float(accel),
        decel=float(decel),
        emergency_decel=float(max(emergency, decel)),
        delta=float(delta),
        action_step_length=action_step,
        max_speed=float(desired_speed),
        speed_factor=1.0,
        speed_deviation=float(np.clip(np.std(speed) / max(np.mean(speed), 1.0), 0.02, 0.30)),
        length=length,
        width=width,
    )
    metrics = {
        "rows": float(len(group)),
        "acceleration_rmse": float(np.sqrt(np.mean(np.square(predicted - observed)))),
        "acceleration_mae": float(np.mean(np.abs(predicted - observed))),
    }
    return parameters, metrics


def fit_idm_types(
    store: NGSIMStore,
    seed: int = 42,
    max_rows_per_type: int = 80_000,
) -> tuple[list[VehicleTypeCalibration], dict[str, Any]]:
    frame, _ = _assign_driver_styles(_calibration_frame(store), seed)
    joined = _leader_join(frame)
    rng = np.random.default_rng(seed)
    vehicle_counts = frame.groupby("vehicle_class")["vehicle_id"].nunique()
    total_vehicles = float(vehicle_counts.sum())
    output: list[VehicleTypeCalibration] = []
    metrics: dict[str, Any] = {}
    for vehicle_class in (1, 2, 3):
        styles = STYLE_NAMES if vehicle_class == 2 else ("normal",)
        class_frame = frame[frame["vehicle_class"] == vehicle_class]
        class_probability = float(vehicle_counts.get(vehicle_class, 0.0) / total_vehicles)
        for style in styles:
            group = joined[(joined["vehicle_class"] == vehicle_class)]
            if vehicle_class == 2:
                group = group[group["style"] == style]
                style_probability = float(
                    class_frame[class_frame["style"] == style]["vehicle_id"].nunique()
                    / max(class_frame["vehicle_id"].nunique(), 1)
                )
            else:
                style_probability = 1.0
            parameters, group_metrics = _fit_group(group, rng, max_rows_per_type)
            type_id = f"ngsim_{SUMO_CLASS[vehicle_class]}_{style}"
            output.append(
                VehicleTypeCalibration(
                    type_id=type_id,
                    vehicle_class=vehicle_class,
                    style=style,
                    probability=class_probability * style_probability,
                    idm=parameters,
                )
            )
            metrics[type_id] = group_metrics
    probability_sum = sum(item.probability for item in output)
    for item in output:
        item.probability /= probability_sum
    return output, metrics


def estimate_demand(store: NGSIMStore) -> dict[str, Any]:
    frame = _calibration_frame(store)
    start = int(frame["time_ms"].min())
    complete = frame[frame["time_ms"] < start + 12 * 60_000]
    entries = complete.groupby("vehicle_id", as_index=False).first()
    lane_rate = (entries.groupby("lane").size() / 0.2).to_dict()
    main_rate = float(np.mean([lane_rate.get(lane, 0.0) for lane in range(1, 7)]))
    minute_profiles = []
    entries = entries.copy()
    entries["minute"] = ((entries["time_ms"] - start) // 60_000).astype(int)
    for minute in range(12):
        minute_frame = entries[entries["minute"] == minute]
        minute_lane_rate = minute_frame.groupby("lane").size() * 60.0
        minute_profiles.append(
            {
                "profile_id": minute,
                "main_per_lane_veh_per_hour": float(
                    np.mean([minute_lane_rate.get(lane, 0.0) for lane in range(1, 7)])
                ),
                "ramp_veh_per_hour": float(minute_lane_rate.get(7, 0.0)),
            }
        )
    return {
        "main_per_lane_veh_per_hour": main_rate,
        "ramp_veh_per_hour": float(lane_rate.get(7, 0.0)),
        "minute_profiles": minute_profiles,
    }


def _apply_scales(
    types: list[VehicleTypeCalibration], scales: dict[str, float]
) -> list[VehicleTypeCalibration]:
    output: list[VehicleTypeCalibration] = []
    for item in types:
        values = asdict(item.idm)
        values["tau"] *= scales["tau_scale"]
        values["min_gap"] *= scales["gap_scale"]
        values["accel"] *= scales["accel_scale"]
        values["decel"] *= scales["decel_scale"]
        values["emergency_decel"] = max(values["decel"], values["emergency_decel"] * scales["decel_scale"])
        values["max_speed"] *= scales["speed_scale"]
        output.append(
            VehicleTypeCalibration(
                item.type_id,
                item.vehicle_class,
                item.style,
                item.probability,
                IDMParameters(**values),
            )
        )
    return output


def _write_trial_files(
    directory: Path,
    types: list[VehicleTypeCalibration],
    demand: dict[str, Any],
    lane_parameters: dict[str, float],
    demand_scale: float,
    duration: float,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    routes = ET.Element("routes")
    distribution = ET.SubElement(routes, "vTypeDistribution", id="ngsim_background")
    for item in types:
        style_scale = {"cautious": 0.7, "normal": 1.0, "aggressive": 1.4}[item.style]
        p = item.idm
        ET.SubElement(
            distribution,
            "vType",
            id=item.type_id,
            probability=f"{item.probability:.8f}",
            vClass=SUMO_CLASS[item.vehicle_class],
            carFollowModel="IDM",
            laneChangeModel="LC2013",
            length=f"{p.length:.3f}",
            width=f"{p.width:.3f}",
            minGap=f"{p.min_gap:.3f}",
            tau=f"{p.tau:.3f}",
            accel=f"{p.accel:.3f}",
            decel=f"{p.decel:.3f}",
            emergencyDecel=f"{p.emergency_decel:.3f}",
            delta=f"{p.delta:.3f}",
            actionStepLength=f"{p.action_step_length:.1f}",
            maxSpeed=f"{p.max_speed:.3f}",
            speedFactor=f"{p.speed_factor:.3f}",
            speedDev=f"{p.speed_deviation:.3f}",
            lcStrategic=f"{lane_parameters['lcStrategic']:.3f}",
            lcCooperative=f"{lane_parameters['lcCooperative']:.3f}",
            lcSpeedGain=f"{lane_parameters['lcSpeedGain'] * style_scale:.3f}",
            lcKeepRight=f"{lane_parameters['lcKeepRight']:.3f}",
            lcAssertive=f"{lane_parameters['lcAssertive'] * style_scale:.3f}",
        )
    ET.SubElement(routes, "route", id="mainRoute", edges="main0 curve mergePre mainAlongRamp mergePost wave recovery")
    ET.SubElement(routes, "route", id="rampRoute", edges="ramp rampAccel rampTaper mergePost wave recovery")
    main_rate = demand["main_per_lane_veh_per_hour"] * demand_scale
    for lane in range(3):
        ET.SubElement(
            routes,
            "flow",
            id=f"calibration_main_{lane}",
            type="ngsim_background",
            route="mainRoute",
            begin="0",
            end=str(duration),
            vehsPerHour=f"{main_rate:.2f}",
            departLane=str(lane),
            departSpeed="desired",
        )
    ET.SubElement(
        routes,
        "flow",
        id="calibration_ramp",
        type="ngsim_background",
        route="rampRoute",
        begin="0",
        end=str(duration),
        vehsPerHour=f"{demand['ramp_veh_per_hour'] * demand_scale:.2f}",
        departLane="0",
        departSpeed="desired",
    )
    route_path = directory / "calibration.rou.xml"
    ET.ElementTree(routes).write(route_path, encoding="utf-8", xml_declaration=True)
    configuration = ET.Element("configuration")
    inputs = ET.SubElement(configuration, "input")
    ET.SubElement(inputs, "net-file", value=str(NET_FILE.resolve()))
    ET.SubElement(inputs, "route-files", value=str(route_path.resolve()))
    time = ET.SubElement(configuration, "time")
    ET.SubElement(time, "begin", value="0")
    ET.SubElement(time, "end", value=str(duration))
    config_path = directory / "calibration.sumocfg"
    ET.ElementTree(configuration).write(config_path, encoding="utf-8", xml_declaration=True)
    return config_path


def _probe_sumo(config_path: Path, seed: int, duration: float) -> dict[str, np.ndarray | float]:
    label = f"calibration-{uuid.uuid4().hex}"
    traci.start(
        [
            _find_sumo_binary("sumo"),
            "-c",
            str(config_path),
            "--step-length",
            "0.1",
            "--seed",
            str(seed),
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
            "--step-method.ballistic",
            "true",
        ],
        label=label,
    )
    connection = traci.getConnection(label)
    speeds: list[float] = []
    accelerations: list[float] = []
    gaps: list[float] = []
    lane_changes = 0
    vehicle_seconds = 0.0
    previous_lane: dict[str, int] = {}
    departed = 0
    try:
        while connection.simulation.getTime() < duration:
            connection.simulationStep()
            current_time = float(connection.simulation.getTime())
            departed += len(connection.simulation.getDepartedIDList())
            vehicles = connection.vehicle.getIDList()
            if current_time < 20.0:
                previous_lane.update(
                    {vehicle: connection.vehicle.getLaneIndex(vehicle) for vehicle in vehicles}
                )
                continue
            vehicle_seconds += len(vehicles) * 0.1
            for vehicle in vehicles:
                speed = float(connection.vehicle.getSpeed(vehicle))
                speeds.append(speed)
                accelerations.append(float(connection.vehicle.getAcceleration(vehicle)))
                leader = connection.vehicle.getLeader(vehicle, 150.0)
                if leader:
                    gaps.append(float(leader[1]))
                lane = int(connection.vehicle.getLaneIndex(vehicle))
                if vehicle in previous_lane and previous_lane[vehicle] != lane:
                    lane_changes += 1
                previous_lane[vehicle] = lane
    finally:
        connection.close()
    return {
        "speed": np.asarray(speeds, np.float32),
        "acceleration": np.asarray(accelerations, np.float32),
        "gap": np.asarray(gaps, np.float32),
        "lane_change_rate": lane_changes / max(vehicle_seconds, 1.0),
        "departed": float(departed),
    }


def _reference_statistics(store: NGSIMStore, split: str) -> dict[str, np.ndarray | float]:
    bounds = store.split(split)
    arrays = store.arrays
    selection = (arrays["time_ms"] >= bounds.start_ms) & (arrays["time_ms"] <= bounds.end_ms)
    vehicle_time = float(selection.sum() * 0.1)
    selected_lane = arrays["lane"][selection]
    selected_vehicle = arrays["vehicle_id"][selection]
    selected_time = arrays["time_ms"][selection]
    order = np.lexsort((selected_time, selected_vehicle))
    lane_ordered = selected_lane[order]
    vehicle_ordered = selected_vehicle[order]
    lane_changes = int(
        np.sum((vehicle_ordered[1:] == vehicle_ordered[:-1]) & (lane_ordered[1:] != lane_ordered[:-1]))
    )
    return {
        "speed": arrays["speed"][selection],
        "acceleration": arrays["acceleration"][selection],
        "gap": arrays["space_headway"][selection & (arrays["space_headway"] > 0.5)],
        "lane_change_rate": lane_changes / max(vehicle_time, 1.0),
    }


def _distribution_objective(reference: dict[str, Any], simulated: dict[str, Any]) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    scales = {"speed": 10.0, "acceleration": 3.0, "gap": 30.0}
    for key, scale in scales.items():
        if len(simulated[key]) == 0:
            components[key] = 10.0
        else:
            components[key] = float(wasserstein_distance(reference[key], simulated[key]) / scale)
    target_lc = float(reference["lane_change_rate"])
    components["lane_change_rate"] = abs(float(simulated["lane_change_rate"]) - target_lc) / max(target_lc, 1e-5)
    total = components["speed"] + components["acceleration"] + components["gap"] + 0.5 * components["lane_change_rate"]
    return total, components


def _vtype_xml(
    types: list[VehicleTypeCalibration], lane_parameters: dict[str, float]
) -> str:
    root = ET.Element("additional")
    for item in types:
        p = item.idm
        style_scale = {"cautious": 0.7, "normal": 1.0, "aggressive": 1.4}[item.style]
        ET.SubElement(
            root,
            "vType",
            id=item.type_id,
            vClass=SUMO_CLASS[item.vehicle_class],
            carFollowModel="IDM",
            laneChangeModel="LC2013",
            length=f"{p.length:.4f}", width=f"{p.width:.4f}",
            minGap=f"{p.min_gap:.4f}", tau=f"{p.tau:.4f}",
            accel=f"{p.accel:.4f}", decel=f"{p.decel:.4f}",
            emergencyDecel=f"{p.emergency_decel:.4f}", delta=f"{p.delta:.4f}",
            actionStepLength=f"{p.action_step_length:.1f}", maxSpeed=f"{p.max_speed:.4f}",
            speedFactor=f"{p.speed_factor:.4f}", speedDev=f"{p.speed_deviation:.4f}",
            lcStrategic=f"{lane_parameters['lcStrategic']:.4f}",
            lcCooperative=f"{lane_parameters['lcCooperative']:.4f}",
            lcSpeedGain=f"{lane_parameters['lcSpeedGain'] * style_scale:.4f}",
            lcKeepRight=f"{lane_parameters['lcKeepRight']:.4f}",
            lcAssertive=f"{lane_parameters['lcAssertive'] * style_scale:.4f}",
        )
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def calibrate_sumo(
    ngsim_dir: str | Path = "artifacts/ngsim",
    output_dir: str | Path = "artifacts/calibration",
    trials: int = 400,
    seeds: int = 3,
    duration: float = 90.0,
    seed: int = 42,
) -> Path:
    """Run direct IDM fitting followed by reproducible closed-loop SUMO search."""
    build_network(force=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    store = NGSIMStore(ngsim_dir)
    base_types, direct_metrics = fit_idm_types(store, seed=seed)
    demand = estimate_demand(store)
    reference = _reference_statistics(store, "validation")
    trial_root = output / "trials"

    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    executor = ProcessPoolExecutor(max_workers=seeds) if seeds > 1 else None
    try:
        for trial_number in range(max(1, trials)):
            if trial_number == 0:
                scales = {key: 1.0 for key in ("tau_scale", "gap_scale", "accel_scale", "decel_scale", "speed_scale")}
                lane = {"lcStrategic": 1.0, "lcCooperative": 1.0, "lcSpeedGain": 1.0, "lcKeepRight": 1.0, "lcAssertive": 1.0}
                demand_scale = 1.0
            else:
                scales = {
                    "tau_scale": float(rng.uniform(0.7, 1.3)),
                    "gap_scale": float(rng.uniform(0.7, 1.3)),
                    "accel_scale": float(rng.uniform(0.7, 1.3)),
                    "decel_scale": float(rng.uniform(0.7, 1.3)),
                    "speed_scale": float(rng.uniform(0.7, 1.4)),
                }
                lane = {
                    "lcStrategic": float(rng.uniform(0.0, 3.0)),
                    "lcCooperative": float(rng.uniform(0.0, 1.0)),
                    "lcSpeedGain": float(rng.uniform(0.0, 4.0)),
                    "lcKeepRight": float(rng.uniform(0.0, 3.0)),
                    "lcAssertive": float(rng.uniform(0.5, 5.0)),
                }
                demand_scale = float(rng.uniform(0.8, 1.2))
            types = _apply_scales(base_types, scales)
            config_path = _write_trial_files(
                trial_root / f"trial_{trial_number}", types, demand, lane, demand_scale, duration
            )
            seed_values = [seed + trial_number * 17 + offset for offset in range(seeds)]
            if executor is None:
                simulations = [_probe_sumo(config_path, seed_values[0], duration)]
            else:
                simulations = list(
                    executor.map(
                        _probe_sumo,
                        repeat(config_path, seeds),
                        seed_values,
                        repeat(duration, seeds),
                    )
                )
            scores = []
            details = []
            for simulated in simulations:
                score, components = _distribution_objective(reference, simulated)
                expected_departures = (3 * demand["main_per_lane_veh_per_hour"] + demand["ramp_veh_per_hour"]) * demand_scale * duration / 3600.0
                flow_error = abs(float(simulated["departed"]) - expected_departures) / max(expected_departures, 1.0)
                score += 0.5 * flow_error
                components["flow"] = flow_error
                scores.append(score)
                details.append(components)
            record = {
                "trial": trial_number,
                **scales,
                **lane,
                "demand_scale": demand_scale,
                "objective": float(np.mean(scores)),
                "components": details,
            }
            history.append(record)
            if best_record is None or record["objective"] < best_record["objective"]:
                best_record = record
            if (trial_number + 1) % 10 == 0 or trial_number + 1 == max(1, trials):
                (output / "search_progress.json").write_text(
                    json.dumps(
                        {
                            "completed_trials": trial_number + 1,
                            "best_trial": best_record["trial"],
                            "best_objective": best_record["objective"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "completed_trials": trial_number + 1,
                            "best_objective": best_record["objective"],
                        }
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    assert best_record is not None
    best = best_record
    scales = {key: float(best[key]) for key in ("tau_scale", "gap_scale", "accel_scale", "decel_scale", "speed_scale")}
    lane = {key: float(best[key]) for key in ("lcStrategic", "lcCooperative", "lcSpeedGain", "lcKeepRight", "lcAssertive")}
    final_types = _apply_scales(base_types, scales)
    free_flow_speed_p99 = {
        str(vehicle_class): float(
            np.quantile(
                store.arrays["speed"][store.arrays["vehicle_class"] == vehicle_class],
                0.99,
            )
        )
        for vehicle_class in SUMO_CLASS
    }
    parameters = {
        "format_version": 1,
        "ngsim_manifest": store.manifest,
        "free_flow_speed_p99": free_flow_speed_p99,
        "types": [
            {**asdict(item), "sumo_class": SUMO_CLASS[item.vehicle_class]}
            for item in final_types
        ],
        "lane_change": lane,
        "demand": demand,
        "demand_scale": float(best["demand_scale"]),
        "best_objective": float(best["objective"]),
        "best_trial": int(best["trial"]),
    }
    parameter_path = output / "calibrated_parameters.json"
    parameter_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    (output / "ngsim_vtypes.add.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + _vtype_xml(final_types, lane),
        encoding="utf-8",
    )
    final_config = _write_trial_files(
        trial_root / "heldout_test", final_types, demand, lane, float(best["demand_scale"]), duration
    )
    test_reference = _reference_statistics(store, "test")
    test_simulated = _probe_sumo(final_config, seed + 100_003, duration)
    test_objective, test_components = _distribution_objective(test_reference, test_simulated)
    test_ks = {
        key: float(ks_2samp(test_reference[key], test_simulated[key]).statistic)
        if len(test_simulated[key]) else 1.0
        for key in ("speed", "acceleration", "gap")
    }
    report = {
        "direct_idm": direct_metrics,
        "study_trials": len(history),
        "best_objective": float(best["objective"]),
        "best_parameters": best,
        "validation_components": best["components"],
        "heldout_test": {"objective": test_objective, "components": test_components, "ks": test_ks},
        "history": history,
    }
    (output / "fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return parameter_path


def calibration_gate(
    output_dir: str | Path = "artifacts/calibration",
    max_objective: float = 2.0,
) -> tuple[bool, dict[str, Any]]:
    output = Path(output_dir)
    parameters = json.loads((output / "calibrated_parameters.json").read_text(encoding="utf-8"))
    report = json.loads((output / "fit_report.json").read_text(encoding="utf-8"))
    passed = bool(parameters["best_objective"] <= max_objective)
    return passed, {
        "passed": passed,
        "best_objective": parameters["best_objective"],
        "threshold": max_objective,
        "trials": report["study_trials"],
    }

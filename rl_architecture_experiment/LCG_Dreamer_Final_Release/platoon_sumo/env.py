"""PettingZoo Parallel environment for the adversarial SUMO platoon scenario."""

from __future__ import annotations

import functools
import json
import math
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium.spaces as spaces
import numpy as np
import traci
import traci.constants as tc
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from .build_network import NET_FILE, SCENARIO_DIR, _find_sumo_binary, build_network


STEP_LENGTH = 0.1
MAX_TIME = 240.0
TARGET_SPEED = 25.0
TARGET_GAP = 12.0
CUTIN_TARGET_GAP = 8.0
MIN_SAFE_GAP = 5.0
MERGE_LANE_CHANGE_DURATION = 2.0
MERGE_LOOKAHEAD = 3.0
MERGE_RETRY_INTERVAL = 2.5
MERGE_SPEED_CONTROL_DISTANCE = 120.0
MERGE_TERMINAL_BUFFER = 25.0
MERGE_COMFORTABLE_DECELERATION = 2.5
MERGE_SPEED_RESPONSE_TIME = 2.0
MERGE_QUEUE_BUFFER = 2.0
MERGE_QUEUE_BRAKING = 4.5
MERGE_QUEUE_RESPONSE_TIME = 0.5
MERGE_QUEUE_CLOSING_TIME = 4.0
MERGE_QUEUE_EMERGENCY_DECELERATION = 6.0
BACKGROUND_PLATOON_GUARD_FRONT = 70.0
BACKGROUND_PLATOON_GUARD_REAR = 40.0
RAMP_STORAGE_VEHICLE_LIMIT = 8
RAMP_STORAGE_REOPEN_LIMIT = 6
RAMP_SPILLBACK_POSITION = 140.0
RAMP_SPILLBACK_SPEED = 3.0
RAMP_METER_MIN_HEADWAY = 1.5
DEFAULT_LANE_CHANGE_MODE = 1621
SAFETY_ONLY_LANE_CHANGE_MODE = 512
# Clear SUMO's car-following safe-speed bit for TraCI-controlled trucks.  The
# environment applies its own TTC/gap collision shield before every command.
PLATOON_SPEED_MODE = 30
V2V_DELAY_EVENT_PROBABILITY = 0.15
V2V_DELAY_MIN = 0.1
V2V_DELAY_MAX = 0.2
PLATOON_LENGTH = 12.0
PLATOON_LANE_INDEX = 1
MAX_ACCELERATION = 1.2
MAX_BRAKING = -3.0
EMERGENCY_BRAKING = -6.0
MAX_JERK = 1.5
# A stopped truck must be able to release a residual negative command within
# one control cycle; applying the cruise jerk limit here recreates the long
# sequential-start tail after a synchronized stop.
# Releasing a brake at zero speed does not create a longitudinal impulse.  The
# full residual service-brake command must clear in one control tick so that a
# shared restart intent is not serialized by actuator-state bookkeeping.
MAX_RESTART_JERK = 45.0
MAX_BRAKE_RELEASE_JERK = 8.0
RESTART_SPEED_THRESHOLD = 0.5
RESTART_FRONT_SPEED = 2.0
FOLLOWER_RESTART_RECOVERY_SPEED = 2.0
FOLLOWER_RESTART_ACCELERATION = 0.3
STALLED_FAILURE_TIMEOUT = 10.0
# An active cooperative lane-change mission may require the platoon to stop
# while a target-lane gap passes.  This bounded grace period keeps that safe
# wait distinct from an unexplained clear-path stall.
LANE_CHANGE_STALL_GRACE = 30.0
REJOIN_GAP_TOLERANCE = 3.0
REJOIN_HOLD_TIME = 1.0
# A cut-in activates recovery immediately.  Gap-only activation needs enough
# separation to distinguish a real split from normal sub-metre tracking error;
# the all-state restart reward still starts closing smaller deviations.
RECOVERY_ACTIVATION_MARGIN = 1.5
TRUCK_MASS = 25_000.0
TRUCK_DRAG_COEFFICIENT = 0.6
TRUCK_FRONTAL_AREA = 10.0
TRUCK_ROLLING_RESISTANCE = 0.006
DRIVETRAIN_EFFICIENCY = 0.9
AIR_DENSITY = 1.225
COHESION_GATE_MU = 0.5
COHESION_POTENTIAL_SCALE = 3.0
COHESION_HOLD_WEIGHT = 2.0
LEADER_MAX_TRACTION_POWER = 300_000.0
LEADER_ENERGY_WEIGHT = 0.2
LEADER_JERK_WEIGHT = 0.2
LEADER_WAVE_WEIGHT = 1.0
LEADER_PBRS_WEIGHT = 1.0
LEADER_PBRS_DISCOUNT = 1.0
LEADER_RESTART_WEIGHT = 4.0
# Keep a clear-path recovery gradient beyond the initial crawl.  The ordinary
# V2I objective takes over above this speed and learns the final cruise target.
LEADER_RESTART_TARGET_SPEED = 15.0
LEADER_SPEED_DEFICIT_WEIGHT = 2.0
WAVE_ACTIVITY_THRESHOLD = 0.5
# Optional downstream-wave objective. It is disabled by default so existing
# checkpoints, evaluations, and replay fingerprints remain unchanged.
DOWNSTREAM_WAVE_WINDOW_STEPS = 50
DOWNSTREAM_WAVE_TARGET_GAIN = 0.85
DOWNSTREAM_WAVE_REWARD_WEIGHT = 1.0
# The lane-change objective is opt-in and therefore does not alter the reward
# fingerprint or behaviour of the already validated longitudinal experiments.
LANE_CHANGE_TRIGGER_DISTANCE = 210.0
LANE_CHANGE_COMPACT_GAP = 10.0
LANE_CHANGE_SETTLE_TIME = 0.5
LANE_CHANGE_PROGRESS_WEIGHT = 6.0
LANE_CHANGE_COMPLETION_BONUS = 8.0
LANE_CHANGE_TIME_WEIGHT = 0.04
LANE_CHANGE_SHAPE_WEIGHT = 0.8
LANE_CHANGE_ROAD_OCCUPANCY_WEIGHT = 0.15
LANE_CHANGE_SPEED_SPREAD_WEIGHT = 0.8
LANE_CHANGE_COMMAND_WEIGHT = 0.03
LANE_CHANGE_SUCCESS_BONUS = 25.0
LANE_CHANGE_TRAFFIC_SPEED_WEIGHT = 1.0
LANE_CHANGE_TRAFFIC_BRAKE_WEIGHT = 0.4
LANE_CHANGE_MIN_FRONT_HEADWAY = 0.35
LANE_CHANGE_MIN_REAR_HEADWAY = 0.50
LANE_CHANGE_REAR_LOOKAHEAD = 1.5
CONTINUOUS_LATERAL_RESOLUTION = 0.25
CONTINUOUS_LANE_CHANGE_REQUEST_DURATION = 12.0
CONTINUOUS_PLATOON_MAX_LATERAL_SPEED = 0.8
CONTINUOUS_PLATOON_LATERAL_ACCELERATION = 0.35
CONTINUOUS_PLATOON_MIN_LATERAL_GAP = 0.3
FOLLOWER_GAP_PBRS_DISCOUNT = 1.0
FOLLOWER_GAP_POTENTIAL_SCALE = 2.0
FOLLOWER_RESTART_WEIGHT = 8.0
FOLLOWER_TRACKING_WEIGHT = 2.0
FOLLOWER_GAP_TRACKING_WEIGHT = 2.0
FOLLOWER_GAP_PBRS_WEIGHT = 2.0
FOLLOWER_STRING_WEIGHT = 1.0
FOLLOWER_BARRIER_WEIGHT = 0.25
FOLLOWER_SHIELD_WEIGHT = 2.0
FOLLOWER_COMFORT_JERK_WEIGHT = 0.1
FOLLOWER_COMFORT_ACCELERATION_WEIGHT = 0.05
FOLLOWER_PREDICTION_HORIZON = 1.5
FOLLOWER_CBF_TIME_HEADWAY = 1.5
HISTORY_STEPS = 50
LEADER_PLAN_STEPS = 10
LEADER_PLAN_INTERVAL = 0.2
NEIGHBOR_HISTORY_STEPS = 20
MAX_FRONT_VEHICLES = 8
MAX_NEIGHBORS = 12
MAIN_EDGES = (
    "main0",
    "curve",
    "mergePre",
    "mainAlongRamp",
    "mergeOpen",
    "mergePost",
    "wave",
    "recovery",
)
MAIN_EDGE_START = {
    "main0": 0.0,
    "curve": 600.0,
    "mergePre": 1400.0,
    "mainAlongRamp": 1600.0,
    "mergeOpen": 1716.667,
    "mergePost": 2100.0,
    "wave": 2300.0,
    "recovery": 3100.0,
}
MAIN_EDGE_LENGTH = {
    "main0": 600.0,
    "curve": 800.0,
    "mergePre": 200.0,
    "mainAlongRamp": 116.667,
    "mergeOpen": 383.333,
    "mergePost": 200.0,
    "wave": 800.0,
    "recovery": 400.0,
}
# SUMO advances vehicles through internal junction lanes that are not part of
# the user-facing 0--3500 m scenario coordinate.  Omitting those lengths makes
# a predecessor jump backwards by 3 m at the ramp opening and by 8 m at the
# lane drop while its follower is still on the previous edge.  Keep a physical
# route coordinate for dynamics and map it back to the nominal coordinate only
# for scenario thresholds and road profiles.
MAIN_EDGE_JUNCTION_LENGTH = {
    "main0": 0.1,
    "curve": 0.1,
    "mergePre": 0.1,
    "mainAlongRamp": 3.0,
    "mergeOpen": 8.0,
    "mergePost": 0.1,
    "wave": 0.1,
    "recovery": 0.0,
}
MAIN_EDGE_DISTANCE_START: dict[str, float] = {}
_route_distance = 0.0
for _edge in MAIN_EDGES:
    MAIN_EDGE_DISTANCE_START[_edge] = _route_distance
    _route_distance += MAIN_EDGE_LENGTH[_edge]
    _route_distance += MAIN_EDGE_JUNCTION_LENGTH[_edge]
MAIN_ROUTE_DISTANCE_END = _route_distance
MAIN_INTERNAL_EDGE_DISTANCE_START = {
    ":n600_0": MAIN_EDGE_DISTANCE_START["main0"] + MAIN_EDGE_LENGTH["main0"],
    ":n1400_0": MAIN_EDGE_DISTANCE_START["curve"] + MAIN_EDGE_LENGTH["curve"],
    ":n1600_0": MAIN_EDGE_DISTANCE_START["mergePre"]
    + MAIN_EDGE_LENGTH["mergePre"],
    ":n1717_0": MAIN_EDGE_DISTANCE_START["mainAlongRamp"]
    + MAIN_EDGE_LENGTH["mainAlongRamp"],
    ":n2100_0": MAIN_EDGE_DISTANCE_START["mergeOpen"]
    + MAIN_EDGE_LENGTH["mergeOpen"],
    ":n2300_0": MAIN_EDGE_DISTANCE_START["mergePost"]
    + MAIN_EDGE_LENGTH["mergePost"],
    ":n3100_0": MAIN_EDGE_DISTANCE_START["wave"] + MAIN_EDGE_LENGTH["wave"],
}
RAMP_EDGE_DISTANCE_START = {
    # Align the ramp's merge point with the mainline physical coordinate.
    "ramp": MAIN_INTERNAL_EDGE_DISTANCE_START[":n1717_0"] - 516.767,
    "rampAccel": MAIN_INTERNAL_EDGE_DISTANCE_START[":n1717_0"] - 116.667,
}
RAMP_INTERNAL_EDGE_DISTANCE_START = {
    ":r1600_0": RAMP_EDGE_DISTANCE_START["ramp"] + 400.0,
    ":n1717_3": MAIN_INTERNAL_EDGE_DISTANCE_START[":n1717_0"],
}
PLATOON_IDS = (
    "leader",
    "follower_1",
    "follower_2",
    "follower_3",
    "follower_4",
    "follower_5",
)
FOLLOWER_IDS = PLATOON_IDS[1:]
PREDECESSOR = {
    "follower_1": "leader",
    "follower_2": "follower_1",
    "follower_3": "follower_2",
    "follower_4": "follower_3",
    "follower_5": "follower_4",
}
INITIAL_POSITIONS = {
    "leader": 170.0,
    "follower_1": 146.0,
    "follower_2": 122.0,
    "follower_3": 98.0,
    "follower_4": 74.0,
    "follower_5": 50.0,
}
CUTIN_ID = "cutin_aggressive"
FRONT_ID_PREFIX = "scenario_front_"
TAIL_BACKGROUND_ID_PREFIX = "tail_background_"
LANE_CHANGE_OBSTACLE_PREFIX = "lane_change_obstacle_"
LANE_CHANGE_TRAFFIC_PREFIX = "lane_change_traffic_"
FIXED_WAVE_PROFILE = np.asarray(
    [-2.2] * 20 + [-1.0] * 10 + [0.0] * 10 + [1.2] * 50 + [0.0] * 10,
    dtype=np.float32,
)
SEVERE_FIXED_WAVE_PROFILE = np.asarray(
    [-3.0] * 25
    + [-1.5] * 10
    + [0.0] * 5
    + [1.2] * 75
    + [0.0] * 5,
    dtype=np.float32,
)


def platoon_lane_index(station: float) -> int:
    """Physical lane index of the middle mainline lane at a route station."""

    nominal = nominal_mainline_station(station)
    return PLATOON_LANE_INDEX + int(1716.667 <= nominal < 2100.0)


def nominal_mainline_station(distance: float) -> float:
    """Map the continuous SUMO route distance to the scenario's 0--3500 m axis."""

    distance = float(np.clip(distance, 0.0, MAIN_ROUTE_DISTANCE_END))
    for edge in MAIN_EDGES:
        physical_start = MAIN_EDGE_DISTANCE_START[edge]
        physical_end = physical_start + MAIN_EDGE_LENGTH[edge]
        nominal_start = MAIN_EDGE_START[edge]
        if distance <= physical_end:
            return float(
                np.clip(
                    nominal_start + distance - physical_start,
                    nominal_start,
                    nominal_start + MAIN_EDGE_LENGTH[edge],
                )
            )
        connector_end = physical_end + MAIN_EDGE_JUNCTION_LENGTH[edge]
        if distance < connector_end:
            return nominal_start + MAIN_EDGE_LENGTH[edge]
    return 3500.0


@dataclass(frozen=True)
class LeaderPacket:
    """Timestamped leader broadcast stored by a follower's FIFO channel."""

    generated_at: float
    deliver_at: float
    station: float
    speed: float
    acceleration: float
    planned_acceleration: float = 0.0
    future_speeds: tuple[float, ...] = ()

    @property
    def delay(self) -> float:
        return self.deliver_at - self.generated_at


class V2VDelayBuffer:
    """FIFO channel with monotonic delivery and constant-acceleration prediction."""

    def __init__(self) -> None:
        self._queue: deque[LeaderPacket] = deque()
        self._latest: LeaderPacket | None = None
        self._last_delivery = -math.inf

    @property
    def latest(self) -> LeaderPacket | None:
        return self._latest

    def prime(
        self,
        timestamp: float,
        station: float,
        speed: float,
        acceleration: float,
        planned_acceleration: float = 0.0,
        future_speeds: tuple[float, ...] = (),
    ) -> None:
        packet = LeaderPacket(
            timestamp,
            timestamp,
            station,
            speed,
            acceleration,
            planned_acceleration,
            future_speeds,
        )
        self._latest = packet
        self._last_delivery = timestamp

    def push(
        self,
        timestamp: float,
        station: float,
        speed: float,
        acceleration: float,
        delay: float,
        planned_acceleration: float = 0.0,
        future_speeds: tuple[float, ...] = (),
    ) -> LeaderPacket:
        if delay != 0.0 and not V2V_DELAY_MIN <= delay <= V2V_DELAY_MAX:
            raise ValueError(
                "V2V delay must be zero for a normal packet or in "
                f"[{V2V_DELAY_MIN}, {V2V_DELAY_MAX}], got {delay}"
            )
        deliver_at = max(timestamp + delay, self._last_delivery)
        packet = LeaderPacket(
            timestamp,
            deliver_at,
            station,
            speed,
            acceleration,
            planned_acceleration,
            future_speeds,
        )
        self._queue.append(packet)
        self._last_delivery = deliver_at
        return packet

    def deliver(self, now: float) -> LeaderPacket | None:
        while self._queue and self._queue[0].deliver_at <= now + 1e-9:
            self._latest = self._queue.popleft()
        return self._latest

    def estimate(self, now: float) -> tuple[float, float, float, float, float]:
        if self._latest is None:
            raise RuntimeError("Delay buffer has not been primed")
        age = max(0.0, now - self._latest.generated_at)
        station = (
            self._latest.station
            + self._latest.speed * age
            + 0.5 * self._latest.acceleration * age * age
        )
        speed = max(0.0, self._latest.speed + self._latest.acceleration * age)
        return station, speed, self._latest.acceleration, self._latest.delay, age

    def plan(self) -> np.ndarray:
        if self._latest is None:
            raise RuntimeError("Delay buffer has not been primed")
        speeds = list(self._latest.future_speeds[:LEADER_PLAN_STEPS])
        speeds.extend([self._latest.speed] * (LEADER_PLAN_STEPS - len(speeds)))
        return np.asarray(
            [self._latest.planned_acceleration, *speeds], dtype=np.float32
        )


class PlatoonParallelEnv(ParallelEnv):
    """Six-truck SUMO platoon environment with causal V2X observations."""

    metadata = {
        "name": "sumo_platoon_v0",
        "render_modes": ["human"],
        "render_fps": 10,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        step_length: float = STEP_LENGTH,
        max_time: float = MAX_TIME,
        calibration_path: str | Path = "artifacts/calibration/calibrated_parameters.json",
        wave_model_path: str | Path = "artifacts/checkpoints/wave_best.pt",
        gui_auto_start: bool = False,
        continuous_lateral_model: bool = False,
        gui_delay_ms: int = 100,
    ) -> None:
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if not math.isclose(step_length, STEP_LENGTH):
            raise ValueError("This scenario is calibrated for a 0.1 s step length")

        self.render_mode = render_mode
        self.step_length = step_length
        self.max_time = max_time
        self.calibration_path = Path(calibration_path)
        self.wave_model_path = Path(wave_model_path)
        self.gui_auto_start = bool(gui_auto_start)
        self.continuous_lateral_model = bool(continuous_lateral_model)
        self.gui_delay_ms = int(gui_delay_ms)
        if self.gui_delay_ms < 0:
            raise ValueError("gui_delay_ms must be non-negative")
        self.possible_agents = list(PLATOON_IDS)
        self.agents: list[str] = []
        self.connection: Any | None = None
        self.connection_label: str | None = None
        self.sim_time = 0.0
        self._station_cache: dict[str, float] = {}
        self._leader_delay_buffers: dict[str, V2VDelayBuffer] = {}
        self._predecessor_delay_buffers: dict[str, V2VDelayBuffer] = {}
        self._v2v_history: dict[str, deque[np.ndarray]] = {}
        self._communication_delay_probability = V2V_DELAY_EVENT_PROBABILITY
        self._communication_delay_min = V2V_DELAY_MIN
        self._communication_delay_max = V2V_DELAY_MAX
        self._communication_delay_events = 0
        self._vehicle_history: dict[str, deque[np.ndarray]] = {}
        self._previous_acceleration: dict[str, float] = {}
        self._last_jerk: dict[str, float] = {}
        self._previous_command: dict[str, float] = {}
        self._collision_shield_active: dict[str, bool] = {}
        self._last_action_info: dict[str, dict[str, Any]] = {}
        self._follower_control_predictions: dict[str, tuple[float, float]] = {}
        self._leader_restart_opportunity = False
        self._leader_stall_duration = 0.0
        self._stalled_failure = False
        self._last_cutin_threat_count = 0
        self._last_platoon_intrusion_count = 0
        self._recovery_leaders: set[str] = set()
        self._recovery_started_at: dict[str, float] = {}
        self._recovery_clear_since: dict[str, float] = {}
        self._disconnect_since: dict[str, float] = {}
        self._previous_pair_gaps: dict[str, float] = {}
        self._previous_gap_potentials: dict[str, float] = {}
        self._previous_cohesion_potential = 0.0
        self._rejoined_this_step = 0
        self._rejoined_vehicles_this_step: set[str] = set()
        self._cutin_state = "idle"
        self._cutin_enabled = True
        self._v2i_enabled = True
        self._speed_reference_override: float | None = None
        self._cutin_configured = False
        self._cutin_merge_committed = False
        self._cutin_open_merge_logged = False
        self._cutin_error: str | None = None
        self._cutin_hold_until = -math.inf
        self._cutin_released = False
        self._cutin_next_lane_change_time = -math.inf
        self._cutin_waiting_lanes: set[tuple[str, int]] = set()
        self._cutin_clearance_scale = 1.0
        self._cutin_force_request = False
        self._cutin_depart_speed_mps = 8.0
        self._wave_started_at: float | None = None
        self._next_wave_time: float | None = None
        self._wave_brake_until = -math.inf
        self._wave_target: str | None = None
        self._wave_pending = False
        self._wave_mode = "learned"
        self._wave_policy: Any | None = None
        self._configured_wave_model_path = self.wave_model_path
        self._wave_profile: np.ndarray | None = None
        self._wave_profile_index = 0
        self._downstream_wave_reward_enabled = False
        self._downstream_wave_target_gain = DOWNSTREAM_WAVE_TARGET_GAIN
        self._downstream_wave_reward_weight = DOWNSTREAM_WAVE_REWARD_WEIGHT
        self._wave_trigger_stations: tuple[float, ...] = ()
        self._wave_trigger_index = 0
        self._wave_trigger_reference = "leader"
        self._front_vehicle_count = 0
        self._front_ids: list[str] = []
        self._front_initial_positions: dict[str, float] = {}
        self._front_initial_speeds: dict[str, float] = {}
        self._front_speed_released = True
        self._selected_wave_source: str | None = None
        self._background_types: list[str] = ["human_normal"]
        self._background_probabilities = np.ones(1, dtype=np.float64)
        self._background_rates = {"main": 900.0, "ramp": 450.0}
        self._background_next: dict[str, float] = {}
        self._background_counter = 0
        self._background_initial_ids: list[str] = []
        self._background_initial_scale = 1.0
        self._background_max_speeds: dict[str, float] = {"human_normal": 36.0}
        self._background_speed_configured: set[str] = set()
        self._background_lane_change_guarded: set[str] = set()
        self._ramp_origin_ids: set[str] = set()
        self._ramp_merge_requested_at: dict[str, float] = {}
        self._ramp_merge_wait_logged: set[str] = set()
        self._ramp_queue_brake_logged: set[str] = set()
        self._ramp_merge_released: set[str] = set()
        self._ramp_meter_closed = False
        self._last_collision_ids: list[str] = []
        self._last_collision_details: list[dict[str, Any]] = []
        self._demand_profile_id = -1
        self._event_log: list[dict[str, Any]] = []
        self._cooperative_lane_change_enabled = False
        self._lane_change_obstacle_stations: tuple[float, ...] = ()
        self._lane_change_obstacle_lanes: tuple[int, ...] = ()
        self._lane_change_target_lanes: tuple[int, ...] = ()
        self._lane_change_obstacle_speed = 8.0
        self._lane_change_goal_station = 1500.0
        self._lane_change_obstacle_ids: list[str] = []
        self._lane_change_event_index = 0
        self._lane_change_event_active = False
        self._lane_change_event_started_at: float | None = None
        self._lane_change_stall_grace = LANE_CHANGE_STALL_GRACE
        self._lane_change_settled_since: float | None = None
        self._lane_change_completed_this_step = 0
        self._lane_change_previous_potential = 0.0
        self._lane_change_command_count = 0
        self._lane_change_commands_this_step = 0
        self._lane_change_dual_lane_time = 0.0
        self._lane_change_first_command_at: float | None = None
        self._lane_change_execution_time = 0.0
        self._lane_change_traffic_enabled = False
        self._lane_change_traffic_front_gaps: tuple[float, ...] = ()
        self._lane_change_traffic_rear_gaps: tuple[float, ...] = ()
        self._lane_change_traffic_front_speeds: tuple[float, ...] = ()
        self._lane_change_traffic_rear_speeds: tuple[float, ...] = ()
        self._lane_change_traffic_ids: list[str] = []
        self._lane_change_traffic_desired_speeds: dict[str, float] = {}
        self._lane_change_traffic_spawned_events: set[int] = set()
        self._lane_change_traffic_delay = 0.0
        self._lane_change_traffic_braking_integral = 0.0
        self._lane_change_traffic_hard_brake_events = 0
        self._lane_change_traffic_hard_brake_active: dict[str, bool] = {}
        self._lane_change_traffic_speed_deficit = 0.0
        self._lane_change_traffic_brake_intensity = 0.0
        self._lane_change_traffic_max_deceleration = 0.0
        self._longitudinal_action_spaces = {
            agent: self._make_action_space(
                lane_change_actions=3 if agent == "leader" else 1
            )
            for agent in self.possible_agents
        }
        self._cooperative_action_spaces = {
            agent: self._make_action_space(lane_change_actions=3)
            for agent in self.possible_agents
        }

    @property
    def event_log(self) -> tuple[dict[str, Any], ...]:
        """Immutable snapshot of scenario events emitted in the current episode."""

        return tuple(event.copy() for event in self._event_log)

    @staticmethod
    def _make_action_space(lane_change_actions: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "accel": spaces.Box(
                    low=np.array([MAX_BRAKING], dtype=np.float32),
                    high=np.array([MAX_ACCELERATION], dtype=np.float32),
                    dtype=np.float32,
                ),
                "lane_change": spaces.Discrete(lane_change_actions),
            }
        )

    def action_space(self, agent: str) -> spaces.Dict:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        action_spaces = (
            self._cooperative_action_spaces
            if self._cooperative_lane_change_enabled
            else self._longitudinal_action_spaces
        )
        return action_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        platoon_low = np.tile(
            np.array([-3600.0, 0.0, -15.0, -1.0, -1.0, 0.0], dtype=np.float32),
            (len(PLATOON_IDS), 1),
        )
        platoon_high = np.tile(
            np.array([3600.0, 60.0, 10.0, 3.0, 3600.0, 1.0], dtype=np.float32),
            (len(PLATOON_IDS), 1),
        )
        track_low = np.tile(
            np.array([-1000.0, -3.0, 0.0, -15.0, 0.0, 0.0], dtype=np.float32),
            (MAX_FRONT_VEHICLES, HISTORY_STEPS, 1),
        )
        track_high = np.tile(
            np.array([1000.0, 3.0, 60.0, 10.0, 30.0, 1.0], dtype=np.float32),
            (MAX_FRONT_VEHICLES, HISTORY_STEPS, 1),
        )
        neighbor_low = np.tile(
            np.array([-500.0, -3.0, 0.0, -15.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            (MAX_NEIGHBORS, NEIGHBOR_HISTORY_STEPS, 1),
        )
        neighbor_high = np.tile(
            np.array([500.0, 3.0, 60.0, 10.0, 30.0, 3.0, 1.0, 1.0], dtype=np.float32),
            (MAX_NEIGHBORS, NEIGHBOR_HISTORY_STEPS, 1),
        )
        return spaces.Dict(
            {
                "ego": spaces.Box(
                    np.array([0.0, 0.0, -15.0, -30.0, -1.0, 0.0, -0.1, 0.0], dtype=np.float32),
                    np.array([3600.0, 60.0, 10.0, 30.0, 3.0, 1.0, 0.1, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "platoon_state": spaces.Box(platoon_low, platoon_high, dtype=np.float32),
                "front_history": spaces.Box(track_low, track_high, dtype=np.float32),
                "front_mask": spaces.Box(0.0, 1.0, shape=(MAX_FRONT_VEHICLES,), dtype=np.float32),
                "neighbor_history": spaces.Box(neighbor_low, neighbor_high, dtype=np.float32),
                "neighbor_mask": spaces.Box(0.0, 1.0, shape=(MAX_NEIGHBORS,), dtype=np.float32),
                "v2v_history": spaces.Box(
                    low=np.tile(np.array([0.0, 0.0, -15.0, 0.0] * 2, dtype=np.float32), (HISTORY_STEPS, 1)),
                    high=np.tile(np.array([3600.0, 60.0, 10.0, 2.0] * 2, dtype=np.float32), (HISTORY_STEPS, 1)),
                    dtype=np.float32,
                ),
                "leader_plan": spaces.Box(
                    low=np.array(
                        [MAX_BRAKING, *([0.0] * LEADER_PLAN_STEPS)],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [MAX_ACCELERATION, *([60.0] * LEADER_PLAN_STEPS)],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
                "local_predecessor": spaces.Box(
                    np.array([-500.0, -60.0, -15.0, -1.0, 0.0, 0.0], dtype=np.float32),
                    np.array([500.0, 60.0, 10.0, 3.0, 500.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "local_front": spaces.Box(
                    np.array([-500.0, -60.0, -15.0, -1.0, 0.0, 0.0], dtype=np.float32),
                    np.array([500.0, 60.0, 10.0, 3.0, 500.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "v2i_flow": spaces.Box(0.0, np.tile(np.array([60.0, 300.0, 60.0, 1.0], dtype=np.float32), (8, 1)), dtype=np.float32),
                "road_profile": spaces.Box(
                    np.tile(np.array([0.0, -0.1, 0.0], dtype=np.float32), (20, 1)),
                    np.tile(np.array([500.0, 0.1, 40.0], dtype=np.float32), (20, 1)),
                    dtype=np.float32,
                ),
                "signal": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "signal_mask": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "communication": spaces.Box(0.0, 2.0, shape=(4,), dtype=np.float32),
                "recovery": spaces.Box(
                    np.array([0.0, 0.0, -60.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([1.0, 500.0, 60.0, 1.0, 40.0, 240.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "lane_mission": spaces.Box(
                    low=np.array(
                        [0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
            }
        )

    def _load_calibration(self) -> tuple[dict[str, Any] | None, Path | None]:
        if not self.calibration_path.exists():
            return None, None
        calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        vtype_file = self.calibration_path.parent / "ngsim_vtypes.add.xml"
        return calibration, vtype_file if vtype_file.exists() else None

    def _configure_episode(
        self, options: dict[str, Any], calibration: dict[str, Any] | None
    ) -> None:
        self._cooperative_lane_change_enabled = bool(
            options.get("cooperative_lane_change_enabled", False)
        )
        default_obstacle_stations = (680.0, 1010.0, 1320.0)
        default_obstacle_lanes = (1, 2, 1)
        default_target_lanes = (2, 1, 0)
        self._lane_change_obstacle_stations = tuple(
            float(value)
            for value in options.get(
                "lane_change_obstacle_stations", default_obstacle_stations
            )
        )
        self._lane_change_obstacle_lanes = tuple(
            int(value)
            for value in options.get(
                "lane_change_obstacle_lanes", default_obstacle_lanes
            )
        )
        self._lane_change_target_lanes = tuple(
            int(value)
            for value in options.get(
                "lane_change_target_lanes", default_target_lanes
            )
        )
        self._lane_change_obstacle_speed = float(
            options.get("lane_change_obstacle_speed_mps", 8.0)
        )
        self._lane_change_goal_station = float(
            options.get("lane_change_goal_station", 1500.0)
        )
        self._lane_change_stall_grace = float(
            options.get(
                "lane_change_gap_wait_grace_s", LANE_CHANGE_STALL_GRACE
            )
        )
        self._lane_change_traffic_enabled = bool(
            options.get("lane_change_traffic_enabled", False)
        )
        self._lane_change_traffic_front_gaps = tuple(
            float(value)
            for value in options.get(
                "lane_change_traffic_front_gaps_m", (38.0, 34.0, 30.0)
            )
        )
        self._lane_change_traffic_rear_gaps = tuple(
            float(value)
            for value in options.get(
                "lane_change_traffic_rear_gaps_m", (24.0, 21.0, 18.0)
            )
        )
        self._lane_change_traffic_front_speeds = tuple(
            float(value)
            for value in options.get(
                "lane_change_traffic_front_speeds_mps", (22.5, 23.0, 22.0)
            )
        )
        self._lane_change_traffic_rear_speeds = tuple(
            float(value)
            for value in options.get(
                "lane_change_traffic_rear_speeds_mps", (26.0, 26.5, 27.0)
            )
        )
        lane_counts = {
            len(self._lane_change_obstacle_stations),
            len(self._lane_change_obstacle_lanes),
            len(self._lane_change_target_lanes),
        }
        if self._cooperative_lane_change_enabled and (
            lane_counts != {3}
            or any(lane not in (0, 1, 2) for lane in self._lane_change_obstacle_lanes)
            or any(lane not in (0, 1, 2) for lane in self._lane_change_target_lanes)
        ):
            raise ValueError(
                "cooperative lane change requires three obstacle stations, "
                "obstacle lanes, and target lanes in [0, 2]"
            )
        if self._cooperative_lane_change_enabled and not (
            tuple(sorted(self._lane_change_obstacle_stations))
            == self._lane_change_obstacle_stations
            and 5.0 <= self._lane_change_obstacle_speed <= 15.0
            and self._lane_change_goal_station
            > self._lane_change_obstacle_stations[-1] + 80.0
        ):
            raise ValueError("invalid cooperative lane-change scenario geometry")
        if self._lane_change_stall_grace < 0.0 or (
            self._cooperative_lane_change_enabled
            and self._lane_change_stall_grace > self.max_time
        ):
            raise ValueError("invalid lane-change gap-wait grace period")
        traffic_lengths = {
            len(self._lane_change_traffic_front_gaps),
            len(self._lane_change_traffic_rear_gaps),
            len(self._lane_change_traffic_front_speeds),
            len(self._lane_change_traffic_rear_speeds),
        }
        if self._lane_change_traffic_enabled and (
            not self._cooperative_lane_change_enabled
            or traffic_lengths != {3}
            or not all(
                24.0 <= value <= 60.0
                for value in self._lane_change_traffic_front_gaps
            )
            or not all(
                14.0 <= value <= 45.0
                for value in self._lane_change_traffic_rear_gaps
            )
            or not all(
                18.0 <= value <= 29.0
                for value in (
                    *self._lane_change_traffic_front_speeds,
                    *self._lane_change_traffic_rear_speeds,
                )
            )
        ):
            raise ValueError(
                "mixed lane-change traffic requires three feasible front/rear "
                "gaps and speeds with cooperative lane changing enabled"
            )
        self._cutin_enabled = bool(options.get("cutin_enabled", True))
        self._cutin_clearance_scale = float(
            options.get("cutin_clearance_scale", 1.0)
        )
        self._cutin_force_request = bool(
            options.get("cutin_force_request", False)
        )
        self._cutin_depart_speed_mps = float(
            options.get("cutin_depart_speed_mps", 8.0)
        )
        if not 0.25 <= self._cutin_clearance_scale <= 1.5:
            raise ValueError("cutin_clearance_scale must be in [0.25, 1.5]")
        if not 4.0 <= self._cutin_depart_speed_mps <= 14.0:
            raise ValueError("cutin_depart_speed_mps must be in [4, 14] m/s")
        self._v2i_enabled = bool(options.get("v2i_enabled", True))
        requested_initial_station = options.get("initial_leader_station_m")
        self._initial_leader_station_m = (
            None
            if requested_initial_station is None
            else float(requested_initial_station)
        )
        if self._initial_leader_station_m is None:
            self._initial_position_offset_m = float(
                options.get("initial_position_offset_m", 0.0)
            )
            if not 0.0 <= self._initial_position_offset_m <= 300.0:
                raise ValueError("initial_position_offset_m must be in [0, 300] m")
        else:
            if not 170.0 <= self._initial_leader_station_m <= 3200.0:
                raise ValueError(
                    "initial_leader_station_m must be in [170, 3200] m"
                )
            self._initial_position_offset_m = (
                self._initial_leader_station_m - INITIAL_POSITIONS["leader"]
            )
        self._episode_initial_positions = {
            vehicle_id: position + self._initial_position_offset_m
            for vehicle_id, position in INITIAL_POSITIONS.items()
        }
        self._tail_background_count = int(options.get("tail_background_count", 0))
        self._tail_background_gap_m = float(
            options.get("tail_background_gap_m", 12.0)
        )
        self._tail_background_depart_speed_mps = float(
            options.get("tail_background_depart_speed_mps", 0.0)
        )
        self._tail_background_type = str(
            options.get("tail_background_type", "human_normal")
        )
        self._downstream_wave_reward_enabled = bool(
            options.get("downstream_wave_reward_enabled", False)
        )
        self._downstream_wave_target_gain = float(
            options.get(
                "downstream_wave_target_gain", DOWNSTREAM_WAVE_TARGET_GAIN
            )
        )
        self._downstream_wave_reward_weight = float(
            options.get(
                "downstream_wave_reward_weight",
                DOWNSTREAM_WAVE_REWARD_WEIGHT,
            )
        )
        if not 0 <= self._tail_background_count <= 5:
            raise ValueError("tail_background_count must be in [0, 5]")
        if not 5.0 <= self._tail_background_gap_m <= 60.0:
            raise ValueError("tail_background_gap_m must be in [5, 60] m")
        if not 0.0 <= self._tail_background_depart_speed_mps <= TARGET_SPEED:
            raise ValueError(
                "tail_background_depart_speed_mps must be in [0, TARGET_SPEED]"
            )
        if self._tail_background_type not in {"human_normal", "human_aggressive"}:
            raise ValueError(
                "tail_background_type must be 'human_normal' or 'human_aggressive'"
            )
        if self._downstream_wave_reward_enabled and self._tail_background_count != 5:
            raise ValueError(
                "downstream_wave_reward_enabled requires five tail background vehicles"
            )
        if not 0.0 < self._downstream_wave_target_gain <= 1.5:
            raise ValueError("downstream_wave_target_gain must be in (0, 1.5]")
        if not 0.0 <= self._downstream_wave_reward_weight <= 10.0:
            raise ValueError("downstream_wave_reward_weight must be in [0, 10]")
        self._tail_background_ids = [
            f"{TAIL_BACKGROUND_ID_PREFIX}{index}"
            for index in range(1, self._tail_background_count + 1)
        ]
        self._tail_initial_positions: dict[str, float] = {}
        speed_reference = options.get("speed_reference")
        self._speed_reference_override = (
            None if speed_reference is None else float(speed_reference)
        )
        if (
            self._speed_reference_override is not None
            and not 0.0 < self._speed_reference_override <= 30.0
        ):
            raise ValueError("speed_reference must be in (0, 30] m/s")
        self._communication_delay_probability = float(
            options.get(
                "communication_delay_probability", V2V_DELAY_EVENT_PROBABILITY
            )
        )
        self._communication_delay_min = float(
            options.get("communication_delay_min", V2V_DELAY_MIN)
        )
        self._communication_delay_max = float(
            options.get("communication_delay_max", V2V_DELAY_MAX)
        )
        if not 0.0 <= self._communication_delay_probability <= 1.0:
            raise ValueError("communication_delay_probability must be in [0, 1]")
        if not (
            V2V_DELAY_MIN
            <= self._communication_delay_min
            <= self._communication_delay_max
            <= V2V_DELAY_MAX
        ):
            raise ValueError(
                "communication delay events must remain within [0.1, 0.2] seconds"
            )
        requested_count = options.get("front_vehicle_count")
        self._front_vehicle_count = (
            int(self.np_random.integers(0, 9))
            if requested_count is None
            else int(requested_count)
        )
        if not 0 <= self._front_vehicle_count <= 8:
            raise ValueError("front_vehicle_count must be in [0, 8]")
        self._front_ids = [
            f"{FRONT_ID_PREFIX}{index}" for index in range(self._front_vehicle_count)
        ]
        self._front_initial_positions = {}
        self._front_initial_speeds = {}
        self._front_speed_released = True
        self._selected_wave_source = (
            str(self.np_random.choice(self._front_ids)) if self._front_ids else None
        )
        self._wave_mode = str(options.get("wave_mode", "learned"))
        if self._wave_mode not in ("learned", "fixed", "off"):
            raise ValueError("wave_mode must be 'learned', 'fixed', or 'off'")
        self._fixed_wave_profile = str(
            options.get("fixed_wave_profile", "standard")
        )
        if self._fixed_wave_profile not in ("standard", "severe"):
            raise ValueError(
                "fixed_wave_profile must be 'standard' or 'severe'"
            )
        if self._front_vehicle_count == 0:
            self._wave_mode = "off"
        self._wave_min_source_distance_m = float(
            options.get("wave_min_source_distance_m", 20.0)
        )
        if not 0.0 <= self._wave_min_source_distance_m <= 50.0:
            raise ValueError("wave_min_source_distance_m must be in [0, 50] m")
        requested_wave_stations = options.get("wave_trigger_stations")
        self._wave_trigger_stations = (
            ()
            if requested_wave_stations is None
            else tuple(float(value) for value in requested_wave_stations)
        )
        if self._wave_trigger_stations and (
            any(
                not 2_400.0 <= station <= 3_000.0
                for station in self._wave_trigger_stations
            )
            or tuple(sorted(set(self._wave_trigger_stations)))
            != self._wave_trigger_stations
        ):
            raise ValueError(
                "wave_trigger_stations must be unique, increasing, and within "
                "[2400, 3000] m"
            )
        self._wave_trigger_reference = str(
            options.get("wave_trigger_reference", "leader")
        )
        if self._wave_trigger_reference not in {"leader", "source"}:
            raise ValueError("wave_trigger_reference must be 'leader' or 'source'")
        if not self._wave_trigger_stations:
            self._wave_trigger_reference = "leader"
        self._wave_policy = None
        if self._wave_mode == "learned":
            self._configured_wave_model_path = Path(
                options.get("wave_model_path", self.wave_model_path)
            )

        if calibration is None:
            self._background_types = ["human_normal"]
            self._background_probabilities = np.ones(1, dtype=np.float64)
            self._background_max_speeds = {"human_normal": 36.0}
            profiles = [{"profile_id": -1, "main_per_lane_veh_per_hour": 900.0, "ramp_veh_per_hour": 450.0}]
            demand_scale = 1.0
        else:
            self._background_types = [item["type_id"] for item in calibration["types"]]
            probabilities = np.asarray(
                [item["probability"] for item in calibration["types"]], dtype=np.float64
            )
            self._background_probabilities = probabilities / probabilities.sum()
            free_flow = calibration.get("free_flow_speed_p99", {})
            self._background_max_speeds = {
                item["type_id"]: max(
                    float(item["idm"]["max_speed"]),
                    float(free_flow.get(str(item["vehicle_class"]), item["idm"]["max_speed"])),
                )
                for item in calibration["types"]
            }
            profiles = calibration["demand"].get("minute_profiles") or [calibration["demand"]]
            demand_scale = float(calibration.get("demand_scale", 1.0))
        requested_profile = options.get("demand_profile")
        if requested_profile is None:
            profile = profiles[int(self.np_random.integers(len(profiles)))]
        else:
            matches = [item for item in profiles if int(item.get("profile_id", -1)) == int(requested_profile)]
            if not matches:
                raise ValueError(f"Unknown demand_profile {requested_profile}")
            profile = matches[0]
        self._demand_profile_id = int(profile.get("profile_id", -1))
        self._background_rates = {
            "main": max(0.0, float(profile["main_per_lane_veh_per_hour"]) * demand_scale),
            "ramp": max(0.0, float(profile["ramp_veh_per_hour"]) * demand_scale),
        }
        self._background_counter = 0
        self._background_initial_ids = []
        self._background_initial_scale = float(
            options.get("background_initial_scale", 1.0)
        )
        if not 0.0 <= self._background_initial_scale <= 2.0:
            raise ValueError("background_initial_scale must be in [0, 2]")
        background_demand_scale = float(
            options.get("background_demand_scale", 1.0)
        )
        if not 0.0 <= background_demand_scale <= 2.0:
            raise ValueError("background_demand_scale must be in [0, 2]")
        self._background_rates = {
            stream: rate * background_demand_scale
            for stream, rate in self._background_rates.items()
        }
        self._background_speed_configured = set()
        self._background_next = {}
        for stream in ("main_0", "main_1", "main_2", "ramp"):
            rate = self._background_rates["ramp" if stream == "ramp" else "main"]
            self._background_next[stream] = (
                5.0 + float(self.np_random.exponential(3600.0 / rate))
                if rate > 0.0
                else math.inf
            )

    def _sample_background_type(self) -> str:
        return str(
            self.np_random.choice(
                self._background_types, p=self._background_probabilities
            )
        )

    def _spawn_front_chain(self) -> None:
        previous_position = self._episode_initial_positions["leader"]
        for vehicle_id in self._front_ids:
            type_id = "wave_generator"
            length = float(self.connection.vehicletype.getLength(type_id))
            net_gap = float(self.np_random.uniform(14.0, 22.0))
            position = previous_position + net_gap + length
            self.connection.vehicle.add(
                vehicle_id,
                routeID="mainRoute",
                typeID=type_id,
                depart="now",
                departLane=str(PLATOON_LANE_INDEX),
                departPos=f"{position:.3f}",
                departSpeed=f"{TARGET_SPEED:.3f}",
            )
            self._front_initial_positions[vehicle_id] = position
            previous_position = position

    def _spawn_lane_change_obstacles(self) -> None:
        if not self._cooperative_lane_change_enabled:
            return
        for index, (station, lane) in enumerate(
            zip(
                self._lane_change_obstacle_stations,
                self._lane_change_obstacle_lanes,
            ),
            start=1,
        ):
            edge, edge_position, _ = self._route_from_station(station)
            vehicle_id = f"{LANE_CHANGE_OBSTACLE_PREFIX}{index}"
            self.connection.vehicle.add(
                vehicle_id,
                routeID=self._background_route_id(edge),
                typeID="wave_generator",
                depart="now",
                departLane=str(lane),
                departPos=f"{edge_position:.3f}",
                departSpeed=f"{self._lane_change_obstacle_speed:.3f}",
            )
            self._lane_change_obstacle_ids.append(vehicle_id)

    def _spawn_lane_change_traffic_gate(
        self, event_index: int, target_lane: int
    ) -> None:
        """Create a paired moving gap around the platoon in the target lane."""

        if (
            not self._lane_change_traffic_enabled
            or event_index in self._lane_change_traffic_spawned_events
        ):
            return
        car_type = "human_aggressive"
        car_length = float(self.connection.vehicletype.getLength(car_type))
        leader_station = nominal_mainline_station(self._station("leader"))
        tail_station = nominal_mainline_station(self._station("follower_5"))
        front_gap = self._lane_change_traffic_front_gaps[event_index]
        rear_gap = self._lane_change_traffic_rear_gaps[event_index]
        front_speed = self._lane_change_traffic_front_speeds[event_index]
        rear_speed = self._lane_change_traffic_rear_speeds[event_index]
        front_position = leader_station + front_gap + car_length
        rear_position = tail_station - PLATOON_LENGTH - rear_gap
        specifications = (
            ("front", front_position, front_speed),
            ("front_queue", front_position + car_length + 30.0, front_speed + 0.5),
            ("rear", rear_position, rear_speed),
            ("rear_queue", rear_position - car_length - 28.0, max(18.0, rear_speed - 0.5)),
        )
        spawned: list[str] = []
        for role, station, desired_speed in specifications:
            edge, edge_position, _ = self._route_from_station(station)
            vehicle_id = (
                f"{LANE_CHANGE_TRAFFIC_PREFIX}{event_index + 1}_{role}"
            )
            self.connection.vehicle.add(
                vehicle_id,
                routeID=self._background_route_id(edge),
                typeID=car_type,
                depart="now",
                departLane=str(target_lane),
                departPos=f"{edge_position:.3f}",
                # SUMO selects the maximum collision-free insertion speed. The
                # desired cruising speed is imposed through maxSpeed below, so
                # traffic gates never rely on an infeasible forced departure.
                departSpeed="max",
            )
            self.connection.vehicle.setLaneChangeMode(
                vehicle_id, SAFETY_ONLY_LANE_CHANGE_MODE
            )
            self.connection.vehicle.setMaxSpeed(vehicle_id, desired_speed)
            self.connection.vehicle.setColor(vehicle_id, (105, 105, 105, 255))
            self._lane_change_traffic_ids.append(vehicle_id)
            self._lane_change_traffic_desired_speeds[vehicle_id] = desired_speed
            self._lane_change_traffic_hard_brake_active[vehicle_id] = False
            spawned.append(vehicle_id)
        self._lane_change_traffic_spawned_events.add(event_index)
        self._event_log.append(
            {
                "time": self.sim_time,
                "event": "lane_change_traffic_gate_spawned",
                "maneuver_index": event_index + 1,
                "target_lane": target_lane,
                "front_gap_m": front_gap,
                "rear_gap_m": rear_gap,
                "front_speed_mps": front_speed,
                "rear_speed_mps": rear_speed,
                "vehicles": spawned,
            }
        )

    def _update_lane_change_traffic_metrics(self, active: set[str]) -> None:
        present = [
            vehicle_id
            for vehicle_id in self._lane_change_traffic_ids
            if vehicle_id in active
        ]
        if not present:
            self._lane_change_traffic_speed_deficit = 0.0
            self._lane_change_traffic_brake_intensity = 0.0
            return
        deficits: list[float] = []
        braking: list[float] = []
        for vehicle_id in present:
            desired = max(
                self._lane_change_traffic_desired_speeds.get(vehicle_id, TARGET_SPEED),
                1.0,
            )
            speed = max(0.0, float(self.connection.vehicle.getSpeed(vehicle_id)))
            acceleration = float(
                self.connection.vehicle.getAcceleration(vehicle_id)
            )
            deficit = max(0.0, 1.0 - speed / desired)
            brake = max(0.0, -acceleration)
            deficits.append(deficit)
            braking.append(brake)
            self._lane_change_traffic_delay += deficit * self.step_length
            self._lane_change_traffic_braking_integral += brake * self.step_length
            self._lane_change_traffic_max_deceleration = max(
                self._lane_change_traffic_max_deceleration, brake
            )
            hard = acceleration <= -3.0
            if hard and not self._lane_change_traffic_hard_brake_active.get(
                vehicle_id, False
            ):
                self._lane_change_traffic_hard_brake_events += 1
            self._lane_change_traffic_hard_brake_active[vehicle_id] = hard
        self._lane_change_traffic_speed_deficit = float(np.mean(deficits))
        self._lane_change_traffic_brake_intensity = float(np.mean(braking))

    def _spawn_tail_background_chain(self) -> None:
        """Insert fixed human-driven followers for downstream wave evaluation."""

        previous_position = self._episode_initial_positions["follower_5"]
        previous_length = PLATOON_LENGTH
        for vehicle_id in self._tail_background_ids:
            position = (
                previous_position
                - previous_length
                - self._tail_background_gap_m
            )
            if position < 5.0:
                raise ValueError(
                    "Tail background chain does not fit on the initial edge; "
                    "increase initial_position_offset_m or reduce its spacing"
                )
            self.connection.vehicle.add(
                vehicle_id,
                routeID="mainRoute",
                typeID=self._tail_background_type,
                depart="now",
                departLane=str(PLATOON_LANE_INDEX),
                departPos=f"{position:.3f}",
                departSpeed=f"{self._tail_background_depart_speed_mps:.3f}",
            )
            self._tail_initial_positions[vehicle_id] = position
            previous_position = position
            previous_length = float(
                self.connection.vehicletype.getLength(self._tail_background_type)
            )

    def _background_route_id(self, edge: str) -> str:
        route_id = f"backgroundFrom_{edge}"
        if route_id not in self.connection.route.getIDList():
            start = MAIN_EDGES.index(edge)
            self.connection.route.add(route_id, list(MAIN_EDGES[start:]))
        return route_id

    def _add_initial_background(self, station: float, lane: int) -> None:
        edge, edge_position, _ = self._route_from_station(station)
        vehicle_id = f"background_{self._background_counter}"
        self._background_counter += 1
        self.connection.vehicle.add(
            vehicle_id,
            routeID=self._background_route_id(edge),
            typeID=self._sample_background_type(),
            depart="now",
            departLane=str(lane + int(edge == "mergeOpen")),
            departPos=f"{edge_position:.3f}",
            departSpeed="desired",
        )
        self._background_initial_ids.append(vehicle_id)

    def _spawn_initial_background(self) -> None:
        rate = self._background_rates["main"] * self._background_initial_scale
        if rate <= 0.0:
            return
        # Convert the calibrated flow to a spatial density using a motorway
        # operating speed. Vehicle-specific IDM parameters still determine
        # the subsequent car-following and lane-changing behavior.
        mean_spacing = float(np.clip(TARGET_SPEED * 3600.0 / rate, 28.0, 180.0))
        protected_start = (
            min(
                self._tail_initial_positions.values(),
                default=self._episode_initial_positions["follower_5"],
            )
            - PLATOON_LENGTH
            - 8.0
        )
        front_end = max(
            self._front_initial_positions.values(),
            default=self._episode_initial_positions["leader"],
        )
        protected_end = front_end + 30.0

        # Ensure that the middle-lane platoon has a real human follower from
        # the first simulation step, while preserving its insertion gap.
        if not self._tail_background_ids:
            self._add_initial_background(
                max(5.0, protected_start - 15.0), PLATOON_LANE_INDEX
            )
        for lane in range(3):
            station = float(self.np_random.uniform(8.0, min(mean_spacing, 80.0)))
            while station < 3470.0:
                if not (
                    lane == PLATOON_LANE_INDEX
                    and station <= protected_end
                ):
                    self._add_initial_background(station, lane)
                station += max(
                    25.0,
                    float(self.np_random.gamma(shape=4.0, scale=mean_spacing / 4.0)),
                )

    def _spawn_background_traffic(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        for stream, next_time in list(self._background_next.items()):
            rate_key = "ramp" if stream == "ramp" else "main"
            rate = self._background_rates[rate_key]
            while rate > 0.0 and self.sim_time + 1e-9 >= next_time:
                if stream == "ramp":
                    available, meter_snapshot = self._ramp_insertion_available(active)
                    if (
                        self._ramp_meter_closed
                        and meter_snapshot["ramp_storage_count"]
                        > RAMP_STORAGE_REOPEN_LIMIT
                    ):
                        available = False
                    if not available:
                        if not self._ramp_meter_closed:
                            self._ramp_meter_closed = True
                            self._event_log.append(
                                {
                                    "time": self.sim_time,
                                    "event": "ramp_meter_closed",
                                    **meter_snapshot,
                                }
                            )
                        break
                    if self._ramp_meter_closed:
                        self._ramp_meter_closed = False
                        self._event_log.append(
                            {
                                "time": self.sim_time,
                                "event": "ramp_meter_reopened",
                                **meter_snapshot,
                            }
                        )
                vehicle_id = f"background_{self._background_counter}"
                self._background_counter += 1
                self.connection.vehicle.add(
                    vehicle_id,
                    routeID="rampRoute" if stream == "ramp" else "mainRoute",
                    typeID=self._sample_background_type(),
                    depart="now",
                    departLane="0" if stream == "ramp" else stream.rsplit("_", 1)[1],
                    departPos="base",
                    departSpeed="desired",
                )
                if stream == "ramp":
                    self._ramp_origin_ids.add(vehicle_id)
                next_time += float(self.np_random.exponential(3600.0 / rate))
                if stream == "ramp":
                    next_time = max(
                        next_time, self.sim_time + RAMP_METER_MIN_HEADWAY
                    )
                    break
            self._background_next[stream] = next_time

    @staticmethod
    def _ramp_meter_should_close(
        ramp_states: list[tuple[str, int, float, float]],
    ) -> bool:
        if len(ramp_states) >= RAMP_STORAGE_VEHICLE_LIMIT:
            return True
        return any(
            road_id == "mergeOpen"
            and lane_index == 0
            and position <= RAMP_SPILLBACK_POSITION
            and speed <= RAMP_SPILLBACK_SPEED
            for road_id, lane_index, position, speed in ramp_states
        )

    def _ramp_insertion_available(
        self, active: set[str]
    ) -> tuple[bool, dict[str, float]]:
        ramp_states: list[tuple[str, int, float, float]] = []
        for vehicle_id in self._ramp_origin_ids.intersection(active):
            road_id = self.connection.vehicle.getRoadID(vehicle_id)
            lane_index = self.connection.vehicle.getLaneIndex(vehicle_id)
            if road_id not in {"ramp", "rampAccel", "mergeOpen"}:
                continue
            if road_id == "mergeOpen" and lane_index != 0:
                continue
            ramp_states.append(
                (
                    road_id,
                    lane_index,
                    float(self.connection.vehicle.getLanePosition(vehicle_id)),
                    float(self.connection.vehicle.getSpeed(vehicle_id)),
                )
            )
        spillback_positions = [
            position
            for road_id, lane_index, position, speed in ramp_states
            if road_id == "mergeOpen"
            and lane_index == 0
            and speed <= RAMP_SPILLBACK_SPEED
        ]
        snapshot = {
            "ramp_storage_count": float(len(ramp_states)),
            "ramp_queue_tail_position": float(
                min(spillback_positions, default=math.inf)
            ),
        }
        return not self._ramp_meter_should_close(ramp_states), snapshot

    def _configure_background_free_flow_speeds(self, active: set[str]) -> None:
        for vehicle_id in active:
            if (
                not vehicle_id.startswith("background_")
                or vehicle_id in self._background_speed_configured
            ):
                continue
            type_id = self.connection.vehicle.getTypeID(vehicle_id)
            maximum = self._background_max_speeds.get(type_id)
            if maximum is not None:
                self.connection.vehicle.setMaxSpeed(vehicle_id, maximum)
            if vehicle_id in self._ramp_origin_ids:
                # Ramp-origin vehicles use the explicit gap-acceptance controller
                # until they have completed the merge.  Mode 512 keeps SUMO's
                # collision checks active while disabling autonomous lane choice.
                self.connection.vehicle.setSpeedMode(vehicle_id, 31)
                self.connection.vehicle.setLaneChangeMode(
                    vehicle_id, SAFETY_ONLY_LANE_CHANGE_MODE
                )
            self._background_speed_configured.add(vehicle_id)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        options = options or {}
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.close()
        build_network(force=False)

        calibration, vtype_file = self._load_calibration()
        self._configure_episode(options, calibration)

        sumo_seed = int(self.np_random.integers(1, 2**31 - 1))
        binary = _find_sumo_binary("sumo-gui" if self.render_mode == "human" else "sumo")
        command = [
            binary,
            "-c",
            str(SCENARIO_DIR / "highway.sumocfg"),
            "--step-length",
            str(self.step_length),
            "--seed",
            str(sumo_seed),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--time-to-teleport",
            "-1",
            "--step-method.ballistic",
            "true",
            "--collision.action",
            "remove",
        ]
        if self.continuous_lateral_model:
            command.extend(
                ["--lateral-resolution", str(CONTINUOUS_LATERAL_RESOLUTION)]
            )
        if vtype_file is not None:
            command.extend(["--additional-files", str(vtype_file.resolve())])
        if self.render_mode == "human":
            command.extend(["--delay", str(self.gui_delay_ms)])
            if self.gui_auto_start:
                command.append("--start")
        self.connection_label = f"platoon-{uuid.uuid4().hex}"
        traci.start(command, label=self.connection_label)
        self.connection = traci.getConnection(self.connection_label)
        if self.continuous_lateral_model:
            self.connection.vehicletype.setMaxSpeedLat(
                "platoon", CONTINUOUS_PLATOON_MAX_LATERAL_SPEED
            )
            self.connection.vehicletype.setMinGapLat(
                "platoon", CONTINUOUS_PLATOON_MIN_LATERAL_GAP
            )
            self.connection.vehicletype.setLateralAlignment("platoon", "center")
            self.connection.vehicletype.setParameter(
                "platoon",
                "laneChangeModel.lcAccelLat",
                str(CONTINUOUS_PLATOON_LATERAL_ACCELERATION),
            )

        self.agents = self.possible_agents[:]
        self.sim_time = 0.0
        self._station_cache = {}
        self._cutin_state = "idle"
        self._cutin_configured = False
        self._cutin_merge_committed = False
        self._cutin_open_merge_logged = False
        self._cutin_error = None
        self._cutin_hold_until = -math.inf
        self._cutin_released = False
        self._cutin_next_lane_change_time = -math.inf
        self._cutin_waiting_lanes = set()
        self._wave_started_at = None
        self._next_wave_time = None
        self._wave_brake_until = -math.inf
        self._wave_target = None
        self._wave_pending = False
        self._wave_profile = None
        self._wave_profile_index = 0
        self._wave_trigger_index = 0
        self._communication_delay_events = 0
        self._leader_restart_opportunity = False
        self._leader_stall_duration = 0.0
        self._stalled_failure = False
        self._last_cutin_threat_count = 0
        self._last_platoon_intrusion_count = 0
        self._recovery_leaders = set()
        self._recovery_started_at = {}
        self._recovery_clear_since = {}
        self._disconnect_since = {}
        self._previous_pair_gaps = {}
        self._previous_gap_potentials = {}
        self._previous_cohesion_potential = 0.0
        self._rejoined_this_step = 0
        self._rejoined_vehicles_this_step = set()
        self._ramp_origin_ids = set()
        self._background_lane_change_guarded = set()
        self._ramp_merge_requested_at = {}
        self._ramp_merge_wait_logged = set()
        self._ramp_queue_brake_logged = set()
        self._ramp_merge_released = set()
        self._ramp_meter_closed = False
        self._last_collision_ids = []
        self._last_collision_details = []
        self._lane_change_obstacle_ids = []
        self._lane_change_event_index = 0
        self._lane_change_event_active = False
        self._lane_change_event_started_at = None
        self._lane_change_settled_since = None
        self._lane_change_completed_this_step = 0
        self._lane_change_previous_potential = 0.0
        self._lane_change_command_count = 0
        self._lane_change_commands_this_step = 0
        self._lane_change_dual_lane_time = 0.0
        self._lane_change_first_command_at = None
        self._lane_change_execution_time = 0.0
        self._lane_change_traffic_ids = []
        self._lane_change_traffic_desired_speeds = {}
        self._lane_change_traffic_spawned_events = set()
        self._lane_change_traffic_delay = 0.0
        self._lane_change_traffic_braking_integral = 0.0
        self._lane_change_traffic_hard_brake_events = 0
        self._lane_change_traffic_hard_brake_active = {}
        self._lane_change_traffic_speed_deficit = 0.0
        self._lane_change_traffic_brake_intensity = 0.0
        self._lane_change_traffic_max_deceleration = 0.0
        self._event_log = [
            {
                "time": 0.0,
                "event": "scenario_configured",
                "front_vehicle_count": self._front_vehicle_count,
                "wave_source": self._selected_wave_source,
                "wave_mode": self._wave_mode,
                "fixed_wave_profile": self._fixed_wave_profile,
                "wave_min_source_distance_m": self._wave_min_source_distance_m,
                "wave_trigger_stations": list(self._wave_trigger_stations),
                "wave_trigger_reference": self._wave_trigger_reference,
                "demand_profile": self._demand_profile_id,
                "communication_delay_probability": self._communication_delay_probability,
                "communication_delay_range": [
                    self._communication_delay_min,
                    self._communication_delay_max,
                ],
                "cutin_enabled": self._cutin_enabled,
                "cutin_clearance_scale": self._cutin_clearance_scale,
                "cutin_force_request": self._cutin_force_request,
                "cutin_depart_speed_mps": self._cutin_depart_speed_mps,
                "tail_background_count": self._tail_background_count,
                "tail_background_gap_m": self._tail_background_gap_m,
                "tail_background_depart_speed_mps": self._tail_background_depart_speed_mps,
                "tail_background_type": self._tail_background_type,
                "initial_leader_station_m": self._episode_initial_positions[
                    "leader"
                ],
                "downstream_wave_reward_enabled": self._downstream_wave_reward_enabled,
                "downstream_wave_target_gain": self._downstream_wave_target_gain,
                "downstream_wave_reward_weight": self._downstream_wave_reward_weight,
                "cooperative_lane_change_enabled": self._cooperative_lane_change_enabled,
                "lane_change_obstacle_stations": list(
                    self._lane_change_obstacle_stations
                ),
                "lane_change_target_lanes": list(self._lane_change_target_lanes),
                "lane_change_gap_wait_grace_s": self._lane_change_stall_grace,
                "lane_change_traffic_enabled": self._lane_change_traffic_enabled,
                "lane_change_traffic_front_gaps_m": list(
                    self._lane_change_traffic_front_gaps
                ),
                "lane_change_traffic_rear_gaps_m": list(
                    self._lane_change_traffic_rear_gaps
                ),
                "lane_change_traffic_front_speeds_mps": list(
                    self._lane_change_traffic_front_speeds
                ),
                "lane_change_traffic_rear_speeds_mps": list(
                    self._lane_change_traffic_rear_speeds
                ),
                "continuous_lateral_model": self.continuous_lateral_model,
                "lateral_resolution_m": (
                    CONTINUOUS_LATERAL_RESOLUTION
                    if self.continuous_lateral_model
                    else None
                ),
                "platoon_max_lateral_speed_mps": (
                    CONTINUOUS_PLATOON_MAX_LATERAL_SPEED
                    if self.continuous_lateral_model
                    else None
                ),
                "platoon_lateral_acceleration_mps2": (
                    CONTINUOUS_PLATOON_LATERAL_ACCELERATION
                    if self.continuous_lateral_model
                    else None
                ),
            }
        ]

        for vehicle_id in PLATOON_IDS:
            initial_station = self._episode_initial_positions[vehicle_id]
            if self._initial_leader_station_m is None:
                route_id = "mainRoute"
                depart_lane = PLATOON_LANE_INDEX
                depart_position = initial_station
            else:
                edge, depart_position, _ = self._route_from_station(
                    initial_station
                )
                route_id = self._background_route_id(edge)
                depart_lane = PLATOON_LANE_INDEX + int(edge == "mergeOpen")
            self.connection.vehicle.add(
                vehicle_id,
                routeID=route_id,
                typeID="platoon",
                depart="now",
                departLane=str(depart_lane),
                departPos=str(depart_position),
                departSpeed=str(TARGET_SPEED),
            )
        # Reserve the adversarial ramp vehicle before stochastic ramp demand is
        # released.  It then traverses the full ramp under arrival-time control
        # and reaches the taper together with followers 2 and 3.
        if self._cutin_enabled:
            self._spawn_cutin()
        self._spawn_front_chain()
        self._spawn_lane_change_obstacles()
        self._spawn_tail_background_chain()
        self._spawn_initial_background()
        self._event_log[0]["background_initial_requested"] = len(
            self._background_initial_ids
        )
        self.connection.simulationStep()
        self.sim_time = float(self.connection.simulation.getTime())
        active = set(self.connection.vehicle.getIDList())
        self._configure_background_free_flow_speeds(active)
        missing = [vehicle_id for vehicle_id in PLATOON_IDS if vehicle_id not in active]
        missing_front = [vehicle_id for vehicle_id in self._front_ids if vehicle_id not in active]
        missing_lane_obstacles = [
            vehicle_id
            for vehicle_id in self._lane_change_obstacle_ids
            if vehicle_id not in active
        ]
        missing_tail = [
            vehicle_id
            for vehicle_id in self._tail_background_ids
            if vehicle_id not in active
        ]
        if missing:
            self.close()
            raise RuntimeError(f"Failed to insert platoon vehicles: {missing}")
        if missing_front:
            self.close()
            raise RuntimeError(f"Failed to insert scenario front vehicles: {missing_front}")
        if missing_lane_obstacles:
            self.close()
            raise RuntimeError(
                "Failed to insert lane-change obstacles: "
                f"{missing_lane_obstacles}"
            )
        if missing_tail:
            self.close()
            raise RuntimeError(
                f"Failed to insert fixed tail background vehicles: {missing_tail}"
            )
        self._event_log[0]["background_initial_active"] = sum(
            vehicle_id in active for vehicle_id in self._background_initial_ids
        )
        for vehicle_id, speed in self._front_initial_speeds.items():
            self.connection.vehicle.setSpeed(vehicle_id, speed)
        for vehicle_id in self._front_ids:
            # These are the episode's explicitly sampled near-front vehicles,
            # not generic background demand.  Keep them in the platoon's lane
            # and let IDM evolve them independently of the learned leader.
            self.connection.vehicle.setLaneChangeMode(vehicle_id, 512)
            self.connection.vehicle.setMaxSpeed(vehicle_id, TARGET_SPEED)
        for vehicle_id in self._lane_change_obstacle_ids:
            self.connection.vehicle.setLaneChangeMode(
                vehicle_id, SAFETY_ONLY_LANE_CHANGE_MODE
            )
            self.connection.vehicle.setMaxSpeed(
                vehicle_id, self._lane_change_obstacle_speed
            )
            self.connection.vehicle.setSpeed(
                vehicle_id, self._lane_change_obstacle_speed
            )
            self.connection.vehicle.setColor(vehicle_id, (220, 40, 40, 255))
        for vehicle_id in self._tail_background_ids:
            # Keep the diagnostic chain in-lane while preserving SUMO's native
            # longitudinal IDM response to the vehicle directly ahead.
            self.connection.vehicle.setLaneChangeMode(vehicle_id, 512)
            self.connection.vehicle.setColor(vehicle_id, (115, 115, 115, 255))

        colors = {
            "leader": (0, 220, 80, 255),
            "follower_1": (0, 140, 255, 255),
            "follower_2": (30, 90, 255, 255),
            "follower_3": (120, 50, 255, 255),
            "follower_4": (190, 40, 220, 255),
            "follower_5": (235, 40, 150, 255),
        }
        for vehicle_id in PLATOON_IDS:
            self.connection.vehicle.setSpeedMode(vehicle_id, PLATOON_SPEED_MODE)
            self.connection.vehicle.setLaneChangeMode(
                vehicle_id,
                (
                    SAFETY_ONLY_LANE_CHANGE_MODE
                    if self.continuous_lateral_model
                    else 0 if self._cooperative_lane_change_enabled else 512
                ),
            )
            self.connection.vehicle.setColor(vehicle_id, colors[vehicle_id])
        if self.render_mode == "human":
            view_id = self.connection.gui.getIDList()[0]
            self.connection.gui.trackVehicle(view_id, "leader")
            self.connection.gui.setZoom(view_id, 1200.0)

        self._refresh_station_cache()
        active = set(self.connection.vehicle.getIDList())
        self._previous_pair_gaps = {
            follower: self._pair_gap_status(follower, active)[0]
            for follower in FOLLOWER_IDS
        }
        self._previous_gap_potentials = {
            follower: self._gap_potential(gap, TARGET_GAP)
            for follower, gap in self._previous_pair_gaps.items()
        }
        self._previous_cohesion_potential = self._cohesion_potential(
            self._previous_pair_gaps.get(FOLLOWER_IDS[0], TARGET_GAP)
        )
        self._vehicle_history = {}
        self._record_vehicle_history()
        leader_state = self._true_vehicle_state("leader")
        initial_plan = tuple([leader_state[1]] * LEADER_PLAN_STEPS)
        self._leader_delay_buffers = {}
        self._predecessor_delay_buffers = {}
        self._v2v_history = {}
        for follower in FOLLOWER_IDS:
            leader_buffer = V2VDelayBuffer()
            leader_buffer.prime(
                self.sim_time,
                leader_state[0],
                leader_state[1],
                leader_state[2],
                0.0,
                initial_plan,
            )
            predecessor_state = self._true_vehicle_state(PREDECESSOR[follower])
            predecessor_buffer = V2VDelayBuffer()
            predecessor_buffer.prime(
                self.sim_time,
                predecessor_state[0],
                predecessor_state[1],
                predecessor_state[2],
            )
            self._leader_delay_buffers[follower] = leader_buffer
            self._predecessor_delay_buffers[follower] = predecessor_buffer
            row = self._v2v_row(follower)
            self._v2v_history[follower] = deque(
                (row.copy() for _ in range(HISTORY_STEPS)),
                maxlen=HISTORY_STEPS,
            )
        self._previous_acceleration = {
            vehicle_id: self.connection.vehicle.getAcceleration(vehicle_id)
            for vehicle_id in PLATOON_IDS
        }
        self._last_jerk = {vehicle_id: 0.0 for vehicle_id in PLATOON_IDS}
        self._previous_command = {vehicle_id: 0.0 for vehicle_id in PLATOON_IDS}
        self._collision_shield_active = {
            vehicle_id: False for vehicle_id in PLATOON_IDS
        }
        self._last_action_info = {
            agent: {"safety_intervened": False, "applied_acceleration": 0.0}
            for agent in self.agents
        }
        self._follower_control_predictions = {}
        observations = self.get_observation()
        infos = {agent: self._base_info(agent) for agent in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[str, dict[str, Any]],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        self._require_connection()
        if not self.agents:
            raise RuntimeError("step() called after the episode ended; call reset()")
        missing_actions = set(self.agents) - set(actions)
        extra_actions = set(actions) - set(self.agents)
        if missing_actions or extra_actions:
            raise ValueError(
                f"Actions must match active agents; missing={missing_actions}, extra={extra_actions}"
            )

        acting_agents = self.agents[:]
        self._lane_change_commands_this_step = 0
        self._spawn_background_traffic()
        self._update_events_before_step()
        self._apply_actions(actions)
        self._broadcast_v2v()
        self.connection.simulationStep()
        self.sim_time = float(self.connection.simulation.getTime())
        if not self._front_speed_released:
            for vehicle_id in self._front_ids:
                if vehicle_id in self.connection.vehicle.getIDList():
                    self.connection.vehicle.setSpeed(vehicle_id, -1.0)
            self._front_speed_released = True
        self._refresh_station_cache()
        self._configure_background_free_flow_speeds(
            set(self.connection.vehicle.getIDList())
        )
        self._record_vehicle_history()
        for buffer in self._leader_delay_buffers.values():
            buffer.deliver(self.sim_time)
        for buffer in self._predecessor_delay_buffers.values():
            buffer.deliver(self.sim_time)
        for follower in FOLLOWER_IDS:
            self._v2v_history[follower].append(self._v2v_row(follower))
        self._update_events_after_step()
        self._update_recovery_roles()
        self._update_leader_stall_state()

        success, collision, failure, truncated = self._episode_state()
        terminal_active = set(self.connection.vehicle.getIDList())
        terminal_pair_gaps, _, _, _, _ = self._platoon_gap_context(
            terminal_active
        )
        terminal_diagnostics = {
            "platoon_active_count": sum(
                vehicle_id in terminal_active for vehicle_id in PLATOON_IDS
            ),
            "platoon_pair_count": len(terminal_pair_gaps),
            "platoon_connected_pair_count": sum(
                gap <= TARGET_GAP + REJOIN_GAP_TOLERANCE
                for gap in terminal_pair_gaps
            ),
            "leader_station": self._station("leader"),
        }
        leader_reward, follower_rewards, components = self._shared_reward(
            success, collision, failure
        )
        done = bool(success or collision or failure or truncated)
        observations = {} if done else self.get_observation()
        rewards = {
            agent: leader_reward if agent == "leader" else follower_rewards[agent]
            for agent in acting_agents
        }
        terminations = {
            agent: bool(success or collision or failure) for agent in acting_agents
        }
        truncations = {agent: bool(truncated) for agent in acting_agents}
        infos = {}
        for agent in acting_agents:
            info = self._base_info(agent)
            info.update(
                {
                    "reward_components": components.copy(),
                    "success": success,
                    "collision": collision,
                    "failure": failure,
                    **terminal_diagnostics,
                    **self._last_action_info[agent],
                }
            )
            infos[agent] = info

        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def get_observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Return observations for all currently active platoon agents."""

        self._require_connection()
        return {agent: self._observation_for(agent) for agent in self.agents}

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        self._require_connection()
        if agent not in self.agents:
            raise KeyError(agent)
        return self._observation_for(agent)

    def state(self) -> np.ndarray:
        """Return the undelayed global state for centralized critics."""

        self._require_connection()
        active = set(self.connection.vehicle.getIDList())
        values: list[float] = []
        for vehicle_id in PLATOON_IDS:
            if vehicle_id in active:
                station, speed, acceleration, lane = self._true_vehicle_state(vehicle_id)
                values.extend([station, speed, acceleration, lane, 1.0])
            else:
                values.extend([0.0, 0.0, 0.0, -1.0, 0.0])
        colliding = set(self.connection.simulation.getCollidingVehiclesIDList())
        values.extend(
            [
                self._cutin_status_value(),
                float(self.sim_time < self._wave_brake_until),
                self.sim_time / self.max_time,
                float(bool(colliding.intersection(PLATOON_IDS))),
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def render(self) -> None:
        if self.render_mode != "human":
            return None
        return None

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except (traci.TraCIException, OSError, FatalTraCIError):
                pass
        self.connection = None
        self.connection_label = None
        self.agents = []

    def _require_connection(self) -> None:
        if self.connection is None:
            raise RuntimeError("Environment is not running; call reset() first")

    def set_follower_control_predictions(
        self, predictions: dict[str, tuple[float, float]]
    ) -> None:
        """Attach CPIDLM predictions used only for reward shaping."""

        self._follower_control_predictions = {
            follower: (max(0.0, float(speed)), float(acceleration))
            for follower, (speed, acceleration) in predictions.items()
            if follower in FOLLOWER_IDS
        }

    def _lane_change_request_safe(
        self, agent: str, target_lane: int, active: set[str]
    ) -> tuple[bool, dict[str, float | str | None]]:
        """Common low-level lateral clearance check without planning intent."""

        ego_station = self._station(agent)
        ego_speed = float(self.connection.vehicle.getSpeed(agent))
        ego_length = float(self.connection.vehicle.getLength(agent))
        front_id: str | None = None
        rear_id: str | None = None
        front_gap = math.inf
        rear_gap = math.inf
        for vehicle_id in active.difference({agent}):
            if int(self.connection.vehicle.getLaneIndex(vehicle_id)) != target_lane:
                continue
            station = self._station(vehicle_id)
            if station >= ego_station:
                gap = station - ego_station - float(
                    self.connection.vehicle.getLength(vehicle_id)
                )
                if gap < front_gap:
                    front_gap = gap
                    front_id = vehicle_id
            else:
                gap = ego_station - ego_length - station
                if gap < rear_gap:
                    rear_gap = gap
                    rear_id = vehicle_id
        front_speed = (
            ego_speed
            if front_id is None
            else float(self.connection.vehicle.getSpeed(front_id))
        )
        rear_speed = (
            ego_speed
            if rear_id is None
            else float(self.connection.vehicle.getSpeed(rear_id))
        )
        front_required = max(
            MIN_SAFE_GAP,
            MIN_SAFE_GAP
            + MERGE_LOOKAHEAD * max(0.0, ego_speed - front_speed),
        )
        rear_required = max(
            MIN_SAFE_GAP,
            MIN_SAFE_GAP
            + LANE_CHANGE_REAR_LOOKAHEAD * max(0.0, rear_speed - ego_speed),
        )
        if front_id not in PLATOON_IDS:
            front_required = max(
                front_required,
                LANE_CHANGE_MIN_FRONT_HEADWAY * ego_speed,
            )
        if rear_id not in PLATOON_IDS:
            rear_required = max(
                rear_required,
                LANE_CHANGE_MIN_REAR_HEADWAY * rear_speed,
            )
        front_ttc = (
            front_gap / max(ego_speed - front_speed, 1e-3)
            if front_id is not None and ego_speed > front_speed
            else math.inf
        )
        rear_ttc = (
            rear_gap / max(rear_speed - ego_speed, 1e-3)
            if rear_id is not None and rear_speed > ego_speed
            else math.inf
        )
        safe = bool(
            front_gap >= front_required
            and rear_gap >= rear_required
            and front_ttc >= 1.5
            and rear_ttc >= 1.5
        )
        return safe, {
            "target_lane": float(target_lane),
            "target_front_id": front_id,
            "target_rear_id": rear_id,
            "target_front_gap": float(min(front_gap, 500.0)),
            "target_rear_gap": float(min(rear_gap, 500.0)),
            "target_front_required": float(front_required),
            "target_rear_required": float(rear_required),
            "target_front_ttc": float(min(front_ttc, 100.0)),
            "target_rear_ttc": float(min(rear_ttc, 100.0)),
        }

    def _apply_actions(self, actions: dict[str, dict[str, Any]]) -> None:
        active = set(self.connection.vehicle.getIDList())
        self._last_action_info = {}
        for agent, action in actions.items():
            if agent not in active:
                self._last_action_info[agent] = {
                    "vehicle_missing": True,
                }
                continue
            if not isinstance(action, dict) or set(action) != {"accel", "lane_change"}:
                raise ValueError(f"Action for {agent} must contain accel and lane_change")
            requested = float(np.asarray(action["accel"], dtype=np.float32).reshape(-1)[0])
            requested_clipped = float(
                np.clip(requested, MAX_BRAKING, MAX_ACCELERATION)
            )
            braking_broadcast = requested_clipped < -0.5
            previous = self._previous_command.get(agent, 0.0)
            speed = float(self.connection.vehicle.getSpeed(agent))
            acceleration, restart_jerk_release = self._jerk_limited_acceleration(
                requested_clipped,
                previous,
                speed,
                self.step_length,
            )
            safety_intervened = not math.isclose(acceleration, requested, abs_tol=1e-6)
            collision_avoidance_intervened = False
            predictive_safety_intervened = False
            formation_guard_intervened = False
            formation_guard_target: float | None = None
            predictive_acceleration_limit: float | None = None
            formation_gap = 0.0
            longitudinal_risk = 0.0

            if agent == "leader" and FOLLOWER_IDS[0] in active:
                formation_gap = max(
                    0.0,
                    self._station("leader")
                    - self._station(FOLLOWER_IDS[0])
                    - PLATOON_LENGTH,
                )
                formation_guard_target = self._formation_guard_acceleration(
                    formation_gap,
                    self.connection.vehicle.getSpeed("leader"),
                    self.connection.vehicle.getSpeed(FOLLOWER_IDS[0]),
                    previous,
                    self.step_length,
                )
                if (
                    formation_guard_target is not None
                    and acceleration > formation_guard_target
                ):
                    acceleration = formation_guard_target
                    safety_intervened = True
                    formation_guard_intervened = True

            front = self.connection.vehicle.getLeader(agent, 200.0)
            if front and front[0] in active:
                ego_speed = self.connection.vehicle.getSpeed(agent)
                front_speed = self.connection.vehicle.getSpeed(front[0])
                net_gap = max(
                    0.0,
                    float(front[1]) + self.connection.vehicle.getMinGap(agent),
                )
                raw_gap = max(0.0, float(front[1]))
                closing_speed = ego_speed - front_speed
                ttc = net_gap / max(closing_speed, 1e-3) if closing_speed > 0 else math.inf
                dynamic_gap = 3.0 + 0.35 * ego_speed + 0.8 * max(0.0, closing_speed)
                required_deceleration = max(
                    0.0,
                    (ego_speed**2 - front_speed**2)
                    / (
                        2.0
                        * max(raw_gap - MIN_SAFE_GAP, 0.5)
                    ),
                )
                cbf_margin = self._cbf_margin(
                    raw_gap, ego_speed, front_speed
                )
                longitudinal_risk = max(
                    max(0.0, (3.0 - ttc) / 3.0),
                    max(0.0, (dynamic_gap - net_gap) / max(dynamic_gap, 1.0)),
                    max(0.0, (required_deceleration - 2.0) / 4.0),
                )
                front_acceleration = float(
                    self._previous_command.get(
                        front[0],
                        self.connection.vehicle.getAcceleration(front[0]),
                    )
                )
                front_intention = float(
                    self._last_action_info.get(front[0], {}).get(
                        "requested_acceleration", front_acceleration
                    )
                )
                front_action_info = self._last_action_info.get(front[0], {})
                # An external brake is an input to the predictive constraint,
                # not an unconditional command broadcast. This lets a safe
                # front-gap buffer absorb the wave while the CBF still takes
                # over as soon as the predicted clearance becomes critical.
                front_safety_braking = bool(
                    front[0] in PLATOON_IDS
                    and front_action_info.get("braking_broadcast", False)
                    and net_gap <= TARGET_GAP + 0.5
                )
                braking_broadcast = (
                    braking_broadcast or front_safety_braking
                )
                coordinated_motion = (
                    net_gap >= MIN_SAFE_GAP
                    and closing_speed <= 0.5
                    and front_intention > -0.5
                    and front_acceleration > -0.5
                    and not front_safety_braking
                )
                predictive_front_acceleration = (
                    max(front_acceleration, front_intention)
                    if coordinated_motion
                    else front_acceleration
                )
                predictive_target_gap = (
                    CUTIN_TARGET_GAP
                    if agent in FOLLOWER_IDS
                    and self._pair_gap_status(agent, active)[1]
                    else (
                        LANE_CHANGE_COMPACT_GAP
                        if self._cooperative_lane_change_enabled
                        and self._lane_change_event_active
                        else TARGET_GAP
                    )
                )
                predictive_acceleration_limit = self._predictive_safe_acceleration(
                    net_gap=net_gap,
                    ego_speed=ego_speed,
                    front_speed=front_speed,
                    # Actions are applied in platoon order.  For an upstream
                    # controlled vehicle, _previous_command therefore already
                    # contains the acceleration accepted in this same control
                    # tick.  Reading TraCI here would return the preceding
                    # simulation tick and falsely reject synchronous commands.
                    front_acceleration=predictive_front_acceleration,
                    minimum_gap=float(
                        self.connection.vehicle.getMinGap(agent)
                    ),
                    horizon=1.0,
                    target_gap=predictive_target_gap,
                    coordinated_motion=coordinated_motion,
                )
                if (
                    front_safety_braking
                    and net_gap >= MIN_SAFE_GAP
                    and closing_speed <= 0.5
                ):
                    # The upstream accepted brake is already the causal V2V
                    # safety ceiling. Reapplying the nominal 12 m tracking
                    # error at a compact post-stop gap would amplify one
                    # synchronized brake into a rearward emergency wave.
                    predictive_acceleration_limit = front_acceleration
                predictive_guard_active = (
                    cbf_margin < 2.0
                    or (closing_speed > 0.5 and ttc < 4.0)
                    or required_deceleration > 2.0
                    # A predecessor brake accepted earlier in this same tick
                    # is a V2V safety broadcast.  Propagating only this braking
                    # ceiling prevents a one-vehicle-at-a-time brake wave while
                    # leaving all nominal longitudinal control to the RL actor.
                    or predictive_acceleration_limit < -0.1
                )
                steady_tracking = (
                    longitudinal_risk < 0.01
                    and net_gap >= predictive_target_gap - 0.5
                    and abs(closing_speed) <= 0.1
                )
                predictive_intervention_tolerance = (
                    0.2
                    if steady_tracking and not front_safety_braking
                    else 1e-3
                )
                if (
                    predictive_guard_active
                    and acceleration
                    > predictive_acceleration_limit
                    + predictive_intervention_tolerance
                ):
                    acceleration = max(
                        MAX_BRAKING, predictive_acceleration_limit
                    )
                    safety_intervened = True
                    predictive_safety_intervened = True
                collision_risk_triggered = agent in PLATOON_IDS and (
                    cbf_margin < 0.0
                    or ttc < FOLLOWER_CBF_TIME_HEADWAY
                    or required_deceleration > 3.5
                    or net_gap < MIN_SAFE_GAP
                )
                # The conservative CBF uses SUMO's gap before the configured
                # minGap is restored, which provides useful braking reserve.
                # Once both vehicles are nearly stopped, that reserve must not
                # become a launch interlock: a positive predecessor motion or
                # same-tick acceleration intent releases the latch while the
                # true physical minimum remains enforced.
                opening_restart = (
                    ego_speed <= RESTART_SPEED_THRESHOLD
                    and coordinated_motion
                    and (
                        front_speed > ego_speed + 0.05
                        or front_intention > 0.05
                        or front_acceleration > 0.05
                    )
                )
                if coordinated_motion:
                    collision_risk_triggered = False
                shield_was_active = self._collision_shield_active.get(agent, False)
                shield_recovered = coordinated_motion or (
                    net_gap >= dynamic_gap and closing_speed <= 0.0
                )
                collision_risk_triggered = collision_risk_triggered or (
                    shield_was_active and not shield_recovered
                )
                if collision_risk_triggered:
                    shield_target = max(
                        EMERGENCY_BRAKING,
                        -max(2.0, required_deceleration),
                    )
                    if net_gap < MIN_SAFE_GAP:
                        shield_target = EMERGENCY_BRAKING
                    acceleration = min(acceleration, shield_target)
                    safety_intervened = True
                    collision_avoidance_intervened = True
                elif shield_was_active and shield_recovered:
                    # Once the front vehicle is pulling away, release emergency
                    # braking promptly.  The normal jerk limiter would otherwise
                    # take about four seconds to climb from -6 m/s2 back to zero.
                    release_jerk = (
                        MAX_RESTART_JERK
                        if opening_restart
                        else MAX_BRAKE_RELEASE_JERK
                    )
                    release_delta = release_jerk * self.step_length
                    requested_clipped = float(
                        np.clip(requested, MAX_BRAKING, MAX_ACCELERATION)
                    )
                    acceleration = float(
                        np.clip(
                            requested_clipped,
                            previous - release_delta,
                            previous + release_delta,
                        )
                    )
                    if formation_guard_target is not None:
                        acceleration = min(acceleration, formation_guard_target)
                    if (
                        requested_clipped > 0.05
                        and predictive_acceleration_limit is not None
                    ):
                        acceleration = min(
                            acceleration, predictive_acceleration_limit
                        )
                    safety_intervened = not math.isclose(
                        acceleration, requested, abs_tol=1e-6
                    )
                if agent in PLATOON_IDS:
                    self._collision_shield_active[agent] = collision_risk_triggered
            elif agent in PLATOON_IDS:
                self._collision_shield_active[agent] = False
            speed = self.connection.vehicle.getSpeed(agent)
            advisory_speed = self._speed_advisory(self._station(agent))
            if speed > advisory_speed + 0.5:
                acceleration = min(acceleration, -1.0)
                safety_intervened = True

            self.connection.vehicle.setAcceleration(agent, acceleration, self.step_length)
            lane_action = int(action["lane_change"])
            if (
                agent in FOLLOWER_IDS
                and lane_action != 0
                and not self._cooperative_lane_change_enabled
            ):
                raise ValueError(
                    f"Follower {agent} only supports longitudinal control "
                    "outside the cooperative lane-change scenario"
                )
            if lane_action not in (0, 1, 2):
                raise ValueError(f"lane_change for {agent} must be 0, 1, or 2")
            lane_index = int(self.connection.vehicle.getLaneIndex(agent))
            lane_count = int(self.connection.edge.getLaneNumber(self.connection.vehicle.getRoadID(agent)))
            direction = 1 if lane_action == 1 else -1 if lane_action == 2 else 0
            lane_change_applied = False
            lane_change_rejected = False
            lane_change_clearance: dict[str, float | str | None] = {}
            if direction and 0 <= lane_index + direction < lane_count:
                safe, lane_change_clearance = self._lane_change_request_safe(
                    agent, lane_index + direction, active
                )
                if safe:
                    request_duration = (
                        CONTINUOUS_LANE_CHANGE_REQUEST_DURATION
                        if self.continuous_lateral_model
                        else 1.0
                    )
                    self.connection.vehicle.changeLaneRelative(
                        agent, direction, request_duration
                    )
                    lane_change_applied = True
                    if self._cooperative_lane_change_enabled:
                        self._lane_change_command_count += 1
                        self._lane_change_commands_this_step += 1
                        if self._lane_change_first_command_at is None:
                            self._lane_change_first_command_at = self.sim_time
                else:
                    lane_change_rejected = True
            self._previous_command[agent] = acceleration
            braking_broadcast = bool(
                braking_broadcast
                or (
                    acceleration < -0.5
                    and (
                        predictive_safety_intervened
                        or collision_avoidance_intervened
                    )
                )
            )
            self._last_action_info[agent] = {
                "requested_acceleration": requested,
                "applied_acceleration": acceleration,
                "safety_intervened": safety_intervened,
                "collision_avoidance_intervened": collision_avoidance_intervened,
                "predictive_safety_intervened": predictive_safety_intervened,
                "formation_guard_intervened": formation_guard_intervened,
                "formation_gap": formation_gap,
                "longitudinal_safety_risk": longitudinal_risk,
                "restart_jerk_release": restart_jerk_release,
                "braking_broadcast": braking_broadcast,
                "requested_lane_change": lane_action,
                "lane_change_applied": lane_change_applied,
                "lane_change_rejected": lane_change_rejected,
                **lane_change_clearance,
            }

    @staticmethod
    def _jerk_limited_acceleration(
        requested: float,
        previous: float,
        speed: float,
        step_length: float,
    ) -> tuple[float, bool]:
        restart_release = bool(
            speed <= RESTART_SPEED_THRESHOLD
            and requested > 0.05
            and previous < -0.05
        )
        moving_brake_release = bool(
            speed > RESTART_SPEED_THRESHOLD
            and requested > 0.05
            and previous < -0.5
        )
        if restart_release:
            positive_jerk = MAX_RESTART_JERK
        elif moving_brake_release:
            # Releasing a service brake is not the same actuator transition as
            # building positive traction. Clear residual coordinated braking
            # promptly, then let the ordinary jerk bound govern acceleration.
            positive_jerk = MAX_BRAKE_RELEASE_JERK
        else:
            positive_jerk = MAX_JERK
        acceleration = float(
            np.clip(
                requested,
                previous - MAX_JERK * step_length,
                previous + positive_jerk * step_length,
            )
        )
        return acceleration, restart_release

    @staticmethod
    def _formation_guard_acceleration(
        gap: float,
        leader_speed: float,
        follower_speed: float,
        previous_acceleration: float,
        step_length: float,
    ) -> float | None:
        if (
            gap <= TARGET_GAP + RECOVERY_ACTIVATION_MARGIN
            or leader_speed <= follower_speed + 0.5
        ):
            return None
        desired = float(np.clip(0.4 * (follower_speed - leader_speed), -1.5, 0.0))
        return max(desired, previous_acceleration - MAX_JERK * step_length)

    @staticmethod
    def _predictive_safe_acceleration(
        net_gap: float,
        ego_speed: float,
        front_speed: float,
        front_acceleration: float,
        minimum_gap: float,
        horizon: float,
        target_gap: float = TARGET_GAP,
        coordinated_motion: bool | None = None,
    ) -> float:
        closing_speed = ego_speed - front_speed
        current_dynamic_gap = (
            3.0 + 0.35 * ego_speed + 0.8 * max(0.0, closing_speed)
        )
        # Preserve the full predictive buffer throughout braking. During a
        # coordinated positive motion, grow the gap target continuously from
        # the physical standstill clearance to the cruise target. This avoids
        # a discontinuous brake command just above the restart threshold while
        # still restoring the full headway as speed rises.
        required_gap = max(target_gap, current_dynamic_gap)
        if coordinated_motion is None:
            coordinated_motion = (
                net_gap >= MIN_SAFE_GAP
                and closing_speed <= 0.5
                and front_acceleration > -0.5
            )
        if coordinated_motion:
            speed_fraction = float(
                np.clip(ego_speed / TARGET_SPEED, 0.0, 1.0)
            )
            recovery_gap = MIN_SAFE_GAP + (
                max(MIN_SAFE_GAP, target_gap) - MIN_SAFE_GAP
            ) * speed_fraction
            required_gap = max(recovery_gap, current_dynamic_gap)
        if (
            net_gap >= required_gap - 0.5
            and abs(closing_speed) <= 0.1
        ):
            # A safety layer should not continuously trim centimetre-scale
            # steady tracking error. The Actor owns nominal gap regulation;
            # TTC, braking broadcasts and the physical CBF remain unchanged.
            required_gap = min(required_gap, net_gap)
        safe_acceleration = front_acceleration + 2.0 * (
            net_gap - required_gap - closing_speed * horizon
        ) / max(horizon**2, 1e-6)
        return float(
            np.clip(safe_acceleration, EMERGENCY_BRAKING, MAX_ACCELERATION)
        )

    def _sample_v2v_delay(self) -> float:
        if self.np_random.random() >= self._communication_delay_probability:
            return 0.0
        self._communication_delay_events += 1
        return float(
            self.np_random.uniform(
                self._communication_delay_min, self._communication_delay_max
            )
        )

    def _broadcast_v2v(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        if "leader" not in active:
            return
        leader_state = self._true_vehicle_state("leader")
        planned_acceleration, planned_speeds = self._leader_motion_plan(leader_state)
        for follower in FOLLOWER_IDS:
            predecessor = PREDECESSOR[follower]
            if predecessor not in active:
                continue
            predecessor_state = self._true_vehicle_state(predecessor)
            self._leader_delay_buffers[follower].push(
                self.sim_time,
                leader_state[0],
                leader_state[1],
                leader_state[2],
                self._sample_v2v_delay(),
                planned_acceleration,
                planned_speeds,
            )
            self._predecessor_delay_buffers[follower].push(
                self.sim_time,
                predecessor_state[0],
                predecessor_state[1],
                predecessor_state[2],
                self._sample_v2v_delay(),
            )

    def _leader_motion_plan(
        self, leader_state: tuple[float, float, float, float]
    ) -> tuple[float, tuple[float, ...]]:
        station, speed, _, _ = leader_state
        planned_acceleration = float(self._previous_command.get("leader", 0.0))
        future_speeds: list[float] = []
        for step in range(1, LEADER_PLAN_STEPS + 1):
            horizon = step * LEADER_PLAN_INTERVAL
            projected_speed = max(0.0, speed + planned_acceleration * horizon)
            projected_station = min(
                MAIN_ROUTE_DISTANCE_END,
                station + speed * horizon + 0.5 * planned_acceleration * horizon**2,
            )
            future_speeds.append(
                min(projected_speed, self._speed_advisory(projected_station))
            )
        return planned_acceleration, tuple(future_speeds)

    def _v2v_row(self, follower: str) -> np.ndarray:
        leader = self._leader_delay_buffers[follower].estimate(self.sim_time)
        predecessor = self._predecessor_delay_buffers[follower].estimate(self.sim_time)
        return np.asarray(
            [leader[0], leader[1], leader[2], leader[4], predecessor[0], predecessor[1], predecessor[2], predecessor[4]],
            dtype=np.float32,
        )

    def _record_vehicle_history(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        for vehicle_id in active:
            station, speed, acceleration, lane = self._true_vehicle_state(vehicle_id)
            road_id = self.connection.vehicle.getRoadID(vehicle_id)
            on_ramp_or_auxiliary = road_id in ("ramp", "rampAccel") or (
                road_id == "mergeOpen"
                and lane < platoon_lane_index(station)
            )
            row = np.asarray(
                [
                    station,
                    lane,
                    speed,
                    acceleration,
                    self.connection.vehicle.getLength(vehicle_id),
                    float(on_ramp_or_auxiliary),
                ],
                dtype=np.float32,
            )
            if vehicle_id not in self._vehicle_history:
                self._vehicle_history[vehicle_id] = deque(
                    (row.copy() for _ in range(HISTORY_STEPS)),
                    maxlen=HISTORY_STEPS,
                )
            else:
                self._vehicle_history[vehicle_id].append(row)

    def _refresh_station_cache(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        for vehicle_id in active:
            road_id = self.connection.vehicle.getRoadID(vehicle_id)
            lane_position = self.connection.vehicle.getLanePosition(vehicle_id)
            if road_id in MAIN_EDGE_DISTANCE_START:
                station = MAIN_EDGE_DISTANCE_START[road_id] + lane_position
            elif road_id in MAIN_INTERNAL_EDGE_DISTANCE_START:
                station = (
                    MAIN_INTERNAL_EDGE_DISTANCE_START[road_id] + lane_position
                )
            elif road_id in RAMP_EDGE_DISTANCE_START:
                station = RAMP_EDGE_DISTANCE_START[road_id] + lane_position
            elif road_id in RAMP_INTERNAL_EDGE_DISTANCE_START:
                station = (
                    RAMP_INTERNAL_EDGE_DISTANCE_START[road_id] + lane_position
                )
            else:
                previous = self._station_cache.get(vehicle_id, 0.0)
                speed = self.connection.vehicle.getSpeed(vehicle_id)
                station = previous + speed * self.step_length
            self._station_cache[vehicle_id] = float(
                np.clip(station, 0.0, MAIN_ROUTE_DISTANCE_END)
            )

    def _station(self, vehicle_id: str) -> float:
        return self._station_cache.get(vehicle_id, 0.0)

    def _true_vehicle_state(self, vehicle_id: str) -> tuple[float, float, float, float]:
        return (
            self._station(vehicle_id),
            float(self.connection.vehicle.getSpeed(vehicle_id)),
            float(self.connection.vehicle.getAcceleration(vehicle_id)),
            float(self.connection.vehicle.getLaneIndex(vehicle_id)),
        )

    def _observation_for(self, observer: str) -> dict[str, np.ndarray]:
        active = set(self.connection.vehicle.getIDList())
        observer_state = self._true_vehicle_state(observer)
        ego_station, ego_speed, ego_acceleration, ego_lane = observer_state
        ego = np.asarray(
            [
                ego_station,
                ego_speed,
                ego_acceleration,
                self._last_jerk.get(observer, 0.0),
                ego_lane,
                PLATOON_IDS.index(observer) / (len(PLATOON_IDS) - 1),
                self._road_curvature(ego_station),
                self._speed_advisory(ego_station) / 40.0,
            ],
            dtype=np.float32,
        )
        platoon = np.zeros((len(PLATOON_IDS), 6), dtype=np.float32)
        visible = {observer, "leader"}
        if observer == "leader":
            visible.update(PLATOON_IDS)
        elif observer in FOLLOWER_IDS:
            visible.add(PREDECESSOR[observer])
        for row, vehicle_id in enumerate(PLATOON_IDS):
            if vehicle_id not in active or vehicle_id not in visible:
                platoon[row, 3] = -1.0
                continue
            station, speed, acceleration, lane = self._true_vehicle_state(vehicle_id)
            if vehicle_id == "leader" and observer in FOLLOWER_IDS:
                station, speed, acceleration, _, _ = self._leader_delay_buffers[observer].estimate(
                    self.sim_time
                )
            gap = self._platoon_gap(vehicle_id, active)
            platoon[row] = np.asarray(
                [station - ego_station, speed, acceleration, lane, gap, 1.0],
                dtype=np.float32,
            )

        front_history, front_mask = self._tracked_humans(
            ego_station, ego_lane, active, ahead_only=True
        )
        neighbor_history, neighbor_mask = self._tracked_humans(
            ego_station, ego_lane, active, ahead_only=False
        )
        if observer in FOLLOWER_IDS:
            leader_packet = self._leader_delay_buffers[observer].estimate(self.sim_time)
            predecessor_packet = self._predecessor_delay_buffers[observer].estimate(self.sim_time)
            communication = np.asarray(
                [leader_packet[3], leader_packet[4], predecessor_packet[3], predecessor_packet[4]],
                dtype=np.float32,
            )
            v2v_history = np.asarray(self._v2v_history[observer], dtype=np.float32)
            leader_plan = self._leader_delay_buffers[observer].plan()
            predecessor = PREDECESSOR[observer]
            pred_station, pred_speed, pred_acceleration, pred_lane = self._true_vehicle_state(predecessor)
            local_predecessor = np.asarray(
                [
                    pred_station - ego_station,
                    pred_speed - ego_speed,
                    pred_acceleration,
                    pred_lane,
                    max(0.0, pred_station - ego_station - PLATOON_LENGTH),
                    1.0,
                ],
                dtype=np.float32,
            )
            physical_front = self.connection.vehicle.getLeader(observer, 500.0)
            if physical_front and physical_front[0] in active:
                front_id, front_gap = physical_front
                front_station, front_speed, front_acceleration, front_lane = (
                    self._true_vehicle_state(front_id)
                )
                local_front = np.asarray(
                    [
                        front_station - ego_station,
                        front_speed - ego_speed,
                        front_acceleration,
                        front_lane,
                        max(
                            0.0,
                            float(front_gap)
                            + self.connection.vehicle.getMinGap(observer),
                        ),
                        1.0,
                    ],
                    dtype=np.float32,
                )
            else:
                local_front = np.zeros(6, dtype=np.float32)
        else:
            communication = np.zeros(4, dtype=np.float32)
            v2v_history = np.zeros((HISTORY_STEPS, 8), dtype=np.float32)
            leader_plan = np.zeros(LEADER_PLAN_STEPS + 1, dtype=np.float32)
            local_predecessor = np.zeros(6, dtype=np.float32)
            local_front = np.zeros(6, dtype=np.float32)
        v2i_flow = (
            self._v2i_flow(ego_station, active)
            if self._v2i_enabled
            else np.zeros((8, 4), dtype=np.float32)
        )
        signal = np.zeros(4, dtype=np.float32)
        signal_mask = np.zeros(1, dtype=np.float32)
        if observer == "leader":
            leader_front = self.connection.vehicle.getLeader("leader", 500.0)
            if leader_front and leader_front[0] in active:
                front_gap = max(0.0, float(leader_front[1]))
                front_speed = max(
                    0.0,
                    float(self.connection.vehicle.getSpeed(leader_front[0])),
                )
            else:
                front_gap = 500.0
                front_speed = self._speed_advisory(ego_station)
            signal = np.asarray(
                [
                    float(self._leader_restart_opportunity),
                    min(1.0, self._leader_stall_duration / STALLED_FAILURE_TIMEOUT),
                    min(1.0, front_gap / 300.0),
                    min(1.0, front_speed / 30.0),
                ],
                dtype=np.float32,
            )
            signal_mask[0] = 1.0
        recovery = np.zeros(6, dtype=np.float32)
        if observer in FOLLOWER_IDS:
            predecessor = PREDECESSOR[observer]
            pair_gap, _, intrusion = self._pair_gap_status(observer, active)
            predecessor_speed = float(
                self.connection.vehicle.getSpeed(predecessor)
            )
            flow_reference = float(v2i_flow[0, 0])
            if flow_reference <= 0.0:
                flow_reference = self._speed_advisory(ego_station)
            recovery = np.asarray(
                [
                    (
                        min(
                            1.0,
                            max(
                                0.0,
                                self.sim_time
                                - self._recovery_started_at.get(
                                    observer, self.sim_time
                                ),
                            )
                            / 1.0,
                        )
                        if observer in self._recovery_leaders
                        else 0.0
                    ),
                    pair_gap,
                    ego_speed - predecessor_speed,
                    float(intrusion),
                    min(self._speed_advisory(ego_station), flow_reference),
                    max(
                        0.0,
                        self.sim_time
                        - self._recovery_started_at.get(observer, self.sim_time),
                    ),
                ],
                dtype=np.float32,
            )
        return {
            "ego": ego,
            "platoon_state": platoon,
            "front_history": front_history[:MAX_FRONT_VEHICLES],
            "front_mask": front_mask[:MAX_FRONT_VEHICLES],
            "neighbor_history": neighbor_history[:MAX_NEIGHBORS, -NEIGHBOR_HISTORY_STEPS:],
            "neighbor_mask": neighbor_mask[:MAX_NEIGHBORS],
            "v2v_history": v2v_history,
            "leader_plan": leader_plan,
            "local_predecessor": local_predecessor,
            "local_front": local_front,
            "v2i_flow": v2i_flow,
            "road_profile": (
                self._road_profile(ego_station)
                if self._v2i_enabled
                else np.zeros((20, 3), dtype=np.float32)
            ),
            "signal": signal,
            "signal_mask": signal_mask,
            "communication": communication,
            "recovery": recovery,
            "lane_mission": self._lane_mission_observation(observer, active),
        }

    def _lane_mission_observation(
        self, observer: str, active: set[str]
    ) -> np.ndarray:
        if not self._cooperative_lane_change_enabled or observer not in active:
            return np.zeros(10, dtype=np.float32)
        current_lane = int(self.connection.vehicle.getLaneIndex(observer))
        complete = self._lane_change_event_index >= len(
            self._lane_change_target_lanes
        )
        if complete:
            target_lane = current_lane
            obstacle_distance = 400.0
        else:
            target_lane = self._lane_change_target_lanes[
                self._lane_change_event_index
            ]
            obstacle_distance = (
                self._lane_change_obstacle_stations[self._lane_change_event_index]
                - nominal_mainline_station(self._station(observer))
            )
        direction = int(np.sign(target_lane - current_lane))
        front_clearance = 200.0
        rear_clearance = 200.0
        observer_station = self._station(observer)
        for vehicle_id in active.difference({observer}):
            if int(self.connection.vehicle.getLaneIndex(vehicle_id)) != target_lane:
                continue
            relative = self._station(vehicle_id) - observer_station
            if relative >= 0.0:
                front_clearance = min(front_clearance, relative - PLATOON_LENGTH)
            else:
                rear_clearance = min(rear_clearance, -relative - PLATOON_LENGTH)
        elapsed = (
            0.0
            if self._lane_change_event_started_at is None
            else self.sim_time - self._lane_change_event_started_at
        )
        return np.asarray(
            [
                1.0,
                self._lane_change_event_index
                / max(len(self._lane_change_target_lanes), 1),
                target_lane / 2.0,
                float(direction),
                float(np.clip(obstacle_distance / 200.0, -2.0, 2.0)),
                current_lane / 2.0,
                float(np.clip(front_clearance / 200.0, 0.0, 1.0)),
                float(np.clip(rear_clearance / 200.0, 0.0, 1.0)),
                float(self._lane_change_event_active),
                float(np.clip(elapsed / 30.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _platoon_gap(self, vehicle_id: str, active: set[str]) -> float:
        if vehicle_id in PREDECESSOR and PREDECESSOR[vehicle_id] in active:
            return float(
                self._station(PREDECESSOR[vehicle_id])
                - self._station(vehicle_id)
                - PLATOON_LENGTH
            )
        if vehicle_id in active:
            leader = self.connection.vehicle.getLeader(vehicle_id, 200.0)
            if leader and leader[0]:
                return float(leader[1] + self.connection.vehicle.getMinGap(vehicle_id))
        return -1.0

    def _tracked_humans(
        self, ego_station: float, ego_lane: float, active: set[str], ahead_only: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        limit = MAX_FRONT_VEHICLES if ahead_only else MAX_NEIGHBORS
        history_steps = HISTORY_STEPS if ahead_only else NEIGHBOR_HISTORY_STEPS
        feature_count = 6 if ahead_only else 8
        tracks = np.zeros((limit, history_steps, feature_count), dtype=np.float32)
        mask = np.zeros(limit, dtype=np.float32)
        candidates: list[tuple[float, str]] = []
        for vehicle_id in active.difference(PLATOON_IDS):
            station = self._station(vehicle_id)
            relative = station - ego_station
            distance = abs(relative)
            if ((ahead_only and 0.0 <= relative <= 800.0) or (not ahead_only and distance <= 250.0)):
                candidates.append((distance, vehicle_id))
        candidates.sort()
        for index, (_, vehicle_id) in enumerate(candidates[:limit]):
            history = list(self._vehicle_history[vehicle_id])[-history_steps:]
            for time_index, sample in enumerate(history):
                base = [sample[0] - ego_station, sample[1] - ego_lane, sample[2], sample[3], sample[4], 1.0]
                if ahead_only:
                    tracks[index, time_index] = np.asarray(base, dtype=np.float32)
                else:
                    tracks[index, time_index] = np.asarray(
                        base[:5] + [sample[1], sample[5], 1.0], dtype=np.float32
                    )
            mask[index] = 1.0
        return tracks, mask

    def _v2i_flow(self, station: float, active: set[str]) -> np.ndarray:
        result = np.zeros((8, 4), dtype=np.float32)
        humans = {
            vehicle_id
            for vehicle_id in active.difference(PLATOON_IDS)
            if self._vehicle_history.get(vehicle_id)
            and self._vehicle_history[vehicle_id][-1][5] < 0.5
        }
        for bin_index in range(8):
            start = station + 100.0 * bin_index
            end = start + 100.0
            speeds = [
                self.connection.vehicle.getSpeed(vehicle_id)
                for vehicle_id in humans
                if start <= self._station(vehicle_id) < end
            ]
            if speeds:
                mean = float(np.mean(speeds))
                deviation = float(np.std(speeds))
                density = min(len(speeds) / 0.3, 300.0)
                result[bin_index] = (mean, density, deviation, min(deviation / 15.0, 1.0))
        return result

    @staticmethod
    def _road_curvature(station: float) -> float:
        station = nominal_mainline_station(station)
        if not 600.0 <= station <= 1400.0:
            return 0.0
        control_y = (0.0, 22.0, 0.0, -22.0, 0.0)
        segment = min(int((station - 600.0) // 200.0), 3)
        u = ((station - 600.0) - 200.0 * segment) / 200.0
        delta = control_y[segment + 1] - control_y[segment]
        first = delta * (30.0 * u**2 * (u - 1.0) ** 2) / 200.0
        second = delta * (60.0 * u * (2.0 * u**2 - 3.0 * u + 1.0)) / 200.0**2
        return float(second / (1.0 + first**2) ** 1.5)

    @staticmethod
    def _speed_advisory(station: float) -> float:
        station = nominal_mainline_station(station)
        return 27.78 if 550.0 <= station <= 1450.0 else 30.0

    def _road_profile(self, station: float) -> np.ndarray:
        profile = np.zeros((20, 3), dtype=np.float32)
        for index in range(20):
            distance = 25.0 * (index + 1)
            sample_station = min(station + distance, MAIN_ROUTE_DISTANCE_END)
            profile[index] = (
                distance,
                self._road_curvature(sample_station),
                self._speed_advisory(sample_station),
            )
        return profile

    def _cutin_status_value(self) -> float:
        return {
            "idle": 0.0,
            "armed": 0.33,
            "aligning": 0.66,
            "success": 1.0,
            "failed": -1.0,
        }.get(self._cutin_state, 0.0)

    def _update_events_before_step(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        if "leader" not in active:
            return
        for vehicle_id in self._lane_change_obstacle_ids:
            if vehicle_id in active:
                self.connection.vehicle.setSpeed(
                    vehicle_id, self._lane_change_obstacle_speed
                )
        leader_station = self._station("leader")
        leader_nominal_station = nominal_mainline_station(leader_station)
        self._protect_platoon_from_ordinary_background_cutins(active)
        self._control_background_ramp_merges(active)
        if (
            self._cutin_enabled
            and self._cutin_state == "idle"
            and leader_nominal_station >= 1400.0
        ):
            self._spawn_cutin()
        self._control_cutin(active, leader_station)
        self._control_wave(active, leader_station)

    def _update_events_after_step(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        if "leader" not in active:
            return
        leader_station = self._station("leader")
        self._control_cutin(active, leader_station)
        self._update_lane_change_traffic_metrics(active)
        self._update_lane_change_mission(active)

    def _update_lane_change_mission(self, active: set[str]) -> None:
        self._lane_change_completed_this_step = 0
        if (
            not self._cooperative_lane_change_enabled
            or self._lane_change_event_index >= len(self._lane_change_target_lanes)
            or not all(vehicle_id in active for vehicle_id in PLATOON_IDS)
        ):
            return
        obstacle_station = self._lane_change_obstacle_stations[
            self._lane_change_event_index
        ]
        leader_station = nominal_mainline_station(self._station("leader"))
        if (
            not self._lane_change_event_active
            and leader_station >= obstacle_station - LANE_CHANGE_TRIGGER_DISTANCE
        ):
            self._lane_change_event_active = True
            self._lane_change_event_started_at = self.sim_time
            self._event_log.append(
                {
                    "time": self.sim_time,
                    "event": "cooperative_lane_change_started",
                    "maneuver_index": self._lane_change_event_index + 1,
                    "target_lane": self._lane_change_target_lanes[
                        self._lane_change_event_index
                    ],
                }
            )
            self._spawn_lane_change_traffic_gate(
                self._lane_change_event_index,
                self._lane_change_target_lanes[self._lane_change_event_index],
            )
        if not self._lane_change_event_active:
            return
        target_lane = self._lane_change_target_lanes[self._lane_change_event_index]
        lanes = {
            vehicle_id: int(self.connection.vehicle.getLaneIndex(vehicle_id))
            for vehicle_id in PLATOON_IDS
        }
        lateral_offsets = {
            vehicle_id: float(
                self.connection.vehicle.getLateralLanePosition(vehicle_id)
            )
            for vehicle_id in PLATOON_IDS
        }
        if self.continuous_lateral_model:
            lateral_speeds = {
                vehicle_id: float(
                    self.connection.vehicle.getLateralSpeed(vehicle_id)
                )
                for vehicle_id in PLATOON_IDS
            }
            lateral_transition_active = any(
                abs(speed) > 0.02 for speed in lateral_speeds.values()
            )
        else:
            lateral_transition_active = any(
                abs(offset) > 0.05 for offset in lateral_offsets.values()
            )
        if len(set(lanes.values())) > 1 or lateral_transition_active:
            self._lane_change_dual_lane_time += self.step_length
        if (
            all(lane == target_lane for lane in lanes.values())
            and not lateral_transition_active
        ):
            if self._lane_change_settled_since is None:
                self._lane_change_settled_since = self.sim_time
            if self.sim_time - self._lane_change_settled_since >= LANE_CHANGE_SETTLE_TIME:
                duration = self.sim_time - float(
                    self._lane_change_event_started_at or self.sim_time
                )
                execution_duration = self.sim_time - float(
                    self._lane_change_first_command_at or self.sim_time
                )
                self._lane_change_execution_time += execution_duration
                completed_index = self._lane_change_event_index
                if completed_index < len(self._lane_change_obstacle_ids):
                    obstacle_id = self._lane_change_obstacle_ids[completed_index]
                    if obstacle_id in active:
                        self.connection.vehicle.remove(obstacle_id)
                self._lane_change_event_index += 1
                self._lane_change_completed_this_step = 1
                self._lane_change_event_active = False
                self._lane_change_event_started_at = None
                self._lane_change_settled_since = None
                self._lane_change_first_command_at = None
                self._event_log.append(
                    {
                        "time": self.sim_time,
                        "event": "cooperative_lane_change_completed",
                        "maneuver_index": self._lane_change_event_index,
                        "duration_s": duration,
                        "execution_duration_s": execution_duration,
                        "cleared_obstacle": self._lane_change_obstacle_ids[
                            completed_index
                        ],
                    }
                )
        else:
            self._lane_change_settled_since = None

    def _spawn_cutin(self) -> None:
        try:
            self.connection.vehicle.add(
                CUTIN_ID,
                routeID="rampRoute",
                typeID="human_aggressive",
                depart="now",
                departLane="0",
                departPos="150",
                departSpeed=f"{self._cutin_depart_speed_mps:.3f}",
            )
            self._cutin_state = "armed"
            self._event_log.append({"time": self.sim_time, "event": "cutin_armed"})
        except traci.TraCIException as error:
            self._cutin_error = str(error)

    def _left_merge_clearance(self, vehicle_id: str) -> tuple[bool, dict[str, float]]:
        """Evaluate a continuous left merge against both target-lane neighbors."""

        ego_speed = float(self.connection.vehicle.getSpeed(vehicle_id))
        leaders = self.connection.vehicle.getLeftLeaders(vehicle_id)
        followers = self.connection.vehicle.getLeftFollowers(vehicle_id)

        front_gap = math.inf
        front_speed = ego_speed
        if leaders:
            front_id, front_gap = min(leaders, key=lambda item: item[1])
            front_gap = float(front_gap)
            front_speed = float(self.connection.vehicle.getSpeed(front_id))

        rear_gap = math.inf
        rear_speed = ego_speed
        if followers:
            rear_id, rear_gap = min(followers, key=lambda item: item[1])
            rear_gap = float(rear_gap)
            rear_speed = float(self.connection.vehicle.getSpeed(rear_id))

        # Keep enough room throughout the two-second lateral maneuver, not only
        # at its start.  The rear margin is deliberately larger because a slow
        # ramp vehicle cutting in front of a fast mainline vehicle is the dominant
        # collision mechanism in this scenario.
        front_headway = 0.25 if vehicle_id == CUTIN_ID else 0.5
        front_required = max(
            MIN_SAFE_GAP,
            front_headway * ego_speed,
            MIN_SAFE_GAP + MERGE_LOOKAHEAD * max(0.0, ego_speed - front_speed),
        )
        rear_required = max(
            MIN_SAFE_GAP + 2.0,
            0.6 * rear_speed,
            MIN_SAFE_GAP
            + 2.0
            + MERGE_LOOKAHEAD * max(0.0, rear_speed - ego_speed),
        )
        if vehicle_id == CUTIN_ID:
            # Stress evaluation scales the accepted headways while retaining
            # hard physical room. This separates controller-created gaps from
            # SUMO's comfortable lane-change decision.
            front_required = max(
                2.0, self._cutin_clearance_scale * front_required
            )
            rear_required = max(
                3.0, self._cutin_clearance_scale * rear_required
            )
        neighbor_speeds = [
            speed
            for present, speed in ((bool(leaders), front_speed), (bool(followers), rear_speed))
            if present
        ]
        target_speed = float(np.mean(neighbor_speeds)) if neighbor_speeds else ego_speed
        speed_aligned = abs(ego_speed - target_speed) <= 5.0
        snapshot = {
            "front_gap": front_gap,
            "front_required": front_required,
            "rear_gap": rear_gap,
            "rear_required": rear_required,
            "ego_speed": ego_speed,
            "target_speed": target_speed,
            "speed_aligned": float(speed_aligned),
        }
        return (
            front_gap >= front_required
            and rear_gap >= rear_required
            and speed_aligned,
            snapshot,
        )

    @staticmethod
    def _needs_background_lane_change_guard(
        vehicle_station: float,
        vehicle_lane: int,
        leader_station: float,
        tail_station: float,
    ) -> bool:
        target_lane = platoon_lane_index(vehicle_station)
        adjacent_to_platoon_lane = (
            vehicle_lane != target_lane
            and abs(vehicle_lane - target_lane) == 1
        )
        return bool(
            adjacent_to_platoon_lane
            and tail_station - BACKGROUND_PLATOON_GUARD_REAR
            <= vehicle_station
            <= leader_station + BACKGROUND_PLATOON_GUARD_FRONT
        )

    def _protect_platoon_from_ordinary_background_cutins(
        self, active: set[str]
    ) -> None:
        """Reserve deliberate cut-in testing for the controlled cut-in actor."""

        if not all(vehicle_id in active for vehicle_id in ("leader", "follower_5")):
            return
        leader_station = self._station("leader")
        tail_station = self._station("follower_5")
        candidates = {
            vehicle_id
            for vehicle_id in active
            if vehicle_id.startswith("background_")
            and vehicle_id not in self._ramp_origin_ids
        }
        guarded_now: set[str] = set()
        for vehicle_id in candidates:
            station = self._station(vehicle_id)
            lane = int(self.connection.vehicle.getLaneIndex(vehicle_id))
            if self._needs_background_lane_change_guard(
                station, lane, leader_station, tail_station
            ):
                guarded_now.add(vehicle_id)
                if vehicle_id not in self._background_lane_change_guarded:
                    self.connection.vehicle.setLaneChangeMode(
                        vehicle_id, SAFETY_ONLY_LANE_CHANGE_MODE
                    )
                    self._event_log.append(
                        {
                            "time": self.sim_time,
                            "event": "background_cutin_guard_activated",
                            "vehicle": vehicle_id,
                            "station": station,
                            "lane": lane,
                        }
                    )
            elif vehicle_id in self._background_lane_change_guarded:
                self.connection.vehicle.setLaneChangeMode(
                    vehicle_id, DEFAULT_LANE_CHANGE_MODE
                )
        self._background_lane_change_guarded = guarded_now

    def _request_safe_left_merge(self, vehicle_id: str) -> tuple[bool, dict[str, float]]:
        clearance, snapshot = self._left_merge_clearance(vehicle_id)
        sumo_allows = bool(self.connection.vehicle.couldChangeLane(vehicle_id, 1))
        snapshot["sumo_allows"] = float(sumo_allows)
        forced_request = bool(
            vehicle_id == CUTIN_ID and self._cutin_force_request
        )
        snapshot["forced_request"] = float(forced_request)
        if not clearance or (not sumo_allows and not forced_request):
            return False, snapshot
        if forced_request:
            self.connection.vehicle.setLaneChangeMode(vehicle_id, 0)
        self.connection.vehicle.changeLaneRelative(
            vehicle_id, 1, MERGE_LANE_CHANGE_DURATION
        )
        return True, snapshot

    @staticmethod
    def _ramp_merge_wait_acceleration(
        remaining_distance: float,
        ego_speed: float,
        target_speed: float,
    ) -> float | None:
        """Align with the target lane and preserve room for a stopped merge."""

        remaining_distance = max(0.0, float(remaining_distance))
        ego_speed = max(0.0, float(ego_speed))
        target_speed = max(0.0, float(target_speed))
        if (
            remaining_distance > MERGE_SPEED_CONTROL_DISTANCE
            and abs(ego_speed - target_speed) <= 5.0
        ):
            return None
        safe_stopping_speed = math.inf
        if remaining_distance <= MERGE_SPEED_CONTROL_DISTANCE:
            usable_distance = max(
                0.0, remaining_distance - MERGE_TERMINAL_BUFFER
            )
            safe_stopping_speed = math.sqrt(
                2.0 * MERGE_COMFORTABLE_DECELERATION * usable_distance
            )
        desired_speed = min(target_speed, safe_stopping_speed)
        return float(
            np.clip(
                (desired_speed - ego_speed) / MERGE_SPEED_RESPONSE_TIME,
                -4.0,
                1.5,
            )
        )

    @staticmethod
    def _ramp_queue_safe_acceleration(
        front_gap: float,
        ego_speed: float,
        front_speed: float,
    ) -> float | None:
        """Cap a merge-wait command by the same-lane braking envelope."""

        available_distance = max(
            0.0,
            float(front_gap) - MIN_SAFE_GAP - MERGE_QUEUE_BUFFER,
        )
        ego_speed = max(0.0, float(ego_speed))
        front_speed = max(0.0, float(front_speed))
        braking_safe_speed = math.sqrt(
            front_speed**2
            + 2.0 * MERGE_QUEUE_BRAKING * available_distance
        )
        closing_safe_speed = (
            front_speed
            + available_distance / MERGE_QUEUE_CLOSING_TIME
        )
        safe_speed = min(braking_safe_speed, closing_safe_speed)
        if ego_speed <= safe_speed:
            return None
        return float(
            np.clip(
                (safe_speed - ego_speed) / MERGE_QUEUE_RESPONSE_TIME,
                -MERGE_QUEUE_EMERGENCY_DECELERATION,
                0.0,
            )
        )

    def _ramp_queue_safety(
        self,
        vehicle_id: str,
        ego_speed: float,
    ) -> tuple[float | None, dict[str, float]]:
        queue_front = self.connection.vehicle.getLeader(
            vehicle_id, MERGE_SPEED_CONTROL_DISTANCE
        )
        if not queue_front:
            return None, {}
        queue_front_id, queue_front_gap = queue_front
        queue_front_speed = float(
            self.connection.vehicle.getSpeed(queue_front_id)
        )
        acceleration = self._ramp_queue_safe_acceleration(
            float(queue_front_gap), ego_speed, queue_front_speed
        )
        return acceleration, {
            "queue_front_gap": float(queue_front_gap),
            "queue_front_speed": queue_front_speed,
        }

    def _control_background_ramp_merges(self, active: set[str]) -> None:
        """Apply explicit gap acceptance to ordinary ramp-origin traffic."""

        for vehicle_id in sorted(self._ramp_origin_ids.intersection(active)):
            if vehicle_id == CUTIN_ID:
                continue
            road_id = self.connection.vehicle.getRoadID(vehicle_id)
            lane_index = self.connection.vehicle.getLaneIndex(vehicle_id)
            if road_id == "mergeOpen":
                self.connection.vehicle.setLaneChangeMode(
                    vehicle_id, SAFETY_ONLY_LANE_CHANGE_MODE
                )
                if lane_index != 0:
                    continue
                last_request = self._ramp_merge_requested_at.get(
                    vehicle_id, -math.inf
                )
                ego_speed = float(
                    self.connection.vehicle.getSpeed(vehicle_id)
                )
                queue_acceleration, queue_snapshot = self._ramp_queue_safety(
                    vehicle_id, ego_speed
                )
                if queue_acceleration is not None:
                    self.connection.vehicle.setAcceleration(
                        vehicle_id, queue_acceleration, self.step_length
                    )
                    if vehicle_id not in self._ramp_queue_brake_logged:
                        self._ramp_queue_brake_logged.add(vehicle_id)
                        self._event_log.append(
                            {
                                "time": self.sim_time,
                                "event": "ramp_queue_braking",
                                "vehicle": vehicle_id,
                                "station": self._station(vehicle_id),
                                "queue_acceleration": queue_acceleration,
                                **queue_snapshot,
                            }
                        )
                if self.sim_time < last_request + MERGE_LANE_CHANGE_DURATION:
                    continue
                requested, snapshot = self._request_safe_left_merge(vehicle_id)
                if requested:
                    self._ramp_merge_requested_at[vehicle_id] = self.sim_time
                    self._ramp_merge_wait_logged.discard(vehicle_id)
                    self._event_log.append(
                        {
                            "time": self.sim_time,
                            "event": "ramp_merge_requested",
                            "vehicle": vehicle_id,
                            "station": self._station(vehicle_id),
                            **snapshot,
                        }
                    )
                    continue
                lane_id = self.connection.vehicle.getLaneID(vehicle_id)
                remaining_distance = max(
                    0.0,
                    float(self.connection.lane.getLength(lane_id))
                    - float(
                        self.connection.vehicle.getLanePosition(vehicle_id)
                    ),
                )
                wait_acceleration = self._ramp_merge_wait_acceleration(
                    remaining_distance,
                    snapshot["ego_speed"],
                    snapshot["target_speed"],
                )
                snapshot.update(queue_snapshot)
                if queue_acceleration is not None:
                    wait_acceleration = (
                        queue_acceleration
                        if wait_acceleration is None
                        else min(wait_acceleration, queue_acceleration)
                    )
                if wait_acceleration is not None:
                    self.connection.vehicle.setAcceleration(
                        vehicle_id, wait_acceleration, self.step_length
                    )
                if vehicle_id not in self._ramp_merge_wait_logged:
                    self._ramp_merge_wait_logged.add(vehicle_id)
                    self._event_log.append(
                        {
                            "time": self.sim_time,
                            "event": "ramp_merge_waiting",
                            "vehicle": vehicle_id,
                            "station": self._station(vehicle_id),
                            "remaining_distance": remaining_distance,
                            "wait_acceleration": wait_acceleration,
                            "queue_acceleration": queue_acceleration,
                            **snapshot,
                        }
                    )
            elif (
                road_id in ("mergePost", "wave", "recovery")
                and vehicle_id not in self._ramp_merge_released
            ):
                self.connection.vehicle.setLaneChangeMode(
                    vehicle_id, DEFAULT_LANE_CHANGE_MODE
                )
                self._ramp_merge_released.add(vehicle_id)
                self._event_log.append(
                    {
                        "time": self.sim_time,
                        "event": "ramp_merge_completed",
                        "vehicle": vehicle_id,
                        "station": self._station(vehicle_id),
                    }
                )

    def _control_cutin(self, active: set[str], leader_station: float) -> None:
        leader_nominal_station = nominal_mainline_station(leader_station)
        if self._cutin_state == "success":
            if CUTIN_ID not in active:
                return
            if self.sim_time < self._cutin_hold_until:
                target_speed = 0.5 * (
                    self.connection.vehicle.getSpeed("follower_2")
                    + self.connection.vehicle.getSpeed("follower_3")
                )
                current_speed = self.connection.vehicle.getSpeed(CUTIN_ID)
                acceleration = float(np.clip(target_speed - current_speed, -4.0, 2.0))
                self.connection.vehicle.setAcceleration(
                    CUTIN_ID, acceleration, self.step_length
                )
            elif not self._cutin_released:
                self.connection.vehicle.setSpeedMode(CUTIN_ID, 31)
                self.connection.vehicle.setLaneChangeMode(
                    CUTIN_ID, DEFAULT_LANE_CHANGE_MODE
                )
                self._cutin_released = True
            return
        if self._cutin_state not in ("armed", "aligning"):
            return
        if CUTIN_ID not in active:
            if leader_nominal_station > 2250.0:
                self._cutin_state = "failed"
            return
        if not self._cutin_configured:
            self.connection.vehicle.setLaneChangeMode(
                CUTIN_ID, SAFETY_ONLY_LANE_CHANGE_MODE
            )
            self.connection.vehicle.setSpeedMode(CUTIN_ID, 31)
            self.connection.vehicle.setColor(CUTIN_ID, (255, 30, 0, 255))
            self._cutin_configured = True
            self._cutin_state = "aligning"

        if "follower_2" not in active or "follower_3" not in active:
            self._cutin_state = "failed"
            return
        target_speed = 0.5 * (
            self.connection.vehicle.getSpeed("follower_2")
            + self.connection.vehicle.getSpeed("follower_3")
        )
        follower_2_station = self._station("follower_2")
        follower_3_station = self._station("follower_3")
        desired_front_clearance = max(MIN_SAFE_GAP, 0.25 * target_speed)
        desired_rear_clearance = max(MIN_SAFE_GAP + 2.0, 0.6 * target_speed)
        rear_constrained_target = (
            follower_3_station
            + self.connection.vehicle.getLength(CUTIN_ID)
            + desired_rear_clearance
        )
        front_constrained_target = (
            follower_2_station
            - self.connection.vehicle.getLength("follower_2")
            - desired_front_clearance
        )
        # Center the aggressive car inside the physically available slot. The
        # explicit gap check below still controls whether the lane change starts.
        target_station = 0.5 * (
            rear_constrained_target + front_constrained_target
        )
        cutin_station = self._station(CUTIN_ID)
        cutin_speed = self.connection.vehicle.getSpeed(CUTIN_ID)
        road_id = self.connection.vehicle.getRoadID(CUTIN_ID)
        lane_index = self.connection.vehicle.getLaneIndex(CUTIN_ID)
        on_acceleration_lane = road_id in ("ramp", "rampAccel") or (
            road_id == "mergeOpen" and lane_index == 0
        )
        merge_end_station = MAIN_EDGE_DISTANCE_START["mergePost"]
        if on_acceleration_lane and target_station < merge_end_station - 5.0:
            target_arrival_time = max(
                (merge_end_station - target_station) / max(target_speed, 1.0),
                self.step_length,
            )
            approach_speed = 27.78 if road_id == "ramp" else 33.33
            distance_to_merge = max(0.0, merge_end_station - cutin_station)
            # Cubic terminal-state guidance: meet both the gap arrival time and
            # its longitudinal speed.  This turns the parallel lane into an
            # actual acceleration lane instead of reaching the mainline slowly.
            acceleration = (
                6.0 * distance_to_merge / target_arrival_time**2
                - (4.0 * cutin_speed + 2.0 * target_speed) / target_arrival_time
            )
            if cutin_speed >= approach_speed and acceleration > 0.0:
                acceleration = 0.0
            if cutin_speed <= 6.0 and acceleration < 0.0:
                acceleration = 0.0
            acceleration = float(np.clip(acceleration, -4.0, 2.8))
        else:
            acceleration = float(
                np.clip(
                    0.45 * (target_station - cutin_station)
                    + 0.7 * (target_speed - cutin_speed),
                    -4.0,
                    2.8,
                )
            )
        self.connection.vehicle.setAcceleration(CUTIN_ID, acceleration, self.step_length)
        if (
            road_id == "mergeOpen"
            and lane_index > 0
            and not self._cutin_open_merge_logged
        ):
            self._cutin_open_merge_logged = True
            self._event_log.append(
                {
                    "time": self.sim_time,
                    "event": "cutin_entered_mainline",
                    "station": cutin_station,
                }
            )

        lane_index = self.connection.vehicle.getLaneIndex(CUTIN_ID)
        target_lane = platoon_lane_index(cutin_station)
        if (
            road_id in ("mergeOpen", "mergePost")
            and lane_index < target_lane
            and self.sim_time >= self._cutin_next_lane_change_time
        ):
            next_lane = lane_index + 1
            longitudinally_aligned = (
                self._station("follower_3")
                < cutin_station
                < self._station("follower_2")
            )
            speed_tolerance = 5.0 if next_lane < target_lane else 3.0
            speed_aligned = abs(cutin_speed - target_speed) <= speed_tolerance
            may_enter_next_lane = speed_aligned and (
                next_lane < target_lane or longitudinally_aligned
            )
            if may_enter_next_lane:
                requested, snapshot = self._request_safe_left_merge(CUTIN_ID)
                waiting_key = (road_id, lane_index)
                if requested:
                    self._cutin_next_lane_change_time = (
                        self.sim_time + MERGE_RETRY_INTERVAL
                    )
                    self._cutin_waiting_lanes.discard(waiting_key)
                    if not self._cutin_merge_committed:
                        self._cutin_merge_committed = True
                        self._event_log.append(
                            {
                                "time": self.sim_time,
                                "event": "cutin_merge_committed",
                                "station": cutin_station,
                                **snapshot,
                            }
                        )
                elif waiting_key not in self._cutin_waiting_lanes:
                    self._cutin_waiting_lanes.add(waiting_key)
                    self._event_log.append(
                        {
                            "time": self.sim_time,
                            "event": "cutin_merge_waiting",
                            "station": cutin_station,
                            "target_lane": next_lane,
                            **snapshot,
                        }
                    )

        inside_platoon = (
            self._station("follower_5")
            < cutin_station
            < self._station("leader")
        )
        if (
            road_id in ("mergeOpen", "mergePost")
            and self.connection.vehicle.getLaneIndex(CUTIN_ID) == target_lane
            and inside_platoon
        ):
            self._mark_cutin_success(cutin_station, target_speed)
        elif leader_nominal_station > 2300.0:
            self._cutin_state = "blocked"
            self._event_log.append(
                {
                    "time": self.sim_time,
                    "event": "cutin_blocked",
                    "road_id": road_id,
                    "station": cutin_station,
                    "target_station": target_station,
                }
            )

    def _mark_cutin_success(self, station: float, target_speed: float) -> None:
        self._cutin_state = "success"
        self._cutin_hold_until = self.sim_time + 2.0
        self._cutin_released = False
        self.connection.vehicle.setSpeedMode(CUTIN_ID, 31)
        current_speed = self.connection.vehicle.getSpeed(CUTIN_ID)
        acceleration = float(np.clip(target_speed - current_speed, -4.0, 2.0))
        self.connection.vehicle.setAcceleration(CUTIN_ID, acceleration, self.step_length)
        self._event_log.append(
            {"time": self.sim_time, "event": "cutin_success", "station": station}
        )

    def _control_wave(self, active: set[str], leader_station: float) -> None:
        if self._wave_mode == "off" or self._selected_wave_source is None:
            return
        target = self._selected_wave_source
        if self._wave_profile is not None:
            if target not in active:
                self._event_log.append(
                    {"time": self.sim_time, "event": "wave_source_lost", "target": target}
                )
                self._wave_profile = None
                return
            if self._wave_profile_index < len(self._wave_profile):
                acceleration = float(self._wave_profile[self._wave_profile_index])
                self.connection.vehicle.setAcceleration(target, acceleration, self.step_length)
                self._wave_profile_index += 1
                return
            self._wave_profile = None
            self.connection.vehicle.setSpeedMode(target, 31)

        leader_nominal_station = nominal_mainline_station(leader_station)
        station_triggered = bool(self._wave_trigger_stations)
        if (
            not station_triggered
            or self._wave_trigger_reference == "leader"
        ) and (leader_nominal_station < 2400.0 or leader_nominal_station > 3100.0):
            return
        if (
            not station_triggered
            and self._next_wave_time is not None
            and self.sim_time + 1e-9 < self._next_wave_time
        ):
            return
        valid_sources = [
            vehicle_id
            for vehicle_id in self._front_ids
            if vehicle_id in active
            and leader_station + self._wave_min_source_distance_m
            <= self._station(vehicle_id)
            <= MAIN_EDGE_DISTANCE_START["recovery"]
        ]
        if target not in valid_sources:
            if not valid_sources:
                self._event_log.append(
                    {"time": self.sim_time, "event": "wave_source_unavailable"}
                )
                if not station_triggered:
                    self._next_wave_time = self.sim_time + 15.0
                return
            target = str(self.np_random.choice(sorted(valid_sources)))
            self._selected_wave_source = target
        if station_triggered:
            if self._wave_trigger_index >= len(self._wave_trigger_stations):
                return
            trigger_progress = (
                nominal_mainline_station(self._station(target))
                if self._wave_trigger_reference == "source"
                else leader_nominal_station
            )
            if (
                trigger_progress + 1e-9
                < self._wave_trigger_stations[self._wave_trigger_index]
            ):
                return
        else:
            if self._wave_started_at is None:
                self._wave_started_at = self.sim_time
                self._next_wave_time = self.sim_time
            if (
                self._next_wave_time is None
                or self.sim_time + 1e-9 < self._next_wave_time
            ):
                return
        if target not in active:
            self._event_log.append(
                {"time": self.sim_time, "event": "wave_source_unavailable", "target": target}
            )
            self._next_wave_time = self.sim_time + 15.0
            return
        if self._wave_mode == "fixed":
            profile_seed = 0
            profile = (
                SEVERE_FIXED_WAVE_PROFILE
                if self._fixed_wave_profile == "severe"
                else FIXED_WAVE_PROFILE
            )
            self._wave_profile = profile.copy()
        else:
            if self._wave_policy is None:
                from .wave_runtime import LearnedWavePolicy

                self._wave_policy = LearnedWavePolicy(self._configured_wave_model_path)
            condition = self._wave_policy.condition(
                self.connection, target, self._vehicle_history[target], self._station_cache
            )
            profile_seed = int(self.np_random.integers(1, 2**31 - 1))
            self._wave_profile = self._wave_policy.sample(condition, profile_seed)
        self._wave_profile_index = 0
        self.connection.vehicle.setSpeedMode(target, 0)
        self.connection.vehicle.setColor(target, (255, 165, 0, 255))
        self._wave_target = target
        self._wave_brake_until = self.sim_time + len(self._wave_profile) * self.step_length
        if station_triggered:
            trigger_station = self._wave_trigger_stations[self._wave_trigger_index]
            self._wave_trigger_index += 1
        else:
            trigger_station = None
            self._next_wave_time = self.sim_time + 15.0
        self._wave_pending = False
        self._event_log.append(
            {
                "time": self.sim_time,
                "event": "wave_brake",
                "target": target,
                "station": self._station(target),
                "configured_trigger_station": trigger_station,
                "trigger_reference": self._wave_trigger_reference,
                "profile_seed": profile_seed,
                "profile_steps": int(len(self._wave_profile)),
                "profile_min_acceleration": float(self._wave_profile.min()),
            }
        )

    @staticmethod
    def _route_from_station(station: float) -> tuple[str, float, list[str]]:
        for index, edge in enumerate(MAIN_EDGES):
            start = MAIN_EDGE_START[edge]
            end = start + MAIN_EDGE_LENGTH[edge]
            if station < end or edge == MAIN_EDGES[-1]:
                return edge, station - start, list(MAIN_EDGES[index:])
        raise ValueError(station)

    def _update_leader_stall_state(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        if "leader" not in active:
            self._leader_restart_opportunity = False
            self._leader_stall_duration = 0.0
            return

        leader_speed = float(self.connection.vehicle.getSpeed("leader"))
        front = self.connection.vehicle.getLeader("leader", 200.0)
        if not front or front[0] not in active:
            restart_opportunity = True
        else:
            front_speed = float(self.connection.vehicle.getSpeed(front[0]))
            front_gap = max(0.0, float(front[1]))
            restart_opportunity = (
                front_gap >= max(10.0, MIN_SAFE_GAP + 0.35 * leader_speed)
                or (front_speed >= RESTART_FRONT_SPEED and front_gap >= MIN_SAFE_GAP)
            )

        lane_change_wait_grace = (
            bool(getattr(self, "_cooperative_lane_change_enabled", False))
            and bool(getattr(self, "_lane_change_event_active", False))
            and getattr(self, "_lane_change_event_started_at", None) is not None
            and self.sim_time - float(self._lane_change_event_started_at)
            < float(
                getattr(
                    self, "_lane_change_stall_grace", LANE_CHANGE_STALL_GRACE
                )
            )
        )
        self._leader_restart_opportunity = restart_opportunity
        if (
            leader_speed < RESTART_SPEED_THRESHOLD
            and restart_opportunity
            and not lane_change_wait_grace
        ):
            self._leader_stall_duration += self.step_length
        else:
            self._leader_stall_duration = 0.0

        if (
            not self._stalled_failure
            and self._leader_stall_duration >= STALLED_FAILURE_TIMEOUT
        ):
            self._stalled_failure = True
            self._event_log.append(
                {
                    "time": self.sim_time,
                    "event": "leader_stalled_with_clear_path",
                    "station": self._station("leader"),
                    "duration": self._leader_stall_duration,
                }
            )

    def _pair_gap_status(
        self, follower: str, active: set[str]
    ) -> tuple[float, bool, bool]:
        predecessor = PREDECESSOR[follower]
        if follower not in active or predecessor not in active:
            return 0.0, False, False
        follower_station = self._station(follower)
        predecessor_station = self._station(predecessor)
        pair_gap = max(
            0.0, predecessor_station - follower_station - PLATOON_LENGTH
        )
        humans = active.difference(PLATOON_IDS)
        follower_lane = int(self.connection.vehicle.getLaneIndex(follower))
        follower_road = self.connection.vehicle.getRoadID(follower)
        corridor_start = follower_station - 10.0
        corridor_end = predecessor_station + 10.0
        threat = False
        intrusion = False
        for vehicle_id in humans:
            station = self._station(vehicle_id)
            if not corridor_start <= station <= corridor_end:
                continue
            lane = int(self.connection.vehicle.getLaneIndex(vehicle_id))
            road = self.connection.vehicle.getRoadID(vehicle_id)
            if abs(lane - follower_lane) == 1:
                direction = 1 if follower_lane > lane else -1
                lane_change_state = int(
                    self.connection.vehicle.getLaneChangeState(vehicle_id, direction)[0]
                )
                intended_direction = tc.LCA_LEFT if direction > 0 else tc.LCA_RIGHT
                scripted_merge = (
                    vehicle_id == CUTIN_ID
                    and self._cutin_state in ("armed", "aligning")
                ) or vehicle_id in self._ramp_merge_requested_at
                threat = threat or bool(
                    lane_change_state & intended_direction
                ) or scripted_merge
            if (
                lane == follower_lane
                and road == follower_road
                and follower_station < station < predecessor_station
            ):
                intrusion = True
        return pair_gap, threat, intrusion

    def _platoon_gap_context(
        self, active: set[str]
    ) -> tuple[list[float], list[float], int, int, float]:
        gaps: list[float] = []
        targets: list[float] = []
        threat_count = 0
        intrusion_count = 0
        compact_defenses = 0
        for follower in FOLLOWER_IDS:
            if follower not in active or PREDECESSOR[follower] not in active:
                continue
            pair_gap, threat, intrusion = self._pair_gap_status(follower, active)
            target = CUTIN_TARGET_GAP if threat else TARGET_GAP
            gaps.append(pair_gap)
            targets.append(target)
            threat_count += int(threat)
            intrusion_count += int(intrusion)
            compact_defenses += int(
                threat and MIN_SAFE_GAP <= pair_gap <= CUTIN_TARGET_GAP + 1.0
            )
        defense_ratio = compact_defenses / max(threat_count, 1)
        return gaps, targets, threat_count, intrusion_count, defense_ratio

    def _update_recovery_roles(self) -> None:
        active = set(self.connection.vehicle.getIDList())
        self._rejoined_this_step = 0
        self._rejoined_vehicles_this_step = set()
        for follower in FOLLOWER_IDS:
            if follower not in active or PREDECESSOR[follower] not in active:
                continue
            pair_gap, _, intrusion = self._pair_gap_status(follower, active)
            disconnected = pair_gap > TARGET_GAP + RECOVERY_ACTIVATION_MARGIN
            if disconnected:
                self._disconnect_since.setdefault(follower, self.sim_time)
            else:
                self._disconnect_since.pop(follower, None)
            if (
                (intrusion or disconnected)
                and follower not in self._recovery_leaders
            ):
                self._recovery_leaders.add(follower)
                self._recovery_started_at[follower] = self.sim_time
                self._recovery_clear_since.pop(follower, None)
                self._event_log.append(
                    {
                        "time": self.sim_time,
                        "event": "subleader_activated",
                        "vehicle": follower,
                        "gap": pair_gap,
                        "reason": "intrusion" if intrusion else "gap_disconnect",
                    }
                )
            if follower not in self._recovery_leaders:
                continue
            reconnected = (
                not intrusion
                and pair_gap <= TARGET_GAP
            )
            if not reconnected:
                self._recovery_clear_since.pop(follower, None)
                continue
            clear_since = self._recovery_clear_since.setdefault(
                follower, self.sim_time
            )
            if self.sim_time - clear_since < REJOIN_HOLD_TIME:
                continue
            self._recovery_leaders.remove(follower)
            self._recovery_started_at.pop(follower, None)
            self._recovery_clear_since.pop(follower, None)
            self._rejoined_this_step += 1
            self._rejoined_vehicles_this_step.add(follower)
            self._event_log.append(
                {
                    "time": self.sim_time,
                    "event": "platoon_rejoined",
                    "vehicle": follower,
                    "gap": pair_gap,
                }
            )

    @staticmethod
    def _restart_path_available(
        ego_speed: float,
        front_speed: float | None = None,
        front_gap: float | None = None,
    ) -> bool:
        if front_speed is None or front_gap is None:
            return True
        required_gap = max(TARGET_GAP, 3.0 + 0.5 * ego_speed)
        return bool(
            front_gap >= required_gap
            and front_speed >= max(RESTART_FRONT_SPEED, ego_speed)
        )

    @staticmethod
    def _restart_reward_terms(
        speed: float,
        acceleration: float,
        path_available: bool,
    ) -> tuple[float, float]:
        if not path_available or speed >= FOLLOWER_RESTART_RECOVERY_SPEED:
            return 0.0, 0.0
        urgency = float(
            np.clip(
                (FOLLOWER_RESTART_RECOVERY_SPEED - speed)
                / FOLLOWER_RESTART_RECOVERY_SPEED,
                0.0,
                1.0,
            )
        )
        drive = urgency * float(
            np.clip(max(0.0, acceleration) / MAX_ACCELERATION, 0.0, 1.0)
        )
        inaction = urgency * float(
            np.clip(
                (FOLLOWER_RESTART_ACCELERATION - acceleration)
                / FOLLOWER_RESTART_ACCELERATION,
                0.0,
                1.0,
            )
        )
        return drive, inaction

    @staticmethod
    def _normalized_huber(error: float, cap: float = 20.0) -> float:
        magnitude = abs(float(error))
        value = 0.5 * magnitude * magnitude if magnitude <= 1.0 else magnitude - 0.5
        return min(value, cap)

    @staticmethod
    def _gap_tracking_error(gaps: list[float], targets: list[float]) -> float:
        if not gaps:
            return 0.0
        return float(
            np.mean(
                [
                    PlatoonParallelEnv._normalized_huber(
                        (gap - target) / max(target, 1.0)
                    )
                    for gap, target in zip(gaps, targets)
                ]
            )
        )

    def _episode_state(self) -> tuple[bool, bool, bool, bool]:
        active = set(self.connection.vehicle.getIDList())
        colliding = set(self.connection.simulation.getCollidingVehiclesIDList())
        get_collisions = getattr(self.connection.simulation, "getCollisions", None)
        collision_items = get_collisions() if callable(get_collisions) else []
        collision_details = [
            {
                "collider": item.collider,
                "victim": item.victim,
                "collider_type": item.colliderType,
                "victim_type": item.victimType,
                "collider_speed": float(item.colliderSpeed),
                "victim_speed": float(item.victimSpeed),
                "collision_type": item.type,
                "lane": item.lane,
                "position": float(item.pos),
            }
            for item in collision_items
        ]
        self._last_collision_ids = sorted(colliding)
        self._last_collision_details = collision_details
        if collision_details:
            self._event_log.append(
                {
                    "time": self.sim_time,
                    "event": "collision",
                    "vehicles": self._last_collision_ids.copy(),
                    "details": [item.copy() for item in collision_details],
                }
            )
        controlled_traffic = set(PLATOON_IDS).union(self._ramp_origin_ids, {CUTIN_ID})
        collision = bool(colliding.intersection(controlled_traffic))
        pair_gaps, _, _, intrusion_count, _ = self._platoon_gap_context(active)
        complete_platoon = all(vehicle_id in active for vehicle_id in PLATOON_IDS)
        formation_connected = (
            complete_platoon
            and len(pair_gaps) == len(FOLLOWER_IDS)
            and all(
            gap <= TARGET_GAP + REJOIN_GAP_TOLERANCE for gap in pair_gaps
            )
        )
        if getattr(self, "_cooperative_lane_change_enabled", False):
            success = bool(
                self._lane_change_event_index
                >= len(self._lane_change_target_lanes)
                and nominal_mainline_station(self._station("leader"))
                >= self._lane_change_goal_station
                and formation_connected
            )
        else:
            success = bool(
                nominal_mainline_station(self._station("leader")) >= 3450.0
                and formation_connected
                and intrusion_count == 0
            )
        missing = not complete_platoon
        failure = bool((missing or self._stalled_failure) and not (success or collision))
        truncated = bool(self.sim_time >= self.max_time and not (success or collision or failure))
        return success, collision, failure, truncated

    def _shared_reward(
        self, success: bool, collision: bool, failure: bool
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        active = set(self.connection.vehicle.getIDList())
        speeds = []
        accelerations = []
        headway_risks = []
        ttc_risks = []
        braking_risks = []
        local_headway_risks: dict[str, float] = {}
        local_ttc_risks: dict[str, float] = {}
        local_braking_risks: dict[str, float] = {}
        normalized_power = []
        fuel_mg_per_second = []
        for vehicle_id in PLATOON_IDS:
            if vehicle_id not in active:
                continue
            speed = self.connection.vehicle.getSpeed(vehicle_id)
            speeds.append(speed)
            acceleration = self.connection.vehicle.getAcceleration(vehicle_id)
            accelerations.append((vehicle_id, acceleration))
            normalized_power.append(
                self._tractive_power(speed, acceleration)
                / 300_000.0
            )
            fuel_mg_per_second.append(
                max(0.0, self.connection.vehicle.getFuelConsumption(vehicle_id))
            )
            if vehicle_id not in FOLLOWER_IDS:
                continue
            physical_front = self.connection.vehicle.getLeader(vehicle_id, 200.0)
            if not physical_front or physical_front[0] not in active:
                continue
            front_speed = self.connection.vehicle.getSpeed(physical_front[0])
            gap = max(
                0.0,
                float(physical_front[1])
                + self.connection.vehicle.getMinGap(vehicle_id),
            )
            closing_speed = max(0.0, speed - front_speed)
            dynamic_gap = 3.0 + 0.35 * speed + 0.8 * closing_speed
            headway_risk = max(
                0.0, (dynamic_gap - gap) / max(dynamic_gap, 1.0)
            ) ** 2
            headway_risks.append(headway_risk)
            ttc = gap / max(closing_speed, 1e-3) if closing_speed > 0.0 else math.inf
            ttc_risk = max(0.0, (3.0 - ttc) / 3.0) ** 2
            ttc_risks.append(ttc_risk)
            required_deceleration = max(
                0.0,
                (speed**2 - front_speed**2) / (2.0 * max(gap - 2.0, 1.0)),
            )
            braking_risk = min(
                1.0, max(0.0, (required_deceleration - 1.5) / 4.5)
            ) ** 2
            braking_risks.append(braking_risk)
            local_headway_risks[vehicle_id] = headway_risk
            local_ttc_risks[vehicle_id] = ttc_risk
            local_braking_risks[vehicle_id] = braking_risk

        (
            platoon_gaps,
            desired_gaps,
            cutin_threat_count,
            intrusion_count,
            compact_defense_ratio,
        ) = self._platoon_gap_context(active)
        self._last_cutin_threat_count = cutin_threat_count
        self._last_platoon_intrusion_count = intrusion_count
        gap_error = self._gap_tracking_error(platoon_gaps, desired_gaps)
        intrusion_ratio = intrusion_count / max(len(FOLLOWER_IDS), 1)
        disconnect_excess = [
            max(0.0, (gap - TARGET_GAP) / TARGET_GAP)
            for gap in platoon_gaps
        ]
        maximum_disconnect_error = max(
            (
                self._normalized_huber(excess)
                for excess in disconnect_excess
            ),
            default=0.0,
        )
        recovery_closing_rates: list[float] = []
        recovery_opening_rates: list[float] = []
        subleader_tracking_errors: list[float] = []
        subleader_tracking_by_follower: dict[str, float] = {}
        pair_gaps: dict[str, float] = {}
        pair_targets: dict[str, float] = {}
        pair_gap_errors: dict[str, float] = {}
        pair_closing_progress: dict[str, float] = {}
        pair_opening_errors: dict[str, float] = {}
        pair_intrusions: dict[str, bool] = {}
        pair_threats: dict[str, bool] = {}
        gap_pbrs_by_follower: dict[str, float] = {}
        physical_gap_by_follower: dict[str, float] = {}
        physical_front_speed_by_follower: dict[str, float] = {}
        predicted_speed_by_follower: dict[str, float] = {}
        predicted_acceleration_by_follower: dict[str, float] = {}
        pair_stalls: dict[str, float] = {}
        restart_drive_by_follower: dict[str, float] = {}
        restart_inaction_by_follower: dict[str, float] = {}
        recovery_stalls = 0
        for follower in FOLLOWER_IDS:
            if follower not in active or PREDECESSOR[follower] not in active:
                continue
            pair_gap, threat, intrusion = self._pair_gap_status(follower, active)
            previous_gap = self._previous_pair_gaps.get(follower, pair_gap)
            self._previous_pair_gaps[follower] = pair_gap
            predecessor = PREDECESSOR[follower]
            predecessor_speed = float(self.connection.vehicle.getSpeed(predecessor))
            speed = float(self.connection.vehicle.getSpeed(follower))
            gap_excess = max(0.0, (pair_gap - TARGET_GAP) / TARGET_GAP)
            opening_speed = max(0.0, predecessor_speed - speed)
            closing_rate = (previous_gap - pair_gap) / self.step_length
            pair_gaps[follower] = pair_gap
            pair_targets[follower] = (
                CUTIN_TARGET_GAP
                if threat
                else (
                    LANE_CHANGE_COMPACT_GAP
                    if self._cooperative_lane_change_enabled
                    and self._lane_change_event_active
                    else TARGET_GAP
                )
            )
            pair_gap_errors[follower] = self._normalized_huber(gap_excess)
            pair_intrusions[follower] = intrusion
            pair_threats[follower] = threat
            current_gap_potential = self._gap_potential(
                pair_gap, pair_targets[follower]
            )
            gap_pbrs_by_follower[follower] = FOLLOWER_GAP_PBRS_WEIGHT * (
                FOLLOWER_GAP_PBRS_DISCOUNT * current_gap_potential
                - self._previous_gap_potentials.get(
                    follower, current_gap_potential
                )
            )
            self._previous_gap_potentials[follower] = current_gap_potential
            pair_closing_progress[follower] = (
                float(np.clip(max(0.0, closing_rate) / 5.0, 0.0, 1.0))
                if pair_gap > TARGET_GAP + REJOIN_GAP_TOLERANCE and not intrusion
                else 0.0
            )
            pair_opening_errors[follower] = (
                self._normalized_huber(opening_speed / 3.0, cap=5.0)
                * min(1.0 + gap_excess, 4.0)
                if pair_gap > TARGET_GAP and not intrusion
                else 0.0
            )
            physical_front = self.connection.vehicle.getLeader(follower, 200.0)
            restart_path_available = self._restart_path_available(speed)
            if physical_front and physical_front[0] in active:
                physical_gap_by_follower[follower] = max(
                    0.0, float(physical_front[1])
                )
                physical_front_speed_by_follower[follower] = float(
                    self.connection.vehicle.getSpeed(physical_front[0])
                )
                restart_path_available = self._restart_path_available(
                    speed,
                    physical_front_speed_by_follower[follower],
                    float(physical_front[1]),
                )
            else:
                physical_gap_by_follower[follower] = 500.0
                physical_front_speed_by_follower[follower] = predecessor_speed
            if follower in self._predecessor_delay_buffers:
                _, predicted_speed, predicted_acceleration, _, _ = (
                    self._predecessor_delay_buffers[follower].estimate(
                        self.sim_time
                    )
                )
            else:
                predicted_speed = predecessor_speed
                predicted_acceleration = float(
                    self.connection.vehicle.getAcceleration(predecessor)
                )
            predicted_speed_by_follower[follower] = max(
                0.0,
                predicted_speed
                + FOLLOWER_PREDICTION_HORIZON * predicted_acceleration,
            )
            predicted_acceleration_by_follower[follower] = predicted_acceleration
            if follower in self._follower_control_predictions:
                (
                    predicted_speed_by_follower[follower],
                    predicted_acceleration_by_follower[follower],
                ) = self._follower_control_predictions[follower]
            restart_drive, restart_inaction = self._restart_reward_terms(
                speed,
                float(self.connection.vehicle.getAcceleration(follower)),
                restart_path_available,
            )
            restart_drive_by_follower[follower] = restart_drive
            restart_inaction_by_follower[follower] = restart_inaction
            if (
                pair_gap > TARGET_GAP + REJOIN_GAP_TOLERANCE
                and not intrusion
            ):
                recovery_closing_rates.append(
                    pair_closing_progress[follower]
                )
                recovery_opening_rates.append(
                    float(np.clip(max(0.0, -closing_rate) / 5.0, 0.0, 1.0))
                )
            if follower not in self._recovery_leaders:
                pair_stalls[follower] = 0.0
                continue
            v2i = self._v2i_flow(self._station(follower), active)
            reference_speed = float(v2i[0, 0])
            if reference_speed <= 0.0:
                reference_speed = self._speed_advisory(self._station(follower))
            reference_speed = min(
                reference_speed, self._speed_advisory(self._station(follower))
            )
            tracking_error = ((speed - reference_speed) / 30.0) ** 2
            subleader_tracking_errors.append(tracking_error)
            subleader_tracking_by_follower[follower] = tracking_error
            if (
                not intrusion
                and pair_gap > TARGET_GAP + REJOIN_GAP_TOLERANCE
                and speed < RESTART_SPEED_THRESHOLD
            ):
                physical_front = self.connection.vehicle.getLeader(follower, 200.0)
                path_available = self._restart_path_available(speed)
                if physical_front and physical_front[0] in active:
                    path_available = self._restart_path_available(
                        speed,
                        float(
                            self.connection.vehicle.getSpeed(physical_front[0])
                        ),
                        float(physical_front[1]),
                    )
                stalled = float(path_available)
                recovery_stalls += int(stalled)
                pair_stalls[follower] = stalled
            else:
                pair_stalls[follower] = 0.0
        recovery_progress = float(
            np.mean(recovery_closing_rates)
        ) if recovery_closing_rates else 0.0
        recovery_opening = float(
            np.mean(recovery_opening_rates)
        ) if recovery_opening_rates else 0.0
        subleader_tracking_error = float(
            np.mean(subleader_tracking_errors)
        ) if subleader_tracking_errors else 0.0
        recovery_stall_ratio = recovery_stalls / max(len(FOLLOWER_IDS), 1)
        restart_drive_ratio = float(
            np.mean(list(restart_drive_by_follower.values()))
        ) if restart_drive_by_follower else 0.0
        restart_inaction_ratio = float(
            np.mean(list(restart_inaction_by_follower.values()))
        ) if restart_inaction_by_follower else 0.0
        recovery_active_ratio = len(self._recovery_leaders) / max(len(FOLLOWER_IDS), 1)
        rejoin_bonus = float(self._rejoined_this_step)
        speed_fluctuation = float(np.std(speeds) / TARGET_SPEED) if speeds else 0.0
        jerks = []
        jerk_by_vehicle: dict[str, float] = {}
        for vehicle_id, acceleration in accelerations:
            previous = self._previous_acceleration.get(vehicle_id, acceleration)
            vehicle_jerk = (acceleration - previous) / self.step_length
            jerk_by_vehicle[vehicle_id] = vehicle_jerk
            self._last_jerk[vehicle_id] = vehicle_jerk
            jerks.append((vehicle_jerk / 10.0) ** 2)
            self._previous_acceleration[vehicle_id] = acceleration
        jerk_error = float(np.mean(jerks)) if jerks else 0.0
        follower_jerk_error = float(
            np.mean(
                [
                    (jerk_by_vehicle[vehicle_id] / 10.0) ** 2
                    for vehicle_id in FOLLOWER_IDS
                    if vehicle_id in jerk_by_vehicle
                ]
            )
        ) if any(vehicle_id in jerk_by_vehicle for vehicle_id in FOLLOWER_IDS) else 0.0
        safety_error = max(headway_risks, default=0.0)
        ttc_error = max(ttc_risks, default=0.0)
        braking_error = max(braking_risks, default=0.0)
        intervention_error = float(
            np.mean(
                [
                    float(
                        self._last_action_info.get(vehicle_id, {}).get(
                            "collision_avoidance_intervened", False
                        )
                    )
                    for vehicle_id in FOLLOWER_IDS
                    if vehicle_id in active
                ]
            )
        ) if any(vehicle_id in active for vehicle_id in FOLLOWER_IDS) else 0.0
        actuator_mismatch_by_follower = {
            vehicle_id: min(
                1.0,
                abs(
                    float(
                        self._last_action_info.get(vehicle_id, {}).get(
                            "requested_acceleration", 0.0
                        )
                    )
                    - float(
                        self._last_action_info.get(vehicle_id, {}).get(
                            "applied_acceleration", 0.0
                        )
                    )
                )
                / 3.0,
            )
            ** 2
            for vehicle_id in FOLLOWER_IDS
            if vehicle_id in active
        }
        actuator_mismatch_error = float(
            np.mean(list(actuator_mismatch_by_follower.values()))
        ) if actuator_mismatch_by_follower else 0.0
        energy_error = float(np.mean(normalized_power)) if normalized_power else 0.0
        acceleration_by_vehicle = dict(accelerations)
        leader_acceleration = acceleration_by_vehicle.get("leader", 0.0)
        amplification = [
            (max(0.0, abs(acceleration) - abs(leader_acceleration)) / 3.0) ** 2
            for vehicle_id, acceleration in accelerations
            if vehicle_id in FOLLOWER_IDS
        ]
        string_error = float(np.mean(amplification)) if amplification else 0.0
        acceleration_alignment = [
            ((acceleration - leader_acceleration) / 3.0) ** 2
            for vehicle_id, acceleration in accelerations
            if vehicle_id in FOLLOWER_IDS
        ]
        alignment_error = (
            float(np.mean(acceleration_alignment))
            if acceleration_alignment
            else 0.0
        )
        progress = float(np.mean(speeds) / 30.0) if speeds else 0.0
        minimum_progress = float(min(speeds) / 30.0) if speeds else 0.0
        cutin_defense_error = 0.0
        cutin_defense_bonus = 0.0
        if (
            self._cutin_state == "aligning"
            and CUTIN_ID in active
            and "follower_2" in active
            and "follower_3" in active
            and nominal_mainline_station(self._station("leader")) >= 1400.0
            and abs(
                self._station(CUTIN_ID)
                - 0.5
                * (
                    self._station("follower_2")
                    + self._station("follower_3")
                )
            )
            <= 120.0
        ):
            defense_gap = self._platoon_gap("follower_3", active)
            cutin_defense_error = (
                max(0.0, defense_gap - CUTIN_TARGET_GAP) / CUTIN_TARGET_GAP
            ) ** 2
            cutin_defense_bonus = float(
                MIN_SAFE_GAP <= defense_gap <= CUTIN_TARGET_GAP + 0.5
            )
        cutin_blocked_now = any(
            event.get("event") == "cutin_blocked"
            and math.isclose(float(event.get("time", -1.0)), self.sim_time, abs_tol=1e-6)
            for event in reversed(self._event_log)
        )
        cutin_success_now = any(
            event.get("event") == "cutin_success"
            and math.isclose(float(event.get("time", -1.0)), self.sim_time, abs_tol=1e-6)
            for event in reversed(self._event_log)
        )

        leader_speed = (
            float(self.connection.vehicle.getSpeed("leader")) if "leader" in active else 0.0
        )
        leader_jerk = jerk_by_vehicle.get("leader", 0.0)

        leader_station = self._station("leader")
        road_target = (
            min(
                self._speed_advisory(
                    min(MAIN_ROUTE_DISTANCE_END, leader_station + distance)
                )
                for distance in np.linspace(0.0, 500.0, 21)
            )
            if self._v2i_enabled
            else self._speed_advisory(leader_station)
        )
        front_speeds: list[float] = []
        front_accelerations: list[float] = []
        front_weights: list[float] = []
        traffic_speed_targets: list[float] = []
        for vehicle_id in active.difference(PLATOON_IDS):
            relative = self._station(vehicle_id) - leader_station
            history = self._vehicle_history.get(vehicle_id)
            if not 0.0 < relative <= 800.0 or not history:
                continue
            latest = history[-1]
            if latest[5] >= 0.5:
                continue
            front_speeds.append(float(latest[2]))
            front_accelerations.append(float(latest[3]))
            front_weights.append(math.exp(-relative / 300.0))
            traffic_speed_targets.append(
                math.sqrt(max(0.0, (float(latest[2]) + 1.0) ** 2 + 1.6 * relative))
            )
        if front_speeds:
            flow_speed = float(np.average(front_speeds, weights=front_weights))
            reference_speed = min(
                road_target, float(np.quantile(traffic_speed_targets, 0.25))
            )
            front_acceleration_rms = float(
                np.sqrt(np.average(np.square(front_accelerations), weights=front_weights))
            )
        else:
            flow_speed = road_target
            reference_speed = road_target
            front_acceleration_rms = 0.0
        if self._speed_reference_override is not None:
            reference_speed = min(
                reference_speed, self._speed_reference_override
            )
            flow_speed = min(flow_speed, self._speed_reference_override)
        reference_speed = float(np.clip(reference_speed, 5.0, 30.0))
        v2i_speed_reward = 1.0 - (
            (leader_speed - reference_speed) / max(reference_speed, 5.0)
        ) ** 2
        speed_deficit = max(
            0.0,
            (reference_speed - leader_speed) / max(reference_speed, 5.0),
        )
        restart_target_speed = min(
            reference_speed, LEADER_RESTART_TARGET_SPEED
        )
        leader_restart_deficit = (
            max(0.0, restart_target_speed - leader_speed)
            / max(restart_target_speed, 1.0)
            if self._leader_restart_opportunity
            else 0.0
        )
        leader_restart_penalty = (
            LEADER_RESTART_WEIGHT * leader_restart_deficit**2
        )
        stalled_with_clear_path = float(
            leader_speed < RESTART_SPEED_THRESHOLD
            and self._leader_restart_opportunity
        )
        curve_overspeed_error = (max(0.0, leader_speed - road_target) / 5.0) ** 2
        current_advisory = self._speed_advisory(leader_station)
        curve_urgency = max(0.0, current_advisory - road_target) / 5.0
        wasted_curve_acceleration = curve_urgency * max(0.0, leader_acceleration) / 1.2
        wave_absorption = self._wave_absorption_reward(
            leader_acceleration, front_acceleration_rms
        )
        downstream_wave = self._downstream_wave_objective(active)
        downstream_wave_reward = downstream_wave["reward"]
        first_follower = FOLLOWER_IDS[0]
        first_gap = pair_gaps.get(first_follower, TARGET_GAP)
        cohesion_gate = self._cohesion_gate(first_gap)
        cohesion_potential = self._cohesion_potential(first_gap)
        cohesion_pbrs = LEADER_PBRS_WEIGHT * (
            LEADER_PBRS_DISCOUNT * cohesion_potential
            - self._previous_cohesion_potential
        )
        cohesion_hold_penalty = COHESION_HOLD_WEIGHT * (
            1.0 - cohesion_gate
        )
        self._previous_cohesion_potential = cohesion_potential
        leader_power_ratio = (
            self._tractive_power(leader_speed, leader_acceleration)
            / LEADER_MAX_TRACTION_POWER
        )
        leader_energy_penalty = LEADER_ENERGY_WEIGHT * leader_power_ratio**1.5
        leader_jerk_penalty = LEADER_JERK_WEIGHT * (leader_jerk / 10.0) ** 2
        gated_v2i_reward = cohesion_gate * v2i_speed_reward
        gated_speed_deficit_penalty = (
            cohesion_gate * LEADER_SPEED_DEFICIT_WEIGHT * speed_deficit**2
        )
        gated_wave_reward = (
            cohesion_gate * LEADER_WAVE_WEIGHT * wave_absorption
        )
        leader_reward = (
            gated_v2i_reward
            + gated_wave_reward
            + downstream_wave_reward
            + cohesion_pbrs
            - cohesion_hold_penalty
            - leader_energy_penalty
            - leader_jerk_penalty
            - leader_restart_penalty
            - gated_speed_deficit_penalty
            - 2.0 * stalled_with_clear_path
            - 5.0 * safety_error
            - 4.0 * ttc_error
            - 4.0 * braking_error
            - 100.0 * float(collision)
            - 50.0 * float(failure)
            + 20.0 * float(success)
        )
        lane_change_reward, lane_change_components = (
            self._cooperative_lane_change_reward(active, success, collision)
        )
        leader_reward += lane_change_reward
        clipped_leader_reward = float(np.clip(leader_reward, -100.0, 50.0))
        follower_rewards: dict[str, float] = {}
        follower_reward_modules: dict[str, list[float]] = {
            "restart": [],
            "tracking": [],
            "anti_cutin_pbrs": [],
            "string": [],
            "barrier": [],
            "comfort": [],
        }
        for follower in FOLLOWER_IDS:
            if follower not in active or PREDECESSOR[follower] not in active:
                follower_rewards[follower] = -50.0 * float(failure)
                continue
            predecessor = PREDECESSOR[follower]
            follower_speed = float(self.connection.vehicle.getSpeed(follower))
            target_gap = pair_targets.get(follower, TARGET_GAP)
            road_speed_limit = self._speed_advisory(self._station(follower))
            follower_reference = (
                min(road_speed_limit, self._speed_reference_override)
                if self._speed_reference_override is not None
                else road_speed_limit
            )
            follower_v2i = (
                self._v2i_flow(self._station(follower), active)
                if self._v2i_enabled
                else np.zeros((8, 4), dtype=np.float32)
            )
            valid_v2i_speed = follower_v2i[:, 0][follower_v2i[:, 0] > 0.0]
            if len(valid_v2i_speed):
                follower_reference = min(
                    follower_reference,
                    float(np.quantile(valid_v2i_speed, 0.25)),
                )
            follower_reference = float(np.clip(follower_reference, 5.0, 30.0))
            physical_gap = physical_gap_by_follower.get(follower, 500.0)
            front_speed = physical_front_speed_by_follower.get(
                follower, follower_reference
            )
            restart_target = self._restart_target_speed(
                physical_gap, front_speed, road_speed_limit
            )
            restart_deficit = max(0.0, restart_target - follower_speed)
            restart_reward = -FOLLOWER_RESTART_WEIGHT * (
                restart_deficit / max(road_speed_limit, 5.0)
            ) ** 2
            predicted_speed = float(
                np.clip(
                    predicted_speed_by_follower.get(follower, front_speed),
                    0.0,
                    follower_reference,
                )
            )
            tracking_reward = -FOLLOWER_TRACKING_WEIGHT * (
                (follower_speed - predicted_speed) / follower_reference
            ) ** 2
            tracking_reward -= FOLLOWER_GAP_TRACKING_WEIGHT * self._normalized_huber(
                (pair_gaps.get(follower, target_gap) - target_gap)
                / max(target_gap, 1.0),
                cap=10.0,
            )
            anti_cutin_reward = gap_pbrs_by_follower.get(follower, 0.0)
            follower_acceleration = acceleration_by_vehicle.get(follower, 0.0)
            predicted_acceleration = predicted_acceleration_by_follower.get(
                follower, 0.0
            )
            acceleration_amplification = max(
                0.0,
                abs(follower_acceleration) - abs(predicted_acceleration),
            )
            recovery_string_gate = (
                0.25
                if pair_gaps.get(follower, target_gap) > target_gap + 1.0
                else 1.0
            )
            string_reward = -FOLLOWER_STRING_WEIGHT * recovery_string_gate * (
                acceleration_amplification / 3.0
            ) ** 2
            local_actuator_mismatch = actuator_mismatch_by_follower.get(
                follower, 0.0
            )
            cbf_margin = self._cbf_margin(
                physical_gap, follower_speed, front_speed
            )
            barrier_penalty = self._cbf_barrier_penalty(cbf_margin)
            barrier_penalty += FOLLOWER_SHIELD_WEIGHT * local_actuator_mismatch
            comfort_penalty = (
                FOLLOWER_COMFORT_JERK_WEIGHT
                * (jerk_by_vehicle.get(follower, 0.0) / 10.0) ** 2
                + FOLLOWER_COMFORT_ACCELERATION_WEIGHT
                * (follower_acceleration / 3.0) ** 2
            )
            local_reward = (
                restart_reward
                + tracking_reward
                + anti_cutin_reward
                + string_reward
                + downstream_wave_reward
                + lane_change_reward
                - barrier_penalty
                - comfort_penalty
                - 100.0 * float(collision)
                - 50.0 * float(failure)
                + 10.0 * float(success)
            )
            follower_rewards[follower] = float(
                np.clip(
                    local_reward,
                    -100.0,
                    35.0 if self._cooperative_lane_change_enabled else 10.0,
                )
            )
            follower_reward_modules["restart"].append(restart_reward)
            follower_reward_modules["tracking"].append(tracking_reward)
            follower_reward_modules["anti_cutin_pbrs"].append(
                anti_cutin_reward
            )
            follower_reward_modules["string"].append(string_reward)
            follower_reward_modules["barrier"].append(-barrier_penalty)
            follower_reward_modules["comfort"].append(-comfort_penalty)
        mean_follower_reward = float(np.mean(list(follower_rewards.values())))
        components = {
            "diagnostic_progress": progress,
            "diagnostic_minimum_progress": minimum_progress,
            "cohesion_gate": cohesion_gate,
            "cohesion_pbrs": cohesion_pbrs,
            "cohesion_hold": -cohesion_hold_penalty,
            "diagnostic_recovery_active_ratio": recovery_active_ratio,
            "diagnostic_platoon_rejoined": rejoin_bonus,
            "v2i_speed_reward": gated_v2i_reward,
            "v2i_speed_deficit": -gated_speed_deficit_penalty,
            "v2i_reference_speed": reference_speed,
            "v2i_flow_speed": flow_speed,
            "v2i_curve_urgency": curve_urgency,
            "v2i_curve_overspeed_error": curve_overspeed_error,
            "v2i_wasted_curve_acceleration": wasted_curve_acceleration,
            "stalled_with_clear_path": -2.0 * stalled_with_clear_path,
            "leader_restart_deficit": -leader_restart_penalty,
            "leader_stall_duration": self._leader_stall_duration,
            "leader_jerk": -leader_jerk_penalty,
            "tractive_energy": -leader_energy_penalty,
            "safety": -5.0 * safety_error,
            "follower_restart": float(
                np.mean(follower_reward_modules["restart"])
            ) if follower_reward_modules["restart"] else 0.0,
            "follower_tracking": float(
                np.mean(follower_reward_modules["tracking"])
            ) if follower_reward_modules["tracking"] else 0.0,
            "follower_anti_cutin_pbrs": float(
                np.mean(follower_reward_modules["anti_cutin_pbrs"])
            ) if follower_reward_modules["anti_cutin_pbrs"] else 0.0,
            "follower_string_prediction": float(
                np.mean(follower_reward_modules["string"])
            ) if follower_reward_modules["string"] else 0.0,
            "follower_cbf_barrier": float(
                np.mean(follower_reward_modules["barrier"])
            ) if follower_reward_modules["barrier"] else 0.0,
            "follower_comfort": float(
                np.mean(follower_reward_modules["comfort"])
            ) if follower_reward_modules["comfort"] else 0.0,
            "cutin_threat_count": float(cutin_threat_count),
            "platoon_intrusion_count": float(intrusion_count),
            "follower_control_reward": mean_follower_reward,
            "wave_absorption": gated_wave_reward,
            "downstream_wave_control": downstream_wave_reward,
            "downstream_wave_active": downstream_wave["active"],
            "downstream_wave_source_speed_sd": downstream_wave[
                "source_speed_sd"
            ],
            "downstream_wave_f5_speed_gain": downstream_wave[
                "formation_tail_speed_gain"
            ],
            "downstream_wave_b5_speed_gain": downstream_wave[
                "background_tail_speed_gain"
            ],
            "downstream_wave_acceleration_gain": downstream_wave[
                "acceleration_gain"
            ],
            "downstream_wave_mobility_ratio": downstream_wave[
                "mobility_ratio"
            ],
            "downstream_wave_mobility_weighted": (
                downstream_wave["active"]
                * downstream_wave["mobility_ratio"]
            ),
            "fuel_mg_per_second": float(np.sum(fuel_mg_per_second)),
            "collision": -100.0 * float(collision),
            "failure": -50.0 * float(failure),
            "success": 20.0 * float(success),
        }
        for follower, reward in follower_rewards.items():
            components[f"{follower}_reward"] = reward
        components.update(lane_change_components)
        return clipped_leader_reward, follower_rewards, components

    def _cooperative_lane_change_reward(
        self, active: set[str], success: bool, collision: bool
    ) -> tuple[float, dict[str, float]]:
        keys = {
            "lane_change_team_reward": 0.0,
            "lane_change_progress_pbrs": 0.0,
            "lane_change_completion": 0.0,
            "lane_change_shape": 0.0,
            "lane_change_road_occupancy": 0.0,
            "lane_change_speed_spread": 0.0,
            "lane_change_command_cost": 0.0,
            "lane_change_traffic_speed": 0.0,
            "lane_change_traffic_braking": 0.0,
            "lane_change_maneuvers_completed": float(
                self._lane_change_event_index
            ),
        }
        if not self._cooperative_lane_change_enabled:
            return 0.0, keys
        complete = all(vehicle_id in active for vehicle_id in PLATOON_IDS)
        if not complete:
            return -25.0, keys
        lanes = np.asarray(
            [
                self.connection.vehicle.getLaneIndex(vehicle_id)
                for vehicle_id in PLATOON_IDS
            ],
            dtype=np.int64,
        )
        speeds = np.asarray(
            [self.connection.vehicle.getSpeed(vehicle_id) for vehicle_id in PLATOON_IDS],
            dtype=np.float64,
        )
        if self._lane_change_event_index < len(self._lane_change_target_lanes):
            target_lane = self._lane_change_target_lanes[
                self._lane_change_event_index
            ]
            completion_ratio = float(np.mean(lanes == target_lane))
        else:
            completion_ratio = 1.0
        progress_potential = (
            self._lane_change_event_index + completion_ratio
        ) / max(len(self._lane_change_target_lanes), 1)
        progress_pbrs = LANE_CHANGE_PROGRESS_WEIGHT * (
            progress_potential - self._lane_change_previous_potential
        )
        self._lane_change_previous_potential = progress_potential

        desired_gap = (
            LANE_CHANGE_COMPACT_GAP
            if self._lane_change_event_active
            else TARGET_GAP
        )
        gaps = [
            self._station(PREDECESSOR[follower])
            - self._station(follower)
            - PLATOON_LENGTH
            for follower in FOLLOWER_IDS
        ]
        shape_error = self._gap_tracking_error(
            gaps, [desired_gap] * len(gaps)
        )
        lane_occupancy = (
            float(max(0, len(set(lanes.tolist())) - 1))
            if self._lane_change_event_active
            else 0.0
        )
        speed_spread = float(np.std(speeds) / 3.0) ** 2
        completion_bonus = (
            LANE_CHANGE_COMPLETION_BONUS
            * float(self._lane_change_completed_this_step)
        )
        time_cost = (
            LANE_CHANGE_TIME_WEIGHT
            if self._lane_change_event_active
            else 0.0
        )
        shape_cost = LANE_CHANGE_SHAPE_WEIGHT * shape_error
        occupancy_cost = (
            LANE_CHANGE_ROAD_OCCUPANCY_WEIGHT * lane_occupancy
        )
        speed_cost = LANE_CHANGE_SPEED_SPREAD_WEIGHT * speed_spread
        command_cost = (
            LANE_CHANGE_COMMAND_WEIGHT
            * self._lane_change_commands_this_step
        )
        traffic_speed_cost = (
            LANE_CHANGE_TRAFFIC_SPEED_WEIGHT
            * self._lane_change_traffic_speed_deficit**2
        )
        traffic_brake_cost = (
            LANE_CHANGE_TRAFFIC_BRAKE_WEIGHT
            * (self._lane_change_traffic_brake_intensity / 3.0) ** 2
        )
        reward = (
            progress_pbrs
            + completion_bonus
            - time_cost
            - shape_cost
            - occupancy_cost
            - speed_cost
            - command_cost
            - traffic_speed_cost
            - traffic_brake_cost
            + LANE_CHANGE_SUCCESS_BONUS * float(success)
            - 25.0 * float(collision)
        )
        keys.update(
            {
                "lane_change_team_reward": float(reward),
                "lane_change_progress_pbrs": float(progress_pbrs),
                "lane_change_completion": float(completion_bonus),
                "lane_change_shape": float(-shape_cost),
                "lane_change_road_occupancy": float(-occupancy_cost),
                "lane_change_speed_spread": float(-speed_cost),
                "lane_change_command_cost": float(-command_cost),
                "lane_change_traffic_speed": float(-traffic_speed_cost),
                "lane_change_traffic_braking": float(-traffic_brake_cost),
                "lane_change_maneuvers_completed": float(
                    self._lane_change_event_index
                ),
            }
        )
        return float(reward), keys

    @staticmethod
    def _restart_target_speed(
        gap: float, front_speed: float, speed_limit: float
    ) -> float:
        safe_space = max(0.0, float(gap) - MIN_SAFE_GAP)
        return min(
            max(0.0, float(speed_limit)),
            math.sqrt(
                max(0.0, float(front_speed)) ** 2
                + 2.0 * MAX_ACCELERATION * safe_space
            ),
        )

    @staticmethod
    def _gap_potential(gap: float, desired_gap: float) -> float:
        normalized_error = (float(gap) - float(desired_gap)) / (
            FOLLOWER_GAP_POTENTIAL_SCALE
        )
        return -PlatoonParallelEnv._normalized_huber(
            normalized_error, cap=20.0
        )

    @staticmethod
    def _cbf_margin(gap: float, ego_speed: float, front_speed: float) -> float:
        closing_speed = max(0.0, float(ego_speed) - float(front_speed))
        return (
            float(gap)
            - MIN_SAFE_GAP
            - FOLLOWER_CBF_TIME_HEADWAY * closing_speed
        )

    @staticmethod
    def _cbf_barrier_penalty(margin: float) -> float:
        if margin >= 0.0:
            return 0.0
        violation = min(6.0, -float(margin))
        return FOLLOWER_BARRIER_WEIGHT * math.expm1(violation)

    @staticmethod
    def _cohesion_gate(gap: float) -> float:
        excess = max(0.0, float(gap) - TARGET_GAP)
        return math.exp(-COHESION_GATE_MU * excess**2)

    @staticmethod
    def _cohesion_potential(gap: float) -> float:
        normalized_excess = max(0.0, float(gap) - TARGET_GAP) / (
            COHESION_POTENTIAL_SCALE
        )
        return -math.log1p(normalized_excess**2)

    @staticmethod
    def _wave_absorption_reward(
        leader_acceleration: float, front_acceleration_amplitude: float
    ) -> float:
        amplitude = abs(float(front_acceleration_amplitude))
        if amplitude < WAVE_ACTIVITY_THRESHOLD:
            return 0.0
        return max(
            0.0,
            1.0 - abs(float(leader_acceleration)) / max(amplitude, 1e-3),
        )

    def _downstream_wave_objective(
        self, active: set[str]
    ) -> dict[str, float]:
        result = {
            "reward": 0.0,
            "active": 0.0,
            "source_speed_sd": 0.0,
            "formation_tail_speed_gain": 0.0,
            "background_tail_speed_gain": 0.0,
            "acceleration_gain": 0.0,
            "mobility_ratio": 1.0,
        }
        if not self._downstream_wave_reward_enabled:
            return result
        if self._selected_wave_source not in active or "follower_5" not in active:
            return result
        if len(self._tail_background_ids) != 5:
            return result
        background_tail = self._tail_background_ids[-1]
        if background_tail not in active:
            return result

        def recent(vehicle_id: str) -> np.ndarray | None:
            history = self._vehicle_history.get(vehicle_id)
            if not history:
                return None
            values = np.asarray(
                list(history)[-DOWNSTREAM_WAVE_WINDOW_STEPS:], dtype=np.float64
            )
            return values if len(values) >= 10 else None

        source = recent(str(self._selected_wave_source))
        formation_tail = recent("follower_5")
        mixed_tail = recent(background_tail)
        if source is None or formation_tail is None or mixed_tail is None:
            return result

        source_speed_sd = float(np.std(source[:, 2]))
        source_acceleration_rms = float(np.sqrt(np.mean(source[:, 3] ** 2)))
        activity = float(
            np.clip(
                max(source_speed_sd / 1.0, source_acceleration_rms / 0.5),
                0.0,
                1.0,
            )
        )
        result["source_speed_sd"] = source_speed_sd
        result["active"] = activity
        if activity <= 0.05:
            return result

        speed_denominator = max(source_speed_sd, 0.25)
        acceleration_denominator = max(source_acceleration_rms, 0.2)
        formation_speed_gain = float(
            np.clip(np.std(formation_tail[:, 2]) / speed_denominator, 0.0, 4.0)
        )
        background_speed_gain = float(
            np.clip(np.std(mixed_tail[:, 2]) / speed_denominator, 0.0, 4.0)
        )
        formation_acceleration_gain = float(
            np.sqrt(np.mean(formation_tail[:, 3] ** 2))
            / acceleration_denominator
        )
        background_acceleration_gain = float(
            np.sqrt(np.mean(mixed_tail[:, 3] ** 2))
            / acceleration_denominator
        )
        acceleration_gain = float(
            np.clip(
                0.35 * formation_acceleration_gain
                + 0.65 * background_acceleration_gain,
                0.0,
                4.0,
            )
        )
        combined_speed_gain = (
            0.35 * formation_speed_gain + 0.65 * background_speed_gain
        )
        source_mean_speed = max(float(np.mean(source[:, 2])), 1.0)
        mobility_ratio = float(
            np.clip(np.mean(mixed_tail[:, 2]) / source_mean_speed, 0.0, 2.0)
        )
        target = self._downstream_wave_target_gain
        speed_excess = max(0.0, combined_speed_gain - target)
        acceleration_excess = max(0.0, acceleration_gain - target)
        mobility_shortfall = max(0.0, 0.8 - mobility_ratio)
        penalty = -self._downstream_wave_reward_weight * activity * (
            speed_excess**2
            + 0.5 * acceleration_excess**2
            + 2.0 * mobility_shortfall**2
        )
        result.update(
            {
                "reward": float(np.clip(penalty, -5.0, 0.0)),
                "formation_tail_speed_gain": formation_speed_gain,
                "background_tail_speed_gain": background_speed_gain,
                "acceleration_gain": acceleration_gain,
                "mobility_ratio": mobility_ratio,
            }
        )
        return result

    @staticmethod
    def _tractive_power(speed: float, acceleration: float) -> float:
        rolling_force = TRUCK_MASS * 9.81 * TRUCK_ROLLING_RESISTANCE
        drag_force = (
            0.5
            * AIR_DENSITY
            * TRUCK_DRAG_COEFFICIENT
            * TRUCK_FRONTAL_AREA
            * speed**2
        )
        inertial_force = TRUCK_MASS * acceleration
        return max(0.0, (rolling_force + drag_force + inertial_force) * speed) / DRIVETRAIN_EFFICIENCY

    def _base_info(self, agent: str) -> dict[str, Any]:
        active = set(self.connection.vehicle.getIDList())
        communication_delay = 0.0
        packet_age = 0.0
        predecessor_delay = 0.0
        predecessor_age = 0.0
        if agent in FOLLOWER_IDS and self._leader_delay_buffers:
            _, _, _, communication_delay, packet_age = self._leader_delay_buffers[agent].estimate(
                self.sim_time
            )
            _, _, _, predecessor_delay, predecessor_age = self._predecessor_delay_buffers[agent].estimate(
                self.sim_time
            )
        return {
            "time": self.sim_time,
            "route_station": self._station(agent),
            "communication_delay": communication_delay,
            "packet_age": packet_age,
            "predecessor_communication_delay": predecessor_delay,
            "predecessor_packet_age": predecessor_age,
            "communication_delay_probability": self._communication_delay_probability,
            "communication_delay_events": self._communication_delay_events,
            "v2i_enabled": self._v2i_enabled,
            "speed_reference": self._speed_reference_override,
            "cutin_state": self._cutin_state,
            "cutin_error": self._cutin_error,
            "wave_active": self.sim_time < self._wave_brake_until,
            "wave_target": self._wave_target,
            "front_vehicle_count": self._front_vehicle_count,
            "selected_wave_source": self._selected_wave_source,
            "demand_profile": self._demand_profile_id,
            "background_active_count": sum(
                vehicle_id.startswith("background_") for vehicle_id in active
            ),
            "leader_restart_opportunity": self._leader_restart_opportunity,
            "leader_stall_duration": self._leader_stall_duration,
            "leader_stalled_failure": self._stalled_failure,
            "cutin_threat_count": self._last_cutin_threat_count,
            "platoon_intrusion_count": self._last_platoon_intrusion_count,
            "temporary_subleaders": sorted(self._recovery_leaders),
            "is_temporary_subleader": agent in self._recovery_leaders,
            "platoon_rejoined_this_step": self._rejoined_this_step,
            "cooperative_lane_change_enabled": self._cooperative_lane_change_enabled,
            "lane_change_event_index": self._lane_change_event_index,
            "lane_change_event_active": self._lane_change_event_active,
            "lane_change_maneuvers_completed": self._lane_change_event_index,
            "lane_change_total_commands": self._lane_change_command_count,
            "lane_change_dual_lane_time": self._lane_change_dual_lane_time,
            "lane_change_execution_time": self._lane_change_execution_time,
            "lane_change_traffic_enabled": self._lane_change_traffic_enabled,
            "lane_change_traffic_active_count": sum(
                vehicle_id in active
                for vehicle_id in self._lane_change_traffic_ids
            ),
            "lane_change_traffic_spawned_count": len(
                self._lane_change_traffic_ids
            ),
            "lane_change_traffic_delay_s": self._lane_change_traffic_delay,
            "lane_change_traffic_braking_integral": (
                self._lane_change_traffic_braking_integral
            ),
            "lane_change_traffic_hard_brake_events": (
                self._lane_change_traffic_hard_brake_events
            ),
            "lane_change_traffic_speed_deficit": (
                self._lane_change_traffic_speed_deficit
            ),
            "lane_change_traffic_brake_intensity": (
                self._lane_change_traffic_brake_intensity
            ),
            "lane_change_traffic_max_deceleration": (
                self._lane_change_traffic_max_deceleration
            ),
            "event_log_size": len(self._event_log),
            "collision_ids": self._last_collision_ids.copy(),
            "collision_details": [
                item.copy() for item in self._last_collision_details
            ],
            "collision_involves_platoon": bool(
                set(self._last_collision_ids).intersection(PLATOON_IDS)
            ),
            "collision_involves_ramp": bool(
                set(self._last_collision_ids).intersection(self._ramp_origin_ids)
            ),
            "collision_involves_cutin": CUTIN_ID in self._last_collision_ids,
        }


try:
    from traci.exceptions import FatalTraCIError
except ImportError:  # pragma: no cover - compatibility with older TraCI packages
    FatalTraCIError = traci.TraCIException

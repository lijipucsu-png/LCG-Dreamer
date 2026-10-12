from __future__ import annotations

from collections import deque
import math

import numpy as np
import pytest
import sumolib
from pettingzoo.test import parallel_api_test

from platoon_sumo.build_network import build_network
from platoon_sumo.env import (
    BACKGROUND_MAX_LC_ASSERTIVENESS,
    DEFAULT_LANE_CHANGE_MODE,
    CUTIN_TARGET_GAP,
    FOLLOWER_IDS,
    INITIAL_POSITIONS,
    MAX_BRAKE_RELEASE_JERK,
    MAX_JERK,
    MAX_RESTART_JERK,
    MAIN_EDGE_DISTANCE_START,
    MAIN_EDGE_JUNCTION_LENGTH,
    MAIN_EDGE_LENGTH,
    MIN_SAFE_GAP,
    PLATOON_IDS,
    PLATOON_LANE_INDEX,
    PREDECESSOR,
    REJOIN_GAP_TOLERANCE,
    TARGET_GAP,
    TARGET_SPEED,
    STALLED_FAILURE_TIMEOUT,
    LANE_CHANGE_STALL_GRACE,
    PlatoonParallelEnv,
    V2VDelayBuffer,
    nominal_mainline_station,
)
from platoon_sumo.run_demo import baseline_actions


def test_downstream_wave_objective_penalizes_amplified_tail_motion() -> None:
    env = PlatoonParallelEnv()
    env._downstream_wave_reward_enabled = True
    env._downstream_wave_target_gain = 0.85
    env._downstream_wave_reward_weight = 1.0
    env._selected_wave_source = "source"
    env._tail_background_ids = [f"tail_background_{index}" for index in range(1, 6)]
    phase = np.linspace(0.0, 2.0 * np.pi, 50)

    def history(speed_amplitude: float, acceleration_amplitude: float) -> deque[np.ndarray]:
        return deque(
            (
                np.asarray(
                    [
                        float(index),
                        1.0,
                        22.0 + speed_amplitude * np.sin(value),
                        acceleration_amplitude * np.cos(value),
                        5.0,
                        0.0,
                    ],
                    dtype=np.float32,
                )
                for index, value in enumerate(phase)
            ),
            maxlen=50,
        )

    env._vehicle_history = {
        "source": history(1.0, 0.5),
        "follower_5": history(1.3, 0.7),
        "tail_background_5": history(1.6, 0.9),
    }
    active = {"source", "follower_5", "tail_background_5"}
    result = env._downstream_wave_objective(active)
    assert result["active"] > 0.70
    assert result["formation_tail_speed_gain"] > 1.0
    assert result["background_tail_speed_gain"] > 1.0
    assert result["reward"] < 0.0


def test_downstream_wave_objective_is_default_off() -> None:
    env = PlatoonParallelEnv()
    assert env._downstream_wave_objective(set())["reward"] == 0.0


def test_network_has_exact_mainline_and_ramp() -> None:
    network_path = build_network(force=True)
    network = sumolib.net.readNet(str(network_path))
    route = [network.getEdge(edge) for edge in ("main0", "curve", "mergePre", "mainAlongRamp", "mergeOpen", "mergePost", "wave", "recovery")]
    assert math.isclose(sum(edge.getLength() for edge in route), 3500.0, abs_tol=0.1)
    assert all(len(edge.getLanes()) == 3 for edge in route if edge.getID() != "mergeOpen")
    merge_open = network.getEdge("mergeOpen")
    assert len(merge_open.getLanes()) == 4
    main_merge = network.getEdge("mainAlongRamp")
    for lane in main_merge.getLanes():
        lane_y = [point[1] for point in lane.getShape()]
        assert math.isclose(max(lane_y), min(lane_y), abs_tol=0.01)
    assert len(network.getEdge("curve").getShape()) >= 150
    assert len(network.getEdge("ramp").getLanes()) == 1
    assert len(network.getEdge("rampAccel").getLanes()) == 1
    assert math.isclose(network.getEdge("rampAccel").getLength(), 350.0 / 3.0, abs_tol=0.1)
    assert math.isclose(
        network.getEdge("rampAccel").getLength() + merge_open.getLength(),
        500.0,
        abs_tol=0.1,
    )
    acceleration_y = network.getEdge("rampAccel").getLanes()[0].getShape()[0][1]
    main_right_y = main_merge.getLanes()[0].getShape()[0][1]
    assert acceleration_y < main_right_y
    assert [lane.getShape()[0][1] for lane in merge_open.getLanes()[1:]] == [176.5, 180.0, 183.5]
    assert [lane.getShape()[-1][1] for lane in merge_open.getLanes()[1:]] == [176.5, 180.0, 183.5]
    assert merge_open.getLanes()[0].getShape()[0][1] < merge_open.getLanes()[1].getShape()[0][1]
    assert math.isclose(
        merge_open.getLanes()[0].getShape()[-1][1],
        merge_open.getLanes()[1].getShape()[-1][1],
        abs_tol=0.01,
    )
    assert network.getEdge("rampAccel") in network.getEdge("ramp").getOutgoing()
    assert merge_open in network.getEdge("rampAccel").getOutgoing()
    assert network.getEdge("mergePost") in merge_open.getOutgoing()


def test_background_lane_changes_remain_autonomous_with_safe_gap_acceptance() -> None:
    class VehicleAPI:
        def __init__(self) -> None:
            self.parameters: list[tuple[str, str, str]] = []

        @staticmethod
        def getTypeID(_vehicle_id: str) -> str:
            return "ngsim_passenger_aggressive"

        @staticmethod
        def setMaxSpeed(_vehicle_id: str, _maximum: float) -> None:
            return None

        def setParameter(self, vehicle_id: str, key: str, value: str) -> None:
            self.parameters.append((vehicle_id, key, value))

        @staticmethod
        def setSpeedMode(_vehicle_id: str, _mode: int) -> None:
            return None

        @staticmethod
        def setLaneChangeMode(_vehicle_id: str, _mode: int) -> None:
            raise AssertionError("ordinary background traffic must keep mode 1621")

    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    vehicle_api = VehicleAPI()
    env.connection = type("Connection", (), {"vehicle": vehicle_api})()
    env._background_speed_configured = set()
    env._background_max_speeds = {"ngsim_passenger_aggressive": 31.0}
    env._ramp_origin_ids = set()

    env._configure_background_free_flow_speeds({"background_1"})

    assert DEFAULT_LANE_CHANGE_MODE == 1621
    assert vehicle_api.parameters == [
        (
            "background_1",
            "laneChangeModel.lcAssertive",
            str(BACKGROUND_MAX_LC_ASSERTIVENESS),
        )
    ]
    assert env._background_speed_configured == {"background_1"}


def test_physical_route_station_includes_internal_junction_lanes() -> None:
    assert math.isclose(
        MAIN_EDGE_DISTANCE_START["mergeOpen"],
        MAIN_EDGE_DISTANCE_START["mainAlongRamp"]
        + MAIN_EDGE_LENGTH["mainAlongRamp"]
        + MAIN_EDGE_JUNCTION_LENGTH["mainAlongRamp"],
    )
    assert math.isclose(
        MAIN_EDGE_DISTANCE_START["mergePost"],
        MAIN_EDGE_DISTANCE_START["mergeOpen"]
        + MAIN_EDGE_LENGTH["mergeOpen"]
        + MAIN_EDGE_JUNCTION_LENGTH["mergeOpen"],
    )
    assert math.isclose(
        nominal_mainline_station(MAIN_EDGE_DISTANCE_START["mergeOpen"]),
        1716.667,
        abs_tol=1e-6,
    )
    assert math.isclose(
        nominal_mainline_station(MAIN_EDGE_DISTANCE_START["mergePost"]),
        2100.0,
        abs_tol=1e-6,
    )


def test_fifo_delay_and_constant_acceleration_compensation() -> None:
    buffer = V2VDelayBuffer()
    buffer.prime(0.0, station=10.0, speed=20.0, acceleration=2.0)
    first = buffer.push(0.1, 12.0, 20.0, 2.0, delay=0.2)
    second = buffer.push(0.2, 14.0, 20.0, 2.0, delay=0.1)
    assert first.deliver_at <= second.deliver_at
    buffer.deliver(0.3)
    station, speed, acceleration, delay, age = buffer.estimate(0.4)
    assert math.isclose(station, 18.04, abs_tol=1e-6)
    assert math.isclose(speed, 20.4, abs_tol=1e-6)
    assert acceleration == 2.0
    assert 0.1 <= delay <= 0.2
    assert math.isclose(age, 0.2)

    normal = V2VDelayBuffer()
    normal.prime(0.0, station=0.0, speed=10.0, acceleration=0.0)
    packet = normal.push(
        0.1,
        1.0,
        10.0,
        0.0,
        delay=0.0,
        planned_acceleration=1.0,
        future_speeds=tuple(10.0 + index for index in range(10)),
    )
    assert packet.deliver_at == packet.generated_at
    normal.deliver(0.1)
    assert normal.estimate(0.1)[3] == 0.0
    assert normal.plan()[0] == 1.0
    assert normal.plan()[-1] == 19.0


def test_communication_delay_events_are_random_optional_and_bounded() -> None:
    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    env.np_random = np.random.default_rng(9)
    env._communication_delay_events = 0
    env._communication_delay_probability = 0.0
    env._communication_delay_min = 0.1
    env._communication_delay_max = 0.2
    assert [env._sample_v2v_delay() for _ in range(20)] == [0.0] * 20

    env._communication_delay_probability = 1.0
    delays = np.asarray([env._sample_v2v_delay() for _ in range(100)])
    assert np.all((0.1 <= delays) & (delays <= 0.2))
    assert float(np.ptp(delays)) > 0.05
    assert env._communication_delay_events == 100


def test_dynamic_gap_reward_and_general_background_cutin_detection() -> None:
    assert PlatoonParallelEnv._gap_tracking_error([8.0], [CUTIN_TARGET_GAP]) == 0.0
    assert PlatoonParallelEnv._gap_tracking_error([8.0], [TARGET_GAP]) > 0.0
    assert PlatoonParallelEnv._normalized_huber((300.0 - TARGET_GAP) / TARGET_GAP) > (
        PlatoonParallelEnv._normalized_huber((48.0 - TARGET_GAP) / TARGET_GAP)
    )

    assert PlatoonParallelEnv._cohesion_gate(12.0) == pytest.approx(1.0)
    assert PlatoonParallelEnv._cohesion_gate(15.0) < 0.02
    assert PlatoonParallelEnv._cohesion_potential(30.0) < (
        PlatoonParallelEnv._cohesion_potential(15.0)
    )
    assert PlatoonParallelEnv._wave_absorption_reward(0.0, 0.49) == 0.0
    assert PlatoonParallelEnv._wave_absorption_reward(1.0, 2.0) == pytest.approx(0.5)
    assert PlatoonParallelEnv._restart_target_speed(5.0, 0.0, 30.0) == 0.0
    assert PlatoonParallelEnv._restart_target_speed(12.0, 25.0, 30.0) > 25.0
    assert PlatoonParallelEnv._restart_target_speed(200.0, 25.0, 30.0) == 30.0
    assert PlatoonParallelEnv._gap_potential(8.0, 8.0) == 0.0
    assert PlatoonParallelEnv._gap_potential(12.0, 8.0) < 0.0
    assert PlatoonParallelEnv._cbf_margin(8.0, 25.0, 25.0) > 0.0
    assert PlatoonParallelEnv._cbf_barrier_penalty(1.0) == 0.0
    assert PlatoonParallelEnv._cbf_barrier_penalty(-2.0) > 0.0

    positions = dict(INITIAL_POSITIONS)
    positions["background_test"] = 110.0
    lanes = {vehicle_id: PLATOON_LANE_INDEX for vehicle_id in PLATOON_IDS}
    lanes["background_test"] = PLATOON_LANE_INDEX - 1
    roads = {vehicle_id: "main0" for vehicle_id in positions}

    class Vehicles:
        @staticmethod
        def getLeader(vehicle_id: str, distance: float) -> tuple[str, float]:
            return PREDECESSOR.get(vehicle_id, ""), 10.0

        @staticmethod
        def getMinGap(vehicle_id: str) -> float:
            return 2.0

        @staticmethod
        def getLaneIndex(vehicle_id: str) -> int:
            return lanes[vehicle_id]

        @staticmethod
        def getRoadID(vehicle_id: str) -> str:
            return roads[vehicle_id]

        @staticmethod
        def getLaneChangeState(vehicle_id: str, direction: int) -> tuple[int, int]:
            return ((2 if direction > 0 else 4), 0)

    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    env.connection = type("Connection", (), {"vehicle": Vehicles()})()
    env._station_cache = positions
    env._cutin_state = "idle"
    env._ramp_merge_requested_at = {}
    active = set(positions)
    gaps, targets, threats, intrusions, _ = env._platoon_gap_context(active)
    assert gaps == [TARGET_GAP] * len(FOLLOWER_IDS)
    assert threats == 1
    assert intrusions == 0
    assert CUTIN_TARGET_GAP in targets
    assert len(gaps) == len(FOLLOWER_IDS)

    lanes["background_test"] = PLATOON_LANE_INDEX
    _, _, threats, intrusions, _ = env._platoon_gap_context(active)
    assert threats == 0
    assert intrusions == 1


def test_ordinary_background_cutin_guard_only_covers_adjacent_platoon_corridor() -> None:
    guard = PlatoonParallelEnv._needs_background_lane_change_guard
    leader_station = 300.0
    tail_station = 180.0

    assert guard(250.0, PLATOON_LANE_INDEX - 1, leader_station, tail_station)
    assert guard(305.0, PLATOON_LANE_INDEX + 1, leader_station, tail_station)
    assert not guard(250.0, PLATOON_LANE_INDEX, leader_station, tail_station)
    assert not guard(120.0, PLATOON_LANE_INDEX - 1, leader_station, tail_station)
    assert not guard(380.0, PLATOON_LANE_INDEX + 1, leader_station, tail_station)


def test_intrusion_activates_subleader_until_gap_is_rejoined() -> None:
    positions = dict(INITIAL_POSITIONS)
    positions["background_test"] = 110.0
    lanes = {vehicle_id: PLATOON_LANE_INDEX for vehicle_id in positions}
    roads = {vehicle_id: "main0" for vehicle_id in positions}
    active = set(positions)

    class Vehicles:
        @staticmethod
        def getIDList() -> list[str]:
            return list(active)

        @staticmethod
        def getLaneIndex(vehicle_id: str) -> int:
            return lanes[vehicle_id]

        @staticmethod
        def getRoadID(vehicle_id: str) -> str:
            return roads[vehicle_id]

        @staticmethod
        def getLaneChangeState(vehicle_id: str, direction: int) -> tuple[int, int]:
            return (0, 0)

    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    env.connection = type("Connection", (), {"vehicle": Vehicles()})()
    env._station_cache = positions
    env._cutin_state = "idle"
    env._ramp_merge_requested_at = {}
    env._recovery_leaders = set()
    env._recovery_started_at = {}
    env._recovery_clear_since = {}
    env._disconnect_since = {}
    env._rejoined_this_step = 0
    env._rejoined_vehicles_this_step = set()
    env._event_log = []
    env.sim_time = 1.0
    env._update_recovery_roles()
    assert "follower_3" in env._recovery_leaders
    assert env._event_log[-1]["event"] == "subleader_activated"

    active.remove("background_test")
    env.sim_time = 2.0
    env._update_recovery_roles()
    env.sim_time = 3.1
    env._update_recovery_roles()
    assert "follower_3" not in env._recovery_leaders
    assert env._rejoined_this_step == 1
    assert env._event_log[-1]["event"] == "platoon_rejoined"


def test_sustained_gap_disconnect_activates_recovery_without_intrusion() -> None:
    positions = dict(INITIAL_POSITIONS)
    positions["follower_1"] = 120.0
    lanes = {vehicle_id: PLATOON_LANE_INDEX for vehicle_id in positions}
    roads = {vehicle_id: "main0" for vehicle_id in positions}

    class Vehicles:
        @staticmethod
        def getIDList() -> list[str]:
            return list(positions)

        @staticmethod
        def getLaneIndex(vehicle_id: str) -> int:
            return lanes[vehicle_id]

        @staticmethod
        def getRoadID(vehicle_id: str) -> str:
            return roads[vehicle_id]

        @staticmethod
        def getLaneChangeState(vehicle_id: str, direction: int) -> tuple[int, int]:
            return (0, 0)

    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    env.connection = type("Connection", (), {"vehicle": Vehicles()})()
    env._station_cache = positions
    env._cutin_state = "idle"
    env._ramp_merge_requested_at = {}
    env._recovery_leaders = set()
    env._recovery_started_at = {}
    env._recovery_clear_since = {}
    env._disconnect_since = {}
    env._rejoined_this_step = 0
    env._rejoined_vehicles_this_step = set()
    env._event_log = []

    env.sim_time = 1.0
    env._update_recovery_roles()
    assert "follower_1" in env._recovery_leaders
    assert env._event_log[-1]["reason"] == "gap_disconnect"


def test_clear_path_stall_becomes_terminal_failure() -> None:
    class Vehicles:
        @staticmethod
        def getIDList() -> list[str]:
            return ["leader"]

        @staticmethod
        def getSpeed(vehicle_id: str) -> float:
            return 0.0

        @staticmethod
        def getLeader(vehicle_id: str, distance: float):
            return None

    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    env.connection = type("Connection", (), {"vehicle": Vehicles()})()
    env.step_length = 0.1
    env.sim_time = 0.0
    env._station_cache = {"leader": 100.0}
    env._leader_restart_opportunity = False
    env._leader_stall_duration = 0.0
    env._stalled_failure = False
    env._event_log = []
    for step in range(int(STALLED_FAILURE_TIMEOUT / env.step_length) + 1):
        env.sim_time = step * env.step_length
        env._update_leader_stall_state()
    assert env._stalled_failure
    assert env._leader_restart_opportunity
    assert env._event_log[-1]["event"] == "leader_stalled_with_clear_path"


def test_active_lane_change_gap_wait_has_bounded_stall_grace() -> None:
    class Vehicles:
        @staticmethod
        def getIDList() -> list[str]:
            return ["leader"]

        @staticmethod
        def getSpeed(vehicle_id: str) -> float:
            return 0.0

        @staticmethod
        def getLeader(vehicle_id: str, distance: float):
            return None

    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    env.connection = type("Connection", (), {"vehicle": Vehicles()})()
    env.step_length = 0.1
    env.sim_time = 0.0
    env._station_cache = {"leader": 100.0}
    env._leader_restart_opportunity = False
    env._leader_stall_duration = 0.0
    env._stalled_failure = False
    env._event_log = []
    env._cooperative_lane_change_enabled = True
    env._lane_change_event_active = True
    env._lane_change_event_started_at = 0.0

    for step in range(int(LANE_CHANGE_STALL_GRACE / env.step_length)):
        env.sim_time = step * env.step_length
        env._update_leader_stall_state()
    assert not env._stalled_failure
    assert env._leader_stall_duration == 0.0

    for step in range(int(STALLED_FAILURE_TIMEOUT / env.step_length) + 2):
        env.sim_time = LANE_CHANGE_STALL_GRACE + step * env.step_length
        env._update_leader_stall_state()
    assert env._stalled_failure
    assert env._event_log[-1]["event"] == "leader_stalled_with_clear_path"


def test_episode_success_requires_all_six_trucks_and_five_connected_gaps() -> None:
    active = set(PLATOON_IDS)

    class Vehicles:
        @staticmethod
        def getIDList() -> list[str]:
            return list(active)

    class Simulation:
        @staticmethod
        def getCollidingVehiclesIDList() -> list[str]:
            return []

    env = PlatoonParallelEnv.__new__(PlatoonParallelEnv)
    env.connection = type(
        "Connection", (), {"vehicle": Vehicles(), "simulation": Simulation()}
    )()
    env._ramp_origin_ids = set()
    env._stalled_failure = False
    env.sim_time = 100.0
    env.max_time = 200.0
    success_station = MAIN_EDGE_DISTANCE_START["recovery"] + 350.0
    env._station = lambda vehicle_id: success_station
    pair_gaps = [TARGET_GAP] * len(FOLLOWER_IDS)
    env._platoon_gap_context = lambda present: (pair_gaps, [], 0, 0, 0.0)

    assert env._episode_state() == (True, False, False, False)

    active.remove("leader")
    pair_gaps.pop()
    assert env._episode_state() == (False, False, True, False)

    active.add("leader")
    pair_gaps.append(TARGET_GAP + REJOIN_GAP_TOLERANCE + 0.1)
    assert env._episode_state() == (False, False, False, False)


def test_restart_reward_requires_positive_acceleration_on_clear_path() -> None:
    assert PlatoonParallelEnv._restart_path_available(0.0)
    assert PlatoonParallelEnv._restart_path_available(0.0, 2.0, 12.0)
    assert not PlatoonParallelEnv._restart_path_available(0.0, 2.0, 5.0)
    assert not PlatoonParallelEnv._restart_path_available(3.0, 2.0, 20.0)

    drive, inaction = PlatoonParallelEnv._restart_reward_terms(
        speed=0.0, acceleration=0.0, path_available=True
    )
    assert drive == 0.0
    assert inaction == 1.0

    drive, inaction = PlatoonParallelEnv._restart_reward_terms(
        speed=0.0, acceleration=1.2, path_available=True
    )
    assert drive == 1.0
    assert inaction == 0.0

    assert PlatoonParallelEnv._restart_reward_terms(0.0, 1.2, False) == (0.0, 0.0)
    assert PlatoonParallelEnv._restart_reward_terms(2.0, 1.2, True) == (0.0, 0.0)


def test_formation_guard_only_limits_a_leader_that_is_escaping() -> None:
    assert (
        PlatoonParallelEnv._formation_guard_acceleration(
            gap=12.0,
            leader_speed=25.0,
            follower_speed=20.0,
            previous_acceleration=0.5,
            step_length=0.1,
        )
        is None
    )
    target = PlatoonParallelEnv._formation_guard_acceleration(
        gap=20.0,
        leader_speed=25.0,
        follower_speed=10.0,
        previous_acceleration=0.5,
        step_length=0.1,
    )
    assert target == pytest.approx(0.35)
    assert (
        PlatoonParallelEnv._formation_guard_acceleration(
            gap=20.0,
            leader_speed=10.0,
            follower_speed=12.0,
            previous_acceleration=0.0,
            step_length=0.1,
        )
        is None
    )


def test_predictive_safety_blocks_acceleration_at_target_gap() -> None:
    at_target = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=12.0,
        ego_speed=25.0,
        front_speed=25.0,
        front_acceleration=0.0,
        minimum_gap=2.5,
        horizon=1.0,
    )
    assert at_target == pytest.approx(0.0)

    steady_deadband = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=11.6,
        ego_speed=25.0,
        front_speed=25.0,
        front_acceleration=0.0,
        minimum_gap=2.5,
        horizon=1.0,
    )
    assert steady_deadband == pytest.approx(0.0)

    with_room = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=25.0,
        ego_speed=25.0,
        front_speed=25.0,
        front_acceleration=0.0,
        minimum_gap=1.0,
        horizon=1.0,
    )
    assert with_room == 1.2

    buffered_external_brake = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=60.0,
        ego_speed=25.0,
        front_speed=25.0,
        front_acceleration=-3.0,
        minimum_gap=1.0,
        horizon=1.0,
    )
    assert buffered_external_brake == 1.2

    emergency_broadcast = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=12.0,
        ego_speed=25.0,
        front_speed=25.0,
        front_acceleration=-4.5,
        minimum_gap=2.5,
        horizon=1.0,
    )
    assert emergency_broadcast == pytest.approx(-4.5)

    compact_restart = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=6.9,
        ego_speed=0.0,
        front_speed=0.2,
        front_acceleration=1.0,
        minimum_gap=2.5,
        horizon=1.0,
    )
    assert compact_restart == pytest.approx(1.2)

    rolling_restart = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=7.0,
        ego_speed=1.0,
        front_speed=1.0,
        front_acceleration=1.0,
        minimum_gap=2.5,
        horizon=1.0,
    )
    assert rolling_restart > 0.0

    compact_coast = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=7.0,
        ego_speed=7.0,
        front_speed=7.0,
        front_acceleration=0.0,
        minimum_gap=2.5,
        horizon=1.0,
    )
    assert compact_coast > -0.1

    buffered_stop = PlatoonParallelEnv._predictive_safe_acceleration(
        net_gap=12.0,
        ego_speed=0.0,
        front_speed=0.0,
        front_acceleration=0.0,
        minimum_gap=2.5,
        horizon=1.0,
    )
    assert buffered_stop == pytest.approx(1.2)


def test_ramp_queue_safety_reserves_reaction_distance() -> None:
    braking = PlatoonParallelEnv._ramp_queue_safe_acceleration(
        front_gap=25.6,
        ego_speed=15.0,
        front_speed=10.2,
    )
    assert braking is not None
    assert braking <= -4.0

    assert (
        PlatoonParallelEnv._ramp_queue_safe_acceleration(
            front_gap=30.0,
            ego_speed=10.0,
            front_speed=10.0,
        )
        is None
    )


def test_cbf_uses_physical_clearance_at_compact_stop() -> None:
    assert PlatoonParallelEnv._cbf_margin(6.9, 0.0, 0.0) > 0.0
    assert PlatoonParallelEnv._cbf_margin(4.9, 0.0, 0.0) < 0.0


def test_stopped_vehicle_uses_fast_positive_jerk_release() -> None:
    acceleration, restart_release = PlatoonParallelEnv._jerk_limited_acceleration(
        requested=1.2,
        previous=-0.13,
        speed=0.0,
        step_length=0.1,
    )
    assert restart_release
    assert acceleration == pytest.approx(
        min(1.2, -0.13 + 0.1 * MAX_RESTART_JERK)
    )

    cruising, restart_release = PlatoonParallelEnv._jerk_limited_acceleration(
        requested=1.2,
        previous=-0.13,
        speed=10.0,
        step_length=0.1,
    )
    assert not restart_release
    assert cruising == pytest.approx(-0.13 + 0.1 * MAX_JERK)

    brake_release, restart_release = (
        PlatoonParallelEnv._jerk_limited_acceleration(
            requested=1.2,
            previous=-3.0,
            speed=10.0,
            step_length=0.1,
        )
    )
    assert not restart_release
    assert brake_release == pytest.approx(-3.0 + 0.1 * MAX_BRAKE_RELEASE_JERK)


def test_synchronous_platoon_acceleration_is_not_rejected_by_safety_layer() -> None:
    env = PlatoonParallelEnv(max_time=20.0)
    try:
        env.reset(
            seed=321,
            options={
                "front_vehicle_count": 0,
                "cutin_enabled": False,
                "wave_mode": "off",
                "communication_delay_probability": 0.0,
                "background_initial_scale": 0.0,
                "background_demand_scale": 0.0,
            },
        )
        maximum_gap_error = 0.0
        for _ in range(50):
            actions = {
                vehicle_id: {
                    "accel": np.asarray([0.12], dtype=np.float32),
                    "lane_change": 0,
                }
                for vehicle_id in PLATOON_IDS
            }
            _, _, _, _, infos = env.step(actions)
            active = set(env.connection.vehicle.getIDList())
            assert not any(
                infos[follower]["predictive_safety_intervened"]
                for follower in FOLLOWER_IDS
            )
            maximum_gap_error = max(
                maximum_gap_error,
                *(
                    abs(env._platoon_gap(follower, active) - TARGET_GAP)
                    for follower in FOLLOWER_IDS
                ),
            )
        assert maximum_gap_error <= 0.2
    finally:
        env.close()


def test_predecessor_braking_ceiling_propagates_in_the_same_control_tick() -> None:
    env = PlatoonParallelEnv(max_time=5.0)
    try:
        env.reset(
            seed=322,
            options={
                "front_vehicle_count": 0,
                "cutin_enabled": False,
                "wave_mode": "off",
                "communication_delay_probability": 0.0,
                "background_initial_scale": 0.0,
                "background_demand_scale": 0.0,
            },
        )
        actions = {
            vehicle_id: {
                "accel": np.asarray(
                    [-3.0 if vehicle_id == "leader" else 0.0],
                    dtype=np.float32,
                ),
                "lane_change": 0,
            }
            for vehicle_id in PLATOON_IDS
        }
        _, _, _, _, infos = env.step(actions)
        leader_applied = infos["leader"]["applied_acceleration"]
        assert leader_applied == pytest.approx(-MAX_JERK * env.step_length)
        assert all(
            infos[follower]["applied_acceleration"]
            == pytest.approx(leader_applied)
            for follower in FOLLOWER_IDS
        )
        assert all(
            infos[follower]["predictive_safety_intervened"]
            for follower in FOLLOWER_IDS
        )
    finally:
        env.close()


def test_restart_intention_propagates_before_residual_brake_releases() -> None:
    env = PlatoonParallelEnv(max_time=5.0)
    try:
        env.reset(
            seed=323,
            options={
                "front_vehicle_count": 0,
                "cutin_enabled": False,
                "wave_mode": "off",
                "communication_delay_probability": 0.0,
                "background_initial_scale": 0.0,
                "background_demand_scale": 0.0,
            },
        )
        for _ in range(100):
            for vehicle_id in PLATOON_IDS:
                env.connection.vehicle.setSpeed(vehicle_id, 0.0)
            env.connection.simulationStep()
            if max(
                env.connection.vehicle.getSpeed(vehicle_id)
                for vehicle_id in PLATOON_IDS
            ) <= 0.01:
                break
        assert all(
            env.connection.vehicle.getSpeed(vehicle_id) <= 0.01
            for vehicle_id in PLATOON_IDS
        )
        env._previous_command = {vehicle_id: -3.0 for vehicle_id in PLATOON_IDS}
        env._collision_shield_active = {
            vehicle_id: True for vehicle_id in PLATOON_IDS
        }
        actions = {
            vehicle_id: {
                "accel": np.asarray([1.0], dtype=np.float32),
                "lane_change": 0,
            }
            for vehicle_id in PLATOON_IDS
        }
        _, _, _, _, infos = env.step(actions)
        expected = min(1.0, -3.0 + MAX_RESTART_JERK * env.step_length)
        assert all(infos[agent]["restart_jerk_release"] for agent in PLATOON_IDS)
        assert all(
            infos[agent]["applied_acceleration"] == pytest.approx(expected)
            for agent in PLATOON_IDS
        )
    finally:
        env.close()


def test_reset_formation_observations_and_parallel_contract() -> None:
    env = PlatoonParallelEnv()
    try:
        observations, infos = env.reset(seed=7)
        assert set(observations) == set(PLATOON_IDS)
        assert set(infos) == set(PLATOON_IDS)
        for agent, observation in observations.items():
            assert env.observation_space(agent).contains(observation)
        positions = [env._station(agent) for agent in PLATOON_IDS]
        assert np.allclose(positions, list(INITIAL_POSITIONS.values()), atol=0.1)
        assert np.allclose(np.diff(positions), [-24.0] * 5, atol=0.1)
        assert all(
            env.connection.vehicle.getLaneIndex(agent) == PLATOON_LANE_INDEX
            for agent in PLATOON_IDS
        )
        assert set(env.action_space("leader").spaces) == {"accel", "lane_change"}
        assert env.action_space("follower_1")["lane_change"].n == 1

        leader_state = observations["leader"]["platoon_state"]
        assert np.all(leader_state[:, 5] == 1.0)
        for follower in FOLLOWER_IDS:
            observation = observations[follower]
            own_row = PLATOON_IDS.index(follower)
            predecessor_row = PLATOON_IDS.index(PREDECESSOR[follower])
            visible = set(np.flatnonzero(observation["platoon_state"][:, 5]))
            assert visible == {0, own_row, predecessor_row}
            assert math.isclose(
                float(observation["local_predecessor"][4]), TARGET_GAP, abs_tol=0.1
            )
            assert math.isclose(
                float(observation["local_front"][4]), TARGET_GAP, abs_tol=0.1
            )
            assert observation["recovery"][0] == 0.0
            assert observation["leader_plan"].shape == (11,)
    finally:
        env.close()


def test_traci_zero_acceleration_is_not_overridden_by_sumo_following() -> None:
    env = PlatoonParallelEnv(max_time=20.0)
    try:
        env.reset(
            seed=123,
            options={
                "front_vehicle_count": 0,
                "cutin_enabled": False,
                "wave_mode": "off",
                "communication_delay_probability": 0.0,
                "background_initial_scale": 0.0,
                "background_demand_scale": 0.0,
            },
        )
        maximum_error = 0.0
        for _ in range(100):
            actions = {
                vehicle_id: {
                    "accel": np.asarray([0.0], dtype=np.float32),
                    "lane_change": 0,
                }
                for vehicle_id in PLATOON_IDS
            }
            env.step(actions)
            active = set(env.connection.vehicle.getIDList())
            maximum_error = max(
                maximum_error,
                *(
                    abs(env._platoon_gap(follower, active) - TARGET_GAP)
                    for follower in FOLLOWER_IDS
                ),
            )
        assert maximum_error <= 0.2
    finally:
        env.close()


def test_v2i_can_be_hidden_during_foundation_curriculum() -> None:
    env = PlatoonParallelEnv(max_time=5.0)
    try:
        observations, infos = env.reset(
            seed=124,
            options={
                "front_vehicle_count": 0,
                "cutin_enabled": False,
                "wave_mode": "off",
                "v2i_enabled": False,
                "speed_reference": 25.0,
                "communication_delay_probability": 0.0,
                "background_initial_scale": 0.0,
                "background_demand_scale": 0.0,
            },
        )
        assert not infos["leader"]["v2i_enabled"]
        assert np.count_nonzero(observations["leader"]["v2i_flow"]) == 0
        assert np.count_nonzero(observations["leader"]["road_profile"]) == 0
        actions = {
            vehicle_id: {
                "accel": np.asarray([0.0], dtype=np.float32),
                "lane_change": 0,
            }
            for vehicle_id in PLATOON_IDS
        }
        _, _, _, _, step_infos = env.step(actions)
        assert step_infos["leader"]["reward_components"][
            "v2i_reference_speed"
        ] == pytest.approx(25.0)
    finally:
        env.close()

    api_env = PlatoonParallelEnv(max_time=5.0)
    parallel_api_test(api_env, num_cycles=20)

    terminal_env = PlatoonParallelEnv(max_time=0.2)
    terminal_observations, _ = terminal_env.reset(seed=11)
    try:
        actions = {
            agent: {
                "accel": np.asarray([0.0], dtype=np.float32),
                "lane_change": 0,
            }
            for agent in terminal_env.agents
        }
        terminal_observations, _, _, truncations, _ = terminal_env.step(actions)
        assert terminal_observations == {}
        assert all(truncations.values())
        assert terminal_env.agents == []
    finally:
        terminal_env.close()


def test_adversarial_events_trigger_with_baseline(tmp_path) -> None:
    torch = __import__("torch")
    from platoon_control.models.wave import ConditionalWaveCVAE

    arguments = {"hidden_dim": 32, "latent_dim": 4, "horizon": 60}
    model = ConditionalWaveCVAE(**arguments)
    wave_path = tmp_path / "wave.pt"
    torch.save({"model_arguments": arguments, "model": model.state_dict()}, wave_path)
    env = PlatoonParallelEnv(wave_model_path=wave_path)
    observations, _ = env.reset(
        seed=42,
        options={
            "front_vehicle_count": 4,
            "wave_mode": "learned",
            "demand_profile": 0,
            "background_initial_scale": 0.0,
        },
    )
    try:
        front_positions = [env._station(vehicle_id) for vehicle_id in env._front_ids]
        front_gaps = np.diff([INITIAL_POSITIONS["leader"], *front_positions]) - 5.0
        assert np.all((14.0 <= front_gaps) & (front_gaps <= 22.0))
        for vehicle_id in env._front_ids:
            assert env.connection.vehicle.getLaneIndex(vehicle_id) == PLATOON_LANE_INDEX
            assert env.connection.vehicle.getLaneChangeMode(vehicle_id) == 512
            assert env.connection.vehicle.getMaxSpeed(vehicle_id) == TARGET_SPEED
        for _ in range(1400):
            if not env.agents:
                break
            commands = baseline_actions(observations)
            leader_station = env._station("leader")
            follower_lane = env.connection.vehicle.getLaneIndex("follower_3")
            follower_road = env.connection.vehicle.getRoadID("follower_3")
            nominal_lane = PLATOON_LANE_INDEX + int(follower_road == "mergeOpen")
            observations, _, terminations, truncations, _ = env.step(
                commands
            )
            if not follower_road.startswith(":"):
                assert follower_lane == nominal_lane
            active = set(env.connection.vehicle.getIDList())
            assert all(
                env.connection.vehicle.getLaneIndex(vehicle_id)
                == (
                    PLATOON_LANE_INDEX
                    + int(env.connection.vehicle.getRoadID(vehicle_id) == "mergeOpen")
                )
                for vehicle_id in env._front_ids
                if vehicle_id in active
                and not env.connection.vehicle.getRoadID(vehicle_id).startswith(":")
            )
            if any(terminations.values()) or any(truncations.values()):
                break
            event_names = {event["event"] for event in env.event_log}
            if (
                {"cutin_success", "cutin_blocked"}.intersection(event_names)
                and "wave_brake" in event_names
                and "ramp_merge_waiting" in event_names
                and "ramp_merge_requested" in event_names
            ):
                break
        event_names = {event["event"] for event in env.event_log}
        assert {"cutin_success", "cutin_blocked"}.intersection(event_names)
        assert "wave_brake" in event_names
        assert "ramp_merge_waiting" in event_names
        assert "ramp_merge_requested" in event_names
        colliding = set(env.connection.simulation.getCollidingVehiclesIDList())
        controlled = set(PLATOON_IDS).union(env._ramp_origin_ids)
        assert not colliding.intersection(controlled)
        cutin_events = [
            event
            for event in env.event_log
            if event["event"] in ("cutin_success", "cutin_blocked")
        ]
        assert 1716.667 <= cutin_events[0]["station"] <= 2300.0
        open_merge_events = [
            event
            for event in env.event_log
            if event["event"] == "cutin_entered_mainline"
        ]
        if "cutin_success" in event_names:
            assert open_merge_events
            assert 1716.667 <= open_merge_events[0]["station"] < 2100.0
        wave_events = [event for event in env.event_log if event["event"] == "wave_brake"]
        assert wave_events[0]["station"] >= 2400.0
        assert wave_events[0]["station"] <= 3100.0
        assert wave_events[0]["target"].startswith("scenario_front_")
    finally:
        env.close()


def test_severe_fixed_wave_profile_is_explicit_and_stronger() -> None:
    from platoon_sumo.env import FIXED_WAVE_PROFILE, SEVERE_FIXED_WAVE_PROFILE

    assert len(SEVERE_FIXED_WAVE_PROFILE) == 120
    assert SEVERE_FIXED_WAVE_PROFILE.min() == pytest.approx(-3.0)
    assert SEVERE_FIXED_WAVE_PROFILE.min() < FIXED_WAVE_PROFILE.min()
    assert abs(float(SEVERE_FIXED_WAVE_PROFILE.sum())) < 1e-4


def test_emergency_stop_prevents_follower_rear_end_collision() -> None:
    env = PlatoonParallelEnv(max_time=20.0)
    observations, _ = env.reset(
        seed=19,
        options={
            "front_vehicle_count": 0,
            "wave_mode": "off",
            "demand_profile": 0,
            "background_initial_scale": 0.0,
        },
    )
    minimum_gap = math.inf
    final_info = {}
    try:
        for step in range(180):
            actions = baseline_actions(observations)
            if step >= 20:
                actions["leader"]["accel"] = np.asarray([-3.0], np.float32)
                for follower in FOLLOWER_IDS:
                    actions[follower]["accel"] = np.asarray([1.2], np.float32)
            observations, _, terminations, truncations, infos = env.step(actions)
            final_info = infos["leader"]
            active = set(env.connection.vehicle.getIDList())
            for follower in FOLLOWER_IDS:
                if follower not in active:
                    continue
                front = env.connection.vehicle.getLeader(follower, 200.0)
                if front:
                    physical_gap = (
                        float(front[1])
                        + env.connection.vehicle.getMinGap(follower)
                    )
                    minimum_gap = min(minimum_gap, physical_gap)
            if any(terminations.values()) or any(truncations.values()):
                break
        assert not final_info.get("collision", False)
        assert minimum_gap >= MIN_SAFE_GAP
        assert math.isclose(
            env.connection.vehicle.getSpeed("leader"), 0.0, abs_tol=0.1
        )
    finally:
        env.close()


def test_calibrated_background_prefills_entire_mainline() -> None:
    env = PlatoonParallelEnv()
    try:
        env.reset(
            seed=42,
            options={"front_vehicle_count": 4, "demand_profile": 0},
        )
        active = set(env.connection.vehicle.getIDList())
        backgrounds = {
            vehicle_id for vehicle_id in active if vehicle_id.startswith("background_")
        }
        assert len(backgrounds) >= 30
        logical_lanes = {
            env.connection.vehicle.getLaneIndex(vehicle_id)
            - int(env.connection.vehicle.getRoadID(vehicle_id) == "mergeOpen")
            for vehicle_id in backgrounds
        }
        assert logical_lanes == {0, 1, 2}
        assert any(
            env._station(vehicle_id) < INITIAL_POSITIONS["follower_5"]
            for vehicle_id in backgrounds
        )
        assert any(
            env._station(vehicle_id) > max(env._front_initial_positions.values())
            for vehicle_id in backgrounds
        )
        assert all(
            env.connection.vehicle.getLaneChangeMode(vehicle_id) != 512
            for vehicle_id in backgrounds
        )
        assert all(
            env.connection.vehicle.getMaxSpeed(vehicle_id)
            >= env.connection.vehicletype.getMaxSpeed(
                env.connection.vehicle.getTypeID(vehicle_id)
            )
            for vehicle_id in backgrounds
        )
    finally:
        env.close()

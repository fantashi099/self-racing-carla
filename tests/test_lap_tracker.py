"""Unit tests for carla_race.lap_tracker (F6)."""
from __future__ import annotations

import math

import pytest

from carla_race.lap_tracker import (
    DEFAULT_CROSSING_THRESHOLD,
    on_lap_complete,
    update_car_progress,
)
from carla_race.race_state import CarState


class Loc:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class Tf:
    def __init__(self, x: float, y: float) -> None:
        self.location = Loc(x, y)


def _circuit_on_circle(n: int, radius: float = 50.0) -> list[Tf]:
    # n waypoints evenly spaced on a circle, starting at (radius, 0)
    out: list[Tf] = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        out.append(Tf(radius * math.cos(angle), radius * math.sin(angle)))
    return out


def _dist(a: Tf, b: Tf) -> float:
    return math.hypot(a.location.x - b.location.x, a.location.y - b.location.y)


def test_update_progress_advances_waypoint_index() -> None:
    circuit = _circuit_on_circle(64)
    car = CarState(actor_id=1, is_player=True, color="r")
    car.waypoint_index = 10
    # move car near waypoint 12
    target = circuit[12]
    update_car_progress(car, target, circuit, distance_fn=_dist)
    assert car.waypoint_index == 12


def test_update_progress_no_circuit_returns_false() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    assert update_car_progress(car, Tf(0, 0), [], distance_fn=_dist) is False


def test_update_progress_returns_false_before_crossing() -> None:
    circuit = _circuit_on_circle(64)
    car = CarState(actor_id=1, is_player=True, color="r")
    car.waypoint_index = 5
    crossed = update_car_progress(car, circuit[5], circuit, distance_fn=_dist)
    assert crossed is False
    assert car.waypoint_index == 5


def test_update_progress_does_not_cross_before_midpoint() -> None:
    """Anti-double-count: a car near the start line that hasn't reached the
    midpoint cannot register a lap by jumping to index 0."""
    circuit = _circuit_on_circle(64)
    car = CarState(actor_id=1, is_player=True, color="r")
    car.waypoint_index = 2  # near start, but hasn't reached midpoint (32)
    # teleport car to start line
    crossed = update_car_progress(car, circuit[0], circuit, distance_fn=_dist)
    assert crossed is False


def test_update_progress_crosses_after_midpoint() -> None:
    circuit = _circuit_on_circle(64)
    car = CarState(actor_id=1, is_player=True, color="r")
    car.waypoint_index = 62  # past midpoint (32), near end
    # move car to start line
    crossed = update_car_progress(car, circuit[0], circuit, distance_fn=_dist)
    assert crossed is True
    assert car.waypoint_index == 0


def test_update_progress_cross_requires_within_threshold() -> None:
    circuit = _circuit_on_circle(64)
    car = CarState(actor_id=1, is_player=True, color="r")
    car.waypoint_index = 62
    # place car far from start line but at index 0's position offset
    far_tf = Tf(circuit[0].location.x + 100, circuit[0].location.y + 100)
    crossed = update_car_progress(
        car, far_tf, circuit, crossing_threshold=8.0, distance_fn=_dist
    )
    assert crossed is False


def test_update_progress_threshold_widens_crossing_zone() -> None:
    circuit = _circuit_on_circle(64)
    car = CarState(actor_id=1, is_player=True, color="r")
    car.waypoint_index = 62
    # 5 units off the start line
    near = Tf(circuit[0].location.x + 5, circuit[0].location.y)
    crossed_default = update_car_progress(car, near, circuit, distance_fn=_dist)
    assert crossed_default is True


def test_update_progress_default_distance_fn_works() -> None:
    """Default 2D distance function should work with structurally compatible
    transforms (carla.Transform has .location.x/y)."""
    circuit = _circuit_on_circle(8)
    car = CarState(actor_id=1, is_player=True, color="r")
    car.waypoint_index = 7
    crossed = update_car_progress(car, circuit[0], circuit)
    assert crossed is True
    assert car.waypoint_index == 0


def test_on_lap_complete_records_split() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    car.laps_finished = 0
    split = on_lap_complete(car, now_s=45.0, race_started_at_s=0.0, num_laps=3)
    assert split.lap_number == 1
    assert split.lap_time_s == 45.0
    assert split.cumulative_time_s == 45.0
    assert car.laps_finished == 1
    assert len(car.splits) == 1
    assert car.lap == 2


def test_on_lap_complete_second_split_subtracts_prev_cumulative() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    on_lap_complete(car, now_s=45.0, race_started_at_s=0.0, num_laps=3)
    split2 = on_lap_complete(car, now_s=100.0, race_started_at_s=0.0, num_laps=3)
    assert split2.lap_number == 2
    assert split2.cumulative_time_s == 100.0
    assert split2.lap_time_s == 55.0
    assert car.laps_finished == 2
    assert car.lap == 3


def test_on_lap_complete_finishes_at_num_laps() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    on_lap_complete(car, now_s=45.0, race_started_at_s=0.0, num_laps=3)
    on_lap_complete(car, now_s=90.0, race_started_at_s=0.0, num_laps=3)
    split3 = on_lap_complete(car, now_s=135.0, race_started_at_s=0.0, num_laps=3)
    assert car.laps_finished == 3
    assert car.finished_at_s == 135.0
    assert car.lap == 3
    # finish_position NOT set here (race_manager assigns it)
    assert car.finish_position is None
    assert split3.lap_number == 3


def test_on_lap_complete_one_lap_race_finishes_immediately() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    split = on_lap_complete(car, now_s=30.0, race_started_at_s=0.0, num_laps=1)
    assert car.laps_finished == 1
    assert car.finished_at_s == 30.0
    assert car.lap == 1
    assert split.lap_number == 1


def test_on_lap_complete_handles_none_started_at() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    split = on_lap_complete(car, now_s=10.0, race_started_at_s=None, num_laps=3)
    assert split.cumulative_time_s == 0.0
    assert split.lap_time_s == 0.0


def test_full_3_lap_flow() -> None:
    circuit = _circuit_on_circle(64)
    car = CarState(actor_id=1, is_player=True, color="r")
    started = 0.0
    crossings = 0
    for lap in range(3):
        # drive from start to midpoint
        for i in range(0, 33):
            update_car_progress(car, circuit[i], circuit, distance_fn=_dist)
        # drive from midpoint back to start
        for i in range(33, 64):
            update_car_progress(car, circuit[i % 64], circuit, distance_fn=_dist)
        # cross start line
        crossed = update_car_progress(car, circuit[0], circuit, distance_fn=_dist)
        if crossed:
            crossings += 1
            on_lap_complete(car, now_s=45.0 * (lap + 1), race_started_at_s=started, num_laps=3)
    assert crossings == 3
    assert car.laps_finished == 3
    assert car.finished_at_s == 135.0
    assert len(car.splits) == 3

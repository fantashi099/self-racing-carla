"""F6 — lap detection on the circuit start line.

Advances a ``CarState.waypoint_index`` along the circuit and detects
start-line crossings to count laps. Pure logic: no CARLA, no I/O.

Contract:
- ``update_car_progress(car, transform, circuit, *, crossing_threshold=8.0,
  distance_fn=None) -> bool``: advance ``car.waypoint_index`` to the nearest
  circuit waypoint ahead, return True on a start-line crossing (lap done).
  Anti-double-count: require the car to reach the midpoint of the circuit
  (``len(circuit) // 2``) before the next crossing can register.
- ``on_lap_complete(car, now_s, race_started_at_s, *, num_laps) -> LapSplit``:
  record the split, bump ``car.laps_finished`` and ``car.lap``, set
  ``car.finish_position`` + ``car.finished_at_s`` if the car has completed
  ``num_laps``. Returns the recorded ``LapSplit``.

Distance is computed in 2D (x, y) by default; a custom ``distance_fn`` can
be injected (used by tests to avoid building real ``carla.Location`` math).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol

from carla_race.race_state import CarState, LapSplit

__all__ = ["on_lap_complete", "update_car_progress"]

DEFAULT_CROSSING_THRESHOLD = 8.0


class _Loc(Protocol):
    x: float
    y: float
    z: float


class _Tf(Protocol):
    location: _Loc


def _default_distance(a: _Tf, b: _Tf) -> float:
    dx = float(a.location.x) - float(b.location.x)
    dy = float(a.location.y) - float(b.location.y)
    return math.hypot(dx, dy)


def _nearest_ahead_index(
    car_index: int,
    car_tf: Any,
    circuit: list[Any],
    distance_fn: Callable[[Any, Any], float],
) -> int:
    """Index of the closest circuit waypoint to ``car_tf`` within a small
    look-ahead window of ``car_index``. Used to advance ``waypoint_index``."""
    n = len(circuit)
    if n == 0:
        return car_index
    best_idx = car_index
    best_d = distance_fn(car_tf, circuit[car_index])
    # look ahead up to 8 waypoints (and a little behind for noise)
    for offset in range(-2, 9):
        idx = (car_index + offset) % n
        d = distance_fn(car_tf, circuit[idx])
        if d < best_d:
            best_d = d
            best_idx = idx
    return best_idx


def update_car_progress(
    car: CarState,
    transform: Any,
    circuit: list[Any],
    *,
    crossing_threshold: float = DEFAULT_CROSSING_THRESHOLD,
    distance_fn: Callable[[Any, Any], float] | None = None,
) -> bool:
    """Advance ``car.waypoint_index``; return True if the car crossed the
    start line (i.e. wrapped from near the end of the circuit back to index 0).

    Anti-double-count: a crossing only registers if the car has previously
    reached the circuit midpoint (``len(circuit) // 2``). This stops a car
    hovering near the start line from racking up laps.
    """
    if not circuit:
        return False
    dfn = distance_fn if distance_fn is not None else _default_distance
    n = len(circuit)

    new_index = _nearest_ahead_index(car.waypoint_index, transform, circuit, dfn)

    midpoint = n // 2
    reached_midpoint = car.waypoint_index >= midpoint

    crossed = False
    if reached_midpoint and new_index == 0:
        # Verify the car is actually near the start line (within threshold).
        d0 = dfn(transform, circuit[0])
        if d0 <= crossing_threshold:
            crossed = True
            car.waypoint_index = 0
        else:
            car.waypoint_index = new_index
    else:
        car.waypoint_index = new_index

    return crossed


def on_lap_complete(
    car: CarState,
    now_s: float,
    race_started_at_s: float,
    *,
    num_laps: int,
) -> LapSplit:
    """Record a lap split. Bumps ``car.laps_finished`` and ``car.lap``.

    If the car has completed ``num_laps``, marks it finished:
    ``car.finished_at_s = now_s``. ``finish_position`` is NOT set here —
    the race manager assigns positions (F-mgr contract) via a separate
    finish counter.
    """
    cumulative = now_s - race_started_at_s if race_started_at_s is not None else 0.0
    prev_cumulative = (
        car.splits[-1].cumulative_time_s if car.splits else 0.0
    )
    lap_time = cumulative - prev_cumulative
    lap_number = car.laps_finished + 1
    split = LapSplit(
        lap_number=lap_number,
        lap_time_s=lap_time,
        cumulative_time_s=cumulative,
    )
    car.splits.append(split)
    car.laps_finished = lap_number

    if car.laps_finished >= num_laps:
        car.finished_at_s = now_s
        car.lap = num_laps
    else:
        car.lap = car.laps_finished + 1

    return split

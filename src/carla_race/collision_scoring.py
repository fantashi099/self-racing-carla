"""F7 — per-car collision sensors + hit classification.

Attaches a ``sensor.other.collision`` to each car actor and wires a
callback that mutates the matching ``CarState`` under a ``threading.Lock``
(CARLA fires sensor callbacks on a background thread). Score is NOT stored
on ``CarState`` — it's recomputed lazily by ``scoring.live_score`` from the
hit counters.

Contract:
- ``classify_hit(type_id) -> str`` — pure: ``walker.*`` → ``"walker"``,
  ``vehicle.*`` → ``"vehicle"``, else ``"other"``.
- ``attach_collision_sensors(world, car_actors, car_states, lock=None)
  -> dict[int, int]`` — spawn one collision sensor per car, register a
  callback that increments ``car_state.walker_hits`` or
  ``car_state.car_hits`` based on ``classify_hit(event.other_actor.type_id)``
  under ``lock``. Returns ``{actor_id: sensor_id}``.
- ``destroy_sensors(world, sensor_ids) -> None`` — best-effort destroy each
  sensor actor.

No ``carla`` import — the world and actors are opaque objects passed in by
race_manager. The collision event is read via ``event.other_actor.type_id``
(structurally compatible with ``carla.CollisionEvent``).
"""
from __future__ import annotations

import threading
from typing import Any

from carla_race.race_state import CarState

__all__ = ["attach_collision_sensors", "classify_hit", "destroy_sensors"]

COLLISION_SENSOR_BP = "sensor.other.collision"


def classify_hit(type_id: str) -> str:
    if type_id.startswith("walker"):
        return "walker"
    if type_id.startswith("vehicle"):
        return "vehicle"
    return "other"


def _identity_transform() -> Any:
    try:
        import carla
    except ImportError:
        return _IdentityTransform()
    return carla.Transform()


class _IdentityTransform:
    def __init__(self) -> None:
        self.location = _ZeroLoc()
        self.rotation = None


class _ZeroLoc:
    x = 0.0
    y = 0.0
    z = 0.0


def attach_collision_sensors(
    world: Any,
    car_actors: dict[int, Any],
    car_states: dict[int, CarState],
    lock: threading.Lock | None = None,
) -> dict[int, int]:
    if lock is None:
        lock = threading.Lock()
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(COLLISION_SENSOR_BP)
    tf = _identity_transform()

    sensor_ids: dict[int, int] = {}
    for actor_id, actor in car_actors.items():
        car_state = car_states[actor_id]
        sensor = world.spawn_actor(bp, tf, attach_to=actor)
        sensor_ids[actor_id] = sensor.id

        def make_cb(state: CarState, lk: threading.Lock) -> Any:
            def cb(event: Any) -> None:
                other = getattr(event, "other_actor", None)
                type_id = getattr(other, "type_id", "") if other is not None else ""
                hit = classify_hit(type_id)
                with lk:
                    if hit == "walker":
                        state.walker_hits += 1
                    elif hit == "vehicle":
                        state.car_hits += 1
            return cb

        sensor.listen(make_cb(car_state, lock))
    return sensor_ids


def destroy_sensors(world: Any, sensor_ids: dict[int, int]) -> None:
    for _actor_id, sid in sensor_ids.items():
        sensor = world.get_actor(sid)
        if sensor is not None:
            sensor.destroy()

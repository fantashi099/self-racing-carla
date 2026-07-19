"""F4 — configure AI cars via TrafficManager.

Applies per-difficulty presets from ``config.AI_DIFFICULTY_PRESETS`` to each
AI car (skipping the player) and hands them the circuit path. No CARLA
import at runtime — the TrafficManager, world map, and actor objects are
passed in by the race manager.

**Signature deviates from PROGRESS.md § Module contracts** (verified at L2
against a real CARLA 0.9.15 build): the real ``TrafficManager.set_path`` takes
an ``Actor`` object (not an actor id) and a list of ``carla.Waypoint`` (not
``carla.Transform``). So ``setup_ai_cars`` now takes ``car_actors`` (id →
Actor dict) + ``carla_map`` (for Transform→Waypoint conversion) instead of
just actor ids + circuit Transforms. See DECISION.md F4 L2 for the rationale.

CARLA TrafficManager method names vary across versions. Preset keys are
applied as ``tm.set_<key>(actor, value)`` defensively: missing methods are
skipped with a one-time stderr warning listing the TM's actual ``set_``
methods, so the L2 smoke passes regardless of the CARLA build.

Contract:
- ``setup_ai_cars(tm, car_actors, carla_map, circuit, *, difficulty,
  player_actor_id) -> None``: ``tm.set_hybrid_physics_mode(True)`` once
  globally, convert ``circuit`` (list[Transform]) to Waypoints via
  ``carla_map.get_waypoint(tf.location)`` and close the loop (append
  ``waypoints[0]`` so the AI crosses the start line), then for every AI
  actor apply each preset key as ``tm.set_<key>(actor, value)`` (defensive)
  and finally ``tm.set_path(actor, waypoints)``. Player is skipped. Raises
  ``ValueError`` if ``difficulty`` is unknown.
- ``reset_ai_cars(tm, car_actor_ids) -> None``: clear each AI car's path
  (``tm.set_path(actor, [])``) so AI control stops before actor destroy.
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any

from carla_race.config import AI_DIFFICULTY_PRESETS

__all__ = ["reset_ai_cars", "setup_ai_cars"]

_warned_missing: set[str] = set()


def _apply_preset_key(tm: Any, key: str, actor: Any, value: Any) -> None:
    method_name = f"set_{key}"
    if key == "percentage_speed_difference":
        method_name = "vehicle_percentage_speed_difference"
    method = getattr(tm, method_name, None)
    if method is None and key == "percentage_speed_difference":
        method_name = "set_percentage_speed_difference"
        method = getattr(tm, method_name, None)
    if method is None:
        if key not in _warned_missing:
            _warned_missing.add(key)
            available = [
                m
                for m in dir(tm)
                if m.startswith("set_") or "speed_difference" in m
            ]
            print(
                f"[ai_driver] TM has no method for {key}; skipping. "
                f"Available matching methods: {available}",
                file=sys.stderr,
            )
        return
    with contextlib.suppress(Exception):
        method(actor, value)


def _circuit_to_path(circuit: list[Any]) -> list[Any]:
    """Build a TM path (list of Locations) from the circuit. Handles two
    circuit shapes:
    - list[Transform]: extract ``tf.location`` (a ``carla.Location``).
    - list[Waypoint]: extract ``wp.transform.location`` (fallback) or
      ``wp.location`` if present.
    Closes the loop by appending the first location so the AI crosses the
    start line and completes a lap. Returns ``[]`` if no locations extracted.

    Verified at L2 against CARLA 0.9.15: ``TrafficManager.set_path`` takes a
    list of ``carla.Location`` (NOT Waypoints) — passing Waypoints raises
    ``TypeError: No registered converter ... Location from ... Waypoint``.
    """
    locations: list[Any] = []
    for tf in circuit:
        loc = getattr(tf, "location", None)
        if loc is None:
            # Waypoint fallback: try .transform.location
            transform = getattr(tf, "transform", None)
            if transform is not None:
                loc = getattr(transform, "location", None)
        if loc is not None:
            locations.append(loc)
    if locations:
        locations.append(locations[0])  # close the loop
    return locations


def setup_ai_cars(
    tm: Any,
    car_actors: dict[int, Any],
    carla_map: Any,
    circuit: list[Any],
    *,
    difficulty: str,
    player_actor_id: int,
) -> None:
    if difficulty not in AI_DIFFICULTY_PRESETS:
        raise ValueError(
            f"unknown difficulty {difficulty!r}; "
            f"expected one of {sorted(AI_DIFFICULTY_PRESETS)}"
        )
    preset = AI_DIFFICULTY_PRESETS[difficulty]
    with contextlib.suppress(Exception):
        tm.set_hybrid_physics_mode(True)
    path = _circuit_to_path(circuit)
    for actor_id, actor in car_actors.items():
        if actor_id == player_actor_id:
            continue
        for key, value in preset.items():
            _apply_preset_key(tm, key, actor, value)
        if path:
            try:
                tm.set_path(actor, path)
            except Exception as e:
                print(
                    f"[ai_driver] set_path failed for actor {actor_id}: {e!r}",
                    file=sys.stderr,
                )


def reset_ai_cars(tm: Any, car_actors: dict[int, Any]) -> None:
    for _actor_id, actor in car_actors.items():
        with contextlib.suppress(Exception):
            tm.set_path(actor, [])

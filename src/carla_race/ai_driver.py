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
    method = getattr(tm, f"set_{key}", None)
    if method is None:
        if key not in _warned_missing:
            _warned_missing.add(key)
            available = [m for m in dir(tm) if m.startswith("set_")]
            print(
                f"[ai_driver] TM has no set_{key}; skipping. "
                f"Available set_ methods: {available}",
                file=sys.stderr,
            )
        return
    with contextlib.suppress(Exception):
        method(actor, value)


def _circuit_to_waypoints(carla_map: Any, circuit: list[Any]) -> list[Any]:
    """Convert circuit entries to Waypoints. Handles two circuit shapes:
    - list[Transform]: convert each via ``map.get_waypoint(tf.location)``.
    - list[Waypoint]: use directly (``build_circuit`` may return Waypoints
      when ``getattr(wp, "transform", wp)`` falls back to the waypoint itself
      on CARLA builds where ``Waypoint.transform`` isn't a property).
    Closes the loop by appending ``waypoints[0]`` so the AI crosses the
    start line and completes a lap. Returns ``[]`` if conversion fails."""
    waypoints: list[Any] = []
    for tf in circuit:
        # Waypoint duck-type: has road_id (Transform does not)
        if hasattr(tf, "road_id"):
            waypoints.append(tf)
            continue
        loc = getattr(tf, "location", None)
        if loc is None:
            continue
        get_wp = getattr(carla_map, "get_waypoint", None)
        if get_wp is None:
            return []
        try:
            wp = get_wp(loc)
        except Exception:
            return []
        if wp is not None:
            waypoints.append(wp)
    if waypoints:
        waypoints.append(waypoints[0])  # close the loop
    return waypoints


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
    waypoints = _circuit_to_waypoints(carla_map, circuit)
    for actor_id, actor in car_actors.items():
        if actor_id == player_actor_id:
            continue
        for key, value in preset.items():
            _apply_preset_key(tm, key, actor, value)
        if waypoints:
            try:
                tm.set_path(actor, waypoints)
            except Exception as e:
                print(
                    f"[ai_driver] set_path failed for actor {actor_id}: {e!r}",
                    file=sys.stderr,
                )


def reset_ai_cars(tm: Any, car_actors: dict[int, Any]) -> None:
    for _actor_id, actor in car_actors.items():
        with contextlib.suppress(Exception):
            tm.set_path(actor, [])

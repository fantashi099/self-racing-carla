"""F4 — configure AI cars via TrafficManager.

Applies per-difficulty presets from ``config.AI_DIFFICULTY_PRESETS`` to each
AI car (skipping the player) and hands them the circuit path. No CARLA
import at runtime — the TrafficManager object is passed in by the race
manager, so this module treats it as an opaque target with ``set_<key>``
methods matching the preset keys.

Contract:
- ``setup_ai_cars(tm, car_actor_ids, circuit, *, difficulty, player_actor_id)
  -> None``: ``tm.set_hybrid_physics_mode(True)`` once globally, then for
  every AI actor id apply each preset key as ``tm.set_<key>(actor_id,
  value)`` and finally ``tm.set_path(actor_id, circuit)``. Player is
  skipped. Raises ``ValueError`` if ``difficulty`` is unknown.
- ``reset_ai_cars(tm, car_actor_ids) -> None``: clear each AI car's path
  (``tm.set_path(actor_id, [])``) so AI control stops before actor destroy.

Preset key → TrafficManager method name convention: a preset key
``foo_bar`` maps to ``tm.set_foo_bar(actor_id, value)``. Keys defined in
``AI_DIFFICULTY_PRESETS`` are ``percentage_speed_difference``,
``global_distance_to_leading_vehicle``, ``auto_lane_change``, ``safety_mode``.
"""
from __future__ import annotations

from typing import Any

from carla_race.config import AI_DIFFICULTY_PRESETS

__all__ = ["reset_ai_cars", "setup_ai_cars"]


def setup_ai_cars(
    tm: Any,
    car_actor_ids: list[int],
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
    tm.set_hybrid_physics_mode(True)
    for actor_id in car_actor_ids:
        if actor_id == player_actor_id:
            continue
        for key, value in preset.items():
            getattr(tm, f"set_{key}")(actor_id, value)
        tm.set_path(actor_id, circuit)


def reset_ai_cars(tm: Any, car_actor_ids: list[int]) -> None:
    for actor_id in car_actor_ids:
        tm.set_path(actor_id, [])

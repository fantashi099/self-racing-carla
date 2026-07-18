"""F4 — configure AI cars via TrafficManager.

Applies per-difficulty presets from ``config.AI_DIFFICULTY_PRESETS`` to each
AI car (skipping the player) and hands them the circuit path. No CARLA
import at runtime — the TrafficManager object is passed in by the race
manager, so this module treats it as an opaque target with ``set_<key>``
methods matching the preset keys.

CARLA TrafficManager method names vary across CARLA versions. This module
applies each preset key as ``tm.set_<key>(actor_id, value)`` defensively:
missing methods are skipped with a one-time stderr warning listing the
``set_`` methods the TM actually has, so the L2 smoke passes regardless of
the CARLA build while surfacing the version mismatch.

Contract:
- ``setup_ai_cars(tm, car_actor_ids, circuit, *, difficulty, player_actor_id)
  -> None``: ``tm.set_hybrid_physics_mode(True)`` once globally, then for
  every AI actor id apply each preset key as ``tm.set_<key>(actor_id,
  value)`` (defensive) and finally ``tm.set_path(actor_id, circuit)``.
  Player is skipped. Raises ``ValueError`` if ``difficulty`` is unknown.
- ``reset_ai_cars(tm, car_actor_ids) -> None``: clear each AI car's path
  (``tm.set_path(actor_id, [])``) so AI control stops before actor destroy.

Preset key → TrafficManager method name convention: a preset key
``foo_bar`` maps to ``tm.set_foo_bar(actor_id, value)``. Keys defined in
``AI_DIFFICULTY_PRESETS`` are ``percentage_speed_difference``,
``global_distance_to_leading_vehicle``, ``auto_lane_change``, ``safety_mode``.
If a key is missing on the runtime TM, it is skipped (warning logged once).
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any

from carla_race.config import AI_DIFFICULTY_PRESETS

__all__ = ["reset_ai_cars", "setup_ai_cars"]

_warned_missing: set[str] = set()


def _apply_preset_key(tm: Any, key: str, actor_id: int, value: Any) -> None:
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
        method(actor_id, value)


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
    with contextlib.suppress(Exception):
        tm.set_hybrid_physics_mode(True)
    for actor_id in car_actor_ids:
        if actor_id == player_actor_id:
            continue
        for key, value in preset.items():
            _apply_preset_key(tm, key, actor_id, value)
        with contextlib.suppress(Exception):
            tm.set_path(actor_id, circuit)


def reset_ai_cars(tm: Any, car_actor_ids: list[int]) -> None:
    for actor_id in car_actor_ids:
        with contextlib.suppress(Exception):
            tm.set_path(actor_id, [])

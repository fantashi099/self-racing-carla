"""Race configuration — env → ``RaceConfig``.

Loads race parameters from environment (with sensible defaults) and validates
them. Frozen dataclass; mutation after construction is blocked.

Env keys:
- ``RACE_NUM_CARS`` (default 10) — total cars including the player. >=2.
- ``RACE_NUM_LAPS`` (default 3) — laps to finish. >=1.
- ``RACE_NUM_WALKERS`` (default 20) — pedestrians to spawn. >=0.
- ``RACE_AI_DIFFICULTY`` (default "normal") — one of easy/normal/hard.

``AI_DIFFICULTY_PRESETS`` maps each difficulty to TrafficManager parameters
consumed by ``ai_driver.setup_ai_cars`` (F4). Values are JSON-serializable
(float/bool/str) so the preset can be surfaced in `/race/state` without
extra adaptation.

Contract:
- ``load_config() -> RaceConfig`` — read env, validate, return frozen config.
- ``RaceConfig.num_ai_cars`` property — ``num_cars - 1`` (player excluded).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

__all__ = ["AI_DIFFICULTY_PRESETS", "DIFFICULTIES", "RaceConfig", "load_config"]

DEFAULT_NUM_CARS = 10
DEFAULT_NUM_LAPS = 3
DEFAULT_NUM_WALKERS = 20
DEFAULT_AI_DIFFICULTY = "normal"

DIFFICULTIES: tuple[str, ...] = ("easy", "normal", "hard")

# TrafficManager parameters per difficulty. Keys mirror the CARLA
# TrafficManager API consumed by ai_driver.setup_ai_cars (F4):
# - percentage_speed_difference: signed delta vs speed limit (+ = slower,
#   - = faster). 0 = drive the limit.
# - global_distance_to_leading_vehicle: meters of following gap.
# - auto_lane_change: allow TM to change lanes to overtake.
# - safety_mode: True = collision-avoidance on; False = aggressive (hard).
AI_DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        "percentage_speed_difference": 20.0,
        "global_distance_to_leading_vehicle": 5.0,
        "auto_lane_change": False,
        "safety_mode": True,
    },
    "normal": {
        "percentage_speed_difference": 0.0,
        "global_distance_to_leading_vehicle": 3.0,
        "auto_lane_change": True,
        "safety_mode": True,
    },
    "hard": {
        "percentage_speed_difference": -20.0,
        "global_distance_to_leading_vehicle": 2.0,
        "auto_lane_change": True,
        "safety_mode": False,
    },
}


@dataclass(frozen=True)
class RaceConfig:
    num_cars: int
    num_laps: int
    num_walkers: int
    ai_difficulty: str

    @property
    def num_ai_cars(self) -> int:
        return self.num_cars - 1


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} is not an integer"
        ) from exc


def load_config() -> RaceConfig:
    """Read race env vars, validate, return a frozen ``RaceConfig``.

    Raises ``ValueError`` on out-of-range or wrong-type values.
    """
    num_cars = _get_int_env("RACE_NUM_CARS", DEFAULT_NUM_CARS)
    num_laps = _get_int_env("RACE_NUM_LAPS", DEFAULT_NUM_LAPS)
    num_walkers = _get_int_env("RACE_NUM_WALKERS", DEFAULT_NUM_WALKERS)
    ai_difficulty = os.environ.get("RACE_AI_DIFFICULTY", DEFAULT_AI_DIFFICULTY)

    if num_cars < 2:
        raise ValueError(f"RACE_NUM_CARS must be >= 2, got {num_cars}")
    if num_laps < 1:
        raise ValueError(f"RACE_NUM_LAPS must be >= 1, got {num_laps}")
    if num_walkers < 0:
        raise ValueError(f"RACE_NUM_WALKERS must be >= 0, got {num_walkers}")
    if ai_difficulty not in DIFFICULTIES:
        raise ValueError(
            f"RACE_AI_DIFFICULTY={ai_difficulty!r} not in {DIFFICULTIES}"
        )

    return RaceConfig(
        num_cars=num_cars,
        num_laps=num_laps,
        num_walkers=num_walkers,
        ai_difficulty=ai_difficulty,
    )

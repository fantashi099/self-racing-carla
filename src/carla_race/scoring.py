"""F8 — pure race score formula.

No CARLA, no I/O, no mutation. Reads progress fields from ``CarState`` and
returns signed integer scores plus a one-line HUD string.

Scoring model:
- ``live_score`` (during race): reward laps completed, penalize elapsed time
  and hits. ``= laps_finished * BASE_PER_LAP_REMAINING - elapsed_s *
  TIME_PENALTY_PER_SEC - walker_hits * WALKER_HIT_PENALTY - car_hits *
  CAR_HIT_PENALTY``. Clamped to >= 0.
- ``finish_bonus`` (at finish): ``= (num_cars - finish_position) *
  FINISH_BONUS_PER_POSITION``. 1st of 10 → 4500; last → 0; DNF
  (``finish_position = num_cars``) → 0.
- ``final_score``: ``live_score`` evaluated at the car's finish time (using
  ``elapsed_s`` passed in by the caller — typically ``car.finished_at_s -
  race.started_at_s``) plus ``finish_bonus``. Clamped to >= 0.

Constants:
- ``WALKER_HIT_PENALTY = 100``
- ``CAR_HIT_PENALTY = 50``
- ``BASE_PER_LAP_REMAINING = 1000``
- ``TIME_PENALTY_PER_SEC = 2.0``
- ``FINISH_BONUS_PER_POSITION = 500``

Contract:
- ``live_score(*, laps_total, laps_finished, elapsed_s, walker_hits,
  car_hits) -> int``
- ``finish_bonus(*, num_cars, finish_position) -> int``
- ``final_score(*, laps_total, num_cars, car, elapsed_s) -> int``
- ``car_score_str(*, laps_total, num_cars, car, elapsed_s) -> str``
"""
from __future__ import annotations

from carla_race.race_state import CarState

__all__ = [
    "BASE_PER_LAP_REMAINING",
    "CAR_HIT_PENALTY",
    "FINISH_BONUS_PER_POSITION",
    "TIME_PENALTY_PER_SEC",
    "WALKER_HIT_PENALTY",
    "car_score_str",
    "final_score",
    "finish_bonus",
    "live_score",
]

WALKER_HIT_PENALTY = 100
CAR_HIT_PENALTY = 50
BASE_PER_LAP_REMAINING = 1000
TIME_PENALTY_PER_SEC = 2.0
FINISH_BONUS_PER_POSITION = 500


def live_score(
    *,
    laps_total: int,
    laps_finished: int,
    elapsed_s: float,
    walker_hits: int,
    car_hits: int,
) -> int:
    """During-race score: laps completed minus time and hit penalties.

    ``laps_total`` is accepted for symmetry with ``final_score`` and to
    support future lap-progress bonus shapes; the current formula rewards
    ``laps_finished`` directly. Result clamped to >= 0 so a slow + hit-heavy
    race never shows a negative score on the HUD.
    """
    _ = laps_total
    raw = (
        laps_finished * BASE_PER_LAP_REMAINING
        - int(elapsed_s * TIME_PENALTY_PER_SEC)
        - walker_hits * WALKER_HIT_PENALTY
        - car_hits * CAR_HIT_PENALTY
    )
    return max(0, raw)


def finish_bonus(*, num_cars: int, finish_position: int) -> int:
    """Position bonus: 1st place gets ``(num_cars - 1) * 500``, last gets 0.

    DNF cars carry ``finish_position = num_cars`` (per the race_manager
    contract), so their bonus is 0 by construction. ``finish_position`` must
    be in ``[1, num_cars]``; out-of-range raises ``ValueError``.
    """
    if not 1 <= finish_position <= num_cars:
        raise ValueError(
            f"finish_position={finish_position} not in [1, {num_cars}]"
        )
    return (num_cars - finish_position) * FINISH_BONUS_PER_POSITION


def final_score(
    *,
    laps_total: int,
    num_cars: int,
    car: CarState,
    elapsed_s: float,
) -> int:
    """End-of-race score for a finished (or DNF) car.

    Uses ``car.finish_position`` for the bonus and ``car.laps_finished`` for
    the lap base. Raises ``ValueError`` if ``car.finish_position`` is None
    (car still racing — call ``live_score`` instead).
    """
    if car.finish_position is None:
        raise ValueError(
            f"car {car.actor_id} has no finish_position; use live_score instead"
        )
    base = live_score(
        laps_total=laps_total,
        laps_finished=car.laps_finished,
        elapsed_s=elapsed_s,
        walker_hits=car.walker_hits,
        car_hits=car.car_hits,
    )
    bonus = finish_bonus(num_cars=num_cars, finish_position=car.finish_position)
    return max(0, base + bonus)


def car_score_str(
    *,
    laps_total: int,
    num_cars: int,
    car: CarState,
    elapsed_s: float,
) -> str:
    """One-line HUD string for a car.

    During race: ``"L 2/3 | 45.2s | W:1 C:0 | Score: 1450"``
    Finished: appends ``" | Pos: 1/10"``
    """
    score = (
        final_score(
            laps_total=laps_total,
            num_cars=num_cars,
            car=car,
            elapsed_s=elapsed_s,
        )
        if car.finish_position is not None
        else live_score(
            laps_total=laps_total,
            laps_finished=car.laps_finished,
            elapsed_s=elapsed_s,
            walker_hits=car.walker_hits,
            car_hits=car.car_hits,
        )
    )
    base = (
        f"L {car.laps_finished}/{laps_total} | {elapsed_s:.1f}s | "
        f"W:{car.walker_hits} C:{car.car_hits} | Score: {score}"
    )
    if car.finish_position is not None:
        return f"{base} | Pos: {car.finish_position}/{num_cars}"
    return base

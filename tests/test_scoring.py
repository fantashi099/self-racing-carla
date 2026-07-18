"""Unit tests for carla_race.scoring (F8)."""
from __future__ import annotations

import pytest

from carla_race.race_state import CarState
from carla_race.scoring import (
    BASE_PER_LAP_REMAINING,
    CAR_HIT_PENALTY,
    FINISH_BONUS_PER_POSITION,
    TIME_PENALTY_PER_SEC,
    WALKER_HIT_PENALTY,
    car_score_str,
    finish_bonus,
    final_score,
    live_score,
)


def test_constants_match_contract() -> None:
    assert WALKER_HIT_PENALTY == 100
    assert CAR_HIT_PENALTY == 50
    assert BASE_PER_LAP_REMAINING == 1000
    assert TIME_PENALTY_PER_SEC == 2.0
    assert FINISH_BONUS_PER_POSITION == 500


def test_live_score_clean_start_zero_laps() -> None:
    assert (
        live_score(
            laps_total=3,
            laps_finished=0,
            elapsed_s=0.0,
            walker_hits=0,
            car_hits=0,
        )
        == 0
    )


def test_live_score_one_lap_clean() -> None:
    # 1 * 1000 - 60 * 2 = 1000 - 120 = 880
    assert (
        live_score(
            laps_total=3,
            laps_finished=1,
            elapsed_s=60.0,
            walker_hits=0,
            car_hits=0,
        )
        == 880
    )


def test_live_score_three_laps_clean() -> None:
    # 3 * 1000 - 120 * 2 = 3000 - 240 = 2760
    assert (
        live_score(
            laps_total=3,
            laps_finished=3,
            elapsed_s=120.0,
            walker_hits=0,
            car_hits=0,
        )
        == 2760
    )


def test_live_score_walker_hit_subtracts_100() -> None:
    base = live_score(laps_total=3, laps_finished=1, elapsed_s=0.0, walker_hits=0, car_hits=0)
    hit = live_score(laps_total=3, laps_finished=1, elapsed_s=0.0, walker_hits=1, car_hits=0)
    assert base - hit == WALKER_HIT_PENALTY


def test_live_score_car_hit_subtracts_50() -> None:
    base = live_score(laps_total=3, laps_finished=1, elapsed_s=0.0, walker_hits=0, car_hits=0)
    hit = live_score(laps_total=3, laps_finished=1, elapsed_s=0.0, walker_hits=0, car_hits=1)
    assert base - hit == CAR_HIT_PENALTY


def test_live_score_multiple_walker_hits_stack() -> None:
    s = live_score(laps_total=3, laps_finished=1, elapsed_s=0.0, walker_hits=3, car_hits=0)
    # 1000 - 0 - 300 = 700
    assert s == 700


def test_live_score_clamps_to_zero() -> None:
    # 0 laps - huge time - hits → would be very negative
    s = live_score(laps_total=3, laps_finished=0, elapsed_s=10000.0, walker_hits=50, car_hits=50)
    assert s == 0


def test_live_score_returns_int() -> None:
    s = live_score(laps_total=3, laps_finished=1, elapsed_s=45.7, walker_hits=0, car_hits=0)
    assert isinstance(s, int)


def test_live_score_truncates_fractional_time_penalty() -> None:
    # 45.7 * 2.0 = 91.4 → int(91.4) = 91; 1000 - 91 = 909
    s = live_score(laps_total=3, laps_finished=1, elapsed_s=45.7, walker_hits=0, car_hits=0)
    assert s == 909


def test_finish_bonus_first_place_max() -> None:
    assert finish_bonus(num_cars=10, finish_position=1) == 9 * FINISH_BONUS_PER_POSITION


def test_finish_bonus_last_place_zero() -> None:
    assert finish_bonus(num_cars=10, finish_position=10) == 0


def test_finish_bonus_dnf_zero() -> None:
    # DNF: finish_position = num_cars → bonus 0
    assert finish_bonus(num_cars=10, finish_position=10) == 0


def test_finish_bonus_second_place() -> None:
    assert finish_bonus(num_cars=10, finish_position=2) == 8 * FINISH_BONUS_PER_POSITION


def test_finish_bonus_two_car_race() -> None:
    assert finish_bonus(num_cars=2, finish_position=1) == 1 * FINISH_BONUS_PER_POSITION
    assert finish_bonus(num_cars=2, finish_position=2) == 0


def test_finish_bonus_rejects_zero_position() -> None:
    with pytest.raises(ValueError, match="finish_position=0"):
        finish_bonus(num_cars=10, finish_position=0)


def test_finish_bonus_rejects_over_num_cars() -> None:
    with pytest.raises(ValueError, match="finish_position=11"):
        finish_bonus(num_cars=10, finish_position=11)


def test_finish_bonus_rejects_negative() -> None:
    with pytest.raises(ValueError, match="finish_position=-1"):
        finish_bonus(num_cars=10, finish_position=-1)


def test_final_score_clean_first_place() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    car.laps_finished = 3
    car.finish_position = 1
    # base = 3*1000 - 120*2 = 2760; bonus = 9*500 = 4500; total = 7260
    assert final_score(laps_total=3, num_cars=10, car=car, elapsed_s=120.0) == 7260


def test_final_score_last_place_no_bonus() -> None:
    car = CarState(actor_id=2, is_player=False, color="b")
    car.laps_finished = 3
    car.finish_position = 10  # last of 10
    # base = 2760; bonus = 0
    assert final_score(laps_total=3, num_cars=10, car=car, elapsed_s=120.0) == 2760


def test_final_score_with_hits() -> None:
    car = CarState(actor_id=3, is_player=False, color="g")
    car.laps_finished = 3
    car.finish_position = 2
    car.walker_hits = 2
    car.car_hits = 1
    # base = 3000 - 240 - 200 - 50 = 2510; bonus = 8*500 = 4000; total = 6510
    assert final_score(laps_total=3, num_cars=10, car=car, elapsed_s=120.0) == 6510


def test_final_score_dnf_zero_bonus() -> None:
    car = CarState(actor_id=4, is_player=False, color="y")
    car.laps_finished = 1
    car.finish_position = 10  # num_cars → DNF
    car.dnf = True
    # base = 1*1000 - 60*2 = 880; bonus = 0
    assert final_score(laps_total=3, num_cars=10, car=car, elapsed_s=60.0) == 880


def test_final_score_rejects_racing_car() -> None:
    car = CarState(actor_id=5, is_player=True, color="r")
    car.laps_finished = 1
    # finish_position is None
    with pytest.raises(ValueError, match="use live_score instead"):
        final_score(laps_total=3, num_cars=10, car=car, elapsed_s=60.0)


def test_final_score_clamps_to_zero() -> None:
    car = CarState(actor_id=6, is_player=False, color="b")
    car.laps_finished = 0
    car.finish_position = 10
    car.walker_hits = 50
    # base would be negative → clamped to 0; bonus = 0; total = 0
    assert final_score(laps_total=3, num_cars=10, car=car, elapsed_s=10000.0) == 0


def test_car_score_str_during_race() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    car.laps_finished = 2
    car.walker_hits = 1
    car.car_hits = 0
    s = car_score_str(laps_total=3, num_cars=10, car=car, elapsed_s=45.7)
    # live: 2*1000 - int(45.7*2) - 100 - 0 = 2000 - 91 - 100 = 1809
    assert "L 2/3" in s
    assert "45.7s" in s
    assert "W:1 C:0" in s
    assert "Score: 1809" in s
    assert "Pos" not in s


def test_car_score_str_finished_includes_position() -> None:
    car = CarState(actor_id=1, is_player=True, color="r")
    car.laps_finished = 3
    car.finish_position = 1
    car.walker_hits = 0
    car.car_hits = 0
    s = car_score_str(laps_total=3, num_cars=10, car=car, elapsed_s=120.0)
    assert "L 3/3" in s
    assert "120.0s" in s
    assert "W:0 C:0" in s
    assert "Pos: 1/10" in s
    # final: 2760 + 4500 = 7260
    assert "Score: 7260" in s


def test_car_score_str_dnf_shows_position() -> None:
    car = CarState(actor_id=2, is_player=False, color="b")
    car.laps_finished = 1
    car.finish_position = 10
    car.dnf = True
    s = car_score_str(laps_total=3, num_cars=10, car=car, elapsed_s=60.0)
    assert "Pos: 10/10" in s


def test_car_score_str_last_place_shows_zero_bonus() -> None:
    car = CarState(actor_id=3, is_player=False, color="g")
    car.laps_finished = 3
    car.finish_position = 10  # last
    s = car_score_str(laps_total=3, num_cars=10, car=car, elapsed_s=120.0)
    # base = 2760, bonus = 0
    assert "Score: 2760" in s
    assert "Pos: 10/10" in s

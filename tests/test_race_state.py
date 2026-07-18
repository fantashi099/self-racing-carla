"""Unit tests for carla_race.race_state."""
from __future__ import annotations

import dataclasses

import pytest

from carla_race.race_state import (
    CarState,
    LapSplit,
    RacePhase,
    RaceState,
)


def test_race_phase_values() -> None:
    assert RacePhase.INIT == "init"
    assert RacePhase.RUNNING == "running"
    assert RacePhase.FINISHED == "finished"
    assert {p.value for p in RacePhase} == {"init", "running", "finished"}


def test_race_phase_is_str_enum() -> None:
    assert isinstance(RacePhase.INIT, str)
    assert RacePhase.INIT == "init"


def test_lap_split_is_frozen() -> None:
    s = LapSplit(lap_number=1, lap_time_s=12.5, cumulative_time_s=12.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.lap_number = 99  # type: ignore[misc]


def test_car_state_defaults() -> None:
    c = CarState(actor_id=10, is_player=True, color="255,0,0")
    assert c.actor_id == 10
    assert c.is_player is True
    assert c.color == "255,0,0"
    assert c.lap == 1
    assert c.waypoint_index == 0
    assert c.laps_finished == 0
    assert c.splits == []
    assert c.walker_hits == 0
    assert c.car_hits == 0
    assert c.finish_position is None
    assert c.finished_at_s is None
    assert c.dnf is False


def test_car_state_is_mutable() -> None:
    c = CarState(actor_id=1, is_player=False, color="0,0,255")
    c.lap = 2
    c.waypoint_index = 17
    c.laps_finished = 1
    c.splits.append(LapSplit(1, 45.0, 45.0))
    c.walker_hits = 3
    c.finish_position = 2
    c.finished_at_s = 120.0
    c.dnf = True
    assert c.lap == 2
    assert c.waypoint_index == 17
    assert c.laps_finished == 1
    assert len(c.splits) == 1
    assert c.walker_hits == 3
    assert c.finish_position == 2
    assert c.finished_at_s == 120.0
    assert c.dnf is True


def test_car_state_splits_independent_per_instance() -> None:
    a = CarState(actor_id=1, is_player=True, color="r")
    b = CarState(actor_id=2, is_player=False, color="b")
    a.splits.append(LapSplit(1, 10.0, 10.0))
    assert b.splits == []


def test_race_state_defaults() -> None:
    rs = RaceState(config_num_cars=10, config_num_laps=3)
    assert rs.config_num_cars == 10
    assert rs.config_num_laps == 3
    assert rs.phase == RacePhase.INIT
    assert rs.started_at_s is None
    assert rs.finished_at_s is None
    assert rs.map_name == ""
    assert rs.cars == {}
    assert rs.circuit_waypoint_count == 0


def test_race_state_cars_independent_per_instance() -> None:
    a = RaceState(config_num_cars=2, config_num_laps=1)
    b = RaceState(config_num_cars=2, config_num_laps=1)
    a.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    assert b.cars == {}


def test_race_state_is_mutable() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=3)
    rs.phase = RacePhase.RUNNING
    rs.started_at_s = 10.0
    rs.map_name = "Town01"
    rs.circuit_waypoint_count = 64
    rs.cars[5] = CarState(actor_id=5, is_player=True, color="255,0,0")
    assert rs.phase == RacePhase.RUNNING
    assert rs.started_at_s == 10.0
    assert rs.map_name == "Town01"
    assert rs.circuit_waypoint_count == 64
    assert 5 in rs.cars


def test_elapsed_s_zero_before_start() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    assert rs.elapsed_s(100.0) == 0.0


def test_elapsed_s_returns_delta_when_running() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    rs.started_at_s = 10.0
    rs.phase = RacePhase.RUNNING
    assert rs.elapsed_s(15.5) == 5.5


def test_elapsed_s_returns_delta_when_finished() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    rs.started_at_s = 10.0
    rs.phase = RacePhase.FINISHED
    rs.finished_at_s = 40.0
    assert rs.elapsed_s(40.0) == 30.0


def test_all_finished_empty_returns_false() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    assert rs.all_finished() is False


def test_all_finished_partial_returns_false() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[1].finish_position = 1
    assert rs.all_finished() is False


def test_all_finished_all_set_returns_true() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[1].finish_position = 1
    rs.cars[2].finish_position = 2
    assert rs.all_finished() is True


def test_all_finished_dnf_counts_as_finished() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[1].finish_position = 1
    rs.cars[2].dnf = True
    rs.cars[2].finish_position = 2
    assert rs.all_finished() is True


def test_player_returns_player_car() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=False, color="b")
    rs.cars[2] = CarState(actor_id=2, is_player=True, color="r")
    p = rs.player()
    assert p.actor_id == 2
    assert p.is_player is True


def test_player_missing_raises() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=False, color="b")
    with pytest.raises(KeyError, match="no player car"):
        rs.player()


def test_player_empty_raises() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    with pytest.raises(KeyError, match="no player car"):
        rs.player()


def test_leaderboard_empty() -> None:
    rs = RaceState(config_num_cars=2, config_num_laps=1)
    assert rs.leaderboard() == []


def test_leaderboard_orders_finished_by_position() -> None:
    rs = RaceState(config_num_cars=3, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[3] = CarState(actor_id=3, is_player=False, color="g")
    rs.cars[1].finish_position = 3
    rs.cars[2].finish_position = 1
    rs.cars[3].finish_position = 2
    order = [c.actor_id for c in rs.leaderboard()]
    assert order == [2, 3, 1]


def test_leaderboard_finished_before_still_racing() -> None:
    rs = RaceState(config_num_cars=3, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[3] = CarState(actor_id=3, is_player=False, color="g")
    rs.cars[1].finish_position = 1
    # cars 2 and 3 still racing
    order = [c.actor_id for c in rs.leaderboard()]
    assert order[0] == 1
    assert set(order[1:]) == {2, 3}


def test_leaderboard_still_racing_by_progress_desc() -> None:
    rs = RaceState(config_num_cars=3, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[3] = CarState(actor_id=3, is_player=False, color="g")
    # none finished
    rs.cars[1].laps_finished = 2
    rs.cars[1].waypoint_index = 40
    rs.cars[2].laps_finished = 2
    rs.cars[2].waypoint_index = 50  # further along
    rs.cars[3].laps_finished = 1  # fewer laps
    rs.cars[3].waypoint_index = 60
    order = [c.actor_id for c in rs.leaderboard()]
    # car 2 (2 laps, wp 50) before car 1 (2 laps, wp 40) before car 3 (1 lap)
    assert order == [2, 1, 3]


def test_leaderboard_dnf_at_back_of_finished() -> None:
    rs = RaceState(config_num_cars=3, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[3] = CarState(actor_id=3, is_player=False, color="g")
    rs.cars[1].finish_position = 1
    rs.cars[2].dnf = True
    rs.cars[2].finish_position = 3  # num_cars per race_manager contract
    rs.cars[3].finish_position = 2
    order = [c.actor_id for c in rs.leaderboard()]
    assert order == [1, 3, 2]
    assert rs.leaderboard()[-1].dnf is True


def test_leaderboard_mixed_finished_dnf_racing() -> None:
    rs = RaceState(config_num_cars=4, config_num_laps=1)
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="r")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="b")
    rs.cars[3] = CarState(actor_id=3, is_player=False, color="g")
    rs.cars[4] = CarState(actor_id=4, is_player=False, color="y")
    rs.cars[1].finish_position = 1
    rs.cars[2].dnf = True
    rs.cars[2].finish_position = 4  # num_cars
    # cars 3, 4 still racing; 3 further along
    rs.cars[3].laps_finished = 2
    rs.cars[3].waypoint_index = 30
    rs.cars[4].laps_finished = 1
    rs.cars[4].waypoint_index = 60
    order = [c.actor_id for c in rs.leaderboard()]
    assert order == [1, 2, 3, 4]

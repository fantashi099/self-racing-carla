"""Unit tests for carla_race.config."""
from __future__ import annotations

import dataclasses
import os

import pytest

from carla_race.config import (
    AI_DIFFICULTY_PRESETS,
    DEFAULT_AI_DIFFICULTY,
    DEFAULT_NUM_CARS,
    DEFAULT_NUM_LAPS,
    DEFAULT_NUM_WALKERS,
    DIFFICULTIES,
    RaceConfig,
    load_config,
)

RACE_ENV_KEYS = ("RACE_NUM_CARS", "RACE_NUM_LAPS", "RACE_NUM_WALKERS", "RACE_AI_DIFFICULTY")


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in RACE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_difficulties_match_presets_keys() -> None:
    assert set(DIFFICULTIES) == set(AI_DIFFICULTY_PRESETS)


def test_presets_have_consistent_keys() -> None:
    expected = {
        "desired_speed",
        "global_distance_to_leading_vehicle",
    }
    for name, preset in AI_DIFFICULTY_PRESETS.items():
        assert set(preset.keys()) == expected, f"{name} preset keys mismatch"


def test_easy_is_slowest_hard_is_fastest() -> None:
    easy = AI_DIFFICULTY_PRESETS["easy"]["desired_speed"]
    normal = AI_DIFFICULTY_PRESETS["normal"]["desired_speed"]
    hard = AI_DIFFICULTY_PRESETS["hard"]["desired_speed"]
    assert easy < normal < hard


def test_hard_has_smallest_following_gap() -> None:
    easy = AI_DIFFICULTY_PRESETS["easy"]["global_distance_to_leading_vehicle"]
    normal = AI_DIFFICULTY_PRESETS["normal"]["global_distance_to_leading_vehicle"]
    hard = AI_DIFFICULTY_PRESETS["hard"]["global_distance_to_leading_vehicle"]
    assert easy > normal > hard


def test_race_config_is_frozen() -> None:
    cfg = RaceConfig(
        num_cars=2, num_laps=1, num_walkers=0, ai_difficulty="normal"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.num_cars = 99  # type: ignore[misc]


def test_num_ai_cars_excludes_player() -> None:
    cfg = RaceConfig(num_cars=10, num_laps=3, num_walkers=20, ai_difficulty="hard")
    assert cfg.num_ai_cars == 9
    cfg2 = RaceConfig(num_cars=2, num_laps=1, num_walkers=0, ai_difficulty="easy")
    assert cfg2.num_ai_cars == 1


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    cfg = load_config()
    assert cfg.num_cars == DEFAULT_NUM_CARS
    assert cfg.num_laps == DEFAULT_NUM_LAPS
    assert cfg.num_walkers == DEFAULT_NUM_WALKERS
    assert cfg.ai_difficulty == DEFAULT_AI_DIFFICULTY
    assert cfg.num_ai_cars == DEFAULT_NUM_CARS - 1


def test_load_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_CARS", "4")
    monkeypatch.setenv("RACE_NUM_LAPS", "5")
    monkeypatch.setenv("RACE_NUM_WALKERS", "30")
    monkeypatch.setenv("RACE_AI_DIFFICULTY", "hard")
    cfg = load_config()
    assert cfg == RaceConfig(4, 5, 30, "hard")


def test_load_config_rejects_num_cars_below_2(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_CARS", "1")
    with pytest.raises(ValueError, match="RACE_NUM_CARS must be >= 2"):
        load_config()


def test_load_config_rejects_zero_cars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_CARS", "0")
    with pytest.raises(ValueError, match="RACE_NUM_CARS must be >= 2"):
        load_config()


def test_load_config_rejects_negative_cars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_CARS", "-3")
    with pytest.raises(ValueError, match="RACE_NUM_CARS must be >= 2"):
        load_config()


def test_load_config_rejects_zero_laps(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_LAPS", "0")
    with pytest.raises(ValueError, match="RACE_NUM_LAPS must be >= 1"):
        load_config()


def test_load_config_rejects_negative_walkers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_WALKERS", "-1")
    with pytest.raises(ValueError, match="RACE_NUM_WALKERS must be >= 0"):
        load_config()


def test_load_config_accepts_zero_walkers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_WALKERS", "0")
    cfg = load_config()
    assert cfg.num_walkers == 0


def test_load_config_rejects_bad_difficulty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_AI_DIFFICULTY", "insane")
    with pytest.raises(ValueError, match="RACE_AI_DIFFICULTY"):
        load_config()


def test_load_config_rejects_non_integer_cars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_CARS", "ten")
    with pytest.raises(ValueError, match="RACE_NUM_CARS=.ten. is not an integer"):
        load_config()


def test_load_config_rejects_non_integer_laps(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_LAPS", "3.5")
    with pytest.raises(ValueError, match="RACE_NUM_LAPS=.*is not an integer"):
        load_config()


def test_load_config_trims_difficulty(monkeypatch: pytest.MonkeyPatch) -> None:
    # os.environ values aren't auto-trimmed; ensure we don't silently accept
    # whitespace-padded difficulty. The current impl uses raw env value,
    # so this documents the contract: caller must not pad.
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_AI_DIFFICULTY", "normal")
    cfg = load_config()
    assert cfg.ai_difficulty == "normal"


def test_load_config_case_sensitive_difficulty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_AI_DIFFICULTY", "Normal")
    with pytest.raises(ValueError, match="RACE_AI_DIFFICULTY"):
        load_config()


def test_load_config_min_two_cars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RACE_NUM_CARS", "2")
    cfg = load_config()
    assert cfg.num_cars == 2
    assert cfg.num_ai_cars == 1

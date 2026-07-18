"""Unit tests for carla_race.ai_driver (F4)."""
from __future__ import annotations

from typing import Any

import pytest

from carla_race.ai_driver import reset_ai_cars, setup_ai_cars


class FakeTM:
    def __init__(self) -> None:
        self.hybrid_physics_mode: list[bool] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.paths: dict[int, list[Any]] = {}
        self.method_calls: dict[str, list[tuple[int, Any]]] = {}

    def set_hybrid_physics_mode(self, enabled: bool) -> None:
        self.hybrid_physics_mode.append(enabled)

    def __getattr__(self, name: str) -> Any:
        # Generic setter recorder: set_<key>(actor_id, value) or set_path(actor_id, path)
        if name.startswith("set_"):
            def _record(*args: Any) -> None:
                self.calls.append((name, args))
                self.method_calls.setdefault(name, []).append(args)
            return _record
        raise AttributeError(name)


def test_setup_ai_cars_skips_player() -> None:
    tm = FakeTM()
    circuit = ["wp0", "wp1", "wp2"]
    setup_ai_cars(
        tm,
        car_actor_ids=[1, 2, 3],
        circuit=circuit,
        difficulty="normal",
        player_actor_id=2,
    )
    set_path_calls = tm.method_calls.get("set_path", [])
    set_path_actors = [args[0] for args in set_path_calls]
    assert 2 not in set_path_actors
    assert set(set_path_actors) == {1, 3}


def test_setup_ai_cars_sets_hybrid_physics_mode_once() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[1, 2],
        circuit=["wp"],
        difficulty="normal",
        player_actor_id=1,
    )
    assert tm.hybrid_physics_mode == [True]


def test_setup_ai_cars_applies_all_preset_keys_per_ai_car() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[1, 2],
        circuit=["wp"],
        difficulty="normal",
        player_actor_id=1,
    )
    # 4 preset keys applied to actor 2 (actor 1 is player, skipped)
    expected_keys = {
        "set_percentage_speed_difference",
        "set_global_distance_to_leading_vehicle",
        "set_auto_lane_change",
        "set_safety_mode",
    }
    applied_keys = {name for (name, _args) in tm.calls if name != "set_path"}
    assert applied_keys == expected_keys
    # Each key called exactly once (for actor 2)
    for key in expected_keys:
        calls = tm.method_calls[key]
        assert len(calls) == 1
        assert calls[0][0] == 2


def test_setup_ai_cars_easy_preset_values() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[1],
        circuit=["wp"],
        difficulty="easy",
        player_actor_id=999,
    )
    speed_calls = tm.method_calls["set_percentage_speed_difference"]
    assert speed_calls[0] == (1, 20.0)
    dist_calls = tm.method_calls["set_global_distance_to_leading_vehicle"]
    assert dist_calls[0] == (1, 5.0)
    lane_calls = tm.method_calls["set_auto_lane_change"]
    assert lane_calls[0] == (1, False)
    safety_calls = tm.method_calls["set_safety_mode"]
    assert safety_calls[0] == (1, True)


def test_setup_ai_cars_hard_preset_values() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[1],
        circuit=["wp"],
        difficulty="hard",
        player_actor_id=999,
    )
    assert tm.method_calls["set_percentage_speed_difference"][0] == (1, -20.0)
    assert tm.method_calls["set_global_distance_to_leading_vehicle"][0] == (1, 2.0)
    assert tm.method_calls["set_auto_lane_change"][0] == (1, True)
    assert tm.method_calls["set_safety_mode"][0] == (1, False)


def test_setup_ai_cars_set_path_uses_circuit() -> None:
    tm = FakeTM()
    circuit = ["wp0", "wp1", "wp2"]
    setup_ai_cars(
        tm,
        car_actor_ids=[1, 2],
        circuit=circuit,
        difficulty="normal",
        player_actor_id=2,
    )
    set_path_calls = tm.method_calls["set_path"]
    assert len(set_path_calls) == 1
    actor_id, path = set_path_calls[0]
    assert actor_id == 1
    assert path is circuit  # same object, not a copy


def test_setup_ai_cars_unknown_difficulty_raises() -> None:
    tm = FakeTM()
    with pytest.raises(ValueError, match="unknown difficulty"):
        setup_ai_cars(
            tm,
            car_actor_ids=[1],
            circuit=["wp"],
            difficulty="insane",
            player_actor_id=999,
        )


def test_setup_ai_cars_empty_car_list() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[],
        circuit=["wp"],
        difficulty="normal",
        player_actor_id=999,
    )
    # hybrid physics still set globally
    assert tm.hybrid_physics_mode == [True]
    assert "set_path" not in tm.method_calls


def test_setup_ai_cars_player_only_no_ai_calls() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[1],
        circuit=["wp"],
        difficulty="normal",
        player_actor_id=1,
    )
    assert "set_path" not in tm.method_calls
    # No per-actor preset applied
    assert "set_percentage_speed_difference" not in tm.method_calls
    # But hybrid physics still set
    assert tm.hybrid_physics_mode == [True]


def test_reset_ai_cars_clears_path_for_each() -> None:
    tm = FakeTM()
    reset_ai_cars(tm, car_actor_ids=[1, 2, 3])
    set_path_calls = tm.method_calls["set_path"]
    assert len(set_path_calls) == 3
    for actor_id, path in set_path_calls:
        assert path == []
    assert [args[0] for args in set_path_calls] == [1, 2, 3]


def test_reset_ai_cars_empty_list_noop() -> None:
    tm = FakeTM()
    reset_ai_cars(tm, car_actor_ids=[])
    assert "set_path" not in tm.method_calls


def test_setup_ai_cars_preserves_call_order_hybrid_first() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[1],
        circuit=["wp"],
        difficulty="normal",
        player_actor_id=999,
    )
    # hybrid_physics_mode is recorded in __init__ via direct call before
    # the per-actor getattr-driven setters
    assert tm.hybrid_physics_mode == [True]
    # First getattr-driven call should be a preset key, not set_path
    first_getattr_call = tm.calls[0][0]
    assert first_getattr_call != "set_path"


def test_setup_ai_cars_skips_missing_tm_methods(capfd: pytest.CaptureFixture[str]) -> None:
    """TM methods that don't exist (CARLA version mismatch) are skipped with
    a one-time warning listing the available set_ methods."""
    # Reset the module-level warning cache so the warning fires in this test.
    from carla_race import ai_driver as mod
    mod._warned_missing.clear()

    class PartialTM:
        def __init__(self) -> None:
            self.hybrid_physics_mode: list[bool] = []
            self.paths: dict[int, list[Any]] = {}

        def set_hybrid_physics_mode(self, enabled: bool) -> None:
            self.hybrid_physics_mode.append(enabled)

        def set_path(self, actor_id: int, path: list[Any]) -> None:
            self.paths[actor_id] = path

        # NOTE: no set_percentage_speed_difference, set_global_distance_to_leading_vehicle,
        # set_auto_lane_change, set_safety_mode — version-mismatch simulation

    tm = PartialTM()
    setup_ai_cars(
        tm,
        car_actor_ids=[1],
        circuit=["wp"],
        difficulty="normal",
        player_actor_id=999,
    )
    # hybrid physics + set_path still applied
    assert tm.hybrid_physics_mode == [True]
    assert tm.paths == {1: ["wp"]}
    # warning was printed to stderr
    captured = capfd.readouterr()
    assert "TM has no set_percentage_speed_difference" in captured.err
    assert "Available set_ methods" in captured.err


def test_setup_ai_cars_missing_method_warning_emitted_once(
    capfd: pytest.CaptureFixture[str],
) -> None:
    from carla_race import ai_driver as mod
    mod._warned_missing.clear()

    class PartialTM:
        def set_hybrid_physics_mode(self, enabled: bool) -> None: pass
        def set_path(self, actor_id: int, path: list[Any]) -> None: pass

    tm = PartialTM()
    # Apply 3 AI cars — missing-method warning should fire once per key, not 3x
    setup_ai_cars(tm, car_actor_ids=[1, 2, 3], circuit=["wp"], difficulty="normal", player_actor_id=999)
    captured = capfd.readouterr()
    # 4 preset keys → 4 warnings (one per missing key)
    err_lines = [l for l in captured.err.splitlines() if "TM has no" in l]
    assert len(err_lines) == 4

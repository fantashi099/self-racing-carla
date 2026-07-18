"""Unit tests for carla_race.ai_driver (F4)."""
from __future__ import annotations

from typing import Any

import pytest

from carla_race.ai_driver import reset_ai_cars, setup_ai_cars


class FakeWaypoint:
    def __init__(self, idx: int) -> None:
        self.idx = idx


class FakeLoc:
    def __init__(self, idx: int) -> None:
        self.idx = idx


class FakeTransform:
    def __init__(self, idx: int) -> None:
        self.location = FakeLoc(idx)


class FakeMap:
    def __init__(self) -> None:
        self.get_waypoint_calls: list[int] = []

    def get_waypoint(self, loc: FakeLoc) -> FakeWaypoint:
        self.get_waypoint_calls.append(loc.idx)
        return FakeWaypoint(loc.idx)


class FakeActor:
    def __init__(self, actor_id: int) -> None:
        self.id = actor_id


class FakeTM:
    def __init__(self) -> None:
        self.hybrid_physics_mode: list[bool] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.method_calls: dict[str, list[tuple[Any, ...]]] = {}
        self.paths: dict[int, list[Any]] = {}

    def set_hybrid_physics_mode(self, enabled: bool) -> None:
        self.hybrid_physics_mode.append(enabled)

    def set_path(self, actor: Any, path: list[Any], empty_buffer: bool = True) -> None:
        self.calls.append(("set_path", (actor, path)))
        self.method_calls.setdefault("set_path", []).append((actor, path))
        # record by actor id for easy lookup
        if hasattr(actor, "id"):
            self.paths[actor.id] = path

    def __getattr__(self, name: str) -> Any:
        if name.startswith("set_"):
            def _record(*args: Any) -> None:
                self.calls.append((name, args))
                self.method_calls.setdefault(name, []).append(args)
            return _record
        raise AttributeError(name)


def _circuit(n: int = 4) -> list[FakeTransform]:
    return [FakeTransform(i) for i in range(n)]


def test_setup_ai_cars_skips_player() -> None:
    tm = FakeTM()
    car_actors = {1: FakeActor(1), 2: FakeActor(2), 3: FakeActor(3)}
    setup_ai_cars(
        tm,
        car_actors,
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=2,
    )
    set_path_calls = tm.method_calls.get("set_path", [])
    set_path_ids = [args[0].id for args in set_path_calls]
    assert 2 not in set_path_ids
    assert set(set_path_ids) == {1, 3}


def test_setup_ai_cars_sets_hybrid_physics_mode_once() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1), 2: FakeActor(2)},
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=1,
    )
    assert tm.hybrid_physics_mode == [True]


def test_setup_ai_cars_applies_all_preset_keys_per_ai_car() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1), 2: FakeActor(2)},
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=1,
    )
    expected_keys = {
        "set_desired_speed",
        "set_global_distance_to_leading_vehicle",
    }
    applied_keys = {name for (name, _args) in tm.calls if name != "set_path"}
    assert applied_keys == expected_keys
    for key in expected_keys:
        calls = tm.method_calls[key]
        assert len(calls) == 1
        # called with the actor object (id=2 since 1 is player)
        assert calls[0][0].id == 2


def test_setup_ai_cars_easy_preset_values() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1)},
        FakeMap(),
        _circuit(),
        difficulty="easy",
        player_actor_id=999,
    )
    speed_calls = tm.method_calls["set_desired_speed"]
    assert speed_calls[0][1] == 20.0
    dist_calls = tm.method_calls["set_global_distance_to_leading_vehicle"]
    assert dist_calls[0][1] == 5.0


def test_setup_ai_cars_hard_preset_values() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1)},
        FakeMap(),
        _circuit(),
        difficulty="hard",
        player_actor_id=999,
    )
    assert tm.method_calls["set_desired_speed"][0][1] == 60.0
    assert tm.method_calls["set_global_distance_to_leading_vehicle"][0][1] == 2.0


def test_setup_ai_cars_set_path_converts_circuit_to_locations_and_closes_loop() -> None:
    tm = FakeTM()
    circuit = _circuit(4)
    setup_ai_cars(
        tm,
        {1: FakeActor(1)},
        FakeMap(),
        circuit,
        difficulty="normal",
        player_actor_id=999,
    )
    set_path_calls = tm.method_calls["set_path"]
    assert len(set_path_calls) == 1
    actor, path = set_path_calls[0]
    assert actor.id == 1
    # 4 locations + 1 closure = 5
    assert len(path) == 5
    # locations extracted from FakeTransform.location (FakeLoc with .idx)
    assert [loc.idx for loc in path] == [0, 1, 2, 3, 0]


def test_setup_ai_cars_set_path_uses_actor_object_not_id() -> None:
    tm = FakeTM()
    actor = FakeActor(1)
    setup_ai_cars(
        tm,
        {1: actor},
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=999,
    )
    set_path_calls = tm.method_calls["set_path"]
    assert set_path_calls[0][0] is actor  # same object reference


def test_setup_ai_cars_unknown_difficulty_raises() -> None:
    tm = FakeTM()
    with pytest.raises(ValueError, match="unknown difficulty"):
        setup_ai_cars(
            tm,
            {1: FakeActor(1)},
            FakeMap(),
            _circuit(),
            difficulty="insane",
            player_actor_id=999,
        )


def test_setup_ai_cars_empty_car_actors() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        {},
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=999,
    )
    assert tm.hybrid_physics_mode == [True]
    assert "set_path" not in tm.method_calls


def test_setup_ai_cars_player_only_no_ai_calls() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1)},
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=1,
    )
    assert "set_path" not in tm.method_calls
    assert "set_desired_speed" not in tm.method_calls
    assert tm.hybrid_physics_mode == [True]


def test_setup_ai_cars_skips_set_path_when_circuit_empty() -> None:
    tm = FakeTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1)},
        FakeMap(),
        [],
        difficulty="normal",
        player_actor_id=999,
    )
    # no circuit → no path → set_path skipped
    assert "set_path" not in tm.method_calls
    # but preset keys still applied
    assert "set_desired_speed" in tm.method_calls


def test_reset_ai_cars_clears_path_for_each() -> None:
    tm = FakeTM()
    car_actors = {1: FakeActor(1), 2: FakeActor(2), 3: FakeActor(3)}
    reset_ai_cars(tm, car_actors)
    set_path_calls = tm.method_calls["set_path"]
    assert len(set_path_calls) == 3
    for actor, path in set_path_calls:
        assert path == []
    assert [args[0].id for args in set_path_calls] == [1, 2, 3]


def test_reset_ai_cars_empty_dict_noop() -> None:
    tm = FakeTM()
    reset_ai_cars(tm, {})
    assert "set_path" not in tm.method_calls


def test_setup_ai_cars_skips_missing_tm_methods(
    capfd: pytest.CaptureFixture[str],
) -> None:
    from carla_race import ai_driver as mod
    mod._warned_missing.clear()

    class PartialTM:
        def __init__(self) -> None:
            self.hybrid_physics_mode: list[bool] = []

        def set_hybrid_physics_mode(self, enabled: bool) -> None:
            self.hybrid_physics_mode.append(enabled)

        def set_path(self, actor: Any, path: list[Any], empty_buffer: bool = True) -> None:
            pass
        # no set_desired_speed, set_global_distance_to_leading_vehicle

    tm = PartialTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1)},
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=999,
    )
    assert tm.hybrid_physics_mode == [True]
    captured = capfd.readouterr()
    assert "TM has no set_desired_speed" in captured.err
    assert "Available set_ methods" in captured.err


def test_setup_ai_cars_missing_method_warning_emitted_once(
    capfd: pytest.CaptureFixture[str],
) -> None:
    from carla_race import ai_driver as mod
    mod._warned_missing.clear()

    class PartialTM:
        def set_hybrid_physics_mode(self, enabled: bool) -> None: pass
        def set_path(self, actor: Any, path: list[Any], empty_buffer: bool = True) -> None: pass

    tm = PartialTM()
    setup_ai_cars(
        tm,
        {1: FakeActor(1), 2: FakeActor(2), 3: FakeActor(3)},
        FakeMap(),
        _circuit(),
        difficulty="normal",
        player_actor_id=999,
    )
    captured = capfd.readouterr()
    err_lines = [l for l in captured.err.splitlines() if "TM has no" in l]
    # 2 preset keys missing → 2 warnings (one per key)
    assert len(err_lines) == 2

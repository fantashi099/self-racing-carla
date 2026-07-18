"""Unit tests for carla_race.map_pool (F1).

Uses lightweight fakes that satisfy the structural surface used by
map_pool: ``client.get_available_maps() -> [obj with .name]``,
``client.load_world(name) -> world``, ``world.get_map() -> obj with .name``.
No ``carla`` pip package required.
"""
from __future__ import annotations

import random

import pytest

from carla_race.map_pool import (
    _basename,
    _exclude_from_env,
    load_map,
    pick_and_load,
    random_map,
)


class FakeMap:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeWorld:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_map(self) -> FakeMap:
        return FakeMap(self._name)


class FakeClient:
    def __init__(self, names: list[str]) -> None:
        self._maps: list[FakeMap] = [FakeMap(n) for n in names]
        self.loaded: list[str] = []

    def get_available_maps(self) -> list[FakeMap]:
        return list(self._maps)

    def load_world(self, name: str) -> FakeWorld:
        self.loaded.append(name)
        return FakeWorld(name)


def test_basename_strips_path() -> None:
    assert _basename("/Game/Carla/Maps/Town01") == "Town01"
    assert _basename("Town02") == "Town02"


def test_exclude_from_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RACE_EXCLUDE_MAPS", raising=False)
    assert _exclude_from_env() == ()


def test_exclude_from_env_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACE_EXCLUDE_MAPS", "Town10HD_Opt,Town12")
    assert _exclude_from_env() == ("Town10HD_Opt", "Town12")


def test_exclude_from_env_trims_and_drops_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACE_EXCLUDE_MAPS", " a , , b ")
    assert _exclude_from_env() == ("a", "b")


def test_random_map_picks_from_pool() -> None:
    c = FakeClient(["/Game/Carla/Maps/Town01", "/Game/Carla/Maps/Town02"])
    picks = {random_map(c, rng=random.Random(i)) for i in range(20)}
    assert picks <= {"Town01", "Town02"}
    assert picks  # non-empty


def test_random_map_respects_explicit_exclude() -> None:
    c = FakeClient(["Town01", "Town02", "Town03"])
    for i in range(20):
        assert (
            random_map(c, exclude=["Town01", "Town03"], rng=random.Random(i))
            == "Town02"
        )


def test_random_map_respects_env_exclude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACE_EXCLUDE_MAPS", "Town02")
    c = FakeClient(["Town01", "Town02"])
    for i in range(20):
        assert random_map(c, rng=random.Random(i)) == "Town01"


def test_random_map_combines_arg_and_env_exclude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RACE_EXCLUDE_MAPS", "Town03")
    c = FakeClient(["Town01", "Town02", "Town03", "Town04"])
    for i in range(20):
        pick = random_map(c, exclude=["Town02"], rng=random.Random(i))
        assert pick in {"Town01", "Town04"}


def test_random_map_empty_pool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RACE_EXCLUDE_MAPS", raising=False)
    c = FakeClient(["Town01"])
    with pytest.raises(RuntimeError, match="no maps available"):
        random_map(c, exclude=["Town01"])


def test_random_map_env_excludes_all_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACE_EXCLUDE_MAPS", "Town01,Town02")
    c = FakeClient(["Town01", "Town02"])
    with pytest.raises(RuntimeError):
        random_map(c)


def test_load_map_returns_map_with_name() -> None:
    c = FakeClient(["Town01"])
    m = load_map(c, "Town01")
    assert m.name == "Town01"
    assert c.loaded == ["Town01"]


def test_pick_and_load_round_trip() -> None:
    c = FakeClient(["Town01", "Town02"])
    name, m = pick_and_load(c, rng=random.Random(0))
    assert name in {"Town01", "Town02"}
    assert m.name == name
    assert c.loaded == [name]


def test_pick_and_load_deterministic_with_seeded_rng() -> None:
    c = FakeClient([f"/Game/Carla/Maps/Town0{i}" for i in range(1, 6)])
    r1 = random.Random(42)
    r2 = random.Random(42)
    assert random_map(c, rng=r1) == random_map(c, rng=r2)


def test_pick_and_load_default_rng_progresses() -> None:
    # default module-level random should still produce valid picks
    c = FakeClient(["Town01", "Town02", "Town03"])
    for _ in range(5):
        name, _m = pick_and_load(c)
        assert name in {"Town01", "Town02", "Town03"}
        c.loaded.clear()

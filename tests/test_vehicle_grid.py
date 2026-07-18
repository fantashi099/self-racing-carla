"""Unit tests for carla_race.vehicle_grid (F2).

Uses lightweight fakes that satisfy the structural surface used by
vehicle_grid: ``world.get_blueprint_library()``,
``world.get_map().get_spawn_points()``, ``world.spawn_actor(bp, tf)``,
``world.get_actor(id)``, ``actor.id``, ``actor.destroy()``,
``bp.set_attribute(name, value)``, ``bp_lib.find(id)``. No ``carla`` pip
package required.
"""
from __future__ import annotations

import pytest

from carla_race.vehicle_grid import (
    GRID_COLORS,
    CarSpawn,
    DEFAULT_BLUEPRINT,
    destroy_grid,
    spawn_grid,
)


class FakeBlueprint:
    def __init__(self, type_id: str) -> None:
        self.type_id = type_id
        self.attributes: dict[str, str] = {}

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value


class FakeBlueprintLibrary:
    def __init__(self) -> None:
        self.find_calls: list[str] = []

    def find(self, type_id: str) -> FakeBlueprint:
        self.find_calls.append(type_id)
        return FakeBlueprint(type_id)


class FakeActor:
    def __init__(self, actor_id: int) -> None:
        self.id = actor_id
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class FakeSpawnPoint:
    def __init__(self, idx: int) -> None:
        self.idx = idx


class FakeMap:
    def __init__(self, n_spawn: int) -> None:
        self._spawn_pts = [FakeSpawnPoint(i) for i in range(n_spawn)]

    def get_spawn_points(self) -> list[FakeSpawnPoint]:
        return list(self._spawn_pts)


class FakeWorld:
    def __init__(self, n_spawn: int, *, next_actor_id: int = 1000) -> None:
        self._map = FakeMap(n_spawn)
        self._bp_lib = FakeBlueprintLibrary()
        self._next_id = next_actor_id
        self.actors: dict[int, FakeActor] = {}
        self.spawn_log: list[tuple[str, FakeSpawnPoint]] = []

    def get_blueprint_library(self) -> FakeBlueprintLibrary:
        return self._bp_lib

    def get_map(self) -> FakeMap:
        return self._map

    def spawn_actor(self, bp: FakeBlueprint, tf: FakeSpawnPoint) -> FakeActor:
        self.spawn_log.append((bp.attributes.get("color", ""), tf))
        actor = FakeActor(self._next_id)
        self.actors[actor.id] = actor
        self._next_id += 1
        return actor

    def get_actor(self, actor_id: int) -> FakeActor | None:
        return self.actors.get(actor_id)


def test_grid_colors_has_ten_distinct() -> None:
    assert len(GRID_COLORS) == 10
    assert len(set(GRID_COLORS)) == 10


def test_grid_colors_are_rgb_strings() -> None:
    for c in GRID_COLORS:
        parts = c.split(",")
        assert len(parts) == 3
        for p in parts:
            assert 0 <= int(p) <= 255


def test_car_spawn_is_frozen() -> None:
    s = CarSpawn(actor_id=42, is_player=True, color="1,2,3", spawn_index=0)
    with pytest.raises(Exception):
        s.actor_id = 99  # type: ignore[misc]


def test_spawn_grid_single_car_is_player() -> None:
    w = FakeWorld(n_spawn=5)
    spawns = spawn_grid(w, num_cars=1)
    assert len(spawns) == 1
    assert spawns[0].is_player is True
    assert spawns[0].spawn_index == 0
    assert spawns[0].color == "255,0,0"
    assert spawns[0].actor_id == 1000
    assert w._bp_lib.find_calls == [DEFAULT_BLUEPRINT]


def test_spawn_grid_player_color_override() -> None:
    w = FakeWorld(n_spawn=5)
    spawns = spawn_grid(w, num_cars=1, player_color="10,20,30")
    assert spawns[0].color == "10,20,30"
    assert w.spawn_log[0][0] == "10,20,30"


def test_spawn_grid_ten_cars_distinct_colors() -> None:
    w = FakeWorld(n_spawn=20)
    spawns = spawn_grid(w, num_cars=10)
    assert len(spawns) == 10
    colors = [s.color for s in spawns]
    assert len(set(colors)) == 10
    assert spawns[0].is_player is True
    assert all(not s.is_player for s in spawns[1:])
    assert [s.spawn_index for s in spawns] == list(range(10))


def test_spawn_grid_skips_player_color_for_ai() -> None:
    w = FakeWorld(n_spawn=20)
    player_color = "0,0,255"  # exists in GRID_COLORS at index 1
    spawns = spawn_grid(w, num_cars=10, player_color=player_color)
    ai_colors = [s.color for s in spawns if not s.is_player]
    assert player_color not in ai_colors
    assert len(set(ai_colors)) == 9
    assert spawns[0].color == player_color


def test_spawn_grid_uses_first_n_spawn_points_in_order() -> None:
    w = FakeWorld(n_spawn=20)
    spawns = spawn_grid(w, num_cars=5)
    assert [s.spawn_index for s in spawns] == [0, 1, 2, 3, 4]
    used_indices = [tf.idx for (_color, tf) in w.spawn_log]
    assert used_indices == [0, 1, 2, 3, 4]


def test_spawn_grid_returns_in_spawn_index_order() -> None:
    w = FakeWorld(n_spawn=20)
    spawns = spawn_grid(w, num_cars=7)
    assert [s.spawn_index for s in spawns] == list(range(7))


def test_spawn_grid_actor_ids_match_world_actors() -> None:
    w = FakeWorld(n_spawn=20)
    spawns = spawn_grid(w, num_cars=4)
    for s in spawns:
        assert s.actor_id in w.actors


def test_spawn_grid_sets_color_attribute_on_blueprint() -> None:
    w = FakeWorld(n_spawn=10)
    spawns = spawn_grid(w, num_cars=3)
    assert w.spawn_log[0][0] == spawns[0].color
    assert w.spawn_log[1][0] == spawns[1].color
    assert w.spawn_log[2][0] == spawns[2].color


def test_spawn_grid_rejects_zero_cars() -> None:
    w = FakeWorld(n_spawn=10)
    with pytest.raises(ValueError, match="num_cars must be >= 1"):
        spawn_grid(w, num_cars=0)


def test_spawn_grid_rejects_negative_cars() -> None:
    w = FakeWorld(n_spawn=10)
    with pytest.raises(ValueError, match="num_cars must be >= 1"):
        spawn_grid(w, num_cars=-3)


def test_spawn_grid_rejects_more_than_spawn_points() -> None:
    w = FakeWorld(n_spawn=4)
    with pytest.raises(ValueError, match="exceeds spawn points available=4"):
        spawn_grid(w, num_cars=5)


def test_spawn_grid_rejects_more_than_distinct_colors() -> None:
    w = FakeWorld(n_spawn=200)
    with pytest.raises(ValueError, match="distinct AI colors"):
        spawn_grid(w, num_cars=12)


def test_spawn_grid_player_color_non_default_excludes_match_from_ai() -> None:
    w = FakeWorld(n_spawn=20)
    spawns = spawn_grid(w, num_cars=10, player_color="255,255,255")
    ai_colors = [s.color for s in spawns if not s.is_player]
    assert "255,255,255" not in ai_colors
    assert spawns[0].color == "255,255,255"


def test_destroy_grid_destroys_all_actors() -> None:
    w = FakeWorld(n_spawn=20)
    spawns = spawn_grid(w, num_cars=5)
    destroy_grid(w, spawns)
    for s in spawns:
        assert w.actors[s.actor_id].destroyed is True


def test_destroy_grid_empty_is_noop() -> None:
    w = FakeWorld(n_spawn=5)
    destroy_grid(w, [])
    assert all(not a.destroyed for a in w.actors.values())


def test_destroy_grid_missing_actor_is_ignored() -> None:
    w = FakeWorld(n_spawn=20)
    spawns = spawn_grid(w, num_cars=2)
    w.actors.pop(spawns[1].actor_id, None)
    destroy_grid(w, spawns)
    assert w.actors[spawns[0].actor_id].destroyed is True


def test_spawn_grid_default_player_color_is_red() -> None:
    w = FakeWorld(n_spawn=10)
    spawns = spawn_grid(w, num_cars=1)
    assert spawns[0].color == "255,0,0"

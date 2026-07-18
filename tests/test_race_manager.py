"""Unit tests for carla_race.race_manager.

Comprehensive fakes satisfying every sub-module surface: map_pool, circuit,
vehicle_grid, collision_scoring, traffic, ai_driver, lap_tracker. No carla
package required.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import pytest

from carla_race.config import RaceConfig
from carla_race.race_manager import RaceManager


# ── fakes ────────────────────────────────────────────────────────────────

class Loc:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class Tf:
    def __init__(self, x: float, y: float) -> None:
        self.location = Loc(x, y)
        self.rotation = None


class FakeActor:
    def __init__(self, actor_id: int) -> None:
        self.id = actor_id
        self._transform = Tf(0.0, 0.0)
        self.alive = True
        self.destroyed = False
        self.type_id = "vehicle.lincoln.mkz_2017"

    def get_transform(self) -> Tf:
        return self._transform

    def set_position(self, x: float, y: float) -> None:
        self._transform = Tf(x, y)

    def is_alive(self) -> bool:
        return self.alive

    def destroy(self) -> None:
        self.destroyed = True
        self.alive = False


class FakeSensor(FakeActor):
    def __init__(self, actor_id: int) -> None:
        super().__init__(actor_id)
        self.type_id = "sensor.other.collision"
        self.callback: Any = None

    def listen(self, cb: Any) -> None:
        self.callback = cb


class FakeWalkerActor(FakeActor):
    def __init__(self, actor_id: int) -> None:
        super().__init__(actor_id)
        self.type_id = "walker.pedestrian.0001"


class FakeControllerActor(FakeActor):
    def __init__(self, actor_id: int, walker: FakeWalkerActor) -> None:
        super().__init__(actor_id)
        self.type_id = "controller.ai.walker"
        self.walker = walker
        self.started = False
        self.destination: Any = None

    def start(self) -> None:
        self.started = True

    def go_to_location(self, loc: Any) -> None:
        self.destination = loc

    def stop(self) -> None:
        pass


class FakeBP:
    def __init__(self, type_id: str) -> None:
        self.type_id = type_id
        self.attributes: dict[str, str] = {}

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value


class FakeBPLibrary:
    def __init__(self) -> None:
        self._walker = FakeBP("walker.pedestrian.0001")

    def find(self, type_id: str) -> FakeBP:
        return FakeBP(type_id)

    def filter(self, pattern: str) -> list[FakeBP]:
        if pattern == "walker.pedestrian.*":
            return [self._walker]
        return []


class FakeMap:
    def __init__(self, name: str, spawn_points: list[Tf]) -> None:
        self.name = name
        self._spawn = spawn_points

    def get_topology(self) -> list[tuple[Any, Any]]:
        return []  # force fallback to spawn points

    def get_spawn_points(self) -> list[Tf]:
        return list(self._spawn)


class FakeWorld:
    def __init__(self, map_obj: FakeMap) -> None:
        self._map = map_obj
        self._bp_lib = FakeBPLibrary()
        self._next_id = 1000
        self.actors: dict[int, FakeActor] = {}
        self._tm = FakeTM()
        self.nav_calls = 0

    def get_map(self) -> FakeMap:
        return self._map

    def get_blueprint_library(self) -> FakeBPLibrary:
        return self._bp_lib

    def get_spawn_points_count(self) -> int:
        return len(self._map.get_spawn_points())

    def spawn_actor(
        self,
        bp: FakeBP,
        tf: Any,
        attach_to: FakeActor | None = None,
    ) -> FakeActor:
        actor_id = self._next_id
        self._next_id += 1
        if bp.type_id == "sensor.other.collision":
            sensor = FakeSensor(actor_id)
            self.actors[actor_id] = sensor
            return sensor
        if bp.type_id.startswith("controller."):
            assert attach_to is not None
            ctrl = FakeControllerActor(actor_id, attach_to)  # type: ignore[arg-type]
            self.actors[actor_id] = ctrl
            return ctrl
        if bp.type_id.startswith("walker."):
            walker = FakeWalkerActor(actor_id)
            self.actors[actor_id] = walker
            return walker
        # vehicle
        actor = FakeActor(actor_id)
        self.actors[actor_id] = actor
        return actor

    def get_actor(self, actor_id: int) -> FakeActor | None:
        return self.actors.get(actor_id)

    def get_random_location_from_navigation(self) -> Loc:
        loc = Loc(float(self.nav_calls), 0.0)
        self.nav_calls += 1
        return loc

    def get_traffic_manager(self) -> "FakeTM":
        return self._tm


class FakeTM:
    def __init__(self) -> None:
        self.hybrid_physics_mode: list[bool] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def set_hybrid_physics_mode(self, enabled: bool) -> None:
        self.hybrid_physics_mode.append(enabled)

    def set_path(self, actor_id: int, path: list[Any]) -> None:
        self.calls.append(("set_path", (actor_id, path)))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("set_"):
            def _record(*args: Any) -> None:
                self.calls.append((name, args))
            return _record
        raise AttributeError(name)


class FakeClient:
    def __init__(self, maps: list[str], spawn_per_map: int = 64) -> None:
        self._maps = maps
        self._spawn_per_map = spawn_per_map
        self.loaded: list[str] = []
        self._world: FakeWorld | None = None

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def load_world(self, name: str) -> FakeWorld:
        self.loaded.append(name)
        spawn_pts = _circle_spawn_points(self._spawn_per_map)
        m = FakeMap(name, spawn_pts)
        self._world = FakeWorld(m)
        return _WorldWrapper(self._world, m)

    def get_world(self) -> FakeWorld:
        if self._world is None:
            raise RuntimeError("load_world not called")
        return self._world

    def get_trafficmanager(self, port: int) -> FakeTM:
        return self.get_world().get_traffic_manager()


class _WorldWrapper:
    """``client.load_world(name)`` returns a world; ``world.get_map()`` returns the map."""

    def __init__(self, world: FakeWorld, map_obj: FakeMap) -> None:
        self._world = world
        self._map = map_obj

    def get_map(self) -> FakeMap:
        return self._map


def _circle_spawn_points(n: int, radius: float = 50.0) -> list[Tf]:
    return [Tf(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _config(**overrides: Any) -> RaceConfig:
    defaults = dict(num_cars=2, num_laps=1, num_walkers=0, ai_difficulty="normal")
    defaults.update(overrides)
    return RaceConfig(**defaults)


# ── tests ────────────────────────────────────────────────────────────────

def test_start_returns_running_state() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    rs = rm.start()
    assert rs.phase.name == "RUNNING"
    assert rs.map_name == "Town01"
    assert rs.config_num_cars == 2
    assert rs.config_num_laps == 1
    assert len(rs.cars) == 2
    assert rs.circuit_waypoint_count == 64
    assert rs.started_at_s is not None
    assert rs.finished_at_s is None


def test_start_creates_player_and_ai_car_states() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    rs = rm.start()
    players = [car for car in rs.cars.values() if car.is_player]
    ais = [car for car in rs.cars.values() if not car.is_player]
    assert len(players) == 1
    assert len(ais) == 1


def test_start_attaches_collision_sensors() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    rm.start()
    assert len(rm._sensor_ids) == 2
    assert set(rm._sensor_ids.keys()) == set(rm._car_actors.keys())


def test_start_sets_up_ai_cars_via_tm() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    rm.start()
    tm = c.get_world().get_traffic_manager()
    assert tm.hybrid_physics_mode == [True]
    # set_path called for AI car only (player skipped)
    set_path_calls = [args for (name, args) in tm.calls if name == "set_path"]
    assert len(set_path_calls) == 1  # one AI car


def test_start_spawns_walkers_when_configured() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_walkers=3))
    rm.start()
    assert len(rm._walkers) == 3


def test_start_no_walkers_when_zero() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_walkers=0))
    rm.start()
    assert rm._walkers == []


def test_start_twice_raises() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    rm.start()
    with pytest.raises(RuntimeError, match="race already running"):
        rm.start()


def test_tick_before_start_returns_none() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    assert rm.tick() is None


def test_tick_with_no_movement_stays_running() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    rm.start()
    rs = rm.tick()
    assert rs is not None
    assert rs.phase.name == "RUNNING"
    # no car finished
    assert all(car.finish_position is None for car in rs.cars.values())


def test_tick_player_completes_one_lap_finishes_position_1() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=1))
    rm.start()
    circuit = rm._circuit
    player = rm._state.player()  # type: ignore[union-attr]
    player_actor = rm._car_actors[player.actor_id]
    # Drive player around the circle and back across the start line.
    for i in list(range(len(circuit))) + [0]:
        wp = circuit[i]
        player_actor.set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
        rm.tick()
        if player.finish_position is not None:
            break
    assert player.finish_position == 1
    assert player.laps_finished == 1
    assert player.finished_at_s is not None


def test_tick_assigns_finish_positions_in_order() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=1))
    rm.start()
    circuit = rm._circuit
    cars = list(rm._state.cars.values())  # type: ignore[union-attr]
    actors = [rm._car_actors[car.actor_id] for car in cars]
    # drive car 0 around the circle first, then car 1
    for i in list(range(len(circuit))) + [0]:
        wp = circuit[i]
        actors[0].set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
        rm.tick()
        if cars[0].finish_position is not None:
            break
    for i in list(range(len(circuit))) + [0]:
        wp = circuit[i]
        actors[1].set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
        rm.tick()
        if cars[1].finish_position is not None:
            break
    assert cars[0].finish_position == 1
    assert cars[1].finish_position == 2


def test_tick_dnf_when_actor_disappears() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=1))
    rm.start()
    cars = list(rm._state.cars.values())  # type: ignore[union-attr]
    # kill the AI car's actor
    ai_car = next(car for car in cars if not car.is_player)
    rm._car_actors[ai_car.actor_id] = None  # type: ignore[index]
    rm.tick()
    assert ai_car.dnf is True
    assert ai_car.finish_position == 2  # num_cars


def test_tick_all_finished_transitions_to_finished() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=1))
    rm.start()
    circuit = rm._circuit
    cars = list(rm._state.cars.values())  # type: ignore[union-attr]
    actors = [rm._car_actors[car.actor_id] for car in cars]
    # drive both cars around the circle and across the start line
    for car_idx, actor in enumerate(actors):
        for i in list(range(len(circuit))) + [0]:
            wp = circuit[i]
            actor.set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
            rm.tick()
            if cars[car_idx].finish_position is not None:
                break
    # one more tick to transition to FINISHED
    rm.tick()
    assert rm._state.phase.name == "FINISHED"  # type: ignore[union-attr]
    assert rm._state.finished_at_s is not None  # type: ignore[union-attr]


def test_state_snapshot_before_start() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    snap = rm.state_snapshot()
    assert snap["phase"] == "init"
    assert snap["cars"] == []


def test_state_snapshot_running_shape() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    rm.start()
    snap = rm.state_snapshot()
    assert snap["phase"] == "running"
    assert snap["map_name"] == "Town01"
    assert snap["num_cars"] == 2
    assert snap["num_laps"] == 1
    assert snap["circuit_waypoint_count"] == 64
    assert len(snap["cars"]) == 2
    car0 = snap["cars"][0]
    assert "actor_id" in car0
    assert "is_player" in car0
    assert "lap" in car0
    assert "finish_position" in car0
    assert "walker_hits" in car0
    assert "car_hits" in car0
    assert "elapsed_s" in snap


def test_state_snapshot_finished_includes_positions() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=1))
    rm.start()
    circuit = rm._circuit
    cars = list(rm._state.cars.values())  # type: ignore[union-attr]
    actors = [rm._car_actors[car.actor_id] for car in cars]
    for car_idx, actor in enumerate(actors):
        for i in list(range(len(circuit))) + [0]:
            wp = circuit[i]
            actor.set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
            rm.tick()
            if cars[car_idx].finish_position is not None:
                break
    rm.tick()
    snap = rm.state_snapshot()
    assert snap["phase"] == "finished"
    positions = [c["finish_position"] for c in snap["cars"]]
    assert sorted(p for p in positions if p is not None) == [1, 2]


def test_destroy_clears_state() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_walkers=2))
    rm.start()
    rm.destroy()
    assert rm._state is None
    assert rm._spawns == []
    assert rm._walkers == []
    assert rm._sensor_ids == {}
    assert rm._car_actors == {}


def test_destroy_destroys_actors() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_walkers=1))
    rm.start()
    world = c.get_world()
    spawned_ids = list(world.actors.keys())
    rm.destroy()
    for aid in spawned_ids:
        if aid in world.actors:
            assert world.actors[aid].destroyed is True


def test_restart_rebuilds_state() -> None:
    c = FakeClient(["Town01", "Town02"])
    rm = RaceManager(c, _config())
    rm.start()
    first_map = rm._state.map_name  # type: ignore[union-attr]
    rm.restart()
    assert rm._state is not None  # type: ignore[union-attr]
    assert rm._state.phase.name == "RUNNING"  # type: ignore[union-attr]
    assert len(rm._state.cars) == 2  # type: ignore[union-attr]


def test_state_snapshot_cars_ordered_by_leaderboard() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=1))
    rm.start()
    circuit = rm._circuit
    cars = list(rm._state.cars.values())  # type: ignore[union-attr]
    actors = [rm._car_actors[car.actor_id] for car in cars]
    # car 0 finishes first
    for i in list(range(len(circuit))) + [0]:
        wp = circuit[i]
        actors[0].set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
        rm.tick()
        if cars[0].finish_position is not None:
            break
    snap = rm.state_snapshot()
    # first entry in cars should be the one with finish_position=1
    assert snap["cars"][0]["finish_position"] == 1


def test_tick_skips_already_finished_cars() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=1))
    rm.start()
    circuit = rm._circuit
    cars = list(rm._state.cars.values())  # type: ignore[union-attr]
    actors = [rm._car_actors[car.actor_id] for car in cars]
    # finish car 0
    for i in list(range(len(circuit))) + [0]:
        wp = circuit[i]
        actors[0].set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
        rm.tick()
        if cars[0].finish_position is not None:
            break
    pos_before = cars[0].laps_finished
    # tick again — car 0 should not advance
    for _ in range(5):
        rm.tick()
    assert cars[0].laps_finished == pos_before


def test_lock_is_threading_lock() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config())
    assert isinstance(rm._lock, type(threading.Lock()))


def test_three_lap_race_player_finishes() -> None:
    c = FakeClient(["Town01"])
    rm = RaceManager(c, _config(num_cars=2, num_laps=3))
    rm.start()
    circuit = rm._circuit
    player = rm._state.player()  # type: ignore[union-attr]
    player_actor = rm._car_actors[player.actor_id]
    laps_seen = 0
    for lap in range(3):
        for i in list(range(len(circuit))) + [0]:
            wp = circuit[i]
            player_actor.set_position(wp.location.x, wp.location.y)  # type: ignore[attr-defined]
            rm.tick()
            if player.laps_finished > laps_seen:
                laps_seen = player.laps_finished
                break
    assert player.laps_finished == 3
    assert player.finish_position is not None
    assert player.finished_at_s is not None

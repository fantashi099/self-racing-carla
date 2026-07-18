"""Unit tests for carla_race.traffic (F5)."""
from __future__ import annotations

import dataclasses

import pytest

from carla_race.traffic import (
    CONTROLLER_BP,
    WALKER_BP_FALLBACK,
    WalkerSpawn,
    destroy_walkers,
    spawn_walkers,
)


class FakeLoc:
    def __init__(self, idx: int) -> None:
        self.idx = idx


class FakeWalkerActor:
    def __init__(self, actor_id: int) -> None:
        self.id = actor_id
        self.destroyed = False
        self.attached_to: "FakeControllerActor | None" = None

    def destroy(self) -> None:
        self.destroyed = True


class FakeControllerActor:
    def __init__(self, actor_id: int, walker: FakeWalkerActor) -> None:
        self.id = actor_id
        self.destroyed = False
        self.started = False
        self.destination: FakeLoc | None = None
        self.stopped = False
        self.walker = walker
        walker.attached_to = self

    def start(self) -> None:
        self.started = True

    def go_to_location(self, loc: FakeLoc) -> None:
        self.destination = loc

    def stop(self) -> None:
        self.stopped = True

    def destroy(self) -> None:
        self.destroyed = True


class FakeBP:
    def __init__(self, type_id: str) -> None:
        self.type_id = type_id
        self.attributes: dict[str, str] = {}

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value


class FakeBPLibrary:
    def __init__(self, walker_bps: list[FakeBP] | None = None) -> None:
        if walker_bps is None:
            walker_bps = [FakeBP("walker.pedestrian.0001")]
        self._walker_bps = walker_bps
        self.find_calls: list[str] = []

    def find(self, type_id: str) -> FakeBP:
        self.find_calls.append(type_id)
        return FakeBP(type_id)

    def filter(self, pattern: str) -> list[FakeBP]:
        if pattern == "walker.pedestrian.*":
            return list(self._walker_bps)
        return []


class FakeWorld:
    def __init__(self, *, next_id: int = 500) -> None:
        self._bp_lib = FakeBPLibrary()
        self._next_id = next_id
        self.actors: dict[int, FakeWalkerActor | FakeControllerActor] = {}
        self.spawn_log: list[tuple[str, object]] = []
        self.nav_calls = 0

    def get_blueprint_library(self) -> FakeBPLibrary:
        return self._bp_lib

    def get_random_location_from_navigation(self) -> FakeLoc:
        loc = FakeLoc(self.nav_calls)
        self.nav_calls += 1
        return loc

    def spawn_actor(
        self,
        bp: FakeBP,
        tf: object,
        attach_to: FakeWalkerActor | None = None,
    ) -> FakeWalkerActor | FakeControllerActor:
        actor_id = self._next_id
        self._next_id += 1
        if bp.type_id.startswith("controller."):
            assert attach_to is not None, "controller must attach to walker"
            ctrl = FakeControllerActor(actor_id, attach_to)
            self.actors[actor_id] = ctrl
            self.spawn_log.append((bp.type_id, attach_to.id))
            return ctrl
        walker = FakeWalkerActor(actor_id)
        self.actors[actor_id] = walker
        self.spawn_log.append((bp.type_id, None))
        return walker

    def get_actor(self, actor_id: int) -> FakeWalkerActor | FakeControllerActor | None:
        return self.actors.get(actor_id)


def test_walker_spawn_is_frozen() -> None:
    s = WalkerSpawn(walker_id=1, controller_id=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.walker_id = 99  # type: ignore[misc]


def test_spawn_walkers_zero_returns_empty() -> None:
    w = FakeWorld()
    assert spawn_walkers(w, num_walkers=0) == []
    assert w.actors == {}


def test_spawn_walkers_negative_raises() -> None:
    w = FakeWorld()
    with pytest.raises(ValueError, match="num_walkers must be >= 0"):
        spawn_walkers(w, num_walkers=-1)


def test_spawn_walkers_one_creates_walker_and_controller() -> None:
    w = FakeWorld()
    spawns = spawn_walkers(w, num_walkers=1)
    assert len(spawns) == 1
    s = spawns[0]
    assert s.walker_id != s.controller_id
    assert s.walker_id in w.actors
    assert s.controller_id in w.actors
    assert isinstance(w.actors[s.walker_id], FakeWalkerActor)
    assert isinstance(w.actors[s.controller_id], FakeControllerActor)


def test_spawn_walkers_controller_attached_to_walker() -> None:
    w = FakeWorld()
    spawns = spawn_walkers(w, num_walkers=1)
    s = spawns[0]
    walker = w.actors[s.walker_id]
    assert isinstance(walker, FakeWalkerActor)
    assert walker.attached_to is not None
    assert walker.attached_to.id == s.controller_id


def test_spawn_walkers_starts_controller_and_sets_destination() -> None:
    w = FakeWorld()
    spawns = spawn_walkers(w, num_walkers=1)
    s = spawns[0]
    ctrl = w.actors[s.controller_id]
    assert isinstance(ctrl, FakeControllerActor)
    assert ctrl.started is True
    assert ctrl.destination is not None


def test_spawn_walkers_uses_walker_blueprint_filter() -> None:
    w = FakeWorld()
    spawn_walkers(w, num_walkers=1)
    # filter is called for walker bp; find is called for controller bp
    assert CONTROLLER_BP in w._bp_lib.find_calls


def test_spawn_walkers_falls_back_to_find_when_filter_empty() -> None:
    w = FakeWorld()
    w._bp_lib = FakeBPLibrary(walker_bps=[])  # empty filter
    spawns = spawn_walkers(w, num_walkers=1)
    assert len(spawns) == 1
    assert WALKER_BP_FALLBACK in w._bp_lib.find_calls


def test_spawn_walkers_multiple_distinct_ids() -> None:
    w = FakeWorld()
    spawns = spawn_walkers(w, num_walkers=5)
    assert len(spawns) == 5
    walker_ids = [s.walker_id for s in spawns]
    controller_ids = [s.controller_id for s in spawns]
    assert len(set(walker_ids)) == 5
    assert len(set(controller_ids)) == 5
    assert set(walker_ids).isdisjoint(set(controller_ids))


def test_spawn_walkers_spawns_in_walker_then_controller_order() -> None:
    w = FakeWorld()
    spawn_walkers(w, num_walkers=2)
    # spawn_log entries: walker, controller, walker, controller
    types = [entry[0] for entry in w.spawn_log]
    assert types[0].startswith("walker.pedestrian")
    assert types[1] == CONTROLLER_BP
    assert types[2].startswith("walker.pedestrian")
    assert types[3] == CONTROLLER_BP


def test_destroy_walkers_stops_controllers_first() -> None:
    w = FakeWorld()
    spawns = spawn_walkers(w, num_walkers=2)
    destroy_walkers(w, spawns)
    for s in spawns:
        ctrl = w.actors[s.controller_id]
        assert isinstance(ctrl, FakeControllerActor)
        assert ctrl.stopped is True
        assert ctrl.destroyed is True
        walker = w.actors[s.walker_id]
        assert isinstance(walker, FakeWalkerActor)
        assert walker.destroyed is True


def test_destroy_walkers_empty_is_noop() -> None:
    w = FakeWorld()
    destroy_walkers(w, [])
    assert w.actors == {}


def test_destroy_walkers_missing_controller_continues() -> None:
    w = FakeWorld()
    spawns = spawn_walkers(w, num_walkers=2)
    # remove one controller from world (simulate already-destroyed)
    w.actors.pop(spawns[0].controller_id, None)
    destroy_walkers(w, spawns)
    # walker 0 still destroyed; controller 1 + walker 1 destroyed
    assert w.actors[spawns[0].walker_id].destroyed is True
    assert w.actors[spawns[1].controller_id].destroyed is True
    assert w.actors[spawns[1].walker_id].destroyed is True


def test_destroy_walkers_missing_walker_continues() -> None:
    w = FakeWorld()
    spawns = spawn_walkers(w, num_walkers=1)
    w.actors.pop(spawns[0].walker_id, None)
    destroy_walkers(w, spawns)
    assert w.actors[spawns[0].controller_id].destroyed is True


def test_destroy_walkers_controller_without_stop_method_is_ok() -> None:
    """If a controller actor lacks stop(), destroy still proceeds."""

    class NoStopController(FakeControllerActor):
        def __init__(self, actor_id: int, walker: FakeWalkerActor) -> None:
            super().__init__(actor_id, walker)
            del self.stopped  # type: ignore[attr-defined]

    w = FakeWorld()
    # Manually plant a controller without stop()
    walker_bp = w._bp_lib.find("walker.pedestrian.0001")
    walker = w.spawn_actor(walker_bp, FakeLoc(0))
    assert isinstance(walker, FakeWalkerActor)
    ctrl_id = w._next_id
    w._next_id += 1
    ctrl = NoStopController(ctrl_id, walker)
    w.actors[ctrl_id] = ctrl
    spawn = WalkerSpawn(walker_id=walker.id, controller_id=ctrl_id)
    destroy_walkers(w, [spawn])
    assert walker.destroyed is True
    assert ctrl.destroyed is True

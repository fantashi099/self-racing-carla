"""Unit tests for carla_race.collision_scoring (F7)."""
from __future__ import annotations

import threading

import pytest

from carla_race.collision_scoring import (
    COLLISION_SENSOR_BP,
    attach_collision_sensors,
    classify_hit,
    destroy_sensors,
)
from carla_race.race_state import CarState


class FakeActor:
    def __init__(self, actor_id: int, type_id: str = "vehicle.lincoln.mkz_2017") -> None:
        self.id = actor_id
        self.type_id = type_id
        self.destroyed = False
        self.listening = False
        self.callback: object | None = None

    def listen(self, cb: object) -> None:
        self.listening = True
        self.callback = cb

    def destroy(self) -> None:
        self.destroyed = True


class FakeSensor(FakeActor):
    pass


class FakeCollisionEvent:
    def __init__(self, other_type_id: str) -> None:
        self.other_actor = FakeActor(-1, other_type_id)


class FakeBP:
    def __init__(self, type_id: str) -> None:
        self.type_id = type_id


class FakeBPLibrary:
    def __init__(self) -> None:
        self.find_calls: list[str] = []

    def find(self, type_id: str) -> FakeBP:
        self.find_calls.append(type_id)
        return FakeBP(type_id)


class FakeWorld:
    def __init__(self, *, next_sensor_id: int = 700) -> None:
        self._bp_lib = FakeBPLibrary()
        self._next_sid = next_sensor_id
        self.actors: dict[int, FakeActor] = {}
        self.spawn_log: list[tuple[int, int | None]] = []  # (bp_type, attach_to_id)

    def get_blueprint_library(self) -> FakeBPLibrary:
        return self._bp_lib

    def spawn_actor(
        self,
        bp: FakeBP,
        tf: object,
        attach_to: FakeActor | None = None,
    ) -> FakeSensor:
        sid = self._next_sid
        self._next_sid += 1
        sensor = FakeSensor(sid)
        self.actors[sid] = sensor
        self.spawn_log.append((attach_to.id if attach_to is not None else None))
        return sensor

    def get_actor(self, actor_id: int) -> FakeActor | None:
        return self.actors.get(actor_id)


def test_classify_hit_walker() -> None:
    assert classify_hit("walker.pedestrian.0001") == "walker"
    assert classify_hit("walker.foo") == "walker"


def test_classify_hit_vehicle() -> None:
    assert classify_hit("vehicle.lincoln.mkz_2017") == "vehicle"
    assert classify_hit("vehicle.tesla.model3") == "vehicle"


def test_classify_hit_other() -> None:
    assert classify_hit("static.street.light") == "other"
    assert classify_hit("") == "other"
    assert classify_hit("sensor.other.collision") == "other"


def test_classify_hit_returns_str() -> None:
    assert isinstance(classify_hit("walker.x"), str)


def test_attach_sensors_returns_sensor_ids_per_car() -> None:
    w = FakeWorld()
    car1 = FakeActor(1)
    car2 = FakeActor(2)
    state1 = CarState(actor_id=1, is_player=True, color="r")
    state2 = CarState(actor_id=2, is_player=False, color="b")
    sensor_ids = attach_collision_sensors(
        w,
        car_actors={1: car1, 2: car2},
        car_states={1: state1, 2: state2},
    )
    assert set(sensor_ids.keys()) == {1, 2}
    assert sensor_ids[1] != sensor_ids[2]
    assert w._bp_lib.find_calls == [COLLISION_SENSOR_BP]


def test_attach_sensors_registers_listen_callback() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    attach_collision_sensors(w, {1: car}, {1: state})
    sensor = w.actors[w.actors.__iter__().__next__()]
    assert sensor.listening is True
    assert sensor.callback is not None


def test_attach_sensors_empty_cars_returns_empty() -> None:
    w = FakeWorld()
    sensor_ids = attach_collision_sensors(w, {}, {})
    assert sensor_ids == {}


def test_attach_sensors_uses_collision_blueprint() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    attach_collision_sensors(w, {1: car}, {1: state})
    assert w._bp_lib.find_calls == [COLLISION_SENSOR_BP]


def test_callback_increments_walker_hits() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state})
    sensor = w.actors[sensor_ids[1]]
    assert sensor.callback is not None
    sensor.callback(FakeCollisionEvent("walker.pedestrian.0001"))  # type: ignore[arg-type]
    assert state.walker_hits == 1
    assert state.car_hits == 0


def test_callback_increments_car_hits() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state})
    sensor = w.actors[sensor_ids[1]]
    sensor.callback(FakeCollisionEvent("vehicle.lincoln.mkz_2017"))  # type: ignore[arg-type]
    assert state.car_hits == 1
    assert state.walker_hits == 0


def test_callback_ignores_other_hits() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state})
    sensor = w.actors[sensor_ids[1]]
    sensor.callback(FakeCollisionEvent("static.street.light"))  # type: ignore[arg-type]
    assert state.walker_hits == 0
    assert state.car_hits == 0


def test_callback_handles_missing_other_actor() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")

    class EventNoOther:
        other_actor = None

    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state})
    sensor = w.actors[sensor_ids[1]]
    sensor.callback(EventNoOther())  # type: ignore[arg-type]
    assert state.walker_hits == 0
    assert state.car_hits == 0


def test_callback_multiple_hits_stack() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state})
    sensor = w.actors[sensor_ids[1]]
    sensor.callback(FakeCollisionEvent("walker.pedestrian.0001"))  # type: ignore[arg-type]
    sensor.callback(FakeCollisionEvent("walker.pedestrian.0002"))  # type: ignore[arg-type]
    sensor.callback(FakeCollisionEvent("vehicle.lincoln.mkz_2017"))  # type: ignore[arg-type]
    assert state.walker_hits == 2
    assert state.car_hits == 1


def test_callback_uses_provided_lock() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    lock = threading.Lock()
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state}, lock=lock)
    sensor = w.actors[sensor_ids[1]]
    sensor.callback(FakeCollisionEvent("walker.pedestrian.0001"))  # type: ignore[arg-type]
    assert state.walker_hits == 1


def test_each_car_gets_own_callback_independent_state() -> None:
    w = FakeWorld()
    car1 = FakeActor(1)
    car2 = FakeActor(2)
    state1 = CarState(actor_id=1, is_player=True, color="r")
    state2 = CarState(actor_id=2, is_player=False, color="b")
    sensor_ids = attach_collision_sensors(
        w, {1: car1, 2: car2}, {1: state1, 2: state2}
    )
    s1 = w.actors[sensor_ids[1]]
    s2 = w.actors[sensor_ids[2]]
    s1.callback(FakeCollisionEvent("walker.pedestrian.0001"))  # type: ignore[arg-type]
    s2.callback(FakeCollisionEvent("vehicle.lincoln.mkz_2017"))  # type: ignore[arg-type]
    s2.callback(FakeCollisionEvent("vehicle.lincoln.mkz_2017"))  # type: ignore[arg-type]
    assert state1.walker_hits == 1
    assert state1.car_hits == 0
    assert state2.walker_hits == 0
    assert state2.car_hits == 2


def test_destroy_sensors_destroys_each() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state})
    destroy_sensors(w, sensor_ids)
    for sid in sensor_ids.values():
        assert w.actors[sid].destroyed is True


def test_destroy_sensors_empty_is_noop() -> None:
    w = FakeWorld()
    destroy_sensors(w, {})
    assert w.actors == {}


def test_destroy_sensors_missing_actor_continues() -> None:
    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state})
    # remove one sensor from world
    sid = sensor_ids[1]
    w.actors.pop(sid, None)
    destroy_sensors(w, sensor_ids)  # should not raise


def test_attach_sensors_thread_safe_concurrent_callbacks() -> None:
    """Concurrent callbacks should not lose hits or corrupt the lock."""
    import threading as t

    w = FakeWorld()
    car = FakeActor(1)
    state = CarState(actor_id=1, is_player=True, color="r")
    lock = threading.Lock()
    sensor_ids = attach_collision_sensors(w, {1: car}, {1: state}, lock=lock)
    sensor = w.actors[sensor_ids[1]]

    def fire(n: int) -> None:
        for _ in range(n):
            sensor.callback(FakeCollisionEvent("walker.pedestrian.0001"))  # type: ignore[arg-type]

    threads = [t.Thread(target=fire, args=(100,)) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert state.walker_hits == 1000

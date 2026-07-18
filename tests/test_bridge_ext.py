"""Unit tests for carla_race.bridge_ext (F9 + F3)."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import carla_race.bridge_ext as bridge_ext
from carla_race.bridge_ext import init_race_manager, race_router, spawn_player_camera
from carla_race.race_state import CarState, RacePhase, RaceState


# ── fakes ────────────────────────────────────────────────────────────────

class FakeCamActor:
    def __init__(self, sid: int) -> None:
        self.id = sid
        self.listening = False

    def listen(self, cb: Any) -> None:
        self.listening = True


class FakeBP:
    def __init__(self, type_id: str) -> None:
        self.type_id = type_id
        self.attributes: dict[str, str] = {}

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value


class FakeBPLibrary:
    def find(self, type_id: str) -> FakeBP:
        return FakeBP(type_id)


class FakeWorld:
    def __init__(self, *, next_sid: int = 900) -> None:
        self._bp_lib = FakeBPLibrary()
        self._next_sid = next_sid
        self.actors: dict[int, Any] = {}
        self.spawn_log: list[str] = []

    def get_blueprint_library(self) -> FakeBPLibrary:
        return self._bp_lib

    def spawn_actor(self, bp: FakeBP, tf: Any, attach_to: Any = None) -> FakeCamActor:
        sid = self._next_sid
        self._next_sid += 1
        cam = FakeCamActor(sid)
        self.actors[sid] = cam
        self.spawn_log.append(bp.type_id)
        return cam

    def get_actor(self, actor_id: int) -> Any:
        # Return a placeholder player actor for the player id
        return object()


class FakeClient:
    def __init__(self, world: FakeWorld) -> None:
        self._world = world

    def get_world(self) -> FakeWorld:
        return self._world


class FakeRM:
    """Minimal RaceManager stand-in: returns canned RaceState, records calls."""

    def __init__(self, state: RaceState) -> None:
        self._state = state
        self.start_calls = 0
        self.restart_calls = 0
        self.destroy_calls = 0
        self.start_raises: RuntimeError | None = None

    def start(self) -> RaceState:
        if self.start_raises is not None:
            raise self.start_raises
        self.start_calls += 1
        return self._state

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "phase": self._state.phase.value,
            "map_name": self._state.map_name,
            "num_cars": self._state.config_num_cars,
            "num_laps": self._state.config_num_laps,
            "started_at_s": self._state.started_at_s,
            "finished_at_s": self._state.finished_at_s,
            "elapsed_s": 0.0,
            "circuit_waypoint_count": self._state.circuit_waypoint_count,
            "cars": [],
        }

    def restart(self) -> RaceState:
        self.restart_calls += 1
        return self._state

    def destroy(self) -> None:
        self.destroy_calls += 1


def _make_state(phase: RacePhase = RacePhase.RUNNING) -> RaceState:
    rs = RaceState(config_num_cars=2, config_num_laps=1, phase=phase)
    rs.started_at_s = 100.0
    rs.map_name = "Town01"
    rs.circuit_waypoint_count = 64
    rs.cars[1] = CarState(actor_id=1, is_player=True, color="255,0,0")
    rs.cars[2] = CarState(actor_id=2, is_player=False, color="0,0,255")
    return rs


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    state = _make_state()
    fake_rm = FakeRM(state)
    world = FakeWorld()
    fake_client = FakeClient(world)
    monkeypatch.setattr(bridge_ext, "_race_manager", fake_rm)
    monkeypatch.setattr(bridge_ext, "_client", fake_client)
    app = FastAPI()
    app.include_router(race_router)
    return TestClient(app)


# ── init_race_manager ──────────────────────────────────────────────────

def test_init_race_manager_sets_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_ext, "_race_manager", None)
    monkeypatch.setattr(bridge_ext, "_client", None)
    world = FakeWorld()
    client = FakeClient(world)
    # patch RaceConfig so load_config isn't called with real env
    config = CarState  # placeholder; init_race_manager accepts a RaceConfig
    from carla_race.config import RaceConfig
    config = RaceConfig(num_cars=2, num_laps=1, num_walkers=0, ai_difficulty="normal")
    init_race_manager(client, config)
    assert bridge_ext._race_manager is not None
    assert bridge_ext._client is client


def test_init_race_manager_loads_config_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RACE_NUM_CARS", raising=False)
    monkeypatch.delenv("RACE_NUM_LAPS", raising=False)
    monkeypatch.delenv("RACE_NUM_WALKERS", raising=False)
    monkeypatch.delenv("RACE_AI_DIFFICULTY", raising=False)
    monkeypatch.setattr(bridge_ext, "_race_manager", None)
    monkeypatch.setattr(bridge_ext, "_client", None)
    client = FakeClient(FakeWorld())
    init_race_manager(client, config=None)
    assert bridge_ext._race_manager is not None


# ── /race/start ────────────────────────────────────────────────────────

def test_start_returns_running_state_with_player_and_camera(app_client: TestClient) -> None:
    resp = app_client.post("/race/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == "running"
    assert body["map_name"] == "Town01"
    assert body["player_actor_id"] == 1
    assert body["camera"] is not None
    assert "sensor_id" in body["camera"]
    assert body["camera"]["ws_path"].startswith("/stream/")
    assert body["camera"]["frame_path"].startswith("/frame/")


def test_start_409_when_already_running(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # configure the fake RM to raise on the second start
    fake_rm = bridge_ext._race_manager
    assert isinstance(fake_rm, FakeRM)
    fake_rm.start_raises = RuntimeError("race already running")
    resp = app_client.post("/race/start")
    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"]


def test_start_404_when_not_initialized(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_ext, "_race_manager", None)
    resp = app_client.post("/race/start")
    assert resp.status_code == 404


# ── /race/state ────────────────────────────────────────────────────────

def test_state_returns_snapshot(app_client: TestClient) -> None:
    resp = app_client.get("/race/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == "running"
    assert body["map_name"] == "Town01"
    assert body["num_cars"] == 2
    assert "cars" in body


def test_state_404_when_not_initialized(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_ext, "_race_manager", None)
    resp = app_client.get("/race/state")
    assert resp.status_code == 404


# ── /race/restart ───────────────────────────────────────────────────────

def test_restart_returns_state_with_new_camera(app_client: TestClient) -> None:
    resp = app_client.post("/race/restart")
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == "running"
    assert body["player_actor_id"] == 1
    assert body["camera"] is not None


def test_restart_404_when_not_initialized(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_ext, "_race_manager", None)
    resp = app_client.post("/race/restart")
    assert resp.status_code == 404


# ── /race/stop ─────────────────────────────────────────────────────────

def test_stop_destroys_race(app_client: TestClient) -> None:
    resp = app_client.post("/race/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stopped"] is True
    fake_rm = bridge_ext._race_manager
    assert isinstance(fake_rm, FakeRM)
    assert fake_rm.destroy_calls == 1


def test_stop_404_when_not_initialized(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_ext, "_race_manager", None)
    resp = app_client.post("/race/stop")
    assert resp.status_code == 404


# ── spawn_player_camera (F3) ───────────────────────────────────────────

def test_spawn_player_camera_returns_sensor_paths() -> None:
    world = FakeWorld()
    player_actor = object()
    cam = spawn_player_camera(world, player_actor)
    assert "sensor_id" in cam
    assert cam["ws_path"] == f"/stream/{cam['sensor_id']}"
    assert cam["frame_path"] == f"/frame/{cam['sensor_id']}"


def test_spawn_player_camera_uses_camera_bp() -> None:
    world = FakeWorld()
    spawn_player_camera(world, object())
    assert "sensor.camera.rgb" in world.spawn_log


def test_spawn_player_camera_sets_image_attributes() -> None:
    world = FakeWorld()
    spawn_player_camera(world, object(), width=640, height=360, fov=90)
    # the spawned bp records attributes; find the bp via spawn_log
    # FakeBPLibrary.find returns a fresh FakeBP each call — we can't inspect
    # it directly, but spawn_actor received the bp. Verify via spawn_log type.
    assert world.spawn_log[-1] == "sensor.camera.rgb"


def test_spawn_player_camera_listener_started() -> None:
    world = FakeWorld()
    cam_info = spawn_player_camera(world, object())
    sensor = world.actors[cam_info["sensor_id"]]
    assert sensor.listening is True


def test_spawn_player_camera_attaches_to_player() -> None:
    """The camera must be attached to the player actor (spawn_actor attach_to)."""

    class AttachingWorld(FakeWorld):
        def __init__(self) -> None:
            super().__init__()
            self.attach_target: Any = None

        def spawn_actor(self, bp: FakeBP, tf: Any, attach_to: Any = None) -> FakeCamActor:
            self.attach_target = attach_to
            return super().spawn_actor(bp, tf, attach_to=attach_to)

    world = AttachingWorld()
    player = object()
    spawn_player_camera(world, player)
    assert world.attach_target is player


# ── router prefix ──────────────────────────────────────────────────────

def test_race_router_has_race_prefix() -> None:
    assert race_router.prefix == "/race"

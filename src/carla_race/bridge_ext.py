"""F9 — /race/* REST endpoints + F3 — 3rd-person camera on the player vehicle.

Exposes the race manager over a FastAPI router. ``init_race_manager(client,
config=None)`` constructs the singleton; mount the router in bridge.py via
``app.include_router(race_router)`` after calling ``init_race_manager``.

Endpoints:
- ``POST /race/start`` — start a race. 409 if already running. Spawns a
  3rd-person camera on the player and returns the state snapshot plus
  ``player_actor_id`` + ``camera`` (sensor_id, ws_path, frame_path).
- ``GET /race/state`` — current race state snapshot.
- ``POST /race/restart`` — destroy + start (re-picks random map). 409 if no
  race has been started yet.
- ``POST /race/stop`` — destroy the race. 404 if no race has been started.

Contract:
- ``race_router = APIRouter(prefix="/race")``
- ``init_race_manager(client, config=None) -> None``
"""
from __future__ import annotations

import sys
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException

from carla_race.config import RaceConfig, load_config
from carla_race.race_manager import RaceManager
from carla_race.race_state import RaceState

__all__ = ["init_race_manager", "race_router", "spawn_player_camera"]

race_router = APIRouter(prefix="/race")

_race_manager: RaceManager | None = None
_client: Any = None
_register_camera_fn: Any = None

CAMERA_BP = "sensor.camera.rgb"
DEFAULT_CAM_WIDTH = 800
DEFAULT_CAM_HEIGHT = 600
DEFAULT_CAM_FOV = 90
DEFAULT_CAM_X = -1.5  # behind the car
DEFAULT_CAM_Y = 0.0
DEFAULT_CAM_Z = 2.5  # above the car


def init_race_manager(
    client: Any,
    config: RaceConfig | None = None,
    register_camera: Any = None,
    carla_lock: Any = None,
) -> None:
    global _race_manager, _client, _register_camera_fn
    _client = client
    _register_camera_fn = register_camera
    if config is None:
        config = load_config()
    _race_manager = RaceManager(client, config, carla_lock=carla_lock)


def _get_rm() -> RaceManager:
    if _race_manager is None:
        raise HTTPException(status_code=404, detail="race manager not initialized")
    return _race_manager


def _get_client() -> Any:
    if _client is None:
        raise HTTPException(status_code=404, detail="race manager not initialized")
    return _client


def spawn_player_camera(
    world: Any,
    player_actor: Any,
    *,
    width: int = DEFAULT_CAM_WIDTH,
    height: int = DEFAULT_CAM_HEIGHT,
    fov: int = DEFAULT_CAM_FOV,
) -> dict[str, Any]:
    """Spawn a 3rd-person RGB camera attached to ``player_actor``. Returns
    ``{sensor_id, ws_path, frame_path}``. Mirrors bridge.py /spawn/camera but
    with 3rd-person defaults (behind + above the car)."""
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(CAMERA_BP)
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))

    transform = _make_camera_transform(DEFAULT_CAM_X, DEFAULT_CAM_Y, DEFAULT_CAM_Z)
    cam = world.spawn_actor(bp, transform, attach_to=player_actor)
    sid = cam.id
    if _register_camera_fn is not None:
        _register_camera_fn(cam, player_actor.id)
    else:
        _register_camera_callback(cam, sid)
    return {
        "sensor_id": sid,
        "ws_path": f"/stream/{sid}",
        "frame_path": f"/frame/{sid}",
    }


def _make_camera_transform(x: float, y: float, z: float) -> Any:
    try:
        import carla
    except ImportError:
        return _CamTransform(x, y, z)
    return carla.Transform(carla.Location(x=x, y=y, z=z))


class _CamLoc:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _CamTransform:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.location = _CamLoc(x, y, z)
        self.rotation = None


def _register_camera_callback(cam: Any, sid: int) -> None:
    """Wire a no-op listener so the sensor is live. The real JPEG encoding
    + broadcast is handled by bridge.py's camera infrastructure when mounted
    there; bridge_ext only ensures the sensor exists and is listening."""
    listen = getattr(cam, "listen", None)
    if listen is None:
        return
    # bridge.py owns the actual frame buffer; here we just start the sensor.
    listen(lambda _frame: None)


@race_router.post("/start")
def start_race() -> dict[str, Any]:
    rm = _get_rm()
    rs: RaceState
    try:
        rs = rm.start()
    except RuntimeError as exc:
        msg = str(exc)
        traceback.print_exc(file=sys.stderr)
        if "already running" not in msg:
            raise HTTPException(status_code=500, detail=f"start failed: {msg}") from exc
        current = getattr(rm, "current_state", None)
        existing = current() if callable(current) else None
        if existing is None:
            raise HTTPException(status_code=409, detail=msg) from exc
        rs = existing
    client = _get_client()
    world = client.get_world()
    player = rs.player()
    player_actor = world.get_actor(player.actor_id)
    camera = spawn_player_camera(world, player_actor) if player_actor is not None else None
    snap = rm.state_snapshot()
    return {
        **snap,
        "player_actor_id": player.actor_id,
        "camera": camera,
    }


@race_router.get("/state")
def get_race_state() -> dict[str, Any]:
    rm = _get_rm()
    try:
        rm.tick()
    except Exception as exc:
        print(f"[race] tick failed: {exc!r}", file=sys.stderr, flush=True)
    return rm.state_snapshot()


@race_router.post("/restart")
def restart_race() -> dict[str, Any]:
    rm = _get_rm()
    try:
        rs = rm.restart()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    client = _get_client()
    world = client.get_world()
    player = rs.player()
    player_actor = world.get_actor(player.actor_id)
    camera = spawn_player_camera(world, player_actor) if player_actor is not None else None
    snap = rm.state_snapshot()
    return {
        **snap,
        "player_actor_id": player.actor_id,
        "camera": camera,
    }


@race_router.post("/stop")
def stop_race() -> dict[str, Any]:
    rm = _get_rm()
    rm.destroy()
    return {"stopped": True}

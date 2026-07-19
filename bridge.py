#!/usr/bin/env python3
"""
CARLA HTTP/WS Bridge — runs next to CARLA server.
Talks CARLA RPC on localhost, exposes REST + WebSocket to thin local clients.
HTTP tunnels cleanly through ngrok/bore (no CARLA RPC port-range issue).

Run (server side, same host as CARLA):
  pip install fastapi uvicorn websockets pillow numpy carla==0.9.15
  python3 bridge.py
  # or with custom CARLA host:
  CARLA_HOST=localhost CARLA_PORT=2000 python3 bridge.py

Tunnel the bridge port (8000) out — any HTTP tunnel works.
"""
import asyncio
import io
import os
import threading
from pathlib import Path
from typing import Dict, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn
from dotenv import load_dotenv

import carla

# ── config ───────────────────────────────────────────────
load_dotenv()
CARLA_HOST = os.environ.get("CARLA_HOST", "localhost")
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8000"))

# ── state ─────────────────────────────────────────────────
app = FastAPI(title="CARLA Bridge")
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(30.0)

vehicles: Dict[int, carla.Actor] = {}
sensors: Dict[int, dict] = {}          # sensor_id -> {actor, subscribers, latest_jpeg}
vehicle_to_sensor: Dict[int, int] = {} # vehicle_id -> primary camera sensor_id
_loop: asyncio.AbstractEventLoop = None
_carla_lock = threading.RLock()

# F2 grid state — set of currently-spawned race cars (player + AI). Lives
# outside `vehicles` because the grid is spawned/destroyed as a unit and
# the /drive minimap polls it for AI car positions.
_race_grid: list = []  # list[CarSpawn]
_race_grid_player_id: int | None = None


# ── cross-thread broadcast (CARLA sensor cb runs on its own thread) ──
def _broadcast(sensor_id: int, jpeg: bytes):
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_async_broadcast(sensor_id, jpeg), _loop)


async def _async_broadcast(sensor_id: int, jpeg: bytes):
    subs = sensors.get(sensor_id, {}).get("subscribers", set())
    full = []
    for q in subs:
        try:
            q.put_nowait(jpeg)
        except asyncio.QueueFull:
            full.append(q)
    for q in full:
        subs.discard(q)


@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_event_loop()


def _register_race_camera(cam, vehicle_id: int) -> None:
    sid = cam.id
    sensors[sid] = {"actor": cam, "subscribers": set(), "latest_jpeg": None}
    vehicle_to_sensor[vehicle_id] = sid
    print(f"[race cam] registered sensor {sid} for vehicle {vehicle_id}", flush=True)

    def cb(image):
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                (image.height, image.width, 4))[:, :, :3]
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=50)
            jpeg = buf.getvalue()
            sensors[sid]["latest_jpeg"] = jpeg
            _broadcast(sid, jpeg)
        except Exception as e:
            print(f"[race cam] cb error sensor {sid}: {e!r}", flush=True)

    cam.listen(cb)


# ── browser canvas client (Option 3) ───────────────────────
_DRIVE_HTML = Path(__file__).with_name("index.html")
_RACE_HTML = Path(__file__).with_name("race.html")


@app.get("/drive", response_class=HTMLResponse)
def drive_page():
    return HTMLResponse(_DRIVE_HTML.read_text(), media_type="text/html")


@app.get("/manual", response_class=HTMLResponse)
def manual_page():
    return HTMLResponse(_DRIVE_HTML.read_text(), media_type="text/html")


# F1 — random map load. Live-verifiable in /drive via the "Random Map" button.
# Endpoints here are F1-only; race endpoints (F9/F10) stay disabled until each
# is live-verified with supervisor sign-off. See PROGRESS.md
# "Session 2026-07-19 (later) — RESET + supervisor rule".
from carla_race.map_pool import pick_and_load  # noqa: E402
from carla_race.vehicle_grid import destroy_grid, spawn_grid  # noqa: E402


@app.get("/map/current")
def get_current_map():
    w = client.get_world()
    return {"map": w.get_map().name.rsplit("/", 1)[-1]}


@app.get("/map/roads")
def get_map_roads():
    """Road network skeleton for the /drive minimap.

    Returns bounds (world coords) + line segments from map.get_topology()
    (entry→exit waypoint per road). Client scales to its minimap canvas.
    """
    with _carla_lock:
        m = client.get_world().get_map()
        topo = m.get_topology()
    segs = []
    xs = []
    ys = []
    for entry, exit_ in topo:
        el = entry.transform.location
        xl = exit_.transform.location
        segs.append([round(el.x, 2), round(el.y, 2),
                      round(xl.x, 2), round(xl.y, 2)])
        xs.extend((el.x, xl.x))
        ys.extend((el.y, xl.y))
    if not xs or not ys:
        return {"bounds": {"min_x": 0, "min_y": 0, "max_x": 1, "max_y": 1}, "segments": []}
    return {
        "bounds": {
            "min_x": round(min(xs), 2), "max_x": round(max(xs), 2),
            "min_y": round(min(ys), 2), "max_y": round(max(ys), 2),
        },
        "segments": segs,
    }


@app.post("/map/random")
def post_random_map():
    """F1: pick a random map (RACE_EXCLUDE_MAPS filter) and load it.

    load_world destroys all current actors, so the client must re-spawn its
    vehicle + camera after calling this (the /drive page does that on success).
    """
    with _carla_lock:
        name, _carla_map = pick_and_load(client)
        # load_world destroys all actors — drop the grid bookkeeping too.
        global _race_grid, _race_grid_player_id
        _race_grid = []
        _race_grid_player_id = None
    return {"map": name}


def _clear_vehicles_near_spawn_points(world, num_cars: int, radius: float = 3.0) -> int:
    """Destroy any vehicle within `radius` meters of the first num_cars spawn
    points. CARLA state persists across bridge restarts, so leftover vehicles
    from a prior session collide with spawn_grid. Best-effort, returns count.
    """
    spawn_pts = world.get_map().get_spawn_points()
    pts = spawn_pts[:num_cars]
    if not pts:
        return 0
    destroyed = 0
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            t = actor.get_transform()
            ax, ay = t.location.x, t.location.y
        except Exception:
            continue
        for sp in pts:
            dx = ax - sp.location.x
            dy = ay - sp.location.y
            if dx * dx + dy * dy <= radius * radius:
                try:
                    actor.destroy()
                    destroyed += 1
                except Exception:
                    pass
                break
    return destroyed


@app.post("/race/grid")
def post_race_grid(body: dict = None):
    """F2: spawn the race grid (1 player + N-1 AI) at distinct spawn points.

    num_cars defaults to RACE_NUM_CARS env (or 10), override via body {num_cars}.
    Returns the player's actor_id (the car the /drive client drives) plus the
    full grid with spawn positions so the minimap can plot all cars.
    """
    body = body or {}
    default_n = int(os.environ.get("RACE_NUM_CARS", "10"))
    num_cars = int(body.get("num_cars", default_n))
    if num_cars < 1:
        return JSONResponse({"error": "num_cars must be >= 1"}, status_code=400)

    global _race_grid, _race_grid_player_id
    with _carla_lock:
        world = client.get_world()
        # Idempotent: if a grid is already spawned, destroy it first so
        # re-clicking Spawn Grid gives a fresh grid instead of a 409 or a
        # collision with the old grid's actors.
        if _race_grid:
            destroy_grid(world, _race_grid)
            for s in _race_grid:
                vehicles.pop(s.actor_id, None)
            _race_grid = []
            _race_grid_player_id = None
        _clear_vehicles_near_spawn_points(world, num_cars)
        try:
            spawns = spawn_grid(world, num_cars=num_cars)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except RuntimeError as e:
            # CARLA collision at a spawn point — return 503 so the client can
            # retry (destroy grid / reload map) rather than a 500 traceback.
            return JSONResponse(
                {"error": f"spawn_grid collided: {e}", "num_cars": num_cars},
                status_code=503,
            )
        _race_grid = list(spawns)
        _race_grid_player_id = spawns[0].actor_id if spawns else None
        cars = []
        for s in spawns:
            actor = world.get_actor(s.actor_id)
            tf = actor.get_transform() if actor is not None else None
            cars.append({
                "actor_id": s.actor_id,
                "is_player": s.is_player,
                "color": s.color,
                "spawn_index": s.spawn_index,
                "x": round(tf.location.x, 2) if tf else 0.0,
                "y": round(tf.location.y, 2) if tf else 0.0,
                "yaw": round(tf.rotation.yaw, 1) if tf else 0.0,
            })
    return {"player_id": _race_grid_player_id, "cars": cars}


@app.get("/race/grid")
def get_race_grid():
    """F2: live positions of every grid car for minimap polling."""
    if not _race_grid:
        return {"cars": []}
    out = []
    with _carla_lock:
        world = client.get_world()
        for s in _race_grid:
            actor = world.get_actor(s.actor_id)
            if actor is None:
                continue
            tf = actor.get_transform()
            out.append({
                "actor_id": s.actor_id,
                "is_player": s.is_player,
                "color": s.color,
                "x": round(tf.location.x, 2),
                "y": round(tf.location.y, 2),
                "yaw": round(tf.rotation.yaw, 1),
            })
    return {"cars": out}


@app.post("/race/grid/destroy")
def post_race_grid_destroy():
    """F2: destroy the current grid (best-effort, missing actors ignored)."""
    global _race_grid, _race_grid_player_id
    if not _race_grid:
        return {"destroyed": 0}
    with _carla_lock:
        world = client.get_world()
        count = len(_race_grid)
        destroy_grid(world, _race_grid)
        # also drop from the manual vehicles dict if present
        for s in _race_grid:
            vehicles.pop(s.actor_id, None)
        _race_grid = []
        _race_grid_player_id = None
    return {"destroyed": count}


# Race mode endpoints (F9/F10) disabled 2026-07-19: re-enable per-feature only
# after live CARLA verification with supervisor sign-off. See PROGRESS.md
# "Session 2026-07-19 (later) — RESET + supervisor rule".
#
# @app.get("/race", response_class=HTMLResponse)
# def race_page():
#     return HTMLResponse(_RACE_HTML.read_text(), media_type="text/html")
#
# # ── race mode router (F9) ─────────────────────────────────
# from carla_race.bridge_ext import init_race_manager, race_router  # noqa: E402
#
# init_race_manager(client, register_camera=_register_race_camera, carla_lock=_carla_lock)
# app.include_router(race_router)
#
# print(
#     "[bridge] RACE_EXCLUDE_MAPS=" + os.environ.get("RACE_EXCLUDE_MAPS", "<unset>"),
#     flush=True,
# )


# ── REST endpoints ────────────────────────────────────────
@app.get("/")
def health():
    w = client.get_world()
    return {
        "status": "ok",
        "carla": f"{CARLA_HOST}:{CARLA_PORT}",
        "map": w.get_map().name,
    }


@app.get("/world")
def get_world():
    w = client.get_world()
    snap = w.get_snapshot()
    actors = w.get_actors()
    return {
        "map": w.get_map().name,
        "frame": snap.frame,
        "elapsed": snap.timestamp.elapsed_seconds,
        "actor_count": len(actors),
        "vehicles": [a.id for a in actors.filter("vehicle.*")],
    }


@app.post("/spawn/vehicle")
def spawn_vehicle(body: dict):
    w = client.get_world()
    bp_lib = w.get_blueprint_library()
    bp = bp_lib.find(body.get("blueprint", "vehicle.lincoln.mkz_2017"))
    if "color" in body:
        bp.set_attribute("color", body["color"])
    spawn_pts = w.get_map().get_spawn_points()
    if not spawn_pts:
        return JSONResponse({"error": "no spawn points on this map"}, status_code=500)
    start_idx = body.get("spawn_index", 0)
    if start_idx >= len(spawn_pts):
        start_idx = 0
    # Retry on collision: walk every spawn point once starting at start_idx.
    # CARLA raises RuntimeError("Spawn failed because of collision ...") when
    # the spot is occupied (previous vehicle, debris, etc.).
    last_err = ""
    for offset in range(len(spawn_pts)):
        idx = (start_idx + offset) % len(spawn_pts)
        try:
            actor = w.spawn_actor(bp, spawn_pts[idx])
            vehicles[actor.id] = actor
            return {"id": actor.id, "type_id": actor.type_id, "spawn_index": idx}
        except RuntimeError as e:
            last_err = str(e)
            continue
    return JSONResponse(
        {"error": f"all {len(spawn_pts)} spawn points collided: {last_err}"},
        status_code=503,
    )


@app.post("/spawn/camera")
def spawn_camera(body: dict):
    w = client.get_world()
    bp_lib = w.get_blueprint_library()
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(body.get("width", 800)))
    bp.set_attribute("image_size_y", str(body.get("height", 600)))
    bp.set_attribute("fov", str(body.get("fov", 90)))

    attach_to = body.get("attach_to")
    if attach_to is None:
        return JSONResponse({"error": "attach_to required (vehicle id)"}, status_code=400)
    parent = vehicles.get(attach_to)
    if parent is None:
        # F2: grid cars aren't in `vehicles` — resolve via world.get_actor
        # (same fallback /step uses) so the camera can attach to the grid player.
        try:
            with _carla_lock:
                parent = w.get_actor(attach_to)
        except Exception as exc:
            return JSONResponse(
                {"error": f"could not resolve vehicle {attach_to}: {exc!r}"},
                status_code=503,
            )
    if parent is None:
        return JSONResponse({"error": f"vehicle {attach_to} not found"}, status_code=404)

    transform = carla.Transform(carla.Location(
        x=float(body.get("x", 1.5)),
        y=float(body.get("y", 0.0)),
        z=float(body.get("z", 2.5)),
    ))
    cam = w.spawn_actor(bp, transform, attach_to=parent)
    sid = cam.id
    sensors[sid] = {"actor": cam, "subscribers": set(), "latest_jpeg": None}

    def cb(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4))[:, :, :3]
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        jpeg = buf.getvalue()
        sensors[sid]["latest_jpeg"] = jpeg
        _broadcast(sid, jpeg)

    cam.listen(cb)
    vehicle_to_sensor[attach_to] = sid
    return {"sensor_id": sid, "ws": f"/stream/{sid}", "frame": f"/frame/{sid}"}


@app.get("/frame/{sensor_id}")
def get_frame(sensor_id: int):
    s = sensors.get(sensor_id)
    if s is None:
        return JSONResponse({"error": "no such sensor"}, status_code=404)
    jpeg = s.get("latest_jpeg")
    if jpeg is None:
        return JSONResponse({"error": "no frame yet"}, status_code=503)
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/step/{vid}")
def step(vid: int, body: dict):
    """Combined: apply vehicle control + return latest camera JPEG + speed header.
    One round-trip per cycle instead of two. Speed in `X-Speed-Kmh` header.
    Race cars (spawned by race_manager, not /spawn/vehicle) resolve via world.get_actor."""
    v = vehicles.get(vid)
    if v is None:
        try:
            with _carla_lock:
                w = client.get_world()
                v = w.get_actor(vid)
        except Exception as exc:
            print(f"[step] CARLA unreachable for vid {vid}: {exc!r}", flush=True)
            return JSONResponse(
                {"error": "simulator unreachable", "vid": vid}, status_code=503,
            )
        if v is None:
            return JSONResponse({"error": f"vehicle {vid} not found"}, status_code=404)
    try:
        ctrl = carla.VehicleControl(
            throttle=float(body.get("throttle", 0.0)),
            steer=float(body.get("steer", 0.0)),
            brake=float(body.get("brake", 0.0)),
            reverse=bool(body.get("reverse", False)),
            hand_brake=bool(body.get("hand_brake", False)),
        )
        with _carla_lock:
            v.apply_control(ctrl)
            speed_kmh = round(v.get_velocity().length() * 3.6, 1)
            tf = v.get_transform()
            pos_x = round(tf.location.x, 2)
            pos_y = round(tf.location.y, 2)
            yaw_deg = round(tf.rotation.yaw, 1)
    except Exception as exc:
        print(f"[step] control/velocity failed for vid {vid}: {exc!r}", flush=True)
        return JSONResponse(
            {"error": "simulator unreachable", "vid": vid}, status_code=503,
        )

    sid = vehicle_to_sensor.get(vid)
    jpeg = sensors.get(sid, {}).get("latest_jpeg") if sid else None
    pos_headers = {
        "X-Speed-Kmh": str(speed_kmh),
        "X-Pos-X": str(pos_x),
        "X-Pos-Y": str(pos_y),
        "X-Pos-Yaw": str(yaw_deg),
        "Access-Control-Expose-Headers": "X-Speed-Kmh, X-Pos-X, X-Pos-Y, X-Pos-Yaw",
    }
    if jpeg is None:
        return JSONResponse(
            {"error": "no frame yet", "speed_kmh": speed_kmh}, status_code=503,
            headers=pos_headers,
        )
    return Response(content=jpeg, media_type="image/jpeg", headers=pos_headers)


@app.post("/control/vehicle/{vid}")
def control_vehicle(vid: int, body: dict):
    v = vehicles.get(vid)
    if v is None:
        return JSONResponse({"error": f"vehicle {vid} not found"}, status_code=404)
    ctrl = carla.VehicleControl(
        throttle=float(body.get("throttle", 0.0)),
        steer=float(body.get("steer", 0.0)),
        brake=float(body.get("brake", 0.0)),
        reverse=bool(body.get("reverse", False)),
        hand_brake=bool(body.get("hand_brake", False)),
    )
    v.apply_control(ctrl)
    speed = v.get_velocity().length()  # m/s
    return {"ok": True, "speed_ms": round(speed, 2), "speed_kmh": round(speed * 3.6, 1)}


@app.post("/destroy/{actor_id}")
def destroy(actor_id: int):
    w = client.get_world()
    actor = w.get_actor(actor_id)
    if actor is not None:
        actor.destroy()
    vehicles.pop(actor_id, None)
    sensors.pop(actor_id, None)
    # clean vehicle_to_sensor both directions
    vehicle_to_sensor.pop(actor_id, None)
    to_drop = [v for v, s in vehicle_to_sensor.items() if s == actor_id]
    for v in to_drop:
        vehicle_to_sensor.pop(v, None)
    return {"destroyed": actor_id}


@app.post("/tick")
def tick():
    w = client.get_world()
    w.tick()
    return {"ok": True}


# ── WS streaming ──────────────────────────────────────────
@app.websocket("/stream/{sensor_id}")
async def stream(ws: WebSocket, sensor_id: int):
    await ws.accept()
    if sensor_id not in sensors:
        await ws.close(code=1008, reason="no such sensor")
        return
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    sensors[sensor_id]["subscribers"].add(q)
    try:
        while True:
            frame = await q.get()
            try:
                await ws.send_bytes(frame)
            except (WebSocketDisconnect, RuntimeError, Exception):
                break
    except (WebSocketDisconnect, RuntimeError, Exception):
        pass
    finally:
        sensors[sensor_id]["subscribers"].discard(q)
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host=BRIDGE_HOST, port=BRIDGE_PORT)

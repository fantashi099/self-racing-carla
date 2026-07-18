#!/usr/bin/env python3
"""
Pygame CARLA remote driver. Single combined HTTP round-trip per cycle:
  POST /step/{vid}  →  apply control + return JPEG + speed header.

Usage:
  pip install pygame requests
  python3 pygame_client.py https://b921-115-73-214-46.ngrok-free.app

Controls (click window first to focus):
  W / ↑       throttle forward
  S / ↓       brake + reverse
  A / ←       steer left
  D / →       steer right
  SPACE       hand brake
  R           respawn (destroy + new vehicle + camera)
  ESC         quit + cleanup
"""
import io
import sys
import threading
import time

import pygame
import requests

BRIDGE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
NUM_WORKERS = int(__import__("os").environ.get("CARLA_WORKERS", "2"))

SESSION = requests.Session()
# pool sized for NUM_WORKERS + headroom; keep-alive so TCP/TLS handshake is paid once
from requests.adapters import HTTPAdapter
_poolext = HTTPAdapter(pool_connections=NUM_WORKERS + 4, pool_maxsize=NUM_WORKERS + 4)
SESSION.mount("http://", _poolext)
SESSION.mount("https://", _poolext)
SESSION.headers.update({"ngrok-skip-browser-warning": "true"})

# ── shared state ──────────────────────────────────────────
latest_frame: bytes | None = None
frame_lock = threading.Lock()
control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0,
           "reverse": False, "hand_brake": False}
control_lock = threading.Lock()
speed_kmh = 0.0
running = True
vid = None
sid = None
step_pause = threading.Event()       # set during respawn to avoid 404 spam
rtt_ms = 0.0
fps_counter = {"frames": 0, "last_print": time.time(), "fps": 0.0}


# ── concurrent step workers ─────────────────────────────
# N workers = N requests in flight at all times.
# Throughput ≈ N / RTT. 1 worker = 20 FPS @ 50ms RTT, 2 = 40 FPS, 3 = 60 FPS.
# Each worker uses its own keep-alive connection from the Session pool.
def step_worker(wid: int):
    global latest_frame, speed_kmh, rtt_ms
    while running:
        if step_pause.is_set() or vid is None:
            time.sleep(0.02)
            continue
        with control_lock:
            c = dict(control)
        t0 = time.time()
        try:
            r = SESSION.post(f"{BRIDGE}/step/{vid}", json=c, timeout=2)
            rtt_ms = (time.time() - t0) * 1000
            if r.status_code == 200:
                with frame_lock:
                    latest_frame = r.content
                speed_kmh = float(r.headers.get("X-Speed-Kmh", 0))
                fps_counter["frames"] += 1
                now = time.time()
                if now - fps_counter["last_print"] >= 2.0:
                    fps_counter["fps"] = fps_counter["frames"] / (now - fps_counter["last_print"])
                    fps_counter["frames"] = 0
                    fps_counter["last_print"] = now
            else:
                print(f"[w{wid}] http {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[w{wid}] error: {e}")
        # tight loop — RTT itself paces us; no sleep keeps pipeline full


# ── spawn helpers ────────────────────────────────────────
def spawn_vehicle_and_camera():
    global vid, sid
    v = SESSION.post(f"{BRIDGE}/spawn/vehicle",
                     json={"color": "255,0,0"}).json()
    vid = v["id"]
    print("Vehicle:", v)
    c = SESSION.post(
        f"{BRIDGE}/spawn/camera",
        json={"attach_to": vid, "width": 640, "height": 360, "fov": 90},
    ).json()
    sid = c["sensor_id"]
    print("Camera:", c)


def respawn():
    global vid, sid
    step_pause.set()
    print("respawning...")
    old_vid, old_sid = vid, sid
    for aid in (old_sid, old_vid):
        try:
            SESSION.post(f"{BRIDGE}/destroy/{aid}", timeout=3)
        except Exception:
            pass
    spawn_vehicle_and_camera()
    step_pause.clear()


# ── main ─────────────────────────────────────────────────
def main():
    global running

    pygame.init()
    pygame.display.set_caption("CARLA Remote Drive — click to focus")
    screen = pygame.display.set_mode((640, 360))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    small = pygame.font.SysFont("monospace", 12)

    spawn_vehicle_and_camera()
    for w in range(NUM_WORKERS):
        threading.Thread(target=step_worker, args=(w,), daemon=True).start()

    print("\n=== READY ===")
    print("Click the pygame window, then use WASD / arrows.")
    print("ESC to quit.\n")

    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        respawn()

            keys = pygame.key.get_pressed()
            with control_lock:
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    control["throttle"] = 0.7
                    control["brake"] = 0.0
                    control["reverse"] = False
                elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                    control["throttle"] = 0.0
                    control["brake"] = 0.4
                    control["reverse"] = True
                else:
                    control["throttle"] = 0.0
                    control["brake"] = 0.0
                    control["reverse"] = False
                if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    control["steer"] = -0.5
                elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    control["steer"] = 0.5
                else:
                    control["steer"] = 0.0
                control["hand_brake"] = keys[pygame.K_SPACE]

            # render
            with frame_lock:
                f = latest_frame
            if f is not None:
                try:
                    img = pygame.image.load(io.BytesIO(f))
                    screen.blit(img, (0, 0))
                except Exception:
                    pass
            else:
                screen.fill((20, 20, 20))
                msg = font.render("waiting for camera...", True, (200, 200, 200))
                screen.blit(msg, (220, 170))

            # HUD
            hud1 = font.render(
                f"WASD | R reset | ESC quit | v{vid}",
                True, (255, 255, 0))
            hud2 = font.render(f"{speed_kmh:5.1f} km/h", True, (0, 255, 0))
            hud3 = small.render(
                f"fps ~{fps_counter['fps']:.1f}  rtt {rtt_ms:.0f}ms  w={NUM_WORKERS}",
                True, (180, 180, 180))
            screen.blit(hud1, (10, 10))
            screen.blit(hud2, (550, 10))
            screen.blit(hud3, (10, 340))

            pygame.display.flip()
            clock.tick(30)
    finally:
        running = False
        time.sleep(0.3)
        for aid in (sid, vid):
            try:
                SESSION.post(f"{BRIDGE}/destroy/{aid}", timeout=3)
            except Exception:
                pass
        pygame.quit()
        print("cleaned up")


if __name__ == "__main__":
    main()

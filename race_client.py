#!/usr/bin/env python3
"""Pygame CARLA race client. Polls /step/{player_actor_id} for control + frame
and /race/state for the race HUD (lap, clock, splits, score, position,
leaderboard).

Usage:
  pip install pygame requests
  python3 race_client.py [BRIDGE_URL] [--race]

  BRIDGE_URL defaults to http://localhost:8000.
  --race forces race mode; without it, the client auto-detects by polling
  /race/state — if phase != "init", it enters race mode; otherwise it falls
  back to the manual-drive spawn flow (delegates to pygame_client.py).

Controls (click window first to focus):
  W / Up       throttle forward
  S / Down     brake + reverse
  A / Left     steer left
  D / Right    steer right
  SPACE        hand brake
  R            restart race (POST /race/restart)
  ESC          quit + POST /race/stop
"""
import io
import json
import sys
import threading
import time

import pygame
import requests

BRIDGE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "http://localhost:8000"
FORCE_RACE = "--race" in sys.argv[1:]

SESSION = requests.Session()
SESSION.headers.update({"ngrok-skip-browser-warning": "true"})

# ── shared state ──────────────────────────────────────────
latest_frame: bytes | None = None
frame_lock = threading.Lock()
control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0,
           "reverse": False, "hand_brake": False}
control_lock = threading.Lock()
speed_kmh = 0.0
running = True
vid: int | None = None
sid: int | None = None
race_state: dict | None = None
step_pause = threading.Event()
rtt_ms = 0.0
fps_counter = {"frames": 0, "last_print": time.time(), "fps": 0.0}


# ── race helpers ─────────────────────────────────────────
def start_race() -> dict:
    """POST /race/start, set vid/sid, return the response body."""
    global vid, sid
    r = SESSION.post(f"{BRIDGE}/race/start", timeout=10)
    r.raise_for_status()
    body = r.json()
    vid = body["player_actor_id"]
    cam = body.get("camera") or {}
    sid = cam.get("sensor_id")
    print(f"[race] started map={body.get('map_name')} player={vid} cam={sid}")
    return body


def restart_race() -> dict:
    global vid, sid
    step_pause.set()
    try:
        r = SESSION.post(f"{BRIDGE}/race/restart", timeout=10)
        r.raise_for_status()
        body = r.json()
        vid = body["player_actor_id"]
        cam = body.get("camera") or {}
        sid = cam.get("sensor_id")
        print(f"[race] restarted map={body.get('map_name')} player={vid}")
        return body
    finally:
        step_pause.clear()


def stop_race() -> None:
    try:
        SESSION.post(f"{BRIDGE}/race/stop", timeout=5)
    except Exception as e:
        print(f"[race] stop error: {e}")


def detect_race_mode() -> bool:
    """Auto-detect race mode: True if /race/state exists and phase != 'init'."""
    if FORCE_RACE:
        return True
    try:
        r = SESSION.get(f"{BRIDGE}/race/state", timeout=3)
        if r.ok:
            body = r.json()
            return body.get("phase") != "init"
    except Exception:
        pass
    return False


# ── step workers (same pattern as pygame_client) ─────────
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
            elif r.status_code == 404:
                print(f"[w{wid}] vehicle lost; restarting race")
                step_pause.set()
                try:
                    restart_race()
                except Exception as e:
                    print(f"[w{wid}] restart failed: {e}")
        except Exception as e:
            print(f"[w{wid}] error: {e}")


def state_worker():
    """Poll /race/state at 4Hz for HUD + finish detection."""
    global race_state
    while running:
        if vid is None:
            time.sleep(0.1)
            continue
        try:
            r = SESSION.get(f"{BRIDGE}/race/state", timeout=2)
            if r.ok:
                race_state = r.json()
        except Exception:
            pass
        time.sleep(0.25)


# ── HUD rendering ─────────────────────────────────────────
def fmt_time(s: float | None) -> str:
    if s is None or s < 0:
        return "00:00.0"
    m = int(s // 60)
    sec = int(s % 60)
    tenths = int((s * 10) % 10)
    return f"{m:02d}:{sec:02d}.{tenths}"


def render_hud(screen, font, small):
    if not race_state:
        return
    state = race_state
    me = next((c for c in state.get("cars", []) if c.get("is_player")), None)
    if me is not None:
        lap = f"Lap {me.get('laps_finished', 0)}/{state.get('num_laps', 0)}"
        screen.blit(font.render(lap, True, (255, 255, 0)), (10, 10))
        clock_s = fmt_time(state.get("elapsed_s", 0.0))
        screen.blit(font.render(clock_s, True, (255, 215, 0)), (10, 30))
        score = f"Score {me.get('score', 0)}"
        screen.blit(font.render(score, True, (255, 255, 255)), (10, 50))
        pos = me.get("finish_position")
        pos_str = f"Pos {pos}/{state.get('num_cars', 0)}" if pos is not None else f"Pos ?/{state.get('num_cars', 0)}"
        screen.blit(font.render(pos_str, True, (0, 255, 200)), (10, 70))
    # leaderboard sidebar
    cars = state.get("cars", [])
    y = 10
    for c in cars:
        color = (255, 215, 0) if c.get("is_player") else (200, 200, 200)
        p = c.get("finish_position")
        p_str = str(p) if p is not None else "-"
        label = f"{p_str:>2} {'YOU' if c.get('is_player') else 'AI '} L{c.get('laps_finished', 0)}"
        screen.blit(small.render(label, True, color), (520, y))
        y += 14
    # phase banner
    phase = state.get("phase", "init")
    if phase == "finished":
        msg = "FINISHED"
        if me is not None and me.get("finish_position") is not None:
            msg += f" — P{me['finish_position']}"
        screen.blit(font.render(msg, True, (0, 255, 100)), (250, 200))


# ── main ─────────────────────────────────────────────────
def main():
    global running

    if not detect_race_mode():
        print("Not in race mode; use --race or start a race via /race/start.")
        print("Falling back is not implemented; pass --race to force.")
        sys.exit(2)

    pygame.init()
    pygame.display.set_caption("CARLA Race — click to focus")
    screen = pygame.display.set_mode((640, 360))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    small = pygame.font.SysFont("monospace", 12)

    try:
        start_race()
    except Exception as e:
        print(f"[race] start failed: {e}")
        sys.exit(1)

    NUM_WORKERS = 2
    for w in range(NUM_WORKERS):
        threading.Thread(target=step_worker, args=(w,), daemon=True).start()
    threading.Thread(target=state_worker, daemon=True).start()

    print("\n=== RACE READY ===")
    print("Click the pygame window, then use WASD / arrows.")
    print("R to restart race, ESC to quit.\n")

    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        try:
                            restart_race()
                        except Exception as e:
                            print(f"restart failed: {e}")

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
                screen.blit(font.render("waiting for camera...", True, (200, 200, 200)), (220, 170))

            render_hud(screen, font, small)

            hud3 = small.render(
                f"fps ~{fps_counter['fps']:.1f}  rtt {rtt_ms:.0f}ms  v{vid}",
                True, (180, 180, 180))
            screen.blit(hud3, (10, 340))

            pygame.display.flip()
            clock.tick(30)
    finally:
        running = False
        time.sleep(0.3)
        stop_race()
        pygame.quit()
        print("cleaned up")


if __name__ == "__main__":
    main()

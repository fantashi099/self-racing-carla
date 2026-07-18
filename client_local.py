#!/usr/bin/env python3
"""
Thin local CARLA client — talks to bridge.py over HTTP + WebSocket.
No `carla` pip package needed locally. Works through any HTTP tunnel.

Usage:
  pip install requests websockets pillow
  python3 client_local.py http://<bridge-url>          # e.g. http://localhost:8000
                                                       # or https://<ngrok>.ngrok.io
"""
import asyncio
import sys
import requests
import websockets

BRIDGE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
WS_BASE = BRIDGE.replace("http://", "ws://").replace("https://", "wss://")


def main():
    # 1. health / map
    print("Health:", requests.get(f"{BRIDGE}/").json())

    # 2. spawn vehicle
    v = requests.post(f"{BRIDGE}/spawn/vehicle", json={"color": "255,0,0"}).json()
    print("Vehicle:", v)
    vid = v["id"]

    # 3. spawn camera attached to vehicle
    c = requests.post(
        f"{BRIDGE}/spawn/camera",
        json={"attach_to": vid, "width": 800, "height": 600, "fov": 90},
    ).json()
    print("Camera:", c)
    sid = c["sensor_id"]

    # 4. stream 5 frames
    asyncio.run(stream(sid))

    # 5. cleanup
    requests.post(f"{BRIDGE}/destroy/{sid}").json()
    requests.post(f"{BRIDGE}/destroy/{vid}").json()
    print("done")


async def stream(sensor_id: int, n: int = 5):
    async with websockets.connect(
        f"{WS_BASE}/stream/{sensor_id}", max_size=None
    ) as ws:
        for i in range(n):
            data = await ws.recv()
            path = f"frame_{i:02d}.jpg"
            with open(path, "wb") as f:
                f.write(data)
            print(f"saved {path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()

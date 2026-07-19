#!/usr/bin/env python3
"""L2 integration smoke for implemented race features against live CARLA.

Per-feature slices are appended here as features are live-verified with the
supervisor.

Contract:
- Read CARLA_HOST / CARLA_PORT from env (default localhost:2000).
- Auto-skip with exit 0 if CARLA unreachable or the `carla` package is missing.
- F1 step: ``carla_race.map_pool.pick_and_load`` → assert non-empty map name,
  print the available-maps list + the excluded set + the picked map so the
  supervisor can confirm the choice is real and not an assumption.
- F4 step: spawn a two-car grid, configure the non-player car through
  TrafficManager with a circuit path, enable autopilot, and assert movement.

Run:
    CARLA_HOST=localhost CARLA_PORT=2000 python scripts/integration_race.py
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _skip(reason: str) -> int:
    print(f"[SKIP] {reason}", file=sys.stderr)
    return 0


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def _basename(map_name: str) -> str:
    return map_name.rsplit("/", 1)[-1]


def _distance(a: Any, b: Any) -> float:
    return float(a.distance(b))


def _verify_f4(client: Any) -> None:
    from carla_race.ai_driver import setup_ai_cars
    from carla_race.circuit import build_circuit
    from carla_race.vehicle_grid import destroy_grid, spawn_grid

    world = client.get_world()
    spawns = spawn_grid(world, num_cars=2, player_color="0,255,0")
    tm_port = int(os.environ.get("RACE_TM_PORT", "8001"))
    actors: dict[int, Any] = {}
    ai_ids: list[int] = []

    try:
        actors = {
            spawn.actor_id: world.get_actor(spawn.actor_id)
            for spawn in spawns
        }
        actors = {actor_id: actor for actor_id, actor in actors.items() if actor is not None}
        player_id = next(spawn.actor_id for spawn in spawns if spawn.is_player)
        ai_ids = [spawn.actor_id for spawn in spawns if not spawn.is_player]
        tm = client.get_trafficmanager(tm_port)
        circuit = build_circuit(world.get_map())
        if not circuit:
            raise RuntimeError("F4 build_circuit returned an empty path")
        setup_ai_cars(
            tm,
            actors,
            world.get_map(),
            circuit,
            difficulty=os.environ.get("RACE_AI_DIFFICULTY", "normal"),
            player_actor_id=player_id,
        )
        initial = {actor_id: actors[actor_id].get_location() for actor_id in ai_ids}
        for actor_id in ai_ids:
            actors[actor_id].set_autopilot(True, tm_port)

        deadline = time.monotonic() + 30.0
        moved = 0.0
        while time.monotonic() < deadline:
            world.wait_for_tick(2.0)
            moved = max(
                _distance(initial[actor_id], actors[actor_id].get_location())
                for actor_id in ai_ids
            )
            if moved >= 2.0:
                break
        if moved < 2.0:
            raise RuntimeError(f"F4 AI car moved only {moved:.2f}m in 30s")
        print(
            f"[OK] F4 integration: {len(ai_ids)} AI car configured on "
            f"{len(circuit)}-point circuit and moved {moved:.2f}m"
        )
    finally:
        for actor_id in ai_ids:
            actor = actors.get(actor_id)
            if actor is not None:
                with contextlib.suppress(Exception):
                    actor.set_autopilot(False, tm_port)
        destroy_grid(world, spawns)
        with contextlib.suppress(Exception):
            world.tick()


def main() -> int:
    host = os.environ.get("CARLA_HOST", "localhost")
    port = int(os.environ.get("CARLA_PORT", "2000"))

    try:
        import carla
    except ImportError:
        return _skip(f"carla package not installed (CARLA_HOST={host}:{port})")

    try:
        client = carla.Client(host, port)
        client.set_timeout(60.0)
        client.get_server_version()
    except Exception as exc:
        return _skip(f"CARLA unreachable at {host}:{port}: {exc!r}")

    # Import after carla is confirmed present so the package's TYPE_CHECKING
    # guard never trips at runtime.
    from carla_race.map_pool import _exclude_from_env, pick_and_load

    # Show the supervisor the real available-maps list + exclude set so the
    # pick is verifiable, not assumed.
    try:
        raw_maps = client.get_available_maps()
    except Exception:
        print("[FAIL] client.get_available_maps() raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    available = sorted({_basename(m) for m in raw_maps})
    excluded = sorted(set(_exclude_from_env()))
    print(f"[info] CARLA server version: {client.get_server_version()}")
    print(f"[info] available maps ({len(available)}): {available}")
    print(f"[info] RACE_EXCLUDE_MAPS excluded ({len(excluded)}): {excluded}")

    pool = [m for m in available if m not in excluded]
    if not pool:
        return _fail(
            f"no maps available after excluding {excluded}; available={available}"
        )

    try:
        name, carla_map = pick_and_load(client)
    except Exception:
        print("[FAIL] pick_and_load raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    if not name or not getattr(carla_map, "name", ""):
        return _fail(f"empty map returned: name={name!r}")

    # Confirm the loaded world actually reports the picked map (not a stale
    # world handle from before load_world).
    try:
        loaded_name = _basename(client.get_world().get_map().name)
    except Exception:
        print("[FAIL] post-load get_map() raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    if loaded_name != name:
        return _fail(
            f"loaded map mismatch: pick_and_load chose {name!r} but world reports "
            f"{loaded_name!r}"
        )

    print(
        f"[OK] F1 integration: picked + loaded map {name!r} "
        f"(map.name={carla_map.name!r}, world confirms {loaded_name!r})"
    )
    try:
        _verify_f4(client)
    except Exception:
        print("[FAIL] F4 integration raised:", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

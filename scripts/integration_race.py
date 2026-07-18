#!/usr/bin/env python3
"""L2 integration smoke for F1 (map_pool) + F2 (vehicle_grid) against live CARLA.

Contract:
- Read CARLA_HOST / CARLA_PORT from env (default localhost:2000).
- Auto-skip with exit 0 if CARLA unreachable or the `carla` package is missing.
- F1 step: ``carla_race.map_pool.pick_and_load`` → assert non-empty map name.
- F2 step: ``carla_race.vehicle_grid.spawn_grid`` on the loaded world with
  ``num_cars=2`` → assert 2 spawns, player=index 0, distinct actor_ids + colors,
  spawn indices [0, 1], actors resolvable in world. Then ``destroy_grid`` →
  assert no exception.

This is the additive F1+F2 slice of the full ``integration_race.py`` contract
(2-car 1-lap race via RaceManager). Later features append their own L2 steps
here; existing steps must keep passing. The 2-car race flow stays the end
goal — per-feature slices build toward it.

Run:
    CARLA_HOST=localhost CARLA_PORT=2000 python scripts/integration_race.py
"""
from __future__ import annotations

import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()


def _skip(reason: str) -> int:
    print(f"[SKIP] {reason}", file=sys.stderr)
    return 0


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


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
        # Touch the server — cheapest reachability probe.
        client.get_server_version()
    except Exception as exc:
        return _skip(f"CARLA unreachable at {host}:{port}: {exc!r}")

    # Import after carla is confirmed present so the package's TYPE_CHECKING
    # guard never trips at runtime.
    from carla_race.map_pool import pick_and_load
    from carla_race.vehicle_grid import destroy_grid, spawn_grid

    try:
        name, carla_map = pick_and_load(client)
    except Exception:
        print("[FAIL] pick_and_load raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    if not name or not getattr(carla_map, "name", ""):
        return _fail(f"empty map returned: name={name!r}")

    print(f"[OK] F1 integration: loaded map {name!r} (map.name={carla_map.name!r})")

    # F2: spawn 2 cars on the loaded world. Use a fresh world handle so the
    # map loaded by F1 is the one we spawn into (load_world may not refresh
    # an existing carla.World reference).
    world = client.get_world()
    spawns = None
    try:
        spawns = spawn_grid(world, num_cars=2)
    except Exception:
        print("[FAIL] spawn_grid raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    if len(spawns) != 2:
        return _fail(f"expected 2 spawns, got {len(spawns)}")
    if not spawns[0].is_player or spawns[0].spawn_index != 0:
        return _fail(
            f"player spawn wrong: is_player={spawns[0].is_player} "
            f"spawn_index={spawns[0].spawn_index}"
        )
    if spawns[1].is_player or spawns[1].spawn_index != 1:
        return _fail(
            f"ai spawn wrong: is_player={spawns[1].is_player} "
            f"spawn_index={spawns[1].spawn_index}"
        )
    actor_ids = [s.actor_id for s in spawns]
    if len(set(actor_ids)) != 2:
        return _fail(f"actor_ids not distinct: {actor_ids}")
    colors = [s.color for s in spawns]
    if len(set(colors)) != 2:
        return _fail(f"colors not distinct: {colors}")

    resolved = [world.get_actor(aid) for aid in actor_ids]
    if any(a is None for a in resolved):
        return _fail(f"some spawned actors not resolvable: ids={actor_ids}")

    print(
        f"[OK] F2 integration: spawned {len(spawns)} cars "
        f"player_id={spawns[0].actor_id} ai_id={spawns[1].actor_id} "
        f"colors={colors}"
    )

    try:
        destroy_grid(world, spawns)
    except Exception:
        print("[FAIL] destroy_grid raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("[OK] F2 integration: destroy_grid completed without error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

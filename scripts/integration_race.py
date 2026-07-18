#!/usr/bin/env python3
"""L2 integration smoke for F1-F9 against live CARLA.

Contract:
- Read CARLA_HOST / CARLA_PORT from env (default localhost:2000).
- Auto-skip with exit 0 if CARLA unreachable or the `carla` package is missing.
- F1 step: ``carla_race.map_pool.pick_and_load`` → assert non-empty map name.
- F2 step: ``carla_race.vehicle_grid.spawn_grid`` + ``destroy_grid`` on the
  loaded world with ``num_cars=2`` → assert 2 spawns, distinct ids/colors,
  actors resolvable.
- F3-F9 step: full 2-car 1-lap race via ``RaceManager`` (map + grid + circuit
  + lap detection + collision sensors + AI autopilot + state snapshot).
  Player car gets autopilot + circuit path so it finishes without a human
  driver. Asserts phase=FINISHED, both cars have finish_position in [1, 2],
  state_snapshot shape. Timeout 120s.

Run:
    CARLA_HOST=localhost CARLA_PORT=2000 python scripts/integration_race.py
"""
from __future__ import annotations

import os
import sys
import time
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

    # F3-F9 integration: full 2-car 1-lap race via RaceManager. Additive —
    # exercises F1 (map), F2 (vehicle_grid), F3 (camera via /race/start), F4
    # (ai_driver + autopilot), F6 (circuit + lap_tracker), F7 (collision
    # sensors), F9 (race_manager FSM). F5 walkers skipped (num_walkers=0)
    # for speed; F8 scoring is pure and unit-tested. The player car has no
    # human driver in this smoke, so we enable TM autopilot + set_path on
    # it too — both cars then follow the circuit and complete the lap.
    from carla_race.config import RaceConfig
    from carla_race.race_manager import RaceManager

    race_cfg = RaceConfig(
        num_cars=2,
        num_laps=1,
        num_walkers=0,
        ai_difficulty="normal",
    )
    rm = RaceManager(client, race_cfg)
    rs = None
    try:
        rs = rm.start()
    except Exception:
        print("[FAIL] RaceManager.start() raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    player = rs.player()
    print(
        f"[OK] F3-F9 integration: race started map={rs.map_name!r} "
        f"player_id={player.actor_id} cars={len(rs.cars)} "
        f"waypoints={rs.circuit_waypoint_count}"
    )

    # Enable autopilot + circuit path on the player too so it finishes
    # without a human driver. race_manager.start() already enabled autopilot
    # on the AI car; the player is normally human-driven. set_path takes an
    # Actor + list of Locations (verified at L2 — Waypoints raise a
    # converter TypeError; Locations work).
    from carla_race.ai_driver import _circuit_to_path

    world = client.get_world()
    tm = client.get_trafficmanager(8000)
    try:
        tm_port = int(tm.get_port())
    except Exception:
        tm_port = 8000
    player_actor = world.get_actor(player.actor_id)
    player_path = _circuit_to_path(rm._circuit)
    if player_path:
        print(f"[debug] player_path locations={len(player_path)} type={type(player_path[0]).__name__}")
    if player_actor is not None and player_path:
        try:
            player_actor.set_autopilot(True, tm_port)
            tm.set_path(player_actor, player_path)
            print(f"[OK] player {player.actor_id} autopilot + path enabled")
        except Exception as e:
            print(f"[WARN] could not enable player autopilot: {e!r}", file=sys.stderr)

    # Tick until FINISHED or 120s timeout.
    deadline = time.monotonic() + 120.0
    last_phase = None
    while time.monotonic() < deadline:
        rs = rm.tick()
        if rs is None:
            break
        if rs.phase != last_phase:
            last_phase = rs.phase
            print(
                f"[race] phase={rs.phase.value} "
                f"player laps={player.laps_finished} wp={player.waypoint_index} "
                f"finish={player.finish_position}"
            )
        if rs.phase.name == "FINISHED":
            break
        time.sleep(0.5)

    if rs is None or rs.phase.name != "FINISHED":
        print(
            f"[FAIL] race did not finish within 120s "
            f"(phase={rs.phase.value if rs else None})",
            file=sys.stderr,
        )
        rm.destroy()
        return 1

    # Both cars must have a finish_position.
    unfinished = [c for c in rs.cars.values() if c.finish_position is None]
    if unfinished:
        print(
            f"[FAIL] {len(unfinished)} cars without finish_position: "
            f"{[c.actor_id for c in unfinished]}",
            file=sys.stderr,
        )
        rm.destroy()
        return 1

    positions = sorted(c.finish_position for c in rs.cars.values())
    if positions != [1, 2]:
        print(
            f"[FAIL] expected finish positions [1, 2], got {positions}",
            file=sys.stderr,
        )
        rm.destroy()
        return 1

    print(
        f"[OK] F3-F9 integration: race FINISHED positions={positions} "
        f"player_pos={player.finish_position} player_laps={player.laps_finished} "
        f"player_hits=w{player.walker_hits}/c{player.car_hits}"
    )

    # state_snapshot shape sanity (F9).
    snap = rm.state_snapshot()
    if snap.get("phase") != "finished":
        print(f"[FAIL] snapshot phase={snap.get('phase')!r}", file=sys.stderr)
        rm.destroy()
        return 1
    if len(snap.get("cars", [])) != 2:
        print(f"[FAIL] snapshot cars={len(snap.get('cars', []))}", file=sys.stderr)
        rm.destroy()
        return 1
    print("[OK] F9 integration: state_snapshot shape ok")

    rm.destroy()
    print("[OK] F3-F9 integration: RaceManager.destroy() completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

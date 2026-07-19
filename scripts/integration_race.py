#!/usr/bin/env python3
"""L2 integration smoke for F1 against live CARLA.

F1 scope only. Per-feature slices get appended here as later features are
live-verified with the supervisor (see PROGRESS.md "Session 2026-07-19
(later) — RESET + supervisor rule").

Contract:
- Read CARLA_HOST / CARLA_PORT from env (default localhost:2000).
- Auto-skip with exit 0 if CARLA unreachable or the `carla` package is missing.
- F1 step: ``carla_race.map_pool.pick_and_load`` → assert non-empty map name,
  print the available-maps list + the excluded set + the picked map so the
  supervisor can confirm the choice is real and not an assumption.

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


def _basename(map_name: str) -> str:
    return map_name.rsplit("/", 1)[-1]


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

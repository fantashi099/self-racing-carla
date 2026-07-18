#!/usr/bin/env python3
"""L2 integration smoke for F1 (map_pool) against a live CARLA server.

Contract (PROGRESS.md § Module contracts / L2):
- Read CARLA_HOST / CARLA_PORT from env (default localhost:2000).
- Auto-skip with exit 0 if CARLA unreachable or the `carla` package is missing.
- Otherwise: call ``carla_race.map_pool.pick_and_load`` and assert a non-empty
  map name is returned.

This is the F1-scoped slice of the full ``integration_race.py`` contract
(2-car 1-lap race via RaceManager). The 2-car race logic will be appended here
as later features (config, race_state, ..., race_manager) land. For now this
gives F1 a real L2 signal: the map pool actually loads a world on live CARLA.

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


def main() -> int:
    host = os.environ.get("CARLA_HOST", "localhost")
    port = int(os.environ.get("CARLA_PORT", "2000"))

    try:
        import carla
    except ImportError:
        return _skip(f"carla package not installed (CARLA_HOST={host}:{port})")

    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        # Touch the server — cheapest reachability probe.
        client.get_server_version()
    except Exception as exc:
        return _skip(f"CARLA unreachable at {host}:{port}: {exc!r}")

    # Import after carla is confirmed present so the package's TYPE_CHECKING
    # guard never trips at runtime.
    from carla_race.map_pool import pick_and_load

    try:
        name, carla_map = pick_and_load(client)
    except Exception:
        print("[FAIL] pick_and_load raised:", file=sys.stderr)
        traceback.print_exc()
        return 1

    if not name or not getattr(carla_map, "name", ""):
        print(f"[FAIL] empty map returned: name={name!r}", file=sys.stderr)
        return 1

    print(f"[OK] F1 integration: loaded map {name!r} (map.name={carla_map.name!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

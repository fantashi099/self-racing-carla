"""F1 — random map pool + load.

Picks a random map from ``client.get_available_maps()``, filtered by the
``RACE_EXCLUDE_MAPS`` env var (comma-separated basenames) and an optional
explicit ``exclude`` list. CARLA map names look like
``/Game/Carla/Maps/Town01``; we filter on the basename (``Town01``) so callers
can pass short names.

Contract:
- ``random_map(client, exclude=(), *, rng=None) -> str``
- ``load_map(client, name) -> carla.Map``
- ``pick_and_load(client, exclude=(), *, rng=None) -> tuple[str, carla.Map]``

CARLA is only imported under ``TYPE_CHECKING`` so unit tests run without the
``carla`` pip package. The mock objects in ``tests/test_map_pool.py``
structurally satisfy the small surface we touch (``get_available_maps``,
``load_world``, ``name``).
"""
from __future__ import annotations

import os
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla

__all__ = ["load_map", "pick_and_load", "random_map"]


def _exclude_from_env() -> tuple[str, ...]:
    raw = os.environ.get("RACE_EXCLUDE_MAPS", "")
    return tuple(tok.strip() for tok in raw.split(",") if tok.strip())


def _basename(map_name: str) -> str:
    # "/Game/Carla/Maps/Town01" -> "Town01"; bare "Town01" -> "Town01"
    return map_name.rsplit("/", 1)[-1]


def random_map(
    client: carla.Client,
    exclude: Sequence[str] = (),
    *,
    rng: random.Random | None = None,
) -> str:
    """Pick a random map basename from the client's available maps.

    Combines the explicit ``exclude`` list with ``RACE_EXCLUDE_MAPS`` env.
    Raises ``RuntimeError`` if the pool is empty after filtering.
    """
    available = [_basename(m) for m in client.get_available_maps()]
    excluded = set(exclude) | set(_exclude_from_env())
    pool = [m for m in available if m not in excluded]
    if not pool:
        raise RuntimeError(
            f"no maps available after excluding {sorted(excluded)}; "
            f"available={sorted(available)}"
        )
    if rng is not None:
        return rng.choice(pool)
    return random.choice(pool)


def load_map(client: carla.Client, name: str) -> carla.Map:
    """Load a world by map basename and return its ``carla.Map``."""
    world = client.load_world(name)
    return world.get_map()


def pick_and_load(
    client: carla.Client,
    exclude: Sequence[str] = (),
    *,
    rng: random.Random | None = None,
) -> tuple[str, carla.Map]:
    """Pick a random map and load it. Returns ``(basename, carla.Map)``."""
    name = random_map(client, exclude, rng=rng)
    return name, load_map(client, name)

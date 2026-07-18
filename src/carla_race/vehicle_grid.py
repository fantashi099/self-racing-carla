"""F2 — spawn N race vehicles at distinct spawn points.

Spawns ``num_cars`` vehicles (1 player at spawn index 0 + AI cars at indices
1..N-1) using the first ``num_cars`` spawn points of the loaded map. Player
color is caller-supplied (defaults to red); AI cars take distinct colors from
``GRID_COLORS`` (10 distinct ``"R,G,B"`` strings), skipping whatever matches
the player color.

Contract:
- ``CarSpawn(actor_id, is_player, color, spawn_index)`` — frozen dataclass
- ``GRID_COLORS`` — tuple of 10 distinct ``"R,G,B"`` strings
- ``spawn_grid(world, *, num_cars, player_color) -> list[CarSpawn]``
- ``destroy_grid(world, spawns) -> None``

CARLA is only imported under ``TYPE_CHECKING`` so unit tests run without the
``carla`` pip package. Tests structurally satisfy the surface touched here:
``world.get_blueprint_library()``, ``world.get_map().get_spawn_points()``,
``world.spawn_actor(bp, transform)``, ``world.get_actor(id)``, ``actor.id``,
``actor.destroy()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla

__all__ = ["GRID_COLORS", "CarSpawn", "destroy_grid", "spawn_grid"]

DEFAULT_BLUEPRINT = "vehicle.lincoln.mkz_2017"

# 10 distinct "R,G,B" strings. Index 0 is the player default (red) and is
# overridden by ``player_color`` at spawn time; AI cars pull from the rest.
GRID_COLORS: tuple[str, ...] = (
    "255,0,0",
    "0,0,255",
    "0,255,0",
    "255,255,0",
    "255,0,255",
    "0,255,255",
    "255,128,0",
    "128,0,255",
    "0,128,255",
    "255,255,255",
)


@dataclass(frozen=True)
class CarSpawn:
    actor_id: int
    is_player: bool
    color: str
    spawn_index: int


def spawn_grid(
    world: carla.World,
    *,
    num_cars: int,
    player_color: str = "255,0,0",
) -> list[CarSpawn]:
    """Spawn ``num_cars`` vehicles at the first N map spawn points.

    Player is at spawn index 0 with ``player_color``; AI cars fill indices
    1..N-1 with distinct colors drawn from ``GRID_COLORS`` minus any color
    equal to ``player_color``. Returns spawns in spawn-index order (player
    first).

    Raises ``ValueError`` if ``num_cars`` < 1, exceeds available spawn
    points, or exceeds the number of distinct AI colors available.
    """
    if num_cars < 1:
        raise ValueError(f"num_cars must be >= 1, got {num_cars}")

    spawn_pts = world.get_map().get_spawn_points()
    if num_cars > len(spawn_pts):
        raise ValueError(
            f"num_cars={num_cars} exceeds spawn points available={len(spawn_pts)}"
        )

    ai_colors = [c for c in GRID_COLORS if c != player_color]
    if num_cars - 1 > len(ai_colors):
        raise ValueError(
            f"need {num_cars - 1} distinct AI colors, only {len(ai_colors)} available"
        )

    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(DEFAULT_BLUEPRINT)

    spawns: list[CarSpawn] = []
    for i in range(num_cars):
        is_player = i == 0
        color = player_color if is_player else ai_colors[i - 1]
        bp.set_attribute("color", color)
        actor = world.spawn_actor(bp, spawn_pts[i])
        spawns.append(
            CarSpawn(
                actor_id=actor.id,
                is_player=is_player,
                color=color,
                spawn_index=i,
            )
        )
    return spawns


def destroy_grid(world: carla.World, spawns: list[CarSpawn]) -> None:
    """Destroy every vehicle in ``spawns``. Best-effort: missing actors ignored."""
    for s in spawns:
        actor = world.get_actor(s.actor_id)
        if actor is not None:
            actor.destroy()

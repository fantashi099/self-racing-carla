"""F5 — spawn walkers + walker AI controllers.

Spawns ``num_walkers`` pedestrian actors each with an attached
``controller.ai.walker`` AI controller, then starts each controller and gives
it a random navigation destination so pedestrians actually move. Destroy
stops controllers first (so they release the walker) before destroying the
walker actors themselves.

Contract:
- ``WalkerSpawn(walker_id, controller_id)`` — frozen record.
- ``spawn_walkers(world, *, num_walkers) -> list[WalkerSpawn]``: 2-step per
  walker — spawn pedestrian blueprint at a random navigation location, then
  spawn ``controller.ai.walker`` attached to it, then ``controller.start()``
  and ``controller.go_to_location(world.get_random_location_from_navigation())``.
  Raises ``ValueError`` if ``num_walkers < 0``.
- ``destroy_walkers(world, walkers) -> None``: stop + destroy controllers
  first, then destroy walkers. Best-effort: missing actors ignored.

No ``carla`` import — the world object is passed in by race_manager and
treated as opaque. Walker blueprint is chosen via
``bp_lib.filter('walker.pedestrian.*')`` + ``random.choice``, falling back to
``bp_lib.find('walker.pedestrian.0001')`` if the filter is empty.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

__all__ = ["WalkerSpawn", "destroy_walkers", "spawn_walkers"]

WALKER_BP_FILTER = "walker.pedestrian.*"
WALKER_BP_FALLBACK = "walker.pedestrian.0001"
CONTROLLER_BP = "controller.ai.walker"


@dataclass(frozen=True)
class WalkerSpawn:
    walker_id: int
    controller_id: int


def _pick_walker_bp(bp_lib: Any) -> Any:
    candidates = bp_lib.filter(WALKER_BP_FILTER)
    if candidates:
        return random.choice(list(candidates))
    return bp_lib.find(WALKER_BP_FALLBACK)


def spawn_walkers(world: Any, *, num_walkers: int) -> list[WalkerSpawn]:
    if num_walkers < 0:
        raise ValueError(f"num_walkers must be >= 0, got {num_walkers}")
    if num_walkers == 0:
        return []

    bp_lib = world.get_blueprint_library()
    walker_bp = _pick_walker_bp(bp_lib)
    controller_bp = bp_lib.find(CONTROLLER_BP)

    spawns: list[WalkerSpawn] = []
    spawned = 0
    attempts = 0
    max_attempts = num_walkers * 5
    while spawned < num_walkers and attempts < max_attempts:
        attempts += 1
        start_loc = world.get_random_location_from_navigation()
        start_tf = _make_transform(start_loc)
        try:
            walker = world.spawn_actor(walker_bp, start_tf)
        except RuntimeError as exc:
            print(f"[traffic] walker spawn retry {attempts}: {exc}", flush=True)
            continue
        try:
            controller = world.spawn_actor(controller_bp, start_tf, attach_to=walker)
        except RuntimeError as exc:
            print(f"[traffic] controller spawn failed: {exc}; destroying walker", flush=True)
            walker.destroy()
            continue
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        spawns.append(WalkerSpawn(walker_id=walker.id, controller_id=controller.id))
        spawned += 1
    if spawned < num_walkers:
        print(
            f"[traffic] spawned {spawned}/{num_walkers} walkers after {attempts} attempts",
            flush=True,
        )
    return spawns


def _make_transform(location: Any) -> Any:
    """Build a transform from a location without importing ``carla``.

    Real CARLA wants ``carla.Transform(location)``. The race_manager passes a
    real ``carla.World``, but this module never imports ``carla`` — we build
    the transform via ``world``'s module by calling ``carla.Transform`` if
    reachable, else fall back to a lightweight wrapper with ``location`` +
    ``rotation`` attributes. The walker spawn only uses ``location`` so a
    bare location-as-transform works on CARLA too (its ``spawn_actor`` reads
    ``transform.location``).
    """
    try:
        import carla
    except ImportError:
        return _LocTransform(location)
    return carla.Transform(location)


class _LocTransform:
    def __init__(self, location: Any) -> None:
        self.location = location
        self.rotation = None


def destroy_walkers(world: Any, walkers: list[WalkerSpawn]) -> None:
    for spawn in walkers:
        controller = world.get_actor(spawn.controller_id)
        if controller is not None:
            stop = getattr(controller, "stop", None)
            if stop is not None:
                stop()
            controller.destroy()
        walker = world.get_actor(spawn.walker_id)
        if walker is not None:
            walker.destroy()

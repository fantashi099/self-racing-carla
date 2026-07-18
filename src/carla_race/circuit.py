"""F6 — build a closed racing loop (circuit) from map topology.

The circuit is an ordered list of ``carla.Transform`` objects forming a
closed loop the player + AI cars follow. Used by:
- ``lap_tracker.update_car_progress`` (F6) to detect start-line crossings.
- ``ai_driver.setup_ai_cars`` (F4) via ``TrafficManager.set_path``.

Contract:
- ``build_circuit(map_obj, *, num_waypoints=64) -> list[carla.Transform]``:
  build a closed loop from ``map.get_topology()`` (longest cycle found via
  bounded DFS) or fall back to ``map.get_spawn_points()``. Resample to
  ``num_waypoints`` entries. Raises ``RuntimeError`` if neither source
  yields a non-empty loop.
- ``start_line_transform(circuit) -> carla.Transform``: ``circuit[0]``.

CARLA is only imported under ``TYPE_CHECKING`` so unit tests run without the
``carla`` pip package. Tests satisfy the surface structurally: a waypoint
has ``.transform``; a transform is opaque (we just carry it through).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import carla

__all__ = ["DEFAULT_NUM_WAYPOINTS", "build_circuit", "start_line_transform"]

DEFAULT_NUM_WAYPOINTS = 64
_CYCLE_SEARCH_BUDGET = 5000


def _adjacency(topology: list[tuple[Any, Any]]) -> tuple[dict[int, list[int]], dict[int, Any]]:
    adj: dict[int, list[int]] = {}
    nodes: dict[int, Any] = {}
    for wp_start, wp_end in topology:
        s, e = id(wp_start), id(wp_end)
        adj.setdefault(s, []).append(e)
        nodes[s] = wp_start
        nodes[e] = wp_end
    return adj, nodes


def _find_cycle_from(
    adj: dict[int, list[int]],
    start: int,
    budget: int,
) -> list[int] | None:
    """Longest-try DFS cycle from ``start`` back to ``start``. Bounded."""
    best: list[int] | None = None
    visited: set[int] = set()
    path: list[int] = []
    iters = 0

    def dfs(node: int) -> None:
        nonlocal best, iters
        if iters >= budget:
            return
        iters += 1
        path.append(node)
        visited.add(node)
        for nxt in adj.get(node, []):
            if nxt == start and len(path) >= 1:
                # closed cycle: path is the cycle
                if best is None or len(path) > len(best):
                    best = list(path)
                continue
            if nxt in visited:
                continue
            dfs(nxt)
        path.pop()
        visited.discard(node)

    dfs(start)
    return best


def _longest_cycle(topology: list[tuple[Any, Any]]) -> list[Any] | None:
    adj, nodes = _adjacency(topology)
    if not adj:
        return None
    budget_per_start = max(100, _CYCLE_SEARCH_BUDGET // max(1, len(adj)))
    best_ids: list[int] | None = None
    for start in list(adj.keys()):
        cycle = _find_cycle_from(adj, start, budget_per_start)
        if cycle and (best_ids is None or len(cycle) > len(best_ids)):
            best_ids = cycle
        if best_ids and len(best_ids) >= _CYCLE_SEARCH_BUDGET:
            break
    if not best_ids:
        return None
    return [nodes[nid] for nid in best_ids]


def _resample(transforms: list[Any], num_waypoints: int) -> list[Any]:
    if num_waypoints <= 0:
        raise ValueError(f"num_waypoints must be > 0, got {num_waypoints}")
    n = len(transforms)
    if n == 0:
        return []
    if n == num_waypoints:
        return list(transforms)
    if n > num_waypoints:
        return [transforms[int(i * n / num_waypoints)] for i in range(num_waypoints)]
    # fewer than requested: repeat the loop to fill, preserving closure
    out: list[Any] = []
    while len(out) < num_waypoints:
        for t in transforms:
            if len(out) >= num_waypoints:
                break
            out.append(t)
    return out


def build_circuit(
    map_obj: carla.Map,
    *,
    num_waypoints: int = DEFAULT_NUM_WAYPOINTS,
) -> list[carla.Transform]:
    """Build an ordered closed loop of ``num_waypoints`` transforms.

    Prefers a cycle from ``map.get_topology()``; falls back to
    ``map.get_spawn_points()`` if no cycle is found. Raises ``RuntimeError``
    if both sources are empty.
    """
    if num_waypoints <= 0:
        raise ValueError(f"num_waypoints must be > 0, got {num_waypoints}")

    topology = map_obj.get_topology()
    if topology:
        cycle = _longest_cycle(list(topology))
        if cycle:
            transforms = [getattr(wp, "transform", wp) for wp in cycle]
            if transforms:
                return _resample(transforms, num_waypoints)

    spawn_pts = map_obj.get_spawn_points()
    if spawn_pts:
        return _resample(list(spawn_pts), num_waypoints)

    raise RuntimeError("map has no topology cycle and no spawn points")


def start_line_transform(circuit: list[carla.Transform]) -> carla.Transform:
    """Return the start/finish line transform (``circuit[0]``)."""
    if not circuit:
        raise ValueError("circuit is empty")
    return circuit[0]

"""Unit tests for carla_race.circuit (F6 prereq)."""
from __future__ import annotations

import pytest

from carla_race.circuit import (
    DEFAULT_NUM_WAYPOINTS,
    build_circuit,
    start_line_transform,
)


class FakeTransform:
    def __init__(self, idx: int) -> None:
        self.idx = idx


class FakeWaypoint:
    def __init__(self, idx: int) -> None:
        self.transform = FakeTransform(idx)


class FakeMap:
    def __init__(
        self,
        topology: list[tuple[FakeWaypoint, FakeWaypoint]] | None = None,
        spawn_points: list[FakeTransform] | None = None,
    ) -> None:
        self._topology = topology or []
        self._spawn = spawn_points or []

    def get_topology(self) -> list[tuple[FakeWaypoint, FakeWaypoint]]:
        return list(self._topology)

    def get_spawn_points(self) -> list[FakeTransform]:
        return list(self._spawn)


def _triangle_topology() -> list[tuple[FakeWaypoint, FakeWaypoint]]:
    a, b, c = FakeWaypoint(1), FakeWaypoint(2), FakeWaypoint(3)
    return [(a, b), (b, c), (c, a)]


def _two_disjoint_cycles() -> list[tuple[FakeWaypoint, FakeWaypoint]]:
    # triangle + square (longer)
    a, b, c = FakeWaypoint(1), FakeWaypoint(2), FakeWaypoint(3)
    p, q, r, s = FakeWaypoint(4), FakeWaypoint(5), FakeWaypoint(6), FakeWaypoint(7)
    return [(a, b), (b, c), (c, a), (p, q), (q, r), (r, s), (s, p)]


def test_build_circuit_from_topology_triangle() -> None:
    m = FakeMap(topology=_triangle_topology())
    circuit = build_circuit(m, num_waypoints=3)
    assert len(circuit) == 3
    assert all(isinstance(t, FakeTransform) for t in circuit)


def test_build_circuit_resamples_down_to_num_waypoints() -> None:
    m = FakeMap(topology=_two_disjoint_cycles())  # longest = square (4 nodes)
    circuit = build_circuit(m, num_waypoints=10)
    assert len(circuit) == 10


def test_build_circuit_resamples_up_when_too_few() -> None:
    m = FakeMap(topology=_triangle_topology())  # 3 nodes
    circuit = build_circuit(m, num_waypoints=64)
    assert len(circuit) == 64


def test_build_circuit_resamples_down_when_too_many() -> None:
    # build a 10-cycle topology
    wps = [FakeWaypoint(i) for i in range(10)]
    topo = [(wps[i], wps[(i + 1) % 10]) for i in range(10)]
    m = FakeMap(topology=topo)
    circuit = build_circuit(m, num_waypoints=4)
    assert len(circuit) == 4


def test_build_circuit_default_num_waypoints() -> None:
    m = FakeMap(topology=_triangle_topology())
    circuit = build_circuit(m)
    assert len(circuit) == DEFAULT_NUM_WAYPOINTS


def test_build_circuit_falls_back_to_spawn_points_when_no_topology() -> None:
    spawn = [FakeTransform(i) for i in range(5)]
    m = FakeMap(topology=[], spawn_points=spawn)
    circuit = build_circuit(m, num_waypoints=5)
    assert len(circuit) == 5
    assert [t.idx for t in circuit] == [0, 1, 2, 3, 4]


def test_build_circuit_falls_back_to_spawn_points_when_no_cycle_found() -> None:
    # acyclic topology: no cycle possible
    a, b, c = FakeWaypoint(1), FakeWaypoint(2), FakeWaypoint(3)
    m = FakeMap(topology=[(a, b), (b, c)], spawn_points=[FakeTransform(99)])
    circuit = build_circuit(m, num_waypoints=1)
    assert len(circuit) == 1
    assert circuit[0].idx == 99


def test_build_circuit_raises_on_empty_map() -> None:
    m = FakeMap(topology=[], spawn_points=[])
    with pytest.raises(RuntimeError, match="no topology cycle and no spawn points"):
        build_circuit(m)


def test_build_circuit_rejects_zero_waypoints() -> None:
    m = FakeMap(topology=_triangle_topology())
    with pytest.raises(ValueError, match="num_waypoints must be > 0"):
        build_circuit(m, num_waypoints=0)


def test_build_circuit_rejects_negative_waypoints() -> None:
    m = FakeMap(topology=_triangle_topology())
    with pytest.raises(ValueError, match="num_waypoints must be > 0"):
        build_circuit(m, num_waypoints=-5)


def test_build_circuit_picks_longest_of_disjoint_cycles() -> None:
    m = FakeMap(topology=_two_disjoint_cycles())  # triangle (3) + square (4)
    circuit = build_circuit(m, num_waypoints=4)
    # square's transforms are idx 4..7
    idxs = {t.idx for t in circuit}
    assert idxs <= {4, 5, 6, 7}


def test_start_line_transform_returns_first() -> None:
    circuit = [FakeTransform(0), FakeTransform(1), FakeTransform(2)]
    assert start_line_transform(circuit).idx == 0


def test_start_line_transform_empty_raises() -> None:
    with pytest.raises(ValueError, match="circuit is empty"):
        start_line_transform([])


def test_build_circuit_returns_copies_not_same_object_identity() -> None:
    # resample should not mutate input topology
    topo = _triangle_topology()
    m = FakeMap(topology=topo)
    build_circuit(m, num_waypoints=64)
    assert len(m.get_topology()) == 3  # unchanged

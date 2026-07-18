"""Race state — FSM + per-car + per-race dataclasses.

Pure data module (no CARLA, no I/O). Holds the authoritative race state
mutated by ``race_manager`` (start/tick/finish) and read by ``scoring`` (score
formula) and ``bridge_ext`` (``/race/state`` JSON snapshot).

Contract:
- ``RacePhase(str, Enum)``: INIT / RUNNING / FINISHED.
- ``LapSplit(lap_number, lap_time_s, cumulative_time_s)`` — frozen record.
- ``CarState(actor_id, is_player, color, lap, waypoint_index, laps_finished,
  splits, walker_hits, car_hits, finish_position, finished_at_s, dnf)`` —
  mutable; identity fields (actor_id/is_player/color) set at construction,
  progress fields mutated by race_manager + collision_scoring.
- ``RaceState(config_num_cars, config_num_laps, phase, started_at_s,
  finished_at_s, map_name, cars, circuit_waypoint_count)`` — mutable; methods
  ``elapsed_s(now)``, ``all_finished()``, ``leaderboard()``, ``player()``.

Leaderboard order: finished cars by ``finish_position`` asc (DNF cars get
``finish_position = num_cars`` per the race_manager contract, so they land
at the back of the finished group); then still-racing cars by laps_finished
desc, then waypoint_index desc (furthest along first).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = ["CarState", "LapSplit", "RacePhase", "RaceState"]


class RacePhase(str, Enum):
    INIT = "init"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class LapSplit:
    lap_number: int
    lap_time_s: float
    cumulative_time_s: float


@dataclass
class CarState:
    actor_id: int
    is_player: bool
    color: str
    lap: int = 1
    waypoint_index: int = 0
    laps_finished: int = 0
    splits: list[LapSplit] = field(default_factory=list)
    walker_hits: int = 0
    car_hits: int = 0
    finish_position: int | None = None
    finished_at_s: float | None = None
    dnf: bool = False


@dataclass
class RaceState:
    config_num_cars: int
    config_num_laps: int
    phase: RacePhase = RacePhase.INIT
    started_at_s: float | None = None
    finished_at_s: float | None = None
    map_name: str = ""
    cars: dict[int, CarState] = field(default_factory=dict)
    circuit_waypoint_count: int = 0

    def elapsed_s(self, now: float) -> float:
        if self.started_at_s is None:
            return 0.0
        return now - self.started_at_s

    def all_finished(self) -> bool:
        if not self.cars:
            return False
        return all(c.finish_position is not None for c in self.cars.values())

    def player(self) -> CarState:
        for c in self.cars.values():
            if c.is_player:
                return c
        raise KeyError("no player car in race state")

    def leaderboard(self) -> list[CarState]:
        def key(c: CarState) -> tuple[int, int, int, int]:
            fp = c.finish_position
            if fp is None:
                return (1, 0, -c.laps_finished, -c.waypoint_index)
            return (0, fp, 0, 0)

        return sorted(self.cars.values(), key=key)

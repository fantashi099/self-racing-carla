"""Race orchestrator — composes map_pool, vehicle_grid, circuit, lap_tracker,
collision_scoring, traffic, ai_driver into a server-side race FSM.

Singleton per ``client`` + ``config``. Holds the authoritative ``RaceState``,
the spawned cars/walkers/sensors, and the finish counter. ``tick()`` advances
every car each call (the bridge calls it on a timer or per /step poll).

Contract:
- ``RaceManager(client, config)``.
- ``start() -> RaceState`` — pick_and_load → build_circuit → spawn_grid →
  CarStates → attach_collision_sensors → spawn_walkers → setup_ai_cars →
  phase=RUNNING. 409-style guard: raises ``RuntimeError`` if already running.
- ``tick() -> RaceState`` — per non-finished car: read transform, update
  progress, on lap completion call on_lap_complete + assign finish_position
  via the monotonic finish counter. DNF if the actor is gone (finish_position
  = num_cars). If all_finished → phase=FINISHED + finished_at_s set.
- ``state_snapshot() -> dict`` — JSON-serializable view of the race.
- ``restart() -> RaceState`` — destroy + start (re-picks random map).
- ``destroy() -> None`` — destroy sensors, walkers, cars in that order.

Finish counter: ``next_finish_pos`` starts at 1, increments per finisher.
DNF cars get ``finish_position = num_cars`` (per the race_state contract).
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time
from typing import Any

from carla_race.ai_driver import setup_ai_cars
from carla_race.circuit import build_circuit
from carla_race.collision_scoring import attach_collision_sensors, destroy_sensors
from carla_race.config import RaceConfig
from carla_race.lap_tracker import on_lap_complete, update_car_progress
from carla_race.map_pool import pick_and_load
from carla_race.race_state import CarState, RacePhase, RaceState
from carla_race.traffic import destroy_walkers, spawn_walkers
from carla_race.vehicle_grid import CarSpawn, destroy_grid, spawn_grid

__all__ = ["RaceManager"]

DEFAULT_TM_PORT = 8000


class RaceManager:
    def __init__(self, client: Any, config: RaceConfig) -> None:
        self._client = client
        self._config = config
        self._state: RaceState | None = None
        self._spawns: list[CarSpawn] = []
        self._walkers: list[Any] = []
        self._sensor_ids: dict[int, int] = {}
        self._circuit: list[Any] = []
        self._car_actors: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._next_finish_pos = 1

    def start(self) -> RaceState:
        if self._state is not None and self._state.phase == RacePhase.RUNNING:
            raise RuntimeError("race already running; call stop or restart first")
        name, carla_map = pick_and_load(self._client)
        world = self._client.get_world()
        self._circuit = build_circuit(carla_map)
        self._spawns = spawn_grid(world, num_cars=self._config.num_cars)
        self._car_actors = {
            s.actor_id: world.get_actor(s.actor_id) for s in self._spawns
        }
        cars: dict[int, CarState] = {
            s.actor_id: CarState(
                actor_id=s.actor_id, is_player=s.is_player, color=s.color
            )
            for s in self._spawns
        }
        self._state = RaceState(
            config_num_cars=self._config.num_cars,
            config_num_laps=self._config.num_laps,
            phase=RacePhase.RUNNING,
            started_at_s=self._now(),
            map_name=name,
            cars=cars,
            circuit_waypoint_count=len(self._circuit),
        )
        self._sensor_ids = attach_collision_sensors(
            world, self._car_actors, cars, lock=self._lock
        )
        if self._config.num_walkers > 0:
            self._walkers = spawn_walkers(world, num_walkers=self._config.num_walkers)
        player_spawn = next((s for s in self._spawns if s.is_player), None)
        player_id = player_spawn.actor_id if player_spawn is not None else -1
        tm = self._get_traffic_manager()
        setup_ai_cars(
            tm,
            self._car_actors,
            carla_map,
            self._circuit,
            difficulty=self._config.ai_difficulty,
            player_actor_id=player_id,
        )
        tm_port = _tm_port(tm)
        for s in self._spawns:
            if s.is_player:
                continue
            actor = self._car_actors.get(s.actor_id)
            if actor is not None:
                _enable_autopilot(actor, tm_port)
        self._next_finish_pos = 1
        return self._state

    def tick(self) -> RaceState | None:
        if self._state is None or self._state.phase != RacePhase.RUNNING:
            return self._state
        world = self._client.get_world()
        now = self._now()
        for actor_id, car in self._state.cars.items():
            if car.finish_position is not None:
                continue
            actor = self._car_actors.get(actor_id)
            if actor is None or not self._actor_alive(world, actor_id, actor):
                car.dnf = True
                car.finish_position = self._config.num_cars
                car.finished_at_s = now
                continue
            tf = actor.get_transform()
            crossed = update_car_progress(car, tf, self._circuit)
            if crossed:
                on_lap_complete(
                    car,
                    now,
                    self._state.started_at_s if self._state.started_at_s is not None else 0.0,
                    num_laps=self._config.num_laps,
                )
                if car.finished_at_s is not None and car.finish_position is None:
                    car.finish_position = self._next_finish_pos
                    self._next_finish_pos += 1
        if self._state.all_finished():
            self._state.phase = RacePhase.FINISHED
            self._state.finished_at_s = now
        return self._state

    def state_snapshot(self) -> dict[str, Any]:
        if self._state is None:
            return {"phase": RacePhase.INIT.value, "cars": []}
        rs = self._state
        now = self._now()
        elapsed = rs.elapsed_s(now)
        cars_json = []
        for car in rs.leaderboard():
            entry: dict[str, Any] = {
                "actor_id": car.actor_id,
                "is_player": car.is_player,
                "color": car.color,
                "lap": car.lap,
                "laps_finished": car.laps_finished,
                "waypoint_index": car.waypoint_index,
                "walker_hits": car.walker_hits,
                "car_hits": car.car_hits,
                "finish_position": car.finish_position,
                "finished_at_s": car.finished_at_s,
                "dnf": car.dnf,
            }
            cars_json.append(entry)
        return {
            "phase": rs.phase.value,
            "map_name": rs.map_name,
            "num_cars": rs.config_num_cars,
            "num_laps": rs.config_num_laps,
            "started_at_s": rs.started_at_s,
            "finished_at_s": rs.finished_at_s,
            "elapsed_s": elapsed,
            "circuit_waypoint_count": rs.circuit_waypoint_count,
            "cars": cars_json,
        }

    def restart(self) -> RaceState:
        self.destroy()
        return self.start()

    def destroy(self) -> None:
        world = self._client.get_world()
        if self._sensor_ids:
            destroy_sensors(world, self._sensor_ids)
            self._sensor_ids = {}
        if self._walkers:
            destroy_walkers(world, self._walkers)
            self._walkers = []
        if self._spawns:
            destroy_grid(world, self._spawns)
            self._spawns = []
        self._car_actors = {}
        self._state = None
        self._next_finish_pos = 1

    def _get_traffic_manager(self) -> Any:
        return self._client.get_trafficmanager(DEFAULT_TM_PORT)

    def _now(self) -> float:
        return time.monotonic()

    def _actor_alive(self, world: Any, actor_id: int, actor: Any) -> bool:
        if actor is None:
            return False
        is_alive = getattr(actor, "is_alive", None)
        if is_alive is not None and callable(is_alive):
            try:
                return bool(is_alive())
            except Exception:
                return False
        return world.get_actor(actor_id) is not None


def _tm_port(tm: Any) -> int:
    """Best-effort read of the TrafficManager's port. CARLA's TM exposes
    ``get_port()``; if absent, fall back to the default port the manager was
    constructed with."""
    get_port = getattr(tm, "get_port", None)
    if get_port is not None:
        with contextlib.suppress(Exception):
            return int(get_port())
    return DEFAULT_TM_PORT


def _enable_autopilot(actor: Any, tm_port: int) -> None:
    """Enable TM autopilot on an actor. Defensive: some mock actors lack
    ``set_autopilot``; skip silently so unit tests stay CARLA-free. Logs
    failures so L2 can see when real CARLA rejects the call."""
    set_autopilot = getattr(actor, "set_autopilot", None)
    if set_autopilot is None:
        return
    try:
        set_autopilot(True, tm_port)
    except Exception as e:
        print(f"[race_manager] set_autopilot failed for actor {getattr(actor, 'id', '?')}: {e!r}", file=sys.stderr)

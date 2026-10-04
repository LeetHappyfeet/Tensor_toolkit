"""Simulation event detection for finite-size bodies and future event hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .state import System, SystemState


@dataclass(frozen=True)
class SimulationEvent:
    """A discrete event detected between two accepted integration states."""

    time: float
    kind: str
    bodies: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, float] = field(default_factory=dict)


class EventDetector(Protocol):
    def detect(
        self,
        previous_time: float,
        previous_state: SystemState,
        time: float,
        state: SystemState,
        system: System,
    ) -> tuple[SimulationEvent, ...]:
        """Return events newly crossed during an accepted integration step."""


@dataclass(frozen=True)
class CollisionDetector:
    """Detect first contact between bodies with non-zero finite radii."""

    def detect(
        self,
        previous_time: float,
        previous_state: SystemState,
        time: float,
        state: SystemState,
        system: System,
    ) -> tuple[SimulationEvent, ...]:
        del previous_time
        events = []
        radii = system.radii
        for i in range(len(system.bodies)):
            for j in range(i + 1, len(system.bodies)):
                contact = radii[i] + radii[j]
                if contact <= 0.0:
                    continue
                previous_distance = float(
                    np.linalg.norm(previous_state.positions[j] - previous_state.positions[i])
                )
                distance = float(np.linalg.norm(state.positions[j] - state.positions[i]))
                if previous_distance > contact and distance <= contact:
                    events.append(
                        SimulationEvent(
                            time=float(time),
                            kind="collision",
                            bodies=(system.names[i], system.names[j]),
                            details={
                                "distance": distance,
                                "contact_distance": float(contact),
                            },
                        )
                    )
        return tuple(events)


@dataclass(frozen=True)
class DistanceCrossingDetector:
    """Detect entry to or exit from a spherical distance threshold."""

    primary: str
    body: str
    radius: float
    direction: str = "enter"
    kind: str = "distance_crossing"

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("distance threshold radius must be finite and positive")
        direction = self.direction.lower()
        if direction not in {"enter", "exit"}:
            raise ValueError("direction must be 'enter' or 'exit'")
        if self.primary == self.body:
            raise ValueError("primary and body must be different")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "direction", direction)

    def detect(
        self,
        previous_time: float,
        previous_state: SystemState,
        time: float,
        state: SystemState,
        system: System,
    ) -> tuple[SimulationEvent, ...]:
        del previous_time
        try:
            primary_index = system.names.index(self.primary)
            body_index = system.names.index(self.body)
        except ValueError as exc:
            raise KeyError("distance-crossing body is not present in the system") from exc

        previous_distance = float(
            np.linalg.norm(
                previous_state.positions[body_index]
                - previous_state.positions[primary_index]
            )
        )
        distance = float(
            np.linalg.norm(state.positions[body_index] - state.positions[primary_index])
        )
        entered = previous_distance > self.radius and distance <= self.radius
        exited = previous_distance < self.radius and distance >= self.radius
        crossed = entered if self.direction == "enter" else exited
        if not crossed:
            return ()
        return (
            SimulationEvent(
                time=float(time),
                kind=self.kind,
                bodies=(self.primary, self.body),
                details={
                    "distance": distance,
                    "threshold_radius": self.radius,
                },
            ),
        )

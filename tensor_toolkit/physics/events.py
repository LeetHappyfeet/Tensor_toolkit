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

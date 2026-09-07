"""Pluggable dynamics models for classical and post-Newtonian simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT
from .gravity import newtonian_gravity_accelerations
from .state import System


class DynamicsModel(Protocol):
    """Acceleration law consumed by the numerical integrators."""

    velocity_dependent: bool

    def accelerations(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        system: System,
    ) -> np.ndarray:
        """Return Cartesian accelerations with shape (N, 3)."""


@dataclass(frozen=True)
class NewtonianGravity:
    """Direct O(N^2) Newtonian gravity reference model."""

    gravitational_constant: float = GRAVITATIONAL_CONSTANT
    softening: float = 0.0
    velocity_dependent: bool = field(default=False, init=False)

    def accelerations(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        system: System,
    ) -> np.ndarray:
        del t, velocities, system
        return newtonian_gravity_accelerations(
            positions,
            masses,
            gravitational_constant=float(self.gravitational_constant),
            softening=float(self.softening),
        )


@dataclass(frozen=True)
class ConstantThrust:
    """Apply constant inertial-frame thrust vectors to selected massive bodies."""

    forces: Mapping[str, np.ndarray]
    velocity_dependent: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        normalized = {}
        for name, value in self.forces.items():
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError("each thrust vector must be a finite shape-(3,) vector")
            normalized[str(name)] = vector.copy()
        object.__setattr__(self, "forces", normalized)

    def accelerations(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        system: System,
    ) -> np.ndarray:
        del t, positions, velocities
        out = np.zeros((len(system.bodies), 3), dtype=np.float64)
        for name, force in self.forces.items():
            try:
                index = system.names.index(name)
            except ValueError as exc:
                raise KeyError(f"unknown thrust body {name!r}") from exc
            mass = float(masses[index])
            if mass <= 0.0:
                raise ValueError(f"thrust requires positive mass for body {name!r}")
            out[index] = force / mass
        return out


@dataclass(frozen=True)
class CompositeDynamics:
    """Add accelerations from multiple independent dynamics models."""

    models: tuple[DynamicsModel, ...]

    def __init__(self, models) -> None:
        models = tuple(models)
        if not models:
            raise ValueError("CompositeDynamics requires at least one model")
        object.__setattr__(self, "models", models)

    @property
    def velocity_dependent(self) -> bool:
        return any(bool(getattr(model, "velocity_dependent", True)) for model in self.models)

    def accelerations(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        system: System,
    ) -> np.ndarray:
        total = np.zeros_like(np.asarray(positions, dtype=np.float64))
        for model in self.models:
            contribution = np.asarray(
                model.accelerations(t, positions, velocities, masses, system),
                dtype=np.float64,
            )
            if contribution.shape != total.shape:
                raise ValueError(
                    f"dynamics model returned {contribution.shape}, expected {total.shape}"
                )
            if not np.all(np.isfinite(contribution)):
                raise ValueError("dynamics model returned non-finite acceleration")
            total += contribution
        return total

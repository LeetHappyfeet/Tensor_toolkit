"""Pluggable dynamics models for classical and post-Newtonian simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT
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


@dataclass(frozen=True)
class CentralBody1PN:
    """Test-particle 1PN motion in the rest frame of ONE fixed spherical primary.

    Harmonic/isotropic weak-field equation, with r and v relative to the
    stationary primary and mu=GM:

      a = -mu*r/r^3
          + mu/(c^2*r^3) * ((4*mu/r - v^2)*r + 4*(r.v)*v)

    This is NOT a general Einstein-Infeld-Hoffmann N-body implementation.
    Only one positive-mass body and any number of passive zero-mass probes
    are permitted. Use RK4, never ordinary velocity-Verlet.
    """
    primary: str
    gravitational_constant: float = GRAVITATIONAL_CONSTANT
    speed_of_light: float = SPEED_OF_LIGHT
    velocity_dependent: bool = field(default=True, init=False)

    def __post_init__(self):
        if not self.primary:
            raise ValueError("1PN requires a named primary")
        if (not np.isfinite(self.gravitational_constant) or self.gravitational_constant <= 0 or
            not np.isfinite(self.speed_of_light) or self.speed_of_light <= 0):
            raise ValueError("1PN physical constants must be finite and positive")

    def accelerations(self, t, positions, velocities, masses, system):
        del t
        p = np.asarray(positions, dtype=np.float64)
        v = np.asarray(velocities, dtype=np.float64)
        mass = np.asarray(masses, dtype=np.float64)
        if p.shape != v.shape or p.shape != (len(system.bodies), 3) or mass.shape != (p.shape[0],):
            raise ValueError("1PN requires matching (N,3) positions and velocities and (N,) masses")
        if self.primary not in system.names:
            raise KeyError(f"unknown 1PN primary {self.primary!r}")
        index = system.names.index(self.primary)
        if mass[index] <= 0 or np.count_nonzero(mass > 0) != 1:
            raise ValueError("central-body 1PN supports exactly one positive-mass primary")
        if not np.allclose(v[index], 0.0, rtol=0., atol=1e-12):
            raise ValueError("1PN central primary must be stationary in the chosen inertial chart")
        mu = float(self.gravitational_constant) * float(mass[index])
        c2 = float(self.speed_of_light)**2
        out = np.zeros_like(p)
        for j in range(len(system.bodies)):
            if j == index:
                continue
            rvec = p[j]-p[index]
            relative_v = v[j]-v[index]
            r = float(np.linalg.norm(rvec))
            v2 = float(np.dot(relative_v, relative_v))
            if r <= max(0.0, system.radii[index]):
                raise ValueError("1PN probe at or inside primary surface/singularity")
            if mu/(r*c2) >= 0.05 or v2/c2 >= 0.05:
                raise ValueError("central-body 1PN requires GM/(rc²)<0.05 and v²/c²<0.05")
            out[j] = (
                -(mu/r**3)*rvec +
                (mu/(c2*r**3))*((4*mu/r-v2)*rvec + 4*np.dot(rvec, relative_v)*relative_v)
            )
        return out

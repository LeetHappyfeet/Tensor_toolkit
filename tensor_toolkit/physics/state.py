"""State containers for classical point-mass dynamics."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _vector3(value, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()

@dataclass(frozen=True)
class Body:
    """A Newtonian point mass in Cartesian SI coordinates."""
    name: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("body name must not be empty")
        mass = float(self.mass)
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError("mass must be finite and non-negative")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "position", _vector3(self.position, name="position"))
        object.__setattr__(self, "velocity", _vector3(self.velocity, name="velocity"))

@dataclass(frozen=True)
class System:
    """A collection of point masses evolved in a shared Cartesian frame."""
    bodies: tuple[Body, ...]

    def __init__(self, bodies) -> None:
        bodies = tuple(bodies)
        if not bodies:
            raise ValueError("system requires at least one body")
        names = [body.name for body in bodies]
        if len(set(names)) != len(names):
            raise ValueError("body names must be unique")
        object.__setattr__(self, "bodies", bodies)

    @property
    def masses(self) -> np.ndarray:
        return np.asarray([body.mass for body in self.bodies], dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        return np.stack([body.position for body in self.bodies])

    @property
    def velocities(self) -> np.ndarray:
        return np.stack([body.velocity for body in self.bodies])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(body.name for body in self.bodies)

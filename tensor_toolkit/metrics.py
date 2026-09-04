"""Metric models for the supported Tensor Toolkit reference path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT


class Metric(Protocol):
    """A spacetime metric that can be sampled on a four-dimensional grid."""

    name: str
    coordinates: tuple[str, str, str, str]

    def evaluate(self, coordinate_grid: tuple[np.ndarray, ...]) -> np.ndarray:
        """Return covariant g_mu_nu with shape (4, 4, Nt, Nx, Ny, Nz)."""


def _grid_shape(coordinate_grid: tuple[np.ndarray, ...]) -> tuple[int, ...]:
    return np.broadcast_shapes(*(np.shape(value) for value in coordinate_grid))


@dataclass(frozen=True)
class MinkowskiMetric:
    name: str = "Minkowski"
    coordinates: tuple[str, str, str, str] = ("t", "x", "y", "z")

    def evaluate(self, coordinate_grid: tuple[np.ndarray, ...]) -> np.ndarray:
        shape = _grid_shape(coordinate_grid)
        g = np.zeros((4, 4, *shape), dtype=np.float64)
        g[0, 0] = -1.0
        g[1, 1] = g[2, 2] = g[3, 3] = 1.0
        return g


@dataclass(frozen=True)
class DeSitterFlatMetric:
    """Flat-slicing de Sitter metric ds^2=-dt^2+exp(2Ht)(dx^2+dy^2+dz^2)."""

    hubble: float = 0.1
    name: str = "de Sitter (flat slicing)"
    coordinates: tuple[str, str, str, str] = ("t", "x", "y", "z")

    def evaluate(self, coordinate_grid: tuple[np.ndarray, ...]) -> np.ndarray:
        t = coordinate_grid[0]
        shape = _grid_shape(coordinate_grid)
        scale2 = np.exp(2.0 * float(self.hubble) * t)
        g = np.zeros((4, 4, *shape), dtype=np.float64)
        g[0, 0] = -1.0
        g[1, 1] = scale2
        g[2, 2] = scale2
        g[3, 3] = scale2
        return g


@dataclass(frozen=True)
class AlcubierreMetric:
    """Alcubierre metric in Cartesian coordinates with a constant x velocity.

    Geometrized units are used (c=G=1). The ship center follows x_s(t)=x0+v*t.
    The standard smooth top-hat shape function is used.
    """

    velocity: float = 0.1
    radius: float = 2.0
    sigma: float = 4.0
    x0: float = 0.0
    name: str = "Alcubierre"
    coordinates: tuple[str, str, str, str] = ("t", "x", "y", "z")

    def shape_function(self, r: np.ndarray) -> np.ndarray:
        R = float(self.radius)
        sigma = float(self.sigma)
        if R <= 0.0 or sigma <= 0.0:
            raise ValueError("radius and sigma must be positive")
        denom = 2.0 * np.tanh(sigma * R)
        return (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / denom

    def evaluate(self, coordinate_grid: tuple[np.ndarray, ...]) -> np.ndarray:
        t, x, y, z = coordinate_grid
        v = float(self.velocity)
        xs = float(self.x0) + v * t
        r = np.sqrt((x - xs) ** 2 + y**2 + z**2)
        f = self.shape_function(r)
        shape = np.broadcast_shapes(np.shape(t), np.shape(x), np.shape(y), np.shape(z))

        # ds^2 = -dt^2 + (dx - v f dt)^2 + dy^2 + dz^2
        g = np.zeros((4, 4, *shape), dtype=np.float64)
        g[0, 0] = -(1.0 - (v * f) ** 2)
        g[0, 1] = g[1, 0] = -v * f
        g[1, 1] = 1.0
        g[2, 2] = 1.0
        g[3, 3] = 1.0
        return g


@dataclass(frozen=True)
class SchwarzschildIsotropicMetric:
    """Schwarzschild exterior in isotropic Cartesian coordinates.

    Coordinates are (ct, x, y, z), all measured in metres, and metric
    components are dimensionless. The isotropic radius rho must remain outside
    the horizon rho = GM/(2 c^2).
    """

    mass_kg: float
    name: str = "Schwarzschild (isotropic Cartesian)"
    coordinates: tuple[str, str, str, str] = ("ct", "x", "y", "z")

    @property
    def geometric_mass(self) -> float:
        mass = float(self.mass_kg)
        if mass <= 0.0:
            raise ValueError("mass_kg must be positive")
        return GRAVITATIONAL_CONSTANT * mass / SPEED_OF_LIGHT**2

    @property
    def isotropic_horizon_radius(self) -> float:
        return 0.5 * self.geometric_mass

    def evaluate(self, coordinate_grid: tuple[np.ndarray, ...]) -> np.ndarray:
        _, x, y, z = coordinate_grid
        rho = np.sqrt(x**2 + y**2 + z**2)
        horizon = self.isotropic_horizon_radius
        if np.any(rho <= horizon):
            raise ValueError(
                "Schwarzschild isotropic coordinates require rho > GM/(2 c^2)"
            )

        u = self.geometric_mass / (2.0 * rho)
        lapse2 = ((1.0 - u) / (1.0 + u)) ** 2
        spatial = (1.0 + u) ** 4
        shape = _grid_shape(coordinate_grid)

        g = np.zeros((4, 4, *shape), dtype=np.float64)
        g[0, 0] = -lapse2
        g[1, 1] = spatial
        g[2, 2] = spatial
        g[3, 3] = spatial
        return g


__all__ = [
    "Metric",
    "MinkowskiMetric",
    "DeSitterFlatMetric",
    "AlcubierreMetric",
    "SchwarzschildIsotropicMetric",
]

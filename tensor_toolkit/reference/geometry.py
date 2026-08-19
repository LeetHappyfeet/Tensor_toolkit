"""Readable float64 CPU reference implementation of the GR tensor pipeline."""

from __future__ import annotations

import numpy as np

from tensor_toolkit.constants import KAPPA_GEOMETRIZED, KAPPA_SI
from tensor_toolkit.validation import validate_metric, validate_spacings


def inverse_metric(metric: np.ndarray) -> np.ndarray:
    """Return ``g^{mu nu}`` for a metric stored as ``(4, 4, *grid)``."""
    g = validate_metric(metric)
    matrix_last = np.moveaxis(g, (0, 1), (-2, -1))
    try:
        inv = np.linalg.inv(matrix_last)
    except np.linalg.LinAlgError as exc:
        raise ValueError("metric is singular at one or more grid points") from exc
    return np.moveaxis(inv, (-2, -1), (0, 1))


def _partial(field: np.ndarray, coordinate: int, spacings: tuple[float, ...], *, field_ndim: int) -> np.ndarray:
    grid_axis = field_ndim + coordinate
    return np.gradient(field, spacings[coordinate], axis=grid_axis, edge_order=2)


def metric_derivatives(metric: np.ndarray, spacings) -> np.ndarray:
    """Return ``d_alpha g_{mu nu}`` as shape ``(4, 4, 4, *grid)``."""
    g = validate_metric(metric)
    grid_shape = g.shape[2:]
    dx = validate_spacings(spacings, grid_shape)
    if len(grid_shape) != 4:
        raise ValueError("the GR reference pipeline currently requires a 4D coordinate grid")
    return np.stack([_partial(g, a, dx, field_ndim=2) for a in range(4)], axis=2)


def christoffel_symbols(metric: np.ndarray, spacings) -> np.ndarray:
    """Compute ``Gamma^rho_{mu nu}`` with shape ``(4, 4, 4, *grid)``."""
    g = validate_metric(metric)
    gu = inverse_metric(g)
    dg = metric_derivatives(g, spacings)
    grid = g.shape[2:]
    gamma = np.zeros((4, 4, 4, *grid), dtype=np.float64)
    for rho in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros(grid, dtype=np.float64)
                for sigma in range(4):
                    acc += gu[rho, sigma] * (
                        dg[sigma, nu, mu] + dg[sigma, mu, nu] - dg[mu, nu, sigma]
                    )
                gamma[rho, mu, nu] = 0.5 * acc
    return gamma


def riemann_tensor(metric: np.ndarray, spacings) -> np.ndarray:
    """Compute ``R^rho_{ sigma mu nu}`` using the documented sign convention."""
    g = validate_metric(metric)
    dx = validate_spacings(spacings, g.shape[2:])
    gamma = christoffel_symbols(g, dx)
    grid = g.shape[2:]
    out = np.zeros((4, 4, 4, 4, *grid), dtype=np.float64)
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    value = _partial(gamma[rho, nu, sigma], mu, dx, field_ndim=0)
                    value -= _partial(gamma[rho, mu, sigma], nu, dx, field_ndim=0)
                    for lam in range(4):
                        value += gamma[rho, mu, lam] * gamma[lam, nu, sigma]
                        value -= gamma[rho, nu, lam] * gamma[lam, mu, sigma]
                    out[rho, sigma, mu, nu] = value
    return out


def ricci_tensor(metric: np.ndarray, spacings) -> np.ndarray:
    """Compute ``R_{sigma nu} = R^rho_{ sigma rho nu}``."""
    return np.einsum("rsrn...->sn...", riemann_tensor(metric, spacings))


def ricci_scalar(metric: np.ndarray, spacings) -> np.ndarray:
    """Compute ``R = g^{mu nu} R_{mu nu}``."""
    gu = inverse_metric(metric)
    ricci = ricci_tensor(metric, spacings)
    return np.einsum("mn...,mn...->...", gu, ricci)


def einstein_tensor(metric: np.ndarray, spacings) -> np.ndarray:
    """Compute the covariant Einstein tensor ``G_{mu nu}``."""
    g = validate_metric(metric)
    ricci = ricci_tensor(g, spacings)
    scalar = np.einsum("mn...,mn...->...", inverse_metric(g), ricci)
    return ricci - 0.5 * g * scalar


def stress_energy_tensor(metric: np.ndarray, spacings, *, units: str = "si") -> np.ndarray:
    """Compute covariant ``T_{mu nu}`` from Einstein's equation with Lambda=0."""
    coupling = {"si": KAPPA_SI, "geometrized": KAPPA_GEOMETRIZED}.get(units)
    if coupling is None:
        raise ValueError("units must be 'si' or 'geometrized'")
    return einstein_tensor(metric, spacings) / coupling


__all__ = [
    "inverse_metric",
    "metric_derivatives",
    "christoffel_symbols",
    "riemann_tensor",
    "ricci_tensor",
    "ricci_scalar",
    "einstein_tensor",
    "stress_energy_tensor",
]

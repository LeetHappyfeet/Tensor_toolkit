"""Readable float64 CPU reference implementation of the GR tensor pipeline."""

from __future__ import annotations

import numpy as np

from tensor_toolkit.constants import KAPPA_GEOMETRIZED, KAPPA_SI
from tensor_toolkit.validation import validate_metric, validate_spacings


def inverse_metric(metric: np.ndarray) -> np.ndarray:
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
    g = validate_metric(metric)
    grid_shape = g.shape[2:]
    dx = validate_spacings(spacings, grid_shape)
    if len(grid_shape) != 4:
        raise ValueError("the GR reference pipeline currently requires a 4D coordinate grid")
    return np.stack([_partial(g, a, dx, field_ndim=2) for a in range(4)], axis=2)


def christoffel_symbols(metric: np.ndarray, spacings) -> np.ndarray:
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


def riemann_from_christoffel(gamma: np.ndarray, spacings) -> np.ndarray:
    dx = tuple(float(x) for x in spacings)
    grid = gamma.shape[3:]
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


def riemann_tensor(metric: np.ndarray, spacings) -> np.ndarray:
    return riemann_from_christoffel(christoffel_symbols(metric, spacings), spacings)


def ricci_from_riemann(riemann: np.ndarray) -> np.ndarray:
    return np.einsum("rsrn...->sn...", riemann)


def ricci_tensor(metric: np.ndarray, spacings) -> np.ndarray:
    return ricci_from_riemann(riemann_tensor(metric, spacings))


def ricci_scalar_from_ricci(metric: np.ndarray, ricci: np.ndarray) -> np.ndarray:
    return np.einsum("mn...,mn...->...", inverse_metric(metric), ricci)


def ricci_scalar(metric: np.ndarray, spacings) -> np.ndarray:
    return ricci_scalar_from_ricci(metric, ricci_tensor(metric, spacings))


def einstein_from_ricci(metric: np.ndarray, ricci: np.ndarray) -> np.ndarray:
    g = validate_metric(metric)
    scalar = ricci_scalar_from_ricci(g, ricci)
    return ricci - 0.5 * g * scalar


def einstein_tensor(metric: np.ndarray, spacings) -> np.ndarray:
    return einstein_from_ricci(metric, ricci_tensor(metric, spacings))


def stress_energy_from_einstein(einstein: np.ndarray, *, units: str = "si") -> np.ndarray:
    coupling = {"si": KAPPA_SI, "geometrized": KAPPA_GEOMETRIZED}.get(units)
    if coupling is None:
        raise ValueError("units must be 'si' or 'geometrized'")
    return einstein / coupling


def stress_energy_tensor(metric: np.ndarray, spacings, *, units: str = "si") -> np.ndarray:
    return stress_energy_from_einstein(einstein_tensor(metric, spacings), units=units)


__all__ = [
    "inverse_metric", "metric_derivatives", "christoffel_symbols",
    "riemann_from_christoffel", "riemann_tensor", "ricci_from_riemann",
    "ricci_tensor", "ricci_scalar_from_ricci", "ricci_scalar",
    "einstein_from_ricci", "einstein_tensor", "stress_energy_from_einstein",
    "stress_energy_tensor",
]

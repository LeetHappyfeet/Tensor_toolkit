"""Validation helpers for the reference metric representation."""

from __future__ import annotations

import numpy as np

from .conventions import DIMENSIONS


def validate_metric(metric: np.ndarray, *, require_float64: bool = True, symmetry_tol: float = 1e-12) -> np.ndarray:
    """Validate and return a metric array with layout ``(4, 4, *grid)``."""
    g = np.asarray(metric)
    if g.ndim < 2 or g.shape[:2] != (DIMENSIONS, DIMENSIONS):
        raise ValueError(f"metric must have shape (4, 4, *grid); got {g.shape}")
    if require_float64 and g.dtype != np.float64:
        raise TypeError(f"reference metrics must use float64; got {g.dtype}")
    if not np.all(np.isfinite(g)):
        raise ValueError("metric contains NaN or infinite values")
    if not np.allclose(g, np.swapaxes(g, 0, 1), rtol=0.0, atol=symmetry_tol):
        raise ValueError("covariant metric must be symmetric in its tensor indices")

    matrices = np.moveaxis(g, (0, 1), (-2, -1))
    determinants = np.linalg.det(matrices)
    if not np.all(np.isfinite(determinants)) or np.any(
        np.isclose(determinants, 0.0, rtol=0.0, atol=1e-14)
    ):
        raise ValueError("metric is singular at one or more grid points")
    return g


def validate_spacings(spacings, grid_shape) -> tuple[float, ...]:
    spacings = tuple(float(x) for x in spacings)
    grid_shape = tuple(grid_shape)
    if len(spacings) != len(grid_shape):
        raise ValueError("one spacing is required for each coordinate-grid axis")
    if any(dx <= 0.0 or not np.isfinite(dx) for dx in spacings):
        raise ValueError("all coordinate spacings must be finite and positive")
    if any(n < 3 for n in grid_shape):
        raise ValueError("each differentiated grid axis needs at least 3 points")
    return spacings


__all__ = ["validate_metric", "validate_spacings"]

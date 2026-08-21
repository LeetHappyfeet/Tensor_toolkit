"""Visualization helpers shared by the Tensor Toolkit GUI and tests.

This module deliberately has no Tkinter or Matplotlib dependency. It turns
rank-2 tensor fields stored as ``(4, 4, Nt, Nx, Ny, Nz)`` into 2-D slices and
provides metric-parameter helpers for the built-in experiment registry.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from tensor_toolkit.metrics import AlcubierreMetric, DeSitterFlatMetric, MinkowskiMetric

COORDINATE_NAMES = ("t", "x", "y", "z")


def editable_metric_parameters(metric) -> dict[str, float]:
    """Return user-editable numeric parameters for a built-in metric."""
    if isinstance(metric, MinkowskiMetric):
        return {}
    if isinstance(metric, DeSitterFlatMetric):
        return {"hubble": float(metric.hubble)}
    if isinstance(metric, AlcubierreMetric):
        return {
            "velocity": float(metric.velocity),
            "radius": float(metric.radius),
            "sigma": float(metric.sigma),
            "x0": float(metric.x0),
        }
    raise TypeError(f"unsupported metric type for GUI parameter editing: {type(metric).__name__}")


def replace_metric_parameters(experiment, parameters: dict[str, float]):
    """Return an experiment whose metric has the supplied numeric parameters."""
    allowed = editable_metric_parameters(experiment.metric)
    unknown = set(parameters) - set(allowed)
    if unknown:
        raise ValueError(f"unknown metric parameter(s): {', '.join(sorted(unknown))}")
    values = {name: float(parameters[name]) for name in parameters}
    return replace(experiment, metric=replace(experiment.metric, **values))


def extract_2d_slice(
    field: np.ndarray,
    *,
    component: tuple[int, int],
    horizontal_axis: int,
    vertical_axis: int,
    fixed_indices: dict[int, int] | None = None,
) -> np.ndarray:
    """Extract a 2-D slice from a rank-2 tensor field.

    The returned array is ordered ``(vertical, horizontal)`` for direct use by
    Matplotlib image/pcolormesh functions.
    """
    value = np.asarray(field)
    if value.ndim != 6 or value.shape[:2] != (4, 4):
        raise ValueError(
            "visualized tensor fields must have shape (4, 4, Nt, Nx, Ny, Nz)"
        )
    mu, nu = (int(component[0]), int(component[1]))
    if not (0 <= mu < 4 and 0 <= nu < 4):
        raise ValueError("tensor component indices must be between 0 and 3")
    if horizontal_axis == vertical_axis:
        raise ValueError("horizontal and vertical plot axes must be different")
    if horizontal_axis not in range(4) or vertical_axis not in range(4):
        raise ValueError("plot axis indices must be between 0 and 3")

    fixed_indices = dict(fixed_indices or {})
    plot_axes = {horizontal_axis, vertical_axis}
    selectors: list[object] = []
    retained_axes: list[int] = []
    for axis, size in enumerate(value.shape[2:]):
        if axis in plot_axes:
            selectors.append(slice(None))
            retained_axes.append(axis)
        else:
            index = int(fixed_indices.get(axis, size // 2))
            if not 0 <= index < size:
                raise IndexError(
                    f"fixed index {index} is outside coordinate axis {axis} with size {size}"
                )
            selectors.append(index)

    sliced = value[(mu, nu, *selectors)]
    if sliced.ndim != 2:
        raise RuntimeError("internal slicing error: expected a 2-D result")
    desired = (vertical_axis, horizontal_axis)
    permutation = tuple(retained_axes.index(axis) for axis in desired)
    return np.transpose(sliced, axes=permutation)


def center_matrix(field: np.ndarray) -> np.ndarray:
    """Return a 4x4 tensor matrix at the geometric center of a 4-D grid."""
    value = np.asarray(field)
    if value.ndim != 6 or value.shape[:2] != (4, 4):
        raise ValueError("center matrix requires shape (4, 4, Nt, Nx, Ny, Nz)")
    center = tuple(size // 2 for size in value.shape[2:])
    return value[(slice(None), slice(None), *center)]


__all__ = [
    "COORDINATE_NAMES",
    "editable_metric_parameters",
    "replace_metric_parameters",
    "extract_2d_slice",
    "center_matrix",
]

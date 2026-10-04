"""Point geometry strategies below the renderer layer.

analytic / finite4 currently supply g, g^-1 and Gamma. Higher-curvature
fields must explicitly use the full numerical reference or a validated saved grid.
The grid backend interpolates saved numerical fields: it does NOT recompute GR.
"""
from __future__ import annotations

from itertools import product
import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT
from tensor_toolkit.metrics import MinkowskiMetric, SchwarzschildIsotropicMetric
from tensor_toolkit.validation import validate_lorentzian_signature

CONNECTION_FIELDS = frozenset({"metric", "inverse_metric", "christoffel"})


def metric_point(metric, event):
    grid = tuple(np.asarray([v], dtype=np.float64) for v in event)
    value = np.asarray(metric.evaluate(grid), dtype=np.float64)
    if value.shape != (4, 4, 1):
        raise ValueError(f"metric point evaluator returned {value.shape}, expected (4,4,1)")
    g = value[:, :, 0].copy()
    validate_lorentzian_signature(g)
    return g


def analytic_metric_derivatives(metric, event):
    """Return dg[mu,nu,alpha], alpha being the coordinate derivative index."""
    event = np.asarray(event, dtype=np.float64)
    if isinstance(metric, MinkowskiMetric):
        return np.zeros((4, 4, 4), dtype=np.float64)
    if not isinstance(metric, SchwarzschildIsotropicMetric):
        raise NotImplementedError(
            f"analytic Christoffel derivatives are unavailable for {type(metric).__name__}; "
            "choose method='numerical' or 'finite4'"
        )
    xyz = event[1:]
    radius = float(np.linalg.norm(xyz))
    # Domain and mass guards are also enforced by metric_point().
    u = metric.geometric_mass / (2 * radius)
    dg = np.zeros((4, 4, 4), dtype=np.float64)
    for axis in range(1, 4):
        du = -u * xyz[axis - 1] / radius**2
        dg[0, 0, axis] = 4 * (1-u) / (1+u)**3 * du
        for diagonal in range(1, 4):
            dg[diagonal, diagonal, axis] = 4 * (1+u)**3 * du
    return dg


def fourth_order_metric_derivatives(metric, event, spacings):
    """Five-point centered, O(h^4) derivatives at one event, not full curvature."""
    event = np.asarray(event, dtype=np.float64)
    dg = np.empty((4, 4, 4), dtype=np.float64)
    for axis, h in enumerate(spacings):
        samples = {}
        for offset in (-2, -1, 1, 2):
            location = event.copy()
            location[axis] += offset * h
            samples[offset] = metric_point(metric, location)
        dg[:, :, axis] = (samples[-2] - 8*samples[-1] +
                           8*samples[1] - samples[2]) / (12*h)
    return dg


def connection_from_derivatives(g, dg):
    """Compute Gamma^rho_{mu nu} using the specified covariant dg layout."""
    inverse = np.linalg.inv(g)
    gamma = np.empty((4, 4, 4), dtype=np.float64)
    for rho in range(4):
        for mu in range(4):
            for nu in range(4):
                value = sum(inverse[rho, sigma] *
                            (dg[sigma, nu, mu] + dg[sigma, mu, nu] -
                             dg[mu, nu, sigma]) for sigma in range(4))
                gamma[rho, mu, nu] = .5 * value
    return inverse, gamma


def point_connection_fields(metric, event, spacings, outputs, *, method):
    outputs = frozenset(outputs)
    unsupported = outputs - CONNECTION_FIELDS
    if unsupported:
        raise NotImplementedError(
            f"{method} point sampler only supports metric, inverse_metric, christoffel; "
            f"unsupported: {sorted(unsupported)}. Use method='numerical' for curvature."
        )
    g = metric_point(metric, event)
    out = {}
    if "metric" in outputs:
        out["metric"] = g
    if outputs & {"inverse_metric", "christoffel"}:
        if method == "analytic":
            dg = analytic_metric_derivatives(metric, event)
        elif method == "finite4":
            dg = fourth_order_metric_derivatives(metric, event, spacings)
        else:
            raise ValueError(f"unrecognized point-derivative method {method!r}")
        inverse, gamma = connection_from_derivatives(g, dg)
        if "inverse_metric" in outputs:
            out["inverse_metric"] = inverse
        if "christoffel" in outputs:
            out["christoffel"] = gamma
    return out


def interpolated_grid_fields(result, metric, event, outputs, *, units):
    """Multilinear 4-D interpolation from a completed, compatible ExperimentResult.

    Reject extrapolation and changed metric parameters. Interpolated curvature is
    approximate even if the input grid was produced by an analytic metric.
    """
    expected = repr(metric)
    if (result.metric_name != metric.name or
        tuple(result.coordinates) != tuple(metric.coordinates) or
        result.metadata.get("metric_configuration") != expected):
        raise ValueError("saved geometry grid does not match metric instance, parameters or chart")
    if ("stress_energy" in outputs and
        result.metadata.get("stress_energy_units") != units):
        raise ValueError("saved stress-energy units disagree with sampler units")
    axes = result.axis_values
    if len(axes) != 4:
        raise ValueError("cached geometry requires four coordinate axes")
    corners = []
    for coordinate, axis in zip(event, axes):
        axis = np.asarray(axis, dtype=np.float64)
        if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError("cached-grid axes must be finite and strictly increasing")
        if coordinate < axis[0] or coordinate > axis[-1]:
            raise ValueError("event is outside the cached geometry grid (no extrapolation)")
        low = int(np.clip(np.searchsorted(axis, coordinate, side="right") - 1, 0, len(axis) - 2))
        weight = float((coordinate - axis[low]) / (axis[low+1] - axis[low]))
        corners.append((low, weight))
    out = {}
    for name in outputs:
        if name not in result.fields:
            raise KeyError(f"cached grid has no retained field {name!r}")
        source = np.asanyarray(result.fields[name])
        prefix = source.shape[:-4]
        if source.shape[-4:] != tuple(len(a) for a in axes):
            raise ValueError("saved field and axes disagree")
        value = np.zeros(prefix, dtype=np.float64)
        for bits in product((0, 1), repeat=4):
            fraction = 1.0
            positions = []
            for bit, (low, weight) in zip(bits, corners):
                fraction *= weight if bit else (1 - weight)
                positions.append(low + bit)
            if fraction:
                value += fraction * source[(Ellipsis, *positions)]
        if not np.all(np.isfinite(value)):
            raise ValueError("interpolated grid field is non-finite")
        if name == "metric":
            validate_lorentzian_signature(value)
        out[name] = float(value) if value.ndim == 0 else value
    return out


__all__ = [
    "CONNECTION_FIELDS", "metric_point", "analytic_metric_derivatives",
    "fourth_order_metric_derivatives", "point_connection_fields",
    "interpolated_grid_fields",
]

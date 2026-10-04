"""Resolution estimates and genuine nested-grid comparisons for tensor experiments.

A wall-cell count is a heuristic, NOT proof of convergence. Comparisons require
identical metrics, coordinate charts, physical domains and nested nodes.
No GUI or VTK imports belong in this module.
"""
from __future__ import annotations

from math import ceil, log
import json
import numpy as np

from .metrics import AlcubierreMetric


def _wall_radius(metric: AlcubierreMetric, level: float) -> float:
    """Solve f(r)=level on the outward monotone branch, without sampling a grid."""
    lower = 0.0
    upper = max(float(metric.radius) * 2, 8 / float(metric.sigma))
    for _ in range(32):
        if float(metric.shape_function(np.asarray(upper))) <= level:
            break
        upper *= 2
    else:
        raise ValueError("unable to bracket Alcubierre wall")
    for _ in range(65):
        middle = (lower + upper) / 2
        if float(metric.shape_function(np.asarray(middle))) > level:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def resolution_report(metric, axes) -> dict[str, object]:
    """Describe actual 4-D spacings and metric-specific under-resolution risks."""
    axes = tuple(np.asarray(a, dtype=np.float64) for a in axes)
    if len(axes) != 4 or any(a.ndim != 1 or a.size < 3 or
                            not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0)
                            for a in axes):
        raise ValueError("resolution requires four increasing axes with >=3 points")
    spacings = tuple(float(np.max(np.diff(a))) for a in axes)
    report: dict[str, object] = {
        "grid_shape": tuple(int(a.size) for a in axes),
        "max_spacings": spacings,
        "derivative_scheme": "numpy.gradient, second-order interior and one-sided boundary",
        "curvature_caveat": "repeated numerical differentiation; refine grid to verify",
        "assessment": "heuristic_only",
        "warnings": [],
    }
    warnings = report["warnings"]
    if isinstance(metric, AlcubierreMetric):
        r90 = _wall_radius(metric, 0.9)
        r10 = _wall_radius(metric, 0.1)
        width = r10 - r90
        if not np.isfinite(width) or width <= 0:
            raise ValueError("non-positive Alcubierre wall width")
        cell_counts = tuple(width / h for h in spacings[1:])
        required = tuple(ceil((a[-1] - a[0]) * 6 / width) + 1 for a in axes[1:])
        report["alcubierre_wall"] = {
            "definition": "radial f(r) transition from 90% to 10%",
            "r90": r90, "r10": r10, "width": width,
            "cells_per_spatial_axis": cell_counts,
            "minimum_cells": min(cell_counts),
            "suggested_points_for_six_cells": required,
            "target_is_heuristic": True,
        }
        if min(cell_counts) < 6.0:
            warnings.append("Alcubierre wall has fewer than six cells along at least one axis; run a convergence study")
        centers = (float(metric.x0) + float(metric.velocity) * axes[0][0],
                   float(metric.x0) + float(metric.velocity) * axes[0][-1])
        # r10 bounds the main transition; clipping it contaminates displayed volume.
        if min(centers) - r10 < axes[1][0] or max(centers) + r10 > axes[1][-1]:
            warnings.append("Alcubierre bubble transition reaches outside the x-domain during the requested time interval")
        if any(-r10 < axis[0] or r10 > axis[-1] for axis in axes[2:]):
            warnings.append("Alcubierre bubble transition reaches outside a transverse spatial domain")
        displacement = abs(float(metric.velocity)) * spacings[0]
        report["bubble_displacement_per_time_step"] = displacement
        report["bubble_displacement_in_x_cells"] = displacement / spacings[1]
    return report


def _nested_indices(coarse_axes, fine_axes):
    if len(coarse_axes) != 4 or len(fine_axes) != 4:
        raise ValueError("both grids must have four axes")
    indices, ratios = [], []
    for c, f in zip(coarse_axes, fine_axes):
        c, f = np.asarray(c), np.asarray(f)
        if c.ndim != 1 or f.ndim != 1 or c.size < 3 or f.size < c.size:
            raise ValueError("comparison requires increasing nested 4-D grids")
        if (f.size - 1) % (c.size - 1):
            raise ValueError("fine-grid node count must be a nested integer refinement")
        ratio = (f.size - 1) // (c.size - 1)
        ix = np.arange(c.size) * ratio
        tolerance = 2e-10 * max(1., float(np.max(np.abs(c))), float(np.max(np.abs(f))))
        if not np.allclose(c, f[ix], rtol=0., atol=tolerance):
            raise ValueError("grid axes must have identical domains and matching nested nodes")
        indices.append(ix)
        ratios.append(ratio)
    return tuple(indices), tuple(ratios)


def _check_compatible(coarse, fine):
    # Reject historical results without parameter provenance: same metric family
    # can represent radically different physical configurations.
    for result in (coarse, fine):
        if not result.metadata.get("metric_configuration") or not result.metadata.get("spacetime"):
            raise ValueError("comparison requires saved metric configuration and spacetime chart metadata")
    # Normalizing JSON handles persisted list-vs-tuple coordinate metadata.
    chart_c = json.dumps(coarse.metadata["spacetime"], sort_keys=True)
    chart_f = json.dumps(fine.metadata["spacetime"], sort_keys=True)
    if (coarse.metric_name != fine.metric_name or tuple(coarse.coordinates) != tuple(fine.coordinates)
        or coarse.metadata["metric_configuration"] != fine.metadata["metric_configuration"]
        or coarse.metadata.get("stress_energy_units") != fine.metadata.get("stress_energy_units")
        or chart_c != chart_f):
        raise ValueError("convergence requires the same metric configuration, chart and units")
    return _nested_indices(coarse.axis_values, fine.axis_values)


def _restrict(array, indices):
    value = np.asanyarray(array)
    prefix = value.ndim - 4
    if prefix < 0:
        raise ValueError("tensor field must end in a four-dimensional coordinate grid")
    # Reduce the last spatial axis first, limiting peak allocation on memmaps.
    for axis in (3, 2, 1, 0):
        value = np.take(value, indices[axis], axis=prefix + axis)
    return value


def _differences(coarse_array, fine_array):
    c = np.asarray(coarse_array, dtype=np.float64)
    f = np.asarray(fine_array, dtype=np.float64)
    if c.shape != f.shape or not np.all(np.isfinite(c)) or not np.all(np.isfinite(f)):
        raise ValueError("compared fields must have identical finite shapes on common nodes")
    diff = c - f
    rms = float(np.sqrt(np.mean(diff * diff)))
    scale = float(np.sqrt(np.mean(f * f)))
    return {"max_abs_difference": float(np.max(np.abs(diff))),
            "rms_difference": rms, "reference_rms": scale,
            "relative_rms_difference": rms / scale if scale else rms,
            "common_values": int(c.size)}


def compare_nested_results(coarse, fine, *, fields=None) -> dict[str, object]:
    """Compare computed values at exactly matching nodes; never interpolate results."""
    indices, ratios = _check_compatible(coarse, fine)
    names = sorted(set(coarse.fields) & set(fine.fields)) if fields is None else sorted(set(fields))
    if not names or any(name not in coarse.fields or name not in fine.fields for name in names):
        raise ValueError("requested comparison fields must be retained in both results")
    return {
        "kind": "nested_grid_field_difference",
        "coarse_shape": tuple(len(a) for a in coarse.axis_values),
        "fine_shape": tuple(len(a) for a in fine.axis_values),
        "refinement_factors": ratios,
        "convergence_certified": False,
        "fields": {name: _differences(coarse.fields[name],
                                      _restrict(fine.fields[name], indices)) for name in names},
    }


def compare_three_resolutions(coarse, medium, fine, *, field: str) -> dict[str, object]:
    """Estimate observed order using BOTH differences restricted to COARSE nodes.

    Requires a common refinement factor on every refined axis. A zero difference
    yields no observed-order estimate, not an assertion of infinite accuracy.
    """
    mid_indices, first_ratios = _check_compatible(coarse, medium)
    fine_indices, second_ratios = _check_compatible(medium, fine)
    factors = [r for r in first_ratios if r > 1]
    if not factors or any(r != factors[0] for r in factors) or first_ratios != second_ratios:
        raise ValueError("observed-order estimate needs the same uniform refinement factor twice")
    if any(field not in result.fields for result in (coarse, medium, fine)):
        raise KeyError(field)
    direct_indices, _ = _check_compatible(coarse, fine)
    c = np.asarray(coarse.fields[field])
    m = _restrict(medium.fields[field], mid_indices)
    f = _restrict(fine.fields[field], direct_indices)
    cm, mf = _differences(c, m), _differences(m, f)
    d1, d2 = cm["rms_difference"], mf["rms_difference"]
    order = log(d1 / d2) / log(factors[0]) if d1 > 0 and d2 > 0 else None
    return {"kind": "three_grid_observed_order", "field": field,
            "refinement_factor": factors[0], "coarse_to_medium": cm,
            "medium_to_fine_on_coarse_nodes": mf, "observed_order": order,
            "convergence_certified": False,
            "note": "Observed order is a diagnostic, not validation of the physical model."}


def spatial_reflection_report(result, field: str, *, axis: int = 2) -> dict[str, object]:
    """Check a justified Cartesian reflection, including tensor-index parity.

    Alcubierre is reflection-invariant in transverse y and z; Minkowski is
    invariant about any centered Cartesian spatial axis. This is an opt-in
    numerical check and cannot establish convergence on its own.
    """
    if result.metric_name == "Alcubierre":
        supported = (2, 3)
    elif result.metric_name == "Minkowski":
        supported = (1, 2, 3)
    else:
        raise ValueError("no justified reflection rule registered for this metric")
    if axis not in supported:
        raise ValueError("requested spatial reflection is not a symmetry of this metric")
    coords = result.axis_values
    if len(coords) != 4:
        raise ValueError("reflection requires four coordinate axes")
    values = np.asarray(coords[axis], dtype=np.float64)
    if not np.allclose(values, -values[::-1], atol=1e-12, rtol=1e-12):
        raise ValueError("reflection axis must be centered at zero with symmetric nodes")
    if field not in result.fields:
        raise KeyError(field)
    source = np.asanyarray(result.fields[field])
    if source.ndim == 4 and field == "ricci_scalar":
        signs = None
    elif source.ndim == 6 and field in {
        "metric", "inverse_metric", "ricci", "einstein", "stress_energy"
    } and source.shape[:2] == (4, 4):
        signs = np.fromfunction(
            lambda i, j: (-1.) ** ((i == axis).astype(int) + (j == axis).astype(int)),
            (4, 4), dtype=int,
        )[:, :, None, None, None]
    else:
        raise ValueError("reflection supports scalar curvature and rank-two metric/GR fields only")
    max_error = 0.
    max_magnitude = 0.
    prefix = source.ndim - 4
    for time_index in range(len(coords[0])):
        slab = np.asarray(source[(slice(None),)*prefix + (time_index, slice(None),
                                                          slice(None), slice(None))])
        if not np.all(np.isfinite(slab)):
            raise ValueError("reflection field contains non-finite values")
        reflected = np.flip(slab, axis=prefix + axis - 1)
        residual = slab - reflected if signs is None else slab - signs * reflected
        max_error = max(max_error, float(np.max(np.abs(residual))))
        max_magnitude = max(max_magnitude, float(np.max(np.abs(slab))))
    return {"axis": axis, "field": field, "max_abs_residual": max_error,
            "relative_max_residual": max_error / max_magnitude if max_magnitude else 0.,
            "tensor_index_parity_applied": signs is not None,
            "convergence_certified": False}


__all__ = ["resolution_report", "compare_nested_results",
           "compare_three_resolutions", "spatial_reflection_report"]

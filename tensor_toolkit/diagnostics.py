"""Numerical validation diagnostics for tensor fields."""

from __future__ import annotations

import numpy as np


def _chunk_slices(value: np.ndarray, chunk_points: int = 8):
    if value.ndim < 3:
        yield (...,)
        return
    grid_axis = 2
    length = value.shape[grid_axis]
    for start in range(0, length, max(1, int(chunk_points))):
        stop = min(length, start + max(1, int(chunk_points)))
        index = [slice(None)] * value.ndim
        index[grid_axis] = slice(start, stop)
        yield tuple(index)


def symmetry_error(tensor: np.ndarray, *, chunk_points: int = 8) -> dict[str, float]:
    """Return absolute and relative symmetry residuals for the first two axes.

    Large arrays and memmaps are scanned in chunks so the diagnostic never
    materializes a full-grid ``tensor - tensor.T`` temporary.
    """
    value = np.asanyarray(tensor)
    if value.ndim < 2 or value.shape[0] != value.shape[1]:
        raise ValueError("symmetry diagnostics require equal first two tensor axes")
    absolute = 0.0
    scale = 0.0
    for index in _chunk_slices(value, chunk_points):
        chunk = np.asarray(value[index])
        residual = chunk - np.swapaxes(chunk, 0, 1)
        if chunk.size:
            absolute = max(absolute, float(np.max(np.abs(residual))))
            scale = max(scale, float(np.max(np.abs(chunk))))
    relative = absolute / scale if scale else 0.0
    return {"absolute": absolute, "relative": relative, "scale": scale}


def field_diagnostics(tensor: np.ndarray, *, chunk_points: int = 8) -> dict[str, object]:
    """Stream finite-value counts, extrema, RMS and rank-2 symmetry over an array.

    Handles memory-mapped fields without converting the entire tensor to RAM.
    """
    value = np.asanyarray(tensor)
    count = 0
    nonfinite = 0
    total_square = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    max_abs = 0.0
    for index in _chunk_slices(value, chunk_points):
        chunk = np.asarray(value[index])
        good = chunk[np.isfinite(chunk)]
        count += int(chunk.size)
        nonfinite += int(chunk.size - good.size)
        if good.size:
            minimum = min(minimum, float(np.min(good)))
            maximum = max(maximum, float(np.max(good)))
            max_abs = max(max_abs, float(np.max(np.abs(good))))
            total_square += float(np.sum(good * good, dtype=np.float64))
    finite_count = count - nonfinite
    out: dict[str, object] = {
        "finite": nonfinite == 0, "nonfinite_count": nonfinite,
        "value_count": count, "finite_count": finite_count,
        "min": minimum if finite_count else None,
        "max": maximum if finite_count else None,
        "max_abs": max_abs, "sum_of_squares": total_square,
        "rms": (total_square / finite_count) ** 0.5 if finite_count else None,
    }
    # Only a full 4-D-grid rank-2 field (or a single 4x4 matrix) has
    # the expected mu,nu symmetry. Gamma (rho,mu,nu) and Riemann
    # (rho,sigma,mu,nu) also begin with 4x4 but MUST NOT use this check.
    if value.ndim in (2, 6) and value.shape[:2] == (4, 4):
        out["symmetry"] = symmetry_error(value, chunk_points=chunk_points) if nonfinite == 0 else None
    return out


def merge_field_diagnostics(current, local):
    """Merge diagnostics over disjoint chunks, retaining exact streaming RMS."""
    if current is None:
        return local
    total = int(current.get("value_count", 0)) + int(local.get("value_count", 0))
    finite_count = int(current.get("finite_count", 0)) + int(local.get("finite_count", 0))
    sum_square = float(current.get("sum_of_squares", 0.)) + float(local.get("sum_of_squares", 0.))
    lows = [v for v in (current.get("min"), local.get("min")) if v is not None]
    highs = [v for v in (current.get("max"), local.get("max")) if v is not None]
    out = {
        "finite": bool(current.get("finite", True) and local.get("finite", True)),
        "nonfinite_count": int(current.get("nonfinite_count", 0)) + int(local.get("nonfinite_count", 0)),
        "value_count": total, "finite_count": finite_count,
        "min": min(lows) if lows else None, "max": max(highs) if highs else None,
        "max_abs": max(float(current.get("max_abs", 0.)), float(local.get("max_abs", 0.))),
        "sum_of_squares": sum_square,
        "rms": (sum_square / finite_count) ** .5 if finite_count else None,
    }
    if current.get("symmetry") is not None and local.get("symmetry") is not None:
        absolute = max(float(current["symmetry"]["absolute"]), float(local["symmetry"]["absolute"]))
        scale = max(float(current["symmetry"]["scale"]), float(local["symmetry"]["scale"]))
        out["symmetry"] = {"absolute": absolute, "scale": scale,
                           "relative": absolute / scale if scale else 0.0}
    return out


def validation_status(
    diagnostics: dict[str, dict[str, object]],
    *,
    warning_relative: float = 1e-3,
) -> str:
    """Classify diagnostics as PASS, WARNING, or FAIL."""
    for item in diagnostics.values():
        if not item.get("finite", True):
            return "FAIL"
    for name in ("metric", "einstein", "stress_energy"):
        item = diagnostics.get(name)
        if not item or not item.get("symmetry"):
            continue
        relative = float(item["symmetry"]["relative"])
        limit = 1e-12 if name == "metric" else warning_relative
        if relative > limit:
            return "WARNING"
    return "PASS"


def result_diagnostics(
    metric: np.ndarray, fields: dict[str, np.ndarray]
) -> dict[str, object]:
    """Create diagnostics for the metric and available derived rank-2 fields."""
    diagnostics: dict[str, dict[str, object]] = {
        "metric": field_diagnostics(metric)
    }
    for name in ("einstein", "stress_energy"):
        if name in fields:
            diagnostics[name] = field_diagnostics(fields[name])
    return {
        "fields": diagnostics,
        "status": validation_status(diagnostics),
        "warning_relative_threshold": 1e-3,
    }


__all__ = [
    "symmetry_error",
    "field_diagnostics",
    "merge_field_diagnostics",
    "validation_status",
    "result_diagnostics",
]

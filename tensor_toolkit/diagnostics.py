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
    """Return finite-value and, for rank-2 tensors, symmetry diagnostics safely."""
    value = np.asanyarray(tensor)
    finite = True
    max_abs = 0.0
    for index in _chunk_slices(value, chunk_points):
        chunk = np.asarray(value[index])
        finite = finite and bool(np.all(np.isfinite(chunk)))
        if chunk.size:
            max_abs = max(max_abs, float(np.max(np.abs(chunk))))
    out: dict[str, object] = {"finite": finite, "max_abs": max_abs}
    if value.ndim >= 2 and value.shape[:2] == (4, 4):
        out["symmetry"] = symmetry_error(value, chunk_points=chunk_points)
    return out


def merge_field_diagnostics(current, local):
    """Merge diagnostics computed over disjoint spatial chunks."""
    if current is None:
        return local
    out = {
        "finite": bool(current.get("finite", True) and local.get("finite", True)),
        "max_abs": max(float(current.get("max_abs", 0.0)), float(local.get("max_abs", 0.0))),
    }
    if "symmetry" in current and "symmetry" in local:
        absolute = max(float(current["symmetry"]["absolute"]), float(local["symmetry"]["absolute"]))
        scale = max(float(current["symmetry"]["scale"]), float(local["symmetry"]["scale"]))
        out["symmetry"] = {
            "absolute": absolute,
            "scale": scale,
            "relative": absolute / scale if scale else 0.0,
        }
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
        if not item or "symmetry" not in item:
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

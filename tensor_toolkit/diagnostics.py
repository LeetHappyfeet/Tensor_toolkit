"""Numerical validation diagnostics for tensor fields."""

from __future__ import annotations

import numpy as np


def symmetry_error(tensor: np.ndarray) -> dict[str, float]:
    """Return absolute and relative symmetry residuals for the first two axes."""
    value = np.asarray(tensor)
    if value.ndim < 2 or value.shape[0] != value.shape[1]:
        raise ValueError("symmetry diagnostics require equal first two tensor axes")
    residual = value - np.swapaxes(value, 0, 1)
    absolute = float(np.max(np.abs(residual))) if value.size else 0.0
    scale = float(np.max(np.abs(value))) if value.size else 0.0
    relative = absolute / scale if scale else 0.0
    return {"absolute": absolute, "relative": relative, "scale": scale}


def field_diagnostics(tensor: np.ndarray) -> dict[str, object]:
    """Return finite-value and, for rank-2 tensors, symmetry diagnostics."""
    value = np.asarray(tensor)
    out: dict[str, object] = {
        "finite": bool(np.all(np.isfinite(value))),
        "max_abs": float(np.max(np.abs(value))) if value.size else 0.0,
    }
    if value.ndim >= 2 and value.shape[:2] == (4, 4):
        out["symmetry"] = symmetry_error(value)
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
    "validation_status",
    "result_diagnostics",
]

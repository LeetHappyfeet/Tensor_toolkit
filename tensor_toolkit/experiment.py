"""Headless experiment definition with memory-aware CPU execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensor_toolkit.backends import require_backend
from tensor_toolkit.diagnostics import field_diagnostics, validation_status
from tensor_toolkit.memory import memory_plan
from tensor_toolkit.metrics import Metric
from tensor_toolkit.reference.geometry import (
    christoffel_symbols,
    inverse_metric,
    ricci_from_christoffel,
    ricci_from_riemann,
    ricci_scalar_from_ricci,
    riemann_from_christoffel,
    stress_energy_from_einstein,
)

SUPPORTED_OUTPUTS = frozenset({
    "metric", "inverse_metric", "christoffel", "riemann",
    "ricci", "ricci_scalar", "einstein", "stress_energy",
})


@dataclass(frozen=True)
class Axis:
    start: float
    stop: float
    points: int

    def values(self) -> np.ndarray:
        if self.points < 3:
            raise ValueError("each experiment axis requires at least 3 points")
        if self.stop <= self.start:
            raise ValueError("axis stop must be greater than start")
        return np.linspace(self.start, self.stop, self.points, dtype=np.float64)


@dataclass(frozen=True)
class Experiment:
    metric: Metric
    axes: tuple[Axis, Axis, Axis, Axis]
    outputs: frozenset[str] = field(default_factory=lambda: frozenset({"metric", "einstein"}))
    stress_energy_units: str = "geometrized"
    backend: str = "cpu"
    memory_mode: str = "auto"
    tile_points: int = 8
    memory_limit_fraction: float = 0.65

    def axis_values(self) -> tuple[np.ndarray, ...]:
        return tuple(axis.values() for axis in self.axes)

    def coordinates(self) -> tuple[np.ndarray, ...]:
        return tuple(np.meshgrid(*self.axis_values(), indexing="ij", sparse=False))

    def spacings(self) -> tuple[float, ...]:
        vectors = self.axis_values()
        return tuple(float(v[1] - v[0]) for v in vectors)


@dataclass
class ExperimentResult:
    metric_name: str
    coordinates: tuple[str, str, str, str]
    axis_values: tuple[np.ndarray, ...]
    fields: dict[str, np.ndarray]
    metadata: dict[str, object]


def _need(outputs, *names: str) -> bool:
    return any(name in outputs for name in names)


def _compute_fields(metric: np.ndarray, spacings, outputs, *, units: str) -> dict[str, np.ndarray]:
    """Compute only requested persistent fields, streaming internal curvature where possible."""
    outputs = frozenset(outputs)
    fields: dict[str, np.ndarray] = {}
    if "metric" in outputs:
        fields["metric"] = metric

    downstream_inverse = _need(
        outputs, "inverse_metric", "christoffel", "riemann", "ricci",
        "ricci_scalar", "einstein", "stress_energy"
    )
    inverse = inverse_metric(metric) if downstream_inverse else None
    if "inverse_metric" in outputs:
        fields["inverse_metric"] = inverse

    downstream_gamma = _need(
        outputs, "christoffel", "riemann", "ricci", "ricci_scalar", "einstein", "stress_energy"
    )
    gamma = christoffel_symbols(metric, spacings, inverse=inverse) if downstream_gamma else None
    if "christoffel" in outputs:
        fields["christoffel"] = gamma

    need_ricci = _need(outputs, "ricci", "ricci_scalar", "einstein", "stress_energy")
    riemann = None
    if "riemann" in outputs:
        riemann = riemann_from_christoffel(gamma, spacings)
        fields["riemann"] = riemann

    ricci = None
    if need_ricci:
        # If Riemann was explicitly requested, reuse it. Otherwise contract directly
        # from Christoffel and never allocate the 256-component Riemann field.
        ricci = ricci_from_riemann(riemann) if riemann is not None else ricci_from_christoffel(gamma, spacings)
        if "ricci" in outputs:
            fields["ricci"] = ricci

    scalar = None
    if _need(outputs, "ricci_scalar", "einstein", "stress_energy"):
        scalar = ricci_scalar_from_ricci(metric, ricci, inverse=inverse)
        if "ricci_scalar" in outputs:
            fields["ricci_scalar"] = scalar

    einstein = None
    if _need(outputs, "einstein", "stress_energy"):
        einstein = ricci - 0.5 * metric * scalar
        if "einstein" in outputs:
            fields["einstein"] = einstein

    if "stress_energy" in outputs:
        fields["stress_energy"] = stress_energy_from_einstein(einstein, units=units)
    return fields


def _diagnostics(metric: np.ndarray | None, fields: dict[str, np.ndarray], metric_diag=None):
    diagnostics: dict[str, dict[str, object]] = {}
    if metric is not None:
        diagnostics["metric"] = field_diagnostics(metric)
    elif metric_diag is not None:
        diagnostics["metric"] = metric_diag
    for name in ("einstein", "stress_energy"):
        if name in fields:
            diagnostics[name] = field_diagnostics(fields[name])
    return {
        "fields": diagnostics,
        "status": validation_status(diagnostics),
        "warning_relative_threshold": 1e-3,
    }


def _run_in_memory(experiment: Experiment, axis_values, spacings):
    coordinate_grid = tuple(np.meshgrid(*axis_values, indexing="ij", sparse=False))
    metric = experiment.metric.evaluate(coordinate_grid)
    del coordinate_grid
    fields = _compute_fields(metric, spacings, experiment.outputs, units=experiment.stress_energy_units)
    diagnostics = _diagnostics(metric, fields)
    return fields, diagnostics


def _merge_metric_diag(current, local):
    if current is None:
        return local
    out = dict(current)
    out["finite"] = bool(current.get("finite", True) and local.get("finite", True))
    out["max_abs"] = max(float(current.get("max_abs", 0.0)), float(local.get("max_abs", 0.0)))
    if "symmetry" in current and "symmetry" in local:
        absolute = max(float(current["symmetry"]["absolute"]), float(local["symmetry"]["absolute"]))
        scale = max(float(current["symmetry"]["scale"]), float(local["symmetry"]["scale"]))
        out["symmetry"] = {
            "absolute": absolute,
            "scale": scale,
            "relative": absolute / scale if scale else 0.0,
        }
    return out


def _crop_core(value: np.ndarray, core_start: int, core_stop: int, halo_start: int):
    prefix = value.ndim - 4
    local_start = core_start - halo_start
    local_stop = core_stop - halo_start
    index = [slice(None)] * prefix + [
        slice(local_start, local_stop), slice(None), slice(None), slice(None)
    ]
    return value[tuple(index)]


def _allocate_global(local: np.ndarray, grid_shape):
    prefix = local.shape[: local.ndim - 4]
    return np.empty((*prefix, *grid_shape), dtype=local.dtype)


def _run_tiled(experiment: Experiment, axis_values, spacings, *, halo: int = 3):
    """Execute t-slabs with a halo that preserves nested second-order edge stencils."""
    grid_shape = tuple(len(axis) for axis in axis_values)
    total_t = grid_shape[0]
    tile_points = max(1, int(experiment.tile_points))
    fields: dict[str, np.ndarray] = {}
    metric_diag = None

    for core_start in range(0, total_t, tile_points):
        core_stop = min(total_t, core_start + tile_points)
        halo_start = max(0, core_start - halo)
        halo_stop = min(total_t, core_stop + halo)
        local_axes = (axis_values[0][halo_start:halo_stop], *axis_values[1:])
        coordinate_grid = tuple(np.meshgrid(*local_axes, indexing="ij", sparse=False))
        metric = experiment.metric.evaluate(coordinate_grid)
        del coordinate_grid
        metric_diag = _merge_metric_diag(metric_diag, field_diagnostics(metric))
        local_fields = _compute_fields(
            metric, spacings, experiment.outputs, units=experiment.stress_energy_units
        )
        for name, local in local_fields.items():
            if name not in fields:
                fields[name] = _allocate_global(local, grid_shape)
            cropped = _crop_core(local, core_start, core_stop, halo_start)
            target_prefix = fields[name].ndim - 4
            target_index = [slice(None)] * target_prefix + [
                slice(core_start, core_stop), slice(None), slice(None), slice(None)
            ]
            fields[name][tuple(target_index)] = cropped

    diagnostics = _diagnostics(fields.get("metric"), fields, metric_diag=metric_diag)
    return fields, diagnostics


def run_experiment(experiment: Experiment) -> ExperimentResult:
    require_backend(experiment.backend)
    unknown = set(experiment.outputs) - SUPPORTED_OUTPUTS
    if unknown:
        raise ValueError(f"unsupported experiment outputs: {sorted(unknown)}")
    if not experiment.outputs:
        raise ValueError("experiment must request at least one output")

    axis_values = experiment.axis_values()
    spacings = tuple(float(v[1] - v[0]) for v in axis_values)
    grid_shape = tuple(len(v) for v in axis_values)
    plan = memory_plan(
        grid_shape,
        experiment.outputs,
        requested_mode=experiment.memory_mode,
        tile_points=experiment.tile_points,
        limit_fraction=experiment.memory_limit_fraction,
    )

    if plan["selected_mode"] == "tiled":
        fields, diagnostics = _run_tiled(
            experiment, axis_values, spacings, halo=int(plan["halo"])
        )
    else:
        fields, diagnostics = _run_in_memory(experiment, axis_values, spacings)

    return ExperimentResult(
        metric_name=experiment.metric.name,
        coordinates=experiment.metric.coordinates,
        axis_values=axis_values,
        fields=fields,
        metadata={
            "spacings": spacings,
            "shape": grid_shape,
            "stress_energy_units": experiment.stress_energy_units,
            "backend": "cpu",
            "diagnostics": diagnostics,
            "memory": plan,
        },
    )


__all__ = ["Axis", "Experiment", "ExperimentResult", "SUPPORTED_OUTPUTS", "run_experiment"]

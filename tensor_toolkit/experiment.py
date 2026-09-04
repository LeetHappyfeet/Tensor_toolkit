"""Headless experiment definition with memory-aware CPU execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import shutil

import numpy as np

from tensor_toolkit.backends import require_backend
from tensor_toolkit.diagnostics import (
    field_diagnostics,
    merge_field_diagnostics,
    validation_status,
)
from tensor_toolkit.memory import memory_plan, output_bytes, select_storage_mode
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
from tensor_toolkit.storage import DiskFieldStore

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
        # Sparse coordinates broadcast to the full grid inside metric definitions
        # without allocating four redundant dense coordinate volumes.
        return tuple(np.meshgrid(*self.axis_values(), indexing="ij", sparse=True))

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


def compute_tensor_fields(metric: np.ndarray, spacings, outputs, *, units: str = "geometrized") -> dict[str, np.ndarray]:
    """Compute requested GR fields from an already-sampled 4-D metric grid.

    This is the reusable field-evaluation entry point for experiments and for
    trajectory-centered spacetime stencils.
    """
    outputs = frozenset(outputs)
    unknown = set(outputs) - SUPPORTED_OUTPUTS
    if unknown:
        raise ValueError(f"unsupported tensor outputs: {sorted(unknown)}")
    if not outputs:
        raise ValueError("at least one tensor output is required")
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


def _diagnostics(metric: np.ndarray | None, fields: dict[str, np.ndarray], metric_diag=None, field_diags=None):
    diagnostics: dict[str, dict[str, object]] = {}
    if metric is not None:
        diagnostics["metric"] = field_diagnostics(metric)
    elif metric_diag is not None:
        diagnostics["metric"] = metric_diag
    if field_diags:
        diagnostics.update(field_diags)
    else:
        for name in ("einstein", "stress_energy"):
            if name in fields:
                diagnostics[name] = field_diagnostics(fields[name])
    return {
        "fields": diagnostics,
        "status": validation_status(diagnostics),
        "warning_relative_threshold": 1e-3,
    }


def _sparse_coordinates(axis_values):
    return tuple(np.meshgrid(*axis_values, indexing="ij", sparse=True))


def _run_in_memory(experiment: Experiment, axis_values, spacings):
    coordinate_grid = _sparse_coordinates(axis_values)
    metric = experiment.metric.evaluate(coordinate_grid)
    del coordinate_grid
    fields = compute_tensor_fields(metric, spacings, experiment.outputs, units=experiment.stress_energy_units)
    diagnostics = _diagnostics(metric, fields)
    return fields, diagnostics


def _crop_core_nd(value: np.ndarray, core_starts, core_stops, halo_starts):
    prefix = value.ndim - 4
    index = [slice(None)] * prefix
    for axis in range(4):
        local_start = int(core_starts[axis]) - int(halo_starts[axis])
        local_stop = int(core_stops[axis]) - int(halo_starts[axis])
        index.append(slice(local_start, local_stop))
    return value[tuple(index)]


def _allocate_global(local: np.ndarray, grid_shape, *, name: str, store: DiskFieldStore | None):
    prefix = local.shape[: local.ndim - 4]
    shape = (*prefix, *grid_shape)
    if store is not None:
        return store.allocate(name, shape, dtype=local.dtype)
    return np.empty(shape, dtype=local.dtype)


def _block_starts(grid_shape, block_shape):
    ranges = [range(0, int(grid_shape[i]), int(block_shape[i])) for i in range(4)]
    yield from product(*ranges)


def _run_tiled(
    experiment: Experiment,
    axis_values,
    spacings,
    *,
    block_shape,
    halo: int = 3,
    store: DiskFieldStore | None = None,
):
    """Execute 4-D core blocks with halo cells and optional direct-to-disk output."""
    grid_shape = tuple(len(axis) for axis in axis_values)
    block_shape = tuple(max(1, min(int(block_shape[i]), grid_shape[i])) for i in range(4))
    fields: dict[str, np.ndarray] = {}
    metric_diag = None
    output_diags: dict[str, dict[str, object]] = {}

    for core_starts in _block_starts(grid_shape, block_shape):
        core_stops = tuple(
            min(grid_shape[i], core_starts[i] + block_shape[i]) for i in range(4)
        )
        halo_starts = tuple(max(0, core_starts[i] - halo) for i in range(4))
        halo_stops = tuple(min(grid_shape[i], core_stops[i] + halo) for i in range(4))
        local_axes = tuple(
            axis_values[i][halo_starts[i]:halo_stops[i]] for i in range(4)
        )
        coordinate_grid = _sparse_coordinates(local_axes)
        metric = experiment.metric.evaluate(coordinate_grid)
        del coordinate_grid

        metric_core = _crop_core_nd(metric, core_starts, core_stops, halo_starts)
        metric_diag = merge_field_diagnostics(metric_diag, field_diagnostics(metric_core))

        local_fields = compute_tensor_fields(
            metric, spacings, experiment.outputs, units=experiment.stress_energy_units
        )
        for name, local in local_fields.items():
            cropped = _crop_core_nd(local, core_starts, core_stops, halo_starts)
            if name in ("einstein", "stress_energy"):
                output_diags[name] = merge_field_diagnostics(
                    output_diags.get(name), field_diagnostics(cropped)
                )
            if name not in fields:
                fields[name] = _allocate_global(local, grid_shape, name=name, store=store)
            prefix = fields[name].ndim - 4
            target_index = [slice(None)] * prefix + [
                slice(core_starts[i], core_stops[i]) for i in range(4)
            ]
            fields[name][tuple(target_index)] = cropped

        del local_fields, metric, metric_core

    if store is not None:
        store.flush()
    diagnostics = _diagnostics(
        fields.get("metric") if store is None else None,
        fields,
        metric_diag=metric_diag,
        field_diags=output_diags,
    )
    return fields, diagnostics


def _check_disk_space(output_path, required_bytes: int) -> None:
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    needed = int(required_bytes * 1.10)
    if free < needed:
        raise OSError(
            f"disk-backed output needs about {needed / 1024**3:.2f} GiB free, "
            f"but only {free / 1024**3:.2f} GiB is available at {path}"
        )


def run_experiment(
    experiment: Experiment,
    *,
    output_path=None,
    storage_mode: str = "memory",
) -> ExperimentResult:
    """Run an experiment with optional disk-backed persistent output."""
    require_backend(experiment.backend)
    unknown = set(experiment.outputs) - SUPPORTED_OUTPUTS
    if unknown:
        raise ValueError(f"unsupported experiment outputs: {sorted(unknown)}")
    if not experiment.outputs:
        raise ValueError("experiment must request at least one output")

    axis_values = experiment.axis_values()
    spacings = tuple(float(v[1] - v[0]) for v in axis_values)
    grid_shape = tuple(len(v) for v in axis_values)
    selected_storage = select_storage_mode(
        grid_shape,
        experiment.outputs,
        requested_mode=storage_mode,
        output_path=output_path,
        limit_fraction=experiment.memory_limit_fraction,
    )
    disk_backed = selected_storage == "disk"
    if disk_backed:
        _check_disk_space(output_path, output_bytes(grid_shape, experiment.outputs))

    plan = memory_plan(
        grid_shape,
        experiment.outputs,
        requested_mode=experiment.memory_mode,
        tile_points=experiment.tile_points,
        limit_fraction=experiment.memory_limit_fraction,
        disk_backed=disk_backed,
    )

    store = DiskFieldStore(output_path) if disk_backed else None
    if plan["selected_mode"] == "tiled":
        fields, diagnostics = _run_tiled(
            experiment,
            axis_values,
            spacings,
            block_shape=tuple(plan["block_shape"]),
            halo=int(plan["halo"]),
            store=store,
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
            "storage": {
                "mode": selected_storage,
                "format": "npy-memmap" if disk_backed else "npz",
            },
        },
    )


__all__ = [
    "Axis", "Experiment", "ExperimentResult", "SUPPORTED_OUTPUTS",
    "compute_tensor_fields", "run_experiment",
]

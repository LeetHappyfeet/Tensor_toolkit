"""Headless experiment definition and single-pass execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensor_toolkit.backends import require_backend
from tensor_toolkit.diagnostics import result_diagnostics
from tensor_toolkit.metrics import Metric
from tensor_toolkit.reference.geometry import (
    christoffel_symbols,
    einstein_from_ricci,
    inverse_metric,
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

    def coordinates(self) -> tuple[np.ndarray, ...]:
        vectors = tuple(axis.values() for axis in self.axes)
        return tuple(np.meshgrid(*vectors, indexing="ij", sparse=False))

    def spacings(self) -> tuple[float, ...]:
        vectors = tuple(axis.values() for axis in self.axes)
        return tuple(float(v[1] - v[0]) for v in vectors)


@dataclass
class ExperimentResult:
    metric_name: str
    coordinates: tuple[str, str, str, str]
    axis_values: tuple[np.ndarray, ...]
    fields: dict[str, np.ndarray]
    metadata: dict[str, object]


def run_experiment(experiment: Experiment) -> ExperimentResult:
    require_backend(experiment.backend)
    unknown = set(experiment.outputs) - SUPPORTED_OUTPUTS
    if unknown:
        raise ValueError(f"unsupported experiment outputs: {sorted(unknown)}")

    coordinate_grid = experiment.coordinates()
    axis_values = tuple(axis.values() for axis in experiment.axes)
    spacings = experiment.spacings()
    metric = experiment.metric.evaluate(coordinate_grid)
    fields: dict[str, np.ndarray] = {}

    def need(*names: str) -> bool:
        return any(name in experiment.outputs for name in names)

    if "metric" in experiment.outputs:
        fields["metric"] = metric

    inverse = inverse_metric(metric) if need(
        "inverse_metric", "christoffel", "riemann", "ricci",
        "ricci_scalar", "einstein", "stress_energy"
    ) else None
    if "inverse_metric" in experiment.outputs:
        fields["inverse_metric"] = inverse

    christoffel = christoffel_symbols(metric, spacings) if need(
        "christoffel", "riemann", "ricci", "ricci_scalar", "einstein", "stress_energy"
    ) else None
    if "christoffel" in experiment.outputs:
        fields["christoffel"] = christoffel

    riemann = riemann_from_christoffel(christoffel, spacings) if need(
        "riemann", "ricci", "ricci_scalar", "einstein", "stress_energy"
    ) else None
    if "riemann" in experiment.outputs:
        fields["riemann"] = riemann

    ricci = ricci_from_riemann(riemann) if need(
        "ricci", "ricci_scalar", "einstein", "stress_energy"
    ) else None
    if "ricci" in experiment.outputs:
        fields["ricci"] = ricci

    scalar = ricci_scalar_from_ricci(metric, ricci) if need(
        "ricci_scalar", "einstein", "stress_energy"
    ) else None
    if "ricci_scalar" in experiment.outputs:
        fields["ricci_scalar"] = scalar

    einstein = einstein_from_ricci(metric, ricci) if need("einstein", "stress_energy") else None
    if "einstein" in experiment.outputs:
        fields["einstein"] = einstein
    if "stress_energy" in experiment.outputs:
        fields["stress_energy"] = stress_energy_from_einstein(
            einstein, units=experiment.stress_energy_units
        )

    return ExperimentResult(
        metric_name=experiment.metric.name,
        coordinates=experiment.metric.coordinates,
        axis_values=axis_values,
        fields=fields,
        metadata={
            "spacings": spacings,
            "shape": metric.shape[2:],
            "stress_energy_units": experiment.stress_energy_units,
            "backend": "cpu",
            "diagnostics": result_diagnostics(metric, fields),
        },
    )


__all__ = ["Axis", "Experiment", "ExperimentResult", "SUPPORTED_OUTPUTS", "run_experiment"]

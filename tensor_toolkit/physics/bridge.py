"""Bridge between classical trajectories and spacetime metric/tensor sampling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tensor_toolkit.experiment import compute_tensor_fields
from tensor_toolkit.metrics import Metric
from .trajectory import Trajectory


CoordinateTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MetricSamples:
    """Metric values evaluated along one body's trajectory."""
    coordinates: np.ndarray
    metric: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    body_name: str

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        metric = np.asarray(self.metric, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("coordinates must have shape (N, 4)")
        if metric.shape != (coordinates.shape[0], 4, 4):
            raise ValueError("metric must have shape (N, 4, 4)")


@dataclass(frozen=True)
class TensorEventSamples:
    """GR tensor fields evaluated at trajectory-centered spacetime events."""
    coordinates: np.ndarray
    fields: dict[str, np.ndarray]
    body_name: str


def trajectory_spacetime_coordinates(
    trajectory: Trajectory, *, body: str | int = 0, times=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arbitrary sampled (t,x,y,z) coordinates from a trajectory."""
    query_times = trajectory.times if times is None else np.asarray(times, dtype=np.float64)
    if query_times.ndim == 0:
        query_times = query_times.reshape(1)
    positions, velocities, accelerations = trajectory.sample(query_times, body=body)
    coordinates = np.column_stack((query_times, positions))
    return coordinates, velocities, accelerations


def _transform_coordinates(
    coordinates: np.ndarray,
    coordinate_transform: CoordinateTransform | None,
) -> np.ndarray:
    if coordinate_transform is None:
        return coordinates
    transformed = np.asarray(coordinate_transform(coordinates.copy()), dtype=np.float64)
    if transformed.shape != coordinates.shape:
        raise ValueError("coordinate_transform must return an array with shape (N, 4)")
    if not np.all(np.isfinite(transformed)):
        raise ValueError("coordinate_transform returned non-finite coordinates")
    return transformed


def sample_metric_along_trajectory(
    metric: Metric,
    trajectory: Trajectory,
    *,
    body: str | int = 0,
    times=None,
    coordinate_transform: CoordinateTransform | None = None,
) -> MetricSamples:
    """Evaluate an existing Tensor Toolkit metric at arbitrary trajectory events.

    Classical states default to SI coordinates. Metrics may use different units,
    especially geometrized coordinates, so callers can supply a coordinate
    transform before evaluation.
    """
    index = trajectory.body_index(body)
    coordinates, velocities, accelerations = trajectory_spacetime_coordinates(
        trajectory, body=index, times=times
    )
    metric_coordinates = _transform_coordinates(coordinates, coordinate_transform)
    t, x, y, z = (metric_coordinates[:, axis] for axis in range(4))
    evaluated = np.asarray(metric.evaluate((t, x, y, z)), dtype=np.float64)
    if evaluated.shape[:2] != (4, 4):
        raise ValueError(f"metric evaluator returned unexpected shape {evaluated.shape}")
    sampled_metric = np.moveaxis(evaluated, -1, 0)
    if sampled_metric.shape != (coordinates.shape[0], 4, 4):
        raise ValueError("metric evaluator did not produce one 4x4 metric per trajectory event")
    return MetricSamples(metric_coordinates, sampled_metric, velocities, accelerations, trajectory.body_names[index])


def sample_tensors_along_trajectory(
    metric: Metric,
    trajectory: Trajectory,
    *,
    spacings: tuple[float, float, float, float],
    outputs=frozenset({"metric", "einstein", "stress_energy"}),
    body: str | int = 0,
    times=None,
    units: str = "geometrized",
    coordinate_transform: CoordinateTransform | None = None,
) -> TensorEventSamples:
    """Evaluate GR fields on local 3x3x3x3 stencils centered on trajectory events.

    The existing reference pipeline differentiates a four-dimensional metric
    grid. For every requested trajectory event this function creates the
    smallest second-order stencil supported by that pipeline, evaluates the
    requested fields, and keeps only the center value.

    Spacings are expressed in the metric's coordinate units, after any supplied
    coordinate transform.
    """
    spacings = tuple(float(value) for value in spacings)
    if len(spacings) != 4 or any(value <= 0.0 for value in spacings):
        raise ValueError("spacings must contain four positive coordinate steps")

    index = trajectory.body_index(body)
    coordinates, _, _ = trajectory_spacetime_coordinates(trajectory, body=index, times=times)
    metric_coordinates = _transform_coordinates(coordinates, coordinate_transform)
    outputs = frozenset(outputs)
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in outputs}

    for event in metric_coordinates:
        axes = tuple(
            event[axis] + spacings[axis] * np.array([-1.0, 0.0, 1.0], dtype=np.float64)
            for axis in range(4)
        )
        coordinate_grid = tuple(np.meshgrid(*axes, indexing="ij", sparse=True))
        sampled_metric = metric.evaluate(coordinate_grid)
        fields = compute_tensor_fields(sampled_metric, spacings, outputs, units=units)

        center = (1, 1, 1, 1)
        for name, field in fields.items():
            prefix = field.ndim - 4
            accumulated[name].append(np.asarray(field[(slice(None),) * prefix + center]).copy())

    stacked = {name: np.stack(values, axis=0) for name, values in accumulated.items()}
    return TensorEventSamples(
        coordinates=metric_coordinates,
        fields=stacked,
        body_name=trajectory.body_names[index],
    )

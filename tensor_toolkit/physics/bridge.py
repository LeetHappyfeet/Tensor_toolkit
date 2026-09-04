"""Bridge between classical trajectories and spacetime metric sampling."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tensor_toolkit.metrics import Metric
from .trajectory import Trajectory

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

def sample_metric_along_trajectory(
    metric: Metric, trajectory: Trajectory, *, body: str | int = 0, times=None
) -> MetricSamples:
    """Evaluate an existing Tensor Toolkit metric at arbitrary trajectory events.

    This samples the metric only. Curvature tensors require derivatives of the
    metric and therefore a local spacetime stencil around each event.
    """
    index = trajectory.body_index(body)
    coordinates, velocities, accelerations = trajectory_spacetime_coordinates(
        trajectory, body=index, times=times
    )
    t, x, y, z = (coordinates[:, axis] for axis in range(4))
    evaluated = np.asarray(metric.evaluate((t, x, y, z)), dtype=np.float64)
    if evaluated.shape[:2] != (4, 4):
        raise ValueError(f"metric evaluator returned unexpected shape {evaluated.shape}")
    sampled_metric = np.moveaxis(evaluated, -1, 0)
    if sampled_metric.shape != (coordinates.shape[0], 4, 4):
        raise ValueError("metric evaluator did not produce one 4x4 metric per trajectory event")
    return MetricSamples(coordinates, sampled_metric, velocities, accelerations, trajectory.body_names[index])

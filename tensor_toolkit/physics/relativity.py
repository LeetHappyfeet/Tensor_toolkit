"""Relativity bridge for sampling classical trajectories in Schwarzschild spacetime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from tensor_toolkit.constants import SPEED_OF_LIGHT
from tensor_toolkit.metrics import SchwarzschildIsotropicMetric
from .bridge import TensorEventSamples, sample_tensors_along_trajectory
from .trajectory import Trajectory
from .worldline import Worldline


@dataclass(frozen=True)
class SchwarzschildTrajectorySamples:
    primary_name: str
    body_name: str
    times: np.ndarray
    coordinates: np.ndarray
    metric: np.ndarray
    proper_time_rate: np.ndarray
    proper_time: np.ndarray
    tensor_samples: TensorEventSamples | None
    worldline: Worldline


def _relative_events(
    trajectory: Trajectory,
    *,
    primary: str | int,
    body: str | int,
    times,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary_index = trajectory.body_index(primary)
    body_index = trajectory.body_index(body)
    if primary_index == body_index:
        raise ValueError("primary and body must be different")

    query = trajectory.times if times is None else np.asarray(times, dtype=np.float64)
    if query.ndim == 0:
        query = query.reshape(1)

    primary_pos, primary_vel, primary_acc = trajectory.sample(query, body=primary_index)
    body_pos, body_vel, body_acc = trajectory.sample(query, body=body_index)

    relative_pos = body_pos - primary_pos
    relative_vel = body_vel - primary_vel
    relative_acc = body_acc - primary_acc
    coordinates = np.column_stack((SPEED_OF_LIGHT * query, relative_pos))
    return coordinates, relative_vel, relative_acc


def sample_schwarzschild_trajectory(
    trajectory: Trajectory,
    *,
    primary: str | int,
    body: str | int,
    primary_mass: float,
    times=None,
    tensor_outputs=None,
    tensor_spacings: tuple[float, float, float, float] | None = None,
) -> SchwarzschildTrajectorySamples:
    """Sample a classical trajectory in a static Schwarzschild background.

    The selected primary defines the spatial origin. This is a one-way bridge:
    the input trajectory is not altered by GR. Coordinates passed to the metric
    are (ct, x, y, z) in metres.
    """
    metric = SchwarzschildIsotropicMetric(primary_mass)
    coordinates, relative_velocity, relative_acceleration = _relative_events(
        trajectory,
        primary=primary,
        body=body,
        times=times,
    )

    ct, x, y, z = (coordinates[:, axis] for axis in range(4))
    evaluated = np.asarray(metric.evaluate((ct, x, y, z)), dtype=np.float64)
    sampled_metric = np.moveaxis(evaluated, -1, 0)

    coordinate_velocity = np.column_stack(
        (
            np.full(len(coordinates), SPEED_OF_LIGHT, dtype=np.float64),
            relative_velocity,
        )
    )
    interval_rate2 = -np.einsum(
        "nij,ni,nj->n",
        sampled_metric,
        coordinate_velocity,
        coordinate_velocity,
    ) / SPEED_OF_LIGHT**2
    if np.any(interval_rate2 <= 0.0):
        raise ValueError("sampled trajectory is not timelike in Schwarzschild spacetime")
    proper_time_rate = np.sqrt(interval_rate2)

    full_coordinates, full_relative_velocity, _ = _relative_events(
        trajectory,
        primary=primary,
        body=body,
        times=trajectory.times,
    )
    ft, fx, fy, fz = (full_coordinates[:, axis] for axis in range(4))
    full_metric = np.moveaxis(
        np.asarray(metric.evaluate((ft, fx, fy, fz)), dtype=np.float64),
        -1,
        0,
    )
    full_coordinate_velocity = np.column_stack(
        (
            np.full(len(full_coordinates), SPEED_OF_LIGHT, dtype=np.float64),
            full_relative_velocity,
        )
    )
    full_rate2 = -np.einsum(
        "nij,ni,nj->n",
        full_metric,
        full_coordinate_velocity,
        full_coordinate_velocity,
    ) / SPEED_OF_LIGHT**2
    if np.any(full_rate2 <= 0.0):
        raise ValueError("trajectory becomes non-timelike in Schwarzschild spacetime")
    full_rate = np.sqrt(full_rate2)
    full_proper_time = np.zeros_like(trajectory.times)
    if len(trajectory.times) > 1:
        dt = np.diff(trajectory.times)
        increments = 0.5 * (full_rate[:-1] + full_rate[1:]) * dt
        full_proper_time[1:] = np.cumsum(increments)

    query_times = coordinates[:, 0] / SPEED_OF_LIGHT
    proper_time = np.interp(query_times, trajectory.times, full_proper_time)
    four_velocity = coordinate_velocity / proper_time_rate[:, np.newaxis]
    coordinate_acceleration = np.column_stack(
        (
            np.zeros(len(coordinates), dtype=np.float64),
            relative_acceleration,
        )
    )
    worldline = Worldline(
        parameter=query_times,
        coordinates=coordinates,
        tangent=coordinate_velocity,
        coordinate_acceleration=coordinate_acceleration,
        proper_time=proper_time,
        four_velocity=four_velocity,
        body_name=trajectory.body_names[trajectory.body_index(body)],
        metadata={
            "parameter": "coordinate_time",
            "coordinate_names": ("ct", "x", "y", "z"),
            "coordinate_units": "m",
            "proper_time_units": "s",
            "background": "SchwarzschildIsotropicMetric",
            "source": "classical_trajectory",
        },
    )

    tensor_samples = None
    if tensor_outputs is not None:
        if tensor_spacings is None:
            raise ValueError(
                "tensor_spacings are required when tensor_outputs are requested"
            )

        primary_index = trajectory.body_index(primary)

        def to_primary_centered_isotropic(events: np.ndarray) -> np.ndarray:
            seconds = events[:, 0]
            primary_positions, _, _ = trajectory.sample(seconds, body=primary_index)
            out = events.copy()
            out[:, 0] = SPEED_OF_LIGHT * seconds
            out[:, 1:] -= primary_positions
            return out

        tensor_samples = sample_tensors_along_trajectory(
            metric,
            trajectory,
            body=body,
            times=query_times,
            spacings=tensor_spacings,
            outputs=tensor_outputs,
            units="geometrized",
            coordinate_transform=to_primary_centered_isotropic,
        )

    return SchwarzschildTrajectorySamples(
        primary_name=trajectory.body_names[trajectory.body_index(primary)],
        body_name=trajectory.body_names[trajectory.body_index(body)],
        times=query_times,
        coordinates=coordinates,
        metric=sampled_metric,
        proper_time_rate=proper_time_rate,
        proper_time=proper_time,
        tensor_samples=tensor_samples,
        worldline=worldline,
    )


def save_schwarzschild_trajectory_samples(
    samples: SchwarzschildTrajectorySamples,
    output_path,
) -> Path:
    """Persist Schwarzschild sampling alongside a classical simulation result."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    arrays = {
        "times": samples.times,
        "coordinates": samples.coordinates,
        "metric": samples.metric,
        "proper_time_rate": samples.proper_time_rate,
        "proper_time": samples.proper_time,
        "four_velocity": samples.worldline.four_velocity,
    }
    if samples.tensor_samples is not None:
        for name, value in samples.tensor_samples.fields.items():
            arrays[f"field_{name}"] = value
    np.savez_compressed(output / "schwarzschild_samples.npz", **arrays)

    metadata = {
        "primary_name": samples.primary_name,
        "body_name": samples.body_name,
        "coordinates": ["ct", "x", "y", "z"],
        "coordinate_units": "m",
        "proper_time_units": "s",
        "worldline_parameter": "coordinate_time",
        "tensor_fields": (
            sorted(samples.tensor_samples.fields)
            if samples.tensor_samples is not None
            else []
        ),
    }
    with (output / "schwarzschild_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output

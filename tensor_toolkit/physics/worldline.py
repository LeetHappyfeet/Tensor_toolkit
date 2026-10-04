"""Spacetime worldline containers shared by classical and relativistic solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .trajectory import Trajectory


@dataclass(frozen=True)
class Worldline:
    """Sampled spacetime history for one physical body.

    Coordinates and tangent are intentionally generic. The parameter may be
    coordinate time, proper time, or another affine parameter depending on the
    solver that produced the worldline.
    """

    parameter: np.ndarray
    coordinates: np.ndarray
    tangent: np.ndarray
    body_name: str
    coordinate_acceleration: np.ndarray | None = None
    proper_time: np.ndarray | None = None
    four_velocity: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameter = np.asarray(self.parameter, dtype=np.float64)
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        tangent = np.asarray(self.tangent, dtype=np.float64)
        if parameter.ndim != 1 or parameter.size < 1:
            raise ValueError("worldline parameter must be a non-empty 1-D array")
        if np.any(np.diff(parameter) <= 0.0):
            raise ValueError("worldline parameter must be strictly increasing")
        expected = (parameter.size, 4)
        if coordinates.shape != expected:
            raise ValueError(f"coordinates must have shape {expected}")
        if tangent.shape != expected:
            raise ValueError(f"tangent must have shape {expected}")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(tangent)):
            raise ValueError("worldline coordinates and tangent must be finite")

        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "tangent", tangent)

        for name in ("coordinate_acceleration", "four_velocity"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.shape != expected or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite array with shape {expected}")
            object.__setattr__(self, name, array)

        if self.proper_time is not None:
            proper_time = np.asarray(self.proper_time, dtype=np.float64)
            if proper_time.shape != parameter.shape or not np.all(np.isfinite(proper_time)):
                raise ValueError("proper_time must be finite and match parameter shape")
            if np.any(np.diff(proper_time) < 0.0):
                raise ValueError("proper_time must be non-decreasing")
            object.__setattr__(self, "proper_time", proper_time)

        object.__setattr__(self, "metadata", dict(self.metadata))


def trajectory_worldline(
    trajectory: Trajectory,
    *,
    body: str | int = 0,
    times=None,
) -> Worldline:
    """Convert one classical body history into a coordinate-time worldline."""
    index = trajectory.body_index(body)
    query = trajectory.times if times is None else np.asarray(times, dtype=np.float64)
    if query.ndim == 0:
        query = query.reshape(1)
    positions, velocities, accelerations = trajectory.sample(query, body=index)
    coordinates = np.column_stack((query, positions))
    tangent = np.column_stack((np.ones(query.size, dtype=np.float64), velocities))
    coordinate_acceleration = np.column_stack(
        (np.zeros(query.size, dtype=np.float64), accelerations)
    )
    return Worldline(
        parameter=query,
        coordinates=coordinates,
        tangent=tangent,
        coordinate_acceleration=coordinate_acceleration,
        body_name=trajectory.body_names[index],
        metadata={
            "parameter": "coordinate_time",
            "coordinate_names": ("t", "x", "y", "z"),
            "coordinate_time_units": "s",
            "spatial_units": "m",
            "source": "classical_trajectory",
        },
    )

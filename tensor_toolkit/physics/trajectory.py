"""Trajectory results and interpolation utilities."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Trajectory:
    """Time history for one or more classical bodies."""
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    body_names: tuple[str, ...]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        velocities = np.asarray(self.velocities, dtype=np.float64)
        accelerations = np.asarray(self.accelerations, dtype=np.float64)
        if times.ndim != 1 or times.size < 1:
            raise ValueError("times must be a non-empty one-dimensional array")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")
        expected = (times.size, len(self.body_names), 3)
        for name, value in (("positions", positions), ("velocities", velocities), ("accelerations", accelerations)):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "accelerations", accelerations)

    def body_index(self, body: str | int) -> int:
        if isinstance(body, str):
            try:
                return self.body_names.index(body)
            except ValueError as exc:
                raise KeyError(f"unknown body {body!r}") from exc
        index = int(body)
        if not 0 <= index < len(self.body_names):
            raise IndexError("body index out of range")
        return index

    def sample(self, times, body: str | int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearly interpolate position, velocity, and acceleration at arbitrary times."""
        query = np.asarray(times, dtype=np.float64)
        if query.ndim == 0:
            query = query.reshape(1)
        if np.any(query < self.times[0]) or np.any(query > self.times[-1]):
            raise ValueError("sample times must lie within the trajectory time range")
        index = self.body_index(body)

        def interpolate(values: np.ndarray) -> np.ndarray:
            return np.stack([np.interp(query, self.times, values[:, index, axis]) for axis in range(3)], axis=-1)

        return interpolate(self.positions), interpolate(self.velocities), interpolate(self.accelerations)

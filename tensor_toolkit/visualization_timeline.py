"""Visualization-only time control and interpolation helpers.

This module never advances a physics solver. It maps wall-clock playback onto
already-computed coordinate/simulation times and samples public result objects.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time as _time

import numpy as np


@dataclass
class VisualizationTimeline:
    start: float = 0.0
    stop: float = 1.0
    current: float = 0.0
    playback_rate: float = 1.0
    playing: bool = False
    loop: bool = False

    def __post_init__(self):
        self.set_range(self.start, self.stop)
        self.seek(self.current)
        self._last_wall_time = None

    def set_range(self, start: float, stop: float) -> None:
        start, stop = float(start), float(stop)
        if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
            raise ValueError("timeline requires finite stop >= start")
        self.start, self.stop = start, stop
        self.current = float(np.clip(getattr(self, "current", start), start, stop))

    def seek(self, value: float) -> float:
        self.current = float(np.clip(float(value), self.start, self.stop))
        self._last_wall_time = None
        return self.current

    def play(self) -> None:
        self.playing = True
        self._last_wall_time = _time.perf_counter()

    def pause(self) -> None:
        self.playing = False
        self._last_wall_time = None

    def toggle(self) -> bool:
        if self.playing:
            self.pause()
        else:
            self.play()
        return self.playing

    def advance(self, wall_seconds: float | None = None) -> float:
        if not self.playing:
            return self.current
        if wall_seconds is None:
            now = _time.perf_counter()
            if self._last_wall_time is None:
                self._last_wall_time = now
                return self.current
            wall_seconds = now - self._last_wall_time
            self._last_wall_time = now
        target = self.current + float(wall_seconds) * float(self.playback_rate)
        if target > self.stop:
            if self.loop and self.stop > self.start:
                span = self.stop - self.start
                target = self.start + ((target - self.start) % span)
            else:
                target = self.stop
                self.pause()
        self.current = float(target)
        return self.current

    def nearest_index(self, samples) -> int:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("timeline samples must be a non-empty 1-D array")
        return int(np.argmin(np.abs(values - self.current)))


def sample_trajectory_positions(trajectory, time_value: float) -> np.ndarray:
    """Smoothly interpolate all body positions at visualization time."""
    t = float(np.clip(time_value, trajectory.times[0], trajectory.times[-1]))
    points = []
    for body in range(len(trajectory.body_names)):
        position, _velocity, _acceleration = trajectory.sample([t], body=body)
        points.append(position[0])
    return np.asarray(points, dtype=np.float64)


def trajectory_trail(trajectory, body: str | int, time_value: float, duration: float | None = None) -> np.ndarray:
    """Return authoritative sampled points up to time, optionally limited to a window."""
    index = trajectory.body_index(body)
    times = np.asarray(trajectory.times, dtype=np.float64)
    end = float(np.clip(time_value, times[0], times[-1]))
    start = times[0] if duration is None or duration <= 0 else max(times[0], end - float(duration))
    mask = (times >= start) & (times <= end)
    points = np.asarray(trajectory.positions[mask, index, :], dtype=np.float64)
    current = sample_trajectory_positions(trajectory, end)[index]
    if points.size == 0:
        return current.reshape(1, 3)
    if not np.allclose(points[-1], current):
        points = np.vstack((points, current))
    return points


class FrameCache:
    """Small LRU cache for visualization frames, including disk-backed results."""

    def __init__(self, capacity: int = 5):
        self.capacity = max(1, int(capacity))
        self._items = OrderedDict()

    def clear(self) -> None:
        self._items.clear()

    def get(self, key, factory):
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value
        value = factory()
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value

    def prefetch(self, keys, factory) -> None:
        for key in keys:
            self.get(key, lambda key=key: factory(key))


__all__ = [
    "VisualizationTimeline", "FrameCache",
    "sample_trajectory_positions", "trajectory_trail",
]

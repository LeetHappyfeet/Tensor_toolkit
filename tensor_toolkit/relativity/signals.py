"""Invariant relativistic signal and frequency-transfer observables."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .frames import normalize_timelike


def measured_frequency(metric: np.ndarray, observer_four_velocity, null_tangent) -> float:
    """Return frequency up to the photon's affine normalization: -u_mu k^mu."""

    metric = np.asarray(metric, dtype=np.float64)
    u = normalize_timelike(metric, observer_four_velocity)
    k = np.asarray(null_tangent, dtype=np.float64)
    if k.shape != (4,):
        raise ValueError("null_tangent must have shape (4,)")
    frequency = -float(np.einsum("m,mn,n->", u, metric, k))
    if frequency <= 0.0:
        raise ValueError("photon tangent is not future-directed for this observer")
    return frequency


@dataclass(frozen=True)
class FrequencyTransfer:
    emitted_frequency_factor: float
    received_frequency_factor: float
    frequency_ratio: float
    redshift: float


def frequency_transfer(
    emitter_metric: np.ndarray,
    emitter_four_velocity,
    emitted_null_tangent,
    receiver_metric: np.ndarray,
    receiver_four_velocity,
    received_null_tangent,
) -> FrequencyTransfer:
    """Compare the same null signal as measured at emission and reception."""

    emitted = measured_frequency(
        emitter_metric, emitter_four_velocity, emitted_null_tangent
    )
    received = measured_frequency(
        receiver_metric, receiver_four_velocity, received_null_tangent
    )
    ratio = received / emitted
    return FrequencyTransfer(
        emitted_frequency_factor=emitted,
        received_frequency_factor=received,
        frequency_ratio=ratio,
        redshift=emitted / received - 1.0,
    )


def coordinate_light_travel_time(coordinates: np.ndarray, *, time_scale: float = 1.0) -> float:
    """Return endpoint coordinate-time difference for a traced signal path."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or len(coordinates) < 2:
        raise ValueError("coordinates must have shape (N,4) with N >= 2")
    return float((coordinates[-1, 0] - coordinates[0, 0]) / float(time_scale))


def radar_distance(round_trip_proper_time: float, *, propagation_speed: float = 1.0) -> float:
    """Radar distance = signal speed times round-trip proper time / 2."""

    round_trip_proper_time = float(round_trip_proper_time)
    propagation_speed = float(propagation_speed)
    if round_trip_proper_time < 0.0 or propagation_speed <= 0.0:
        raise ValueError("proper time must be non-negative and propagation speed positive")
    return 0.5 * propagation_speed * round_trip_proper_time

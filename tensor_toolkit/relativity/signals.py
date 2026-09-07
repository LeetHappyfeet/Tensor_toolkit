"""Invariant relativistic signal and frequency-transfer observables."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .debug import debug_log
from .frames import normalize_timelike


def measured_frequency(
    metric: np.ndarray,
    observer_four_velocity,
    null_tangent,
    *,
    debug: bool = False,
) -> float:
    """Return frequency up to the photon's affine normalization: -u_mu k^mu."""

    metric = np.asarray(metric, dtype=np.float64)
    u = normalize_timelike(metric, observer_four_velocity)
    k = np.asarray(null_tangent, dtype=np.float64)
    if k.shape != (4,):
        raise ValueError("null_tangent must have shape (4,)")
    frequency = -float(np.einsum("m,mn,n->", u, metric, k))
    if frequency <= 0.0:
        raise ValueError("photon tangent is not future-directed for this observer")
    debug_log(
        debug,
        "signals",
        "measured_frequency",
        frequency=frequency,
        observer=np.array2string(u, precision=6),
        null_tangent=np.array2string(k, precision=6),
    )
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
    *,
    debug: bool = False,
) -> FrequencyTransfer:
    """Compare the same null signal as measured at emission and reception."""

    emitted = measured_frequency(
        emitter_metric, emitter_four_velocity, emitted_null_tangent, debug=debug
    )
    received = measured_frequency(
        receiver_metric, receiver_four_velocity, received_null_tangent, debug=debug
    )
    ratio = received / emitted
    result = FrequencyTransfer(
        emitted_frequency_factor=emitted,
        received_frequency_factor=received,
        frequency_ratio=ratio,
        redshift=emitted / received - 1.0,
    )
    debug_log(
        debug,
        "signals",
        "frequency_transfer",
        emitted=emitted,
        received=received,
        ratio=result.frequency_ratio,
        redshift=result.redshift,
    )
    return result


def coordinate_light_travel_time(
    coordinates: np.ndarray,
    *,
    time_scale: float = 1.0,
    debug: bool = False,
) -> float:
    """Return endpoint coordinate-time difference for a traced signal path."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or len(coordinates) < 2:
        raise ValueError("coordinates must have shape (N,4) with N >= 2")
    value = float((coordinates[-1, 0] - coordinates[0, 0]) / float(time_scale))
    debug_log(debug, "signals", "coordinate_light_travel_time", value=value)
    return value


def radar_distance(
    round_trip_proper_time: float,
    *,
    propagation_speed: float = 1.0,
    debug: bool = False,
) -> float:
    """Radar distance = signal speed times round-trip proper time / 2."""

    round_trip_proper_time = float(round_trip_proper_time)
    propagation_speed = float(propagation_speed)
    if round_trip_proper_time < 0.0 or propagation_speed <= 0.0:
        raise ValueError("proper time must be non-negative and propagation speed positive")
    value = 0.5 * propagation_speed * round_trip_proper_time
    debug_log(debug, "signals", "radar_distance", value=value)
    return value

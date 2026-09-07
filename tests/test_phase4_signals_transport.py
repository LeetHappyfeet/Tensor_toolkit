import numpy as np

from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.physics import Worldline
from tensor_toolkit.relativity import (
    SpacetimeSampler,
    coordinate_light_travel_time,
    frequency_transfer,
    measured_frequency,
    parallel_transport,
    radar_distance,
)


def test_invariant_frequency_transfer_is_unity_for_equal_minkowski_observers():
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    observer = np.array([1.0, 0.0, 0.0, 0.0])
    photon = np.array([1.0, 1.0, 0.0, 0.0])
    assert measured_frequency(metric, observer, photon) == 1.0
    transfer = frequency_transfer(
        metric,
        observer,
        photon,
        metric,
        observer,
        photon,
    )
    assert transfer.frequency_ratio == 1.0
    assert transfer.redshift == 0.0


def test_signal_time_and_radar_distance_helpers():
    coordinates = np.array([[0.0, 0, 0, 0], [10.0, 10, 0, 0]], dtype=float)
    assert coordinate_light_travel_time(coordinates) == 10.0
    assert radar_distance(8.0, propagation_speed=2.0) == 8.0


def test_parallel_transport_is_constant_in_minkowski():
    sampler = SpacetimeSampler(MinkowskiMetric(), (0.2, 0.2, 0.2, 0.2))
    parameter = np.linspace(0.0, 1.0, 5)
    coordinates = np.column_stack(
        (parameter, parameter, np.zeros((len(parameter), 2)))
    )
    tangent = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), (len(parameter), 1))
    worldline = Worldline(
        parameter=parameter,
        coordinates=coordinates,
        tangent=tangent,
        body_name="ray",
    )
    initial = np.array([0.0, 0.0, 1.0, 0.0])
    transported = parallel_transport(sampler, worldline, initial)
    assert np.allclose(transported, initial)

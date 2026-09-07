import numpy as np

from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.relativity import (
    SpacetimeSampler,
    comoving_frame,
    curvature_diagnostics,
    geodesic_deviation_acceleration,
    static_frame,
    tidal_eigensystem,
    tidal_tensor,
)


def test_minkowski_observer_frame_is_orthonormal_and_measures_vectors():
    sampler = SpacetimeSampler(MinkowskiMetric(), (0.1, 0.1, 0.1, 0.1))
    event = np.zeros(4)
    metric = sampler.metric_at(event)
    frame = comoving_frame(metric, event, [2.0, 0.0, 0.0, 0.0])
    gram = np.einsum("am,mn,bn->ab", frame.tetrad, metric, frame.tetrad)
    assert np.allclose(gram, np.diag([-1.0, 1.0, 1.0, 1.0]))
    measured = frame.measure_vector(np.array([1.0, 0.25, 0.0, 0.0]))
    assert np.allclose(measured, [1.0, 0.25, 0.0, 0.0])


def test_static_frame_and_local_stress_energy_projection():
    sampler = SpacetimeSampler(MinkowskiMetric(), (0.1, 0.1, 0.1, 0.1))
    event = np.zeros(4)
    frame = static_frame(sampler.metric_at(event), event)
    stress = np.diag([5.0, 2.0, 3.0, 4.0])
    local = frame.measure_stress_energy(stress)
    assert local.energy_density == 5.0
    assert np.allclose(local.momentum_density, 0.0)
    assert np.allclose(local.spatial_stress, np.diag([2.0, 3.0, 4.0]))


def test_minkowski_curvature_tides_and_deviation_are_zero():
    sampler = SpacetimeSampler(MinkowskiMetric(), (0.1, 0.1, 0.1, 0.1))
    event = np.zeros(4)
    metric = sampler.metric_at(event)
    riemann = sampler.riemann_at(event)
    diagnostics = curvature_diagnostics(metric, riemann)
    assert diagnostics.ricci_scalar == 0.0
    assert diagnostics.ricci_square == 0.0
    assert diagnostics.kretschmann == 0.0

    frame = static_frame(metric, event)
    tidal = tidal_tensor(metric, riemann, frame)
    values, vectors = tidal_eigensystem(tidal)
    assert np.allclose(tidal, 0.0)
    assert np.allclose(values, 0.0)
    assert vectors.shape == (3, 3)
    assert np.allclose(
        geodesic_deviation_acceleration(tidal, [100.0, 0.0, 0.0]),
        0.0,
    )

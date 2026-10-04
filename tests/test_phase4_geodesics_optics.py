import numpy as np

from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.relativity import (
    SpacetimeSampler,
    integrate_geodesic,
    static_frame,
    trace_null_ray,
    trace_ray_bundle,
)


def test_timelike_minkowski_geodesic_is_straight_and_preserves_norm():
    sampler = SpacetimeSampler(MinkowskiMetric(), (0.2, 0.2, 0.2, 0.2))
    result = integrate_geodesic(
        sampler,
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0, 0.0],
        affine_step=0.1,
        steps=10,
        causal_type="timelike",
    )
    expected = np.column_stack(
        (
            result.worldline.parameter,
            0.1 * result.worldline.parameter,
            np.zeros((len(result.worldline.parameter), 2)),
        )
    )
    assert np.allclose(result.worldline.coordinates, expected)
    assert result.max_normalization_error < 1e-14


def test_null_minkowski_geodesic_and_ray_bundle_are_straight():
    sampler = SpacetimeSampler(MinkowskiMetric(), (0.2, 0.2, 0.2, 0.2))
    event = np.zeros(4)
    frame = static_frame(sampler.metric_at(event), event, name="camera")
    ray = trace_null_ray(
        sampler,
        frame,
        [1.0, 0.0, 0.0],
        affine_step=0.25,
        steps=4,
    )
    assert ray.causal_type == "null"
    assert ray.max_normalization_error < 1e-14
    assert np.allclose(ray.worldline.coordinates[:, 0], ray.worldline.parameter)
    assert np.allclose(ray.worldline.coordinates[:, 1], ray.worldline.parameter)

    bundle = trace_ray_bundle(
        sampler,
        frame,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        affine_step=0.25,
        steps=2,
    )
    assert bundle.observer_name == "camera"
    assert len(bundle.rays) == 2

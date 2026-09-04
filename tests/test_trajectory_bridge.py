import numpy as np

from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.physics import (
    Body,
    System,
    sample_metric_along_trajectory,
    sample_tensors_along_trajectory,
    simulate,
)


def test_trajectory_can_be_sampled_at_arbitrary_spacetime_events():
    system = System([Body("probe", 0.0, [1, 2, 3], [4, 0, 0])])
    trajectory = simulate(system, duration=2.0, dt=0.25, gravitational_constant=1.0)
    samples = sample_metric_along_trajectory(
        MinkowskiMetric(), trajectory, body="probe", times=np.array([0.125, 0.5, 1.75])
    )
    assert samples.coordinates.shape == (3, 4)
    assert samples.metric.shape == (3, 4, 4)
    assert np.allclose(samples.coordinates[:, 1], [1.5, 3.0, 8.0])
    expected = np.diag([-1.0, 1.0, 1.0, 1.0])
    assert np.allclose(samples.metric, expected[np.newaxis, :, :])


def test_local_tensor_stencil_feeds_existing_gr_pipeline():
    system = System([Body("probe", 0.0, [0, 0, 0], [1, 0, 0])])
    trajectory = simulate(system, duration=1.0, dt=0.25, gravitational_constant=1.0)
    samples = sample_tensors_along_trajectory(
        MinkowskiMetric(),
        trajectory,
        body="probe",
        times=np.array([0.25, 0.75]),
        spacings=(0.01, 0.01, 0.01, 0.01),
        outputs={"metric", "einstein", "stress_energy"},
    )
    assert samples.coordinates.shape == (2, 4)
    assert samples.fields["metric"].shape == (2, 4, 4)
    assert samples.fields["einstein"].shape == (2, 4, 4)
    assert samples.fields["stress_energy"].shape == (2, 4, 4)
    assert np.allclose(samples.fields["einstein"], 0.0)
    assert np.allclose(samples.fields["stress_energy"], 0.0)


def test_coordinate_transform_is_applied_before_metric_sampling():
    system = System([Body("probe", 0.0, [10, 0, 0], [2, 0, 0])])
    trajectory = simulate(system, duration=1.0, dt=0.5, gravitational_constant=1.0)

    def scale(events):
        events[:, 0] *= 0.5
        events[:, 1:] *= 0.1
        return events

    samples = sample_metric_along_trajectory(
        MinkowskiMetric(), trajectory, body="probe", times=[0.5], coordinate_transform=scale
    )
    assert np.allclose(samples.coordinates[0], [0.25, 1.1, 0.0, 0.0])

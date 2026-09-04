import numpy as np
from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.physics import Body, System, sample_metric_along_trajectory, simulate

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

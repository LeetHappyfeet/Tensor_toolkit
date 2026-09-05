import numpy as np

from tensor_toolkit.physics.events import SimulationEvent
from tensor_toolkit.physics.trajectory import Trajectory
from tensor_toolkit.visualization_timeline import (
    FrameCache,
    VisualizationTimeline,
    sample_trajectory_positions,
    trajectory_trail,
)


def _trajectory():
    times = np.array([0.0, 10.0, 20.0])
    positions = np.zeros((3, 1, 3))
    positions[:, 0, 0] = [0.0, 10.0, 20.0]
    velocities = np.zeros_like(positions)
    velocities[:, 0, 0] = 1.0
    return Trajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=np.zeros_like(positions),
        body_names=("probe",),
        events=(SimulationEvent(time=10.0, kind="midpoint", bodies=("probe",)),),
    )


def test_timeline_playback_rate_and_pause():
    timeline = VisualizationTimeline(0.0, 100.0, 10.0, playback_rate=20.0)
    timeline.play()
    assert timeline.advance(0.5) == 20.0
    timeline.pause()
    assert timeline.advance(10.0) == 20.0


def test_timeline_loop_wraps_without_running_solver():
    timeline = VisualizationTimeline(0.0, 10.0, 9.0, playback_rate=4.0, loop=True)
    timeline.play()
    np.testing.assert_allclose(timeline.advance(1.0), 3.0)


def test_nearest_tensor_frame_is_discrete():
    timeline = VisualizationTimeline(0.0, 10.0, 6.0)
    assert timeline.nearest_index(np.array([0.0, 5.0, 10.0])) == 1


def test_trajectory_position_is_smoothly_interpolated():
    trajectory = _trajectory()
    position = sample_trajectory_positions(trajectory, 5.0)
    np.testing.assert_allclose(position[0], [5.0, 0.0, 0.0])


def test_trail_uses_stored_points_plus_current_interpolated_position():
    trajectory = _trajectory()
    trail = trajectory_trail(trajectory, "probe", 15.0, duration=10.0)
    np.testing.assert_allclose(trail[:, 0], [10.0, 15.0])


def test_frame_cache_is_bounded_lru():
    cache = FrameCache(2)
    cache.get("a", lambda: 1)
    cache.get("b", lambda: 2)
    assert cache.get("a", lambda: 99) == 1
    cache.get("c", lambda: 3)
    assert list(cache._items) == ["a", "c"]

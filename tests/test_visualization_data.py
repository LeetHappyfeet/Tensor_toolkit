import numpy as np

from tensor_toolkit.experiment import ExperimentResult
from tensor_toolkit.physics.events import SimulationEvent
from tensor_toolkit.physics.trajectory import Trajectory
from tensor_toolkit.physics.worldline import Worldline
from tensor_toolkit.visualization_data import (
    experiment_volume,
    trajectory_event_points,
    trajectory_polylines,
    worldline_polyline,
)


def test_experiment_volume_selects_time_and_component():
    field = np.zeros((4, 4, 2, 3, 4, 5), dtype=float)
    field[0, 1, 1] = 7.0
    result = ExperimentResult(
        metric_name="test",
        coordinates=("t", "x", "y", "z"),
        axis_values=(
            np.array([0.0, 1.0]),
            np.arange(3.0),
            np.arange(4.0),
            np.arange(5.0),
        ),
        fields={"stress_energy": field},
        metadata={"diagnostics": {"status": "PASS"}},
    )
    volume = experiment_volume(result, "stress_energy", component=(0, 1), time_index=1)
    assert volume.values.shape == (3, 4, 5)
    assert np.all(volume.values == 7.0)
    assert volume.metadata["time"] == 1.0


def test_trajectory_and_events_become_renderer_geometry():
    times = np.array([0.0, 1.0, 2.0])
    positions = np.zeros((3, 1, 3))
    positions[:, 0, 0] = times
    trajectory = Trajectory(
        times=times,
        positions=positions,
        velocities=np.zeros_like(positions),
        accelerations=np.zeros_like(positions),
        body_names=("probe",),
        events=(SimulationEvent(time=1.1, kind="marker", bodies=("probe",)),),
    )
    lines = trajectory_polylines(trajectory)
    assert len(lines) == 1
    assert lines[0].points.shape == (3, 3)
    events = trajectory_event_points(trajectory)
    assert events.points.shape == (1, 3)
    assert events.labels == ("marker",)


def test_worldline_projects_spatial_coordinates():
    parameter = np.array([0.0, 1.0])
    coordinates = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 4.0, 5.0, 6.0]])
    worldline = Worldline(
        parameter=parameter,
        coordinates=coordinates,
        tangent=np.ones((2, 4)),
        body_name="ray",
    )
    line = worldline_polyline(worldline)
    np.testing.assert_allclose(line.points, coordinates[:, 1:])

import numpy as np

from tensor_toolkit.constants import SPEED_OF_LIGHT
from tensor_toolkit.physics import (
    Body,
    System,
    sample_schwarzschild_trajectory,
    simulate,
    trajectory_worldline,
)


def test_classical_trajectory_converts_to_coordinate_time_worldline():
    system = System([
        Body("probe", 0.0, [1, 2, 3], [4, 0, 0]),
    ])
    trajectory = simulate(
        system,
        duration=1.0,
        dt=0.25,
        gravitational_constant=1.0,
    )
    worldline = trajectory_worldline(
        trajectory,
        body="probe",
        times=[0.25, 0.75],
    )
    assert worldline.coordinates.shape == (2, 4)
    assert np.allclose(worldline.coordinates[:, 0], [0.25, 0.75])
    assert np.allclose(worldline.tangent[:, 0], 1.0)
    assert np.allclose(worldline.tangent[:, 1], 4.0)
    assert np.allclose(worldline.coordinate_acceleration, 0.0)
    assert worldline.proper_time is None
    assert worldline.four_velocity is None


def test_schwarzschild_bridge_populates_proper_time_worldline_and_four_velocity():
    mass = 1.89813e27
    radius = 1.0e9
    system = System([
        Body("primary", mass, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [radius, 0, 0], [0, 1000, 0]),
    ])
    trajectory = simulate(
        system,
        duration=2.0,
        dt=1.0,
        gravitational_constant=1e-30,
    )
    samples = sample_schwarzschild_trajectory(
        trajectory,
        primary="primary",
        body="probe",
        primary_mass=mass,
        times=[0.0, 2.0],
    )
    worldline = samples.worldline
    assert np.allclose(worldline.coordinates, samples.coordinates)
    assert np.allclose(worldline.proper_time, samples.proper_time)
    assert worldline.four_velocity.shape == (2, 4)
    assert np.allclose(worldline.tangent[:, 0], SPEED_OF_LIGHT)
    assert np.allclose(
        worldline.four_velocity,
        worldline.tangent / samples.proper_time_rate[:, np.newaxis],
    )

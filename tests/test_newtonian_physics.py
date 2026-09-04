import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT
from tensor_toolkit.physics import (
    Body,
    System,
    circular_orbit_system,
    conservation_diagnostics,
    hyperbolic_flyby_system,
    newtonian_gravity_accelerations,
    simulate,
)


def test_two_body_acceleration_matches_newtonian_gravity():
    separation = 10.0
    masses = np.array([2.0, 3.0])
    positions = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    acceleration = newtonian_gravity_accelerations(positions, masses)
    assert np.isclose(acceleration[0, 0], GRAVITATIONAL_CONSTANT * masses[1] / separation**2)
    assert np.isclose(acceleration[1, 0], -GRAVITATIONAL_CONSTANT * masses[0] / separation**2)
    assert np.allclose(acceleration[:, 1:], 0.0)


def test_isolated_body_has_zero_acceleration():
    acceleration = newtonian_gravity_accelerations(
        np.array([[1.0, 2.0, 3.0]]), np.array([5.0])
    )
    assert np.array_equal(acceleration, np.zeros((1, 3)))


def test_rk4_preserves_simple_circular_orbit_reasonably():
    system = System([
        Body("primary", 1.0, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [1, 0, 0], [0, 1, 0]),
    ])
    trajectory = simulate(
        system,
        duration=2.0 * np.pi,
        dt=0.01,
        gravitational_constant=1.0,
    )
    final = trajectory.positions[-1, 1]
    assert np.linalg.norm(final - np.array([1.0, 0.0, 0.0])) < 2e-3


def test_velocity_verlet_preserves_circular_orbit_and_radius():
    system = circular_orbit_system(
        primary_mass=1.0,
        radius=1.0,
        gravitational_constant=1.0,
    )
    trajectory = simulate(
        system,
        duration=20.0 * np.pi,
        dt=0.01,
        method="verlet",
        gravitational_constant=1.0,
    )
    radius = np.linalg.norm(trajectory.positions[:, 1], axis=1)
    assert np.max(np.abs(radius - 1.0)) < 2e-4


def test_conservation_diagnostics_track_energy_and_angular_momentum():
    # Use two finite masses so the conserved system quantities are non-zero.
    system = System([
        Body("a", 1.0, [-0.5, 0, 0], [0, -np.sqrt(0.5), 0]),
        Body("b", 1.0, [0.5, 0, 0], [0, np.sqrt(0.5), 0]),
    ])
    trajectory = simulate(
        system,
        duration=4.0 * np.pi,
        dt=0.005,
        method="verlet",
        gravitational_constant=1.0,
    )
    diagnostics = conservation_diagnostics(
        trajectory,
        system.masses,
        gravitational_constant=1.0,
    )
    assert diagnostics.energy_relative_drift < 2e-4
    assert diagnostics.momentum_absolute_drift < 1e-12
    assert diagnostics.angular_momentum_relative_drift < 1e-12


def test_flyby_helper_builds_expected_incoming_state():
    system = hyperbolic_flyby_system(
        primary_mass=10.0,
        initial_distance=100.0,
        impact_parameter=5.0,
        incoming_speed=3.0,
    )
    assert np.allclose(system.positions[1], [-100.0, 5.0, 0.0])
    assert np.allclose(system.velocities[1], [3.0, 0.0, 0.0])
    assert system.masses[1] == 0.0

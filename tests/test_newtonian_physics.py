import numpy as np
from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT
from tensor_toolkit.physics import Body, System, newtonian_gravity_accelerations, simulate

def test_two_body_acceleration_matches_newtonian_gravity():
    separation = 10.0
    masses = np.array([2.0, 3.0])
    positions = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    acceleration = newtonian_gravity_accelerations(positions, masses)
    assert np.isclose(acceleration[0, 0], GRAVITATIONAL_CONSTANT * masses[1] / separation**2)
    assert np.isclose(acceleration[1, 0], -GRAVITATIONAL_CONSTANT * masses[0] / separation**2)
    assert np.allclose(acceleration[:, 1:], 0.0)

def test_isolated_body_has_zero_acceleration():
    acceleration = newtonian_gravity_accelerations(np.array([[1.0, 2.0, 3.0]]), np.array([5.0]))
    assert np.array_equal(acceleration, np.zeros((1, 3)))

def test_rk4_preserves_simple_circular_orbit_reasonably():
    system = System([
        Body("primary", 1.0, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [1, 0, 0], [0, 1, 0]),
    ])
    trajectory = simulate(system, duration=2.0 * np.pi, dt=0.01, gravitational_constant=1.0)
    final = trajectory.positions[-1, 1]
    assert np.linalg.norm(final - np.array([1.0, 0.0, 0.0])) < 2e-3

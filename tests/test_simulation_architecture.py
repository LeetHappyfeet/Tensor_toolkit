import numpy as np
import pytest

from tensor_toolkit.physics import (
    Body,
    CollisionDetector,
    CompositeDynamics,
    ConstantThrust,
    NewtonianGravity,
    System,
    simulate,
)


def test_explicit_newtonian_model_matches_legacy_default_path():
    system = System([
        Body("primary", 1.0, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [1, 0, 0], [0, 1, 0]),
    ])
    legacy = simulate(
        system,
        duration=2.0,
        dt=0.01,
        method="rk4",
        gravitational_constant=1.0,
    )
    explicit = simulate(
        system,
        duration=2.0,
        dt=0.01,
        method="rk4",
        dynamics=NewtonianGravity(gravitational_constant=1.0),
    )
    assert np.array_equal(legacy.times, explicit.times)
    assert np.allclose(legacy.positions, explicit.positions)
    assert np.allclose(legacy.velocities, explicit.velocities)
    assert np.allclose(legacy.accelerations, explicit.accelerations)


def test_constant_thrust_composes_with_gravity_model():
    system = System([
        Body("ship", 2.0, [0, 0, 0], [0, 0, 0]),
    ])
    dynamics = CompositeDynamics([
        NewtonianGravity(gravitational_constant=1.0),
        ConstantThrust({"ship": np.array([4.0, 0.0, 0.0])}),
    ])
    trajectory = simulate(
        system,
        duration=2.0,
        dt=0.1,
        method="rk4",
        dynamics=dynamics,
    )
    assert np.isclose(trajectory.velocities[-1, 0, 0], 4.0, atol=1e-12)
    assert np.isclose(trajectory.positions[-1, 0, 0], 4.0, atol=1e-12)
    assert np.allclose(trajectory.accelerations[:, 0, 0], 2.0)


def test_collision_detector_records_first_finite_radius_contact():
    system = System([
        Body("left", 0.0, [-1, 0, 0], [1, 0, 0], radius=0.5),
        Body("right", 0.0, [1, 0, 0], [-1, 0, 0], radius=0.5),
    ])
    trajectory = simulate(
        system,
        duration=1.0,
        dt=0.25,
        method="rk4",
        gravitational_constant=1.0,
        event_detectors=(CollisionDetector(),),
    )
    assert len(trajectory.events) == 1
    event = trajectory.events[0]
    assert event.kind == "collision"
    assert event.bodies == ("left", "right")
    assert np.isclose(event.time, 0.5)
    assert np.isclose(event.details["contact_distance"], 1.0)


def test_verlet_rejects_velocity_dependent_dynamics():
    class VelocityDependentModel:
        velocity_dependent = True

        def accelerations(self, t, positions, velocities, masses, system):
            del t, masses, system
            return -0.01 * velocities + np.zeros_like(positions)

    system = System([
        Body("probe", 1.0, [0, 0, 0], [1, 0, 0]),
    ])
    with pytest.raises(ValueError, match="velocity-independent"):
        simulate(
            system,
            duration=1.0,
            dt=0.1,
            method="verlet",
            dynamics=VelocityDependentModel(),
        )


def test_zero_radius_bodies_do_not_generate_collision_events():
    system = System([
        Body("a", 0.0, [-1, 0, 0], [1, 0, 0]),
        Body("b", 0.0, [1, 0, 0], [-1, 0, 0]),
    ])
    trajectory = simulate(
        system,
        duration=1.0,
        dt=0.25,
        gravitational_constant=1.0,
        event_detectors=(CollisionDetector(),),
    )
    assert trajectory.events == ()

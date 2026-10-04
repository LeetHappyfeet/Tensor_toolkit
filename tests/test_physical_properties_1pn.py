"""Compatibility, physical-component and central-body PN regression tests."""
import numpy as np
import pytest

from tensor_toolkit.physics import (
    Body, System, simulate, NewtonianGravity, CentralBody1PN,
    PhysicalProperties, RotationProperties, MaterialProperties,
    ElectromagneticProperties, PropulsionProperties,
)


def _system(v=0.0, secondary_mass=0.0):
    return System([
        Body("star", 10., [0, 0, 0], [0, 0, 0], radius=1.),
        Body("probe", secondary_mass, [1000, 0, 0], [0, v, 0]),
    ])


def test_body_backwards_compatible_and_composable_descriptive_properties():
    legacy = Body("plain", 1., [0, 0, 0], [0, 0, 0], 2.)
    assert legacy.radius == 2. and legacy.properties == PhysicalProperties()
    physical = PhysicalProperties(
        rotation=RotationProperties(inertia_tensor=np.diag([2., 3., 4.])),
        material=MaterialProperties(density_kg_m3=2., restitution=0.5),
        electromagnetic=ElectromagneticProperties(charge_coulombs=0.1),
        propulsion=PropulsionProperties(propellant_mass_kg=0.5, maximum_thrust_newtons=10.),
    )
    body = Body("ship", 2., [0, 0, 0], [0, 0, 0], properties=physical)
    assert body.properties.rotation.inertia_tensor.shape == (3, 3)
    assert body.properties.propulsion.propellant_mass_kg == 0.5
    assert not body.properties.rotation.inertia_tensor.flags.writeable
    with pytest.raises(ValueError, match="propellant mass"):
        Body("bad", 0.2, [0, 0, 0], [0, 0, 0], properties=physical)
    with pytest.raises(ValueError, match="unit normalized"):
        RotationProperties(orientation=[2, 0, 0, 0])


def test_one_pn_stationary_probe_acceleration_has_expected_radial_correction():
    system = _system()
    state = system.initial_state()
    model = CentralBody1PN(primary="star", gravitational_constant=1., speed_of_light=100.)
    a = model.accelerations(0., state.positions, state.velocities, state.masses, system)
    r, mu, c = 1000., 10., 100.
    expected = -mu/r**2 + 4*mu**2/(c**2*r**3)
    assert a.shape == (2, 3)
    assert np.array_equal(a[0], np.zeros(3))
    assert a[1, 0] == pytest.approx(expected)
    assert a[1, 1] == 0.


def test_one_pn_is_velocity_dependent_and_rejects_verlet():
    system = _system(v=1.)
    model = CentralBody1PN(primary="star", gravitational_constant=1., speed_of_light=100.)
    state = system.initial_state()
    a = model.accelerations(0, state.positions, state.velocities, state.masses, system)
    baseline = NewtonianGravity(gravitational_constant=1.).accelerations(
        0, state.positions, state.velocities, state.masses, system)
    assert np.linalg.norm(a[1]-baseline[1]) > 0
    with pytest.raises(ValueError, match="velocity-independent"):
        simulate(system, duration=2, dt=0.1, method="verlet", dynamics=model)
    result = simulate(system, duration=2, dt=0.1, method="rk4", dynamics=model)
    assert np.all(np.isfinite(result.positions))


def test_one_pn_requires_stationary_primary_and_passive_probes():
    system = _system(secondary_mass=0.1)
    state = system.initial_state()
    model = CentralBody1PN(primary="star", gravitational_constant=1., speed_of_light=100.)
    with pytest.raises(ValueError, match="exactly one"):
        model.accelerations(0, state.positions, state.velocities, state.masses, system)
    moving = System([
        Body("star", 10., [0, 0, 0], [1, 0, 0]),
        Body("probe", 0., [1000, 0, 0], [0, 0, 0]),
    ])
    initial = moving.initial_state()
    with pytest.raises(ValueError, match="stationary"):
        model.accelerations(0, initial.positions, initial.velocities, initial.masses, moving)

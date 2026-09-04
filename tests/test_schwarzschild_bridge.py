import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT
from tensor_toolkit.metrics import SchwarzschildIsotropicMetric
from tensor_toolkit.physics import (
    Body,
    System,
    sample_schwarzschild_trajectory,
    save_schwarzschild_trajectory_samples,
    simulate,
)


def test_schwarzschild_isotropic_metric_matches_closed_form():
    mass = 1.89813e27
    metric = SchwarzschildIsotropicMetric(mass)
    radius = 1.0e9
    g = metric.evaluate(
        (
            np.array([0.0]),
            np.array([radius]),
            np.array([0.0]),
            np.array([0.0]),
        )
    )
    m = GRAVITATIONAL_CONSTANT * mass / SPEED_OF_LIGHT**2
    u = m / (2.0 * radius)
    expected_g00 = -((1.0 - u) / (1.0 + u)) ** 2
    expected_space = (1.0 + u) ** 4

    assert np.isclose(g[0, 0, 0], expected_g00)
    assert np.isclose(g[1, 1, 0], expected_space)
    assert np.isclose(g[2, 2, 0], expected_space)
    assert np.isclose(g[3, 3, 0], expected_space)


def test_static_probe_proper_time_rate_matches_schwarzschild_lapse():
    mass = 1.89813e27
    radius = 1.0e9
    system = System([
        Body("primary", mass, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [radius, 0, 0], [0, 0, 0]),
    ])
    trajectory = simulate(
        system,
        duration=10.0,
        dt=1.0,
        gravitational_constant=1e-30,
    )
    samples = sample_schwarzschild_trajectory(
        trajectory,
        primary="primary",
        body="probe",
        primary_mass=mass,
        times=[0.0, 10.0],
    )

    m = GRAVITATIONAL_CONSTANT * mass / SPEED_OF_LIGHT**2
    u = m / (2.0 * radius)
    expected = (1.0 - u) / (1.0 + u)
    assert np.isclose(samples.proper_time_rate[0], expected, rtol=1e-13)
    assert np.isclose(samples.proper_time[-1], 10.0 * expected, rtol=1e-12)


def test_schwarzschild_bridge_can_call_existing_gr_tensor_pipeline():
    mass = 1.89813e27
    system = System([
        Body("primary", mass, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [1.0e9, 0, 0], [0, 1000, 0]),
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
        times=[1.0],
        tensor_outputs={"metric", "christoffel"},
        tensor_spacings=(1.0e6, 1.0e6, 1.0e6, 1.0e6),
    )

    assert samples.tensor_samples is not None
    assert samples.tensor_samples.fields["metric"].shape == (1, 4, 4)
    assert samples.tensor_samples.fields["christoffel"].shape == (1, 4, 4, 4)
    assert np.all(np.isfinite(samples.tensor_samples.fields["christoffel"]))
    assert np.max(np.abs(samples.tensor_samples.fields["christoffel"])) > 0.0


def test_schwarzschild_samples_can_be_saved(tmp_path):
    mass = 1.89813e27
    system = System([
        Body("primary", mass, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [1.0e9, 0, 0], [0, 1000, 0]),
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
    output = save_schwarzschild_trajectory_samples(samples, tmp_path)
    saved = np.load(output / "schwarzschild_samples.npz")
    assert saved["coordinates"].shape == (2, 4)
    assert saved["metric"].shape == (2, 4, 4)
    assert saved["proper_time"].shape == (2,)
    assert (output / "schwarzschild_metadata.json").exists()

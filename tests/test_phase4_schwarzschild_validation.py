import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT
from tensor_toolkit.metrics import SchwarzschildIsotropicMetric
from tensor_toolkit.relativity import (
    analytic_kretschmann,
    analytic_static_tidal_eigenvalues,
    areal_radius_from_isotropic,
    isotropic_radius_from_areal,
    photon_sphere_areal_radius,
    photon_sphere_isotropic_radius,
    static_frequency_ratio,
    validate_schwarzschild_circular_geodesic,
    validate_schwarzschild_kretschmann,
    validate_schwarzschild_light_bending,
    validate_schwarzschild_phase4,
    validate_schwarzschild_photon_sphere,
    validate_schwarzschild_radial_geodesic,
    validate_schwarzschild_redshift,
    validate_schwarzschild_tidal_eigenvalues,
)


def _metric_with_geometric_mass(m=1.0):
    return SchwarzschildIsotropicMetric(m * SPEED_OF_LIGHT**2 / GRAVITATIONAL_CONSTANT)


def test_schwarzschild_areal_isotropic_radius_round_trip():
    metric = _metric_with_geometric_mass()
    areal = np.array([3.0, 6.0, 10.0, 50.0])
    isotropic = isotropic_radius_from_areal(metric, areal)
    recovered = areal_radius_from_isotropic(metric, isotropic)
    assert np.allclose(recovered, areal, rtol=1e-14, atol=1e-14)


def test_schwarzschild_analytic_reference_values():
    metric = _metric_with_geometric_mass()
    rho = float(isotropic_radius_from_areal(metric, 10.0))
    assert np.isclose(analytic_kretschmann(metric, rho), 48.0 / 10.0**6)
    assert np.allclose(
        analytic_static_tidal_eigenvalues(metric, rho),
        [-2.0 / 10.0**3, 1.0 / 10.0**3, 1.0 / 10.0**3],
    )
    assert np.isclose(photon_sphere_areal_radius(metric), 3.0)
    photon_rho = photon_sphere_isotropic_radius(metric)
    assert np.isclose(areal_radius_from_isotropic(metric, photon_rho), 3.0)


def test_schwarzschild_static_redshift_reference():
    metric = _metric_with_geometric_mass()
    rho_e = float(isotropic_radius_from_areal(metric, 6.0))
    rho_r = float(isotropic_radius_from_areal(metric, 20.0))
    expected = np.sqrt(1.0 - 2.0 / 6.0) / np.sqrt(1.0 - 2.0 / 20.0)
    assert np.isclose(static_frequency_ratio(metric, rho_e, rho_r), expected)


def test_schwarzschild_phase4_kretschmann_and_redshift_validation():
    metric = _metric_with_geometric_mass()
    assert validate_schwarzschild_kretschmann(metric).passed
    assert validate_schwarzschild_redshift(metric).passed


def test_schwarzschild_phase4_timelike_geodesic_validation():
    metric = _metric_with_geometric_mass()
    radial = validate_schwarzschild_radial_geodesic(metric)
    circular = validate_schwarzschild_circular_geodesic(metric)
    assert radial.passed, radial
    assert circular.passed, circular


def test_schwarzschild_phase4_photon_sphere_and_tides_validation():
    metric = _metric_with_geometric_mass()
    photon = validate_schwarzschild_photon_sphere(metric)
    tides = validate_schwarzschild_tidal_eigenvalues(metric)
    assert photon.passed, photon
    assert tides.passed, tides


def test_schwarzschild_phase4_weak_field_light_bending_validation():
    metric = _metric_with_geometric_mass()
    bending = validate_schwarzschild_light_bending(metric)
    assert bending.passed, bending
    assert bending.numerical > 0.0


def test_schwarzschild_phase4_report_collects_checks():
    metric = _metric_with_geometric_mass()
    report = validate_schwarzschild_phase4(metric, include_light_bending=False)
    names = {check.name for check in report.checks}
    assert report.passed, report.failed_checks
    assert names == {
        "kretschmann",
        "gravitational_redshift",
        "radial_timelike_geodesic",
        "circular_timelike_geodesic",
        "photon_sphere",
        "tidal_eigenvalues",
    }

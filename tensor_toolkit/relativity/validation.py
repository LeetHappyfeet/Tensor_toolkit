"""Executable Schwarzschild Phase 4 validation suite."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tensor_toolkit.metrics import SchwarzschildIsotropicMetric
from .curvature import kretschmann_scalar, tidal_eigensystem, tidal_tensor
from .debug import debug_log
from .frames import static_frame
from .geodesics import integrate_geodesic
from .optics import trace_null_ray
from .sampling import SpacetimeSampler
from .schwarzschild import (
    analytic_kretschmann,
    analytic_static_tidal_eigenvalues,
    areal_radius_from_isotropic,
    circular_timelike_tangent,
    isotropic_radius_from_areal,
    photon_sphere_areal_radius,
    photon_sphere_tangent,
    static_frequency_ratio,
    static_radial_null_tangent,
    static_timelike_tangent,
    weak_field_light_deflection,
)
from .signals import frequency_transfer


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    numerical: object
    expected: object
    absolute_error: float
    relative_error: float
    details: dict[str, object]


@dataclass(frozen=True)
class SchwarzschildValidationReport:
    metric_name: str
    geometric_mass: float
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)


def _relative_error(numerical, expected) -> tuple[float, float]:
    numerical_array = np.asarray(numerical, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    absolute = float(np.max(np.abs(numerical_array - expected_array)))
    scale = float(np.max(np.abs(expected_array)))
    relative = absolute if scale == 0.0 else absolute / scale
    return absolute, relative


def _check(
    name: str,
    numerical,
    expected,
    *,
    rtol: float,
    atol: float = 0.0,
    details: dict[str, object] | None = None,
    extra_condition: bool = True,
    debug: bool = False,
) -> ValidationCheck:
    absolute, relative = _relative_error(numerical, expected)
    passed = bool(
        extra_condition
        and np.allclose(
            np.asarray(numerical, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=rtol,
            atol=atol,
        )
    )
    result = ValidationCheck(
        name=name,
        passed=passed,
        numerical=numerical,
        expected=expected,
        absolute_error=absolute,
        relative_error=relative,
        details={} if details is None else dict(details),
    )
    debug_log(
        debug,
        "validation",
        name,
        passed=passed,
        numerical=numerical,
        expected=expected,
        absolute_error=absolute,
        relative_error=relative,
    )
    return result


def validate_schwarzschild_kretschmann(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
) -> ValidationCheck:
    m = metric.geometric_mass
    R = 10.0 * m
    rho = float(isotropic_radius_from_areal(metric, R))
    spacing = 0.02 * m
    sampler = SpacetimeSampler(metric, (spacing,) * 4)
    event = np.array([0.0, rho, 0.0, 0.0])
    numerical = kretschmann_scalar(sampler.metric_at(event), sampler.riemann_at(event))
    expected = float(analytic_kretschmann(metric, rho))
    return _check(
        "kretschmann",
        numerical,
        expected,
        rtol=0.04,
        details={"areal_radius": R, "isotropic_radius": rho, "spacing": spacing},
        debug=debug,
    )


def validate_schwarzschild_redshift(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
) -> ValidationCheck:
    m = metric.geometric_mass
    rho_e = float(isotropic_radius_from_areal(metric, 6.0 * m))
    rho_r = float(isotropic_radius_from_areal(metric, 20.0 * m))
    sampler = SpacetimeSampler(metric, (0.02 * m,) * 4)
    event_e = np.array([0.0, rho_e, 0.0, 0.0])
    event_r = np.array([0.0, rho_r, 0.0, 0.0])
    transfer = frequency_transfer(
        sampler.metric_at(event_e),
        static_timelike_tangent(metric, rho_e),
        static_radial_null_tangent(metric, rho_e, outward=True),
        sampler.metric_at(event_r),
        static_timelike_tangent(metric, rho_r),
        static_radial_null_tangent(metric, rho_r, outward=True),
        debug=debug,
    )
    expected = static_frequency_ratio(metric, rho_e, rho_r)
    return _check(
        "gravitational_redshift",
        transfer.frequency_ratio,
        expected,
        rtol=1e-12,
        atol=1e-12,
        details={
            "emitter_areal_radius": 6.0 * m,
            "receiver_areal_radius": 20.0 * m,
            "redshift": transfer.redshift,
        },
        debug=debug,
    )


def validate_schwarzschild_radial_geodesic(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
) -> ValidationCheck:
    m = metric.geometric_mass
    R0 = 10.0 * m
    rho0 = float(isotropic_radius_from_areal(metric, R0))
    sampler = SpacetimeSampler(metric, (0.02 * m,) * 4)
    result = integrate_geodesic(
        sampler,
        [0.0, rho0, 0.0, 0.0],
        static_timelike_tangent(metric, rho0),
        affine_step=0.05 * m,
        steps=100,
        causal_type="timelike",
        name="schwarzschild-radial",
        debug=debug,
        debug_every=20,
    )
    rho = np.linalg.norm(result.worldline.coordinates[:, 1:], axis=1)
    R = areal_radius_from_isotropic(metric, rho)
    alpha = np.array(
        [
            np.sqrt(-sampler.metric_at(event)[0, 0])
            for event in result.worldline.coordinates
        ]
    )
    killing_energy = alpha**2 * result.worldline.tangent[:, 0]
    energy_drift = float(
        np.max(np.abs(killing_energy - killing_energy[0])) / abs(killing_energy[0])
    )
    radial_drop = float(R0 - R[-1])
    transverse = float(np.max(np.abs(result.worldline.coordinates[:, 2:])))
    passed_condition = radial_drop > 0.0 and transverse < 1e-8 * m and energy_drift < 2e-3
    return _check(
        "radial_timelike_geodesic",
        energy_drift,
        0.0,
        rtol=0.0,
        atol=2e-3,
        extra_condition=passed_condition,
        details={
            "initial_areal_radius": R0,
            "final_areal_radius": float(R[-1]),
            "radial_drop": radial_drop,
            "max_transverse_coordinate": transverse,
            "max_normalization_error": result.max_normalization_error,
        },
        debug=debug,
    )


def validate_schwarzschild_circular_geodesic(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
) -> ValidationCheck:
    m = metric.geometric_mass
    R0 = 10.0 * m
    rho0, tangent = circular_timelike_tangent(metric, R0)
    sampler = SpacetimeSampler(metric, (0.02 * m,) * 4)
    result = integrate_geodesic(
        sampler,
        [0.0, rho0, 0.0, 0.0],
        tangent,
        affine_step=0.1 * m,
        steps=100,
        causal_type="timelike",
        name="schwarzschild-circular",
        debug=debug,
        debug_every=20,
    )
    rho = np.linalg.norm(result.worldline.coordinates[:, 1:], axis=1)
    R = areal_radius_from_isotropic(metric, rho)
    max_relative_radius_drift = float(np.max(np.abs(R - R0)) / R0)
    passed_condition = result.max_normalization_error < 2e-3
    return _check(
        "circular_timelike_geodesic",
        max_relative_radius_drift,
        0.0,
        rtol=0.0,
        atol=5e-3,
        extra_condition=passed_condition,
        details={
            "areal_radius": R0,
            "isotropic_radius": rho0,
            "max_normalization_error": result.max_normalization_error,
        },
        debug=debug,
    )


def validate_schwarzschild_light_bending(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
) -> ValidationCheck:
    m = metric.geometric_mass
    b = 50.0 * m
    x_start = -300.0 * m
    x_stop = 300.0 * m
    spacing = 0.2 * m
    sampler = SpacetimeSampler(metric, (spacing,) * 4)
    event = np.array([0.0, x_start, b, 0.0])
    frame = static_frame(sampler.metric_at(event), event, name="weak-field-emitter")

    def stop(coordinates: np.ndarray, tangent: np.ndarray) -> bool:
        del tangent
        return bool(coordinates[1] >= x_stop)

    ray = trace_null_ray(
        sampler,
        frame,
        [1.0, 0.0, 0.0],
        affine_step=2.0 * m,
        steps=350,
        stop_condition=stop,
        name="schwarzschild-bending",
        debug=debug,
        debug_every=50,
    )
    final_event = ray.worldline.coordinates[-1]
    final_frame = static_frame(
        sampler.metric_at(final_event),
        final_event,
        name="weak-field-receiver",
    )
    final_local = final_frame.measure_vector(ray.worldline.tangent[-1])
    numerical = float(np.arctan2(abs(final_local[2]), final_local[1]))
    expected = weak_field_light_deflection(metric, b)
    reached_far_side = bool(final_event[1] >= x_stop)
    return _check(
        "weak_field_light_bending",
        numerical,
        expected,
        rtol=0.35,
        extra_condition=reached_far_side,
        details={
            "impact_parameter": b,
            "start_x": x_start,
            "final_x": float(final_event[1]),
            "max_normalization_error": ray.max_normalization_error,
        },
        debug=debug,
    )


def validate_schwarzschild_photon_sphere(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
) -> ValidationCheck:
    m = metric.geometric_mass
    rho0, tangent = photon_sphere_tangent(metric)
    expected_R = photon_sphere_areal_radius(metric)
    sampler = SpacetimeSampler(metric, (0.005 * m,) * 4)
    result = integrate_geodesic(
        sampler,
        [0.0, rho0, 0.0, 0.0],
        tangent,
        affine_step=0.01 * m,
        steps=100,
        causal_type="null",
        name="schwarzschild-photon-sphere",
        debug=debug,
        debug_every=20,
    )
    rho = np.linalg.norm(result.worldline.coordinates[:, 1:], axis=1)
    R = areal_radius_from_isotropic(metric, rho)
    max_relative_radius_drift = float(np.max(np.abs(R - expected_R)) / expected_R)
    passed_condition = result.max_normalization_error < 2e-3
    return _check(
        "photon_sphere",
        max_relative_radius_drift,
        0.0,
        rtol=0.0,
        atol=0.03,
        extra_condition=passed_condition,
        details={
            "expected_areal_radius": expected_R,
            "isotropic_radius": rho0,
            "max_normalization_error": result.max_normalization_error,
        },
        debug=debug,
    )


def validate_schwarzschild_tidal_eigenvalues(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
) -> ValidationCheck:
    m = metric.geometric_mass
    R = 10.0 * m
    rho = float(isotropic_radius_from_areal(metric, R))
    spacing = 0.02 * m
    sampler = SpacetimeSampler(metric, (spacing,) * 4)
    event = np.array([0.0, rho, 0.0, 0.0])
    point_metric = sampler.metric_at(event)
    frame = static_frame(point_metric, event)
    tidal = tidal_tensor(point_metric, sampler.riemann_at(event), frame)
    numerical, _ = tidal_eigensystem(tidal)
    expected = analytic_static_tidal_eigenvalues(metric, rho)
    return _check(
        "tidal_eigenvalues",
        numerical,
        expected,
        rtol=0.05,
        details={
            "areal_radius": R,
            "isotropic_radius": rho,
            "trace": float(np.trace(tidal)),
        },
        debug=debug,
    )


def validate_schwarzschild_phase4(
    metric: SchwarzschildIsotropicMetric,
    *,
    debug: bool = False,
    include_light_bending: bool = True,
) -> SchwarzschildValidationReport:
    """Run the Phase 4 Schwarzschild validation gate."""

    debug_log(
        debug,
        "validation",
        "schwarzschild_phase4:start",
        mass_kg=metric.mass_kg,
        geometric_mass=metric.geometric_mass,
    )
    checks = [
        validate_schwarzschild_kretschmann(metric, debug=debug),
        validate_schwarzschild_redshift(metric, debug=debug),
        validate_schwarzschild_radial_geodesic(metric, debug=debug),
        validate_schwarzschild_circular_geodesic(metric, debug=debug),
        validate_schwarzschild_photon_sphere(metric, debug=debug),
        validate_schwarzschild_tidal_eigenvalues(metric, debug=debug),
    ]
    if include_light_bending:
        checks.append(validate_schwarzschild_light_bending(metric, debug=debug))
    report = SchwarzschildValidationReport(
        metric_name=metric.name,
        geometric_mass=metric.geometric_mass,
        checks=tuple(checks),
    )
    debug_log(
        debug,
        "validation",
        "schwarzschild_phase4:done",
        passed=report.passed,
        failed=",".join(report.failed_checks) if report.failed_checks else "none",
    )
    return report

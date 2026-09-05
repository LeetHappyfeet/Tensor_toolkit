"""Analytic Schwarzschild reference quantities for Phase 4 validation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tensor_toolkit.metrics import SchwarzschildIsotropicMetric


def areal_radius_from_isotropic(metric: SchwarzschildIsotropicMetric, rho) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    m = metric.geometric_mass
    if np.any(rho <= 0.5 * m):
        raise ValueError("isotropic radius must lie outside the Schwarzschild horizon")
    return rho * (1.0 + m / (2.0 * rho)) ** 2


def isotropic_radius_from_areal(metric: SchwarzschildIsotropicMetric, radius) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64)
    m = metric.geometric_mass
    if np.any(radius <= 2.0 * m):
        raise ValueError("areal radius must lie outside the Schwarzschild horizon")
    return 0.5 * (radius - m + np.sqrt(radius * (radius - 2.0 * m)))


def schwarzschild_lapse(metric: SchwarzschildIsotropicMetric, rho) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    m = metric.geometric_mass
    u = m / (2.0 * rho)
    if np.any(u >= 1.0):
        raise ValueError("isotropic radius must lie outside the Schwarzschild horizon")
    return (1.0 - u) / (1.0 + u)


def analytic_kretschmann(metric: SchwarzschildIsotropicMetric, rho) -> np.ndarray:
    """K = 48 m^2 / R^6 using areal radius R."""

    radius = areal_radius_from_isotropic(metric, rho)
    m = metric.geometric_mass
    return 48.0 * m**2 / radius**6


def static_frequency_ratio(
    metric: SchwarzschildIsotropicMetric,
    emitter_rho: float,
    receiver_rho: float,
) -> float:
    """Received/emitted frequency ratio for static Schwarzschild observers."""

    alpha_e = float(schwarzschild_lapse(metric, emitter_rho))
    alpha_r = float(schwarzschild_lapse(metric, receiver_rho))
    return alpha_e / alpha_r


def static_redshift(
    metric: SchwarzschildIsotropicMetric,
    emitter_rho: float,
    receiver_rho: float,
) -> float:
    ratio = static_frequency_ratio(metric, emitter_rho, receiver_rho)
    return 1.0 / ratio - 1.0


def weak_field_light_deflection(metric: SchwarzschildIsotropicMetric, impact_parameter: float) -> float:
    """Leading-order asymptotic deflection angle 4m/b."""

    b = float(impact_parameter)
    if b <= 0.0:
        raise ValueError("impact_parameter must be positive")
    return 4.0 * metric.geometric_mass / b


def photon_sphere_areal_radius(metric: SchwarzschildIsotropicMetric) -> float:
    return 3.0 * metric.geometric_mass


def photon_sphere_isotropic_radius(metric: SchwarzschildIsotropicMetric) -> float:
    return float(isotropic_radius_from_areal(metric, photon_sphere_areal_radius(metric)))


def circular_timelike_tangent(
    metric: SchwarzschildIsotropicMetric,
    areal_radius: float,
) -> tuple[float, np.ndarray]:
    """Return isotropic radius and proper-length-normalized circular tangent.

    The orbit begins on the +x axis in the equatorial plane and moves +y.
    Coordinates are (ct, x, y, z), so the affine parameter has length units.
    """

    R = float(areal_radius)
    m = metric.geometric_mass
    if R <= 3.0 * m:
        raise ValueError("circular timelike geodesics require areal radius > 3m")
    rho = float(isotropic_radius_from_areal(metric, R))
    u_t = 1.0 / np.sqrt(1.0 - 3.0 * m / R)
    u_phi = np.sqrt(m / R**3) / np.sqrt(1.0 - 3.0 * m / R)
    tangent = np.array([u_t, 0.0, rho * u_phi, 0.0], dtype=np.float64)
    return rho, tangent


def photon_sphere_tangent(
    metric: SchwarzschildIsotropicMetric,
    *,
    energy: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Return a circular null tangent at the Schwarzschild photon sphere."""

    m = metric.geometric_mass
    rho = photon_sphere_isotropic_radius(metric)
    E = float(energy)
    if E <= 0.0:
        raise ValueError("energy must be positive")
    k_t = 3.0 * E
    k_phi = E / (np.sqrt(3.0) * m)
    tangent = np.array([k_t, 0.0, rho * k_phi, 0.0], dtype=np.float64)
    return rho, tangent


def analytic_static_tidal_eigenvalues(
    metric: SchwarzschildIsotropicMetric,
    rho: float,
) -> np.ndarray:
    """Sorted E_ij eigenvalues for a static orthonormal observer."""

    R = float(areal_radius_from_isotropic(metric, rho))
    scale = metric.geometric_mass / R**3
    return np.array([-2.0 * scale, scale, scale], dtype=np.float64)


@dataclass(frozen=True)
class SchwarzschildReference:
    geometric_mass: float
    isotropic_radius: float
    areal_radius: float
    lapse: float
    kretschmann: float
    tidal_eigenvalues: np.ndarray


def reference_at_isotropic_radius(
    metric: SchwarzschildIsotropicMetric,
    rho: float,
) -> SchwarzschildReference:
    return SchwarzschildReference(
        geometric_mass=metric.geometric_mass,
        isotropic_radius=float(rho),
        areal_radius=float(areal_radius_from_isotropic(metric, rho)),
        lapse=float(schwarzschild_lapse(metric, rho)),
        kretschmann=float(analytic_kretschmann(metric, rho)),
        tidal_eigenvalues=analytic_static_tidal_eigenvalues(metric, rho),
    )

"""Conservation, orbital-invariant, and encounter diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT
from .trajectory import Trajectory


@dataclass(frozen=True)
class ConservationDiagnostics:
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    linear_momentum: np.ndarray
    angular_momentum: np.ndarray
    energy_relative_drift: float
    momentum_absolute_drift: float
    angular_momentum_relative_drift: float


@dataclass(frozen=True)
class TestParticleDiagnostics:
    primary_name: str
    probe_name: str
    orbit_class: str
    specific_energy: np.ndarray
    specific_angular_momentum: np.ndarray
    specific_energy_relative_drift: float
    specific_angular_momentum_relative_drift: float
    eccentricity: float
    semi_major_axis: float | None
    periapsis_distance: float
    apoapsis_distance: float | None
    orbital_period: float | None


@dataclass(frozen=True)
class HyperbolicReference:
    primary_name: str
    probe_name: str
    eccentricity: float
    v_infinity: float
    periapsis_distance: float
    periapsis_speed: float
    asymptotic_deflection_angle: float
    numerical_periapsis_distance_error: float
    numerical_periapsis_speed_error: float
    finite_window_deflection_error: float


@dataclass(frozen=True)
class EncounterDiagnostics:
    primary_name: str
    probe_name: str
    closest_approach_time: float
    closest_approach_distance: float
    periapsis_relative_speed: float
    initial_relative_speed: float
    final_relative_speed: float
    deflection_angle: float


def _relative_drift(values: np.ndarray) -> float:
    baseline = float(np.linalg.norm(values[0]))
    delta = (
        float(np.max(np.linalg.norm(values - values[0], axis=-1)))
        if values.ndim > 1
        else float(np.max(np.abs(values - values[0])))
    )
    if baseline == 0.0:
        return delta
    return delta / baseline


def conservation_diagnostics(
    trajectory: Trajectory,
    masses,
    *,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
    softening: float = 0.0,
) -> ConservationDiagnostics:
    """Compute total-system Newtonian conservation histories.

    Passive test particles with mass=0 do not contribute to these quantities.
    Use test_particle_diagnostics for their orbital invariants.
    """
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (len(trajectory.body_names),):
        raise ValueError("masses must match trajectory body count")
    if np.any(masses < 0.0) or np.any(~np.isfinite(masses)):
        raise ValueError("masses must be finite and non-negative")

    velocities = trajectory.velocities
    positions = trajectory.positions
    kinetic = 0.5 * np.sum(
        masses[np.newaxis, :, np.newaxis] * velocities**2,
        axis=(1, 2),
    )

    potential = np.zeros(trajectory.times.size, dtype=np.float64)
    eps2 = float(softening) ** 2
    for i in range(masses.size):
        for j in range(i + 1, masses.size):
            if masses[i] == 0.0 or masses[j] == 0.0:
                continue
            separation = positions[:, j] - positions[:, i]
            distance = np.sqrt(
                np.einsum("ti,ti->t", separation, separation) + eps2
            )
            potential -= (
                float(gravitational_constant) * masses[i] * masses[j] / distance
            )

    total = kinetic + potential
    momentum = np.sum(
        masses[np.newaxis, :, np.newaxis] * velocities,
        axis=1,
    )
    angular = np.sum(
        np.cross(
            positions,
            masses[np.newaxis, :, np.newaxis] * velocities,
        ),
        axis=1,
    )

    energy_baseline = abs(float(total[0]))
    energy_delta = float(np.max(np.abs(total - total[0])))
    energy_relative = (
        energy_delta
        if energy_baseline == 0.0
        else energy_delta / energy_baseline
    )

    return ConservationDiagnostics(
        kinetic_energy=kinetic,
        potential_energy=potential,
        total_energy=total,
        linear_momentum=momentum,
        angular_momentum=angular,
        energy_relative_drift=energy_relative,
        momentum_absolute_drift=float(
            np.max(np.linalg.norm(momentum - momentum[0], axis=1))
        ),
        angular_momentum_relative_drift=_relative_drift(angular),
    )


def test_particle_diagnostics(
    trajectory: Trajectory,
    *,
    primary: str | int = 0,
    probe: str | int = 1,
    primary_mass: float,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> TestParticleDiagnostics:
    """Compute specific orbital invariants for a passive probe."""
    if primary_mass <= 0.0:
        raise ValueError("primary_mass must be positive")

    primary_index = trajectory.body_index(primary)
    probe_index = trajectory.body_index(probe)
    if primary_index == probe_index:
        raise ValueError("primary and probe must identify different bodies")

    relative_position = (
        trajectory.positions[:, probe_index]
        - trajectory.positions[:, primary_index]
    )
    relative_velocity = (
        trajectory.velocities[:, probe_index]
        - trajectory.velocities[:, primary_index]
    )
    radius = np.linalg.norm(relative_position, axis=1)
    if np.any(radius == 0.0):
        raise ValueError("test particle intersects the primary")

    mu = float(gravitational_constant) * float(primary_mass)
    speed2 = np.einsum("ti,ti->t", relative_velocity, relative_velocity)
    specific_energy = 0.5 * speed2 - mu / radius
    specific_angular_momentum = np.cross(
        relative_position,
        relative_velocity,
    )

    energy0 = float(specific_energy[0])
    h0 = float(np.linalg.norm(specific_angular_momentum[0]))
    eccentricity = float(np.sqrt(max(0.0, 1.0 + 2.0 * energy0 * h0**2 / mu**2)))
    tolerance = max(1e-12, abs(energy0) * 1e-12)
    if energy0 < -tolerance:
        orbit_class = "elliptic"
        semi_major_axis = float(-mu / (2.0 * energy0))
        periapsis_distance = float(semi_major_axis * (1.0 - eccentricity))
        apoapsis_distance = float(semi_major_axis * (1.0 + eccentricity))
        orbital_period = float(2.0 * np.pi * np.sqrt(semi_major_axis**3 / mu))
    elif energy0 > tolerance:
        orbit_class = "hyperbolic"
        semi_major_axis = float(-mu / (2.0 * energy0))
        periapsis_distance = float(h0**2 / (mu * (1.0 + eccentricity)))
        apoapsis_distance = None
        orbital_period = None
    else:
        orbit_class = "parabolic"
        semi_major_axis = None
        eccentricity = 1.0
        periapsis_distance = float(h0**2 / (2.0 * mu))
        apoapsis_distance = None
        orbital_period = None

    return TestParticleDiagnostics(
        primary_name=trajectory.body_names[primary_index],
        probe_name=trajectory.body_names[probe_index],
        orbit_class=orbit_class,
        specific_energy=specific_energy,
        specific_angular_momentum=specific_angular_momentum,
        specific_energy_relative_drift=_relative_drift(specific_energy),
        specific_angular_momentum_relative_drift=_relative_drift(
            specific_angular_momentum
        ),
        eccentricity=eccentricity,
        semi_major_axis=semi_major_axis,
        periapsis_distance=periapsis_distance,
        apoapsis_distance=apoapsis_distance,
        orbital_period=orbital_period,
    )



# Public scientific API name begins with 'test_' for domain terminology.\n# Prevent pytest from collecting it when imported into a test module.\ntest_particle_diagnostics.__test__ = False\n
def encounter_diagnostics(
    trajectory: Trajectory,
    *,
    primary: str | int = 0,
    probe: str | int = 1,
) -> EncounterDiagnostics:
    """Summarize a two-body encounter from the simulated relative trajectory."""
    primary_index = trajectory.body_index(primary)
    probe_index = trajectory.body_index(probe)
    if primary_index == probe_index:
        raise ValueError("primary and probe must identify different bodies")

    relative_position = (
        trajectory.positions[:, probe_index]
        - trajectory.positions[:, primary_index]
    )
    relative_velocity = (
        trajectory.velocities[:, probe_index]
        - trajectory.velocities[:, primary_index]
    )
    distance = np.linalg.norm(relative_position, axis=1)
    closest_index = int(np.argmin(distance))
    speeds = np.linalg.norm(relative_velocity, axis=1)

    initial_speed = float(speeds[0])
    final_speed = float(speeds[-1])
    if initial_speed == 0.0 or final_speed == 0.0:
        raise ValueError(
            "deflection angle requires non-zero initial and final relative speeds"
        )
    cosine = float(
        np.dot(relative_velocity[0], relative_velocity[-1])
        / (initial_speed * final_speed)
    )
    deflection = float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return EncounterDiagnostics(
        primary_name=trajectory.body_names[primary_index],
        probe_name=trajectory.body_names[probe_index],
        closest_approach_time=float(trajectory.times[closest_index]),
        closest_approach_distance=float(distance[closest_index]),
        periapsis_relative_speed=float(speeds[closest_index]),
        initial_relative_speed=initial_speed,
        final_relative_speed=final_speed,
        deflection_angle=deflection,
    )


def hyperbolic_reference(
    trajectory: Trajectory,
    *,
    primary: str | int = 0,
    probe: str | int = 1,
    primary_mass: float,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> HyperbolicReference:
    """Compare a numerical encounter against analytic Newtonian hyperbola invariants."""
    if primary_mass <= 0.0:
        raise ValueError("primary_mass must be positive")

    primary_index = trajectory.body_index(primary)
    probe_index = trajectory.body_index(probe)
    relative_position0 = (
        trajectory.positions[0, probe_index]
        - trajectory.positions[0, primary_index]
    )
    relative_velocity0 = (
        trajectory.velocities[0, probe_index]
        - trajectory.velocities[0, primary_index]
    )

    mu = float(gravitational_constant) * float(primary_mass)
    r0 = float(np.linalg.norm(relative_position0))
    v0_sq = float(np.dot(relative_velocity0, relative_velocity0))
    specific_energy = 0.5 * v0_sq - mu / r0
    if specific_energy <= 0.0:
        raise ValueError("trajectory is not hyperbolic from the initial state")

    h_vec = np.cross(relative_position0, relative_velocity0)
    h = float(np.linalg.norm(h_vec))
    eccentricity = float(
        np.sqrt(1.0 + 2.0 * specific_energy * h**2 / mu**2)
    )
    v_infinity = float(np.sqrt(2.0 * specific_energy))
    periapsis_distance = float(h**2 / (mu * (1.0 + eccentricity)))
    periapsis_speed = float(h / periapsis_distance)
    asymptotic_deflection = float(
        2.0 * np.arcsin(1.0 / eccentricity)
    )

    numerical = encounter_diagnostics(
        trajectory,
        primary=primary_index,
        probe=probe_index,
    )

    return HyperbolicReference(
        primary_name=trajectory.body_names[primary_index],
        probe_name=trajectory.body_names[probe_index],
        eccentricity=eccentricity,
        v_infinity=v_infinity,
        periapsis_distance=periapsis_distance,
        periapsis_speed=periapsis_speed,
        asymptotic_deflection_angle=asymptotic_deflection,
        numerical_periapsis_distance_error=(
            numerical.closest_approach_distance - periapsis_distance
        ),
        numerical_periapsis_speed_error=(
            numerical.periapsis_relative_speed - periapsis_speed
        ),
        finite_window_deflection_error=(
            numerical.deflection_angle - asymptotic_deflection
        ),
    )

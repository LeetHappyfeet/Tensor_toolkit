"""Conservation diagnostics for Newtonian trajectories."""

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

def _relative_drift(values: np.ndarray) -> float:
    baseline = float(np.linalg.norm(values[0]))
    delta = float(np.max(np.linalg.norm(values - values[0], axis=-1))) if values.ndim > 1 else float(np.max(np.abs(values - values[0])))
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
    """Compute energy and momentum conservation histories for an N-body trajectory."""
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (len(trajectory.body_names),):
        raise ValueError("masses must match trajectory body count")
    if np.any(masses < 0.0) or np.any(~np.isfinite(masses)):
        raise ValueError("masses must be finite and non-negative")

    velocities = trajectory.velocities
    positions = trajectory.positions
    kinetic = 0.5 * np.sum(masses[np.newaxis, :, np.newaxis] * velocities**2, axis=(1, 2))

    potential = np.zeros(trajectory.times.size, dtype=np.float64)
    eps2 = float(softening) ** 2
    for i in range(masses.size):
        for j in range(i + 1, masses.size):
            if masses[i] == 0.0 or masses[j] == 0.0:
                continue
            separation = positions[:, j] - positions[:, i]
            distance = np.sqrt(np.einsum("ti,ti->t", separation, separation) + eps2)
            potential -= float(gravitational_constant) * masses[i] * masses[j] / distance

    total = kinetic + potential
    momentum = np.sum(masses[np.newaxis, :, np.newaxis] * velocities, axis=1)
    angular = np.sum(np.cross(positions, masses[np.newaxis, :, np.newaxis] * velocities), axis=1)

    energy_baseline = abs(float(total[0]))
    energy_delta = float(np.max(np.abs(total - total[0])))
    energy_relative = energy_delta if energy_baseline == 0.0 else energy_delta / energy_baseline

    return ConservationDiagnostics(
        kinetic_energy=kinetic,
        potential_energy=potential,
        total_energy=total,
        linear_momentum=momentum,
        angular_momentum=angular,
        energy_relative_drift=energy_relative,
        momentum_absolute_drift=float(np.max(np.linalg.norm(momentum - momentum[0], axis=1))),
        angular_momentum_relative_drift=_relative_drift(angular),
    )


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

def encounter_diagnostics(
    trajectory: Trajectory,
    *,
    primary: str | int = 0,
    probe: str | int = 1,
) -> EncounterDiagnostics:
    """Summarize a two-body encounter from the simulated relative trajectory.

    Deflection angle is the angle in radians between the initial and final
    relative-velocity directions over the supplied trajectory interval.
    """
    primary_index = trajectory.body_index(primary)
    probe_index = trajectory.body_index(probe)
    if primary_index == probe_index:
        raise ValueError("primary and probe must identify different bodies")

    relative_position = (
        trajectory.positions[:, probe_index] - trajectory.positions[:, primary_index]
    )
    relative_velocity = (
        trajectory.velocities[:, probe_index] - trajectory.velocities[:, primary_index]
    )
    distance = np.linalg.norm(relative_position, axis=1)
    closest_index = int(np.argmin(distance))
    speeds = np.linalg.norm(relative_velocity, axis=1)

    initial_speed = float(speeds[0])
    final_speed = float(speeds[-1])
    if initial_speed == 0.0 or final_speed == 0.0:
        raise ValueError("deflection angle requires non-zero initial and final relative speeds")
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

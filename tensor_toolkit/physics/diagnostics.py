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

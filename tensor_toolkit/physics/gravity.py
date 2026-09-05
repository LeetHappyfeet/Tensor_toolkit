"""Newtonian gravitational force models."""

from __future__ import annotations

import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT


def newtonian_gravity_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
    softening: float = 0.0,
) -> np.ndarray:
    """Return N-body gravitational accelerations in Cartesian coordinates.

    Parameters use SI units by default: positions in metres, masses in kilograms,
    and accelerations are returned in m/s^2. Softening is an optional numerical
    length scale in metres and defaults to zero.

    Coincident zero-mass source particles contribute exactly zero acceleration.
    Coincident positive-mass sources remain a Newtonian point-mass singularity
    unless non-zero softening is supplied.
    """
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have shape (N,)")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(masses)):
        raise ValueError("positions and masses must be finite")
    if np.any(masses < 0.0):
        raise ValueError("masses must be non-negative")
    if gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be positive")
    if softening < 0.0:
        raise ValueError("softening must be non-negative")

    displacement = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    raw_distance2 = np.einsum("ijk,ijk->ij", displacement, displacement)
    np.fill_diagonal(raw_distance2, np.inf)

    if softening:
        distance2 = raw_distance2 + float(softening) ** 2
    else:
        coincident = raw_distance2 == 0.0
        positive_source = masses[np.newaxis, :] > 0.0
        if np.any(coincident & positive_source):
            raise ValueError(
                "coincident positive-mass point sources are singular; "
                "use finite radii/collision handling or non-zero softening"
            )
        distance2 = raw_distance2.copy()
        # A coincident massless source has no gravitational effect. Treat its
        # inverse-distance factor as zero rather than forming 0 * infinity.
        distance2[coincident] = np.inf

    inv_r3 = distance2 ** -1.5
    source_weight = masses[np.newaxis, :] * inv_r3
    weighted = displacement * source_weight[:, :, np.newaxis]
    return float(gravitational_constant) * np.sum(weighted, axis=1)

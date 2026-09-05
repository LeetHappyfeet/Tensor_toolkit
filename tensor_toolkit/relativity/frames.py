"""Local orthonormal observer frames and tensor measurements."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def metric_inner(metric: np.ndarray, a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.einsum("m,mn,n->", a, metric, b))


def normalize_timelike(metric: np.ndarray, vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm2 = metric_inner(metric, vector, vector)
    if norm2 >= 0.0:
        raise ValueError("observer four-velocity must be timelike")
    return vector / np.sqrt(-norm2)


def _orthogonalize(metric: np.ndarray, candidate: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    value = candidate.astype(np.float64, copy=True)
    for existing in basis:
        denom = metric_inner(metric, existing, existing)
        value -= metric_inner(metric, value, existing) / denom * existing
    return value


@dataclass(frozen=True)
class ObserverFrame:
    """Orthonormal tetrad at one spacetime event.

    tetrad[a, mu] stores local basis vector a in coordinate components.
    """

    coordinates: np.ndarray
    four_velocity: np.ndarray
    tetrad: np.ndarray
    metric: np.ndarray
    name: str = "observer"

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        four_velocity = np.asarray(self.four_velocity, dtype=np.float64)
        tetrad = np.asarray(self.tetrad, dtype=np.float64)
        metric = np.asarray(self.metric, dtype=np.float64)
        if coordinates.shape != (4,):
            raise ValueError("coordinates must have shape (4,)")
        if four_velocity.shape != (4,):
            raise ValueError("four_velocity must have shape (4,)")
        if tetrad.shape != (4, 4):
            raise ValueError("tetrad must have shape (4, 4)")
        if metric.shape != (4, 4):
            raise ValueError("metric must have shape (4, 4)")
        gram = np.einsum("am,mn,bn->ab", tetrad, metric, tetrad)
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        if not np.allclose(gram, eta, rtol=1e-9, atol=1e-10):
            raise ValueError("tetrad is not orthonormal under the supplied metric")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "four_velocity", four_velocity)
        object.__setattr__(self, "tetrad", tetrad)
        object.__setattr__(self, "metric", metric)

    @property
    def time_basis(self) -> np.ndarray:
        return self.tetrad[0]

    @property
    def spatial_basis(self) -> np.ndarray:
        return self.tetrad[1:]

    def measure_vector(self, vector) -> np.ndarray:
        """Return local tetrad components of a contravariant vector."""
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (4,):
            raise ValueError("vector must have shape (4,)")
        covector = self.metric @ vector
        components = self.tetrad @ covector
        components[0] *= -1.0
        return components

    def measure_covariant_rank2(self, tensor) -> np.ndarray:
        tensor = np.asarray(tensor, dtype=np.float64)
        if tensor.shape != (4, 4):
            raise ValueError("rank-2 tensor must have shape (4, 4)")
        return np.einsum("am,bn,mn->ab", self.tetrad, self.tetrad, tensor)

    def spatial_projection(self, tensor) -> np.ndarray:
        return self.measure_covariant_rank2(tensor)[1:, 1:]


def comoving_frame(metric: np.ndarray, coordinates, four_velocity, *, name="observer") -> ObserverFrame:
    """Construct an orthonormal tetrad using metric Gram-Schmidt."""

    metric = np.asarray(metric, dtype=np.float64)
    if metric.shape != (4, 4):
        raise ValueError("metric must have shape (4, 4)")
    e0 = normalize_timelike(metric, four_velocity)
    basis: list[np.ndarray] = [e0]
    for seed in np.eye(4, dtype=np.float64):
        candidate = _orthogonalize(metric, seed, basis)
        norm2 = metric_inner(metric, candidate, candidate)
        if norm2 > 1e-12:
            basis.append(candidate / np.sqrt(norm2))
        if len(basis) == 4:
            break
    if len(basis) != 4:
        raise ValueError("could not construct three spacelike tetrad axes")
    return ObserverFrame(
        coordinates=np.asarray(coordinates, dtype=np.float64),
        four_velocity=np.asarray(four_velocity, dtype=np.float64),
        tetrad=np.stack(basis),
        metric=metric,
        name=name,
    )

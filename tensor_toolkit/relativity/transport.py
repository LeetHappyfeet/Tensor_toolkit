"""Parallel and Fermi-Walker transport along sampled worldlines."""

from __future__ import annotations

import numpy as np

from tensor_toolkit.physics.worldline import Worldline
from .sampling import SpacetimeSampler


def _parallel_rhs(
    sampler: SpacetimeSampler,
    coordinates: np.ndarray,
    tangent: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    gamma = sampler.connection_at(coordinates)
    return -np.einsum("mab,a,b->m", gamma, tangent, vector, optimize=True)


def parallel_transport(
    sampler: SpacetimeSampler,
    worldline: Worldline,
    initial_vector,
) -> np.ndarray:
    """Transport a vector along an already sampled worldline.

    A midpoint predictor-corrector step is used between worldline samples.
    """

    vector = np.asarray(initial_vector, dtype=np.float64)
    if vector.shape != (4,):
        raise ValueError("initial_vector must have shape (4,)")
    output = np.empty((len(worldline.parameter), 4), dtype=np.float64)
    output[0] = vector

    for i in range(len(worldline.parameter) - 1):
        h = float(worldline.parameter[i + 1] - worldline.parameter[i])
        x0 = worldline.coordinates[i]
        x1 = worldline.coordinates[i + 1]
        u0 = worldline.tangent[i]
        u1 = worldline.tangent[i + 1]
        k1 = _parallel_rhs(sampler, x0, u0, vector)
        midpoint_vector = vector + 0.5 * h * k1
        midpoint_x = 0.5 * (x0 + x1)
        midpoint_u = 0.5 * (u0 + u1)
        k2 = _parallel_rhs(sampler, midpoint_x, midpoint_u, midpoint_vector)
        vector = vector + h * k2
        output[i + 1] = vector
    return output


def fermi_walker_transport(
    sampler: SpacetimeSampler,
    worldline: Worldline,
    four_acceleration,
    initial_vector,
) -> np.ndarray:
    """Fermi-Walker transport using a unit-timelike tangent and supplied 4-acceleration.

    The caller is responsible for using a proper-time-parameterized worldline.
    """

    acceleration = np.asarray(four_acceleration, dtype=np.float64)
    if acceleration.shape != worldline.coordinates.shape:
        raise ValueError("four_acceleration must match worldline coordinate shape")
    vector = np.asarray(initial_vector, dtype=np.float64)
    if vector.shape != (4,):
        raise ValueError("initial_vector must have shape (4,)")
    output = np.empty_like(acceleration)
    output[0] = vector

    for i in range(len(worldline.parameter) - 1):
        h = float(worldline.parameter[i + 1] - worldline.parameter[i])
        metric = sampler.metric_at(worldline.coordinates[i])
        gamma = sampler.connection_at(worldline.coordinates[i])
        u = worldline.tangent[i]
        a = acceleration[i]
        u_cov = metric @ u
        a_cov = metric @ a
        connection_term = -np.einsum("mab,a,b->m", gamma, u, vector, optimize=True)
        fw = (np.outer(u, a_cov) - np.outer(a, u_cov)) @ vector
        vector = vector + h * (connection_term + fw)
        output[i + 1] = vector
    return output

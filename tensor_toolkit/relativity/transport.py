"""Parallel and Fermi-Walker transport along sampled worldlines."""

from __future__ import annotations

import numpy as np

from tensor_toolkit.physics.worldline import Worldline
from .debug import debug_log
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
    *,
    debug: bool = False,
    debug_every: int = 1,
) -> np.ndarray:
    """Transport a vector along an already sampled worldline."""

    vector = np.asarray(initial_vector, dtype=np.float64)
    debug_every = int(debug_every)
    if vector.shape != (4,):
        raise ValueError("initial_vector must have shape (4,)")
    if debug_every < 1:
        raise ValueError("debug_every must be at least 1")
    debug_enabled = bool(debug or sampler.debug)
    output = np.empty((len(worldline.parameter), 4), dtype=np.float64)
    output[0] = vector
    debug_log(
        debug_enabled,
        "transport",
        "parallel:start",
        samples=len(worldline.parameter),
        vector=np.array2string(vector, precision=6),
    )

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
        if debug_enabled and ((i + 1) % debug_every == 0 or i == len(worldline.parameter) - 2):
            debug_log(
                True,
                "transport",
                "parallel:step",
                index=i + 1,
                vector=np.array2string(vector, precision=6),
            )
    return output


def fermi_walker_transport(
    sampler: SpacetimeSampler,
    worldline: Worldline,
    four_acceleration,
    initial_vector,
    *,
    debug: bool = False,
    debug_every: int = 1,
) -> np.ndarray:
    """Fermi-Walker transport using a unit-timelike tangent and supplied 4-acceleration."""

    acceleration = np.asarray(four_acceleration, dtype=np.float64)
    debug_every = int(debug_every)
    if acceleration.shape != worldline.coordinates.shape:
        raise ValueError("four_acceleration must match worldline coordinate shape")
    vector = np.asarray(initial_vector, dtype=np.float64)
    if vector.shape != (4,):
        raise ValueError("initial_vector must have shape (4,)")
    if debug_every < 1:
        raise ValueError("debug_every must be at least 1")
    debug_enabled = bool(debug or sampler.debug)
    output = np.empty_like(acceleration)
    output[0] = vector
    debug_log(debug_enabled, "transport", "fermi_walker:start", samples=len(worldline.parameter))

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
        if debug_enabled and ((i + 1) % debug_every == 0 or i == len(worldline.parameter) - 2):
            debug_log(
                True,
                "transport",
                "fermi_walker:step",
                index=i + 1,
                vector=np.array2string(vector, precision=6),
            )
    return output

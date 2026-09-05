"""Timelike and null geodesic integration on sampled spacetimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tensor_toolkit.physics.worldline import Worldline
from .sampling import SpacetimeSampler


StopCondition = Callable[[np.ndarray, np.ndarray], bool]


@dataclass(frozen=True)
class GeodesicResult:
    worldline: Worldline
    causal_type: str
    normalization: np.ndarray
    normalization_error: np.ndarray
    max_normalization_error: float


def tangent_norm(metric: np.ndarray, tangent: np.ndarray) -> float:
    return float(np.einsum("m,mn,n->", tangent, metric, tangent))


def _rhs(sampler: SpacetimeSampler, state: np.ndarray) -> np.ndarray:
    coordinates = state[:4]
    tangent = state[4:]
    gamma = sampler.connection_at(coordinates)
    acceleration = -np.einsum(
        "mab,a,b->m",
        gamma,
        tangent,
        tangent,
        optimize=True,
    )
    return np.concatenate((tangent, acceleration))


def _rk4_step(sampler: SpacetimeSampler, state: np.ndarray, step: float) -> np.ndarray:
    k1 = _rhs(sampler, state)
    k2 = _rhs(sampler, state + 0.5 * step * k1)
    k3 = _rhs(sampler, state + 0.5 * step * k2)
    k4 = _rhs(sampler, state + step * k3)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_geodesic(
    sampler: SpacetimeSampler,
    initial_coordinates,
    initial_tangent,
    *,
    affine_step: float,
    steps: int,
    causal_type: str = "timelike",
    stop_condition: StopCondition | None = None,
    name: str = "geodesic",
) -> GeodesicResult:
    """Integrate the geodesic equation with a fixed-step RK4 reference solver."""

    coordinates0 = np.asarray(initial_coordinates, dtype=np.float64)
    tangent0 = np.asarray(initial_tangent, dtype=np.float64)
    if coordinates0.shape != (4,) or tangent0.shape != (4,):
        raise ValueError("initial coordinates and tangent must have shape (4,)")
    if not np.all(np.isfinite(coordinates0)) or not np.all(np.isfinite(tangent0)):
        raise ValueError("initial geodesic state must be finite")
    affine_step = float(affine_step)
    steps = int(steps)
    if affine_step <= 0.0 or steps < 1:
        raise ValueError("affine_step must be positive and steps must be at least 1")
    causal_type = causal_type.lower()
    if causal_type not in {"timelike", "null"}:
        raise ValueError("causal_type must be 'timelike' or 'null'")

    initial_norm = tangent_norm(sampler.metric_at(coordinates0), tangent0)
    if causal_type == "timelike" and initial_norm >= 0.0:
        raise ValueError("timelike geodesic requires a timelike initial tangent")
    if causal_type == "null":
        scale = max(1.0, float(np.max(np.abs(tangent0))))
        if abs(initial_norm) > 1e-8 * scale**2:
            raise ValueError("null geodesic initial tangent must satisfy g(k,k)=0")

    target = initial_norm if causal_type == "timelike" else 0.0
    parameter = [0.0]
    coordinates = [coordinates0.copy()]
    tangents = [tangent0.copy()]
    norms = [initial_norm]

    state = np.concatenate((coordinates0, tangent0))
    for index in range(steps):
        state = _rk4_step(sampler, state, affine_step)
        event = state[:4].copy()
        tangent = state[4:].copy()
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("geodesic integration produced non-finite state")
        parameter.append((index + 1) * affine_step)
        coordinates.append(event)
        tangents.append(tangent)
        norms.append(tangent_norm(sampler.metric_at(event), tangent))
        if stop_condition is not None and stop_condition(event, tangent):
            break

    parameter_array = np.asarray(parameter, dtype=np.float64)
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    tangent_array = np.asarray(tangents, dtype=np.float64)
    normalization = np.asarray(norms, dtype=np.float64)
    normalization_error = np.abs(normalization - target)

    worldline = Worldline(
        parameter=parameter_array,
        coordinates=coordinate_array,
        tangent=tangent_array,
        body_name=name,
        metadata={
            "source": "geodesic",
            "causal_type": causal_type,
            "parameter": "affine",
            "integrator": "rk4",
            "affine_step": affine_step,
            "normalization_target": float(target),
        },
    )
    return GeodesicResult(
        worldline=worldline,
        causal_type=causal_type,
        normalization=normalization,
        normalization_error=normalization_error,
        max_normalization_error=float(np.max(normalization_error)),
    )

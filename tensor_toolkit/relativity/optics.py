"""Scientific null-ray and ray-bundle construction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .debug import debug_log
from .frames import ObserverFrame
from .geodesics import GeodesicResult, integrate_geodesic
from .sampling import SpacetimeSampler


def null_tangent_from_local_direction(
    frame: ObserverFrame,
    direction,
    *,
    debug: bool = False,
) -> np.ndarray:
    """Construct future null k^mu from a unit spatial direction in an observer frame."""

    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("direction must have shape (3,)")
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    n = direction / norm
    tangent = frame.time_basis + np.einsum("i,im->m", n, frame.spatial_basis)
    norm2 = float(np.einsum("m,mn,n->", tangent, frame.metric, tangent))
    if not np.isclose(norm2, 0.0, atol=1e-10, rtol=1e-10):
        raise ValueError("constructed ray tangent is not null")
    debug_log(
        debug,
        "optics",
        "null_tangent",
        observer=frame.name,
        direction=np.array2string(n, precision=6),
        tangent=np.array2string(tangent, precision=6),
        norm=norm2,
    )
    return tangent


@dataclass(frozen=True)
class RayBundle:
    observer_name: str
    directions: np.ndarray
    rays: tuple[GeodesicResult, ...]


def trace_null_ray(
    sampler: SpacetimeSampler,
    frame: ObserverFrame,
    direction,
    *,
    affine_step: float,
    steps: int,
    stop_condition=None,
    name: str = "ray",
    debug: bool = False,
    debug_every: int = 1,
) -> GeodesicResult:
    debug_enabled = bool(debug or sampler.debug)
    tangent = null_tangent_from_local_direction(frame, direction, debug=debug_enabled)
    debug_log(debug_enabled, "optics", "trace_null_ray:start", name=name)
    result = integrate_geodesic(
        sampler,
        frame.coordinates,
        tangent,
        affine_step=affine_step,
        steps=steps,
        causal_type="null",
        stop_condition=stop_condition,
        name=name,
        debug=debug_enabled,
        debug_every=debug_every,
    )
    debug_log(
        debug_enabled,
        "optics",
        "trace_null_ray:done",
        name=name,
        steps=len(result.worldline.parameter) - 1,
    )
    return result


def trace_ray_bundle(
    sampler: SpacetimeSampler,
    frame: ObserverFrame,
    directions,
    *,
    affine_step: float,
    steps: int,
    stop_condition=None,
    debug: bool = False,
    debug_every: int = 1,
) -> RayBundle:
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (N,3)")
    debug_enabled = bool(debug or sampler.debug)
    debug_log(
        debug_enabled,
        "optics",
        "trace_ray_bundle:start",
        observer=frame.name,
        rays=len(directions),
    )
    rays = tuple(
        trace_null_ray(
            sampler,
            frame,
            direction,
            affine_step=affine_step,
            steps=steps,
            stop_condition=stop_condition,
            name=f"ray-{index}",
            debug=debug_enabled,
            debug_every=debug_every,
        )
        for index, direction in enumerate(directions)
    )
    debug_log(debug_enabled, "optics", "trace_ray_bundle:done", rays=len(rays))
    return RayBundle(frame.name, directions.copy(), rays)

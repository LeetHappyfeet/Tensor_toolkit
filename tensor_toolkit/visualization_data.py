"""Renderer-neutral visualization data adapters.

This module is the contract between validated physics/GR results and any GUI
renderer.  It depends only on NumPy and public Tensor Toolkit result objects;
VTK/PyVista code must live downstream of these adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class VolumeData:
    name: str
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PolylineData:
    name: str
    points: np.ndarray
    scalars: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GlyphData:
    name: str
    origins: np.ndarray
    vectors: np.ndarray
    magnitudes: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PointData:
    name: str
    points: np.ndarray
    labels: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


def _spatial_axes(result):
    if len(result.axis_values) != 4:
        raise ValueError("experiment result must provide four coordinate axes")
    return tuple(np.asarray(v, dtype=np.float64) for v in result.axis_values[1:4])


def experiment_volume(result, field_name: str, *, component=(0, 0), time_index: int = 0) -> VolumeData:
    """Extract one spatial 3-D scalar field at a selected coordinate-time index."""
    if field_name not in result.fields:
        raise KeyError(f"field {field_name!r} is not present in the result")
    value = np.asarray(result.fields[field_name])
    if value.ndim == 6 and value.shape[:2] == (4, 4):
        mu, nu = map(int, component)
        if mu not in range(4) or nu not in range(4):
            raise ValueError("rank-2 component indices must be in [0, 3]")
        scalar = value[mu, nu]
    elif value.ndim == 4:
        scalar = value
    else:
        raise ValueError(
            f"{field_name!r} cannot be mapped to a scalar volume; expected "
            "(4,4,Nt,Nx,Ny,Nz) or (Nt,Nx,Ny,Nz)"
        )
    nt = scalar.shape[0]
    index = int(time_index)
    if index < 0:
        index += nt
    if not 0 <= index < nt:
        raise IndexError(f"time index {time_index} outside [0, {nt - 1}]")
    x, y, z = _spatial_axes(result)
    values = np.asarray(scalar[index], dtype=np.float64)
    expected = (x.size, y.size, z.size)
    if values.shape != expected:
        raise ValueError(f"spatial volume shape {values.shape} does not match axes {expected}")
    return VolumeData(
        name=f"{field_name}[{component[0]},{component[1]}]",
        values=values,
        x=x,
        y=y,
        z=z,
        metadata={
            "field": field_name,
            "component": tuple(map(int, component)),
            "time_index": index,
            "time": float(result.axis_values[0][index]),
            "metric": result.metric_name,
            "validation": result.metadata.get("diagnostics", {}),
        },
    )


def trajectory_polylines(trajectory) -> tuple[PolylineData, ...]:
    """Convert all classical bodies in a Trajectory to independent 3-D polylines."""
    positions = np.asarray(trajectory.positions, dtype=np.float64)
    return tuple(
        PolylineData(
            name=str(name),
            points=positions[:, i, :].copy(),
            scalars=np.asarray(trajectory.times, dtype=np.float64).copy(),
            metadata={"source": "trajectory", "body": str(name), "scalar_name": "time"},
        )
        for i, name in enumerate(trajectory.body_names)
    )


def worldline_polyline(worldline) -> PolylineData:
    """Project a four-dimensional worldline to its spatial x,y,z coordinates."""
    coordinates = np.asarray(worldline.coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("worldline coordinates must have shape (N,4)")
    return PolylineData(
        name=str(worldline.body_name),
        points=coordinates[:, 1:4].copy(),
        scalars=np.asarray(worldline.parameter, dtype=np.float64).copy(),
        metadata={"source": "worldline", **dict(getattr(worldline, "metadata", {}))},
    )


def ray_bundle_polylines(bundle) -> tuple[PolylineData, ...]:
    """Convert a relativity.optics.RayBundle to renderer-neutral polylines."""
    lines = []
    for index, ray in enumerate(bundle.rays):
        line = worldline_polyline(ray.worldline)
        lines.append(
            PolylineData(
                name=f"{bundle.observer_name}:ray-{index}",
                points=line.points,
                scalars=line.scalars,
                metadata={
                    **line.metadata,
                    "source": "ray_bundle",
                    "observer": str(bundle.observer_name),
                    "ray_index": index,
                },
            )
        )
    return tuple(lines)


def observer_frame_glyphs(frame, *, scale: float = 1.0) -> GlyphData:
    """Return the three spatial tetrad axes as glyph vectors at the observer event."""
    origin = np.asarray(frame.coordinates, dtype=np.float64)[1:4]
    vectors = np.asarray(frame.spatial_basis, dtype=np.float64)[:, 1:4] * float(scale)
    return GlyphData(
        name=str(frame.name),
        origins=np.repeat(origin[None, :], 3, axis=0),
        vectors=vectors,
        magnitudes=np.linalg.norm(vectors, axis=1),
        metadata={"source": "observer_frame"},
    )


def tidal_eigen_glyphs(origin, tidal, *, scale: float = 1.0, name: str = "tidal") -> GlyphData:
    """Represent the principal directions of a symmetric 3x3 tidal tensor."""
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape == (4,):
        origin = origin[1:4]
    if origin.shape != (3,):
        raise ValueError("tidal glyph origin must have shape (3,) or (4,)")
    tensor = np.asarray(tidal, dtype=np.float64)
    if tensor.shape != (3, 3):
        raise ValueError("tidal tensor must have shape (3,3)")
    values, vectors = np.linalg.eigh(0.5 * (tensor + tensor.T))
    arrows = vectors.T * (np.abs(values) * float(scale))[:, None]
    return GlyphData(
        name=name,
        origins=np.repeat(origin[None, :], 3, axis=0),
        vectors=arrows,
        magnitudes=values,
        metadata={"source": "tidal_tensor", "eigenvalues": values.copy()},
    )


def trajectory_event_points(trajectory) -> PointData:
    """Map trajectory events to body positions at the nearest accepted time sample."""
    points = []
    labels = []
    times = np.asarray(trajectory.times, dtype=np.float64)
    for event in trajectory.events:
        index = int(np.argmin(np.abs(times - float(event.time))))
        if event.bodies:
            try:
                body_index = trajectory.body_names.index(event.bodies[-1])
            except ValueError:
                body_index = 0
        else:
            body_index = 0
        points.append(np.asarray(trajectory.positions[index, body_index], dtype=np.float64))
        labels.append(str(event.kind))
    array = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    return PointData(
        name="events",
        points=array,
        labels=tuple(labels),
        metadata={"source": "trajectory_events"},
    )


__all__ = [
    "VolumeData", "PolylineData", "GlyphData", "PointData",
    "experiment_volume", "trajectory_polylines", "worldline_polyline",
    "ray_bundle_polylines", "observer_frame_glyphs", "tidal_eigen_glyphs",
    "trajectory_event_points",
]

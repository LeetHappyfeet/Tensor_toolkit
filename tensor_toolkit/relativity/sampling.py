"""Metric and curvature sampling at arbitrary spacetime events."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

from tensor_toolkit.experiment import compute_tensor_fields, SUPPORTED_OUTPUTS
from tensor_toolkit.metrics import Metric
from tensor_toolkit.physics.worldline import Worldline
from .debug import debug_log
from .point_geometry import metric_point, point_connection_fields, interpolated_grid_fields


@dataclass(frozen=True)
class EventGeometry:
    """Locally sampled geometry at one spacetime event."""

    coordinates: np.ndarray
    metric: np.ndarray
    inverse_metric: np.ndarray | None = None
    christoffel: np.ndarray | None = None
    riemann: np.ndarray | None = None
    ricci: np.ndarray | None = None
    ricci_scalar: float | None = None
    einstein: np.ndarray | None = None
    stress_energy: np.ndarray | None = None


@dataclass(frozen=True)
class WorldlineFieldSamples:
    coordinates: np.ndarray
    fields: dict[str, np.ndarray]
    body_name: str


@dataclass(frozen=True)
class SpacetimeSampler:
    """Sample an analytic metric and its derived local geometry."""

    metric: Metric
    spacings: tuple[float, float, float, float]
    units: str = "geometrized"
    debug: bool = False
    method: str = "numerical"  # numerical, analytic, finite4, cached_grid
    grid_result: object | None = field(default=None, repr=False, compare=False)
    cache_size: int = 32
    _field_cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False, compare=False)
    _cache_hits: int = field(default=0, init=False, repr=False, compare=False)
    _cache_misses: int = field(default=0, init=False, repr=False, compare=False)
    _metric_signature: tuple = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        spacings = tuple(float(value) for value in self.spacings)
        if len(spacings) != 4 or any(value <= 0.0 for value in spacings):
            raise ValueError("spacings must contain four positive values")
        if not all(np.isfinite(spacings)):
            raise ValueError("spacings must be finite")
        if self.method not in {"numerical", "analytic", "finite4", "cached_grid"}:
            raise ValueError("method must be numerical, analytic, finite4 or cached_grid")
        if self.method == "cached_grid" and self.grid_result is None:
            raise ValueError("cached_grid requires a completed ExperimentResult")
        if not isinstance(self.cache_size, int) or self.cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        object.__setattr__(self, "spacings", spacings)
        object.__setattr__(self, "_metric_signature", (
            repr(self.metric), tuple(getattr(self.metric, "coordinates", ())),
            repr(getattr(self.metric, "definition", None))))
        debug_log(
            self.debug,
            "sampler",
            "initialized",
            metric=getattr(self.metric, "name", type(self.metric).__name__),
            spacings=self.spacings,
            units=self.units,
        )

    @staticmethod
    def _event(event) -> np.ndarray:
        event = np.asarray(event, dtype=np.float64)
        if event.shape != (4,) or not np.all(np.isfinite(event)):
            raise ValueError("event must be a finite shape-(4,) coordinate")
        return event

    def cache_info(self) -> dict[str, int]:
        """Per-instance bounded memoization (do not mutate/share the metric concurrently)."""
        return {"events": len(self._field_cache), "max_events": self.cache_size,
                "hits": self._cache_hits, "misses": self._cache_misses}

    def clear_cache(self) -> None:
        self._field_cache.clear()
        object.__setattr__(self, "_cache_hits", 0)
        object.__setattr__(self, "_cache_misses", 0)

    def strategy_info(self) -> dict[str, object]:
        return {"method": self.method, "spacings": self.spacings, "units": self.units,
                "cache_size": self.cache_size,
                "field_interpolation": "multilinear 4D (no extrapolation)" if self.method == "cached_grid" else None,
                "derivative_order": 4 if self.method == "finite4" else (2 if self.method == "numerical" else None),
                "curvature_from_point_derivatives": self.method == "numerical"}

    def metric_at(self, event) -> np.ndarray:
        event = self._event(event)
        if self.method == "cached_grid":
            return np.asarray(self.fields_at(event, {"metric"})["metric"]).copy()
        return metric_point(self.metric, event)

    def inverse_metric_at(self, event) -> np.ndarray:
        if self.method == "cached_grid":
            return np.asarray(self.fields_at(event, {"inverse_metric"})["inverse_metric"]).copy()
        return np.linalg.inv(self.metric_at(event))

    def _compute_fields(self, event, outputs):
        if self.method in {"analytic", "finite4"}:
            return point_connection_fields(self.metric, event, self.spacings, outputs,
                                           method=self.method)
        if self.method == "cached_grid":
            return interpolated_grid_fields(self.grid_result, self.metric, event,
                                            outputs, units=self.units)
        # Historical general reference: a full 3^4 local stencil. Curvature
        # differentiation here is second order and must be convergence-tested.
        axes = tuple(
            event[i] + self.spacings[i] * np.array([-1., 0., 1.], dtype=np.float64)
            for i in range(4)
        )
        grid = tuple(np.meshgrid(*axes, indexing="ij", sparse=True))
        metric = self.metric.evaluate(grid)
        fields = compute_tensor_fields(metric, self.spacings, outputs, units=self.units)
        center = (1, 1, 1, 1)
        out = {}
        for name, value in fields.items():
            prefix = value.ndim - 4
            part = np.asarray(value[(slice(None),) * prefix + center]).copy()
            out[name] = float(part) if part.ndim == 0 else part
        return out

    def fields_at(self, event, outputs) -> dict[str, np.ndarray | float]:
        event = self._event(event)
        if self._metric_signature != (repr(self.metric), tuple(getattr(self.metric, "coordinates", ())),
                                      repr(getattr(self.metric, "definition", None))):
            self.clear_cache()
            raise ValueError("metric parameters/chart changed after sampler construction; create a new sampler")
        outputs = frozenset(outputs)
        if not outputs or outputs - SUPPORTED_OUTPUTS:
            raise ValueError(f"invalid field request: {sorted(outputs)}")
        key = tuple(float(x) for x in event)
        entry = self._field_cache.get(key) if self.cache_size else None
        if entry is not None and outputs.issubset(entry):
            object.__setattr__(self, "_cache_hits", self._cache_hits + 1)
            self._field_cache.move_to_end(key)
            return {name: (value.copy() if isinstance(value, np.ndarray) else value)
                    for name, value in entry.items() if name in outputs}
        object.__setattr__(self, "_cache_misses", self._cache_misses + 1)
        needed = outputs - set(entry or ())
        computed = self._compute_fields(event, needed)
        if self.cache_size:
            if entry is None:
                entry = {}
            entry.update({name: (value.copy() if isinstance(value, np.ndarray) else value)
                          for name, value in computed.items()})
            self._field_cache[key] = entry
            self._field_cache.move_to_end(key)
            while len(self._field_cache) > self.cache_size:
                self._field_cache.popitem(last=False)
            selected = entry
        else:
            selected = computed
        return {name: (selected[name].copy() if isinstance(selected[name], np.ndarray)
                       else selected[name]) for name in outputs}

    def fields_along_worldline(self, worldline: Worldline, outputs) -> WorldlineFieldSamples:
        outputs = frozenset(outputs)
        accumulated: dict[str, list[np.ndarray]] = {name: [] for name in outputs}
        for index, event in enumerate(worldline.coordinates):
            debug_log(
                self.debug,
                "sampler",
                "worldline_sample",
                index=index,
                body=worldline.body_name,
            )
            fields = self.fields_at(event, outputs)
            for name in outputs:
                accumulated[name].append(np.asarray(fields[name]))
        return WorldlineFieldSamples(
            coordinates=worldline.coordinates.copy(),
            fields={name: np.stack(values, axis=0) for name, values in accumulated.items()},
            body_name=worldline.body_name,
        )

    def connection_at(self, event) -> np.ndarray:
        return np.asarray(self.fields_at(event, {"christoffel"})["christoffel"])

    def riemann_at(self, event) -> np.ndarray:
        return np.asarray(self.fields_at(event, {"riemann"})["riemann"])

    def geometry_at(
        self,
        event,
        *,
        include=("inverse_metric", "christoffel", "riemann", "ricci", "ricci_scalar"),
    ) -> EventGeometry:
        event = self._event(event)
        requested = frozenset(include)
        fields = self.fields_at(event, {"metric", *requested})
        return EventGeometry(
            coordinates=event.copy(),
            metric=np.asarray(fields["metric"]),
            inverse_metric=np.asarray(fields["inverse_metric"]) if "inverse_metric" in fields else None,
            christoffel=np.asarray(fields["christoffel"]) if "christoffel" in fields else None,
            riemann=np.asarray(fields["riemann"]) if "riemann" in fields else None,
            ricci=np.asarray(fields["ricci"]) if "ricci" in fields else None,
            ricci_scalar=float(fields["ricci_scalar"]) if "ricci_scalar" in fields else None,
            einstein=np.asarray(fields["einstein"]) if "einstein" in fields else None,
            stress_energy=np.asarray(fields["stress_energy"]) if "stress_energy" in fields else None,
        )

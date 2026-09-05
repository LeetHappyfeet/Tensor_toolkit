"""Metric and curvature sampling at arbitrary spacetime events."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tensor_toolkit.experiment import compute_tensor_fields
from tensor_toolkit.metrics import Metric
from tensor_toolkit.physics.worldline import Worldline


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

    def __post_init__(self) -> None:
        spacings = tuple(float(value) for value in self.spacings)
        if len(spacings) != 4 or any(value <= 0.0 for value in spacings):
            raise ValueError("spacings must contain four positive values")
        object.__setattr__(self, "spacings", spacings)

    @staticmethod
    def _event(event) -> np.ndarray:
        event = np.asarray(event, dtype=np.float64)
        if event.shape != (4,) or not np.all(np.isfinite(event)):
            raise ValueError("event must be a finite shape-(4,) coordinate")
        return event

    def metric_at(self, event) -> np.ndarray:
        event = self._event(event)
        coordinates = tuple(np.asarray([event[i]], dtype=np.float64) for i in range(4))
        value = np.asarray(self.metric.evaluate(coordinates), dtype=np.float64)
        if value.shape != (4, 4, 1):
            raise ValueError(f"metric evaluator returned unexpected point shape {value.shape}")
        return value[..., 0]

    def inverse_metric_at(self, event) -> np.ndarray:
        return np.linalg.inv(self.metric_at(event))

    def fields_at(self, event, outputs) -> dict[str, np.ndarray | float]:
        event = self._event(event)
        outputs = frozenset(outputs)
        if not outputs:
            raise ValueError("at least one field must be requested")
        axes = tuple(
            event[i] + self.spacings[i] * np.array([-1.0, 0.0, 1.0], dtype=np.float64)
            for i in range(4)
        )
        grid = tuple(np.meshgrid(*axes, indexing="ij", sparse=True))
        metric = self.metric.evaluate(grid)
        fields = compute_tensor_fields(metric, self.spacings, outputs, units=self.units)
        center = (1, 1, 1, 1)
        out: dict[str, np.ndarray | float] = {}
        for name, field in fields.items():
            prefix = field.ndim - 4
            value = np.asarray(field[(slice(None),) * prefix + center]).copy()
            out[name] = float(value) if value.ndim == 0 else value
        return out

    def fields_along_worldline(self, worldline: Worldline, outputs) -> WorldlineFieldSamples:
        outputs = frozenset(outputs)
        accumulated: dict[str, list[np.ndarray]] = {name: [] for name in outputs}
        for event in worldline.coordinates:
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

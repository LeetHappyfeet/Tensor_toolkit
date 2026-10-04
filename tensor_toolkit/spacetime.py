"""Spacetime catalogue metadata; no renderer or numerical-solver dependency.

A coordinate chart is part of a metric's contract. In particular, (t,x,y,z)
and (ct,r,theta,phi) are NOT interchangeable input arrays.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateChart:
    names: tuple[str, str, str, str]
    units: tuple[str, str, str, str]
    kind: str
    domain: str

    def __post_init__(self):
        if len(self.names) != 4 or len(self.units) != 4:
            raise ValueError("a spacetime chart requires four named, unit-bearing coordinates")
        if len(set(self.names)) != 4:
            raise ValueError("coordinate names must be distinct")
        if self.kind not in {"cartesian", "spherical", "isotropic-cartesian", "kerr-schild-cartesian"}:
            raise ValueError("unsupported chart kind")


@dataclass(frozen=True)
class SpacetimeDefinition:
    family: str
    chart: CoordinateChart
    source_type: str  # exact, approximate, prescribed, flat
    reference: str
    notes: str = ""

    def __post_init__(self):
        if self.source_type not in {"exact", "approximate", "prescribed", "flat"}:
            raise ValueError("invalid source type")


CATALOGUE_REFERENCE = "Mueller et al., Catalogue of Spacetimes, arXiv:0904.4184v3"


def describe_spacetime(metric) -> SpacetimeDefinition:
    """Return explicit, immutable metric metadata; fail rather than guessing units."""
    definition = getattr(metric, "definition", None)
    if not isinstance(definition, SpacetimeDefinition):
        raise TypeError(f"{type(metric).__name__} has no spacetime definition")
    if tuple(metric.coordinates) != definition.chart.names:
        raise ValueError("metric coordinates disagree with its chart definition")
    return definition


__all__ = ["CoordinateChart", "SpacetimeDefinition", "CATALOGUE_REFERENCE", "describe_spacetime"]

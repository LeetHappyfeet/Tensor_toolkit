"""Headless Alcubierre parameter exploration and sampling diagnostics.

No VTK or GUI imports belong in this module. Width is measured directly from
the metric's normalized shape function at its 10% and 90% radial crossings.
"""
from dataclasses import dataclass, replace
import numpy as np
from tensor_toolkit.metrics import AlcubierreMetric
from tensor_toolkit.experiment import Experiment, run_experiment


@dataclass(frozen=True)
class WallResolution:
    width_10_90: float
    spacing: float
    cells_across_wall: float
    status: str


def wall_resolution(metric: AlcubierreMetric, spatial_axes) -> WallResolution:
    if not isinstance(metric, AlcubierreMetric):
        raise TypeError("wall diagnostic requires AlcubierreMetric")
    if not np.isfinite(metric.radius) or not np.isfinite(metric.sigma) or metric.radius <= 0 or metric.sigma <= 0:
        raise ValueError("radius and sigma must be finite and positive")
    spacings = []
    for axis in spatial_axes:
        values = axis.values()
        spacings.append(float(np.max(np.diff(values))))
    spacing = max(spacings)
    # The radial profile is monotone for r>=0. Locate actual 90% and 10%
    # crossings rather than assuming a dimensionally incorrect radius/sigma.
    f0 = float(metric.shape_function(np.array(0.0)))
    if f0 < 0.9:
        raise ValueError("shape function has no 90% crossing")
    def crossing(level):
        lo, hi = 0.0, max(metric.radius, 1.0 / metric.sigma)
        for _ in range(80):
            if float(metric.shape_function(np.array(hi))) <= level:
                break
            hi *= 2.0
        else:
            raise ValueError("could not bracket shape-function crossing")
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if float(metric.shape_function(np.array(mid))) > level:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    width = crossing(0.1) - crossing(0.9)
    cells = width / spacing
    status = "under-resolved" if cells < 4 else "marginal" if cells < 8 else "resolved"
    return WallResolution(width, spacing, cells, status)


SIGMA_PRESETS = {"Diffuse": 0.5, "Standard": 2.0, "Sharp": 10.0, "Extreme": 50.0}


def sigma_sweep(experiment: Experiment, values):
    """Yield sigma and authoritative results; never interpolate tensor fields."""
    if not isinstance(experiment.metric, AlcubierreMetric):
        raise TypeError("sigma sweep requires AlcubierreMetric")
    values = tuple(float(v) for v in values)
    if not values or any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("sigma values must be finite and positive")
    for sigma in values:
        configured = replace(experiment, metric=replace(experiment.metric, sigma=sigma))
        yield sigma, wall_resolution(configured.metric, configured.axes[1:]), run_experiment(configured)

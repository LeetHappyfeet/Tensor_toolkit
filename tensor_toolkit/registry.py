"""Built-in Phase-2 experiment registry and grid configuration."""

from dataclasses import replace

from tensor_toolkit.experiment import Axis, Experiment
from tensor_toolkit.metrics import (
    AlcubierreMetric, DeSitterFlatMetric, MinkowskiMetric,
    SchwarzschildIsotropicMetric, KerrBoyerLindquistMetric, KerrSchildMetric,
    ReissnerNordstromMetric, FLRWMetric, LinearizedPlaneWaveMetric, WeakField1PNMetric,
)
from tensor_toolkit.constants import SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT
from tensor_toolkit.spacetime import describe_spacetime


def _axes(extent: float = 1.0, points: int = 3):
    return tuple(Axis(-extent, extent, points) for _ in range(4))


def _axes4(time, first, second, third, points=5):
    return tuple(Axis(*bounds, points) for bounds in (time, first, second, third))


def builtins() -> dict[str, Experiment]:
    common = frozenset({"metric", "einstein", "stress_energy"})
    # m=GM/c² = 1 metre for stable small default compact-object stencils.
    mass_ref = SPEED_OF_LIGHT**2 / GRAVITATIONAL_CONSTANT
    exterior_cart = _axes4((-1, 1), (5, 7), (-1, 1), (-1, 1))
    exterior_spherical = _axes4((-1, 1), (5, 7), (0.5, 2.5), (-1, 1))
    return {
        "minkowski": Experiment(MinkowskiMetric(), _axes(), common),
        "de-sitter": Experiment(DeSitterFlatMetric(hubble=0.1), _axes(), common),
        "alcubierre": Experiment(
            AlcubierreMetric(velocity=0.1, radius=1.0, sigma=2.0),
            _axes(2.0, 5),
            common,
        ),
        "schwarzschild": Experiment(SchwarzschildIsotropicMetric(mass_ref), exterior_cart, common),
        "kerr": Experiment(KerrSchildMetric(mass_ref, spin=0.5), exterior_cart, common),
        "kerr-bl": Experiment(KerrBoyerLindquistMetric(mass_ref, spin=0.5), exterior_spherical, common),
        "reissner-nordstrom": Experiment(
            ReissnerNordstromMetric(mass_ref, charge_coulombs=2e16), exterior_spherical, common),
        "weak-field-1pn": Experiment(
            WeakField1PNMetric(mass_ref), _axes4((-1, 1), (25, 29), (-2, 2), (-2, 2)), common),
        "flrw": Experiment(FLRWMetric(hubble=0.1), _axes(0.8, 5), common),
        "flrw-closed": Experiment(FLRWMetric(hubble=0.1, curvature=1), _axes(0.8, 5), common),
        "flrw-open": Experiment(FLRWMetric(hubble=0.1, curvature=-1), _axes(0.8, 5), common),
        "plane-wave": Experiment(LinearizedPlaneWaveMetric(plus=0.03), _axes(1.0, 5), common),
    }


def get_experiment(name: str) -> Experiment:
    experiments = builtins()
    if name not in experiments:
        raise KeyError(
            f"unknown experiment {name!r}; choose from: {', '.join(sorted(experiments))}"
        )
    return experiments[name]


def configure_grid(
    experiment: Experiment,
    *,
    points: int | None = None,
    extent: float | None = None,
    time_points: int | None = None,
    time_start: float | None = None,
    time_stop: float | None = None,
) -> Experiment:
    """Return an experiment with independent time and spatial grid overrides.

    points and extent apply to the three spatial axes. For backwards
    compatibility, when no explicit time override is supplied, they continue
    to apply to all four axes as before.
    """
    if points is not None and points < 3:
        raise ValueError("--points must be at least 3")
    if extent is not None and extent <= 0:
        raise ValueError("--extent must be positive")
    if extent is not None and describe_spacetime(experiment.metric).chart.kind != "cartesian":
        raise ValueError(
            "symmetric --extent is only valid for unrestricted Cartesian charts; "
            "configure each axis explicitly for compact-object or spherical charts"
        )
    if time_points is not None and time_points < 3:
        raise ValueError("time_points must be at least 3")
    if (time_start is None) ^ (time_stop is None):
        raise ValueError("time_start and time_stop must be provided together")
    if time_start is not None and float(time_stop) <= float(time_start):
        raise ValueError("time_stop must be greater than time_start")

    explicit_time = time_points is not None or time_start is not None
    axes = []
    for index, axis in enumerate(experiment.axes):
        if index == 0 and explicit_time:
            axis_points = axis.points if time_points is None else int(time_points)
            start = axis.start if time_start is None else float(time_start)
            stop = axis.stop if time_stop is None else float(time_stop)
        else:
            axis_points = axis.points if points is None else int(points)
            if extent is None:
                start, stop = axis.start, axis.stop
            else:
                start, stop = -float(extent), float(extent)
        axes.append(Axis(start, stop, axis_points))
    return replace(experiment, axes=tuple(axes))


__all__ = ["builtins", "get_experiment", "configure_grid", "gui_builtins"]


# The Cartesian-only volume GUI must not mistake (r,theta,phi) for (x,y,z).
def gui_builtins() -> dict[str, Experiment]:
    return {key: value for key, value in builtins().items()
            if key in {"minkowski", "de-sitter", "alcubierre", "flrw",
                       "flrw-closed", "plane-wave"}}

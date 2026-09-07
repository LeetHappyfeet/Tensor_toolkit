"""Built-in Phase-2 experiment registry and grid configuration."""

from dataclasses import replace

from tensor_toolkit.experiment import Axis, Experiment
from tensor_toolkit.metrics import AlcubierreMetric, DeSitterFlatMetric, MinkowskiMetric


def _axes(extent: float = 1.0, points: int = 3):
    return tuple(Axis(-extent, extent, points) for _ in range(4))


def builtins() -> dict[str, Experiment]:
    common = frozenset({"metric", "einstein", "stress_energy"})
    return {
        "minkowski": Experiment(MinkowskiMetric(), _axes(), common),
        "de-sitter": Experiment(DeSitterFlatMetric(hubble=0.1), _axes(), common),
        "alcubierre": Experiment(
            AlcubierreMetric(velocity=0.1, radius=1.0, sigma=2.0),
            _axes(2.0, 5),
            common,
        ),
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


__all__ = ["builtins", "get_experiment", "configure_grid"]

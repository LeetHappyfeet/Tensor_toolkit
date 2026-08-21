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
) -> Experiment:
    """Return an experiment with a uniform four-axis grid override."""
    if points is not None and points < 3:
        raise ValueError("--points must be at least 3")
    if extent is not None and extent <= 0:
        raise ValueError("--extent must be positive")

    axes = []
    for axis in experiment.axes:
        axis_points = axis.points if points is None else int(points)
        if extent is None:
            start, stop = axis.start, axis.stop
        else:
            start, stop = -float(extent), float(extent)
        axes.append(Axis(start, stop, axis_points))
    return replace(experiment, axes=tuple(axes))


__all__ = ["builtins", "get_experiment", "configure_grid"]

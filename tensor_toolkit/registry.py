"""Built-in Phase-2 experiment registry."""

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


__all__ = ["builtins", "get_experiment"]

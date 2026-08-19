import numpy as np

from tensor_toolkit.experiment import Axis, Experiment, run_experiment
from tensor_toolkit.metrics import AlcubierreMetric, MinkowskiMetric


def small_axes(n=5):
    return (
        Axis(-0.2, 0.2, n),
        Axis(-2.0, 2.0, n),
        Axis(-2.0, 2.0, n),
        Axis(-2.0, 2.0, n),
    )


def test_headless_minkowski_experiment_returns_requested_fields():
    experiment = Experiment(
        metric=MinkowskiMetric(),
        axes=small_axes(),
        outputs=frozenset({"metric", "einstein", "stress_energy"}),
    )
    result = run_experiment(experiment)
    assert result.metric_name == "Minkowski"
    assert set(result.fields) == {"metric", "einstein", "stress_energy"}
    assert result.fields["metric"].dtype == np.float64
    assert np.count_nonzero(result.fields["einstein"]) == 0
    assert np.count_nonzero(result.fields["stress_energy"]) == 0


def test_alcubierre_metric_is_time_dependent_and_lorentzian():
    experiment = Experiment(
        metric=AlcubierreMetric(velocity=0.2, radius=1.0, sigma=3.0),
        axes=small_axes(),
        outputs=frozenset({"metric"}),
    )
    result = run_experiment(experiment)
    g = result.fields["metric"]
    assert not np.array_equal(g[:, :, 0], g[:, :, -1])
    center = (2, 2, 2, 2)
    matrix = g[:, :, *center]
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.count_nonzero(eigenvalues < 0.0) == 1
    assert np.count_nonzero(eigenvalues > 0.0) == 3


def test_alcubierre_zero_velocity_reduces_to_minkowski():
    axes = small_axes()
    alcubierre = run_experiment(Experiment(AlcubierreMetric(velocity=0.0), axes, frozenset({"metric"})))
    minkowski = run_experiment(Experiment(MinkowskiMetric(), axes, frozenset({"metric"})))
    assert np.allclose(alcubierre.fields["metric"], minkowski.fields["metric"])

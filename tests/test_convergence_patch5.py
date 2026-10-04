"""Patch 5 regression: truthful resolution heuristics and same-node comparison."""
from dataclasses import replace

import numpy as np
import pytest

from tensor_toolkit.convergence import (
    resolution_report, compare_nested_results, compare_three_resolutions,
)
from tensor_toolkit.diagnostics import field_diagnostics, merge_field_diagnostics
from tensor_toolkit.experiment import Axis, ExperimentResult, run_experiment
from tensor_toolkit.metrics import AlcubierreMetric
from tensor_toolkit.registry import builtins


def test_sharp_wall_reports_underresolution_and_domain_clipping():
    metric = AlcubierreMetric(velocity=.9, radius=1., sigma=10.)
    axes = (np.linspace(-2, 2, 21),) + (np.linspace(-2, 2, 48),) * 3
    report = resolution_report(metric, axes)
    wall = report["alcubierre_wall"]
    assert wall["width"] == pytest.approx(.2197, rel=.03)
    assert wall["minimum_cells"] < 2
    assert wall["suggested_points_for_six_cells"][0] > 100
    assert any("six cells" in message for message in report["warnings"])
    assert any("x-domain" in message for message in report["warnings"])
    assert report["bubble_displacement_in_x_cells"] > 2
    assert report["assessment"] == "heuristic_only"


def test_standard_wall_has_more_cells_and_no_false_certification():
    metric = AlcubierreMetric(velocity=0, radius=1., sigma=2.)
    axes = tuple(np.linspace(-2, 2, 33) for _ in range(4))
    report = resolution_report(metric, axes)
    assert report["alcubierre_wall"]["minimum_cells"] > 6
    assert report["assessment"] == "heuristic_only"


def test_streamed_statistics_and_disjoint_chunk_merge():
    data = np.array([[1., -2., 3.], [4., 0., -6.]])
    first, second = field_diagnostics(data[:1]), field_diagnostics(data[1:])
    combined = merge_field_diagnostics(first, second)
    direct = field_diagnostics(data)
    for name in ("min", "max", "max_abs", "rms", "value_count", "sum_of_squares"):
        assert combined[name] == pytest.approx(direct[name])
    invalid = field_diagnostics(np.asarray([1., np.nan, np.inf]))
    assert not invalid["finite"]
    assert invalid["nonfinite_count"] == 2
    assert invalid["finite_count"] == 1


def _synthetic(n, *, config="fixed"):
    axes = tuple(np.linspace(-2, 2, n) for _ in range(4))
    values = np.meshgrid(*axes, sparse=True, indexing="ij")
    # Truncation-error stand-in with formal O(h^2), same physical function
    h = float(axes[0][1] - axes[0][0])
    f = np.broadcast_to(sum(a**2 for a in values), (n,) * 4).copy()
    f += h*h
    return ExperimentResult(
        metric_name="synthetic", coordinates=("t", "x", "y", "z"),
        axis_values=axes, fields={"ricci_scalar": f},
        metadata={"metric_configuration": config, "stress_energy_units": "geometrized",
                  "spacetime": {"chart": "synthetic"}},
    )


def test_nested_comparisons_have_expected_observed_order():
    coarse, medium, fine = (_synthetic(n) for n in (5, 9, 17))
    pair = compare_nested_results(coarse, medium, fields={"ricci_scalar"})
    assert pair["fields"]["ricci_scalar"]["rms_difference"] > 0
    assert pair["refinement_factors"] == (2, 2, 2, 2)
    observed = compare_three_resolutions(coarse, medium, fine, field="ricci_scalar")
    assert observed["observed_order"] == pytest.approx(2., abs=1e-10)
    assert not observed["convergence_certified"]


def test_rejects_incompatible_configuration_and_unnested_nodes():
    a, b = _synthetic(5), _synthetic(7)
    with pytest.raises(ValueError, match="nested"):
        compare_nested_results(a, b)
    with pytest.raises(ValueError, match="configuration"):
        compare_nested_results(a, _synthetic(9, config="different"))


def test_experiment_persists_resolution_statistics_and_provenance():
    experiment = replace(builtins()["minkowski"], outputs=frozenset({"metric", "einstein"}))
    result = run_experiment(experiment)
    assert result.metadata["resolution"]["grid_shape"] == (3, 3, 3, 3)
    assert result.metadata["metric_configuration"] == repr(experiment.metric)
    assert result.metadata["field_statistics"]["einstein"]["max_abs"] == pytest.approx(0.)
    assert result.metadata["derivative_provenance"]["interior_formal_order"] == 2
    assert result.metadata["derivative_provenance"]["convergence_certified"] is False


def test_connection_and_riemann_statistics_do_not_use_rank_two_symmetry():
    gamma = np.zeros((4, 4, 4, 3, 3, 3, 3), dtype=np.float64)
    gamma[0, 1, 2] = 1.
    riemann = np.zeros((4, 4, 4, 4, 3, 3, 3, 3), dtype=np.float64)
    assert "symmetry" not in field_diagnostics(gamma)
    assert "symmetry" not in field_diagnostics(riemann)
    rank2 = np.zeros((4, 4, 3, 3, 3, 3), dtype=np.float64)
    assert "symmetry" in field_diagnostics(rank2)

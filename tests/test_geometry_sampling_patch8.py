"""Patch 8 regression: exact and FD connection, grid reuse, cache integrity."""
from dataclasses import replace

import numpy as np
import pytest

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT as G, SPEED_OF_LIGHT as C
from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.metrics import MinkowskiMetric, SchwarzschildIsotropicMetric
from tensor_toolkit.registry import builtins
from tensor_toolkit.relativity.sampling import SpacetimeSampler

SPACING = (.01, .01, .01, .01)


def test_analytic_flat_connection_vanishes_and_cache_is_bounded():
    sampler = SpacetimeSampler(MinkowskiMetric(), SPACING, method="analytic",
                               cache_size=1)
    event = (0., 1., 2., 3.)
    first = sampler.connection_at(event)
    assert first.shape == (4, 4, 4)
    assert np.array_equal(first, np.zeros_like(first))
    first[0, 0, 0] = 999.
    assert sampler.connection_at(event)[0, 0, 0] == 0.
    assert sampler.cache_info()["hits"] >= 1
    sampler.connection_at((.1, 1., 2., 3.))
    assert sampler.cache_info()["events"] == 1
    sampler.clear_cache()
    assert sampler.cache_info()["events"] == 0


def test_analytic_schwarzschild_connection_agrees_with_fd_reference():
    metric = SchwarzschildIsotropicMetric(C*C/G)
    event = np.array([0., 8., 1., 0.])
    exact = SpacetimeSampler(metric, SPACING, method="analytic").connection_at(event)
    reference = SpacetimeSampler(metric, SPACING, method="numerical").connection_at(event)
    order4 = SpacetimeSampler(metric, SPACING, method="finite4").connection_at(event)
    err2 = float(np.max(np.abs(reference - exact)))
    err4 = float(np.max(np.abs(order4 - exact)))
    assert err2 < 1e-5
    assert err4 < err2
    assert np.allclose(exact, np.swapaxes(exact, 1, 2), atol=1e-14)


def test_point_connection_modes_do_not_claim_to_supply_curvature():
    sampler = SpacetimeSampler(MinkowskiMetric(), SPACING, method="finite4")
    with pytest.raises(NotImplementedError, match="only supports"):
        sampler.fields_at((0., 0., 0., 0.), {"riemann"})
    with pytest.raises(NotImplementedError, match="unavailable"):
        from tensor_toolkit.metrics import AlcubierreMetric
        SpacetimeSampler(AlcubierreMetric(), SPACING, method="analytic").connection_at(
            (0., 1., 0., 0.)
        )


def test_saved_grid_interpolates_retained_fields_with_contract_checks():
    experiment = replace(builtins()["minkowski"],
                         outputs=frozenset({"metric", "inverse_metric", "christoffel"}))
    result = run_experiment(experiment)
    sampler = SpacetimeSampler(experiment.metric, SPACING, method="cached_grid",
                               grid_result=result)
    event = (.13, .22, -.11, .32)
    fields = sampler.fields_at(event, {"metric", "inverse_metric", "christoffel"})
    assert np.allclose(fields["metric"], np.diag([-1., 1., 1., 1.]))
    assert np.allclose(fields["inverse_metric"], fields["metric"])
    assert np.allclose(fields["christoffel"], 0.)
    assert sampler.cache_info()["events"] == 1
    assert sampler.strategy_info()["field_interpolation"].startswith("multilinear")
    with pytest.raises(ValueError, match="outside"):
        sampler.metric_at((0., 3., 0., 0.))
    with pytest.raises(KeyError, match="no retained field"):
        sampler.fields_at(event, {"stress_energy"})


def test_grid_rejects_parameter_mismatch_and_stress_units():
    experiment = replace(builtins()["minkowski"],
                         outputs=frozenset({"metric", "stress_energy"}))
    result = run_experiment(experiment)
    altered = replace(experiment.metric, name="other")
    sampler = SpacetimeSampler(altered, SPACING, method="cached_grid", grid_result=result)
    with pytest.raises(ValueError, match="does not match"):
        sampler.metric_at((0., 0., 0., 0.))
    units = SpacetimeSampler(experiment.metric, SPACING, method="cached_grid",
                             grid_result=result, units="si")
    with pytest.raises(ValueError, match="units"):
        units.fields_at((0., 0., 0., 0.), {"stress_energy"})


def test_invalid_strategy_and_no_grid_are_rejected():
    with pytest.raises(ValueError, match="method"):
        SpacetimeSampler(MinkowskiMetric(), SPACING, method="unknown")
    with pytest.raises(ValueError, match="ExperimentResult"):
        SpacetimeSampler(MinkowskiMetric(), SPACING, method="cached_grid")


def test_saved_grid_roundtrip_can_be_sampled_after_reload(tmp_path):
    from tensor_toolkit.io import save_result, load_experiment_result
    experiment = replace(builtins()["minkowski"],
                         outputs=frozenset({"metric", "christoffel"}))
    original = run_experiment(experiment)
    save_result(original, tmp_path)
    loaded = load_experiment_result(tmp_path)
    sampler = SpacetimeSampler(experiment.metric, SPACING, method="cached_grid",
                               grid_result=loaded)
    assert np.allclose(sampler.metric_at((0., .3, .1, .2)),
                       np.diag([-1., 1., 1., 1.]))
    assert np.allclose(sampler.connection_at((0., .3, .1, .2)), 0.)


def test_reference_sampler_reuses_repeated_local_stencil():
    class CountingMinkowski:
        name = "Minkowski counting"
        coordinates = ("t", "x", "y", "z")
        def __init__(self):
            self.evaluations = 0
        def evaluate(self, grid):
            self.evaluations += 1
            return MinkowskiMetric().evaluate(grid)
    metric = CountingMinkowski()
    sampler = SpacetimeSampler(metric, SPACING, cache_size=2)
    event = (0., 0., 0., 0.)
    sampler.connection_at(event)
    initial_calls = metric.evaluations
    assert initial_calls == 1
    sampler.connection_at(event)
    assert metric.evaluations == initial_calls
    assert sampler.cache_info()["hits"] == 1
    sampler.connection_at((1., 0., 0., 0.))
    assert metric.evaluations == initial_calls + 1

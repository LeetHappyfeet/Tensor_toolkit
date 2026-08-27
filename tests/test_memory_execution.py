import numpy as np
from dataclasses import replace

from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.memory import estimate_peak_bytes
from tensor_toolkit.reference.geometry import (
    christoffel_symbols,
    ricci_from_christoffel,
    ricci_from_riemann,
    riemann_from_christoffel,
)
from tensor_toolkit.registry import configure_grid, get_experiment


def test_streamed_ricci_matches_full_riemann_contraction():
    experiment = configure_grid(get_experiment("de-sitter"), points=5, extent=1.0)
    metric = experiment.metric.evaluate(experiment.coordinates())
    gamma = christoffel_symbols(metric, experiment.spacings())
    streamed = ricci_from_christoffel(gamma, experiment.spacings())
    explicit = ricci_from_riemann(
        riemann_from_christoffel(gamma, experiment.spacings())
    )
    assert np.allclose(streamed, explicit, rtol=0.0, atol=1e-12)


def test_tiled_execution_matches_full_grid_including_edges():
    base = configure_grid(get_experiment("de-sitter"), points=7, extent=1.0)
    outputs = frozenset({"metric", "einstein", "stress_energy"})
    full = run_experiment(
        replace(base, outputs=outputs, memory_mode="in_memory")
    )
    tiled = run_experiment(
        replace(base, outputs=outputs, memory_mode="tiled", tile_points=3)
    )
    assert tiled.metadata["memory"]["halo"] == 3
    for name in outputs:
        assert np.allclose(tiled.fields[name], full.fields[name], rtol=0.0, atol=1e-12)


def test_only_requested_outputs_are_retained():
    base = configure_grid(get_experiment("minkowski"), points=3, extent=1.0)
    result = run_experiment(
        replace(base, outputs=frozenset({"stress_energy"}), memory_mode="in_memory")
    )
    assert set(result.fields) == {"stress_energy"}
    assert np.max(np.abs(result.fields["stress_energy"])) == 0.0


def test_tiled_peak_estimate_is_lower_for_large_grid():
    shape = (41, 41, 41, 41)
    outputs = frozenset({"einstein"})
    full = estimate_peak_bytes(shape, outputs, mode="in_memory", tile_points=8)
    tiled = estimate_peak_bytes(shape, outputs, mode="tiled", tile_points=8)
    assert tiled < full

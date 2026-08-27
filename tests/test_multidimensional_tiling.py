from dataclasses import dataclass, replace

import numpy as np

from tensor_toolkit.experiment import Axis, Experiment, _run_in_memory, _run_tiled
from tensor_toolkit.memory import block_working_bytes, choose_block_shape


@dataclass(frozen=True)
class SpaceTimeVaryingMetric:
    """Smooth nonsingular metric that varies in t and x for tiling regression tests."""

    name: str = "space-time varying test metric"
    coordinates: tuple[str, str, str, str] = ("t", "x", "y", "z")

    def evaluate(self, coordinate_grid):
        t, x, y, z = coordinate_grid
        shape = np.broadcast_shapes(*(np.shape(v) for v in coordinate_grid))
        g = np.zeros((4, 4, *shape), dtype=np.float64)
        g[0, 0] = -(1.0 + 0.02 * x**2 + 0.01 * t**2)
        g[1, 1] = 1.0 + 0.01 * t * x
        g[2, 2] = 1.0 + 0.005 * x**2
        g[3, 3] = 1.0
        return g


def _experiment(points=9):
    axes = tuple(Axis(-1.0, 1.0, points) for _ in range(4))
    return Experiment(
        SpaceTimeVaryingMetric(),
        axes,
        outputs=frozenset({"metric", "einstein", "stress_energy"}),
        memory_mode="tiled",
        tile_points=3,
    )


def test_multidimensional_blocks_match_full_grid():
    experiment = _experiment(9)
    axes = experiment.axis_values()
    spacings = tuple(float(v[1] - v[0]) for v in axes)

    full_fields, _ = _run_in_memory(experiment, axes, spacings)
    tiled_fields, _ = _run_tiled(
        experiment,
        axes,
        spacings,
        block_shape=(3, 3, 3, 3),
        halo=3,
    )

    for name in ("metric", "einstein", "stress_energy"):
        np.testing.assert_allclose(tiled_fields[name], full_fields[name], rtol=1e-12, atol=1e-12)


def test_block_planner_reduces_spatial_dimensions_to_fit_budget():
    shape = (65, 65, 65, 65)
    outputs = frozenset({"metric", "einstein", "stress_energy"})
    budget = 512 * 1024**2
    block = choose_block_shape(
        shape,
        outputs,
        working_budget_bytes=budget,
        tile_points=2,
        halo=3,
    )
    assert block[0] <= 2
    assert any(block[i] < shape[i] for i in (1, 2, 3))
    assert block_working_bytes(shape, outputs, block, halo=3) <= budget


def test_sparse_coordinate_grid_broadcasts_to_full_metric_shape():
    experiment = _experiment(7)
    coordinates = experiment.coordinates()
    assert [np.count_nonzero(np.array(v.shape) > 1) for v in coordinates] == [1, 1, 1, 1]
    metric = experiment.metric.evaluate(coordinates)
    assert metric.shape == (4, 4, 7, 7, 7, 7)

import pytest
from dataclasses import replace
from tensor_toolkit.alcubierre_study import wall_resolution, sigma_sweep
from tensor_toolkit.registry import get_experiment, configure_grid


def test_sharper_wall_requires_more_spatial_resolution():
    exp = configure_grid(get_experiment("alcubierre"), points=33, extent=2)
    soft = wall_resolution(replace(exp.metric, sigma=2), exp.axes[1:])
    sharp = wall_resolution(replace(exp.metric, sigma=10), exp.axes[1:])
    assert soft.width_10_90 > sharp.width_10_90
    assert soft.cells_across_wall > sharp.cells_across_wall


def test_sweep_rejects_nonpositive_sigma_without_running_solver():
    exp = get_experiment("alcubierre")
    with pytest.raises(ValueError):
        list(sigma_sweep(exp, [1, 0]))

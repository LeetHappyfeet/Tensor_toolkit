import numpy as np

from tensor_toolkit.cli import main
from tensor_toolkit.diagnostics import symmetry_error
from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.io import load_result, save_result
from tensor_toolkit.registry import configure_grid, get_experiment


def test_configure_grid_overrides_points_and_extent():
    experiment = configure_grid(
        get_experiment("alcubierre"), points=7, extent=3.0
    )
    assert all(axis.points == 7 for axis in experiment.axes)
    assert all((axis.start, axis.stop) == (-3.0, 3.0) for axis in experiment.axes)


def test_symmetry_error_detects_asymmetry():
    value = np.zeros((4, 4, 3, 3, 3, 3))
    value[0, 1] = 2.0
    diagnostics = symmetry_error(value)
    assert diagnostics["absolute"] == 2.0
    assert diagnostics["relative"] == 1.0


def test_saved_result_can_be_reloaded_and_inspected(tmp_path, capsys):
    result = run_experiment(get_experiment("minkowski"))
    save_result(result, tmp_path)
    metadata, fields, axes = load_result(tmp_path)
    assert metadata["metric_name"] == "Minkowski"
    assert "einstein" in fields
    assert len(axes) == 4

    assert main([
        "inspect", str(tmp_path), "--field", "einstein", "--center"
    ]) == 0
    output = capsys.readouterr().out
    assert "Stored validation status: PASS" in output
    assert "center index=" in output


def test_run_configurable_resolution(capsys):
    assert main([
        "run", "minkowski", "--points", "5", "--extent", "2"
    ]) == 0
    output = capsys.readouterr().out
    assert "grid=5x5x5x5" in output
    assert "status: PASS" in output


def test_convergence_command(capsys):
    assert main([
        "convergence", "minkowski", "--points", "3", "5"
    ]) == 0
    output = capsys.readouterr().out
    assert "Convergence study: minkowski" in output
    assert "symmetry_rel" in output

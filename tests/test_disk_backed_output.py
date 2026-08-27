from dataclasses import replace

import numpy as np

from tensor_toolkit.diagnostics import field_diagnostics
from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.io import load_result, save_result
from tensor_toolkit.registry import configure_grid, get_experiment


def _experiment():
    base = get_experiment("de-sitter")
    base = configure_grid(base, points=5, extent=1.0)
    return replace(
        base,
        outputs=frozenset({"einstein", "stress_energy"}),
        memory_mode="tiled",
        tile_points=2,
    )


def test_disk_backed_matches_memory_and_loads_as_memmap(tmp_path):
    experiment = _experiment()
    memory = run_experiment(experiment, storage_mode="memory")

    target = tmp_path / "disk_result"
    disk = run_experiment(
        experiment,
        output_path=target,
        storage_mode="disk",
    )
    assert disk.metadata["storage"]["mode"] == "disk"
    assert disk.metadata["memory"]["disk_backed"] is True
    assert isinstance(disk.fields["einstein"], np.memmap)
    np.testing.assert_allclose(
        disk.fields["einstein"], memory.fields["einstein"], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        disk.fields["stress_energy"], memory.fields["stress_energy"], rtol=0.0, atol=1e-12
    )

    save_result(disk, target)
    metadata, fields, axes = load_result(target)
    assert metadata["storage"]["format"] == "npy-memmap"
    assert len(axes) == 4
    assert isinstance(fields["einstein"], np.memmap)
    assert fields["einstein"].shape == (4, 4, 5, 5, 5, 5)


def test_memmap_diagnostics_do_not_require_full_residual(tmp_path):
    path = tmp_path / "field.npy"
    value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(4, 4, 9, 3, 3, 3))
    value[:] = 0.0
    value[0, 1] = 2.0
    value[1, 0] = 1.5
    value.flush()

    reopened = np.load(path, mmap_mode="r", allow_pickle=False)
    diagnostics = field_diagnostics(reopened, chunk_points=2)
    assert diagnostics["finite"] is True
    assert diagnostics["max_abs"] == 2.0
    assert diagnostics["symmetry"]["absolute"] == 0.5
    assert diagnostics["symmetry"]["relative"] == 0.25

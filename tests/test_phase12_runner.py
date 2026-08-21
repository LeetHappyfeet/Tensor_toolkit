import numpy as np
import pytest

from tensor_toolkit.backends import require_backend
from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.registry import get_experiment


def test_minkowski_pipeline_is_vacuum():
    result = run_experiment(get_experiment("minkowski"))
    assert np.max(np.abs(result.fields["einstein"])) < 1e-12
    assert np.max(np.abs(result.fields["stress_energy"])) < 1e-12


def test_gpu_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        require_backend("gpu")


def test_result_records_cpu_backend():
    result = run_experiment(get_experiment("minkowski"))
    assert result.metadata["backend"] == "cpu"

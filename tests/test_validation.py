import numpy as np
import pytest

from tensor_toolkit.validation import validate_metric


def test_metric_validation_rejects_float32():
    g = np.zeros((4, 4, 3, 3, 3, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        validate_metric(g)


def test_metric_validation_rejects_asymmetry():
    g = np.zeros((4, 4, 3, 3, 3, 3), dtype=np.float64)
    g[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        validate_metric(g)

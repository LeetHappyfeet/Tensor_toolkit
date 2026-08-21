import numpy as np

from tensor_toolkit.experiment import Axis, Experiment
from tensor_toolkit.metrics import DeSitterFlatMetric, MinkowskiMetric
from tensor_toolkit.visualization import (
    center_matrix,
    editable_metric_parameters,
    extract_2d_slice,
    replace_metric_parameters,
)


def test_extract_2d_slice_orders_vertical_horizontal():
    field = np.zeros((4, 4, 3, 4, 5, 6), dtype=np.float64)
    for x in range(4):
        for y in range(5):
            field[0, 0, :, x, y, :] = 10 * y + x
    sliced = extract_2d_slice(
        field,
        component=(0, 0),
        horizontal_axis=1,
        vertical_axis=2,
        fixed_indices={0: 1, 3: 2},
    )
    assert sliced.shape == (5, 4)
    assert sliced[3, 2] == 32


def test_center_matrix_uses_each_grid_center():
    field = np.zeros((4, 4, 3, 5, 7, 9), dtype=np.float64)
    field[:, :, 1, 2, 3, 4] = np.eye(4)
    np.testing.assert_array_equal(center_matrix(field), np.eye(4))


def test_metric_parameter_helpers():
    axes = tuple(Axis(-1, 1, 3) for _ in range(4))
    minkowski = Experiment(MinkowskiMetric(), axes)
    assert editable_metric_parameters(minkowski.metric) == {}

    de_sitter = Experiment(DeSitterFlatMetric(hubble=0.1), axes)
    changed = replace_metric_parameters(de_sitter, {"hubble": 0.25})
    assert changed.metric.hubble == 0.25

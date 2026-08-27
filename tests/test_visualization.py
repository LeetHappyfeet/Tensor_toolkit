import numpy as np

from tensor_toolkit.experiment import Axis, Experiment
from tensor_toolkit.metrics import DeSitterFlatMetric, MinkowskiMetric
from tensor_toolkit.visualization import (
    center_matrix,
    component_choice,
    component_index,
    editable_metric_parameters,
    extract_2d_slice,
    replace_metric_parameters,
    tensor_component_label,
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


def test_coordinate_component_labels_round_trip():
    assert component_choice(0) == "t (0)"
    assert component_choice(3) == "z (3)"
    assert component_index("x (1)") == 1
    assert component_index("y") == 2
    assert tensor_component_label("stress_energy", 0, 1) == "T_tx [0,1]"
    assert tensor_component_label("einstein", 3, 3) == "G_zz [3,3]"

import numpy as np

from tensor_toolkit.experiment import Axis, Experiment, run_experiment
from tensor_toolkit.metrics import DeSitterFlatMetric


def _residual(n):
    H = 0.08
    axes = (
        Axis(-1.0, 1.0, n),
        Axis(-0.2, 0.2, 5),
        Axis(-0.2, 0.2, 5),
        Axis(-0.2, 0.2, 5),
    )
    result = run_experiment(Experiment(
        metric=DeSitterFlatMetric(hubble=H),
        axes=axes,
        outputs=frozenset({"metric", "ricci", "ricci_scalar", "einstein"}),
    ))
    g = result.fields["metric"]
    ricci = result.fields["ricci"]
    scalar = result.fields["ricci_scalar"]
    einstein = result.fields["einstein"]

    core_tensor = (slice(None), slice(None), slice(2, -2), slice(1, -1), slice(1, -1), slice(1, -1))
    core_scalar = (slice(2, -2), slice(1, -1), slice(1, -1), slice(1, -1))
    ricci_error = np.max(np.abs((ricci - 3.0 * H**2 * g)[core_tensor]))
    scalar_error = np.max(np.abs((scalar - 12.0 * H**2)[core_scalar]))
    einstein_error = np.max(np.abs((einstein + 3.0 * H**2 * g)[core_tensor]))
    return max(ricci_error, scalar_error, einstein_error)


def test_de_sitter_nonzero_curvature_converges_to_analytic_solution():
    coarse = _residual(7)
    fine = _residual(11)
    assert fine < coarse

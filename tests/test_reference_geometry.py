import numpy as np

from tensor_toolkit.reference import christoffel_symbols, einstein_tensor, inverse_metric, riemann_tensor


def minkowski(shape=(5, 5, 5, 5)):
    g = np.zeros((4, 4, *shape), dtype=np.float64)
    g[0, 0] = -1.0
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    return g


def test_minkowski_pipeline_is_exactly_flat():
    g = minkowski()
    dx = (0.2, 0.2, 0.2, 0.2)
    gu = inverse_metric(g)
    assert np.array_equal(gu, g)
    assert np.count_nonzero(christoffel_symbols(g, dx)) == 0
    assert np.count_nonzero(riemann_tensor(g, dx)) == 0
    assert np.count_nonzero(einstein_tensor(g, dx)) == 0


def _rindler_residual(n):
    t = np.linspace(-0.3, 0.3, n)
    x = np.linspace(1.0, 2.0, n)
    y = np.linspace(-0.3, 0.3, n)
    z = np.linspace(-0.3, 0.3, n)
    shape = (n, n, n, n)
    g = np.zeros((4, 4, *shape), dtype=np.float64)
    x4 = x.reshape(1, n, 1, 1)
    g[0, 0] = -(x4**2)
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    dx = (t[1] - t[0], x[1] - x[0], y[1] - y[0], z[1] - z[0])

    gamma = christoffel_symbols(g, dx)
    interior_gamma = (slice(None),) * 3 + (slice(2, -2),) * 4
    assert np.max(np.abs(gamma[interior_gamma])) > 0.1

    riemann = riemann_tensor(g, dx)
    core = (slice(None),) * 4 + (slice(2, -2),) * 4
    return np.max(np.abs(riemann[core]))


def test_rindler_curvature_residual_decreases_under_refinement():
    coarse = _rindler_residual(7)
    fine = _rindler_residual(11)
    assert fine < coarse

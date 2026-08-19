import json
from pathlib import Path

import numpy as np

from tensor_toolkit.reference import einstein_tensor, ricci_tensor, riemann_tensor


REFERENCE = Path(__file__).parent / "reference_data" / "schwarzschild.json"


def _reference():
    return json.loads(REFERENCE.read_text())


def _schwarzschild(n):
    ref = _reference()
    mass = ref["parameters"]["M"]
    domain = ref["domain"]

    # Three points are sufficient in the ignorable t and phi directions.
    t = np.linspace(-0.2, 0.2, 3)
    r = np.linspace(domain["r_min"], domain["r_max"], n)
    theta = np.linspace(domain["theta_min"], domain["theta_max"], n)
    phi = np.linspace(-0.2, 0.2, 3)

    shape = (len(t), len(r), len(theta), len(phi))
    g = np.zeros((4, 4, *shape), dtype=np.float64)
    rr = r.reshape(1, n, 1, 1)
    tt = theta.reshape(1, 1, n, 1)
    f = 1.0 - 2.0 * mass / rr

    g[0, 0] = -f
    g[1, 1] = 1.0 / f
    g[2, 2] = rr**2
    g[3, 3] = rr**2 * np.sin(tt) ** 2

    spacings = (t[1] - t[0], r[1] - r[0], theta[1] - theta[0], phi[1] - phi[0])
    return ref, r, theta, g, spacings


def _center(n):
    return (1, n // 2, n // 2, 1)


def test_schwarzschild_has_nonzero_riemann_curvature_with_expected_component():
    n = 11
    ref, r, _, g, dx = _schwarzschild(n)
    riemann = riemann_tensor(g, dx)
    point = _center(n)
    radius = r[n // 2]
    mass = ref["parameters"]["M"]

    # For the documented convention: R^r_{theta r theta} = -M/r.
    numerical = riemann[(1, 2, 1, 2) + point]
    expected = -mass / radius
    assert numerical != 0.0
    assert np.isclose(numerical, expected, rtol=0.03, atol=0.0)


def _vacuum_residual(n):
    _, _, _, g, dx = _schwarzschild(n)
    point = _center(n)
    ricci = ricci_tensor(g, dx)
    einstein = einstein_tensor(g, dx)
    return max(
        np.max(np.abs(ricci[(slice(None), slice(None)) + point])),
        np.max(np.abs(einstein[(slice(None), slice(None)) + point])),
    )


def test_schwarzschild_vacuum_residual_decreases_under_refinement():
    # Exact Schwarzschild exterior has R_mn = G_mn = 0. The finite-difference
    # reference solver should approach that identity as the grid is refined.
    coarse = _vacuum_residual(7)
    medium = _vacuum_residual(9)
    fine = _vacuum_residual(11)
    assert medium < coarse
    assert fine < medium

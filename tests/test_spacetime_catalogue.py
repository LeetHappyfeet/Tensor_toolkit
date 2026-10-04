"""Independent regression tests for paper-backed and supplemental metrics."""
import numpy as np
import pytest

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT as G, SPEED_OF_LIGHT as C
from tensor_toolkit.metrics import (
    MinkowskiMetric, DeSitterFlatMetric, KerrBoyerLindquistMetric,
    KerrSchildMetric, ReissnerNordstromMetric, FLRWMetric,
    LinearizedPlaneWaveMetric, WeakField1PNMetric, SchwarzschildIsotropicMetric,
)
from tensor_toolkit.registry import builtins, gui_builtins, configure_grid
from tensor_toolkit.spacetime import describe_spacetime


M = C**2/G  # Schwarzschild length GM/c^2 = 1 metre.


def point(metric, *event):
    assert len(event) == 4
    return np.asarray(metric.evaluate(tuple(np.array([x]) for x in event)))[:, :, 0]


@pytest.mark.parametrize("name", sorted(builtins()))
def test_registered_metric_exterior_shapes_and_lorentzian_signature(name):
    experiment = builtins()[name]
    definition = describe_spacetime(experiment.metric)
    assert definition.chart.names == experiment.metric.coordinates
    g = np.asarray(experiment.metric.evaluate(experiment.coordinates()))
    assert g.shape == (4, 4, *map(lambda a: a.points, experiment.axes))
    assert np.all(np.isfinite(g))
    assert np.allclose(g, np.swapaxes(g, 0, 1))
    eig = np.linalg.eigvalsh(np.moveaxis(g, (0, 1), (-2, -1)))
    assert np.all(np.sum(eig < 0, axis=-1) == 1)
    assert np.all(np.sum(eig > 0, axis=-1) == 3)


def test_kerr_bl_spin_zero_reduces_to_schwarzschild_areal_chart():
    g = point(KerrBoyerLindquistMetric(M, spin=0), 0, 10, 1.1, 0)
    assert np.isclose(g[0, 0], -(1-2/10))
    assert np.isclose(g[1, 1], 1/(1-2/10))
    assert np.isclose(g[2, 2], 100)
    assert np.isclose(g[3, 3], 100*np.sin(1.1)**2)
    assert np.isclose(g[0, 3], 0)


def test_kerr_ks_spin_zero_recovers_schwarzschild_ks_not_isotropic_chart():
    g = point(KerrSchildMetric(M, spin=0), 0, 10, 0, 0)
    assert np.isclose(g[0, 0], -(1-2/10))
    assert np.isclose(g[0, 1], 2/10)  # ingoing Kerr-Schild off-diagonal term
    assert np.isclose(g[1, 1], 1+2/10)


def test_rn_charge_zero_reduces_to_schwarzschild_areal_chart():
    g = point(ReissnerNordstromMetric(M, 0), 0, 10, 1.1, 0)
    assert np.isclose(g[0, 0], -(1-2/10))
    assert np.isclose(g[1, 1], 1/(1-2/10))
    charged = point(ReissnerNordstromMetric(M, 2e16), 0, 10, 1.1, 0)
    assert charged[0, 0] < g[0, 0]  # charge makes f larger


def test_flrw_zero_curvature_matches_de_sitter_flat_slicing():
    grid = tuple(np.array([v]) for v in (0.4, 0.2, -0.1, 0.5))
    assert np.allclose(FLRWMetric(hubble=0.1).evaluate(grid),
                       DeSitterFlatMetric(hubble=0.1).evaluate(grid))


def test_zero_amplitude_wave_recovers_minkowski_and_nonzero_wave_is_symmetric():
    grid = tuple(np.array([v]) for v in (0.4, 0.2, -0.1, 0.5))
    assert np.array_equal(LinearizedPlaneWaveMetric(plus=0, cross=0).evaluate(grid),
                          MinkowskiMetric().evaluate(grid))
    wave = point(LinearizedPlaneWaveMetric(plus=0.03, cross=0.02), 0, 0, 0, 0)
    assert wave[1, 1] == pytest.approx(1.03)
    assert wave[2, 2] == pytest.approx(0.97)
    assert wave[1, 2] == pytest.approx(0.02)
    assert wave[2, 1] == pytest.approx(0.02)


def test_weak_field_1pn_agrees_with_isotropic_schwarzschild_to_omitted_order():
    r = 200.0  # epsilon=0.005
    exact = point(SchwarzschildIsotropicMetric(M), 0, r, 0, 0)
    approx = point(WeakField1PNMetric(M), 0, r, 0, 0)
    assert abs(exact[0, 0]-approx[0, 0]) < 5e-7
    assert abs(exact[1, 1]-approx[1, 1]) < 5e-5


def test_chart_domain_guards_prevent_polar_and_horizon_stencils():
    with pytest.raises(ValueError):
        point(KerrBoyerLindquistMetric(M), 0, 5, 0, 0)
    with pytest.raises(ValueError):
        point(KerrSchildMetric(M), 0, 0, 0, 0)
    with pytest.raises(ValueError):
        point(ReissnerNordstromMetric(M), 0, 1, 1, 0)
    with pytest.raises(ValueError):
        point(WeakField1PNMetric(M), 0, 1, 0, 0)
    with pytest.raises(ValueError):
        point(FLRWMetric(curvature=-1), 0, 3, 0, 0)


def test_gui_never_lists_spherical_or_offset_compact_object_grids():
    names = set(gui_builtins())
    assert {"minkowski", "de-sitter", "alcubierre"}.issubset(names)
    assert names.isdisjoint({"kerr-bl", "kerr", "reissner-nordstrom", "weak-field-1pn", "schwarzschild"})
    with pytest.raises(ValueError, match="symmetric --extent"):
        configure_grid(builtins()["kerr-bl"], extent=2)

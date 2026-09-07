from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.relativity import SpacetimeSampler, integrate_geodesic


def test_phase4_debug_flag_emits_sampler_and_geodesic_trace(capsys):
    sampler = SpacetimeSampler(
        MinkowskiMetric(),
        (0.1, 0.1, 0.1, 0.1),
        debug=True,
    )
    integrate_geodesic(
        sampler,
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        affine_step=0.1,
        steps=2,
        debug=True,
        debug_every=1,
    )
    captured = capsys.readouterr()
    assert "[tensor-toolkit:sampler]" in captured.err
    assert "[tensor-toolkit:geodesic] start" in captured.err
    assert "[tensor-toolkit:geodesic] step" in captured.err
    assert "[tensor-toolkit:geodesic] done" in captured.err

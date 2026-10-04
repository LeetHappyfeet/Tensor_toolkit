# Numerical reliability: patches 5 and 8

These APIs operate below VTK, on analytic metrics, completed numerical experiment
results and trajectories. The visualizer remains a viewer, not a curvature engine.

## Patch 5: grid diagnostics and actual convergence

Every `run_experiment` result now includes:

- `metadata["metric_configuration"]`: exact metric-instance representation,
  in addition to the spacetime coordinate-chart definition.
- `metadata["resolution"]`: axis sizes, maximum spacings, and heuristic
  warnings. Alcubierre reports its *radial shape function* 90%-to-10%
  transition width, the corresponding cells per spatial axis, an indicative
  six-cell grid size, time-frame displacement and potential boundary clipping.
- `metadata["field_statistics"]`: streaming extrema, RMS, finite and nonfinite
  counts, and symmetry residuals for retained outputs.
- `metadata["derivative_provenance"]`: second-order NumPy gradients in the
  reference 4-D pipeline, including repeated differentiation to curvature.

A wall-width estimate is a **resolution heuristic**, not a convergence proof.
The initial CLI `convergence` command previously compared Einstein tensor
symmetry alone; it now also compares Einstein values at matching coarse-grid
nodes. Use *nested* resolutions with the same domain, e.g.:

```bat
tensor-toolkit convergence alcubierre --points 5 9 17 --extent 2
tensor-toolkit convergence minkowski --points 5 9 17
```

The 3-grid case reports observed order from the two differences restricted to
the *same coarse-grid nodes*. A zero-error (e.g. exactly flat Minkowski) field
has no meaningful observed-order estimate. Choose grids sufficiently small
for available memory: all GR fields use 4-dimensional finite differences.

Python:

```python
from dataclasses import replace
from tensor_toolkit.registry import get_experiment, configure_grid
from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.convergence import compare_nested_results, compare_three_resolutions

base = replace(get_experiment("alcubierre"), outputs=frozenset({"einstein"}))
results = [run_experiment(configure_grid(base, points=n, extent=2))
           for n in (5, 9, 17)]
print(results[1].metadata["resolution"])
print(compare_nested_results(results[0], results[1], fields={"einstein"}))
print(compare_three_resolutions(*results, field="einstein"))
```

Comparisons refuse mismatched metric configurations, chart descriptions, units,
physical domains and non-nested coordinate nodes. They compare raw physical
components without smoothing or interpolating data.

For opt-in spatial symmetry checks, `spatial_reflection_report(result, "stress_energy",
axis=2)` checks Alcubierre's transverse y-reflection. The rank-two tensor
indices receive the appropriate reflection parity; scalar curvature is even.
The current justified rules cover y/z for Alcubierre and centered Cartesian
spatial axes for Minkowski, not arbitrary charts or spacetimes.

## Patch 8: explicit geometry sampling strategies

`SpacetimeSampler` retains its default historical second-order numerical
reference (a local `3x3x3x3` grid for derived fields). It now supports:

| `method` | Capability | Limitation |
|---|---|---|
| `numerical` | Full supported reference GR field pipeline | Rebuilds a local 3^4 stencil for an uncached event; refine spacing |
| `analytic` | Exact point metric/inverse and analytic Christoffel for Minkowski and *isotropic* Schwarzschild | Curvature requires the numerical path; other analytic Gamma metrics not implemented |
| `finite4` | Five-point fourth-order metric derivatives and point Christoffel | Not a fourth-order Riemann/Ricci solver; stencil must stay inside the chart |
| `cached_grid` | Four-dimensional multilinear interpolation of *retained* fields of a completed compatible `ExperimentResult` | No extrapolation; interpolated curvature is approximate, not recalculated |

All four modes support an optional bounded per-instance event/field cache
(`cache_size=32` by default, 0 disables it). A repeated request returns
a defensive copy. The cache holds events for the specific metric instance,
units, requested field set and numerical strategy. Call `clear_cache()` if
changing any externally supplied grid arrays in place; normal use should
treat completed results as immutable.

```python
from tensor_toolkit.constants import SPEED_OF_LIGHT as c, GRAVITATIONAL_CONSTANT as G
from tensor_toolkit.metrics import SchwarzschildIsotropicMetric
from tensor_toolkit.relativity import SpacetimeSampler

metric = SchwarzschildIsotropicMetric(c*c/G)
event = [0.0, 8.0, 1.0, 0.0]  # (ct,x,y,z), metres
sampler = SpacetimeSampler(metric, (.01,)*4, method="analytic")
gamma = sampler.connection_at(event)
print(sampler.strategy_info(), sampler.cache_info())
```

For interpolation, create an experiment that retains the requested fields,
then pass its completed result with `method="cached_grid", grid_result=result`.
A result written with `save_result(result, directory)` can be rehydrated with
`tensor_toolkit.io.load_experiment_result(directory)` and passed directly as
`grid_result` (including read-only memmaps for disk-backed saves).
The sampler rejects an incompatible metric name, parameter representation,
coordinate chart, out-of-domain event or mismatched stress-energy units.
An analytic or fourth-order *connection-only* request for Riemann/Ricci raises
`NotImplementedError` rather than silently mixing accuracy claims.

## Limitations and next steps

- No general fourth-order *curvature* path is claimed: repeated first
  derivatives require wider, correctly haloed stencils and boundary testing.
- No arbitrary-chart metric-specific analytic derivatives except Minkowski
  and isotropic Schwarzschild in this patch.
- The 4-D cached-grid backend is interpolation, not a replacement for resolving
  a physically sharp stress-energy wall.
- Follow-up: validate field convergence against exact references, benchmark
  event-sampling costs, and introduce specialized derivatives when justified.

Tests: `python -m pytest tests/test_convergence_patch5.py tests/test_geometry_sampling_patch8.py -v`.

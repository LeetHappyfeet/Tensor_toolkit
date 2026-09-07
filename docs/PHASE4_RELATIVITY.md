# Phase 4: relativistic observation and propagation

Phase 4 turns Tensor Toolkit's tensor and worldline machinery into a physical
observer/propagation analysis layer. It remains downstream of the validated
metric/tensor reference path: it does not evolve spacetime and is not numerical
relativity.

## Architecture

The central boundary is SpacetimeSampler. Analytic metrics currently provide
the geometry. The same interface is intended to support interpolated numerical
spacetimes later.

    Metric
      |
      v
    SpacetimeSampler
      |-- metric at event
      |-- connection at event
      |-- Riemann at event
      |-- fields along Worldline
      |
      +--> ObserverFrame
      |      |-- local vector/tensor measurements
      |      |-- stress-energy seen by observer
      |      +-- tidal/Weyl projections
      |
      +--> Geodesic solver
      |      |-- timelike
      |      +-- null
      |
      +--> signal observables
      +--> scientific ray bundles
      +--> transport

## Observer frames

ObserverFrame stores an orthonormal tetrad e_(a)^mu at one event. The comoving
constructor normalizes an arbitrary timelike four-velocity and uses metric
Gram-Schmidt to construct three spacelike axes.

Static coordinate observers are supported where the coordinate-time direction
is timelike.

Rank-2 covariant tensors can be projected into the local tetrad. Stress-energy
measurements expose local energy density, momentum density, and spatial stress.
These quantities are observer-dependent by construction.

## Curvature diagnostics

The Phase 4 curvature layer consumes the existing Riemann convention and
provides pointwise:

- Ricci scalar,
- Ricci contraction R_mn R^mn,
- Kretschmann scalar R_abcd R^abcd,
- fully covariant Weyl tensor,
- observer-frame electric Weyl curvature,
- observer-frame tidal tensor,
- tidal eigensystem,
- geodesic-deviation acceleration.

No second curvature implementation is introduced. Riemann data comes from the
reference geometry pipeline through SpacetimeSampler.

## Worldline sampling

SpacetimeSampler can evaluate requested tensor fields directly along a
Worldline. This is the preferred Phase 4 path because a Worldline can originate
from Newtonian dynamics, future 1PN dynamics, a geodesic solver, or another
trajectory generator.

The older trajectory-centered bridge remains available for compatibility.

## Geodesics

integrate_geodesic solves

    dx^mu/dlambda = k^mu

and

    dk^mu/dlambda = -Gamma^mu_ab k^a k^b

using a fixed-step RK4 reference integrator.

Timelike and null causal classes are explicit. Each result records the
normalization history g(k,k), normalization error, and maximum drift. This is
the relativistic analogue of the conservation diagnostics used by the
classical engine.

The first implementation intentionally prioritizes a transparent reference
solver over performance. Each connection evaluation currently uses a local
finite-difference stencil. A cached/interpolated connection field should be
added before production-scale ray tracing.

## Transport

Parallel transport is available for vectors carried along sampled worldlines.
Fermi-Walker transport is also exposed when the caller supplies a
proper-time-parameterized worldline and four-acceleration.

This establishes the machinery needed for nonrotating spacecraft frames and
transported observer tetrads.

## Signals and redshift

Photon frequency measured by an observer is evaluated from the invariant
contraction

    omega = -u_mu k^mu.

frequency_transfer compares emission and reception measurements of the same
null signal and returns the received/emitted frequency ratio and redshift.

Coordinate light-travel-time and radar-distance helpers are also provided.
Full emitter-to-receiver shooting and reflection solvers remain future work.

## Scientific ray tracing

Observer tetrads can construct future-directed null tangents from local spatial
directions. Individual rays and ray bundles can then be integrated through the
same geodesic solver.

The current ray layer is scientific geometry, not a renderer. It is intended to
feed the later VTK/PyVista visualizer with worldlines, ray bundles, event points,
tidal eigendirections, and scalar curvature fields.

Future optical expansion should add bundle Jacobians, expansion, shear,
magnification, caustics, and lens maps after null-geodesic validation is
established.

## Validation status

The exact flat-spacetime baseline covers:

- orthonormal observer tetrads,
- local tensor measurements,
- zero curvature and zero tides,
- straight timelike geodesics,
- straight null geodesics,
- null ray bundles,
- geodesic normalization preservation,
- equal-observer zero redshift,
- constant parallel transport.

Schwarzschild is now the Phase 4 curved-spacetime validation gate. The
Development branch includes analytic/numerical checks for:

- Kretschmann curvature,
- static gravitational redshift,
- radial timelike geodesics and Killing-energy conservation,
- circular timelike geodesics,
- weak-field light bending,
- the R=3m photon sphere,
- static-observer tidal eigenvalues.

The executable report is available through validate_schwarzschild_phase4().
Light bending can be skipped for fast iteration.

See docs/SCHWARZSCHILD_PHASE4_VALIDATION.md for the analytic formulas,
coordinate conversions, tolerances, and usage examples.

## Debugging

Phase 4 scientific paths have opt-in debug tracing. Normal calls remain silent.

SpacetimeSampler accepts debug=True. Geodesic, optics, and transport routines
accept debug=True, with debug_every controlling step cadence where appropriate.
The Schwarzschild validation report also accepts debug=True and reports each
reference comparison.

Messages are written to stderr with component-prefixed lines so debugging does
not alter saved scientific data.

## Phase boundary

Phase 4 analyzes prescribed spacetime geometry. It does not evolve the metric.
ADM/BSSN decomposition, constraint evolution, matter backreaction, gauge
conditions, and PDE evolution remain Phase 5.

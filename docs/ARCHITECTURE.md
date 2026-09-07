# Tensor Toolkit architecture

Tensor Toolkit is experimental scientific-computing software for classical and relativistic physics. The authoritative Development-branch implementation lives in the installable `tensor_toolkit` package. Legacy root-level, `Metrics/`, `solver/`, and `visualizer/` code remains for migration and historical comparison but is not the supported execution path.

## Major layers

```text
Metric models
    |
    v
Reference differential geometry
    |
    +--> Grid experiments --> memory/storage --> CLI / desktop visualizer
    |
    +--> SpacetimeSampler
             |
             +--> observer frames
             +--> curvature diagnostics
             +--> timelike/null geodesics
             +--> signal/redshift observables
             +--> ray bundles
             +--> vector transport

Classical systems
    |
    v
Dynamics models + integrators + events
    |
    v
Trajectory
    |
    v
Worldline
    |
    +----------------------> relativistic sampling and analysis
```

## Reference geometry path

The CPU/NumPy `float64` reference solver follows

```text
g_munu
  -> g^munu
  -> Gamma^rho_munu
  -> R^rho_sigma_mu_nu or streamed Ricci contraction
  -> R_munu
  -> R
  -> G_munu
  -> T_munu
```

The mathematical conventions are centralized in `tensor_toolkit.conventions` and documented in [CONVENTIONS.md](CONVENTIONS.md). The implementation is in `tensor_toolkit.reference.geometry`.

## Metric-grid experiments

`tensor_toolkit.experiment` defines the headless grid experiment boundary. A metric, four coordinate axes, requested retained fields, unit convention, backend choice, and memory policy become an `ExperimentResult`.

The CLI `tensor-toolkit run` and the current desktop visualizer both use this same execution path.

Registered CLI grid experiments currently include Minkowski, flat-slicing de Sitter, and Alcubierre. Schwarzschild exists as a supported metric class and is used programmatically, by validation, and by the trajectory bridge, but is not currently registered as a general `tensor-toolkit run schwarzschild` experiment.

## Classical simulation path

`tensor_toolkit.physics` separates evolving state, acceleration laws, numerical integration, event detection, diagnostics, and experiment persistence.

The core path is:

```text
System + DynamicsModel
    -> simulate()
    -> Trajectory
    -> Worldline
```

The direct Newtonian N-body implementation is the current reference dynamics model. RK4 supports future velocity-dependent dynamics; velocity-Verlet remains the supported symplectic path for velocity-independent acceleration laws.

## Relativistic analysis path

`tensor_toolkit.relativity` is downstream of prescribed spacetime geometry. `SpacetimeSampler` provides event-level metric, connection, and curvature access to observer frames, curvature invariants, tidal projections, geodesic integration, signal observables, null-ray tracing, and vector transport.

This layer analyzes spacetime; it does not evolve Einstein's equations.

## Worldlines as the shared boundary

`Worldline` is the common sampled spacetime-history container. Classical trajectories can be converted into worldlines, while relativistic geodesics naturally produce spacetime curves. This gives later analysis and visualization code a common representation without requiring all motion to originate from the same dynamics engine.

## Memory and storage

Large four-dimensional tensor grids are handled by a resource planner. The current implementation supports:

- full-grid in-memory execution,
- multidimensional `t x x x y x z` core blocking,
- three-cell halos,
- automatic spatial block reduction to fit a safe RAM budget,
- sparse/broadcast coordinate construction,
- selective field retention,
- streamed Ricci contraction when full Riemann output is unnecessary,
- direct-to-disk `.npy` memmaps,
- chunked diagnostics for large stored fields.

See [MEMORY_AND_STORAGE.md](MEMORY_AND_STORAGE.md).

## Visualization

The current Tkinter/Matplotlib interface is a downstream client of `run_experiment()`. It does not implement an independent GR solver. It supports rank-2 field slices, 4x4 overviews, center-point tensor inspection, validation diagnostics, resource planning, and reopening either NPZ or disk-backed results.

A future VTK/PyVista layer is intended for 3-D scalar fields, worldlines, ray bundles, tidal eigendirections, and other spatial scientific visualization. It should remain downstream of the same physics and storage interfaces.

## Validation boundary

Validation is part of the architecture rather than an optional presentation layer. Current gates include flat-spacetime tensor checks, curved reference metrics, classical conservation/orbital references, and Schwarzschild Phase 4 observables.

See [VALIDATION.md](VALIDATION.md).

## Current phase boundary

Tensor Toolkit does not yet implement a self-consistent numerical-relativity evolution system. ADM/BSSN-type initial data, gauge evolution, Hamiltonian/momentum constraints, PDE evolution, and matter backreaction remain Phase 5 work.

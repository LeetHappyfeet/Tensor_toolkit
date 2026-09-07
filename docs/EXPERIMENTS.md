# Headless experiments

Tensor Toolkit separates metric construction, tensor calculation, experiment configuration, storage, simulation, and visualization.

There are now two complementary experiment paths:

```text
Metric -> grid Experiment -> reference geometry engine -> ExperimentResult
```

and

```text
System + DynamicsModel -> simulation -> Trajectory -> Worldline
                                      -> relativistic sampling/analysis
```

Neither numerical engine depends on the desktop UI.

## Metric-grid experiment contract

A supported metric exposes a name, coordinate ordering, and an `evaluate()` method returning a NumPy `float64` covariant metric with shape

`(4, 4, N0, N1, N2, N3)`.

Current metric classes include Minkowski, flat-slicing de Sitter, Alcubierre, and Schwarzschild isotropic Cartesian.

The built-in `tensor-toolkit run` registry currently exposes:

- `minkowski`
- `de-sitter`
- `alcubierre`

Schwarzschild is currently used programmatically, by validation, and by the classical-trajectory bridge rather than as a general registered grid experiment.

An `Experiment` owns four coordinate axes, a metric model, requested outputs, stress-energy units, backend choice, and memory policy.

Requested retained outputs are selected from the supported tensor fields. Internal intermediates are discarded unless required or explicitly requested.

## Storage and memory

`run_experiment()` supports in-memory and multidimensional tiled execution. Persistent results can remain in memory/NPZ form or be written incrementally as disk-backed NumPy memmaps.

See [MEMORY_AND_STORAGE.md](MEMORY_AND_STORAGE.md).

## Classical simulation experiments

`tensor_toolkit.physics` provides a separate experiment layer for classical trajectories. A `SimulationExperiment` combines a system, duration, timestep, integrator, dynamics model, events, and metadata.

The built-in `demo-flyby` experiment provides a reproducible Jupiter/probe encounter:

```bash
tensor-toolkit simulate demo-flyby
```

Simulation results can be saved and reused for encounter diagnostics, worldline conversion, Schwarzschild sampling, and local GR tensor evaluation.

## Worldlines

`Worldline` is the shared spacetime-history representation between classical and relativistic layers. A worldline can originate from a Newtonian trajectory, a Schwarzschild bridge, a geodesic solver, or future dynamics implementations.

This is the preferred boundary for downstream relativistic observation, propagation, and visualization.

## Alcubierre research direction

The current Alcubierre metric remains a prescribed geometry, not a numerical evolution of Einstein's equations. Tensor Toolkit samples it over spacetime and infers the corresponding curvature and stress-energy fields.

That supports parameter studies while keeping numerical-relativity evolution as a separate future milestone.

## Phase boundary

The experiment system does not yet self-consistently evolve spacetime and matter together. ADM/BSSN-type initial data, constraint evolution, gauge evolution, and PDE integration remain Phase 5 work.

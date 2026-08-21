# Headless experiments

Tensor Toolkit now separates metric construction, tensor calculation, experiment configuration, and visualization.

The supported flow is:

`Metric -> Experiment -> reference geometry engine -> ExperimentResult -> visualization/storage`

The numerical engine has no UI dependency.

## Metric contract

A supported metric exposes a name, coordinate ordering, and an `evaluate()` method returning a NumPy float64 covariant metric with layout `(4, 4, Nt, Nx, Ny, Nz)`.

Initial models are Minkowski, flat-slicing de Sitter, and Alcubierre. The Alcubierre model uses geometrized units and a moving center `x_s(t) = x0 + v t`, so it supports genuine multi-time-slice evaluation instead of the legacy constructor's single-time-slice restriction.

## Experiment contract

An `Experiment` owns four coordinate axes, a metric model, requested outputs, and the stress-energy unit convention. `run_experiment()` evaluates only requested retained outputs. This is the first step toward an explicit retention policy: expensive intermediates such as the full Riemann tensor need not be kept when the user only wants Einstein or stress-energy fields.

The current implementation is intentionally simple and may recompute intermediates when several downstream outputs are requested. A later calculation-plan/cache layer should remove that duplication before large production runs.

## Alcubierre research direction

The current Alcubierre metric is a prescribed geometry, not a numerical evolution of Einstein's equations. Tensor Toolkit evaluates the geometry over `(t,x,y,z)` and infers the corresponding Einstein/stress-energy tensors. This supports parameter scans over velocity, radius, wall thickness, and trajectory while the evolution-code problem remains a separate future milestone.

The next analysis layer should derive observer-dependent quantities from `T_mu_nu`, including Eulerian energy density, momentum density, spatial stress, and energy-condition diagnostics. Those quantities should become the primary UI visualization targets rather than raw tensor components alone.

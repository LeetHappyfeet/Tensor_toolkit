# Metric Tensor Simulator

The desktop visualizer is a front end to the same CPU/NumPy reference pipeline used by `tensor-toolkit run`. It does not contain a second symbolic GR implementation.

## Launch

From an editable development checkout:

```text
git pull
python -m pip install -e .
tensor-toolkit visualize
```

The compatibility launcher also remains available:

```text
python visualizer.py
```

## Current workflow

1. Select a registered metric. Minkowski is the default baseline.
2. Set metric parameters, grid resolution, and uniform coordinate extent.
3. Run the simulation on the supported CPU/NumPy float64 backend.
4. Select a rank-2 field: covariant metric, inverse metric, Ricci tensor, Einstein tensor, or stress-energy tensor.
5. Select tensor indices mu and nu.
6. Select the two coordinate axes to plot. The remaining two coordinates are fixed by grid index.
7. Inspect the tensor matrix at the grid center and the numerical validation tab.
8. Save the run in the same `result.npz` + `metadata.json` format used by the CLI, or open a previously saved result.

## Architecture

The visualizer calls `tensor_toolkit.experiment.run_experiment`. It does not import or call the legacy `analyticalEnergyTensor.py` path. This keeps CLI runs, saved results, validation, and plots on one mathematical implementation.

The GUI requests the rank-2 fields that are naturally represented by a 4x4 component selector:

- `metric`
- `inverse_metric`
- `ricci`
- `einstein`
- `stress_energy`

Christoffel symbols and the Riemann tensor remain available to the solver but require dedicated higher-rank visualization controls and are intentionally not flattened into the current 4x4 viewer.

## Scope

This is currently a metric-field simulator, not a dynamical numerical-relativity evolution code. Time is one coordinate of the sampled metric field. The program does not yet evolve initial data forward with ADM/BSSN or couple moving matter back into spacetime.

A validation `PASS` means only that the implemented numerical checks passed. It is not a statement that a metric is physically realizable.

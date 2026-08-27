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
3. Choose which rank-2 outputs should be retained. The default is metric + Einstein + stress-energy.
4. Choose a memory mode (`auto`, `in_memory`, or `tiled`) and optional t-slab tile size.
5. Use **Estimate memory** to preview in-memory and tiled peak-RAM estimates. The same preflight runs automatically before every simulation.
6. Run the simulation on the supported CPU/NumPy float64 backend. In `auto` mode the solver switches to tiled execution when the in-memory estimate exceeds the safe RAM budget.
7. Select one of the retained rank-2 fields for visualization.
8. Select tensor indices mu and nu, or use the 4x4 overview to see all 16 components on the same slice.
9. Select the two coordinate axes to plot. The remaining two coordinates are fixed by grid index.
10. Inspect the tensor matrix at the grid center and the numerical validation tab.
11. Save the run in the same `result.npz` + `metadata.json` format used by the CLI, or open a previously saved result.

## Memory-aware execution

The GUI uses the same memory planner as the CLI. The preflight reports:

- selected execution mode
- estimated selected peak RAM
- estimated in-memory peak RAM
- estimated tiled peak RAM
- available physical RAM when detectable
- the configured safe RAM budget
- tile core size and three-cell halo when tiled mode is selected

If the requested calculation exceeds the safe budget even in tiled mode, the GUI stops before allocating the large tensors and reports a preflight error rather than allowing the operating system to kill the process.

Tiling currently decomposes the four-dimensional calculation into slabs along the t axis with a three-cell halo. Requested final output fields are still retained in RAM, so very large runs should select only the fields needed for analysis. Disk-backed output is a later extension.

## Architecture

The visualizer calls `tensor_toolkit.experiment.run_experiment`. It does not import or call the legacy `analyticalEnergyTensor.py` path. This keeps CLI runs, saved results, validation, memory planning, and plots on one mathematical implementation.

The GUI exposes the rank-2 fields that are naturally represented by a 4x4 component selector:

- `metric`
- `inverse_metric`
- `ricci`
- `einstein`
- `stress_energy`

Christoffel symbols and the Riemann tensor remain available to the solver but require dedicated higher-rank visualization controls and are intentionally not flattened into the current 4x4 viewer.

## Scope

This is currently a metric-field simulator, not a dynamical numerical-relativity evolution code. Time is one coordinate of the sampled metric field. The program does not yet evolve initial data forward with ADM/BSSN or couple moving matter back into spacetime.

A validation `PASS` means only that the implemented numerical checks passed. It is not a statement that a metric is physically realizable.

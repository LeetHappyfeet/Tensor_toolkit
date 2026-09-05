# Metric Tensor Simulator

The current desktop visualizer is a Tkinter/Matplotlib front end to the same CPU/NumPy reference experiment pipeline used by `tensor-toolkit run`. It does not contain a second GR implementation.

## Launch

From an editable Development checkout:

```text
python -m pip install -e .
tensor-toolkit visualize
```

The compatibility launcher remains available:

```text
python visualizer.py
```

## Current workflow

1. Select a registered metric.
2. Edit supported metric parameters.
3. Choose uniform grid resolution and extent.
4. Choose which rank-2 outputs should be retained.
5. Choose memory mode: `auto`, `in_memory`, or `tiled`.
6. Optionally provide a time-core hint with `tile_points`; spatial block sizes are chosen automatically when required.
7. Choose output storage: `auto`, `memory`, or `disk`.
8. Estimate resources before calculation.
9. Run the experiment on the authoritative CPU/NumPy `float64` backend.
10. Inspect retained rank-2 tensor fields by component and two-dimensional slice.
11. View the complete 4x4 field overview and center-point tensor.
12. Inspect stored validation diagnostics.
13. Save or reopen either conventional NPZ results or disk-backed memmapped results.

## Memory-aware execution

The GUI uses the same planner as the CLI.

The resource panel reports the selected execution strategy, four-dimensional core block shape, maximum halo-expanded block shape, block count, estimated peak RAM, persistent-output size, available RAM, and free disk space where applicable.

Tiled execution can decompose all four coordinates. `tile_points` is a time-core hint rather than a statement that only time is tiled.

See [MEMORY_AND_STORAGE.md](MEMORY_AND_STORAGE.md).

## Persistent results

For memory-backed runs, saved results use:

```text
result.npz
metadata.json
```

For disk-backed runs, the result directory contains:

```text
metadata.json
fields/
  <field>.npy
axes/
  axis_0.npy
  axis_1.npy
  axis_2.npy
  axis_3.npy
```

Large field arrays are reopened with NumPy memory mapping so selected slices can be inspected without loading the complete tensor field into RAM.

## Architecture

The visualizer calls `tensor_toolkit.experiment.run_experiment`. It does not call the legacy `analyticalEnergyTensor.py` path.

The current GUI is intentionally focused on rank-2 fields:

- `metric`
- `inverse_metric`
- `ricci`
- `einstein`
- `stress_energy`

Christoffel symbols and the full Riemann tensor require dedicated higher-rank visualization controls and are not flattened into the current 4x4 viewer.

## Current visualization boundary

The present interface is primarily a two-dimensional scientific inspection tool over four-dimensional stored fields. It is useful for component slices, matrix overviews, diagnostics, and resource-aware experiment control.

It is not yet the intended final visualization environment for worldlines, ray bundles, three-dimensional scalar fields, tidal eigendirections, or observer-local glyphs.

## Planned VTK/PyVista layer

A later visualization milestone should add a VTK/PyVista renderer downstream of the existing solver and storage APIs.

That layer is intended to support:

- 3-D scalar and tensor-derived fields,
- volume and slice rendering,
- worldlines and trajectories,
- null rays and ray bundles,
- event markers,
- observer frames,
- tidal eigendirection glyphs,
- curvature isosurfaces,
- time-dependent animation.

The VTK/PyVista renderer should consume existing `ExperimentResult`, disk-backed fields, `Trajectory`, `Worldline`, and Phase 4 analysis outputs rather than introducing a parallel physics implementation.

## Scope

The visualizer shows prescribed spacetime geometry and derived observables. Tensor Toolkit does not yet evolve ADM/BSSN initial data or self-consistently couple matter back into the metric.

A validation `PASS` means that the implemented numerical checks passed. It is not a claim that a prescribed metric is physically realizable.

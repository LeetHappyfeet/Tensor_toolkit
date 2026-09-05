# VTK Visualizer

Tensor Toolkit's primary desktop visualization path now uses VTK for 3-D rendering. The renderer is intentionally the final stage of the scientific pipeline: it displays already-computed NumPy data and does not implement metric, curvature, dynamics, observer, or ray-tracing physics.

## Installation and launch

From the Development branch:

```text
python -m pip install -e ".[visualization]"
tensor-toolkit visualize
```

The compatibility launcher remains:

```text
python visualizer.py
```

## Architecture

The boundary between physics and rendering is `tensor_toolkit.visualization_data`. It contains renderer-neutral data classes and adapters for:

- 3-D scalar volumes selected from `ExperimentResult`,
- classical `Trajectory` objects,
- relativistic `Worldline` objects,
- `RayBundle` null-ray collections,
- observer-frame spatial tetrads,
- tidal-tensor eigendirections, and
- trajectory event markers.

VTK code belongs downstream of those adapters. New rendering features should consume `VolumeData`, `PolylineData`, `GlyphData`, or `PointData` instead of importing solver internals.

## Initial VTK milestone

The current VTK GUI can:

- select a built-in metric and edit its supported parameters,
- choose grid resolution and extent,
- select retained GR fields,
- run the same validated CPU/NumPy experiment path as the CLI,
- open and save Tensor Toolkit result directories,
- select a rank-2 tensor component and coordinate-time index,
- render the resulting x-y-z scalar field as a volume or isosurface, and
- keep the experiment validation status visible beside the rendering controls.

The VTK scene is therefore a view of the authoritative result, not an alternate computation path.

## Next renderer work

The neutral adapters are already available for the next scene layers. The next implementation pass should add trajectory/worldline tubes, null-ray bundles, observer-frame and tidal glyphs, event markers, clipping planes, scalar bars, camera presets, and animation over the stored time axis. Those additions should not require changes to the solver.

Large disk-backed results should continue to be sliced at a selected time before VTK conversion so the GUI never materializes the complete 4-D field merely to display one 3-D frame.

## Scientific caution

A visually smooth or symmetric VTK scene is not evidence that the underlying metric or physical model is correct. The GUI exposes stored validation status for that reason. Tensor Toolkit still treats validation and convergence testing as upstream requirements, with visualization used only to help a human inspect the computed geometry and observables.

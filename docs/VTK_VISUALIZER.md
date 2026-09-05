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

## Current VTK milestone

The current VTK GUI can:

- select a built-in metric and edit its supported parameters,
- choose grid resolution and extent,
- select retained GR fields,
- run the same validated CPU/NumPy experiment path as the CLI,
- open and save Tensor Toolkit tensor result directories,
- open saved classical simulation directories containing `trajectory.npz`,
- render rank-2 tensor components as 3-D volumes or isosurfaces,
- play, pause, scrub, loop, and change visualization playback rate,
- keep tensor fields locked to the nearest authoritative stored time sample,
- smoothly interpolate displayed classical body positions through the public `Trajectory.sample()` API,
- draw past trajectory trails from stored samples,
- render simulation event markers and jump directly to event times,
- follow a selected body with the VTK camera,
- update VTK scalar arrays and transfer functions in place during playback, and
- keep the experiment validation status visible beside the rendering controls.

The master clock is implemented by `tensor_toolkit.visualization_timeline.VisualizationTimeline`. It maps wall-clock playback onto already-computed simulation/coordinate time; it never calls an integrator or generates new physics samples.

`FrameCache` keeps a small LRU window of nearby 3-D tensor frames. This is especially important for disk-backed results because only the displayed and neighboring time slices need to become active in memory.

The VTK scene is therefore a view of authoritative results plus explicitly visualization-only interpolation for moving objects, not an alternate computation path.

## Time semantics

Tensor fields remain discrete by default: at visualization time `t`, the GUI displays the nearest stored tensor frame. Classical object positions may move smoothly between accepted trajectory samples using the existing interpolation supplied by `Trajectory.sample()`.

This distinction is intentional. Smooth object motion is a rendering operation; it does not claim that a new physics step was solved between stored samples.

When both a tensor result and a trajectory are loaded, they share one master visualization clock. If their time domains differ, each layer is clamped to the part of its own stored interval that overlaps the current visualization time.

## Next renderer work

The neutral adapters are already available for worldlines, null-ray bundles, observer frames, and tidal tensors. The next implementation pass should add worldline/ray tubes, observer-frame and tidal glyphs, clipping planes, scalar bars, camera presets, proper-time overlays, and optional clearly labeled temporal interpolation of scalar field data.

Large disk-backed results should continue to be sliced at a selected time before VTK conversion so the GUI never materializes the complete 4-D field merely to display one 3-D frame.

## Scientific caution

A visually smooth or symmetric VTK scene is not evidence that the underlying metric or physical model is correct. The GUI exposes stored validation status for that reason. Tensor Toolkit still treats validation and convergence testing as upstream requirements, with visualization used only to help a human inspect the computed geometry and observables.

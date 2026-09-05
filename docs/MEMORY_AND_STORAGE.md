# Memory and storage architecture

Four-dimensional tensor grids grow rapidly: a uniform `N x N x N x N` domain contains `N^4` spacetime points before tensor components are counted. Tensor Toolkit therefore separates working-memory planning from persistent-output storage.

## Execution modes

The current CPU reference engine supports:

- `in_memory`: compute the requested grid as one full domain,
- `tiled`: decompose the four-dimensional grid into core blocks with halos,
- `auto`: choose full-grid or tiled execution from the estimated safe RAM budget.

The planner treats only a configurable fraction of currently available physical RAM as usable working memory.

## Multidimensional block decomposition

Tiled execution is not limited to the time axis.

`--tile-points` is retained as a time-core hint for compatibility, but the planner automatically reduces spatial `x`, `y`, and `z` core sizes when necessary.

Each core block is expanded by a three-cell halo where neighboring grid points exist:

```text
full 4-D grid
    |
    v
core block (t,x,y,z)
    + three-cell halo
    |
    v
metric -> inverse -> Christoffel -> curvature contraction
    |
    v
crop halo
    |
    +--> in-memory output
    `--> disk-backed output
```

The halo prevents artificial internal block boundaries from becoming numerical boundaries of the calculation.

## Working-memory estimate

The planner uses a deliberately conservative transient component allowance to cover metric inversion, derivatives, curvature work arrays, NumPy temporaries, allocator overhead, and optional full-Riemann output.

It reports:

- requested and selected memory modes,
- core block shape,
- maximum halo-expanded local shape,
- block count,
- estimated in-memory peak,
- estimated tiled peak,
- selected peak estimate,
- available RAM,
- safe RAM budget,
- working-memory budget,
- persistent output size.

## Selective field retention

Only requested outputs are persisted. Expensive intermediates are discarded unless explicitly requested.

When the full Riemann tensor is not requested, the reference path contracts curvature directly into the Ricci tensor rather than materializing a persistent 256-component Riemann field.

## Sparse coordinates

Metric construction uses sparse/broadcast coordinate arrays instead of creating four complete dense coordinate volumes. Dense metric and derivative arrays still exist inside each computational block.

## Storage modes

Persistent output can use:

- `memory`: retain arrays in RAM and save a conventional compressed `result.npz`,
- `disk`: allocate standard NumPy `.npy` memmaps and write block cores directly to disk,
- `auto`: choose based on requested output size, available RAM, and whether an output directory was supplied.

Disk-backed execution requires `auto` or `tiled` memory mode.

## Disk-backed layout

A completed disk-backed result directory contains:

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

These files are ordinary NumPy arrays. `tensor-toolkit inspect` and the desktop visualizer reopen large tensor fields with memory mapping so selected slices can be read without loading the complete result into RAM.

An incomplete marker is used while a disk-backed calculation is in progress. Incomplete results are rejected on load.

## Disk safety

Before disk-backed execution, Tensor Toolkit checks free space for the requested persistent arrays and reserves additional headroom. A run is rejected before expensive calculation if the target location cannot safely hold the expected output.

## CLI examples

Automatic planning:

```bash
tensor-toolkit run alcubierre --points 31
```

Force tiled execution:

```bash
tensor-toolkit run alcubierre --points 41 --memory-mode tiled --tile-points 6
```

Explicit disk-backed output:

```bash
tensor-toolkit run alcubierre \
  --points 65 \
  --fields einstein stress_energy \
  --memory-mode tiled \
  --tile-points 4 \
  --storage-mode disk \
  --output results/alcubierre65
```

## Practical limits

Blocking controls peak RAM; it does not change the total `N^4` amount of spacetime work. Very high resolutions can therefore remain CPU-expensive and can produce very large persistent fields.

Memory planning is a safety mechanism, not a performance substitute for future optimized or accelerated backends.

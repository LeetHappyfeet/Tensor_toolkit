# Disk-backed high-resolution output

Tensor Toolkit can persist large requested tensor fields directly to disk while a multidimensional CPU calculation is running. This prevents the final result arrays from needing to fit in physical RAM and allows the working GR calculation to be decomposed across `t × x × y × z` blocks.

## CLI

For an explicitly disk-backed run:

```bash
tensor-toolkit run alcubierre \
  --points 65 \
  --fields einstein stress_energy \
  --memory-mode tiled \
  --tile-points 4 \
  --storage-mode disk \
  --output results/alcubierre65
```

`--storage-mode auto` is the default. If an output directory is supplied, Tensor Toolkit may choose disk-backed storage when the requested persistent arrays are large relative to available RAM. Use `--storage-mode disk` when disk persistence is required explicitly.

`--tile-points` is retained as a time-core hint for compatibility. It no longer means that only time is tiled. The planner uses available RAM to choose safe spatial core sizes automatically for `x`, `y`, and `z` as well.

## Four-dimensional block decomposition

A large grid is divided into core blocks. Every block is extended by a three-cell halo in each tiled coordinate direction before derivatives are evaluated. Only the core is copied into the final result. Halo values are discarded.

Conceptually:

```text
full 4-D grid
    ↓
core block (t,x,y,z)
    + three-cell halo on each available side
    ↓
metric → inverse → Christoffel → streamed Ricci → Einstein / stress-energy
    ↓
crop halo
    ↓
write core directly to global memory or disk-backed field
```

The halo is required because curvature contains nested finite differences. Artificial block boundaries therefore do not become numerical boundaries of the spacetime calculation.

The memory planner reports:

- selected execution mode;
- chosen core block shape `t × x × y × z`;
- maximum local block shape including halos;
- number of blocks;
- estimated peak working RAM;
- persistent output size.

## Sparse coordinates

Metric sampling uses sparse/broadcast coordinate arrays. Tensor Toolkit no longer creates four complete dense copies of the `t`, `x`, `y`, and `z` coordinate volumes merely to evaluate a metric. The metric tensor itself and its required derivative intermediates are still dense inside each computational block.

## On-disk layout

A disk-backed result directory contains:

```text
metadata.json
fields/
  einstein.npy
  stress_energy.npy
axes/
  axis_0.npy
  axis_1.npy
  axis_2.npy
  axis_3.npy
```

The field files are standard NumPy `.npy` arrays created with `open_memmap`. They are portable NumPy files, not private Tensor Toolkit binary blobs.

`tensor-toolkit inspect` opens them using `mmap_mode="r"`, and the desktop visualizer can open the same result directory. Only pages needed for selected slices need to become resident in RAM.

## GUI

The metric tensor simulator exposes:

- **Memory mode**: `auto`, `in_memory`, or `tiled`
- **Time-core hint**: preferred maximum number of core time samples before automatic block planning
- **Output storage**: `auto`, `memory`, or `disk`
- **Result directory**: target directory for saved or disk-backed calculations
- **Outputs to retain**: limits persistent tensor fields

The resource panel displays the automatically chosen 4-D core block, maximum halo-expanded block, block count, peak RAM estimate, retained-output size, available RAM, and free disk space.

## Safety checks

Before allocating large arrays Tensor Toolkit checks:

1. estimated peak RAM for the selected execution/storage strategy;
2. available physical RAM and the configured safe fraction;
3. estimated persistent result size;
4. available free space at the disk result location, including 10% headroom.

Validation diagnostics are accumulated block by block. Opening or inspecting a memory-mapped tensor also uses chunked diagnostics so a symmetry check does not allocate a second full-size tensor residual.

## Practical limits

Multidimensional tiling removes the previous `tile_t × N³` RAM wall, but it does not make arbitrarily large simulations free. Increasing a uniform four-dimensional grid remains an `N⁴` problem in total grid points, disk output, and CPU work. For example, `101⁴` contains more than 104 million spacetime points before tensor components are counted.

The block planner controls peak RAM; it does not reduce the total amount of curvature computation. Very high resolutions can therefore take a long time on the current CPU reference backend and can generate tens or hundreds of gigabytes of output depending on retained fields.

# Disk-backed high-resolution output

Tensor Toolkit can persist large requested tensor fields directly to disk while a tiled CPU calculation is running. This prevents the final result arrays from needing to fit in physical RAM.

## CLI

For an explicitly disk-backed run:

```bash
tensor-toolkit run alcubierre \
  --points 41 \
  --fields einstein stress_energy \
  --memory-mode tiled \
  --tile-points 4 \
  --storage-mode disk \
  --output results/alcubierre41
```

`--storage-mode auto` is the default. If an output directory is supplied, Tensor Toolkit may choose disk-backed storage when the requested persistent arrays are large relative to available RAM. Use `--storage-mode disk` when disk persistence is required explicitly.

Disk-backed output requires `--output`. It is intentionally paired with tiled execution because an ordinary full-grid calculation would still construct full-grid intermediates in RAM.

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

`tensor-toolkit inspect` opens them using `mmap_mode="r"`, and the desktop visualizer can open the same result directory. Only pages needed for the selected slices are read into memory by the operating system.

## GUI

The metric tensor simulator exposes:

- **Memory mode**: `auto`, `in_memory`, or `tiled`
- **Output storage**: `auto`, `memory`, or `disk`
- **Disk result dir**: target directory for a disk-backed calculation
- **Tile t-points**: core time samples processed per slab

Choose a disk result directory before selecting explicit `disk` storage. The result is finalized in that directory automatically after the computation succeeds.

## Safety checks

Before allocating large arrays Tensor Toolkit checks:

1. estimated peak RAM for the selected execution/storage strategy;
2. available physical RAM and the configured safe fraction;
3. estimated persistent result size;
4. available free space at the disk result location, including 10% headroom.

Validation diagnostics are accumulated tile by tile. Opening or inspecting a memory-mapped tensor also uses chunked diagnostics so a symmetry check does not allocate a second full-size tensor residual.

## Remaining scaling limit

Disk-backed output removes persistent result arrays from the required resident-RAM budget. It does **not** eliminate the RAM required by one computational tile.

The current domain decomposition is along the time axis only. A tile still spans the complete `x × y × z` grid plus a three-cell time halo. For very large uniform `N^4` grids, the working set therefore still scales approximately with `tile_t × N^3`.

The next major scaling improvement, if needed, is multidimensional spatial tiling with halos in `x`, `y`, and `z` as well as `t`. The disk-backed field format introduced here is designed to support that later without changing saved-result compatibility.

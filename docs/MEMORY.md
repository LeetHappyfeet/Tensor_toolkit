# Memory-aware execution

Tensor Toolkit 0.2 uses float64 CPU arrays. Four-dimensional grids grow as `N^4`, so memory becomes the limiting resource quickly.

The reference solver now applies four safeguards:

1. **Stream Ricci when full Riemann is not requested.** The solver contracts curvature directly from Christoffel symbols into `R_mu_nu`, avoiding the persistent 256-component Riemann field. If `riemann` is explicitly requested, it is still computed and returned.
2. **Preflight RAM estimates.** A run estimates peak memory before allocating large tensor arrays. By default only 65% of currently available physical RAM is treated as safe working memory.
3. **Retain only requested outputs.** Internal tensors are computed as necessary and discarded unless they were requested as outputs. The CLI can override built-in outputs with `--fields`.
4. **Tiled execution.** Large jobs can run as slabs along the time axis. Each slab includes a three-cell halo. The halo is deliberately wider than the centered stencil alone requires because the existing `edge_order=2` global boundary stencil differentiates Christoffel symbols that themselves depend on metric derivatives. Three cells reproduce the full-grid result at the global edges as well as in the interior.

## CLI examples

Automatic memory selection:

```bash
tensor-toolkit run alcubierre --points 31
```

Force tiled mode:

```bash
tensor-toolkit run alcubierre --points 31 --memory-mode tiled --tile-points 6
```

Keep only the Einstein tensor:

```bash
tensor-toolkit run alcubierre --points 41 --fields einstein --memory-mode tiled
```

Requesting fewer persistent fields can reduce memory substantially. Full output arrays must still fit in RAM in 0.2; disk-backed output is a later step.

## Current tiling boundary

The first tiled implementation decomposes only the `t` axis. This is enough to reduce peak working memory significantly without changing the finite-difference mathematics. It is not yet multidimensional domain decomposition and it is not a substitute for disk-backed storage when the requested output arrays themselves exceed available RAM.

The `memory` object stored in experiment metadata records the chosen mode, tile size, halo width, estimated peak bytes, available bytes, and safe RAM budget.

# Supported reference pipeline

The `tensor_toolkit.reference` package is the authoritative CPU/NumPy `float64` geometry path. Legacy code under root-level modules, `Metrics/`, and `solver/` remains for migration and historical comparison but is not the supported solver.

## Representation and conventions

A numerical metric is a NumPy `float64` array with shape

`(4, 4, N0, N1, N2, N3)`.

The first two axes are covariant tensor indices. The remaining axes follow the metric's explicit coordinate ordering.

The reference signature is `(-+++)`.

The Riemann convention is

`R^rho_{ sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}`.

The Ricci contraction is

`R_{sigma nu} = R^rho_{ sigma rho nu}`.

The Einstein equation assumes zero cosmological constant:

`G_{mu nu} = (8 pi G / c^4) T_{mu nu}`

in SI units, or

`G_{mu nu} = 8 pi T_{mu nu}`

in geometrized units.

Physical constants live in `tensor_toolkit.constants`; conventions live in `tensor_toolkit.conventions`.

## Calculation path

The reference geometry functions implement

```text
g_munu
  -> g^munu
  -> Gamma^rho_munu
  -> R^rho_sigma_mu_nu
  -> R_munu
  -> R
  -> G_munu
  -> T_munu
```

When full Riemann output is not requested, the engine can contract curvature directly into the Ricci tensor and avoid persisting the 256-component Riemann field.

Finite derivatives use NumPy's second-order gradient implementation, including one-sided second-order stencils at global boundaries.

Multidimensional tiled execution evaluates halo-expanded local domains and crops each block back to its core before persistence.

## Validation path

Validation now extends well beyond the original flat-space rehabilitation gate.

Current coverage includes:

1. shared constants and Einstein coupling;
2. metric dtype, shape, symmetry, and finiteness validation;
3. exact Minkowski flat-spacetime checks;
4. curved/reference metric tests including de Sitter and Schwarzschild;
5. grid-refinement and field diagnostics;
6. disk-backed and multidimensional-tiling behavior;
7. classical orbital and encounter validation;
8. Schwarzschild Phase 4 validation for curvature, redshift, geodesics, light bending, photon-sphere behavior, and tidal eigenvalues.

See [VALIDATION.md](VALIDATION.md) and [SCHWARZSCHILD_PHASE4_VALIDATION.md](SCHWARZSCHILD_PHASE4_VALIDATION.md).

## Backend policy

CPU/NumPy `float64` is the supported reference backend. The old `tryGPU` path was not a real CUDA implementation.

A future GPU or accelerated backend should be implemented independently and validated component-by-component against this reference path before being enabled as supported scientific output.

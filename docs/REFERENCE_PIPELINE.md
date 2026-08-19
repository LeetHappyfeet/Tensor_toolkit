# Supported reference pipeline

This package is the clean CPU reference path introduced after the technical audit. Legacy code in `solver/` is intentionally left in place until it can be compared component-by-component against this implementation.

## Representation and conventions

The supported numerical metric representation is a NumPy `float64` array with shape `(4, 4, Nt, Nx, Ny, Nz)`. The first two axes are covariant tensor indices and the remaining axes are the coordinate grid in `(t, x, y, z)` order. Metrics must be finite and symmetric.

The reference signature is `(-+++)`.

The Riemann convention is

`R^rho_{ sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}`.

The Ricci contraction is `R_{sigma nu} = R^rho_{ sigma rho nu}`.

The Einstein equation assumes zero cosmological constant:

`G_{mu nu} = (8 pi G / c^4) T_{mu nu}` in SI, or `G_{mu nu} = 8 pi T_{mu nu}` in geometrized units (`G=c=1`).

Physical constants and couplings live in `tensor_toolkit.constants`; conventions live in `tensor_toolkit.conventions`.

## Calculation path

The supported functions in `tensor_toolkit.reference` follow one explicit sequence:

`g_munu -> g^munu -> Gamma^rho_munu -> R^rho_sigma_mu_nu -> R_munu -> R -> G_munu -> T_munu`.

Finite derivatives use NumPy's second-order gradient implementation, including one-sided second-order boundary stencils. This replaces the legacy behavior that copied interior derivatives onto boundaries or silently zeroed mixed-derivative boundary regions.

## Validation path

`pytest` is the first validation gate. Current tests establish:

1. the Einstein SI coupling is derived from the shared constants;
2. float32 and asymmetric reference metrics are rejected;
3. Minkowski spacetime produces exactly zero connection, Riemann tensor, and Einstein tensor;
4. Rindler coordinates produce a nonzero connection while the numerical curvature residual decreases under grid refinement.

Next validation targets are an analytic curved spacetime, measured grid-refinement convergence, comparison with the existing symbolic implementation, and only then component-by-component comparison against the legacy finite-difference solver.

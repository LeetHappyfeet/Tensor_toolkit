# Tensor Toolkit mathematical conventions

The reference solver uses four ordered coordinates

`x^mu = (x^0, x^1, x^2, x^3)`

with metric signature `(-,+,+,+)`, a covariant input metric `g_{mu nu}`, and array layout

`(mu, nu, N0, N1, N2, N3)`.

Most built-in metrics use `(t, x, y, z)`. The Schwarzschild isotropic Cartesian metric uses `(ct, x, y, z)`, with all four coordinates measured in metres.

The connection is

`Gamma^rho_{mu nu} = 1/2 g^{rho sigma}(d_mu g_{sigma nu} + d_nu g_{sigma mu} - d_sigma g_{mu nu})`.

The Riemann convention is

`R^rho_{ sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}`.

The Ricci contraction is

`R_{sigma nu}=R^rho_{ sigma rho nu}`.

The Ricci scalar and Einstein tensor are

`R=g^{mu nu}R_{mu nu}`

and

`G_{mu nu}=R_{mu nu}-(1/2)g_{mu nu}R`.

In SI units,

`G_{mu nu}=(8*pi*G/c^4)T_{mu nu}`,

so

`T_{mu nu}=(c^4/(8*pi*G))G_{mu nu}`.

In geometrized units `G=c=1`,

`T_{mu nu}=G_{mu nu}/(8*pi)`.

Scientific reference calculations use NumPy `float64`. A future accelerated backend must numerically agree with the CPU reference implementation before being considered supported.

Finite differences must never silently wrap. The current reference implementation uses NumPy second-order gradients, including one-sided second-order stencils at global boundaries. Multidimensional tiled execution uses halo cells so internal block boundaries do not become physical/numerical boundaries of the spacetime calculation.

Metric-specific coordinate definitions and unit systems must remain explicit. In particular, SI classical trajectories must not be silently mixed with geometrized metric coordinates.

# Tensor Toolkit mathematical conventions

The reference solver uses coordinates `(t, x, y, z)`, metric signature `(-,+,+,+)`, a covariant input metric `g_{mu nu}`, and array layout `(mu, nu, Nt, Nx, Ny, Nz)`. Grid spacing is `(dt, dx, dy, dz)`.

The connection is

`Gamma^rho_{mu nu} = 1/2 g^{rho sigma}(d_mu g_{sigma nu} + d_nu g_{sigma mu} - d_sigma g_{mu nu})`.

The Riemann convention is

`R^rho_{ sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}`.

The Ricci contraction is `R_{sigma nu}=R^rho_{ sigma rho nu}`. The Ricci scalar is `R=g^{mu nu}R_{mu nu}` and the Einstein tensor is `G_{mu nu}=R_{mu nu}-(1/2)g_{mu nu}R`.

In SI units the field equation is `G_{mu nu}=(8*pi*G/c^4)T_{mu nu}`, hence `T_{mu nu}=(c^4/(8*pi*G))G_{mu nu}`. In geometrized units `G=c=1`, so `T_{mu nu}=G_{mu nu}/(8*pi)`.

Scientific reference calculations use NumPy float64. GPU implementations must also default to float64 and numerically agree with the reference implementation before being considered supported.

Finite differences must never silently wrap. Centered second-order first derivatives require a one-cell stencil margin; centered fourth-order first derivatives require two. Curvature differentiates the connection again, so a solver using only centered stencils has a two-cell trustworthy interior margin at second order and a four-cell margin at fourth order. Boundary behavior must be explicit: documented one-sided stencils or cells marked invalid, never copied interior values masquerading as boundary derivatives.

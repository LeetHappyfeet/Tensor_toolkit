# Schwarzschild Phase 4 validation

Tensor Toolkit validates the Phase 4 observer, curvature, geodesic, signal, and
optics machinery against analytic Schwarzschild results before numerical
relativity work begins.

The supported Schwarzschild metric uses isotropic Cartesian coordinates

    (ct, x, y, z)

with geometric mass

    m = GM/c^2.

Analytic Schwarzschild formulas are usually written using areal radius R, not
the isotropic Cartesian radius rho. Validation therefore converts between them:

    R = rho (1 + m/(2 rho))^2

and

    rho = 1/2 [R - m + sqrt(R(R - 2m))].

Comparing an areal-radius formula directly against rho is not a valid
Schwarzschild reference test.

## Analytic reference helpers

The module

    tensor_toolkit.relativity.schwarzschild

contains independent reference expressions for:

- isotropic/areal radius conversion,
- Schwarzschild lapse,
- static timelike observers,
- radial null tangents with fixed Killing energy,
- static gravitational redshift,
- Kretschmann scalar,
- weak-field light deflection,
- circular timelike initial data,
- photon-sphere initial data,
- static-observer tidal eigenvalues.

These helpers do not call the numerical curvature or geodesic routines they
validate.

## Kretschmann curvature

For Schwarzschild exterior spacetime,

    K = R_abcd R^abcd = 48 m^2 / R^6.

The validation samples the numerical Riemann tensor at R=10m using the existing
reference finite-difference pipeline, contracts it to K, and compares it with
the analytic expression.

This checks more than Ricci=0: Schwarzschild has zero Ricci curvature in vacuum
but nonzero Riemann/Weyl curvature.

## Static gravitational redshift

For static observers at emitter and receiver radii,

    omega_r / omega_e = alpha_e / alpha_r

where alpha is the Schwarzschild lapse. In areal coordinates this is equivalent
to

    sqrt(1 - 2m/R_e) / sqrt(1 - 2m/R_r).

The numerical observable is evaluated through the invariant contraction

    omega = -u_mu k^mu

rather than by directly returning the analytic lapse ratio. This validates the
observer and signal-measurement path.

## Radial timelike geodesic

A particle is released from rest relative to a static observer at R=10m.

Validation checks that:

- areal radius decreases,
- no transverse coordinate motion is generated,
- the timelike normalization remains controlled,
- the Schwarzschild Killing energy alpha^2 u^0 remains conserved.

This tests the Christoffel sampler and the timelike geodesic integrator
together.

## Circular timelike geodesic

For a circular equatorial geodesic,

    u^t = 1 / sqrt(1 - 3m/R)

and

    u^phi = sqrt(m/R^3) / sqrt(1 - 3m/R).

The reference helper transforms this tangent into isotropic Cartesian
coordinates. A stable orbit at R=10m is integrated through a finite arc and the
maximum areal-radius drift and geodesic-normalization drift are checked.

## Weak-field light bending

The leading asymptotic Schwarzschild light-deflection result is

    delta = 4m/b

for impact parameter b >> m.

The validation launches a null ray from a static observer in the weak field,
integrates it past the mass, projects the final tangent into a static observer
frame on the far side, and compares the measured angular deflection against
4m/b.

Because the numerical experiment begins and ends at finite radius and the
analytic reference is only leading-order weak field, this check intentionally
uses a wider tolerance than the exact invariant tests.

## Photon sphere

The Schwarzschild photon sphere lies at

    R = 3m.

Its isotropic radius is obtained through the areal/isotropic conversion. The
analytic null circular tangent is integrated for a short arc.

The test monitors:

- areal-radius drift from 3m,
- null-normalization drift.

The photon sphere is physically unstable, so this is intentionally a short-arc
reference test rather than a demand that a finite-precision trajectory remain
circular indefinitely.

## Static tidal eigenvalues

For a static orthonormal observer in Schwarzschild vacuum, the electric/tidal
Riemann tensor has sorted principal values

    (-2m/R^3, +m/R^3, +m/R^3)

using Tensor Toolkit's documented Riemann convention and geodesic-deviation
sign.

The test obtains the numerical Riemann tensor, constructs a static tetrad,
projects curvature into the observer frame, diagonalizes the 3x3 tidal tensor,
and compares its eigenvalues against this analytic result.

The zero trace is also recorded as a useful vacuum diagnostic.

## Executable validation report

Run the complete gate from Python with:

    from tensor_toolkit.metrics import SchwarzschildIsotropicMetric
    from tensor_toolkit.relativity import validate_schwarzschild_phase4

    metric = SchwarzschildIsotropicMetric(1.89813e27)
    report = validate_schwarzschild_phase4(metric)

    print(report.passed)
    for check in report.checks:
        print(
            check.name,
            check.passed,
            check.numerical,
            check.expected,
            check.relative_error,
        )

Light bending is the longest check and can be omitted during quick development
iterations:

    report = validate_schwarzschild_phase4(
        metric,
        include_light_bending=False,
    )

## Debug mode

Phase 4 debugging is opt-in. Normal scientific-library calls remain silent.

A sampler can expose metric/tensor sampling activity:

    sampler = SpacetimeSampler(
        metric,
        spacings,
        debug=True,
    )

Geodesics support step tracing with adjustable cadence:

    result = integrate_geodesic(
        sampler,
        coordinates,
        tangent,
        affine_step=...,
        steps=...,
        debug=True,
        debug_every=20,
    )

Ray tracing and transport expose the same debug/debug_every pattern where
appropriate.

The complete Schwarzschild gate can also be traced:

    report = validate_schwarzschild_phase4(
        metric,
        debug=True,
    )

Debug messages are written to stderr with component prefixes such as:

    [tensor-toolkit:sampler]
    [tensor-toolkit:geodesic]
    [tensor-toolkit:optics]
    [tensor-toolkit:signals]
    [tensor-toolkit:transport]
    [tensor-toolkit:validation]

Geodesic debug records include the accepted step number, affine parameter,
coordinates, tangent, normalization, and normalization drift. Stop-condition
activation is reported explicitly.

For large ray integrations, use debug_every to avoid producing one line per
accepted step.

## Validation gate

These Schwarzschild checks are intended to be passed before Phase 5 begins.
Failures should be treated as geometry, coordinate, convention, interpolation,
or integration problems rather than having their tolerances silently widened.

The next improvement after this suite is stable should be cached/interpolated
metric and Christoffel fields for large geodesic and ray-bundle workloads,
followed by the planned VTK/PyVista visualizer.

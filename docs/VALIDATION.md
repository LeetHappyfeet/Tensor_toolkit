# Validation strategy

Tensor Toolkit treats validation as a staged scientific gate. Passing software tests is necessary but does not by itself establish that a model is physically realizable or that every quantity is accurate at production resolution.

## 1. Reference geometry validation

The CPU/NumPy `float64` geometry path is tested first.

Current checks include:

- shared physical constants and Einstein coupling,
- metric shape, symmetry, finiteness, and dtype validation,
- exact Minkowski vacuum behavior,
- nontrivial flat-coordinate behavior,
- de Sitter reference behavior,
- Schwarzschild reference data and curved-spacetime calculations,
- symmetry and convergence diagnostics,
- disk-backed versus in-memory execution consistency where applicable.

The reference geometry path is documented in [REFERENCE_PIPELINE.md](REFERENCE_PIPELINE.md).

## 2. Classical dynamics validation

The simulation layer is tested independently of relativistic analysis.

Current checks include:

- Newtonian N-body acceleration,
- RK4 and velocity-Verlet behavior,
- conservation diagnostics for massive systems,
- passive test-particle orbital invariants,
- circular-orbit helpers,
- analytic hyperbolic-orbit quantities,
- closest approach and encounter diagnostics,
- event detection,
- trajectory/worldline conversion.

A zero-mass passive probe does not contribute to total system conservation quantities, so probe-specific energy and angular-momentum diagnostics are used where system totals would be uninformative.

See [NEWTONIAN_MECHANICS.md](NEWTONIAN_MECHANICS.md).

## 3. Relativistic observable validation

Phase 4 is validated against analytic Schwarzschild results in isotropic Cartesian coordinates, with explicit conversion to areal radius where the analytic formula requires it.

The current gate covers:

- Kretschmann curvature,
- static gravitational redshift,
- radial timelike geodesics and Killing-energy behavior,
- circular timelike geodesics,
- weak-field light bending,
- photon-sphere behavior at areal radius `R=3m`,
- static-observer tidal eigenvalues,
- null/timelike normalization drift,
- observer-frame and transport baselines.

The executable validation report is `validate_schwarzschild_phase4()`.

See [SCHWARZSCHILD_PHASE4_VALIDATION.md](SCHWARZSCHILD_PHASE4_VALIDATION.md).

## Numerical interpretation

Finite-difference results should be judged using resolution studies, symmetry residuals, invariant checks, and independent analytic references where available. Tolerances should not be widened merely to hide a failing geometry, coordinate, interpolation, or integrator path.

The Development branch currently prioritizes a transparent CPU reference implementation over performance. Future accelerated backends should be validated component-by-component against this path before being treated as supported.

## Phase 5 gate

Numerical-relativity evolution should not become authoritative until the earlier geometry, dynamics, observer, geodesic, propagation, and Schwarzschild validation layers remain trustworthy under regression testing.

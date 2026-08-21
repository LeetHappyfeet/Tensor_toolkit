# Validation reference data

This directory contains provenance-bearing reference cases used to validate the supported CPU geometry pipeline.

A reference case must record:

- spacetime and coordinate names;
- metric signature and unit convention;
- parameter values and a nonsingular numerical domain;
- the metric components used by the test;
- expected tensor identities or component values;
- the Riemann/Ricci convention when a sign-sensitive quantity is stored;
- primary published or independent-software provenance.

Reference data is an oracle, not generated test output. Do not replace expected values with values produced by the numerical routine under test.

## Validation hierarchy

1. Published analytic result or standard exact solution.
2. Independent computer-algebra implementation (for example EinsteinPy or xAct).
3. Tensor Toolkit symbolic implementation, used only as a cross-check and never as the sole oracle.
4. Legacy finite-difference implementation, treated as code under test.

Numerical tests must avoid coordinate singularities and horizons unless the singular behavior itself is the subject of the test. Finite-difference vacuum identities are tested by convergence under refinement rather than by demanding exact floating-point zero.

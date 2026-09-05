# Tensor Toolkit technical audit

## Historical status

This document is the original rehabilitation audit that preceded the current installable `tensor_toolkit` package. It is preserved for project history.

The current architecture is documented in [ARCHITECTURE.md](../ARCHITECTURE.md), and the supported reference solver is documented in [REFERENCE_PIPELINE.md](../REFERENCE_PIPELINE.md).

## Original audit

Tensor Toolkit is experimental research software split across `Metrics/`, `solver/`, `solver/tools/`, `analysis/`, and `visualizer/`, with separate symbolic examples at repository root. The legacy public data model is primarily dictionaries containing nested component arrays. There is currently no installable package, dependency lock/specification, or coherent pytest validation suite.

### Confirmed defects

- `solver/met2den.py` imports `ricciS`, `ricciT`, `einT`, and `einE` from `solver.tools`, while `solver/tools/__init__.py` exports none of them. It imports `c4Inv2` but calls `c4Inv`.
- `solver/tools/ricciT.py` defines `ricciT2`, not `ricciT`, and references NumPy, finite-difference functions, and `c` without importing/defining them locally.
- `solver/getEnergyTensor.py` calls undefined `changeTensorIndex` and, on the second-order path, `met2den2`; it also assumes a `name` key exists.
- The `tryGPU` path is not a CUDA backend. It creates NumPy arrays and explicitly casts metric data/scaling to float32 before calling CPU Python functions.
- Legacy finite differences copy nearest interior derivatives into pure-derivative boundary cells and leave mixed-derivative boundary regions at zero. These are inconsistent, undocumented boundary conditions that can hide error.
- Tensor validation is duplicated with different assumptions.
- Multiple Minkowski constructors/definitions exist; at least one standalone definition uses `(+---)`, conflicting with the intended `(-+++)` reference convention.
- Visualization code includes incomplete/stubbed functions and cannot serve as a validation layer.
- Legacy numerical code mixes nested lists, NumPy arrays, and optional Torch type checks without a backend abstraction.
- No authoritative Riemann sign convention, Ricci contraction, array layout, unit policy, or precision policy is enforced.
- Symbolic `analyticalEnergyTensor` and finite-difference solver paths are separate implementations with no cross-validation suite.

### Rehabilitation strategy

Do not rewrite or delete the legacy solver yet. Add a clean float64 CPU reference package alongside it, validate it against analytic spacetimes and convergence tests, then compare legacy paths component-by-component. GPU work remains blocked until the CPU reference is trustworthy.

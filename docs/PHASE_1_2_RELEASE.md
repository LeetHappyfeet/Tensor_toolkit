# Tensor Toolkit Phase 1/2 CPU Reference Release

Version 0.2.0 intentionally limits scope to two objectives:

1. Maintain one readable, float64 CPU reference path for `g_cov -> g_contra -> Gamma -> Riemann -> Ricci -> R -> Einstein -> T` using the documented `(-,+,+,+)` convention.
2. Provide an installable command-line application for selecting and running built-in experiments.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
tensor-toolkit doctor
tensor-toolkit list
tensor-toolkit run minkowski
tensor-toolkit run de-sitter --output results/de_sitter
tensor-toolkit run alcubierre --output results/alcubierre
```

Running `tensor-toolkit` without arguments in an interactive terminal opens a basic experiment selector.

## GPU status

GPU execution is explicitly unsupported in 0.2.0. The previous `tryGPU` path used NumPy arrays and therefore did not constitute a GPU implementation. The supported backend is CPU/NumPy float64. A future GPU backend should be implemented separately and validated component-by-component against the CPU reference pathway before being enabled.

## Output

When `--output DIRECTORY` is supplied, Tensor Toolkit writes:

- `result.npz`: compressed NumPy arrays for requested fields and coordinate axes.
- `metadata.json`: metric name, coordinates, grid spacing and shape, units, backend, and field names.

## Phase boundary

This release does not add particle dynamics, classical mechanics, electromagnetism, 3+1 evolution, or matter backreaction. Those remain later phases after the reference GR path and experiment runtime are established.

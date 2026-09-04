# Tensor Toolkit

Tensor Toolkit is an experimental scientific-computing package for classical and relativistic physics simulation. It combines Newtonian many-body trajectory integration with a validated general-relativity tensor pipeline, allowing moving bodies and test particles to be followed through classical simulations and sampled in spacetime metrics.

The current development direction is a validated CPU reference pipeline, a Newtonian point-mass dynamics layer, and bridges between simulated trajectories and relativistic spacetime calculations. Legacy symbolic, plotting, and GPU-era code remains in the repository for migration/reference purposes, but it is not the authoritative execution path.
<img width="1920" height="1017" alt="python_nt60fyL2VS" src="https://github.com/user-attachments/assets/02f5bb28-b256-4136-ae66-e6caa7cbcfdc" />

## Current GR pipeline

The supported reference path is:

```text
g_cov
  -> g_contra
  -> Christoffel symbols
  -> Riemann tensor
  -> Ricci tensor
  -> Ricci scalar
  -> Einstein tensor
  -> stress-energy tensor
```

Tensor conventions, constants, validation, and experiment execution live in the installable `tensor_toolkit` package.

The supported backend is CPU / NumPy `float64`. GPU execution is intentionally disabled until a real independently validated GPU backend is implemented.

## Development installation

Python 3.10 or newer is required.

```text
git clone --branch Development https://github.com/LeetHappyfeet/Tensor_toolkit.git Tensor_toolkit-dev
cd Tensor_toolkit-dev
python -m pip install -e .
```

On Windows, Miniconda/Conda environments work well. The desktop simulator uses Tkinter and Matplotlib.

## Command line

Check the installation:

```text
tensor-toolkit doctor
tensor-toolkit list
```

Run reference experiments:

```text
tensor-toolkit run minkowski
tensor-toolkit run de-sitter
tensor-toolkit run alcubierre --points 7
```

Save and inspect results:

```text
tensor-toolkit run de-sitter --points 7 --output results/de-sitter7
tensor-toolkit inspect results/de-sitter7
tensor-toolkit inspect results/de-sitter7 --field einstein --center
```

Run a resolution study:

```text
tensor-toolkit convergence de-sitter --points 5 7 9
tensor-toolkit convergence alcubierre --points 5 7 9
```

## Newtonian mechanics and relativity bridge

The Development branch now includes a classical point-mass simulation layer designed to provide physically meaningful trajectories for the relativistic engine.

Current capabilities include:

- Newtonian N-body gravity with massive bodies and passive test particles,
- velocity-Verlet and RK4 trajectory integration,
- reusable orbital and flyby initial conditions,
- energy, momentum, and angular-momentum diagnostics,
- passive-probe specific energy and angular-momentum validation,
- elliptic, parabolic, and hyperbolic orbit classification,
- bound-orbit elements and analytic hyperbolic reference calculations,
- pairwise closest-approach and encounter diagnostics,
- arbitrary-time trajectory sampling, and
- trajectory-to-metric and local tensor sampling.

Run the built-in Jupiter/probe experiment with:

```text
tensor-toolkit simulate demo-flyby
```

A Newtonian trajectory can also be sampled in a Schwarzschild spacetime without changing the classical motion:

```text
tensor-toolkit simulate demo-flyby --schwarzschild jupiter probe --relativity-samples 5
```

The bridge uses Schwarzschild isotropic Cartesian coordinates `(ct, x, y, z)`, allowing the Cartesian Newtonian trajectory to feed directly into metric evaluation. Proper time is integrated along the trajectory, and selected events such as closest approach can be passed through the existing GR finite-difference tensor pipeline:

```text
tensor-toolkit simulate demo-flyby --schwarzschild jupiter probe --relativity-samples 5 --gr-fields metric christoffel ricci einstein --gr-spacing 1000000
```

This is currently a one-way coupling: Newtonian dynamics generates the trajectory and GR evaluates the spacetime along it. Schwarzschild sampling treats one selected massive body as an isolated, non-rotating spherical source; the project does not yet solve a self-consistent many-body relativistic spacetime.

See `docs/NEWTONIAN_MECHANICS.md` for the classical simulation and relativity-bridge details.

## Metric Tensor Simulator

Launch the desktop simulator with:

```text
tensor-toolkit visualize
```

or, from a repository checkout:

```text
python visualizer.py
```

The visualizer is now a front end to the same `run_experiment()` pipeline used by the CLI. It no longer constructs a separate symbolic metric or calls the legacy `analyticalEnergyTensor.py` implementation.

The current viewer supports:

- selecting a registered metric (Minkowski is the default baseline),
- editing metric parameters,
- setting grid resolution and uniform coordinate extent,
- viewing `metric`, `inverse_metric`, `ricci`, `einstein`, and `stress_energy` rank-2 fields,
- selecting tensor components and two-dimensional coordinate slices,
- inspecting the complete 4x4 tensor at the grid center,
- seeing stored numerical validation results,
- saving and reopening the same NPZ/JSON result format used by the CLI.

See `docs/VISUALIZER.md` for details.

## Numerical status

Tensor Toolkit is still research/development software. A successful run is not the same thing as a validated physical result.

Minkowski vacuum is the basic exact sanity check. Curved metrics are being used for analytic and convergence validation. The CLI and GUI surface numerical symmetry warnings rather than hiding them.

The GR grid simulator samples metric fields over `(t, x, y, z)`, while the classical layer evolves moving bodies over time and can feed selected trajectory events into metric/tensor evaluation. Tensor Toolkit is not yet a full numerical-relativity evolution code: it does not evolve ADM/BSSN initial data or self-consistently couple moving matter back into spacetime.

## Tests

Install test dependencies and run:

```text
python -m pip install -e ".[test]"
python -m pytest
```

## License

See the repository license files for the applicable project license.

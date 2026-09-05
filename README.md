# Tensor Toolkit

Tensor Toolkit is an experimental scientific-computing package for classical and relativistic physics simulation. It combines Newtonian many-body trajectory integration with a validated general-relativity tensor pipeline, allowing moving bodies and test particles to be followed through classical simulations and sampled in spacetime metrics.

The current development direction is a validated CPU reference pipeline, a generalized classical dynamics layer, shared worldlines, and a relativistic observer/propagation analysis layer. Legacy symbolic, plotting, and GPU-era code remains in the repository for migration/reference purposes, but it is not the authoritative execution path.
<img width="1920" height="1017" alt="python_nt60fyL2VS" src="https://github.com/user-attachments/assets/02f5bb28-b256-4136-ae66-e6caa7cbcfdc" />
The current development direction is a validated CPU reference pipeline, a Newtonian point-mass dynamics layer, and bridges between simulated trajectories and relativistic spacetime calculations. Legacy symbolic, plotting, and GPU-era code remains in the repository for migration/reference purposes, but it is not the authoritative execution path.


# Tensor Toolkit
What's new?
Added new memory handling to prevent memory overflows. This can be changed in either the UI or terminal window. For example:
```
tensor-toolkit run alcubierre --points 31 --fields einstein stress_energy --memory-mode auto
```
or
```
tensor-toolkit run alcubierre \
    --points 31 \
    --fields einstein stress_energy \
    --memory-mode auto
```
Added more changes to the GUI.

<img width="1920" height="1017" alt="python_nt60fyL2VS" src="https://github.com/user-attachments/assets/f8308a17-6831-4e61-b80c-06263dde65d4" />




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

The Development branch includes a classical simulation layer designed to provide physically meaningful trajectories for the relativistic engine.

Current capabilities include:

- Newtonian N-body gravity with massive bodies and passive test particles,
- pluggable dynamics models and composite external forces,
- velocity-Verlet and RK4 trajectory integration,
- finite body radii and event detection,
- reusable orbital and flyby initial conditions,
- energy, momentum, and angular-momentum diagnostics,
- passive-probe orbital validation,
- pairwise closest-approach and encounter diagnostics,
- trajectory/worldline conversion, and
- trajectory-to-metric and local tensor sampling.

Run the built-in Jupiter/probe experiment with:

```text
tensor-toolkit simulate demo-flyby
```

A Newtonian trajectory can also be sampled in a Schwarzschild spacetime without changing the classical motion:

```text
tensor-toolkit simulate demo-flyby --schwarzschild jupiter probe --relativity-samples 5
```

The bridge uses Schwarzschild isotropic Cartesian coordinates `(ct, x, y, z)`, allowing the Cartesian Newtonian trajectory to feed directly into metric evaluation. Proper time is integrated along the trajectory, selected events can be passed through the existing GR finite-difference tensor pipeline, and Schwarzschild samples now expose a shared `Worldline` with four-velocity.

```text
tensor-toolkit simulate demo-flyby --schwarzschild jupiter probe --relativity-samples 5 --gr-fields metric christoffel ricci einstein --gr-spacing 1000000
```

This remains one-way coupling: classical dynamics generates the trajectory and GR evaluates the spacetime along it. Schwarzschild sampling treats one selected massive body as an isolated, non-rotating spherical source; the project does not yet solve a self-consistent many-body relativistic spacetime.

See `docs/NEWTONIAN_MECHANICS.md` and `docs/SIMULATION_ARCHITECTURE.md`.

## Phase 4 relativistic observation and propagation

The Development branch now includes a dedicated `tensor_toolkit.relativity` analysis package built downstream of the validated reference geometry pipeline.

Current Phase 4 infrastructure includes:

- `SpacetimeSampler` for arbitrary-event metric, connection, curvature, and worldline field sampling,
- orthonormal observer tetrads and static/comoving observer frames,
- observer-local vector, tensor, and stress-energy measurements,
- Ricci-square and Kretschmann curvature invariants,
- Weyl and observer-frame electric Weyl curvature,
- tidal tensors, principal tidal directions, and geodesic-deviation acceleration,
- timelike and null geodesic integration with normalization-drift diagnostics,
- parallel and Fermi-Walker vector transport,
- invariant photon frequency/redshift measurements,
- coordinate light-travel-time and radar-distance observables,
- null-ray construction from local observer directions, and
- scientific ray-bundle integration.

Phase 4 intentionally analyzes prescribed geometry rather than evolving it. The geodesic solver is currently a transparent fixed-step RK4 reference path whose connection evaluations use local finite-difference stencils; cached/interpolated fields should be introduced before large ray-tracing workloads.

See `docs/PHASE4_RELATIVITY.md`.

## Metric Tensor Simulator

Launch the desktop simulator with:

```text
tensor-toolkit visualize
```

or, from a repository checkout:

```text
python visualizer.py
```

The visualizer is a front end to the same `run_experiment()` pipeline used by the CLI. It no longer constructs a separate symbolic metric or calls the legacy `analyticalEnergyTensor.py` implementation.

The current viewer supports:

- selecting a registered metric,
- editing metric parameters,
- setting grid resolution and coordinate extent,
- viewing rank-2 metric/curvature/stress-energy fields,
- selecting tensor components and two-dimensional coordinate slices,
- inspecting complete tensors at the grid center,
- seeing stored numerical validation results, and
- saving and reopening the same NPZ/JSON result format used by the CLI.

A later visualization milestone will move the 3-D field, worldline, tidal-glyph, and ray-bundle views to VTK/PyVista while keeping the renderer downstream of the solver.

See `docs/VISUALIZER.md`.

## Numerical status

Tensor Toolkit is still research/development software. A successful run is not the same thing as a validated physical result.

Minkowski vacuum is the basic exact sanity check. Curved metrics are being used for analytic and convergence validation. The CLI and GUI surface numerical symmetry warnings rather than hiding them.

The GR grid simulator samples prescribed metric fields over `(t, x, y, z)`. The classical and Phase 4 layers can generate and analyze worldlines, observer measurements, geodesics, and light propagation in those prescribed spacetimes.

Tensor Toolkit is not yet a full numerical-relativity evolution code: it does not evolve ADM/BSSN initial data, solve Einstein's equations as an initial-value PDE system, or self-consistently couple moving matter back into spacetime. Those remain Phase 5 work after the earlier machinery passes stronger Schwarzschild and propagation validation.

## Tests

Install test dependencies and run:

```text
python -m pip install -e ".[test]"
python -m pytest
```

## License

See the repository license files for the applicable project license.

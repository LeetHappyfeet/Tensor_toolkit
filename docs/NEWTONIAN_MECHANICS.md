# Newtonian mechanics and trajectory bridge

Tensor Toolkit includes a classical point-mass dynamics layer under `tensor_toolkit.physics`. Its purpose is to generate physically meaningful trajectories that can be validated independently and then converted into shared worldlines for relativistic sampling and analysis.

## Architecture

The classical path is:

```text
Body/System
   |
   v
DynamicsModel
   |
   v
simulate()
   |
   v
Trajectory
   |
   +--> diagnostics/events
   |
   +--> Worldline
            |
            +--> metric sampling
            +--> local GR tensor sampling
            +--> Schwarzschild analysis
            +--> Phase 4 relativistic analysis
```

See [SIMULATION_ARCHITECTURE.md](SIMULATION_ARCHITECTURE.md) for the generalized dynamics and state model.

## Units

Classical body states use SI units by default:

- position: metres,
- time: seconds,
- mass: kilograms,
- velocity: metres per second.

Metric coordinate systems may use different conventions. Coordinate conversion must therefore be explicit.

For example, `SchwarzschildIsotropicMetric` uses `(ct,x,y,z)` with all four coordinates measured in metres, while the Alcubierre metric currently uses geometrized coordinates.

A Newtonian SI trajectory must not be silently inserted into a geometrized metric.

## Bodies and systems

`Body` is the initial-condition/configuration object. Bodies can have positive mass, zero mass for passive test particles, and an optional finite radius.

A zero-mass probe responds to gravity but does not accelerate massive bodies.

`SystemState` is the evolving translational state and stores positions, velocities, and masses.

## Dynamics models

The default model is direct Newtonian N-body gravity.

`simulate()` also accepts an explicit `DynamicsModel`. Current infrastructure includes:

- `NewtonianGravity`,
- `ConstantThrust`,
- `CompositeDynamics`.

Dynamics models declare whether acceleration depends on velocity.

The current velocity-Verlet implementation is supported only for velocity-independent acceleration laws. RK4 is the reference path for velocity-dependent models and future post-Newtonian extensions.

## Integrators

Supported fixed-step integrators are:

- fourth-order Runge-Kutta,
- velocity-Verlet.

Example:

```python
from tensor_toolkit.physics import Body, System, simulate

system = System([
    Body("star", 1.98847e30, [0, 0, 0], [0, 0, 0]),
    Body("probe", 0.0, [1.495978707e11, 0, 0], [0, 29_784.7, 0]),
])

trajectory = simulate(
    system,
    duration=365.25 * 86400,
    dt=3600,
    method="verlet",
)
```

## Events and finite radii

Event detectors run after accepted integration steps.

Current event infrastructure includes finite-radius collision detection and distance-crossing detection. Events are recorded on the returned trajectory.

Collision detection is observational at this stage. It does not automatically merge, bounce, fragment, or stop bodies unless future event-response logic explicitly implements that behavior.

## Conservation diagnostics

For massive systems, Tensor Toolkit reports:

- total-energy relative drift,
- linear-momentum absolute drift,
- angular-momentum relative drift.

```python
from tensor_toolkit.physics import conservation_diagnostics

diagnostics = conservation_diagnostics(trajectory, system.masses)
```

These diagnostics make integrator and timestep quality visible before a trajectory is used by relativistic analysis.

## Passive test-particle validation

A zero-mass probe contributes nothing to the total Newtonian system energy or momentum. System conservation values can therefore be exactly zero even when the probe trajectory is inaccurate.

For massive-primary to passive-probe encounters, Tensor Toolkit additionally evaluates probe-specific quantities:

- specific orbital energy,
- specific angular momentum,
- relative specific-energy drift,
- relative specific-angular-momentum drift.

This is the appropriate validation layer for spacecraft/test-particle runs.

## Orbit helpers

Reusable initial-condition helpers include circular orbits and hyperbolic flybys.

```python
from tensor_toolkit.physics import circular_orbit_system, hyperbolic_flyby_system
```

The hyperbolic helper interprets the supplied incoming speed at the specified finite initial distance. It does not silently reinterpret it as asymptotic velocity at infinity.

## Encounter diagnostics

Pairwise encounter diagnostics can report:

- closest approach,
- periapsis relative speed,
- initial/final relative speed,
- finite-window velocity-direction change or deflection where appropriate.

Passive test-particle hyperbolic encounters can also be compared with the analytic Newtonian reference orbit:

- eccentricity,
- asymptotic velocity `v_inf`,
- analytic periapsis distance,
- analytic periapsis speed,
- asymptotic scattering angle,
- numerical periapsis error,
- numerical periapsis-speed error.

The finite-window simulated direction change is kept distinct from the asymptotic analytic scattering angle.

## Orbit classification

Passive test-particle motion is classified from specific orbital energy:

```text
specific energy < 0  -> elliptic / bound
specific energy ~ 0  -> parabolic
specific energy > 0  -> hyperbolic / unbound
```

Bound elliptic diagnostics include eccentricity, semi-major axis, periapsis, apoapsis, and orbital period.

## Classical simulation CLI

The built-in flyby experiment can be run with:

```bash
tensor-toolkit simulate demo-flyby
```

Optional controls include duration, timestep, integration method, sample interval, and output directory.

Saved trajectory data uses:

```text
times          (T,)
positions      (T, N, 3)
velocities     (T, N, 3)
accelerations  (T, N, 3)
body_names     (N,)
```

The layout is many-body rather than hard-coded to two-body motion.

## Metric sampling

A trajectory can be sampled against a metric without recomputing the classical motion.

```python
from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.physics import sample_metric_along_trajectory
```

Returned event samples include metric coordinates, classical velocity/acceleration, and the local 4x4 metric.

## Local tensor sampling

Curvature cannot be derived from a single isolated metric value with the current finite-difference reference solver.

`sample_tensors_along_trajectory()` therefore constructs a local four-dimensional stencil around each selected event, evaluates the normal GR pipeline, and retains the center value.

The supplied spacings belong to the metric coordinate system.

## Worldlines

`trajectory_worldline()` converts a selected classical body into the common `Worldline` representation.

The classical worldline uses coordinates `(t,x,y,z)` and can store coordinate acceleration. Coordinate acceleration must not be confused with covariant four-acceleration.

The shared worldline boundary allows later observer, propagation, and visualization code to operate without depending on the original trajectory generator.

## Schwarzschild relativity bridge

A validated Newtonian trajectory can be sampled in a static Schwarzschild background without changing the classical path.

For the built-in flyby:

```bash
tensor-toolkit simulate demo-flyby \
  --schwarzschild jupiter probe \
  --relativity-samples 5
```

The bridge uses isotropic Cartesian Schwarzschild coordinates:

```text
(ct, x, y, z)
```

For the selected pair, positions are measured relative to the chosen primary.

The bridge reports quantities including isotropic radius, proper-time rate `dτ/dt`, accumulated proper time, and a shared worldline/four-velocity representation.

If closest approach is available from encounter diagnostics, it is included in the selected relativity events.

## Local GR fields along the flyby

The Schwarzschild bridge can also invoke the existing finite-difference tensor pipeline:

```bash
tensor-toolkit simulate demo-flyby \
  --schwarzschild jupiter probe \
  --relativity-samples 5 \
  --gr-fields metric christoffel ricci einstein \
  --gr-spacing 1000000
```

The GR spacing sets the local stencil spacing in metres for each of `(ct,x,y,z)`.

## Physical limitation

The Schwarzschild bridge is deliberately one-way.

The Newtonian simulation creates the trajectory; Schwarzschild geometry is then sampled along it. The relativistic field does not yet feed back into the trajectory.

Schwarzschild also represents one isolated, nonrotating spherical source. Other Newtonian bodies can exist in the classical simulation, but their masses are not combined into an exact many-body relativistic spacetime.

## Current development direction

The classical engine now has the state, dynamics, event, diagnostics, trajectory, and worldline interfaces needed for more advanced motion models.

Likely later additions include validated velocity-dependent post-Newtonian dynamics, rotating/Kerr backgrounds, improved spacecraft force models, and tighter integration with Phase 4 geodesic/observer analysis.

Self-consistent matter/spacetime evolution remains a numerical-relativity problem rather than an extension of the Newtonian bridge.

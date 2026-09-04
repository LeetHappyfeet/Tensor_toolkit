# Simulation architecture foundation

The Development branch now separates numerical integration from the physical
acceleration law. This is the first-generation simulation foundation intended
to support Newtonian, post-Newtonian, spacecraft, and eventually geodesic
dynamics without creating parallel simulation engines.

## Dynamics models

`simulate()` retains its historical API. When no dynamics model is supplied,
it constructs `NewtonianGravity` and produces the same direct O(N^2)
Newtonian dynamics as before.

A caller can now supply an explicit dynamics model:

```python
from tensor_toolkit.physics import NewtonianGravity, simulate

trajectory = simulate(
    system,
    duration=1000.0,
    dt=1.0,
    method="rk4",
    dynamics=NewtonianGravity(),
)
```

Independent acceleration laws can be combined with `CompositeDynamics`.
The first external-force implementation is `ConstantThrust`, which applies
constant inertial-frame force vectors to named positive-mass bodies.

```python
from tensor_toolkit.physics import CompositeDynamics, ConstantThrust, NewtonianGravity

dynamics = CompositeDynamics([
    NewtonianGravity(),
    ConstantThrust({"spacecraft": [1000.0, 0.0, 0.0]}),
])
```

Dynamics models declare whether they depend on velocity. Velocity-Verlet is
rejected for velocity-dependent models because the current Verlet formulation
is only a supported reference integrator for velocity-independent acceleration
laws. RK4 is the current path for future 1PN velocity-dependent dynamics.

## Evolving state

`Body` remains the initial-condition/configuration object and now optionally
contains a finite radius. `SystemState` is the evolving translational state
used internally by the simulator:

```text
positions  (N, 3)
velocities (N, 3)
masses     (N,)
```

Mass is present in the evolving state so variable-mass propulsion can be added
without redesigning the state boundary. This milestone does not yet implement
mass-flow integration.

## Events and finite radii

Event detectors run after accepted integration steps. `CollisionDetector`
uses the sum of two bodies' finite radii and records first contact as a
`SimulationEvent` attached to the returned `Trajectory`.

Collision events are observational in this milestone. They do not yet stop the
integrator, merge bodies, bounce bodies, fragment bodies, or apply material
physics. Those behaviors should be implemented as explicit event responses
rather than hidden inside the gravity model.

Saved `SimulationExperiment` metadata now includes detected events.

## Worldlines

`Worldline` is the common sampled spacetime-history container. It stores:

```text
parameter
coordinates            x^mu
tangent                 dx^mu/dlambda
coordinate_acceleration optional
proper_time             optional
four_velocity           optional
metadata
```

`trajectory_worldline()` converts one body from a classical `Trajectory`
into a coordinate-time worldline with coordinates `(t, x, y, z)`.

The Schwarzschild bridge now also returns a `Worldline`. Its coordinates are
`(ct, x, y, z)`; proper time is integrated as before, and the sampled
four-velocity is calculated from

```text
u^mu = dx^mu/dtau = (dx^mu/dt) / (dtau/dt).
```

The currently stored acceleration on these worldlines is coordinate
acceleration, not covariant four-acceleration. A geodesic can have changing
coordinate velocity while having zero physical four-acceleration, so
four-acceleration will be added at the relativistic worldline layer rather than
mislabeling the classical acceleration.

## Compatibility

Existing calls such as

```python
simulate(system, duration=..., dt=..., method="verlet")
```

remain supported. Existing trajectory arrays and Schwarzschild sample arrays
remain present. Schwarzschild saved data now additionally contains
`four_velocity`.

The direct Newtonian gravity implementation remains the authoritative reference
model. Future Barnes-Hut or other accelerated gravity models should be separate
dynamics implementations that can be validated against it.

## Next physics milestone

With integration decoupled from dynamics, the next physics addition can be a
well-scoped velocity-dependent 1PN model using the existing RK4 path. The same
initial conditions can then be evolved with Newtonian and 1PN dynamics and
converted into the same `Worldline` interface for comparison.

# Newtonian mechanics and trajectory bridge

Tensor Toolkit now includes a small classical point-mass dynamics layer under
`tensor_toolkit.physics`. Its purpose is to generate physically meaningful
trajectories that can be sampled by the existing metric and GR tensor pipeline.

## Scope of the first implementation

The initial supported model is intentionally narrow:

- Cartesian point masses.
- Newtonian N-body gravity.
- Fixed-step fourth-order Runge-Kutta integration.
- Symplectic velocity-Verlet integration for long orbital runs.
- Conservation diagnostics for energy, linear momentum, and angular momentum.
- Reusable circular-orbit and hyperbolic-flyby initial-condition helpers.
- Arbitrary-time interpolation along a simulated trajectory.
- Direct metric evaluation at trajectory events.
- Local 4-D tensor stencils centered on trajectory events.

Classical body states use SI units by default: metres, seconds, kilograms, and
metres per second. Some Tensor Toolkit metrics use geometrized coordinates.
The bridge therefore accepts an optional `coordinate_transform` so unit and
coordinate conversion is explicit rather than implicit.

## Basic orbit simulation

```python
import numpy as np

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

A zero-mass body is supported as a passive test particle: it responds to
gravity but does not accelerate the massive bodies.

## Metric sampling

```python
from tensor_toolkit.metrics import MinkowskiMetric
from tensor_toolkit.physics import sample_metric_along_trajectory

samples = sample_metric_along_trajectory(
    MinkowskiMetric(),
    trajectory,
    body="probe",
    times=[0.0, 86400.0, 2 * 86400.0],
)
```

Each returned event contains its metric-coordinate `(t, x, y, z)`, classical
velocity and acceleration, and a 4x4 metric tensor.

## Tensor sampling

Curvature cannot be derived from one isolated metric value because the current
reference pipeline obtains derivatives numerically. The trajectory bridge
therefore constructs a local 3x3x3x3 coordinate stencil centered on each event,
feeds that metric grid into the normal Tensor Toolkit tensor evaluator, and
keeps the tensor values at the center.

```python
from tensor_toolkit.physics import sample_tensors_along_trajectory

samples = sample_tensors_along_trajectory(
    MinkowskiMetric(),
    trajectory,
    body="probe",
    times=[0.0, 86400.0],
    spacings=(1.0, 1000.0, 1000.0, 1000.0),
    outputs={"metric", "einstein", "stress_energy"},
)
```

The four spacings are in the metric coordinate system and correspond to
`(dt, dx, dy, dz)`.

## Important unit boundary

A Newtonian SI trajectory must not be fed blindly into a metric defined in
geometrized units. For metrics such as Alcubierre, callers should provide a
coordinate transform appropriate to that metric and chosen unit convention.
This is deliberately explicit because silently mixing SI and geometrized
coordinates would produce numerically valid but physically meaningless output.

## Conservation diagnostics

The simulator can now report Newtonian conservation behavior directly:

```python
from tensor_toolkit.physics import conservation_diagnostics

diagnostics = conservation_diagnostics(
    trajectory,
    system.masses,
)

print(diagnostics.energy_relative_drift)
print(diagnostics.momentum_absolute_drift)
print(diagnostics.angular_momentum_relative_drift)
```

These diagnostics are intended to make integrator and timestep quality visible
before trajectories are used as inputs to relativistic analysis.

## Reusable orbit and flyby setups

```python
from tensor_toolkit.physics import circular_orbit_system, hyperbolic_flyby_system

orbit = circular_orbit_system(
    primary_mass=1.98847e30,
    radius=1.495978707e11,
)

flyby = hyperbolic_flyby_system(
    primary_mass=1.89813e27,
    initial_distance=5.0e9,
    impact_parameter=7.0e8,
    incoming_speed=15_000.0,
)
```

The flyby helper uses the specified finite-distance initial speed. It does not
silently reinterpret that value as asymptotic velocity at infinity.

After integration, encounter diagnostics can report closest approach, periapsis
relative speed, initial/final relative speed, and the net deflection angle:

```python
from tensor_toolkit.physics import encounter_diagnostics, simulate

trajectory = simulate(
    flyby,
    duration=1.0e6,
    dt=10.0,
    method="verlet",
)

encounter = encounter_diagnostics(
    trajectory,
    primary="primary",
    probe="probe",
)

print(encounter.closest_approach_distance)
print(encounter.periapsis_relative_speed)
print(encounter.deflection_angle)
```

## Next steps

The next classical milestone is a CLI-accessible orbit/flyby experiment that
records closest approach, deflection angle, periapsis speed, conservation
diagnostics, and trajectory samples. The next relativity milestone is to feed
those same events into Schwarzschild and later Kerr metric sampling, followed by
observer-frame proper time and four-velocity calculations.

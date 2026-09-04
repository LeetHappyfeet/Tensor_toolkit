"""Numerical integration for classical point-mass systems."""

from __future__ import annotations
import numpy as np
from .gravity import newtonian_gravity_accelerations
from .state import System
from .trajectory import Trajectory

def _derivative(state: np.ndarray, masses: np.ndarray, *, gravitational_constant: float, softening: float):
    n = masses.size
    positions = state[:n]
    velocities = state[n:]
    accelerations = newtonian_gravity_accelerations(
        positions, masses, gravitational_constant=gravitational_constant, softening=softening
    )
    return np.concatenate((velocities, accelerations), axis=0)

def _rk4_step(state, dt, masses, *, gravitational_constant, softening):
    kwargs = {"gravitational_constant": gravitational_constant, "softening": softening}
    k1 = _derivative(state, masses, **kwargs)
    k2 = _derivative(state + 0.5 * dt * k1, masses, **kwargs)
    k3 = _derivative(state + 0.5 * dt * k2, masses, **kwargs)
    k4 = _derivative(state + dt * k3, masses, **kwargs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _velocity_verlet_step(positions, velocities, accelerations, dt, masses, *, gravitational_constant, softening):
    next_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    next_accelerations = newtonian_gravity_accelerations(
        next_positions,
        masses,
        gravitational_constant=gravitational_constant,
        softening=softening,
    )
    next_velocities = velocities + 0.5 * (accelerations + next_accelerations) * dt
    return next_positions, next_velocities, next_accelerations

def simulate(
    system: System,
    *,
    duration: float,
    dt: float,
    method: str = "rk4",
    gravitational_constant: float | None = None,
    softening: float = 0.0,
) -> Trajectory:
    """Integrate an N-body Newtonian system with RK4 or velocity-Verlet."""
    from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT
    duration = float(duration)
    dt = float(dt)
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
    method = method.lower()
    if method not in {"rk4", "verlet"}:
        raise ValueError("supported integration methods: 'rk4', 'verlet'")
    G = GRAVITATIONAL_CONSTANT if gravitational_constant is None else float(gravitational_constant)

    steps = int(np.ceil(duration / dt))
    times = np.linspace(0.0, duration, steps + 1, dtype=np.float64)
    n = len(system.bodies)
    positions = np.empty((steps + 1, n, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    accelerations = np.empty_like(positions)
    state = np.concatenate((system.positions, system.velocities), axis=0)
    positions[0] = state[:n]
    velocities[0] = state[n:]
    accelerations[0] = newtonian_gravity_accelerations(
        positions[0], system.masses, gravitational_constant=G, softening=softening
    )

    for step in range(steps):
        step_dt = float(times[step + 1] - times[step])
        if method == "rk4":
            state = _rk4_step(state, step_dt, system.masses, gravitational_constant=G, softening=softening)
            positions[step + 1] = state[:n]
            velocities[step + 1] = state[n:]
            accelerations[step + 1] = newtonian_gravity_accelerations(
                positions[step + 1], system.masses, gravitational_constant=G, softening=softening
            )
        else:
            p, v, a = _velocity_verlet_step(
                positions[step],
                velocities[step],
                accelerations[step],
                step_dt,
                system.masses,
                gravitational_constant=G,
                softening=softening,
            )
            positions[step + 1] = p
            velocities[step + 1] = v
            accelerations[step + 1] = a
            state = np.concatenate((p, v), axis=0)

    return Trajectory(times, positions, velocities, accelerations, system.names)

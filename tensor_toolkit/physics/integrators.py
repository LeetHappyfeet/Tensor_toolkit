"""Numerical integration for classical translational systems."""

from __future__ import annotations

import numpy as np

from .dynamics import DynamicsModel, NewtonianGravity
from .events import EventDetector, SimulationEvent
from .state import System, SystemState
from .trajectory import Trajectory


def _accelerations(
    dynamics: DynamicsModel,
    t: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    system: System,
) -> np.ndarray:
    acceleration = np.asarray(
        dynamics.accelerations(t, positions, velocities, masses, system),
        dtype=np.float64,
    )
    if acceleration.shape != positions.shape:
        raise ValueError(
            f"dynamics model returned acceleration shape {acceleration.shape}, "
            f"expected {positions.shape}"
        )
    if not np.all(np.isfinite(acceleration)):
        raise ValueError("dynamics model returned non-finite acceleration")
    return acceleration


def _derivative(
    t: float,
    state: np.ndarray,
    masses: np.ndarray,
    system: System,
    dynamics: DynamicsModel,
) -> np.ndarray:
    n = masses.size
    positions = state[:n]
    velocities = state[n:]
    acceleration = _accelerations(
        dynamics, t, positions, velocities, masses, system
    )
    return np.concatenate((velocities, acceleration), axis=0)


def _rk4_step(
    t: float,
    state: np.ndarray,
    dt: float,
    masses: np.ndarray,
    system: System,
    dynamics: DynamicsModel,
) -> np.ndarray:
    k1 = _derivative(t, state, masses, system, dynamics)
    k2 = _derivative(t + 0.5 * dt, state + 0.5 * dt * k1, masses, system, dynamics)
    k3 = _derivative(t + 0.5 * dt, state + 0.5 * dt * k2, masses, system, dynamics)
    k4 = _derivative(t + dt, state + dt * k3, masses, system, dynamics)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _velocity_verlet_step(
    t: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt: float,
    masses: np.ndarray,
    system: System,
    dynamics: DynamicsModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    next_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    predicted_velocities = velocities + accelerations * dt
    next_accelerations = _accelerations(
        dynamics,
        t + dt,
        next_positions,
        predicted_velocities,
        masses,
        system,
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
    dynamics: DynamicsModel | None = None,
    event_detectors: tuple[EventDetector, ...] | list[EventDetector] = (),
) -> Trajectory:
    """Integrate a system with a pluggable acceleration law.

    When dynamics is omitted, the historical behavior is preserved by using
    direct Newtonian gravity with the supplied gravitational constant and
    softening. Velocity-Verlet is restricted to dynamics models that declare
    themselves velocity-independent.
    """
    from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT

    duration = float(duration)
    dt = float(dt)
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
    method = method.lower()
    if method not in {"rk4", "verlet"}:
        raise ValueError("supported integration methods: 'rk4', 'verlet'")

    if dynamics is None:
        G = (
            GRAVITATIONAL_CONSTANT
            if gravitational_constant is None
            else float(gravitational_constant)
        )
        dynamics = NewtonianGravity(
            gravitational_constant=G,
            softening=float(softening),
        )

    if method == "verlet" and bool(getattr(dynamics, "velocity_dependent", True)):
        raise ValueError(
            "velocity-Verlet requires a velocity-independent dynamics model; "
            "use RK4 for velocity-dependent dynamics such as post-Newtonian models"
        )

    detectors = tuple(event_detectors)
    steps = int(np.ceil(duration / dt))
    times = np.linspace(0.0, duration, steps + 1, dtype=np.float64)
    n = len(system.bodies)

    positions = np.empty((steps + 1, n, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    accelerations = np.empty_like(positions)

    initial = system.initial_state()
    masses = initial.masses
    state = np.concatenate((initial.positions, initial.velocities), axis=0)
    positions[0] = initial.positions
    velocities[0] = initial.velocities
    accelerations[0] = _accelerations(
        dynamics,
        float(times[0]),
        positions[0],
        velocities[0],
        masses,
        system,
    )

    detected_events: list[SimulationEvent] = []

    for step in range(steps):
        step_time = float(times[step])
        next_time = float(times[step + 1])
        step_dt = next_time - step_time
        previous_state = SystemState(positions[step], velocities[step], masses)

        if method == "rk4":
            state = _rk4_step(
                step_time,
                state,
                step_dt,
                masses,
                system,
                dynamics,
            )
            positions[step + 1] = state[:n]
            velocities[step + 1] = state[n:]
            accelerations[step + 1] = _accelerations(
                dynamics,
                next_time,
                positions[step + 1],
                velocities[step + 1],
                masses,
                system,
            )
        else:
            p, v, a = _velocity_verlet_step(
                step_time,
                positions[step],
                velocities[step],
                accelerations[step],
                step_dt,
                masses,
                system,
                dynamics,
            )
            positions[step + 1] = p
            velocities[step + 1] = v
            accelerations[step + 1] = a
            state = np.concatenate((p, v), axis=0)

        current_state = SystemState(
            positions[step + 1],
            velocities[step + 1],
            masses,
        )
        for detector in detectors:
            detected_events.extend(
                detector.detect(
                    step_time,
                    previous_state,
                    next_time,
                    current_state,
                    system,
                )
            )

    return Trajectory(
        times,
        positions,
        velocities,
        accelerations,
        system.names,
        events=tuple(detected_events),
    )

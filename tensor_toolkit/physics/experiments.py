"""Classical simulation experiments built on the physics layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from .diagnostics import (
    ConservationDiagnostics,
    EncounterDiagnostics,
    HyperbolicReference,
    TestParticleDiagnostics,
    conservation_diagnostics,
    encounter_diagnostics,
    hyperbolic_reference,
    test_particle_diagnostics,
)
from .dynamics import DynamicsModel
from .events import EventDetector
from .integrators import simulate
from .state import System
from .trajectory import Trajectory


@dataclass(frozen=True)
class SimulationExperiment:
    """Definition for a classical many-body simulation experiment."""

    name: str
    system: System
    duration: float
    dt: float
    method: str = "verlet"
    sample_every: int = 1
    encounters: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    dynamics: DynamicsModel | None = None
    event_detectors: tuple[EventDetector, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name must not be empty")
        if self.duration <= 0.0 or self.dt <= 0.0:
            raise ValueError("duration and dt must be positive")
        if self.sample_every < 1:
            raise ValueError("sample_every must be at least 1")
        names = set(self.system.names)
        for primary, probe in self.encounters:
            if primary not in names or probe not in names:
                raise ValueError("encounter body names must exist in the experiment system")
            if primary == probe:
                raise ValueError("encounter bodies must be different")


@dataclass(frozen=True)
class SimulationExperimentResult:
    name: str
    trajectory: Trajectory
    conservation: ConservationDiagnostics
    encounters: dict[str, EncounterDiagnostics]
    test_particles: dict[str, TestParticleDiagnostics]
    hyperbolic_references: dict[str, HyperbolicReference]
    metadata: dict[str, object]


def run_simulation_experiment(
    experiment: SimulationExperiment,
    *,
    gravitational_constant: float | None = None,
    softening: float = 0.0,
) -> SimulationExperimentResult:
    trajectory = simulate(
        experiment.system,
        duration=experiment.duration,
        dt=experiment.dt,
        method=experiment.method,
        gravitational_constant=gravitational_constant,
        softening=softening,
        dynamics=experiment.dynamics,
        event_detectors=experiment.event_detectors,
    )
    G = gravitational_constant
    conservation = conservation_diagnostics(
        trajectory,
        experiment.system.masses,
        **({} if G is None else {"gravitational_constant": G}),
        softening=softening,
    )
    encounters = {}
    test_particles = {}
    hyperbolic_references = {}
    masses_by_name = dict(zip(experiment.system.names, experiment.system.masses))

    for primary, probe in experiment.encounters:
        key = f"{primary}->{probe}"
        encounters[key] = encounter_diagnostics(
            trajectory,
            primary=primary,
            probe=probe,
        )
        if masses_by_name[probe] == 0.0 and masses_by_name[primary] > 0.0:
            test_particles[key] = test_particle_diagnostics(
                trajectory,
                primary=primary,
                probe=probe,
                primary_mass=float(masses_by_name[primary]),
                **({} if G is None else {"gravitational_constant": G}),
            )
            try:
                hyperbolic_references[key] = hyperbolic_reference(
                    trajectory,
                    primary=primary,
                    probe=probe,
                    primary_mass=float(masses_by_name[primary]),
                    **({} if G is None else {"gravitational_constant": G}),
                )
            except ValueError:
                pass

    return SimulationExperimentResult(
        name=experiment.name,
        trajectory=trajectory,
        conservation=conservation,
        encounters=encounters,
        test_particles=test_particles,
        hyperbolic_references=hyperbolic_references,
        metadata={
            "method": experiment.method,
            "duration": float(experiment.duration),
            "dt": float(experiment.dt),
            "body_names": experiment.system.names,
            "body_count": len(experiment.system.bodies),
            "sample_every": int(experiment.sample_every),
            "dynamics": (
                type(experiment.dynamics).__name__
                if experiment.dynamics is not None
                else "NewtonianGravity"
            ),
            "event_count": len(trajectory.events),
        },
    )


def save_simulation_experiment_result(
    result: SimulationExperimentResult,
    output_path,
) -> Path:
    """Save many-body trajectory data and diagnostics in a stable directory layout."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    stride = int(result.metadata.get("sample_every", 1))
    trajectory = result.trajectory
    np.savez_compressed(
        output / "trajectory.npz",
        times=trajectory.times[::stride],
        positions=trajectory.positions[::stride],
        velocities=trajectory.velocities[::stride],
        accelerations=trajectory.accelerations[::stride],
        body_names=np.asarray(trajectory.body_names),
    )

    metadata = dict(result.metadata)
    metadata["name"] = result.name
    metadata["conservation"] = {
        "energy_relative_drift": result.conservation.energy_relative_drift,
        "momentum_absolute_drift": result.conservation.momentum_absolute_drift,
        "angular_momentum_relative_drift": result.conservation.angular_momentum_relative_drift,
    }
    metadata["events"] = [
        {
            "time": event.time,
            "kind": event.kind,
            "bodies": list(event.bodies),
            "details": event.details,
        }
        for event in trajectory.events
    ]
    metadata["test_particles"] = {
        key: {
            "primary_name": value.primary_name,
            "probe_name": value.probe_name,
            "orbit_class": value.orbit_class,
            "specific_energy_initial": float(value.specific_energy[0]),
            "specific_energy_relative_drift": value.specific_energy_relative_drift,
            "specific_angular_momentum_initial": value.specific_angular_momentum[0].tolist(),
            "specific_angular_momentum_relative_drift": value.specific_angular_momentum_relative_drift,
            "eccentricity": value.eccentricity,
            "semi_major_axis": value.semi_major_axis,
            "periapsis_distance": value.periapsis_distance,
            "apoapsis_distance": value.apoapsis_distance,
            "orbital_period": value.orbital_period,
        }
        for key, value in result.test_particles.items()
    }
    metadata["hyperbolic_references"] = {
        key: {
            "primary_name": value.primary_name,
            "probe_name": value.probe_name,
            "eccentricity": value.eccentricity,
            "v_infinity": value.v_infinity,
            "periapsis_distance": value.periapsis_distance,
            "periapsis_speed": value.periapsis_speed,
            "asymptotic_deflection_angle": value.asymptotic_deflection_angle,
            "numerical_periapsis_distance_error": value.numerical_periapsis_distance_error,
            "numerical_periapsis_speed_error": value.numerical_periapsis_speed_error,
            "finite_window_deflection_error": value.finite_window_deflection_error,
        }
        for key, value in result.hyperbolic_references.items()
    }
    metadata["encounters"] = {
        key: {
            "primary_name": value.primary_name,
            "probe_name": value.probe_name,
            "closest_approach_time": value.closest_approach_time,
            "closest_approach_distance": value.closest_approach_distance,
            "periapsis_relative_speed": value.periapsis_relative_speed,
            "initial_relative_speed": value.initial_relative_speed,
            "final_relative_speed": value.final_relative_speed,
            "deflection_angle": value.deflection_angle,
        }
        for key, value in result.encounters.items()
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output

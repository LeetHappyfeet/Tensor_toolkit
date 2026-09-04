"""Classical simulation experiments built on the Newtonian physics layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from .diagnostics import ConservationDiagnostics, EncounterDiagnostics, conservation_diagnostics, encounter_diagnostics
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
    )
    G = gravitational_constant
    conservation = conservation_diagnostics(
        trajectory,
        experiment.system.masses,
        **({} if G is None else {"gravitational_constant": G}),
        softening=softening,
    )
    encounters = {
        f"{primary}->{probe}": encounter_diagnostics(
            trajectory,
            primary=primary,
            probe=probe,
        )
        for primary, probe in experiment.encounters
    }
    return SimulationExperimentResult(
        name=experiment.name,
        trajectory=trajectory,
        conservation=conservation,
        encounters=encounters,
        metadata={
            "method": experiment.method,
            "duration": float(experiment.duration),
            "dt": float(experiment.dt),
            "body_names": experiment.system.names,
            "body_count": len(experiment.system.bodies),
            "sample_every": int(experiment.sample_every),
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

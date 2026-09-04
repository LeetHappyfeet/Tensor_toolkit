"""Classical mechanics and trajectory interfaces for Tensor Toolkit."""

from .bridge import (
    MetricSamples,
    TensorEventSamples,
    sample_metric_along_trajectory,
    sample_tensors_along_trajectory,
    trajectory_spacetime_coordinates,
)
from .diagnostics import (
    ConservationDiagnostics,
    EncounterDiagnostics,
    conservation_diagnostics,
    encounter_diagnostics,
)
from .demos import demo_flyby_experiment, simulation_demos
from .experiments import (
    SimulationExperiment,
    SimulationExperimentResult,
    run_simulation_experiment,
    save_simulation_experiment_result,
)
from .gravity import newtonian_gravity_accelerations
from .integrators import simulate
from .orbits import circular_orbit_system, hyperbolic_flyby_system
from .state import Body, System
from .trajectory import Trajectory

__all__ = [
    "Body",
    "System",
    "Trajectory",
    "MetricSamples",
    "TensorEventSamples",
    "ConservationDiagnostics",
    "EncounterDiagnostics",
    "SimulationExperiment",
    "SimulationExperimentResult",
    "newtonian_gravity_accelerations",
    "conservation_diagnostics",
    "encounter_diagnostics",
    "circular_orbit_system",
    "hyperbolic_flyby_system",
    "demo_flyby_experiment",
    "simulation_demos",
    "run_simulation_experiment",
    "save_simulation_experiment_result",
    "simulate",
    "trajectory_spacetime_coordinates",
    "sample_metric_along_trajectory",
    "sample_tensors_along_trajectory",
]

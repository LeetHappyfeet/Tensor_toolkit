"""Built-in classical simulation demos."""

from __future__ import annotations

from .experiments import SimulationExperiment
from .orbits import hyperbolic_flyby_system


def demo_flyby_experiment() -> SimulationExperiment:
    """Jupiter-scale passive-probe flyby demo in SI units."""
    system = hyperbolic_flyby_system(
        primary_mass=1.89813e27,
        initial_distance=5.0e9,
        impact_parameter=7.0e8,
        incoming_speed=15_000.0,
        primary_name="jupiter",
        probe_name="probe",
    )
    return SimulationExperiment(
        name="demo-flyby",
        system=system,
        duration=700_000.0,
        dt=20.0,
        method="verlet",
        sample_every=50,
        encounters=(("jupiter", "probe"),),
    )


def simulation_demos() -> dict[str, SimulationExperiment]:
    return {
        "demo-flyby": demo_flyby_experiment(),
    }

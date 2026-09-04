"""Reusable Newtonian orbital and flyby initial-condition helpers."""

from __future__ import annotations
import numpy as np
from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT
from .state import Body, System

def circular_orbit_system(
    *,
    primary_mass: float,
    radius: float,
    probe_mass: float = 0.0,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
    primary_name: str = "primary",
    probe_name: str = "probe",
) -> System:
    """Return a primary at the origin and a probe in a circular +y orbit."""
    if primary_mass <= 0.0 or radius <= 0.0:
        raise ValueError("primary_mass and radius must be positive")
    speed = np.sqrt(float(gravitational_constant) * float(primary_mass) / float(radius))
    return System([
        Body(primary_name, primary_mass, [0, 0, 0], [0, 0, 0]),
        Body(probe_name, probe_mass, [radius, 0, 0], [0, speed, 0]),
    ])

def hyperbolic_flyby_system(
    *,
    primary_mass: float,
    initial_distance: float,
    impact_parameter: float,
    incoming_speed: float,
    probe_mass: float = 0.0,
    primary_name: str = "primary",
    probe_name: str = "probe",
) -> System:
    """Return a simple incoming flyby initial state in the xy plane.

    The probe begins at x=-initial_distance, y=impact_parameter and moves in +x.
    incoming_speed is the finite-distance initial speed, not asymptotic v-infinity.
    """
    if primary_mass <= 0.0 or initial_distance <= 0.0 or incoming_speed <= 0.0:
        raise ValueError("primary_mass, initial_distance, and incoming_speed must be positive")
    if impact_parameter == 0.0:
        raise ValueError("impact_parameter must be non-zero for a flyby")
    return System([
        Body(primary_name, primary_mass, [0, 0, 0], [0, 0, 0]),
        Body(probe_name, probe_mass, [-initial_distance, impact_parameter, 0], [incoming_speed, 0, 0]),
    ])

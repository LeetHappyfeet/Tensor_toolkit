"""Classical mechanics and trajectory interfaces for Tensor Toolkit."""

from .bridge import MetricSamples, sample_metric_along_trajectory, trajectory_spacetime_coordinates
from .gravity import newtonian_gravity_accelerations
from .integrators import simulate
from .state import Body, System
from .trajectory import Trajectory

__all__ = [
    "Body",
    "System",
    "Trajectory",
    "MetricSamples",
    "newtonian_gravity_accelerations",
    "simulate",
    "trajectory_spacetime_coordinates",
    "sample_metric_along_trajectory",
]

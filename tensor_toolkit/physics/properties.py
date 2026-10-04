"""Optional, validated physical properties for classical bodies.

These are configuration/identity values, not evolved constitutive laws.
In particular, spin, fuel and charge do not cause forces, torques, mass flow
or stress-energy without a separately selected DynamicsModel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64).copy()
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape-({size},) vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RotationProperties:
    """Quaternion orientation (w,x,y,z), angular velocity (rad/s), optional SI inertia."""
    orientation: np.ndarray = field(default_factory=lambda: (1., 0., 0., 0.))
    angular_velocity: np.ndarray = field(default_factory=lambda: (0., 0., 0.))
    inertia_tensor: np.ndarray | None = None

    def __post_init__(self):
        q = _vector(self.orientation, 4, "orientation")
        if not np.isclose(np.linalg.norm(q), 1., rtol=0., atol=1e-8):
            raise ValueError("orientation quaternion must be unit normalized")
        w = _vector(self.angular_velocity, 3, "angular_velocity")
        object.__setattr__(self, "orientation", q)
        object.__setattr__(self, "angular_velocity", w)
        if self.inertia_tensor is not None:
            tensor = np.asarray(self.inertia_tensor, dtype=np.float64).copy()
            if (tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)) or
                not np.allclose(tensor, tensor.T, atol=1e-10, rtol=0) or
                np.any(np.linalg.eigvalsh(tensor) <= 0)):
                raise ValueError("inertia_tensor must be finite symmetric positive-definite (kg m^2)")
            tensor.setflags(write=False)
            object.__setattr__(self, "inertia_tensor", tensor)


@dataclass(frozen=True)
class MaterialProperties:
    density_kg_m3: float | None = None
    restitution: float | None = None
    drag_coefficient: float | None = None
    cross_section_m2: float | None = None

    def __post_init__(self):
        for name in ("density_kg_m3", "drag_coefficient", "cross_section_m2"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.restitution is not None and (
            not np.isfinite(self.restitution) or not 0 <= self.restitution <= 1
        ):
            raise ValueError("coefficient of restitution must be in [0,1]")


@dataclass(frozen=True)
class ElectromagneticProperties:
    charge_coulombs: float = 0.0
    magnetic_dipole_am2: np.ndarray = field(default_factory=lambda: (0., 0., 0.))

    def __post_init__(self):
        if not np.isfinite(self.charge_coulombs):
            raise ValueError("charge_coulombs must be finite")
        object.__setattr__(self, "magnetic_dipole_am2",
                           _vector(self.magnetic_dipole_am2, 3, "magnetic_dipole_am2"))


@dataclass(frozen=True)
class PropulsionProperties:
    propellant_mass_kg: float = 0.0
    maximum_thrust_newtons: float = 0.0
    specific_impulse_seconds: float | None = None

    def __post_init__(self):
        for name in ("propellant_mass_kg", "maximum_thrust_newtons"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        isp = self.specific_impulse_seconds
        if isp is not None and (not np.isfinite(isp) or isp <= 0):
            raise ValueError("specific impulse must be finite and positive")


@dataclass(frozen=True)
class PhysicalProperties:
    """Independent components: absence is distinct from a zero-valued model."""
    rotation: RotationProperties | None = None
    material: MaterialProperties | None = None
    electromagnetic: ElectromagneticProperties | None = None
    propulsion: PropulsionProperties | None = None

    def __post_init__(self):
        for name, cls in (
            ("rotation", RotationProperties), ("material", MaterialProperties),
            ("electromagnetic", ElectromagneticProperties), ("propulsion", PropulsionProperties)
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, cls):
                raise TypeError(f"{name} must be {cls.__name__} or None")


__all__ = [
    "RotationProperties", "MaterialProperties", "ElectromagneticProperties",
    "PropulsionProperties", "PhysicalProperties"
]

"""Physical constants and Einstein-equation couplings used by Tensor Toolkit."""

from math import pi

SPEED_OF_LIGHT = 299_792_458.0  # m s^-1, exact
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2

# Einstein equation, with cosmological constant omitted:
# G_{mu nu} = KAPPA_SI T_{mu nu}
KAPPA_SI = 8.0 * pi * GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**4
KAPPA_GEOMETRIZED = 8.0 * pi

__all__ = [
    "SPEED_OF_LIGHT",
    "GRAVITATIONAL_CONSTANT",
    "KAPPA_SI",
    "KAPPA_GEOMETRIZED",
]

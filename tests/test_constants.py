from math import pi

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT, KAPPA_SI, SPEED_OF_LIGHT


def test_einstein_si_coupling_is_derived_from_constants():
    assert KAPPA_SI == 8.0 * pi * GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**4

#Hey Heads Up!. The constants will need to be updated shortly to deal with quantum physics. While they are constants, they actually aren't. This is an assumption.
# Called by met2den.py
from math import pi

def einE(E, gu, units):
    # Constants
    G = 6.674 * 10**-11 * units[2]**2 * units[1] / units[0]**3
    c = 2.99792 * 10**8 * units[2] / units[0]

    # Initialize energy density tensor
    enDen_ = [[0 for _ in range(4)] for _ in range(4)]

    # Calculate energy density tensor components
    for mu in range(4):
        for nu in range(4):
            enDen_[mu][nu] = c**4 / (8 * pi * G) * E[mu][nu]

    # Initialize contravariant form of energy density tensor
    enDen = [[0 for _ in range(4)] for _ in range(4)]

    # Calculate contravariant form of energy density tensor
    for mu in range(4):
        for nu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    enDen[mu][nu] += enDen_[alpha][beta] * gu[alpha][mu] * gu[beta][nu]

    return enDen

# Example usage:
# E = [[E11, E12, E13, E14], [E21, E22, E23, E24], [E31, E32, E33, E34], [E41, E42, E43, E44]]
# gu = [[gu11, gu12, gu13, gu14], [gu21, gu22, gu23, gu24], [gu31, gu32, gu33, gu34], [gu41, gu42, gu43, gu44]]
# units = [length_unit, mass_unit, time_unit]
# enDen = einE2(E, gu, units)

from solver.tools.c4Inv import c4Inv2
from solver.tools.ricciT import ricciT2
from solver.tools.ricciS import ricciS2
from solver.tools.einT import einT2
from solver.tools.einE import einE

def met2den2(metricTensor, delta=None, units=None):
    # Set default values if not provided
    if delta is None:
        delta = [1, 1, 1, 1]
    if units is None:
        units = [1, 1, 1]

    # Metric tensor and its inverse
    gl = metricTensor
    gu = c4Inv2(gl)

    # Calculate the Ricci tensor
    R_munu = ricciT2(gu, gl, delta)

    # Calculate the Ricci scalar
    R = ricciS2(R_munu, gu)

    # Calculate Einstein tensor
    E = einT2(R_munu, R, gl)

    # Calculate energy density
    energyDensity = einE2(E, gu, units)

    return energyDensity

# Example usage:
# energyDensity = met2den2(metricTensor)


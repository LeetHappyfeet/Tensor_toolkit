import numpy as np
from solver.tools.ricciS import ricciS2
from solver.tools.ricciT import ricciT2
from solver.tools.einT import einT2
from solver.tools.einE import einE
from solver.tools.c4Inv import c4Inv2

def met2den(gl, delta=None, units=None):
    """
    Coverts a Cartesian metric tensor to the corresponding energy density tensor using the Einstein Field Equations
    
    Parameters:
    gl : list of lists
        4x4 array. Elements of the array are 4-D matrices of double type, representing the metric tensor.
    delta : array-like, optional
        1x4 array of the uniform delta step size in each coordinate direction. Defaults to [1, 1, 1, 1].
    
    Returns:
    energy_density : list of lists
        4x4 array. Elements of the array are 4-D matrices of double type, representing the energy density tensor.
    """
    # Convert gl to numpy array
    gl_np = np.array(gl)
    
    # Handle default input arguments
    if delta is None:
        delta = [1, 1, 1, 1]
    if units is None:
        units = [1, 1, 1]
    
    # Calculate metric tensor inverse
    gu = c4Inv2(gl_np)
    
    # Calculate the Ricci tensor
    R_munu = ricciT2(gu, gl_np, delta)
    
    # Calculate the Ricci scalar
    R = ricciS2(R_munu, gu)
    
    # Calculate Einstein tensor
    E = einT2(R_munu, R, gl_np)
    
    # Calculate Energy density
    energy_density = einE(E, gu, units)
    
    return energy_density

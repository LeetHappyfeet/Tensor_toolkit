import numpy as np

def met2den(gl, delta=None):
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
    # Handle default input arguments
    if delta is None:
        delta = [1, 1, 1, 1]
    
    # Calculate metric tensor inverse
    gu = c4Inv(gl)
    
    # Calculate the Ricci tensor
    R_munu = ricciT(gu, gl, delta)
    
    # Calculate the Ricci scalar
    R = ricciS(R_munu, gu)
    
    # Calculate Einstein tensor
    E = einT(R_munu, R, gl)
    
    # Calculate Energy density
    energy_density = einE(E, gu)
    
    return energy_density

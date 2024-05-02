# Filename: setMinkowskiThreePlusOne.py
import numpy as np

def setMinkowskiThreePlusOne(gridSize):
    """
    Returns the 3+1 format for flat space.

    Args:
    gridSize (tuple): World size in [t, x, y, z].

    Returns:
    alpha (numpy.ndarray): Lapse rate 4D array.
    beta (list): Shift vector, 1x3 list of 4D arrays.
    gamma (list): Spatial terms, 3x3 list of 4D arrays.
    """

    # Lapse rate alpha
    alpha = np.ones(gridSize)

    # Shift vector beta
    beta = [np.zeros(gridSize) for _ in range(3)]

    # Spatial terms gamma
    gamma = [[np.ones(gridSize) if i == j else np.zeros(gridSize) for j in range(3)] for i in range(3)]

    return alpha, beta, gamma
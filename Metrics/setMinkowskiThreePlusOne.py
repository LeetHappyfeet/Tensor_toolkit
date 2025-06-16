# Filename: setMinkowskiThreePlusOne.py
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

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
    
    logging.info(f"Alpha shape: {alpha.shape}")
    for i in range(3):
        logging.info(f"Beta[{i}] shape: {beta[i].shape}")
    for i in range(3):
        for j in range(3):
            logging.info(f"Gamma[{i}][{j}] shape: {gamma[i][j].shape}")
            logging.info(f"Gamma[{i}][{j}] initial values: {gamma[i][j][0,0,0]}")

    return alpha, beta, gamma


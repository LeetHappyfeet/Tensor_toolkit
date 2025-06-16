# Filename: setMinkowski.py
import numpy as np

def setMinkowski(gridSize):
    """
    Builds metric terms for a flat Minkowski space.

    Args:
    gridSize (tuple): World size in [t, x, y, z].

    Returns:
    metric (dict): The metric tensor as a dictionary.
    """

    # Initialize the metric tensor as a dictionary
    metric = {}

    # dt^2 term
    metric[(1, 1)] = -np.ones(gridSize)

    # Non-time diagonal terms
    metric[(2, 2)] = np.ones(gridSize)
    metric[(3, 3)] = np.ones(gridSize)
    metric[(4, 4)] = np.ones(gridSize)

    # Cross terms
    metric[(1, 2)] = np.zeros(gridSize)
    metric[(2, 1)] = np.zeros(gridSize)
    metric[(1, 3)] = np.zeros(gridSize)
    metric[(3, 1)] = np.zeros(gridSize)
    metric[(2, 3)] = np.zeros(gridSize)
    metric[(3, 2)] = np.zeros(gridSize)
    metric[(2, 4)] = np.zeros(gridSize)
    metric[(4, 2)] = np.zeros(gridSize)
    metric[(3, 4)] = np.zeros(gridSize)
    metric[(4, 3)] = np.zeros(gridSize)
    metric[(1, 4)] = np.zeros(gridSize)
    metric[(4, 1)] = np.zeros(gridSize)

    return metric


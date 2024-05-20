# Filename: threePlusOneBuilder.py
import numpy as np
from Metrics.verifyTensor import verifyTensor

def threePlusOneBuilder(alpha, beta, gamma, threshold=1e-10):
    """
    Builds the metric tensor given input 3+1 components of alpha, beta, and gamma.

    Args:
    alpha (numpy.ndarray): Lapse rate map across spacetime.
    beta (list of numpy.ndarray): Covariant shift vector map across spacetime.
    gamma (list of list of numpy.ndarray): Covariant spatial term map across spacetime.
    threshold (float): Threshold value for the signature criterion.

    Returns:
    metricTensor (dict): Metric tensor represented as a dictionary.
    """

    # Set spatial components
    gamma_up = [np.linalg.inv(g) for g in gamma]

    # Find gridSize
    s = alpha.shape

    # Caluculate beta_i
    beta_up = [np.zeros(s) for _ in range(3)]
    for i in range(3):
        for j in range(3):
            beta_up[i] += gamma_up[i][j] * beta[j]

    # Create time-time component
    metricTensor = {(1, 1): -alpha ** 2}
    for i in range(3):
        metricTensor[(1, 1)] += beta_up[i] * beta[i]

    # Create time-space components
    for i in range(2, 5):
        metricTensor[(1, i)] = beta[i - 2]
        metricTensor[(i, 1)] = beta[i - 2]

    # Create space-space components
    for i in range(2, 5):
        for j in range(2, 5):
            metricTensor[(i, j)] = gamma[i - 2][j - 2]

    # Verify the metric tensor
    if not verifyTensor(metricTensor, threshold):
        raise ValueError("Constructed metric tensor does not satisfy symmetry and signature criteria.")

    return metricTensor

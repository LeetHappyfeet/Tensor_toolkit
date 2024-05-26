import numpy as np
from Metrics.verifyTensor import verifyTensor
import logging

logging.basicConfig(level=logging.INFO)

def is_symmetric(tensor, tol=1e-10):
    for (i, j), value in tensor.items():
        if not np.allclose(value, tensor.get((j, i), np.zeros_like(value)), atol=tol):
            logging.error(f"Tensor component {(i, j)} is not symmetric with {(j, i)}.")
            return False
    return True

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
    
    logging.info(f"Shape of alpha: {alpha.shape}")
    for i in range(3):
        logging.info(f"Shape of beta[{i}]: {beta[i].shape}")
    
    for i in range(3):
        for j in range(3):
            logging.info(f"Shape of gamma[{i}][{j}]: {gamma[i][j].shape}")
            logging.info(f"gamma[{i}][{j}] sample values: {gamma[i][j][0,0,0]}")

    gamma_up = []

    for idx, g in enumerate(gamma):
        dets = np.linalg.det(g)
        logging.info(f"Determinant of gamma[{idx}]: {dets}")
        
        if np.any(np.abs(dets) < threshold):
            logging.warning(f"Singular matrix detected at gamma[{idx}] with determinant {dets}. Using pseudoinverse instead.")
            gamma_up.append(np.linalg.pinv(g))  # Use pseudo-inverse for singular matrices
        else:
            gamma_up.append(np.linalg.inv(g))

    # Find gridSize
    s = alpha.shape

    # Calculate beta_i
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

    # Log the metric tensor before verification
    logging.info(f"Constructed metric tensor: {metricTensor}")

    # Verify the symmetry of the metric tensor
    if not is_symmetric(metricTensor, threshold):
        logging.error("Constructed metric tensor is not symmetric.")
        raise ValueError("Constructed metric tensor does not satisfy symmetry criteria.")

    # Verify the metric tensor using external function
    if not verifyTensor(metricTensor, threshold):
        logging.error(f"Metric tensor failed verification: {metricTensor}")
        raise ValueError("Constructed metric tensor does not satisfy signature criteria.")

    return metricTensor
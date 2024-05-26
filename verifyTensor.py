# Filename: verifyTensor.py
import numpy as np

def verifyTensor(metricTensor, threshold):
    """
    Verifies the metric tensor based on symmetry and signature criteria.

    Args:
    metricTensor (dict): Metric tensor represented as a dictionary.
    threshold (float): Threshold value for the signature criterion.

    Returns:
    bool: True if the metric tensor is verified, False otherwise.
    """
    # Convert the metric tensor dictionary to a numpy array
    tensor = np.array([[metricTensor.get((i, j), 0) for j in range(1, 5)] for i in range(1, 5)])

    # Check if the tensor is symmetric
    if not np.array_equal(tensor, tensor.T):
        return False

    # Check the signature of the metric tensor
    diag_elems = np.diag(tensor)
    if any(diag_elems < -threshold) or any(diag_elems > threshold):
        return False

    return True

import numpy as np
from scipy.sparse import coo_matrix
import logging

def verifyTensor(metricTensor, threshold=1e-10):
    """
    Verifies that the metric tensor meets the expected criteria.

    Args:
    metricTensor (dict): Metric tensor represented as a dictionary.
    threshold (float): Threshold value for the signature criterion.

    Returns:
    bool: True if the metric tensor is valid, False otherwise.
    """
    logging.info(f"Verifying metric tensor: {metricTensor}")

    rows, cols, data = [], [], []
    for key, value in metricTensor.items():
        if len(key) != 2:
            logging.error(f"Unexpected key in metricTensor: {key}")
            raise ValueError(f"Unexpected key in metricTensor: {key}")

        (i, j) = key
        flat_value = value.flatten()
        index_shift = (i-1) * 4 + (j-1)
        for idx, val in enumerate(flat_value):
            rows.append(index_shift)
            cols.append(idx)
            data.append(val)

    tensor_sparse = coo_matrix((data, (rows, cols)), shape=(16, len(data) // 16))

    tol = max(threshold, 1e-10 * np.max(np.abs(data)))
    for (i, j), value in metricTensor.items():
        if not np.allclose(value, metricTensor.get((j, i), np.zeros_like(value)), atol=tol):
            return False

    return True

# Filename: getScalars.py
import numpy as np
import tensorflow as tf
from tensor_utils import c3_inv, change_tensor_index

def three_plus_one_decomposer(metric):
    """
    Finds 3+1 terms from the metric tensor.

    Args:
    metric (dict): Dictionary containing the metric tensor.

    Returns:
    alpha (numpy.ndarray): 4D array representing the lapse rate.
    beta_down (list): List of 3 4D arrays representing the covariant shift vectors.
    gamma_down (list): List of 3x3 4D arrays representing the covariant spatial terms.
    beta_up (list): List of 3 4D arrays representing the contravariant shift vectors.
    gamma_up (list): List of 3x3 4D arrays representing the contravariant spatial terms.
    """

    # Check that the metric is covariant and change index if not
    metric = change_tensor_index(metric, "covariant")

    # Covariant shift vector maps to the covariant tensor terms g_0i
    beta_down = [metric['tensor'][0][i+1] for i in range(3)]

    # Covariant gamma maps to the covariant tensor terms g_ij
    gamma_down = [[metric['tensor'][i+1][j+1] for j in range(3)] for i in range(3)]

    # Transform gamma to contravariant
    gamma_up = c3_inv(gamma_down)

    # Find the world grid size
    s = np.shape(metric['tensor'][0][0])

    # Transform beta to contravariant
    beta_up = []
    for i in range(3):
        temp = np.zeros(s)
        for j in range(3):
            temp += gamma_up[i][j] * beta_down[j]
        beta_up.append(temp)

    # Find lapse using beta and g_00
    alpha = tf.sqrt(beta_up[0] * beta_down[0] + beta_up[1] * beta_down[1] + beta_up[2] * beta_down[2] - metric['tensor'][0][0])

    return alpha, beta_down, gamma_down, beta_up, gamma_up


# Filename: getInnerProduct.py
import numpy as np
from verifyTensor import verifyTensor

##ensure that you have defined the verifyTensor and c4Inv functions or integrate their functionalities within this code.
##Also, make sure that the input vecA, vecB, and Metric are dictionaries containing the appropriate keys.

def getInnerProduct(vecA, vecB, Metric):
    """
    GETINNERPRODUCT: takes the inner product of two vector fields with their metric

    Args:
    vecA (dict): Dictionary containing a vector field represented by a list of arrays.
    vecB (dict): Dictionary containing a vector field represented by a list of arrays.
    Metric (dict): Dictionary containing a metric tensor represented by a 4x4 array.

    Returns:
    innerprod (numpy.ndarray): Inner product of the two vector fields.
    """

    # Verify Metric
    if not verifyTensor(Metric, 1):
        raise ValueError("Metric is not verified. Please verify metric using verifyTensor(metric).")

    s = np.shape(Metric['tensor'][0][0])
    innerprod = np.zeros(s)

    if vecA['index'] != vecB['index']:
        for mu in range(4):
            for nu in range(4):
                innerprod += vecA['field'][mu] * vecB['field'][nu]

    elif vecA['index'] == vecB['index']:
        if vecA['index'] == Metric['index']:
            Metric['tensor'] = c4Inv(Metric['tensor'])  # flip index
        for mu in range(4):
            for nu in range(4):
                innerprod += vecA['field'][mu] * vecB['field'][nu] * Metric['tensor'][mu][nu]

    return innerprod

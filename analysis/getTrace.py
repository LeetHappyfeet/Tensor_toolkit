# Filename: getTrace
import numpy as np
from solver.verifyTensor import verifyTensor
from solver.tools.c4Inv import c4Inv2 as c4Inv

def getTrace(tensor, metric):
    """
    GETTRACE: take the trace of a tensor

    Args:
    tensor (dict): Dictionary containing a tensor represented by a 4x4 array.
    metric (dict): Dictionary containing a metric tensor represented by a 4x4 array.

    Returns:
    Trace (numpy.ndarray): Trace of the tensor.
    """

    # Verify metric
    if not verifyTensor(metric, 1):
        raise ValueError("Metric is not verified. Please verify metric using verifyTensor(metric).")

    Trace = np.zeros_like(metric['tensor'][0][0])

    if tensor['index'] == metric['index']:
        metric['tensor'] = c4Inv(metric['tensor'])

    for a in range(4):
        for b in range(4):
            Trace += metric['tensor'][a][b] * tensor['tensor'][a][b]

    return Trace


# Filename: getEvenPointsOnSphere.py
import numpy as np

def getEvenPointsOnSphere(R, numberOfPoints, useGPU=False):
    """
    GETEVENPOINTSONSPHERE: Generate evenly distributed points on a sphere.

    Args:
    R (float): Radius of the sphere.
    numberOfPoints (int): Number of points to generate.
    useGPU (bool): Flag indicating whether to use GPU computation (default is False).

    Returns:
    Vector (numpy.ndarray): Array of shape (3, numberOfPoints) containing the generated points.
    """

    if useGPU:
        goldenRatio = (1 + 5 ** 0.5) / 2
        Vector = np.zeros((3, numberOfPoints), dtype=np.float32)
    else:
        goldenRatio = (1 + 5 ** 0.5) / 2
        Vector = np.zeros((3, numberOfPoints), dtype=np.float64)

    for i in range(numberOfPoints):
        theta = 2 * np.pi * i / goldenRatio
        phi = np.arccos(1 - 2 * (i + 0.5) / numberOfPoints)

        Vector[0, i] = R * np.cos(theta) * np.sin(phi)
        Vector[1, i] = R * np.sin(theta) * np.sin(phi)
        Vector[2, i] = R * np.cos(phi)

    return Vector
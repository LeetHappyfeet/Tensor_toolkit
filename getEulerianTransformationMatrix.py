# Filename: getEulerianTransformationMatrix.py
import numpy as np

def get_eulerian_transformation_matrix(g):
    if g.ndim == 2:  # For simple 4x4 metrics
        assert np.array_equal(g, g.T), "g is not symmetric"

        factor0 = g[3, 3]
        factor1 = (-g[2, 3] ** 2 + g[2, 2] * factor0)
        factor2 = (2 * g[1, 2] * g[1, 3] * g[2, 3] - g[3, 3] * g[1, 2] ** 2 - g[2, 2] * g[1, 3] ** 2 + g[1, 1] * factor1)
        factor3 = (-2 * g[3, 3] * g[0, 1] * g[0, 2] * g[1, 2] + 2 * g[0, 2] * g[0, 3] * g[1, 2] * g[2, 3] +
                   2 * g[0, 1] * g[0, 2] * g[1, 3] * g[2, 3] + 2 * g[0, 1] * g[0, 3] * g[1, 2] * g[2, 3] -
                   g[0, 1] ** 2 * g[2, 3] ** 2 - g[0, 2] ** 2 * g[1, 3] ** 2 - g[0, 3] ** 2 * g[1, 2] ** 2 +
                   g[2, 2] * (-2 * g[0, 1] * g[0, 3] * g[2, 3] + g[3, 3] * g[0, 1] ** 2) +
                   g[1, 1] * (-2 * g[0, 2] * g[0, 3] * g[2, 3] + g[3, 3] * g[0, 2] ** 2 + g[2, 2] * g[0, 3] ** 2) -
                   g[0, 0] * factor2)

        M = np.zeros((4, 4))
        M[0, 0] = np.sqrt(factor2 / factor3)
        M[1, 0] = (g[0, 1] * g[2, 3] ** 2 + g[0, 2] * g[1, 2] * g[3, 3] - g[0, 2] * g[1, 3] * g[2, 3] -
                   g[0, 3] * g[1, 2] * g[2, 3] + g[0, 3] * g[1, 3] * g[2, 2] - g[0, 1] * g[2, 2] * g[3, 3]) / np.sqrt(
            factor2 * factor3)
        M[2, 0] = (g[0, 2] * g[1, 3] ** 2 - g[0, 3] * g[1, 2] * g[1, 3] + g[0, 1] * g[1, 2] * g[3, 3] -
                   g[0, 1] * g[1, 3] * g[2, 3] - g[0, 2] * g[1, 1] * g[3, 3] + g[0, 3] * g[1, 1] * g[2, 3]) / np.sqrt(
            factor2 * factor3)
        M[3, 0] = (g[0, 3] * g[1, 2] ** 2 - g[0, 2] * g[1, 2] * g[1, 3] - g[0, 1] * g[1, 2] * g[2, 3] +
                   g[0, 1] * g[1, 3] * g[2, 2] + g[0, 2] * g[1, 1] * g[2, 3] - g[0, 3] * g[1, 1] * g[2, 2]) / np.sqrt(
            factor2 * factor3)

        M[1, 1] = (g[2, 3] ** 2 - g[1, 2] ** 2 - g[1, 3] ** 2 + g[1, 1] * factor0) / np.sqrt(factor0 * factor2)
        M[2, 1] = (-g[1, 2] * g[1, 3] + g[1, 1] * g[2, 3]) / np.sqrt(factor0 * factor2)
        M[3, 1] = (-g[1, 3] * g[2, 3] + g[1, 2] * factor0) / np.sqrt(factor0 * factor2)

        M[2, 2] = np.sqrt(factor0 / factor1)
        M[3, 2] = (-g[2, 3]) / np.sqrt(factor0 * factor1)

        M[3, 3] = np.sqrt(factor0)

   

    return M

# Example usage:
g = np.array([[1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, -1, 0],
              [0, 0, 0, -1]])

M = get_eulerian_transformation_matrix(g)
print(M)
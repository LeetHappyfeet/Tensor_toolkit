import numpy as np

def redblue(value, gradient_num=1024):
    """
    Generates a red-blue color map based on the input values.

    Parameters:
        value (numpy.ndarray): Input values.
        gradient_num (int): Number of points in the gradient. Default is 1024.

    Returns:
        numpy.ndarray: Red-blue color map.
    """
    min_value = np.min(value)
    max_value = np.max(value)

    if not (min_value <= 0 and max_value >= 0):
        if min_value > 0 and max_value > 0:
            return np.vstack([np.linspace(1, 0, round(gradient_num)),
                              np.linspace(1, 0, round(gradient_num)),
                              np.ones(round(gradient_num))]).T
        if min_value < 0 and max_value < 0:
            return np.vstack([np.ones(round(gradient_num)),
                              np.linspace(0, 1, round(gradient_num)),
                              np.linspace(0, 1, round(gradient_num))]).T

    if min_value == 0 and max_value == 0:
        return np.array([[1, 1, 1]])

    center_val = abs(max_value) / (abs(max_value) + abs(min_value))
    first_half = np.vstack([np.ones(round((1 - center_val) * gradient_num)),
                            np.linspace(0, 1, round((1 - center_val) * gradient_num)),
                            np.linspace(0, 1, round((1 - center_val) * gradient_num))]).T
    second_half = np.vstack([np.linspace(1, 0, round(center_val * gradient_num)),
                             np.linspace(1, 0, round(center_val * gradient_num)),
                             np.ones(round(center_val * gradient_num))]).T

    return np.vstack((first_half, second_half))

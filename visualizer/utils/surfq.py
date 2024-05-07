import numpy as np
import matplotlib.pyplot as plt

def surfq(*args):
    """
    Surfs the squeeze of the array.

    Parameters:
        *args: Variable number of input arguments. Must provide at least one array.

    Returns:
        None
    """
    if len(args) >= 3 and len(args[0]) > 3 and len(args[1]) > 3 and len(args[2]) > 3 and \
            isinstance(args[1], np.ndarray) and isinstance(args[2], np.ndarray):
        plt.figure()
        plt.gca(projection='3d').plot_surface(args[0], args[1], np.squeeze(args[2]), *args[3:])
        plt.colormaps(redblue(np.squeeze(args[2])))
    else:
        plt.figure()
        plt.gca(projection='3d').plot_surface(np.squeeze(args[0]).T, *args[1:])
        plt.colormaps(redblue(np.squeeze(args[0])))

    plt.show()

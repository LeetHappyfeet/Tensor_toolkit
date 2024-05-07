import matplotlib.pyplot as plt

def plotq(*args):
    """
    Plots the squeeze of the array.

    Parameters:
        *args: Variable number of arguments, where the first argument is the array to be squeezed,
               and the rest are optional arguments for the plot function.

    Returns:
        None
    """
    if len(args) >= 2 and len(args[0]) > 3 and len(args[1]) > 3:
        plt.plot(*[squeeze(arg) for arg in args], args[2:])
    else:
        plt.plot(*[squeeze(arg) for arg in args])

def squeeze(array):
    """
    Squeezes the input array.

    Parameters:
        array (numpy.ndarray): Input array to be squeezed.

    Returns:
        squeezed_array (numpy.ndarray): Squeezed array.
    """
    return array.squeeze()

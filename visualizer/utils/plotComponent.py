import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_component(array, title_text, x_label_text, y_label_text, alpha):
    """
    Plots a component of the tensor array.

    Parameters:
        array (numpy.ndarray): The component of the tensor array to plot.
        title_text (str): Title of the plot.
        x_label_text (str): Label for the x-axis.
        y_label_text (str): Label for the y-axis.
        alpha (float): Alpha value of the surface grid display from 0 to 1.

    Returns:
        None
    """
    # Create a new figure
    plt.figure()

    # Plot the surface
    plt_surf = plt.imshow(array, cmap=ListedColormap(redblue(array)), aspect='auto', interpolation='none', alpha=alpha)

    # Set title and labels
    plt.title(title_text)
    plt.xlabel(x_label_text)
    plt.ylabel(y_label_text)

    # Set the color limits
    plt.colorbar(plt_surf)

    # Show the plot
    plt.show()

def redblue(array):
    """
    Custom colormap function.

    Parameters:
        array (numpy.ndarray): Input array for colormap scaling.

    Returns:
        cmap (list): List of RGBA color tuples.
    """
    # Define your custom colormap here, based on the input array
    cmap = ['blue', 'white', 'red']
    return cmap

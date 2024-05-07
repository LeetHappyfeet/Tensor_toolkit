import numpy as np
import matplotlib.pyplot as plt

def plot_tensor(tensor, alpha=0.2, sliced_planes=[1, 4], slice_locations=None):
    """
    Plots the unique elements of the tensor based on the slice plane.

    Parameters:
        tensor (dict): The tensor struct object representing either a metric or stress-energy tensor.
        alpha (float): Alpha value of the surface grid display, ranging from 0 to 1.
        sliced_planes (list): Coordinates that are sliced, represented by index values from 1 to 4.
        slice_locations (list): Location of the slice in the specified coordinates.

    Returns:
        None
    """
    # Handle default input arguments
    if slice_locations is None:
        slice_locations = []

    # Verify tensor
    if not verify_tensor(tensor, suppress_msgs=True):
        raise ValueError("Tensor is not verified. Please verify tensor using verify_tensor(tensor).")

    # Check that the sliced planes are different
    if sliced_planes[0] == sliced_planes[1]:
        raise ValueError("Selected planes must not be the same. Select two different planes to slice along.")

    # Round slice locations
    slice_locations = [round(loc) for loc in slice_locations]

    # Check that the slice locations are inside the world
    if any(loc < 1 for loc in slice_locations) or \
            any(loc > tensor['tensor'][0][0].shape[dim - 1] for dim, loc in zip(sliced_planes, slice_locations)):
        raise ValueError("Slice locations are outside the world.")

    # Check tensor type
    if tensor['type'].lower() == "metric":
        title_character = "g"
    elif tensor['type'].lower() == "stress-energy":
        title_character = "T"
    else:
        raise ValueError("Unknown tensor type. Supported types are 'Metric' and 'Stress-Energy'.")

    # Check tensor index
    index_map = {
        "covariant": ["_{", "}"],
        "contravariant": ["^{", "}"],
        "mixedupdown": ["^{", "}_{ "],
        "mixeddownup": ["_{", "}^{ "]
    }
    if tensor['index'].lower() not in index_map:
        raise ValueError("Unknown tensor index. Supported indices are 'covariant', 'contravariant', 'mixedupdown', "
                         "and 'mixeddownup'.")
    title_augment1, title_augment2 = index_map[tensor['index'].lower()]

    # Check that the coords are cartesian
    if tensor['coords'].lower() == "cartesian":
        # Label cartesian axis
        x_label_text, y_label_text = label_cartesian_axis(sliced_planes)

        # Get slice data
        idx = get_slice_data(sliced_planes, slice_locations, tensor)

        # Plot components
        for i in range(len(idx)):
            component_title = f"{title_character}{title_augment1}{idx[i][0]}{title_augment2}{idx[i][1]}"
            plot_component(tensor['tensor'][idx[i][0] - 1][idx[i][1] - 1][idx[i][2] - 1][idx[i][3] - 1],
                           component_title, x_label_text, y_label_text, alpha)
    else:
        raise ValueError("Unknown coordinate system. Must be 'cartesian'.")

def verify_tensor(tensor, suppress_msgs=False):
    """
    Verifies the metric tensor and stress energy tensor structs.

    Parameters:
        tensor (dict): The tensor struct object.
        suppress_msgs (bool): Whether to suppress warning messages. Defaults to False.

    Returns:
        verified (bool): True if tensor is verified, False otherwise.
    """
    # Implementation of verify_tensor function
    pass

def label_cartesian_axis(sliced_planes):
    """
    Label cartesian axis based on sliced planes.

    Parameters:
        sliced_planes (list): Coordinates that are sliced.

    Returns:
        x_label_text (str): Label for x-axis.
        y_label_text (str): Label for y-axis.
    """
    # Implementation of label_cartesian_axis function
    pass

def get_slice_data(sliced_planes, slice_locations, tensor):
    """
    Get slice data based on sliced planes and locations.

    Parameters:
        sliced_planes (list): Coordinates that are sliced.
        slice_locations (list): Location of the slice in the specified coordinates.
        tensor (dict): The tensor struct object.

    Returns:
        slice_data (list): List of tuples containing slice data.
    """
    # Implementation of get_slice_data function
    pass

def plot_component(data, title, x_label, y_label, alpha):
    """
    Plot a component of the tensor.

    Parameters:
        data (numpy.ndarray): The data to be plotted.
        title (str): Title of the plot.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        alpha (float): Alpha value of the surface grid display, ranging from 0 to 1.

    Returns:
        None
    """
    # Implementation of plot_component function
    pass

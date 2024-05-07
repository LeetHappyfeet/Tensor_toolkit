def label_cartesian_axis(plane):
    """
    Labels the Cartesian axes based on the specified plane.

    Parameters:
        plane (list): Coordinates of the slice plane, represented by index values from 1 to 4.

    Returns:
        xlabel_name (str): Label for the x-axis.
        ylabel_name (str): Label for the y-axis.
    """
    labels = ["t", "x", "y", "z"]
    shown_planes = sorted(set(range(1, 5)) - set(plane))

    xlabel_name = labels[shown_planes[0] - 1]
    ylabel_name = labels[shown_planes[1] - 1]

    return xlabel_name, ylabel_name

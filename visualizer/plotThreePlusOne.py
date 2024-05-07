def plot_three_plus_one(metric, sliced_planes=[1, 4], slice_locations=None, alpha=0.2):
    """
    Plots the 3+1 elements of the metric tensor based on the slice plane.

    Parameters:
        metric (dict): The metric tensor struct object.
        sliced_planes (list): Coordinates that are sliced, represented by index values from 1 to 4.
        slice_locations (list): Location of the slice in the specified coordinates.
        alpha (float): Alpha value of the surface grid display, ranging from 0 to 1.

    Returns:
        None
    """
    # Handle input arguments
    if slice_locations is None:
        slice_locations = []

    # Verify tensor
    if not verify_tensor(metric, suppress_msgs=True):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")

    # Check that the sliced planes are different
    if sliced_planes[0] == sliced_planes[1]:
        raise ValueError("Selected planes must not be the same. Select two different planes to slice along.")

    # Round slice locations
    slice_locations = [round(loc) for loc in slice_locations]

    # Check that the slice locations are inside the world
    if any(loc < 1 for loc in slice_locations) or \
            any(loc > metric['tensor'][0][0].shape[dim - 1] for dim, loc in zip(sliced_planes, slice_locations)):
        raise ValueError("Slice locations are outside the world.")

    # Check that the coords are cartesian
    if metric['coords'].lower() == "cartesian":
        # Decompose the metric tensor
        alpha_lapse, beta_down, gamma_down, _, _ = three_plus_one_decomposer(metric)

        # Label cartesian axis
        x_label_text, y_label_text = label_cartesian_axis(sliced_planes)

        # Get slice data
        idx = get_slice_data(sliced_planes, slice_locations, metric)

        # Plot alpha
        title_text = r"$\alpha$"
        plot_component(np.squeeze(alpha_lapse[idx[0], idx[1], idx[2], idx[3]]), title_text, x_label_text, y_label_text, alpha)

        # Plot beta
        for i in range(3):
            title_text = r"$\beta_{}$".format(i+1)
            plot_component(np.squeeze(beta_down[i][idx[0], idx[1], idx[2], idx[3]]), title_text, x_label_text, y_label_text, alpha)

        # Plot gamma
        for c in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]:
            title_text = r"$\gamma_{{{}}}$".format(c[0]) + "{}".format(c[1])
            plot_component(np.squeeze(gamma_down[c[0] - 1][c[1] - 1][idx[0], idx[1], idx[2], idx[3]]), title_text, x_label_text, y_label_text, alpha)

    else:
        raise ValueError("Unknown coordinate system. Must be 'cartesian'.")

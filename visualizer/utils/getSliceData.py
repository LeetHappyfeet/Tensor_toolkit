def get_slice_data(plane, slice_center, tensor):
    """
    Retrieves slice data based on the specified slice plane and center.

    Parameters:
        plane (list): Coordinates of the slice plane, represented by index values from 1 to 4.
        slice_center (list): Location of the slice center in the specified coordinates.
        tensor (dict): The tensor object.

    Returns:
        index_data (list): Slice data indices.
    """
    s = tensor['tensor'][0][0].shape  # assuming all tensors in the tensor struct have the same shape
    index_data = [list(range(1, dim + 1)) for dim in s]

    index_data[plane[0] - 1] = slice_center[0]
    index_data[plane[1] - 1] = slice_center[1]

    return index_data

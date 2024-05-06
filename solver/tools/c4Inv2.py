import numpy as np

def c4Inv2(cellArray):
    # Check if the input cell array is of size 4x4
    assert cellArray.shape == (4, 4), "Cell array is not 4x4"
    
    # Convert cell array to numpy array for easier indexing
    r = cellArray
    
    # Calculate determinant
    det = np.linalg.det(r)
    
    # Calculate elements of the inverse matrix
    invCellArray = np.empty_like(r, dtype=float)
    for i in range(4):
        for j in range(4):
            minor = np.delete(np.delete(r, i, axis=0), j, axis=1)
            cofactor = (-1) ** (i + j) * np.linalg.det(minor)
            invCellArray[j, i] = cofactor / det
    
    return invCellArray

# Example usage:
# cellArray = np.array([[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]])
# invArray = c4Inv2(cellArray)

import numpy as np

def cDet2(cellArray):
    # Get the dimensions of the cell array
    h, w = len(cellArray), len(cellArray[0])
    
    if h == 2 and w == 2:
        # Base case for 2x2 matrix
        cellDet = cellArray[0][0] * cellArray[1][1] - cellArray[0][1] * cellArray[1][0]
        return cellDet
    
    cellDet = 0
    for i in range(h):
        # Create a subarray by removing the first row and the i-th column
        subArray = [row[:i] + row[i+1:] for row in cellArray[1:]]
        
        # Calculate the determinant of the subarray recursively
        subDet = cDet2(subArray)
        
        # Add the contribution of the current element to the determinant
        cellDet += (2 * (i % 2) - 1) * cellArray[0][i] * subDet
    
    return cellDet

# Example usage:
# cellArray = [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]
# det = cDet2(cellArray)

import tensorflow as tf

def c4Inv2(cellArray):
    # Convert cell array to TensorFlow tensor
    r = tf.constant(cellArray, dtype=tf.float64)
    
    # Calculate determinant
    det = tf.linalg.det(r)
    
    # Calculate elements of the inverse matrix
    invCellArray = tf.linalg.inv(r)
    
    return invCellArray

# Example usage:
# cellArray = np.array([[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]])
# invArray = c4Inv2(cellArray)

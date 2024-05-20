import numpy as np
import tensorflow as tf
from Metrics.setMinkowskiThreePlusOne import setMinkowskiThreePlusOne
from Metrics.threePlusOneBuilder import threePlusOneBuilder

tf.compat.v1.disable_eager_execution()
# Define shape function for Alcubierre metric
def shapeFunction_Alcubierre(r, R, sigma):
    return tf.exp(-((r - R) / sigma) ** 2)

# Define function to build the Alcubierre metric in a Galilean comoving frame
def metricGet_AlcubierreComoving(gridSize, worldCenter, v, R, sigma, gridScale):
    # Ensure worldCenter has enough elements
    if len(worldCenter) < 4:
        raise ValueError("worldCenter must have at least 4 elements.")

    # Initialize TensorFlow session
    with tf.compat.v1.Session() as sess:
        # Define placeholders for r, R, and sigma
        r_placeholder = tf.compat.v1.placeholder(tf.float32)
        R_placeholder = tf.compat.v1.placeholder(tf.float32)
        sigma_placeholder = tf.compat.v1.placeholder(tf.float32)

        # Compute shape function
        shape_function = shapeFunction_Alcubierre(r_placeholder, R_placeholder, sigma_placeholder)

        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Define values for r, R, and sigma
        r_value = 5.0  # provide appropriate value
        R_value = 10.0  # provide appropriate value
        sigma_value = 2.0  # provide appropriate value

        # Evaluate shape function
        result = sess.run(shape_function, feed_dict={r_placeholder: r_value, R_placeholder: R_value, sigma_placeholder: sigma_value})

        # Declare a Minkowski space
        alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)

        # Add the Alcubierre modification
        t = 0  # only one timeslice is used
        for i in range(gridSize[1]):
            for j in range(gridSize[2]):
                for k in range(gridSize[3]):
                    print(f"Processing: i={i}, j={j}, k={k}")  # Monitoring output
                    # Find grid center x, y, z
                    x = (i + 1) * gridScale[1] - worldCenter[1]
                    y = (j + 1) * gridScale[2] - worldCenter[2]
                    z = (k + 1) * gridScale[3] - worldCenter[3]

                    # Find the radius from the center of the bubble
                    r_value = np.sqrt(x**2 + y**2 + z**2)

                    # Evaluate shape function using placeholders
                    fs_value = sess.run(shape_function, feed_dict={r_placeholder: r_value, R_placeholder: R, sigma_placeholder: sigma})

                    # Add alcubierre modification to beta
                    beta[0][t, i, j, k] = v * (1 - fs_value)

        # Make tensor from the 3+1 functions
        metric = {}
        metric['params'] = {
            'gridSize': gridSize,
            'worldCenter': worldCenter,
            'velocity': v,
            'R': R,
            'sigma': sigma
        }
        metric['type'] = "metric"
        metric['name'] = 'Alcubierre Comoving'
        metric['scaling'] = gridScale
        metric['coords'] = "cartesian"
        metric['index'] = "covariant"
        metric['tensor'] = threePlusOneBuilder(alpha, beta, gamma)

        return metric
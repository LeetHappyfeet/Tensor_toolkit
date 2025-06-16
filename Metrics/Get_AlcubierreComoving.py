import numpy as np
import tensorflow as tf
import logging
import traceback
from Metrics.setMinkowskiThreePlusOne import setMinkowskiThreePlusOne
from Metrics.threePlusOneBuilder import threePlusOneBuilder

logging.basicConfig(level=logging.INFO, filename='singularities.log', filemode='w')

tf.compat.v1.disable_eager_execution()

# Define shape function for Alcubierre metric
def shapeFunction_Alcubierre(r, R, sigma):
    return tf.exp(-((r - R) / sigma) ** 2)

# Define function to build the Alcubierre metric in a Galilean comoving frame
def metricGet_AlcubierreComoving(gridSize, worldCenter, v, R, sigma, gridScale):
    try:
        # Ensure worldCenter has enough elements
        if len(worldCenter) < 4:
            raise ValueError("worldCenter must have at least 4 elements.")

        # Initialize TensorFlow session
        with tf.compat.v1.Session() as sess:
            # Define placeholders for r, R, and sigma
            r_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None,))
            R_placeholder = tf.compat.v1.placeholder(tf.float32)
            sigma_placeholder = tf.compat.v1.placeholder(tf.float32)

            # Compute shape function
            shape_function = shapeFunction_Alcubierre(r_placeholder, R_placeholder, sigma_placeholder)

            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())

            # Collect all r_value values
            r_values = []
            for i in range(gridSize[1]):
                for j in range(gridSize[2]):
                    for k in range(gridSize[3]):
                        x = (i + 1) * gridScale[1] - worldCenter[1]
                        y = (j + 1) * gridScale[2] - worldCenter[2]
                        z = (k + 1) * gridScale[3] - worldCenter[3]
                        r_value = np.sqrt(x**2 + y**2 + z**2)
                        r_values.append(r_value)

            # Evaluate shape function for all r_values
            fs_values = sess.run(shape_function, feed_dict={r_placeholder: r_values, R_placeholder: R, sigma_placeholder: sigma})

        # Check if gridSize in time is 1 and raise error if not
        if gridSize[0] > 1:
            raise ValueError('The time grid is greater than 1, only a size of 1 can be used in comoving')

        # Assign parameters to metric struct
        metric = {}
        metric['params'] = {
            'gridSize': gridSize,
            'worldCenter': worldCenter,
            'velocity': v,
            'R': R,
            'sigma': sigma
        }

        # Assign quantities to metric struct
        metric['type'] = "metric"
        metric['name'] = 'Alcubierre Comoving'
        metric['scaling'] = gridScale
        metric['coords'] = "cartesian"
        metric['index'] = "covariant"

        # Declare a Minkowski space
        alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)

        # Add the Alcubierre modification
        t = 0  # only one timeslice is used
        idx = 0  # index for fs_values
        for i in range(gridSize[1]):
            for j in range(gridSize[2]):
                for k in range(gridSize[3]):
                    fs_value = fs_values[idx]
                    beta[0][t, i, j, k] = v * (1 - fs_value)
                    idx += 1

        # Make tensor from the 3+1 functions
        metric_tensor = threePlusOneBuilder(alpha, beta, gamma)
        
        if metric_tensor is None:
            raise ValueError("Unable to construct metric tensor.")

        metric['tensor'] = metric_tensor

        return metric
    except Exception as e:
        logging.error(f"Error occurred while computing the metric: {e}")
        logging.error(traceback.format_exc())
        return None


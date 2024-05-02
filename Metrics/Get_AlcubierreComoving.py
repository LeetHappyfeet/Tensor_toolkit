import numpy as np
import tensorflow as tf
from setMinkowskiThreePlusOne import setMinkowskiThreePlusOne
from threePlusOneBuilder import threePlusOneBuilder

# Define shape function for Alcubierre metric
def shapeFunction_Alcubierre(r, R, sigma):
    return tf.exp(-((r - R) / sigma) ** 2)

# Define function to build the Alcubierre metric in a Galilean comoving frame
def metricGet_AlcubierreComoving(gridSize, worldCenter, v, R, sigma, gridScale):
    # Handle default input arguments
    if len(gridScale) < 4:
        gridScale += [1] * (4 - len(gridScale))

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
    metric['date'] = tf.strings.format("{}", tf.date)

    # Declare a Minkowski space
    alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)

    # Add the Alcubierre modification
    t = 0  # only one timeslice is used
    for i in range(gridSize[1]):
        for j in range(gridSize[2]):
            for k in range(gridSize[3]):
                # Find grid center x, y, z
                x = (i + 1) * gridScale[1] - worldCenter[1]
                y = (j + 1) * gridScale[2] - worldCenter[2]
                z = (k + 1) * gridScale[3] - worldCenter[3]

                # Find the radius from the center of the bubble
                r = tf.sqrt(tf.pow(x, 2) + tf.pow(y, 2) + tf.pow(z, 2))

                # Find shape function at this point in r
                fs = shapeFunction_Alcubierre(r, R, sigma)

                # Add alcubierre modification to dxdt
                beta[0][t, i, j, k] = v * (1 - fs)

    # Make tensor from the 3+1 functions
    metric['tensor'] = threePlusOneBuilder(alpha, beta, gamma)

    return metric

# Import setMinkowskiThreePlusOne function from provided file
from setMinkowskiThreePlusOne import setMinkowskiThreePlusOne

# Import threePlusOneBuilder function from provided file
from threePlusOneBuilder import threePlusOneBuilder
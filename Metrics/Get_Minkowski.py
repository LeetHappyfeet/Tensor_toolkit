# Filename: metricGet_Minkowski.py

import numpy as np
import tensorflow as tf

# Define function to build the Minkowski metric
def metricGet_Minkowski(gridSize, gridScaling):
    # Handle default input arguments
    if len(gridScaling) < 4:
        gridScaling += [1] * (4 - len(gridScaling))

    # Assign quantities to metric struct
    metric = {}
    metric['type'] = "metric"
    metric['name'] = "Minkowski"
    metric['scaling'] = gridScaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = tf.strings.format("{}", tf.date)

    # Initialize tensor dictionary
    metric['tensor'] = {}

    # dt^2 term
    metric['tensor'][(1, 1)] = -tf.ones(gridSize)

    # Non-time diagonal terms
    metric['tensor'][(2, 2)] = tf.ones(gridSize)
    metric['tensor'][(3, 3)] = tf.ones(gridSize)
    metric['tensor'][(4, 4)] = tf.ones(gridSize)

    # Cross terms
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                metric['tensor'][(i, j)] = tf.zeros(gridSize)

    return metric


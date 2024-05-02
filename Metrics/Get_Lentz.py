# Filename: metricGet_Lentz.py


import numpy as np
import tensorflow as tf
from setMinkowskiThreePlusOne import setMinkowskiThreePlusOne
from threePlusOneBuilder import threePlusOneBuilder

# Define function to build the Lentz metric
def metricGet_Lentz(gridSize, worldCenter, v, scale, gridScale):
    # Handle default input arguments
    if len(scale) < 1:
        scale = max(gridSize[1:4]) / 7
    if len(gridScale) < 4:
        gridScale += [1] * (4 - len(gridScale))

    # Assign parameters to metric struct
    metric = {}
    metric['params'] = {
        'gridSize': gridSize,
        'worldCenter': worldCenter,
        'velocity': v
    }

    # Assign quantities to metric struct
    metric['type'] = "metric"
    metric['name'] = "Lentz"
    metric['scaling'] = gridScale
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = tf.strings.format("{}", tf.date)

    # Declare a Minkowski space
    alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)

    # Lentz Soliton Terms
    for i in range(gridSize[1]):
        for j in range(gridSize[2]):
            for k in range(gridSize[3]):
                x = (i + 1) * gridScale[1] - worldCenter[1]
                y = (j + 1) * gridScale[2] - worldCenter[2]

                for t in range(gridSize[0]):
                    # Determine the x offset of the center of the bubble, centered in time
                    xs = (t + 1) * gridScale[0] - worldCenter[0] * v * tf.constant(299792458.0)

                    xp = x - xs

                    # Get Lentz template values
                    WFX, WFY = getWarpFactorByRegion(xp, y, scale)

                    # Assign dxdt term
                    beta[0][t, i, j, k] = -WFX * v

                    # Assign dydt term
                    beta[1][t, i, j, k] = WFY * v

    # Make tensor from the 3+1 functions
    metric['tensor'] = threePlusOneBuilder(alpha, beta, gamma)

    return metric

# Define function to get Lentz warp factor by region
def getWarpFactorByRegion(xIn, yIn, sizeScale):
    x = xIn
    y = tf.abs(yIn)
    WFX = tf.constant(0, dtype=tf.float32)
    WFY = tf.constant(0, dtype=tf.float32)

    # Lentz shift vector template
    conditions = [
        (x >= sizeScale) & (x <= 2*sizeScale) & (x-sizeScale >= y),
        (x > sizeScale) & (x <= 2*sizeScale) & (x-sizeScale <= y) & (-y+3*sizeScale >= x),
        (x > 0) & (x <= sizeScale) & (x+sizeScale > y) & (-y+sizeScale < x),
        (x > 0) & (x <= sizeScale) & (x+sizeScale <= y) & (-y+3*sizeScale >= x),
        (x > -sizeScale) & (x <= 0) & (-x+sizeScale < y) & (-y+3*sizeScale >= -x),
        (x > -sizeScale) & (x <= 0) & (x+sizeScale <= y) & (-y+sizeScale >= x),
        (x >= -sizeScale) & (x <= sizeScale) & (x+sizeScale > y)
    ]

    choices_WFX = [-2, -1, 0, -0.5, 0.5, 1, 1]
    choices_WFY = [0, 1, 1, 0.5, 0.5, 0, 0]

    WFX = tf.case([(conditions[i], lambda: choices_WFX[i]) for i in range(len(conditions))], default=lambda: 1.0)
    WFY = tf.case([(conditions[i], lambda: choices_WFY[i]) for i in range(len(conditions))], default=lambda: 0.0)

    WFY = tf.sign(yIn) * WFY

    return WFX, WFY
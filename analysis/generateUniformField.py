# Filename: generateUniformField.py
import tensorflow as tf
import numpy as np

def generateUniformField(type, numAngularVec, numTimeVec, tryGPU):

    if not (type.lower() == "nulllike" or type.lower() == "timelike"):
        raise ValueError('Vector field type not generated, use either: "nulllike", "timelike"')

    if type.lower() == "timelike":
        # Generate timelike vectors c^2t^2 > r^2
        bb = tf.linspace(0.0, 1.0, numTimeVec)
        VecField = tf.ones((4, numAngularVec, numTimeVec))

        for jj in range(numTimeVec):
            # Build vector field in Cartesian coordinates
            spherical_points = tf.stack([tf.ones(numAngularVec), getEvenPointsOnSphere(1.0 - bb[jj], numAngularVec, 1)], axis=0)
            VecField[:,:,jj].assign(spherical_points / tf.norm(spherical_points, axis=0))

    elif type.lower() == "nulllike":
        # Build vector field in Cartesian coordinates
        spherical_points = getEvenPointsOnSphere(1.0, numAngularVec, 1)
        VecField = tf.ones((4, numAngularVec))
        VecField[1:,:] = spherical_points
        VecField = VecField / tf.norm(VecField, axis=0)

    # Convert to GPU array if requested
    if tryGPU:
        VecField = tf.Variable(VecField)
    
    return VecField

def getEvenPointsOnSphere(radius, numPoints, dim):
    # Function to generate evenly spaced points on a sphere
    theta = np.linspace(0, np.pi, numPoints)
    phi = np.linspace(0, 2 * np.pi, numPoints)
    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    spherical_points = np.stack([x, y, z], axis=-1)
    return tf.convert_to_tensor(spherical_points, dtype=tf.float32)

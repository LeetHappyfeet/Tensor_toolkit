# Filename: build_minkowski_metric.py

import tensorflow as tf

def build_minkowski_metric(grid_size, grid_scaling=None):
    if grid_scaling is None:
        grid_scaling = [1, 1, 1, 1]

    metric = {}
    metric['type'] = "metric"
    metric['name'] = "Minkowski"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = tf.strings.format("{}", tf.date)

    # Initialize tensor components
    for i in range(1, 5):
        for j in range(1, 5):
            if i == j:
                val = -tf.ones(grid_size) if i == 1 else tf.ones(grid_size)
            else:
                val = tf.zeros(grid_size)
            metric[f"tensor_{i},{j}"] = val

    return metric
import numpy as np

def getEnergyTensor(metric, tryGPU=False, diffOrder='fourth'):
    """
    Converts the metric into the stress energy tensor

    Parameters:
    - metric: A metric dictionary containing tensor components and other information
    - tryGPU: A flag indicating whether to use GPU computation (default: False)
    - diffOrder: Order of finite difference, either 'second' or 'fourth' (default: 'fourth')

    Returns:
    - energy: A dictionary representing the energy tensor
    """

    # Handle default input arguments
    if diffOrder not in ['second', 'fourth']:
        raise ValueError("Order Flag Not Specified Correctly. Options: 'second' or 'fourth'")

    # Check that the metric is verified and covariant
    if not verifyTensor(metric, 1):
        raise ValueError("Metric is not verified. Please verify metric using verifyTensor(metric).")
    if metric['index'] != "covariant":
        metric = changeTensorIndex(metric, "covariant")
        print(f"Changed metric from {metric['index']} index to {'covariant'} index")

    # Compute on GPU if tryGPU flag is True
    if tryGPU:
        # Convert metric to GPU array
        metricTensorGPU = [[np.array(component, dtype='float32') for component in row] for row in metric['tensor']]
        metricTensorGPU = np.array(metricTensorGPU)

        # Compute on GPU
        metric['scaling'] = np.array(metric['scaling'], dtype='float32')
        if diffOrder == 'fourth':
            enDenGPU = met2den(metricTensorGPU, metric['scaling'])
        elif diffOrder == 'second':
            enDenGPU = met2den2(metricTensorGPU, metric['scaling'])

        # Gather results from GPU
        energyTensor = [[np.array(component) for component in row] for row in enDenGPU]

    else:
        # Compute on CPU
        if diffOrder == 'fourth':
            energyTensor = met2den(metric['tensor'], metric['scaling'])
        elif diffOrder == 'second':
            energyTensor = met2den2(metric['tensor'], metric['scaling'])

    # Assign struct values
    energy = {
        'type': 'Stress-Energy',
        'tensor': energyTensor,
        'coords': metric['coords'],
        'index': 'contravariant',
        'order': diffOrder,
        'name': metric['name'],
        'date': np.datetime64('today')
    }

    return energy

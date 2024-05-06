def verifyTensor(inputTensor, suppressMsgs=False):
    """
    Verifies the metric tensor and stress-energy tensor structs

    Parameters:
    - inputTensor: A dictionary representing the tensor
    - suppressMsgs: A flag indicating whether to suppress warning messages (default: False)

    Returns:
    - verified: A boolean indicating whether the tensor is verified
    """

    # Handle input arguments
    if not suppressMsgs:
        suppressMsgs = 0

    verified = 1

    # Check if 'type' field exists
    if 'type' in inputTensor:
        # Check tensor type
        if inputTensor['type'].lower() == "metric":
            # Metric tensor
            dispMessage("type: Metric", suppressMsgs)
        elif inputTensor['type'].lower() == "stress-energy":
            # Stress-Energy Tensor
            dispMessage("Type: Stress-Energy", suppressMsgs)
        else:
            warning("Unknown type")
            verified = 0

        # Check other properties
        # Tensor
        if 'tensor' in inputTensor:
            if isinstance(inputTensor['tensor'], list) and len(inputTensor['tensor']) == 4 and len(inputTensor['tensor'][0][0]) == 4:
                dispMessage("tensor: Verified", suppressMsgs)
            else:
                warning("Tensor is not formatted correctly. Tensor must be a 4x4 list of 4D values.")
                verified = 0
        else:
            warning("tensor: Empty")
            verified = 0

        # Coords
        if 'coords' in inputTensor:
            if inputTensor['coords'].lower() == "cartesian":
                dispMessage("coords: " + inputTensor['coords'], suppressMsgs)
            else:
                warning("Non-cartesian coordinates are not supported at this time. Set .coords to 'cartesian'.")
        else:
            warning("coords: Empty")
            verified = 0

        # Index
        if 'index' in inputTensor:
            if inputTensor['index'].lower() in ["contravariant", "covariant", "mixedupdown", "mixeddownup"]:
                dispMessage("index: " + inputTensor['index'], suppressMsgs)
            else:
                warning("Unknown index")
                verified = 0
        else:
            warning("index: Empty")
            verified = 0
    else:
        warning('Tensor type does not exist. Must be Either "Metric" or "Stress-Energy"')
        verified = 0

    return verified

# Function for displaying messages
def dispMessage(msg, sM):
    if not sM:
        print(msg)

# Function for warning messages
def warning(msg):
    print("Warning:", msg)

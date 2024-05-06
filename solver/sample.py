from verifyTensor import verifyTensor
from getEnergyTensor import getEnergyTensor

# Sample metric tensor (4x4 list)
metric = {
    'type': 'Metric',
    'tensor': [
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, -1]],

        [[0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0]]
    ],
    'coords': 'cartesian',
    'index': 'covariant'
}

# Verify the metric tensor
if verifyTensor(metric):
    print("Metric tensor verified successfully!")
else:
    print("Metric tensor verification failed.")

# Calculate the stress-energy tensor
energy = getEnergyTensor(metric)

# Display the stress-energy tensor
if energy:
    print("\nStress-Energy Tensor:")
    for i in range(4):
        for j in range(4):
            print(energy['tensor'][i][j], end="\t")
        print()

from Metrics.Get_AlcubierreComoving import metricGet_AlcubierreComoving
from solver.getEnergyTensor import getEnergyTensor
from solver.metric_definition import metric
import time

# Define monitoring function
def monitor_progress(start_time, step, total_steps):
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / step * total_steps
    estimated_remaining_time = estimated_total_time - elapsed_time
    print(f"Step {step}/{total_steps} - Elapsed Time: {elapsed_time:.2f}s - Estimated Remaining Time: {estimated_remaining_time:.2f}s")

# Define parameters for the Alcubierre metric
gridSize = [1, 100, 100, 100]
worldCenter = [0, 0, 0, 0]  # You need to add the fourth element here
v = 0.5
r_value = 5.0
R_value = 10.0
sigma_value = 2.0
gridScale = [1, 0.1, 0.1, 0.1]

# Define R and sigma
R = R_value
sigma = sigma_value

# Check lengths of gridScale and worldCenter
print(len(gridScale))
print(len(worldCenter))

# Get the Alcubierre metric
total_steps = gridSize[1] * gridSize[2] * gridSize[3]
start_time = time.time()
metric = metricGet_AlcubierreComoving(gridSize, worldCenter, v, R, sigma, gridScale)
monitor_progress(start_time, total_steps, total_steps)

# Compute the stress-energy tensor from the Alcubierre metric
energy_tensor = getEnergyTensor(metric, tryGPU=False, diffOrder='fourth')

# Print the resulting stress-energy tensor
print(energy_tensor)
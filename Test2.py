import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from analyticalEnergyTensor import analyticalEnergyTensor
import time

def monitor_progress(step, total_steps, start_time):
    elapsed_time = time.time() - start_time
    progress = (step / total_steps) * 100
    estimated_total_time = (elapsed_time / step) * total_steps if step != 0 else 0
    estimated_remaining_time = estimated_total_time - elapsed_time
    print(f"Step {step}/{total_steps} - {progress:.2f}% complete - Elapsed Time: {elapsed_time:.2f}s - Estimated Remaining Time: {estimated_remaining_time:.2f}s")

# Define symbolic variables
t, x, y, z = sp.symbols('t x y z')

# Define an arbitrary metric tensor (for example, a weak gravitational wave)
g = sp.zeros(4, 4)
g[0, 0] = -1
g[1, 1] = 1 + 0.1 * sp.sin(x) * sp.sin(t)
g[2, 2] = 1 + 0.1 * sp.sin(y) * sp.sin(t)
g[3, 3] = 1 + 0.1 * sp.sin(z) * sp.sin(t)
g[1, 2] = 0.05 * sp.cos(x) * sp.cos(y) * sp.cos(t)
g[2, 1] = g[1, 2]  # Symmetric component

# Define the coordinates
coords = [t, x, y, z]

# Start time for progress monitoring
start_time = time.time()

# Compute the stress-energy tensor
T_out = analyticalEnergyTensor(g, coords)

# Print the resulting stress-energy tensor component-wise
print("Stress-Energy Tensor:")
for i in range(4):
    for j in range(4):
        print(f"T[{i},{j}] = {T_out[i, j]}")

# Progress monitoring
total_steps = 1  # Since this is a single calculation, only one step
monitor_progress(1, total_steps, start_time)

# Define a numerical grid for evaluation
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
t_val = 0  # Fixed time value for visualization
z_val = 0  # Fixed z value for visualization

# Function to evaluate tensor components numerically
def evaluate_component(component, t_val, x_grid, y_grid, z_val):
    component_func = sp.lambdify((t, x, y, z), component, 'numpy')
    return component_func(t_val, x_grid, y_grid, z_val)

# Visualize multiple components of the stress-energy tensor
components_to_plot = [(1, 1), (1, 2), (2, 2), (3, 3)]  # List of tensor components to visualize
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for ax, (i, j) in zip(axs.flatten(), components_to_plot):
    component = T_out[i, j]
    component_values = evaluate_component(component, t_val, x_grid, y_grid, z_val)
    
    c = ax.contourf(x_grid, y_grid, component_values, cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_title(f'T[{i},{j}]')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()
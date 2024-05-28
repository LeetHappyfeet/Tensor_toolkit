import tkinter as tk
from tkinter import ttk
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from analyticalEnergyTensor import analyticalEnergyTensor

# Define a function to create and return the metric tensor
def create_metric_tensor(t_amp, x_amp, y_amp, z_amp, xy_coupling, t, x, y, z):
    g = sp.zeros(4, 4)
    g[0, 0] = -1
    g[1, 1] = 1 + x_amp * sp.sin(x) * sp.sin(t)
    g[2, 2] = 1 + y_amp * sp.sin(y) * sp.sin(t)
    g[3, 3] = 1 + z_amp * sp.sin(z) * sp.sin(t)
    g[1, 2] = xy_coupling * sp.cos(x) * sp.cos(y) * sp.cos(t)
    g[2, 1] = g[1, 2]  # Symmetric component
    return g

# Function to evaluate tensor components numerically
def evaluate_component(component, t_val, x_grid, y_grid, z_val, t, x, y, z):
    component_func = sp.lambdify((t, x, y, z), component, 'numpy')
    return component_func(t_val, x_grid, y_grid, z_val)

# Function to update the plot
def update_plot():
    # Get parameter values from UI
    t_amp = float(t_amp_var.get())
    x_amp = float(x_amp_var.get())
    y_amp = float(y_amp_var.get())
    z_amp = float(z_amp_var.get())
    xy_coupling = float(xy_coupling_var.get())
    
    # Define symbolic variables
    t, x, y, z = sp.symbols('t x y z')
    
    # Create metric tensor
    g = create_metric_tensor(t_amp, x_amp, y_amp, z_amp, xy_coupling, t, x, y, z)
    
    # Define the coordinates
    coords = [t, x, y, z]
    
    # Compute the stress-energy tensor
    T_out = analyticalEnergyTensor(g, coords)
    
    # Define a numerical grid for evaluation
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    t_val = 0  # Fixed time value for visualization
    z_val = 0  # Fixed z value for visualization

    # Visualize multiple components of the stress-energy tensor
    components_to_plot = [(1, 1), (1, 2), (2, 2), (3, 3)]  # List of tensor components to visualize
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (i, j) in zip(axs.flatten(), components_to_plot):
        component = T_out[i, j]
        component_values = evaluate_component(component, t_val, x_grid, y_grid, z_val, t, x, y, z)

        c = ax.contourf(x_grid, y_grid, component_values, cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_title(f'T[{i},{j}]')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()

    # Clear the old plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Embed the new plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Create the main window
root = tk.Tk()
root.title("Stress-Energy Tensor Visualization")

# Create input controls
control_frame = ttk.Frame(root)
control_frame.pack(side=tk.LEFT, fill=tk.Y)

tk.Label(control_frame, text="t amplitude").pack()
t_amp_var = tk.StringVar(value="0.1")
tk.Entry(control_frame, textvariable=t_amp_var).pack()

tk.Label(control_frame, text="x amplitude").pack()
x_amp_var = tk.StringVar(value="0.1")
tk.Entry(control_frame, textvariable=x_amp_var).pack()

tk.Label(control_frame, text="y amplitude").pack()
y_amp_var = tk.StringVar(value="0.1")
tk.Entry(control_frame, textvariable=y_amp_var).pack()

tk.Label(control_frame, text="z amplitude").pack()
z_amp_var = tk.StringVar(value="0.1")
tk.Entry(control_frame, textvariable=z_amp_var).pack()

tk.Label(control_frame, text="xy coupling").pack()
xy_coupling_var = tk.StringVar(value="0.05")
tk.Entry(control_frame, textvariable=xy_coupling_var).pack()

update_button = ttk.Button(control_frame, text="Update Plot", command=update_plot)
update_button.pack(pady=10)

# Create plot area
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Initial plot
update_plot()

# Run the tkinter main loop
root.mainloop()
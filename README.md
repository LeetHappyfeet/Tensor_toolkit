




# Stress-Energy Tensor Visualization

This project provides a visualization tool for the stress-energy tensor of a given metric tensor in General Relativity. The visualization allows users to adjust input values and dynamically see the changes in the tensor components.
![python_3ipoIVblvX](https://github.com/LeetHappyfeet/Tensor_toolkit/assets/138872496/f8811393-1f73-4690-a344-32cc53c0ce53)


## Features

- Compute the stress-energy tensor analytically for a given metric tensor.
- Visualize multiple components of the stress-energy tensor.
- Adjust input parameters using a graphical user interface (GUI) built with Tkinter.
- Interactive plots using Matplotlib.

## Requirements

- Python 3.x
- SymPy
- NumPy
- Matplotlib
- Tkinter

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/LeetHappyfeet/Tensor_toolkit/stress-energy-tensor-visualization.git
   cd stress-energy-tensor-visualization
   ```

2. Install the required Python packages:
   ```sh
   pip install sympy numpy matplotlib
   ```

## Usage

1. Run the `visualizer.py` script to start the GUI:
   ```sh
   python visualizer.py
   ```

2. Adjust the input parameters (t amplitude, x amplitude, y amplitude, z amplitude, xy coupling) using the provided entry fields.

3. Click the "Update Plot" button to see the changes in the stress-energy tensor components.

## Project Structure

- `analyticalEnergyTensor.py`: Contains the function to compute the stress-energy tensor analytically.
- `visualizer.py`: Main script to run the GUI and visualize the tensor components.
- `README.md`: Project documentation.

## Example

Here's an example of how to use the `analyticalEnergyTensor` function directly in Python:

```python
import sympy as sp
from analyticalEnergyTensor import analyticalEnergyTensor

# Define symbolic variables
t, x, y, z = sp.symbols('t x y z')

# Define a simple metric tensor
g = sp.zeros(4, 4)
g[0, 0] = -1
g[1, 1] = 1 + 0.1 * sp.sin(x) * sp.sin(t)
g[2, 2] = 1 + 0.1 * sp.sin(y) * sp.sin(t)
g[3, 3] = 1 + 0.1 * sp.sin(z) * sp.sin(t)
g[1, 2] = 0.05 * sp.cos(x) * sp.cos(y) * sp.cos(t)
g[2, 1] = g[1, 2]  # Symmetric component

# Define the coordinates
coords = [t, x, y, z]

# Compute the stress-energy tensor
T_out = analyticalEnergyTensor(g, coords)

# Print the stress-energy tensor
print(T_out)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.


```

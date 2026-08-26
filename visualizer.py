"""Compatibility launcher for the Tensor Toolkit metric tensor simulator.

The visualizer implementation now lives inside the installable ``tensor_toolkit``
package and uses the same experiment/validation pipeline as the command-line
runner.  This file is retained so existing checkouts can still run
``python visualizer.py``.
"""

from tensor_toolkit.gui_overview import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Compatibility entry point for the Tensor Toolkit visualizer.

The base :mod:`tensor_toolkit.gui` implementation now owns the complete
high-resolution interface: tensor overview, memory tiling, disk-backed output,
resource estimates, retained-output selection, and coordinate-aware tensor
component controls.  This module is retained because older launchers import
``tensor_toolkit.gui_overview:main``.
"""
from __future__ import annotations

from tensor_toolkit.gui import TensorToolkitGUI, _gui_dependencies


class TensorToolkitOverviewGUI(TensorToolkitGUI):
    """Backward-compatible name for the unified Tensor Toolkit GUI."""



def main() -> int:
    """Launch the unified Tensor Toolkit desktop visualizer."""
    tk, _ttk, _filedialog, _messagebox, _Figure, _Canvas = _gui_dependencies()
    root = tk.Tk()
    TensorToolkitOverviewGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual GUI entry point
    raise SystemExit(main())

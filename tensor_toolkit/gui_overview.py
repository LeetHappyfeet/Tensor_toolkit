"""Enhanced Tensor Toolkit GUI with a 4x4 tensor-component overview."""
from __future__ import annotations

import numpy as np

from tensor_toolkit.gui import TensorToolkitGUI, _gui_dependencies
from tensor_toolkit.visualization import COORDINATE_NAMES, extract_2d_slice


class TensorToolkitOverviewGUI(TensorToolkitGUI):
    """Metric simulator with a full rank-2 tensor slice overview."""

    def _build_workspace(self):
        ttk = self.ttk
        workspace = ttk.Notebook(self.root)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.workspace = workspace

        plot_tab = ttk.Frame(workspace)
        overview_tab = ttk.Frame(workspace)
        details_tab = ttk.Frame(workspace)
        validation_tab = ttk.Frame(workspace)
        workspace.add(plot_tab, text="Field slice")
        workspace.add(overview_tab, text="4×4 overview")
        workspace.add(details_tab, text="Tensor at center")
        workspace.add(validation_tab, text="Validation")

        plot_tab.rowconfigure(0, weight=1)
        plot_tab.columnconfigure(0, weight=1)
        self.figure = self.Figure(figsize=(8, 6), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Run a simulation to display a tensor field")
        self._canvas = self.FigureCanvasTkAgg(self.figure, master=plot_tab)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        overview_tab.rowconfigure(0, weight=1)
        overview_tab.columnconfigure(0, weight=1)
        self.overview_figure = self.Figure(figsize=(10, 8), dpi=100)
        self.overview_figure.text(
            0.5,
            0.5,
            "Run a simulation to display all 16 tensor components",
            ha="center",
            va="center",
        )
        self._overview_canvas = self.FigureCanvasTkAgg(
            self.overview_figure, master=overview_tab
        )
        self._overview_canvas.draw()
        self._overview_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        details_tab.rowconfigure(0, weight=1)
        details_tab.columnconfigure(0, weight=1)
        self.center_text = self.tk.Text(details_tab, wrap="none", font=("Consolas", 10))
        self.center_text.grid(row=0, column=0, sticky="nsew")
        self.center_text.insert("1.0", "Run a simulation to inspect the center tensor.\n")
        self.center_text.configure(state="disabled")

        validation_tab.rowconfigure(0, weight=1)
        validation_tab.columnconfigure(0, weight=1)
        self.validation_text = self.tk.Text(
            validation_tab, wrap="word", font=("Consolas", 10)
        )
        self.validation_text.grid(row=0, column=0, sticky="nsew")
        self.validation_text.insert(
            "1.0", "Validation diagnostics will appear after a run.\n"
        )
        self.validation_text.configure(state="disabled")

    def _refresh_view(self):
        super()._refresh_view()
        self._refresh_overview()

    def _refresh_overview(self):
        """Render all 16 components of the selected rank-2 field on one slice."""
        if not self.fields or self.field_var.get() not in self.fields:
            return

        field_name = self.field_var.get()
        field = np.asarray(self.fields[field_name])
        if field.ndim != 6 or field.shape[:2] != (4, 4):
            return

        try:
            horizontal = self._axis_index(self.horizontal_var.get())
            vertical = self._axis_index(self.vertical_var.get())
            fixed = {
                axis: int(variable.get())
                for axis, variable in self.fixed_index_vars.items()
            }
            slices = [
                [
                    extract_2d_slice(
                        field,
                        component=(mu, nu),
                        horizontal_axis=horizontal,
                        vertical_axis=vertical,
                        fixed_indices=fixed,
                    )
                    for nu in range(4)
                ]
                for mu in range(4)
            ]
        except (ValueError, IndexError):
            return

        x_values = self.axis_values[horizontal]
        y_values = self.axis_values[vertical]
        max_abs = max(
            float(np.max(np.abs(component)))
            for row in slices
            for component in row
        )
        if max_abs == 0.0:
            max_abs = 1.0

        self.overview_figure.clear()
        axes = self.overview_figure.subplots(4, 4, sharex=True, sharey=True)
        last_mesh = None
        for mu in range(4):
            for nu in range(4):
                ax = axes[mu, nu]
                last_mesh = ax.pcolormesh(
                    x_values,
                    y_values,
                    slices[mu][nu],
                    shading="auto",
                    cmap="coolwarm",
                    vmin=-max_abs,
                    vmax=max_abs,
                )
                ax.set_title(f"[{mu},{nu}]", fontsize=9)
                if mu == 3:
                    ax.set_xlabel(self.horizontal_var.get(), fontsize=8)
                if nu == 0:
                    ax.set_ylabel(self.vertical_var.get(), fontsize=8)
                ax.tick_params(labelsize=7)

        fixed_description = ", ".join(
            f"{COORDINATE_NAMES[axis]}={self.axis_values[axis][index]:g}"
            for axis, index in sorted(fixed.items())
        )
        title = f"{self.metric_name}: {field_name} tensor overview"
        if fixed_description:
            title += f" at {fixed_description}"
        self.overview_figure.suptitle(title, fontsize=11)

        if last_mesh is not None:
            self.overview_figure.colorbar(
                last_mesh,
                ax=axes.ravel().tolist(),
                shrink=0.82,
                pad=0.02,
                label=f"{field_name} component value",
            )
        self.overview_figure.subplots_adjust(
            left=0.07,
            right=0.88,
            bottom=0.07,
            top=0.91,
            wspace=0.18,
            hspace=0.28,
        )
        self._overview_canvas.draw_idle()


def main() -> int:
    """Launch the enhanced Tensor Toolkit desktop visualizer."""
    tk, _ttk, _filedialog, _messagebox, _Figure, _Canvas = _gui_dependencies()
    root = tk.Tk()
    TensorToolkitOverviewGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual GUI entry point
    raise SystemExit(main())

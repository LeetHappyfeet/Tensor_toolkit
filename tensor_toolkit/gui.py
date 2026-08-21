"""Tkinter/Matplotlib desktop interface for the Tensor Toolkit CPU solver."""
from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.io import load_result, save_result
from tensor_toolkit.registry import builtins, configure_grid, get_experiment
from tensor_toolkit.visualization import (
    COORDINATE_NAMES,
    center_matrix,
    editable_metric_parameters,
    extract_2d_slice,
    replace_metric_parameters,
)

GUI_OUTPUTS = frozenset(
    {"metric", "inverse_metric", "ricci", "einstein", "stress_energy"}
)


def _gui_dependencies():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "Tkinter is required for the Tensor Toolkit visualizer. "
            "On Windows, use a Python/Conda installation that includes Tk."
        ) from exc
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Matplotlib is required for the Tensor Toolkit visualizer. "
            "Install the package with its GUI dependencies."
        ) from exc
    return tk, ttk, filedialog, messagebox, Figure, FigureCanvasTkAgg


class TensorToolkitGUI:
    """Desktop metric-field simulator backed by the authoritative CPU pipeline."""

    def __init__(self, root):
        tk, ttk, filedialog, messagebox, Figure, FigureCanvasTkAgg = _gui_dependencies()
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.root = root

        self.result = None
        self.fields: dict[str, np.ndarray] = {}
        self.axis_values: tuple[np.ndarray, ...] = ()
        self.metadata: dict[str, object] = {}
        self.metric_name = ""
        self.metric_parameter_vars: dict[str, object] = {}
        self.fixed_index_vars: dict[int, object] = {}
        self._canvas = None

        root.title("Tensor Toolkit — Metric Tensor Simulator")
        root.geometry("1450x880")
        root.minsize(1100, 700)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_controls()
        self._build_workspace()
        self._load_metric_defaults()

    def _build_controls(self):
        ttk = self.ttk
        tk = self.tk
        panel = ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=0, sticky="nsw")

        ttk.Label(panel, text="Simulation", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Label(panel, text="Metric").grid(row=1, column=0, sticky="w")
        self.metric_var = tk.StringVar(value="minkowski")
        metric_box = ttk.Combobox(
            panel,
            textvariable=self.metric_var,
            values=sorted(builtins()),
            state="readonly",
            width=18,
        )
        metric_box.grid(row=1, column=1, sticky="ew")
        metric_box.bind("<<ComboboxSelected>>", lambda _event: self._load_metric_defaults())

        self.metric_parameter_frame = ttk.LabelFrame(panel, text="Metric parameters", padding=6)
        self.metric_parameter_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)

        grid_frame = ttk.LabelFrame(panel, text="Coordinate grid", padding=6)
        grid_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(grid_frame, text="Points / axis").grid(row=0, column=0, sticky="w")
        self.points_var = tk.StringVar(value="3")
        ttk.Entry(grid_frame, textvariable=self.points_var, width=10).grid(row=0, column=1)
        ttk.Label(grid_frame, text="Uniform extent ±").grid(row=1, column=0, sticky="w")
        self.extent_var = tk.StringVar(value="1")
        ttk.Entry(grid_frame, textvariable=self.extent_var, width=10).grid(row=1, column=1)
        ttk.Label(
            grid_frame,
            text="The same t/x/y/z grid is used by this Phase-2 simulator.",
            wraplength=240,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        button_frame = ttk.Frame(panel)
        button_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(button_frame, text="Run simulation", command=self._start_run)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(button_frame, text="Open result", command=self._open_result).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )
        self.save_button = ttk.Button(button_frame, text="Save result", command=self._save_result)
        self.save_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        ttk.Separator(panel).grid(row=5, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(panel, text="Tensor view", font=("TkDefaultFont", 11, "bold")).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Label(panel, text="Field").grid(row=7, column=0, sticky="w")
        self.field_var = tk.StringVar(value="metric")
        self.field_box = ttk.Combobox(panel, textvariable=self.field_var, state="readonly", width=18)
        self.field_box.grid(row=7, column=1, sticky="ew")
        self.field_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_view())

        ttk.Label(panel, text="Component μ").grid(row=8, column=0, sticky="w")
        self.mu_var = tk.IntVar(value=0)
        mu_box = ttk.Combobox(panel, textvariable=self.mu_var, values=(0, 1, 2, 3), state="readonly", width=5)
        mu_box.grid(row=8, column=1, sticky="w")
        mu_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_view())
        ttk.Label(panel, text="Component ν").grid(row=9, column=0, sticky="w")
        self.nu_var = tk.IntVar(value=0)
        nu_box = ttk.Combobox(panel, textvariable=self.nu_var, values=(0, 1, 2, 3), state="readonly", width=5)
        nu_box.grid(row=9, column=1, sticky="w")
        nu_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_view())

        ttk.Label(panel, text="Horizontal axis").grid(row=10, column=0, sticky="w")
        self.horizontal_var = tk.StringVar(value="x")
        horizontal_box = ttk.Combobox(
            panel, textvariable=self.horizontal_var, values=COORDINATE_NAMES, state="readonly", width=7
        )
        horizontal_box.grid(row=10, column=1, sticky="w")
        horizontal_box.bind("<<ComboboxSelected>>", lambda _event: self._axes_changed())
        ttk.Label(panel, text="Vertical axis").grid(row=11, column=0, sticky="w")
        self.vertical_var = tk.StringVar(value="y")
        vertical_box = ttk.Combobox(
            panel, textvariable=self.vertical_var, values=COORDINATE_NAMES, state="readonly", width=7
        )
        vertical_box.grid(row=11, column=1, sticky="w")
        vertical_box.bind("<<ComboboxSelected>>", lambda _event: self._axes_changed())

        self.fixed_frame = ttk.LabelFrame(panel, text="Fixed coordinates", padding=6)
        self.fixed_frame.grid(row=12, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(self.fixed_frame, text="Run or open a result to select slices.").grid(row=0, column=0)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.status_var, wraplength=250).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

    def _build_workspace(self):
        ttk = self.ttk
        workspace = ttk.Notebook(self.root)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        plot_tab = ttk.Frame(workspace)
        details_tab = ttk.Frame(workspace)
        validation_tab = ttk.Frame(workspace)
        workspace.add(plot_tab, text="Field slice")
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

        details_tab.rowconfigure(0, weight=1)
        details_tab.columnconfigure(0, weight=1)
        self.center_text = self.tk.Text(details_tab, wrap="none", font=("Consolas", 10))
        self.center_text.grid(row=0, column=0, sticky="nsew")
        self.center_text.insert("1.0", "Run a simulation to inspect the center tensor.\n")
        self.center_text.configure(state="disabled")

        validation_tab.rowconfigure(0, weight=1)
        validation_tab.columnconfigure(0, weight=1)
        self.validation_text = self.tk.Text(validation_tab, wrap="word", font=("Consolas", 10))
        self.validation_text.grid(row=0, column=0, sticky="nsew")
        self.validation_text.insert("1.0", "Validation diagnostics will appear after a run.\n")
        self.validation_text.configure(state="disabled")

    def _load_metric_defaults(self):
        experiment = get_experiment(self.metric_var.get())
        self.points_var.set(str(experiment.axes[0].points))
        extent = max(abs(experiment.axes[0].start), abs(experiment.axes[0].stop))
        self.extent_var.set(f"{extent:g}")
        for child in self.metric_parameter_frame.winfo_children():
            child.destroy()
        self.metric_parameter_vars.clear()
        parameters = editable_metric_parameters(experiment.metric)
        if not parameters:
            self.ttk.Label(self.metric_parameter_frame, text="No parameters").grid(row=0, column=0, sticky="w")
            return
        for row, (name, value) in enumerate(parameters.items()):
            self.ttk.Label(self.metric_parameter_frame, text=name).grid(row=row, column=0, sticky="w")
            variable = self.tk.StringVar(value=f"{value:g}")
            self.ttk.Entry(self.metric_parameter_frame, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="e"
            )
            self.metric_parameter_vars[name] = variable

    def _build_experiment(self):
        points = int(self.points_var.get())
        extent = float(self.extent_var.get())
        experiment = get_experiment(self.metric_var.get())
        experiment = configure_grid(experiment, points=points, extent=extent)
        parameters = {
            name: float(variable.get()) for name, variable in self.metric_parameter_vars.items()
        }
        experiment = replace_metric_parameters(experiment, parameters)
        return replace(experiment, outputs=GUI_OUTPUTS, backend="cpu")

    def _start_run(self):
        try:
            experiment = self._build_experiment()
        except (ValueError, TypeError) as exc:
            self.messagebox.showerror("Invalid simulation settings", str(exc))
            return
        self.run_button.configure(state="disabled")
        self.status_var.set("Calculating tensors on CPU…")

        def worker():
            try:
                result = run_experiment(experiment)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._run_failed(exc))
                return
            self.root.after(0, lambda: self._run_finished(result))

        threading.Thread(target=worker, daemon=True).start()

    def _run_failed(self, exc: Exception):
        self.run_button.configure(state="normal")
        self.status_var.set("Simulation failed")
        self.messagebox.showerror("Simulation failed", f"{type(exc).__name__}: {exc}")

    def _run_finished(self, result):
        self.run_button.configure(state="normal")
        self.result = result
        self.fields = result.fields
        self.axis_values = result.axis_values
        self.metadata = result.metadata
        self.metric_name = result.metric_name
        self.status_var.set(
            f"Complete: {result.metric_name}; validation "
            f"{result.metadata.get('diagnostics', {}).get('status', 'UNKNOWN')}"
        )
        self._populate_result_controls()
        self._refresh_validation()
        self._refresh_view()

    def _populate_result_controls(self):
        names = sorted(
            name
            for name, value in self.fields.items()
            if np.asarray(value).ndim == 6 and np.asarray(value).shape[:2] == (4, 4)
        )
        self.field_box.configure(values=names)
        if self.field_var.get() not in names and names:
            self.field_var.set("metric" if "metric" in names else names[0])
        self._build_fixed_controls()

    def _axis_index(self, name: str) -> int:
        return COORDINATE_NAMES.index(name)

    def _axes_changed(self):
        if self.horizontal_var.get() == self.vertical_var.get():
            alternatives = [name for name in COORDINATE_NAMES if name != self.horizontal_var.get()]
            self.vertical_var.set(alternatives[0])
        self._build_fixed_controls()
        self._refresh_view()

    def _build_fixed_controls(self):
        for child in self.fixed_frame.winfo_children():
            child.destroy()
        self.fixed_index_vars.clear()
        if not self.axis_values:
            self.ttk.Label(self.fixed_frame, text="No result loaded").grid(row=0, column=0)
            return
        horizontal = self._axis_index(self.horizontal_var.get())
        vertical = self._axis_index(self.vertical_var.get())
        fixed_axes = [axis for axis in range(4) if axis not in (horizontal, vertical)]
        for row, axis in enumerate(fixed_axes):
            values = self.axis_values[axis]
            center = len(values) // 2
            variable = self.tk.IntVar(value=center)
            self.fixed_index_vars[axis] = variable
            self.ttk.Label(self.fixed_frame, text=f"{COORDINATE_NAMES[axis]} index").grid(
                row=row, column=0, sticky="w"
            )
            spin = self.ttk.Spinbox(
                self.fixed_frame,
                from_=0,
                to=len(values) - 1,
                textvariable=variable,
                width=7,
                command=self._refresh_view,
            )
            spin.grid(row=row, column=1, sticky="e")
            spin.bind("<Return>", lambda _event: self._refresh_view())

    def _refresh_view(self):
        if not self.fields or self.field_var.get() not in self.fields:
            return
        try:
            field_name = self.field_var.get()
            field = self.fields[field_name]
            horizontal = self._axis_index(self.horizontal_var.get())
            vertical = self._axis_index(self.vertical_var.get())
            fixed = {axis: int(variable.get()) for axis, variable in self.fixed_index_vars.items()}
            data = extract_2d_slice(
                field,
                component=(int(self.mu_var.get()), int(self.nu_var.get())),
                horizontal_axis=horizontal,
                vertical_axis=vertical,
                fixed_indices=fixed,
            )
        except (ValueError, IndexError) as exc:
            self.status_var.set(f"View error: {exc}")
            return

        x_values = self.axis_values[horizontal]
        y_values = self.axis_values[vertical]
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        mesh = self.axes.pcolormesh(x_values, y_values, data, shading="auto")
        self.figure.colorbar(
            mesh,
            ax=self.axes,
            label=f"{field_name}[{self.mu_var.get()},{self.nu_var.get()}]",
        )
        self.axes.set_xlabel(self.horizontal_var.get())
        self.axes.set_ylabel(self.vertical_var.get())
        fixed_description = ", ".join(
            f"{COORDINATE_NAMES[axis]}={self.axis_values[axis][index]:g}"
            for axis, index in sorted(fixed.items())
        )
        title = (
            f"{self.metric_name}: {field_name}[{self.mu_var.get()},{self.nu_var.get()}]"
            + (f" at {fixed_description}" if fixed_description else "")
        )
        self.axes.set_title(title)
        self.figure.tight_layout()
        self._canvas.draw_idle()
        self._refresh_center_text()

    def _refresh_center_text(self):
        name = self.field_var.get()
        if name not in self.fields:
            return
        try:
            matrix = center_matrix(self.fields[name])
        except ValueError:
            return
        text = (
            f"Metric: {self.metric_name}\n"
            f"Field: {name}\n"
            f"Grid center indices: {tuple(len(axis) // 2 for axis in self.axis_values)}\n\n"
            + np.array2string(matrix, precision=10, suppress_small=False)
            + "\n"
        )
        self.center_text.configure(state="normal")
        self.center_text.delete("1.0", "end")
        self.center_text.insert("1.0", text)
        self.center_text.configure(state="disabled")

    def _refresh_validation(self):
        diagnostics = self.metadata.get("diagnostics", {})
        lines = [
            f"Metric: {self.metric_name}",
            f"Status: {diagnostics.get('status', 'UNKNOWN')}",
            "",
        ]
        for name, item in diagnostics.get("fields", {}).items():
            lines.append(name)
            lines.append(f"  finite: {item.get('finite', 'unknown')}")
            lines.append(f"  max |value|: {item.get('max_abs', 0):.10g}")
            symmetry = item.get("symmetry")
            if symmetry:
                lines.append(f"  symmetry absolute residual: {symmetry['absolute']:.10g}")
                lines.append(f"  symmetry relative residual: {symmetry['relative']:.10g}")
            lines.append("")
        lines.extend(
            [
                "Interpretation",
                "  PASS means the currently implemented numerical checks passed.",
                "  WARNING means the run completed but a numerical residual exceeded the",
                "  configured warning threshold. It is not a statement that the metric",
                "  itself is physically valid or experimentally realizable.",
            ]
        )
        self.validation_text.configure(state="normal")
        self.validation_text.delete("1.0", "end")
        self.validation_text.insert("1.0", "\n".join(lines))
        self.validation_text.configure(state="disabled")

    def _save_result(self):
        if self.result is None:
            self.messagebox.showinfo("No simulation result", "Run a simulation before saving it.")
            return
        target = self.filedialog.askdirectory(title="Choose result directory")
        if not target:
            return
        path = save_result(self.result, Path(target))
        self.status_var.set(f"Saved result to {path}")

    def _open_result(self):
        target = self.filedialog.askdirectory(title="Open Tensor Toolkit result directory")
        if not target:
            return
        try:
            metadata, fields, axes = load_result(target)
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.messagebox.showerror("Could not open result", str(exc))
            return
        self.result = None
        self.metadata = metadata
        self.fields = fields
        self.axis_values = axes
        self.metric_name = str(metadata.get("metric_name", "Loaded result"))
        self.status_var.set(f"Loaded {target}")
        self._populate_result_controls()
        self._refresh_validation()
        self._refresh_view()


def main() -> int:
    """Launch the Tensor Toolkit desktop visualizer."""
    tk, _ttk, _filedialog, _messagebox, _Figure, _Canvas = _gui_dependencies()
    root = tk.Tk()
    TensorToolkitGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual GUI entry point
    raise SystemExit(main())

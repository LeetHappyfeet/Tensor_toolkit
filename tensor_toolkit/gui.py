"""Tkinter/Matplotlib desktop interface for the Tensor Toolkit CPU solver."""
from __future__ import annotations

import shutil
import threading
from dataclasses import replace
from math import prod
from pathlib import Path

import numpy as np

from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.io import load_result, save_result
from tensor_toolkit.memory import available_memory_bytes, memory_plan, output_bytes, select_storage_mode
from tensor_toolkit.registry import builtins, configure_grid, get_experiment
from tensor_toolkit.visualization import (
    COORDINATE_NAMES,
    center_matrix,
    component_choice,
    component_index,
    editable_metric_parameters,
    extract_2d_slice,
    replace_metric_parameters,
    tensor_component_label,
)

GUI_OUTPUTS = ("metric", "inverse_metric", "ricci", "einstein", "stress_energy")
COMPONENT_CHOICES = tuple(component_choice(i) for i in range(4))


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 1024**3:.2f} GiB"


def _gui_dependencies():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Tkinter is required for the Tensor Toolkit visualizer.") from exc
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Matplotlib is required for the Tensor Toolkit visualizer.") from exc
    return tk, ttk, filedialog, messagebox, Figure, FigureCanvasTkAgg


class TensorToolkitGUI:
    """Desktop metric-field simulator backed by the authoritative CPU pipeline."""

    def __init__(self, root):
        tk, ttk, filedialog, messagebox, Figure, FigureCanvasTkAgg = _gui_dependencies()
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.Figure, self.FigureCanvasTkAgg = Figure, FigureCanvasTkAgg
        self.root = root
        self.result = None
        self.fields: dict[str, np.ndarray] = {}
        self.axis_values: tuple[np.ndarray, ...] = ()
        self.metadata: dict[str, object] = {}
        self.metric_name = ""
        self.metric_parameter_vars: dict[str, object] = {}
        self.fixed_index_vars: dict[int, object] = {}
        self.output_vars: dict[str, object] = {}
        self._canvas = None

        root.title("Tensor Toolkit — Metric Tensor Simulator")
        root.geometry("1520x920")
        root.minsize(1180, 740)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self._build_controls()
        self._build_workspace()
        self._load_metric_defaults()

    def _build_controls(self):
        ttk, tk = self.ttk, self.tk
        outer = ttk.Frame(self.root)
        outer.grid(row=0, column=0, sticky="nsw")
        canvas = tk.Canvas(outer, width=330, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        panel = ttk.Frame(canvas, padding=10)
        panel.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=panel, anchor="nw", width=315)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0
        ttk.Label(panel, text="Simulation", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)); row += 1
        ttk.Label(panel, text="Metric").grid(row=row, column=0, sticky="w")
        self.metric_var = tk.StringVar(value="minkowski")
        metric_box = ttk.Combobox(panel, textvariable=self.metric_var, values=sorted(builtins()), state="readonly", width=20)
        metric_box.grid(row=row, column=1, sticky="ew")
        metric_box.bind("<<ComboboxSelected>>", lambda _e: self._load_metric_defaults()); row += 1

        self.metric_parameter_frame = ttk.LabelFrame(panel, text="Metric parameters", padding=6)
        self.metric_parameter_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7); row += 1

        grid_frame = ttk.LabelFrame(panel, text="Coordinate grid", padding=6)
        grid_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7); row += 1
        ttk.Label(grid_frame, text="Points / axis").grid(row=0, column=0, sticky="w")
        self.points_var = tk.StringVar(value="3")
        points_entry = ttk.Entry(grid_frame, textvariable=self.points_var, width=12)
        points_entry.grid(row=0, column=1, sticky="e")
        ttk.Label(grid_frame, text="Uniform extent ±").grid(row=1, column=0, sticky="w")
        self.extent_var = tk.StringVar(value="1")
        ttk.Entry(grid_frame, textvariable=self.extent_var, width=12).grid(row=1, column=1, sticky="e")
        self.grid_summary_var = tk.StringVar(value="")
        ttk.Label(grid_frame, textvariable=self.grid_summary_var, wraplength=270).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        output_frame = ttk.LabelFrame(panel, text="Outputs to retain", padding=6)
        output_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7); row += 1
        defaults = {"metric", "einstein", "stress_energy"}
        for i, name in enumerate(GUI_OUTPUTS):
            var = tk.BooleanVar(value=name in defaults)
            self.output_vars[name] = var
            ttk.Checkbutton(output_frame, text=name, variable=var, command=self._update_resource_estimate).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 8))

        execution = ttk.LabelFrame(panel, text="Execution & storage", padding=6)
        execution.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7); row += 1
        ttk.Label(execution, text="Memory mode").grid(row=0, column=0, sticky="w")
        self.memory_mode_var = tk.StringVar(value="auto")
        ttk.Combobox(execution, textvariable=self.memory_mode_var, values=("auto", "in_memory", "tiled"), state="readonly", width=12).grid(row=0, column=1, sticky="e")
        ttk.Label(execution, text="Tile t-points").grid(row=1, column=0, sticky="w")
        self.tile_points_var = tk.StringVar(value="8")
        ttk.Entry(execution, textvariable=self.tile_points_var, width=12).grid(row=1, column=1, sticky="e")
        ttk.Label(execution, text="Output storage").grid(row=2, column=0, sticky="w")
        self.storage_mode_var = tk.StringVar(value="auto")
        ttk.Combobox(execution, textvariable=self.storage_mode_var, values=("auto", "memory", "disk"), state="readonly", width=12).grid(row=2, column=1, sticky="e")
        ttk.Label(execution, text="Result directory").grid(row=3, column=0, sticky="w")
        self.output_path_var = tk.StringVar(value=str(Path("results") / "gui_run"))
        ttk.Entry(execution, textvariable=self.output_path_var, width=19).grid(row=3, column=1, sticky="e")
        ttk.Button(execution, text="Choose…", command=self._choose_output_dir).grid(row=4, column=1, sticky="e", pady=(3, 0))
        self.resource_var = tk.StringVar(value="")
        ttk.Label(execution, textvariable=self.resource_var, wraplength=270, justify="left").grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(execution, text="Estimate resources", command=self._update_resource_estimate).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        buttons = ttk.Frame(panel)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 10)); row += 1
        buttons.columnconfigure((0, 1), weight=1)
        self.run_button = ttk.Button(buttons, text="Run simulation", command=self._start_run)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(buttons, text="Open result", command=self._open_result).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.save_button = ttk.Button(buttons, text="Save result", command=self._save_result)
        self.save_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
        ttk.Label(panel, text="Tensor view", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)); row += 1
        ttk.Label(panel, text="Field").grid(row=row, column=0, sticky="w")
        self.field_var = tk.StringVar(value="metric")
        self.field_box = ttk.Combobox(panel, textvariable=self.field_var, state="readonly", width=20)
        self.field_box.grid(row=row, column=1, sticky="ew")
        self.field_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_view()); row += 1

        ttk.Label(panel, text="Component μ").grid(row=row, column=0, sticky="w")
        self.mu_var = tk.StringVar(value=COMPONENT_CHOICES[0])
        mu_box = ttk.Combobox(panel, textvariable=self.mu_var, values=COMPONENT_CHOICES, state="readonly", width=10)
        mu_box.grid(row=row, column=1, sticky="w"); mu_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_view()); row += 1
        ttk.Label(panel, text="Component ν").grid(row=row, column=0, sticky="w")
        self.nu_var = tk.StringVar(value=COMPONENT_CHOICES[0])
        nu_box = ttk.Combobox(panel, textvariable=self.nu_var, values=COMPONENT_CHOICES, state="readonly", width=10)
        nu_box.grid(row=row, column=1, sticky="w"); nu_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_view()); row += 1
        self.component_label_var = tk.StringVar(value="Selected: g_tt [0,0]")
        ttk.Label(panel, textvariable=self.component_label_var, font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 5)); row += 1

        ttk.Label(panel, text="Horizontal axis").grid(row=row, column=0, sticky="w")
        self.horizontal_var = tk.StringVar(value="x")
        hbox = ttk.Combobox(panel, textvariable=self.horizontal_var, values=COORDINATE_NAMES, state="readonly", width=8)
        hbox.grid(row=row, column=1, sticky="w"); hbox.bind("<<ComboboxSelected>>", lambda _e: self._axes_changed()); row += 1
        ttk.Label(panel, text="Vertical axis").grid(row=row, column=0, sticky="w")
        self.vertical_var = tk.StringVar(value="y")
        vbox = ttk.Combobox(panel, textvariable=self.vertical_var, values=COORDINATE_NAMES, state="readonly", width=8)
        vbox.grid(row=row, column=1, sticky="w"); vbox.bind("<<ComboboxSelected>>", lambda _e: self._axes_changed()); row += 1

        self.fixed_frame = ttk.LabelFrame(panel, text="Fixed coordinates", padding=6)
        self.fixed_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=7); row += 1
        ttk.Label(self.fixed_frame, text="Run or open a result to select slices.").grid(row=0, column=0)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.status_var, wraplength=280).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

        for variable in (self.points_var, self.extent_var, self.tile_points_var, self.output_path_var):
            variable.trace_add("write", lambda *_args: self.root.after_idle(self._update_resource_estimate))
        self.memory_mode_var.trace_add("write", lambda *_args: self.root.after_idle(self._update_resource_estimate))
        self.storage_mode_var.trace_add("write", lambda *_args: self.root.after_idle(self._update_resource_estimate))

    def _build_workspace(self):
        ttk = self.ttk
        workspace = ttk.Notebook(self.root)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        plot_tab, overview_tab, details_tab, validation_tab = (ttk.Frame(workspace) for _ in range(4))
        workspace.add(plot_tab, text="Field slice")
        workspace.add(overview_tab, text="4×4 overview")
        workspace.add(details_tab, text="Tensor at center")
        workspace.add(validation_tab, text="Validation")
        plot_tab.rowconfigure(0, weight=1); plot_tab.columnconfigure(0, weight=1)
        self.figure = self.Figure(figsize=(8, 6), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Run a simulation to display a tensor field")
        self._canvas = self.FigureCanvasTkAgg(self.figure, master=plot_tab)
        self._canvas.draw(); self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        overview_tab.rowconfigure(0, weight=1); overview_tab.columnconfigure(0, weight=1)
        self.overview_figure = self.Figure(figsize=(9, 7), dpi=100)
        self.overview_canvas = self.FigureCanvasTkAgg(self.overview_figure, master=overview_tab)
        self.overview_canvas.draw(); self.overview_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        for tab, attr, initial in ((details_tab, "center_text", "Run a simulation to inspect the center tensor.\n"), (validation_tab, "validation_text", "Validation diagnostics will appear after a run.\n")):
            tab.rowconfigure(0, weight=1); tab.columnconfigure(0, weight=1)
            widget = self.tk.Text(tab, wrap="none" if attr == "center_text" else "word", font=("Consolas", 10))
            widget.grid(row=0, column=0, sticky="nsew"); widget.insert("1.0", initial); widget.configure(state="disabled")
            setattr(self, attr, widget)

    def _load_metric_defaults(self):
        experiment = get_experiment(self.metric_var.get())
        self.points_var.set(str(experiment.axes[0].points))
        extent = max(abs(experiment.axes[0].start), abs(experiment.axes[0].stop))
        self.extent_var.set(f"{extent:g}")
        for child in self.metric_parameter_frame.winfo_children(): child.destroy()
        self.metric_parameter_vars.clear()
        parameters = editable_metric_parameters(experiment.metric)
        if not parameters:
            self.ttk.Label(self.metric_parameter_frame, text="No parameters").grid(row=0, column=0, sticky="w")
        for r, (name, value) in enumerate(parameters.items()):
            self.ttk.Label(self.metric_parameter_frame, text=name).grid(row=r, column=0, sticky="w")
            variable = self.tk.StringVar(value=f"{value:g}")
            self.ttk.Entry(self.metric_parameter_frame, textvariable=variable, width=12).grid(row=r, column=1, sticky="e")
            self.metric_parameter_vars[name] = variable
        self._update_resource_estimate()

    def _selected_outputs(self):
        outputs = frozenset(name for name, var in self.output_vars.items() if bool(var.get()))
        if not outputs:
            raise ValueError("select at least one output field to retain")
        return outputs

    def _build_experiment(self):
        points = int(self.points_var.get())
        if points < 3:
            raise ValueError("points / axis must be at least 3")
        extent = float(self.extent_var.get())
        if extent <= 0:
            raise ValueError("uniform extent must be positive")
        tile_points = int(self.tile_points_var.get())
        experiment = configure_grid(get_experiment(self.metric_var.get()), points=points, extent=extent)
        parameters = {name: float(variable.get()) for name, variable in self.metric_parameter_vars.items()}
        experiment = replace_metric_parameters(experiment, parameters)
        return replace(experiment, outputs=self._selected_outputs(), backend="cpu", memory_mode=self.memory_mode_var.get(), tile_points=tile_points)

    def _choose_output_dir(self):
        target = self.filedialog.askdirectory(title="Choose simulation result directory")
        if target:
            self.output_path_var.set(target)

    def _update_resource_estimate(self):
        try:
            points = int(self.points_var.get()); tile = int(self.tile_points_var.get())
            if points < 3 or tile < 1: raise ValueError
            outputs = self._selected_outputs()
            shape = (points,) * 4
            path_text = self.output_path_var.get().strip()
            path = Path(path_text) if path_text else None
            storage = select_storage_mode(shape, outputs, requested_mode=self.storage_mode_var.get(), output_path=path, limit_fraction=0.65)
            plan = memory_plan(shape, outputs, requested_mode=self.memory_mode_var.get(), tile_points=tile, limit_fraction=0.65, disk_backed=storage == "disk")
            persistent = output_bytes(shape, outputs)
            free_disk = None
            if path is not None:
                probe = path if path.exists() else next((p for p in (path, *path.parents) if p.exists()), Path.cwd())
                free_disk = shutil.disk_usage(probe).free
            spacing = (2 * float(self.extent_var.get())) / (points - 1)
            self.grid_summary_var.set(f"{points**4:,} spacetime points; spacing Δ≈{spacing:g}")
            self.resource_var.set(
                f"Plan: {plan['selected_mode']} + {storage}\n"
                f"Peak RAM: {_gib(plan['estimated_selected_bytes'])}\n"
                f"Retained output: {_gib(persistent)}\n"
                f"Available RAM: {_gib(available_memory_bytes())}\n"
                f"Free disk: {_gib(free_disk)}"
            )
        except Exception as exc:
            self.resource_var.set(f"Resource estimate unavailable: {exc}")

    def _start_run(self):
        try:
            experiment = self._build_experiment()
            output_path = Path(self.output_path_var.get().strip()) if self.output_path_var.get().strip() else None
            storage_mode = self.storage_mode_var.get()
            if storage_mode == "disk" and output_path is None:
                raise ValueError("disk storage requires a result directory")
        except (ValueError, TypeError) as exc:
            self.messagebox.showerror("Invalid simulation settings", str(exc)); return
        self.run_button.configure(state="disabled")
        self.status_var.set("Calculating tensors on CPU…")

        def worker():
            try:
                result = run_experiment(experiment, output_path=output_path, storage_mode=storage_mode)
                if output_path is not None:
                    save_result(result, output_path)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._run_failed(exc)); return
            self.root.after(0, lambda: self._run_finished(result, output_path))
        threading.Thread(target=worker, daemon=True).start()

    def _run_failed(self, exc):
        self.run_button.configure(state="normal"); self.status_var.set("Simulation failed")
        self.messagebox.showerror("Simulation failed", f"{type(exc).__name__}: {exc}")

    def _run_finished(self, result, output_path=None):
        self.run_button.configure(state="normal")
        self.result, self.fields, self.axis_values = result, result.fields, result.axis_values
        self.metadata, self.metric_name = result.metadata, result.metric_name
        suffix = f"; saved {output_path}" if output_path is not None else ""
        self.status_var.set(f"Complete: {result.metric_name}; validation {result.metadata.get('diagnostics', {}).get('status', 'UNKNOWN')}{suffix}")
        self._populate_result_controls(); self._refresh_validation(); self._refresh_view(); self._update_resource_estimate()

    def _populate_result_controls(self):
        names = sorted(name for name, value in self.fields.items() if np.asarray(value).ndim == 6 and np.asarray(value).shape[:2] == (4, 4))
        self.field_box.configure(values=names)
        if self.field_var.get() not in names and names:
            self.field_var.set("metric" if "metric" in names else names[0])
        self._build_fixed_controls()

    def _axis_index(self, name): return COORDINATE_NAMES.index(name)

    def _axes_changed(self):
        if self.horizontal_var.get() == self.vertical_var.get():
            self.vertical_var.set(next(name for name in COORDINATE_NAMES if name != self.horizontal_var.get()))
        self._build_fixed_controls(); self._refresh_view()

    def _build_fixed_controls(self):
        for child in self.fixed_frame.winfo_children(): child.destroy()
        self.fixed_index_vars.clear()
        if not self.axis_values:
            self.ttk.Label(self.fixed_frame, text="No result loaded").grid(row=0, column=0); return
        horizontal, vertical = self._axis_index(self.horizontal_var.get()), self._axis_index(self.vertical_var.get())
        for row, axis in enumerate(a for a in range(4) if a not in (horizontal, vertical)):
            values = self.axis_values[axis]; center = len(values) // 2
            variable = self.tk.IntVar(value=center); self.fixed_index_vars[axis] = variable
            value_label = self.tk.StringVar(value=f"{COORDINATE_NAMES[axis]}={values[center]:g}")
            self.ttk.Label(self.fixed_frame, text=f"{COORDINATE_NAMES[axis]} index").grid(row=row, column=0, sticky="w")
            spin = self.ttk.Spinbox(self.fixed_frame, from_=0, to=len(values)-1, textvariable=variable, width=6, command=lambda a=axis, v=variable, l=value_label: self._fixed_changed(a, v, l))
            spin.grid(row=row, column=1, sticky="e")
            self.ttk.Label(self.fixed_frame, textvariable=value_label).grid(row=row, column=2, sticky="w", padx=(5, 0))

    def _fixed_changed(self, axis, variable, label):
        try: label.set(f"{COORDINATE_NAMES[axis]}={self.axis_values[axis][int(variable.get())]:g}")
        except Exception: return
        self._refresh_view()

    def _current_component(self):
        return component_index(self.mu_var.get()), component_index(self.nu_var.get())

    def _refresh_view(self):
        if not self.fields or self.field_var.get() not in self.fields: return
        try:
            field_name = self.field_var.get(); field = self.fields[field_name]
            mu, nu = self._current_component()
            horizontal, vertical = self._axis_index(self.horizontal_var.get()), self._axis_index(self.vertical_var.get())
            fixed = {axis: int(variable.get()) for axis, variable in self.fixed_index_vars.items()}
            data = extract_2d_slice(field, component=(mu, nu), horizontal_axis=horizontal, vertical_axis=vertical, fixed_indices=fixed)
        except (ValueError, IndexError) as exc:
            self.status_var.set(f"View error: {exc}"); return
        label = tensor_component_label(field_name, mu, nu); self.component_label_var.set(f"Selected: {label}")
        x_values, y_values = self.axis_values[horizontal], self.axis_values[vertical]
        self.figure.clear(); self.axes = self.figure.add_subplot(111)
        mesh = self.axes.pcolormesh(x_values, y_values, data, shading="auto")
        self.figure.colorbar(mesh, ax=self.axes, label=label)
        self.axes.set_xlabel(self.horizontal_var.get()); self.axes.set_ylabel(self.vertical_var.get())
        fixed_desc = ", ".join(f"{COORDINATE_NAMES[a]}={self.axis_values[a][i]:g}" for a, i in sorted(fixed.items()))
        self.axes.set_title(f"{self.metric_name}: {label}" + (f" at {fixed_desc}" if fixed_desc else ""))
        self.figure.tight_layout(); self._canvas.draw_idle()
        self._refresh_overview(field_name, horizontal, vertical, fixed)
        self._refresh_center_text()

    def _refresh_overview(self, field_name, horizontal, vertical, fixed):
        field = self.fields[field_name]
        slices = [[extract_2d_slice(field, component=(mu, nu), horizontal_axis=horizontal, vertical_axis=vertical, fixed_indices=fixed) for nu in range(4)] for mu in range(4)]
        vmax = max((float(np.max(np.abs(data))) for row in slices for data in row), default=0.0)
        vmax = vmax if vmax > 0 else 1.0
        self.overview_figure.clear(); axes = self.overview_figure.subplots(4, 4, sharex=True, sharey=True)
        x_values, y_values = self.axis_values[horizontal], self.axis_values[vertical]
        mesh = None
        for mu in range(4):
            for nu in range(4):
                ax = axes[mu, nu]
                mesh = ax.pcolormesh(x_values, y_values, slices[mu][nu], shading="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
                ax.set_title(tensor_component_label(field_name, mu, nu).split(" [", 1)[0], fontsize=8)
                if mu == 3: ax.set_xlabel(self.horizontal_var.get(), fontsize=7)
                if nu == 0: ax.set_ylabel(self.vertical_var.get(), fontsize=7)
                ax.tick_params(labelsize=6)
        if mesh is not None: self.overview_figure.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.7, label=field_name)
        self.overview_figure.suptitle(f"{self.metric_name}: {field_name} 4×4 component overview")
        self.overview_figure.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.91, wspace=0.28, hspace=0.38)
        self.overview_canvas.draw_idle()

    def _refresh_center_text(self):
        name = self.field_var.get()
        if name not in self.fields: return
        try: matrix = center_matrix(self.fields[name])
        except ValueError: return
        text = f"Metric: {self.metric_name}\nField: {name}\nGrid center indices: {tuple(len(axis)//2 for axis in self.axis_values)}\n\n" + np.array2string(matrix, precision=10, suppress_small=False) + "\n"
        self.center_text.configure(state="normal"); self.center_text.delete("1.0", "end"); self.center_text.insert("1.0", text); self.center_text.configure(state="disabled")

    def _refresh_validation(self):
        diagnostics = self.metadata.get("diagnostics", {})
        memory = self.metadata.get("memory", {}); storage = self.metadata.get("storage", {})
        lines = [f"Metric: {self.metric_name}", f"Status: {diagnostics.get('status', 'UNKNOWN')}", f"Execution: {memory.get('selected_mode', 'unknown')}", f"Storage: {storage.get('mode', 'unknown')} ({storage.get('format', 'unknown')})", ""]
        for name, item in diagnostics.get("fields", {}).items():
            lines += [name, f"  finite: {item.get('finite', 'unknown')}", f"  max |value|: {item.get('max_abs', 0):.10g}"]
            symmetry = item.get("symmetry")
            if symmetry: lines += [f"  symmetry absolute residual: {symmetry['absolute']:.10g}", f"  symmetry relative residual: {symmetry['relative']:.10g}"]
            lines.append("")
        lines += ["PASS means the implemented numerical checks passed.", "WARNING means a numerical residual exceeded the configured threshold."]
        self.validation_text.configure(state="normal"); self.validation_text.delete("1.0", "end"); self.validation_text.insert("1.0", "\n".join(lines)); self.validation_text.configure(state="disabled")

    def _save_result(self):
        if self.result is None:
            self.messagebox.showinfo("No simulation result", "Run a simulation before saving it."); return
        target = self.filedialog.askdirectory(title="Choose result directory")
        if not target: return
        try: path = save_result(self.result, Path(target))
        except Exception as exc:
            self.messagebox.showerror("Could not save result", str(exc)); return
        self.status_var.set(f"Saved result to {path}")

    def _open_result(self):
        target = self.filedialog.askdirectory(title="Open Tensor Toolkit result directory")
        if not target: return
        try: metadata, fields, axes = load_result(target)
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.messagebox.showerror("Could not open result", str(exc)); return
        self.result = None; self.metadata = metadata; self.fields = fields; self.axis_values = axes
        self.metric_name = str(metadata.get("metric_name", "Loaded result")); self.status_var.set(f"Loaded {target}")
        self._populate_result_controls(); self._refresh_validation(); self._refresh_view()


def main() -> int:
    tk, _ttk, _filedialog, _messagebox, _Figure, _Canvas = _gui_dependencies()
    root = tk.Tk(); TensorToolkitGUI(root); root.mainloop(); return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

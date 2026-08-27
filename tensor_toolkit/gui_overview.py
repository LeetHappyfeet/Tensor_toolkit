"""Enhanced Tensor Toolkit GUI with tensor overview and memory-aware execution."""
from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np

from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.gui import TensorToolkitGUI, _gui_dependencies
from tensor_toolkit.io import save_result
from tensor_toolkit.memory import memory_plan, select_storage_mode
from tensor_toolkit.visualization import COORDINATE_NAMES, extract_2d_slice

VIEWABLE_OUTPUTS = (
    "metric",
    "inverse_metric",
    "ricci",
    "einstein",
    "stress_energy",
)
DEFAULT_OUTPUTS = frozenset({"metric", "einstein", "stress_energy"})


def selected_output_names(flags: dict[str, bool]) -> frozenset[str]:
    """Return validated GUI output names from checkbox-like boolean flags."""
    selected = frozenset(name for name in VIEWABLE_OUTPUTS if bool(flags.get(name, False)))
    if not selected:
        raise ValueError("select at least one tensor output")
    return selected


def format_memory_plan(plan: dict[str, object]) -> str:
    """Compact human-readable memory preflight summary for the GUI."""
    gib = 1024**3
    selected = float(plan.get("estimated_selected_bytes", 0)) / gib
    in_memory = float(plan.get("estimated_in_memory_bytes", 0)) / gib
    tiled = float(plan.get("estimated_tiled_bytes", 0)) / gib
    persistent = float(plan.get("persistent_output_bytes", 0)) / gib
    available = plan.get("available_bytes")
    safe = plan.get("safe_budget_bytes")
    lines = [
        f"Selected mode: {plan.get('selected_mode', 'unknown')}",
        f"Output storage: {'disk-backed' if plan.get('disk_backed') else 'memory'}",
        f"Estimated peak RAM: {selected:.2f} GiB",
        f"Persistent result size: {persistent:.2f} GiB",
        f"In-memory estimate: {in_memory:.2f} GiB",
        f"Tiled estimate: {tiled:.2f} GiB",
    ]
    if available is not None:
        lines.append(f"Available RAM: {float(available) / gib:.2f} GiB")
    if safe is not None:
        lines.append(f"Safe budget: {float(safe) / gib:.2f} GiB")
    if plan.get("selected_mode") == "tiled":
        lines.append(
            f"Tile core: {plan.get('tile_points')} t-points; halo: {plan.get('halo')}"
        )
    return "\n".join(lines)


class TensorToolkitOverviewGUI(TensorToolkitGUI):
    """Metric simulator with tensor overview, tiling, and disk-backed output."""

    def _build_controls(self):
        super()._build_controls()
        ttk = self.ttk
        tk = self.tk

        panels = self.root.grid_slaves(row=0, column=0)
        if not panels:
            raise RuntimeError("Tensor Toolkit GUI control panel was not created")
        panel = panels[0]

        memory_frame = ttk.LabelFrame(panel, text="Memory / execution", padding=6)
        memory_frame.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        ttk.Label(memory_frame, text="Memory mode").grid(row=0, column=0, sticky="w")
        self.memory_mode_var = tk.StringVar(value="auto")
        memory_box = ttk.Combobox(
            memory_frame,
            textvariable=self.memory_mode_var,
            values=("auto", "in_memory", "tiled"),
            state="readonly",
            width=12,
        )
        memory_box.grid(row=0, column=1, sticky="e")
        memory_box.bind("<<ComboboxSelected>>", lambda _event: self._clear_memory_preview())

        ttk.Label(memory_frame, text="Tile t-points").grid(row=1, column=0, sticky="w")
        self.tile_points_var = tk.StringVar(value="8")
        ttk.Spinbox(
            memory_frame,
            from_=1,
            to=256,
            textvariable=self.tile_points_var,
            width=10,
            command=self._clear_memory_preview,
        ).grid(row=1, column=1, sticky="e")

        ttk.Label(memory_frame, text="Output storage").grid(row=2, column=0, sticky="w")
        self.storage_mode_var = tk.StringVar(value="auto")
        storage_box = ttk.Combobox(
            memory_frame,
            textvariable=self.storage_mode_var,
            values=("auto", "memory", "disk"),
            state="readonly",
            width=12,
        )
        storage_box.grid(row=2, column=1, sticky="e")
        storage_box.bind("<<ComboboxSelected>>", lambda _event: self._clear_memory_preview())

        self.storage_path_var = tk.StringVar(value="")
        ttk.Label(memory_frame, text="Disk result dir").grid(row=3, column=0, sticky="w")
        ttk.Entry(memory_frame, textvariable=self.storage_path_var, width=15).grid(
            row=3, column=1, sticky="ew"
        )
        ttk.Button(
            memory_frame,
            text="Choose…",
            command=self._choose_storage_directory,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        output_frame = ttk.LabelFrame(panel, text="Retained outputs", padding=6)
        output_frame.grid(row=15, column=0, columnspan=2, sticky="ew", pady=4)
        self.output_vars: dict[str, object] = {}
        for row, name in enumerate(VIEWABLE_OUTPUTS):
            variable = tk.BooleanVar(value=name in DEFAULT_OUTPUTS)
            self.output_vars[name] = variable
            ttk.Checkbutton(
                output_frame,
                text=name,
                variable=variable,
                command=self._clear_memory_preview,
            ).grid(row=row, column=0, sticky="w")

        ttk.Button(
            output_frame,
            text="Estimate memory",
            command=self._show_memory_preflight,
        ).grid(row=len(VIEWABLE_OUTPUTS), column=0, sticky="ew", pady=(6, 0))

        self.memory_preview_var = tk.StringVar(
            value="Memory estimate will be calculated before each run."
        )
        ttk.Label(
            panel,
            textvariable=self.memory_preview_var,
            justify="left",
            wraplength=270,
        ).grid(row=16, column=0, columnspan=2, sticky="w", pady=(4, 8))

        self.root.geometry("1520x1020")
        self.root.minsize(1150, 760)

    def _choose_storage_directory(self):
        target = self.filedialog.askdirectory(title="Choose disk-backed result directory")
        if target:
            self.storage_path_var.set(target)
            self._clear_memory_preview()

    def _selected_outputs(self) -> frozenset[str]:
        flags = {name: bool(variable.get()) for name, variable in self.output_vars.items()}
        return selected_output_names(flags)

    def _build_experiment(self):
        experiment = super()._build_experiment()
        outputs = self._selected_outputs()
        memory_mode = self.memory_mode_var.get()
        tile_points = int(self.tile_points_var.get())
        if tile_points < 1:
            raise ValueError("tile points must be at least 1")
        return replace(
            experiment,
            outputs=outputs,
            backend="cpu",
            memory_mode=memory_mode,
            tile_points=tile_points,
        )

    def _storage_settings(self, experiment):
        requested = self.storage_mode_var.get()
        raw_path = self.storage_path_var.get().strip()
        output_path = raw_path or None
        shape = tuple(axis.points for axis in experiment.axes)
        selected = select_storage_mode(
            shape,
            experiment.outputs,
            requested_mode=requested,
            output_path=output_path,
            limit_fraction=experiment.memory_limit_fraction,
        )
        return requested, selected, output_path

    def _preflight_for_experiment(self, experiment):
        _requested, selected_storage, _path = self._storage_settings(experiment)
        shape = tuple(axis.points for axis in experiment.axes)
        return memory_plan(
            shape,
            experiment.outputs,
            requested_mode=experiment.memory_mode,
            tile_points=experiment.tile_points,
            limit_fraction=experiment.memory_limit_fraction,
            disk_backed=selected_storage == "disk",
        )

    def _clear_memory_preview(self):
        self.memory_preview_var.set("Memory estimate changed; recalculate or run simulation.")

    def _show_memory_preflight(self):
        try:
            experiment = self._build_experiment()
            plan = self._preflight_for_experiment(experiment)
        except (ValueError, TypeError, MemoryError, OSError) as exc:
            self.memory_preview_var.set(f"Preflight failed: {exc}")
            return
        self.memory_preview_var.set(format_memory_plan(plan))

    def _start_run(self):
        try:
            experiment = self._build_experiment()
            requested_storage, selected_storage, output_path = self._storage_settings(experiment)
            if selected_storage == "disk" and output_path is None:
                raise ValueError("choose a disk result directory before a disk-backed run")
            plan = self._preflight_for_experiment(experiment)
        except (ValueError, TypeError, MemoryError, OSError) as exc:
            self.messagebox.showerror("Simulation preflight failed", str(exc))
            self.memory_preview_var.set(f"Preflight failed: {exc}")
            return

        self.memory_preview_var.set(format_memory_plan(plan))
        selected_mode = str(plan.get("selected_mode", experiment.memory_mode))
        selected_gib = float(plan.get("estimated_selected_bytes", 0)) / 1024**3
        self.run_button.configure(state="disabled")
        self.status_var.set(
            f"Calculating on CPU; {selected_mode}, {selected_storage} storage, "
            f"estimated peak {selected_gib:.2f} GiB…"
        )

        def worker():
            try:
                result = run_experiment(
                    experiment,
                    output_path=output_path,
                    storage_mode=requested_storage,
                )
                if result.metadata.get("storage", {}).get("mode") == "disk":
                    save_result(result, output_path)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._run_failed(exc))
                return
            self.root.after(0, lambda: self._run_finished(result))

        threading.Thread(target=worker, daemon=True).start()

    def _run_finished(self, result):
        super()._run_finished(result)
        memory = result.metadata.get("memory", {})
        storage = result.metadata.get("storage", {})
        if memory:
            self.memory_preview_var.set(format_memory_plan(memory))
            mode = memory.get("selected_mode", "unknown")
            store = storage.get("mode", "memory")
            validation = result.metadata.get("diagnostics", {}).get("status", "UNKNOWN")
            self.status_var.set(
                f"Complete: {result.metric_name}; mode {mode}; storage {store}; validation {validation}"
            )

    def _save_result(self):
        if self.result is not None and self.result.metadata.get("storage", {}).get("mode") == "disk":
            self.messagebox.showinfo(
                "Disk-backed result",
                "This result was written incrementally to its selected directory during the run.",
            )
            return
        super()._save_result()

    def _open_result(self):
        super()._open_result()
        memory = self.metadata.get("memory", {}) if self.metadata else {}
        storage = self.metadata.get("storage", {}) if self.metadata else {}
        if memory:
            self.memory_preview_var.set(format_memory_plan(memory))
        if storage.get("mode") == "disk":
            self.status_var.set(
                f"Loaded disk-backed {self.metric_name}; fields remain memory-mapped"
            )

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
        field = np.asanyarray(self.fields[field_name])
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

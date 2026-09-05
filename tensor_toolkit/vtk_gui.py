"""VTK desktop visualizer for Tensor Toolkit.

The GUI is deliberately downstream of tensor_toolkit.visualization_data.  VTK
receives normalized NumPy-backed volumes and geometry; it never computes
physics or reaches into solver internals.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

import numpy as np

from tensor_toolkit.experiment import ExperimentResult, run_experiment
from tensor_toolkit.io import load_result, save_result
from tensor_toolkit.registry import builtins, configure_grid, get_experiment
from tensor_toolkit.visualization import editable_metric_parameters, replace_metric_parameters
from tensor_toolkit.visualization_data import experiment_volume


RANK2_FIELDS = ("metric", "inverse_metric", "ricci", "einstein", "stress_energy")
GUI_OUTPUTS = ("metric", "inverse_metric", "ricci", "ricci_scalar", "einstein", "stress_energy")


def _dependencies():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is required for the Tensor Toolkit visualizer.") from exc
    try:
        from vtkmodules.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
        from vtkmodules.vtkCommonCore import vtkDoubleArray
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkColorTransferFunction,
            vtkPiecewiseFunction,
            vtkRenderer,
            vtkVolume,
            vtkVolumeProperty,
        )
        from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
        from vtkmodules.vtkFiltersCore import vtkContourFilter
        from vtkmodules.vtkRenderingCore import vtkDataSetMapper
        from vtkmodules.util.numpy_support import numpy_to_vtk
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'VTK is required for the 3-D visualizer. Install with '
            'python -m pip install -e ".[visualization]"'
        ) from exc
    return {
        "tk": tk, "ttk": ttk, "filedialog": filedialog, "messagebox": messagebox,
        "vtkTkRenderWindowInteractor": vtkTkRenderWindowInteractor,
        "vtkImageData": vtkImageData, "vtkRenderer": vtkRenderer,
        "vtkVolume": vtkVolume, "vtkVolumeProperty": vtkVolumeProperty,
        "vtkColorTransferFunction": vtkColorTransferFunction,
        "vtkPiecewiseFunction": vtkPiecewiseFunction,
        "vtkSmartVolumeMapper": vtkSmartVolumeMapper,
        "vtkContourFilter": vtkContourFilter, "vtkDataSetMapper": vtkDataSetMapper,
        "vtkActor": vtkActor, "numpy_to_vtk": numpy_to_vtk,
    }


class TensorToolkitVTKGUI:
    """Interactive 3-D field viewer backed by the validated experiment pipeline."""

    def __init__(self, root):
        self.d = _dependencies()
        self.tk = self.d["tk"]
        self.ttk = self.d["ttk"]
        self.root = root
        self.root.title("Tensor Toolkit — VTK Scientific Visualizer")
        self.root.geometry("1500x900")
        self.result = None
        self._actors = []
        self._parameter_vars = {}

        self.metric_var = self.tk.StringVar(value="alcubierre")
        self.points_var = self.tk.IntVar(value=9)
        self.extent_var = self.tk.DoubleVar(value=2.0)
        self.field_var = self.tk.StringVar(value="stress_energy")
        self.mu_var = self.tk.IntVar(value=0)
        self.nu_var = self.tk.IntVar(value=0)
        self.time_var = self.tk.IntVar(value=0)
        self.mode_var = self.tk.StringVar(value="volume")
        self.status_var = self.tk.StringVar(value="Ready")
        self.validation_var = self.tk.StringVar(value="No result loaded")
        self.output_vars = {
            name: self.tk.BooleanVar(value=name in {"metric", "einstein", "stress_energy"})
            for name in GUI_OUTPUTS
        }

        self._build_layout()
        self._metric_changed()
        self._update_field_choices()

    def _build_layout(self):
        ttk = self.ttk
        outer = ttk.Panedwindow(self.root, orient="horizontal")
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, padding=8)
        view = ttk.Frame(outer)
        outer.add(controls, weight=0)
        outer.add(view, weight=1)

        row = 0
        ttk.Label(controls, text="Experiment", font=("", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        metric_box = ttk.Combobox(controls, textvariable=self.metric_var, values=sorted(builtins()), state="readonly", width=20)
        metric_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        metric_box.bind("<<ComboboxSelected>>", lambda _e: self._metric_changed())
        row += 1

        self.parameter_frame = ttk.LabelFrame(controls, text="Metric parameters", padding=6)
        self.parameter_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1

        ttk.Label(controls, text="Points / axis").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(controls, from_=3, to=257, textvariable=self.points_var, width=9).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(controls, text="Extent ±").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.extent_var, width=10).grid(row=row, column=1, sticky="ew")
        row += 1

        outputs = ttk.LabelFrame(controls, text="Retained fields", padding=6)
        outputs.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        for i, name in enumerate(GUI_OUTPUTS):
            ttk.Checkbutton(outputs, text=name, variable=self.output_vars[name]).grid(row=i, column=0, sticky="w")
        row += 1

        ttk.Button(controls, text="Run experiment", command=self._run).grid(row=row, column=0, sticky="ew", pady=3)
        ttk.Button(controls, text="Open result", command=self._open).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        ttk.Button(controls, text="Save result", command=self._save).grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        row += 1

        render = ttk.LabelFrame(controls, text="3-D view", padding=6)
        render.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(render, text="Field").grid(row=0, column=0, sticky="w")
        self.field_box = ttk.Combobox(render, textvariable=self.field_var, state="readonly", width=18)
        self.field_box.grid(row=0, column=1, sticky="ew")
        self.field_box.bind("<<ComboboxSelected>>", lambda _e: self.render())
        ttk.Label(render, text="μ").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(render, from_=0, to=3, textvariable=self.mu_var, width=5, command=self.render).grid(row=1, column=1, sticky="ew")
        ttk.Label(render, text="ν").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(render, from_=0, to=3, textvariable=self.nu_var, width=5, command=self.render).grid(row=2, column=1, sticky="ew")
        ttk.Label(render, text="Time index").grid(row=3, column=0, sticky="w")
        self.time_spin = ttk.Spinbox(render, from_=0, to=0, textvariable=self.time_var, width=7, command=self.render)
        self.time_spin.grid(row=3, column=1, sticky="ew")
        ttk.Label(render, text="Mode").grid(row=4, column=0, sticky="w")
        mode = ttk.Combobox(render, textvariable=self.mode_var, values=("volume", "isosurface"), state="readonly", width=12)
        mode.grid(row=4, column=1, sticky="ew")
        mode.bind("<<ComboboxSelected>>", lambda _e: self.render())
        ttk.Button(render, text="Refresh 3-D scene", command=self.render).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        row += 1

        ttk.Label(controls, text="Validation", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        ttk.Label(controls, textvariable=self.validation_var, wraplength=285, justify="left").grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1
        ttk.Label(controls, textvariable=self.status_var, wraplength=285, justify="left").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        controls.columnconfigure(1, weight=1)

        self.renderer = self.d["vtkRenderer"]()
        self.renderer.SetBackground(0.06, 0.07, 0.09)
        self.interactor = self.d["vtkTkRenderWindowInteractor"](view, width=1000, height=800)
        self.interactor.pack(fill="both", expand=True)
        self.interactor.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor.Initialize()

    def _metric_changed(self):
        for child in self.parameter_frame.winfo_children():
            child.destroy()
        self._parameter_vars.clear()
        experiment = get_experiment(self.metric_var.get())
        for row, (name, value) in enumerate(editable_metric_parameters(experiment.metric).items()):
            self.ttk.Label(self.parameter_frame, text=name).grid(row=row, column=0, sticky="w")
            var = self.tk.DoubleVar(value=value)
            self.ttk.Entry(self.parameter_frame, textvariable=var, width=12).grid(row=row, column=1, sticky="ew")
            self._parameter_vars[name] = var

    def _selected_outputs(self):
        outputs = frozenset(name for name, var in self.output_vars.items() if var.get())
        if not outputs:
            raise ValueError("select at least one retained field")
        return outputs

    def _build_experiment(self):
        experiment = get_experiment(self.metric_var.get())
        experiment = replace_metric_parameters(
            experiment, {name: var.get() for name, var in self._parameter_vars.items()}
        )
        experiment = configure_grid(experiment, points=int(self.points_var.get()), extent=float(self.extent_var.get()))
        return replace(experiment, outputs=self._selected_outputs())

    def _run(self):
        try:
            experiment = self._build_experiment()
        except Exception as exc:
            self.d["messagebox"].showerror("Invalid experiment", str(exc))
            return
        self.status_var.set("Calculating on the validated CPU/NumPy pipeline…")
        def worker():
            try:
                result = run_experiment(experiment)
            except Exception as exc:
                self.root.after(0, lambda: self._worker_failed(exc))
                return
            self.root.after(0, lambda: self._accept_result(result))
        threading.Thread(target=worker, daemon=True).start()

    def _worker_failed(self, exc):
        self.status_var.set("Calculation failed")
        self.d["messagebox"].showerror("Tensor Toolkit", str(exc))

    def _accept_result(self, result):
        self.result = result
        nt = len(result.axis_values[0])
        self.time_spin.configure(to=max(0, nt - 1))
        self.time_var.set(min(self.time_var.get(), nt - 1))
        self._update_field_choices()
        diagnostics = result.metadata.get("diagnostics", {})
        status = diagnostics.get("status", "unknown")
        self.validation_var.set(f"Pipeline validation status: {status}")
        self.status_var.set(f"Loaded {result.metric_name}; grid {result.metadata.get('shape', '?')}")
        self.render()

    def _update_field_choices(self):
        fields = [name for name in RANK2_FIELDS if self.result is None or name in self.result.fields]
        if self.result is not None and "ricci_scalar" in self.result.fields:
            fields.append("ricci_scalar")
        self.field_box.configure(values=fields)
        if fields and self.field_var.get() not in fields:
            self.field_var.set(fields[0])

    def _open(self):
        path = self.d["filedialog"].askdirectory(title="Open Tensor Toolkit result")
        if not path:
            return
        try:
            metadata, fields, axes = load_result(path)
            result = ExperimentResult(
                metric_name=str(metadata.get("metric_name", "unknown")),
                coordinates=tuple(metadata.get("coordinates", ("t", "x", "y", "z"))),
                axis_values=tuple(axes),
                fields=fields,
                metadata={key: value for key, value in metadata.items() if key not in {"metric_name", "coordinates", "fields"}},
            )
            self._accept_result(result)
        except Exception as exc:
            self.d["messagebox"].showerror("Open result", str(exc))

    def _save(self):
        if self.result is None:
            self.d["messagebox"].showinfo("Save result", "Run or open a result first.")
            return
        path = self.d["filedialog"].askdirectory(title="Choose result directory")
        if not path:
            return
        try:
            save_result(self.result, Path(path))
            self.status_var.set(f"Saved result to {path}")
        except Exception as exc:
            self.d["messagebox"].showerror("Save result", str(exc))

    def _vtk_image(self, volume):
        image = self.d["vtkImageData"]()
        nx, ny, nz = volume.values.shape
        image.SetDimensions(nx, ny, nz)
        spacing = []
        origin = []
        for axis in (volume.x, volume.y, volume.z):
            origin.append(float(axis[0]))
            spacing.append(float(axis[1] - axis[0]) if len(axis) > 1 else 1.0)
        image.SetOrigin(*origin)
        image.SetSpacing(*spacing)
        flat = np.ascontiguousarray(volume.values).ravel(order="F")
        vtk_values = self.d["numpy_to_vtk"](flat, deep=True)
        vtk_values.SetName(volume.name)
        image.GetPointData().SetScalars(vtk_values)
        return image

    def render(self):
        if self.result is None:
            return
        try:
            volume = experiment_volume(
                self.result,
                self.field_var.get(),
                component=(int(self.mu_var.get()), int(self.nu_var.get())),
                time_index=int(self.time_var.get()),
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            return

        self.renderer.RemoveAllViewProps()
        image = self._vtk_image(volume)
        finite = volume.values[np.isfinite(volume.values)]
        if finite.size == 0:
            self.status_var.set("Selected volume contains no finite values")
            return
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if np.isclose(vmin, vmax):
            vmax = vmin + max(1.0, abs(vmin)) * 1e-12

        if self.mode_var.get() == "isosurface":
            contour = self.d["vtkContourFilter"]()
            contour.SetInputData(image)
            contour.SetValue(0, 0.5 * (vmin + vmax))
            mapper = self.d["vtkDataSetMapper"]()
            mapper.SetInputConnection(contour.GetOutputPort())
            actor = self.d["vtkActor"]()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
        else:
            mapper = self.d["vtkSmartVolumeMapper"]()
            mapper.SetInputData(image)
            color = self.d["vtkColorTransferFunction"]()
            color.AddRGBPoint(vmin, 0.1, 0.2, 0.8)
            color.AddRGBPoint(0.5 * (vmin + vmax), 0.9, 0.9, 0.9)
            color.AddRGBPoint(vmax, 0.8, 0.2, 0.1)
            opacity = self.d["vtkPiecewiseFunction"]()
            opacity.AddPoint(vmin, 0.0)
            opacity.AddPoint(0.5 * (vmin + vmax), 0.08)
            opacity.AddPoint(vmax, 0.65)
            prop = self.d["vtkVolumeProperty"]()
            prop.SetColor(color)
            prop.SetScalarOpacity(opacity)
            prop.ShadeOn()
            prop.SetInterpolationTypeToLinear()
            actor = self.d["vtkVolume"]()
            actor.SetMapper(mapper)
            actor.SetProperty(prop)
            self.renderer.AddVolume(actor)

        self.renderer.ResetCamera()
        self.interactor.GetRenderWindow().Render()
        self.status_var.set(
            f"{volume.name} at t={volume.metadata['time']:.6g}; "
            f"range [{vmin:.6g}, {vmax:.6g}]"
        )


def main() -> int:
    d = _dependencies()
    root = d["tk"].Tk()
    TensorToolkitVTKGUI(root)
    root.mainloop()
    return 0


__all__ = ["TensorToolkitVTKGUI", "main"]

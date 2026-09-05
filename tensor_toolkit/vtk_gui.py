"""VTK desktop visualizer for Tensor Toolkit.

The GUI is downstream of tensor_toolkit.visualization_data and
visualization_timeline. VTK receives normalized stored results and never
advances the physics solver.
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
from tensor_toolkit.visualization_data import experiment_volume, trajectory_event_points
from tensor_toolkit.visualization_io import load_saved_trajectory
from tensor_toolkit.visualization_timeline import (
    FrameCache,
    VisualizationTimeline,
    sample_trajectory_positions,
    trajectory_trail,
)


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
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData, vtkPolyLine
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        from vtkmodules.vtkRenderingCore import (
            vtkActor, vtkColorTransferFunction, vtkDataSetMapper, vtkPiecewiseFunction,
            vtkPolyDataMapper, vtkRenderer, vtkVolume, vtkVolumeProperty,
        )
        from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
        from vtkmodules.vtkFiltersCore import vtkContourFilter
        from vtkmodules.util.numpy_support import numpy_to_vtk
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'VTK is required for the 3-D visualizer. Install with '
            'python -m pip install -e ".[visualization]"'
        ) from exc
    return locals()


class TensorToolkitVTKGUI:
    """Interactive 3-D field and trajectory viewer over stored solver results."""

    TICK_MS = 16

    def __init__(self, root):
        self.d = _dependencies()
        self.tk = self.d["tk"]
        self.ttk = self.d["ttk"]
        self.root = root
        self.root.title("Tensor Toolkit — VTK Scientific Visualizer")
        self.root.geometry("1550x920")

        self.result = None
        self.trajectory = None
        self.timeline = VisualizationTimeline(0.0, 1.0, 0.0)
        self.frame_cache = FrameCache(5)
        self._parameter_vars = {}
        self._volume_state = None
        self._body_actors = {}
        self._trail_actors = {}
        self._event_actors = []
        self._camera_initialized = False
        self._scrubbing = False

        self.metric_var = self.tk.StringVar(value="alcubierre")
        self.points_var = self.tk.IntVar(value=9)
        self.extent_var = self.tk.DoubleVar(value=2.0)
        self.field_var = self.tk.StringVar(value="stress_energy")
        self.mu_var = self.tk.IntVar(value=0)
        self.nu_var = self.tk.IntVar(value=0)
        self.mode_var = self.tk.StringVar(value="volume")
        self.status_var = self.tk.StringVar(value="Ready")
        self.validation_var = self.tk.StringVar(value="No result loaded")

        self.timeline_var = self.tk.DoubleVar(value=0.0)
        self.timeline_text_var = self.tk.StringVar(value="t = 0")
        self.play_text_var = self.tk.StringVar(value="▶ Play")
        self.playback_var = self.tk.StringVar(value="1")
        self.loop_var = self.tk.BooleanVar(value=False)
        self.trail_var = self.tk.DoubleVar(value=0.0)
        self.follow_var = self.tk.StringVar(value="World")
        self.event_var = self.tk.StringVar(value="")

        self.output_vars = {
            name: self.tk.BooleanVar(value=name in {"metric", "einstein", "stress_energy"})
            for name in GUI_OUTPUTS
        }

        self._build_layout()
        self._metric_changed()
        self._update_field_choices()
        self.root.after(self.TICK_MS, self._tick)

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
        ttk.Button(controls, text="Open tensor result", command=self._open).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        ttk.Button(controls, text="Open trajectory", command=self._open_trajectory).grid(row=row, column=0, sticky="ew", pady=3)
        ttk.Button(controls, text="Save tensor result", command=self._save).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        render = ttk.LabelFrame(controls, text="3-D field", padding=6)
        render.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(render, text="Field").grid(row=0, column=0, sticky="w")
        self.field_box = ttk.Combobox(render, textvariable=self.field_var, state="readonly", width=18)
        self.field_box.grid(row=0, column=1, sticky="ew")
        self.field_box.bind("<<ComboboxSelected>>", lambda _e: self._field_changed())
        ttk.Label(render, text="μ").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(render, from_=0, to=3, textvariable=self.mu_var, width=5, command=self._field_changed).grid(row=1, column=1, sticky="ew")
        ttk.Label(render, text="ν").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(render, from_=0, to=3, textvariable=self.nu_var, width=5, command=self._field_changed).grid(row=2, column=1, sticky="ew")
        ttk.Label(render, text="Mode").grid(row=3, column=0, sticky="w")
        mode = ttk.Combobox(render, textvariable=self.mode_var, values=("volume", "isosurface"), state="readonly", width=12)
        mode.grid(row=3, column=1, sticky="ew")
        mode.bind("<<ComboboxSelected>>", lambda _e: self._field_changed())
        row += 1

        timeline_box = ttk.LabelFrame(controls, text="Visualization timeline", padding=6)
        timeline_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(timeline_box, textvariable=self.timeline_text_var).grid(row=0, column=0, columnspan=4, sticky="w")
        self.timeline_scale = ttk.Scale(
            timeline_box, variable=self.timeline_var, from_=0.0, to=1.0,
            command=self._timeline_scrub,
        )
        self.timeline_scale.grid(row=1, column=0, columnspan=4, sticky="ew", pady=3)
        self.play_button = ttk.Button(timeline_box, textvariable=self.play_text_var, command=self._toggle_play)
        self.play_button.grid(row=2, column=0, sticky="ew")
        ttk.Label(timeline_box, text="Rate").grid(row=2, column=1, sticky="e")
        rate = ttk.Combobox(timeline_box, textvariable=self.playback_var, values=("0.1", "1", "10", "100", "1000", "3600", "86400"), width=8)
        rate.grid(row=2, column=2, sticky="ew")
        rate.bind("<<ComboboxSelected>>", lambda _e: self._playback_changed())
        rate.bind("<Return>", lambda _e: self._playback_changed())
        ttk.Checkbutton(timeline_box, text="Loop", variable=self.loop_var, command=self._playback_changed).grid(row=2, column=3, sticky="w")
        ttk.Label(timeline_box, text="Trail seconds").grid(row=3, column=0, sticky="w")
        ttk.Entry(timeline_box, textvariable=self.trail_var, width=10).grid(row=3, column=1, sticky="ew")
        ttk.Label(timeline_box, text="Follow").grid(row=3, column=2, sticky="e")
        self.follow_box = ttk.Combobox(timeline_box, textvariable=self.follow_var, values=("World",), state="readonly", width=12)
        self.follow_box.grid(row=3, column=3, sticky="ew")
        self.follow_box.bind("<<ComboboxSelected>>", lambda _e: self._render_time_state())
        ttk.Label(timeline_box, text="Event").grid(row=4, column=0, sticky="w")
        self.event_box = ttk.Combobox(timeline_box, textvariable=self.event_var, values=(), state="readonly", width=18)
        self.event_box.grid(row=4, column=1, columnspan=2, sticky="ew")
        ttk.Button(timeline_box, text="Jump", command=self._jump_event).grid(row=4, column=3, sticky="ew")
        timeline_box.columnconfigure(0, weight=1)
        timeline_box.columnconfigure(2, weight=1)
        row += 1

        ttk.Label(controls, text="Validation", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        ttk.Label(controls, textvariable=self.validation_var, wraplength=310, justify="left").grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1
        ttk.Label(controls, textvariable=self.status_var, wraplength=310, justify="left").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        self.renderer = self.d["vtkRenderer"]()
        self.renderer.SetBackground(0.06, 0.07, 0.09)
        self.interactor = self.d["vtkTkRenderWindowInteractor"](view, width=1050, height=850)
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
        experiment = replace_metric_parameters(experiment, {n: v.get() for n, v in self._parameter_vars.items()})
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
        self.frame_cache.clear()
        self._volume_state = None
        self._update_field_choices()
        diagnostics = result.metadata.get("diagnostics", {})
        self.validation_var.set(f"Pipeline validation status: {diagnostics.get('status', 'unknown')}")
        self.status_var.set(f"Loaded {result.metric_name}; grid {result.metadata.get('shape', '?')}")
        self._sync_timeline_range()
        self._field_changed()

    def _update_field_choices(self):
        fields = [name for name in RANK2_FIELDS if self.result is None or name in self.result.fields]
        if self.result is not None and "ricci_scalar" in self.result.fields:
            fields.append("ricci_scalar")
        self.field_box.configure(values=fields)
        if fields and self.field_var.get() not in fields:
            self.field_var.set(fields[0])

    def _open(self):
        path = self.d["filedialog"].askdirectory(title="Open Tensor Toolkit tensor result")
        if not path:
            return
        try:
            metadata, fields, axes = load_result(path)
            self._accept_result(ExperimentResult(
                metric_name=str(metadata.get("metric_name", "unknown")),
                coordinates=tuple(metadata.get("coordinates", ("t", "x", "y", "z"))),
                axis_values=tuple(axes),
                fields=fields,
                metadata={k: v for k, v in metadata.items() if k not in {"metric_name", "coordinates", "fields"}},
            ))
        except Exception as exc:
            self.d["messagebox"].showerror("Open result", str(exc))

    def _open_trajectory(self):
        path = self.d["filedialog"].askdirectory(title="Open saved classical simulation")
        if not path:
            return
        try:
            self.trajectory = load_saved_trajectory(path)
        except Exception as exc:
            self.d["messagebox"].showerror("Open trajectory", str(exc))
            return
        self._build_trajectory_scene()
        self._sync_timeline_range()
        self.status_var.set(
            f"Loaded trajectory with {len(self.trajectory.body_names)} bodies and "
            f"{len(self.trajectory.events)} events"
        )
        self._render_time_state()

    def _save(self):
        if self.result is None:
            self.d["messagebox"].showinfo("Save result", "Run or open a tensor result first.")
            return
        path = self.d["filedialog"].askdirectory(title="Choose result directory")
        if path:
            try:
                save_result(self.result, Path(path))
                self.status_var.set(f"Saved result to {path}")
            except Exception as exc:
                self.d["messagebox"].showerror("Save result", str(exc))

    def _sync_timeline_range(self):
        ranges = []
        if self.result is not None:
            t = np.asarray(self.result.axis_values[0], dtype=float)
            ranges.append((float(t[0]), float(t[-1])))
        if self.trajectory is not None:
            ranges.append((float(self.trajectory.times[0]), float(self.trajectory.times[-1])))
        if not ranges:
            return
        start = min(v[0] for v in ranges)
        stop = max(v[1] for v in ranges)
        self.timeline.set_range(start, stop)
        self.timeline.seek(start)
        self.timeline_scale.configure(from_=start, to=stop if stop > start else start + 1.0)
        self.timeline_var.set(start)
        self._update_time_text()

    def _toggle_play(self):
        self._playback_changed()
        playing = self.timeline.toggle()
        self.play_text_var.set("❚❚ Pause" if playing else "▶ Play")

    def _playback_changed(self):
        try:
            rate = float(self.playback_var.get())
            if rate <= 0 or not np.isfinite(rate):
                raise ValueError
        except ValueError:
            self.status_var.set("Playback rate must be a positive finite number")
            return
        self.timeline.playback_rate = rate
        self.timeline.loop = bool(self.loop_var.get())

    def _timeline_scrub(self, value):
        self.timeline.pause()
        self.play_text_var.set("▶ Play")
        self.timeline.seek(float(value))
        self.timeline_var.set(self.timeline.current)
        self._render_time_state()

    def _tick(self):
        before = self.timeline.current
        now = self.timeline.advance()
        if now != before:
            self.timeline_var.set(now)
            self._render_time_state()
        if not self.timeline.playing and self.play_text_var.get() != "▶ Play":
            self.play_text_var.set("▶ Play")
        self.root.after(self.TICK_MS, self._tick)

    def _field_changed(self):
        self.frame_cache.clear()
        self._volume_state = None
        self._render_time_state(force_field_rebuild=True)

    def _tensor_frame_index(self):
        if self.result is None:
            return None
        return self.timeline.nearest_index(self.result.axis_values[0])

    def _cached_volume(self, index):
        key = (self.field_var.get(), int(self.mu_var.get()), int(self.nu_var.get()), int(index))
        return self.frame_cache.get(
            key,
            lambda: experiment_volume(
                self.result, self.field_var.get(),
                component=(int(self.mu_var.get()), int(self.nu_var.get())),
                time_index=int(index),
            ),
        )

    def _prefetch_nearby(self, index):
        if self.result is None:
            return
        nt = len(self.result.axis_values[0])
        for i in range(max(0, index - 2), min(nt, index + 3)):
            self._cached_volume(i)

    def _vtk_image(self, volume):
        image = self.d["vtkImageData"]()
        nx, ny, nz = volume.values.shape
        image.SetDimensions(nx, ny, nz)
        axes = (volume.x, volume.y, volume.z)
        image.SetOrigin(*(float(a[0]) for a in axes))
        image.SetSpacing(*(float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in axes))
        self._set_image_scalars(image, volume)
        return image

    def _set_image_scalars(self, image, volume):
        flat = np.ascontiguousarray(volume.values).ravel(order="F")
        vtk_values = self.d["numpy_to_vtk"](flat, deep=True)
        vtk_values.SetName(volume.name)
        image.GetPointData().SetScalars(vtk_values)
        image.GetPointData().Modified()
        image.Modified()

    def _ensure_field_actor(self, volume, force=False):
        spec = (self.field_var.get(), int(self.mu_var.get()), int(self.nu_var.get()), self.mode_var.get())
        if not force and self._volume_state is not None and self._volume_state["spec"] == spec:
            self._set_image_scalars(self._volume_state["image"], volume)
            return

        if self._volume_state is not None:
            self.renderer.RemoveViewProp(self._volume_state["actor"])

        image = self._vtk_image(volume)
        finite = volume.values[np.isfinite(volume.values)]
        if finite.size == 0:
            raise ValueError("selected volume contains no finite values")
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
            pipeline = contour
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
            pipeline = mapper

        self._volume_state = {"spec": spec, "image": image, "actor": actor, "pipeline": pipeline, "range": (vmin, vmax)}

    def _polyline_actor(self, points, width=2.0):
        vtk_points = self.d["vtkPoints"]()
        polyline = self.d["vtkPolyLine"]()
        n = len(points)
        polyline.GetPointIds().SetNumberOfIds(n)
        for i, point in enumerate(points):
            vtk_points.InsertNextPoint(*map(float, point))
            polyline.GetPointIds().SetId(i, i)
        cells = self.d["vtkCellArray"]()
        if n >= 2:
            cells.InsertNextCell(polyline)
        data = self.d["vtkPolyData"]()
        data.SetPoints(vtk_points)
        data.SetLines(cells)
        mapper = self.d["vtkPolyDataMapper"]()
        mapper.SetInputData(data)
        actor = self.d["vtkActor"]()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(width)
        return actor, data

    def _build_trajectory_scene(self):
        trail_actors = [value[0] for value in self._trail_actors.values()]
        for actor in list(self._body_actors.values()) + trail_actors + self._event_actors:
            self.renderer.RemoveActor(actor)
        self._body_actors.clear()
        self._trail_actors.clear()
        self._event_actors.clear()

        if self.trajectory is None:
            return

        spans = np.ptp(self.trajectory.positions.reshape(-1, 3), axis=0)
        radius = max(float(np.max(spans)) * 0.01, 1e-6)
        for name in self.trajectory.body_names:
            sphere = self.d["vtkSphereSource"]()
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            mapper = self.d["vtkPolyDataMapper"]()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = self.d["vtkActor"]()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
            self._body_actors[name] = actor

            trail_actor, trail_data = self._polyline_actor(np.zeros((1, 3)), width=2.0)
            self.renderer.AddActor(trail_actor)
            self._trail_actors[name] = (trail_actor, trail_data)

        event_labels = []
        event_points = trajectory_event_points(self.trajectory)
        event_radius = radius * 0.65
        for i, event in enumerate(self.trajectory.events):
            event_labels.append(f"{i}: {event.kind} @ {event.time:.6g}")
            sphere = self.d["vtkSphereSource"]()
            sphere.SetRadius(event_radius)
            sphere.SetThetaResolution(12)
            sphere.SetPhiResolution(12)
            mapper = self.d["vtkPolyDataMapper"]()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = self.d["vtkActor"]()
            actor.SetMapper(mapper)
            if i < len(event_points.points):
                actor.SetPosition(*map(float, event_points.points[i]))
            self.renderer.AddActor(actor)
            self._event_actors.append(actor)
        self.event_box.configure(values=event_labels)
        if event_labels:
            self.event_var.set(event_labels[0])

        self.follow_box.configure(values=("World", *self.trajectory.body_names))
        if self.follow_var.get() not in ("World", *self.trajectory.body_names):
            self.follow_var.set("World")

    def _update_polydata_line(self, polydata, points):
        vtk_points = self.d["vtkPoints"]()
        cells = self.d["vtkCellArray"]()
        if len(points) >= 2:
            line = self.d["vtkPolyLine"]()
            line.GetPointIds().SetNumberOfIds(len(points))
            for i, point in enumerate(points):
                vtk_points.InsertNextPoint(*map(float, point))
                line.GetPointIds().SetId(i, i)
            cells.InsertNextCell(line)
        elif len(points) == 1:
            vtk_points.InsertNextPoint(*map(float, points[0]))
        polydata.SetPoints(vtk_points)
        polydata.SetLines(cells)
        polydata.Modified()

    def _update_trajectory_scene(self):
        if self.trajectory is None:
            return
        t = float(np.clip(self.timeline.current, self.trajectory.times[0], self.trajectory.times[-1]))
        positions = sample_trajectory_positions(self.trajectory, t)
        duration = float(self.trail_var.get())
        for i, name in enumerate(self.trajectory.body_names):
            self._body_actors[name].SetPosition(*map(float, positions[i]))
            trail = trajectory_trail(self.trajectory, name, t, duration=duration)
            actor, data = self._trail_actors[name]
            self._update_polydata_line(data, trail)

    def _update_follow_camera(self):
        if self.trajectory is None or self.follow_var.get() == "World":
            return
        name = self.follow_var.get()
        if name not in self.trajectory.body_names:
            return
        idx = self.trajectory.body_names.index(name)
        t = float(np.clip(self.timeline.current, self.trajectory.times[0], self.trajectory.times[-1]))
        target = sample_trajectory_positions(self.trajectory, t)[idx]
        camera = self.renderer.GetActiveCamera()
        old_focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        old_pos = np.asarray(camera.GetPosition(), dtype=float)
        offset = old_pos - old_focal
        camera.SetFocalPoint(*map(float, target))
        camera.SetPosition(*map(float, target + offset))

    def _jump_event(self):
        if self.trajectory is None or not self.trajectory.events:
            return
        try:
            index = int(self.event_var.get().split(":", 1)[0])
            event = self.trajectory.events[index]
        except Exception:
            return
        self.timeline.pause()
        self.play_text_var.set("▶ Play")
        self.timeline.seek(event.time)
        self.timeline_var.set(self.timeline.current)
        self._render_time_state()

    def _update_time_text(self, frame_index=None):
        text = f"t = {self.timeline.current:.6g}"
        if frame_index is not None and self.result is not None:
            frame_time = float(self.result.axis_values[0][frame_index])
            text += f"   tensor frame {frame_index} @ {frame_time:.6g}"
        self.timeline_text_var.set(text)

    def _render_time_state(self, force_field_rebuild=False):
        frame_index = self._tensor_frame_index()
        if frame_index is not None:
            try:
                volume = self._cached_volume(frame_index)
                self._ensure_field_actor(volume, force=force_field_rebuild)
                self._prefetch_nearby(frame_index)
            except Exception as exc:
                self.status_var.set(str(exc))
        self._update_trajectory_scene()
        self._update_follow_camera()
        self._update_time_text(frame_index)

        if not self._camera_initialized and (self.result is not None or self.trajectory is not None):
            self.renderer.ResetCamera()
            self._camera_initialized = True
        self.interactor.GetRenderWindow().Render()


def main() -> int:
    d = _dependencies()
    root = d["tk"].Tk()
    TensorToolkitVTKGUI(root)
    root.mainloop()
    return 0


__all__ = ["TensorToolkitVTKGUI", "main"]

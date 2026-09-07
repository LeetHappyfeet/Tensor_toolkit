"""Qt/VTK desktop visualizer for Tensor Toolkit.

The GUI is downstream of tensor_toolkit.visualization_data and
visualization_timeline. It displays stored solver results and never advances
or modifies the physics engine.
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
INDEX_CHOICES = ("t (0)", "x (1)", "y (2)", "z (3)")
FIELD_DESCRIPTIONS = {
    "metric": "Metric tensor g_μν: spacetime geometry in the selected coordinate basis.",
    "inverse_metric": "Inverse metric g^μν: inverse spacetime geometry in the selected coordinate basis.",
    "ricci": "Ricci tensor R_μν: curvature contraction associated with matter/energy content.",
    "ricci_scalar": "Ricci scalar R: scalar curvature; μ and ν do not apply.",
    "einstein": "Einstein tensor G_μν: curvature combination used by Einstein's field equation.",
    "stress_energy": "Stress-energy tensor T_μν: energy density, momentum density, and stresses.",
}


def _dependencies():
    try:
        from PySide6 import QtCore, QtWidgets
        from PySide6.QtCore import Signal
    except ImportError as exc:
        raise RuntimeError(
            'Qt is required for the VTK desktop visualizer. Install with '
            'python -m pip install -e ".[visualization]"'
        ) from exc
    try:
        import vtk
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.util.numpy_support import numpy_to_vtk
    except Exception as exc:
        raise RuntimeError(
            "VTK is installed but its Qt rendering bindings could not be loaded "
            f"({type(exc).__name__}: {exc})."
        ) from exc
    return QtCore, QtWidgets, Signal, vtk, QVTKRenderWindowInteractor, numpy_to_vtk


QtCore, QtWidgets, Signal, vtk, QVTKRenderWindowInteractor, numpy_to_vtk = _dependencies()


class _WorkerSignals(QtCore.QObject):
    finished = Signal(object)
    failed = Signal(str)


class TensorToolkitVTKGUI(QtWidgets.QMainWindow):
    """Interactive VTK field/trajectory viewer over authoritative stored results."""

    TICK_MS = 16
    SLIDER_STEPS = 10000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tensor Toolkit — VTK Scientific Visualizer")
        self.resize(1550, 920)

        self.result = None
        self.trajectory = None
        self.timeline = VisualizationTimeline(0.0, 1.0, 0.0)
        self.frame_cache = FrameCache(5)
        self._volume_state = None
        self._body_actors = {}
        self._trail_actors = {}
        self._event_actors = []
        self._camera_initialized = False
        self._parameter_edits = {}
        self._worker_signals = _WorkerSignals()
        self._worker_signals.finished.connect(self._accept_result)
        self._worker_signals.failed.connect(self._worker_failed)

        self._build_layout()
        self._metric_changed()
        self._update_field_choices()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.TICK_MS)

    def _build_layout(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls = QtWidgets.QWidget()
        controls_scroll.setWidget(controls)
        controls.setMinimumWidth(330)
        layout = QtWidgets.QVBoxLayout(controls)

        exp_group = QtWidgets.QGroupBox("Experiment")
        exp_layout = QtWidgets.QFormLayout(exp_group)
        self.metric_box = QtWidgets.QComboBox()
        self.metric_box.addItems(sorted(builtins()))
        self.metric_box.setCurrentText("alcubierre")
        self.metric_box.currentTextChanged.connect(self._metric_changed)
        exp_layout.addRow("Metric", self.metric_box)

        self.parameter_group = QtWidgets.QGroupBox("Metric parameters")
        self.parameter_layout = QtWidgets.QFormLayout(self.parameter_group)
        exp_layout.addRow(self.parameter_group)

        self.points_spin = QtWidgets.QSpinBox()
        self.points_spin.setRange(3, 257)
        self.points_spin.setValue(9)
        exp_layout.addRow("Points / axis", self.points_spin)

        self.extent_spin = QtWidgets.QDoubleSpinBox()
        self.extent_spin.setRange(1e-12, 1e12)
        self.extent_spin.setDecimals(6)
        self.extent_spin.setValue(2.0)
        exp_layout.addRow("Extent ±", self.extent_spin)
        layout.addWidget(exp_group)

        output_group = QtWidgets.QGroupBox("Retained fields")
        output_layout = QtWidgets.QVBoxLayout(output_group)
        self.output_checks = {}
        for name in GUI_OUTPUTS:
            check = QtWidgets.QCheckBox(name)
            check.setChecked(name in {"metric", "einstein", "stress_energy"})
            output_layout.addWidget(check)
            self.output_checks[name] = check
        layout.addWidget(output_group)

        run_row = QtWidgets.QGridLayout()
        self.run_button = QtWidgets.QPushButton("Run experiment")
        self.run_button.clicked.connect(self._run)
        run_row.addWidget(self.run_button, 0, 0)
        open_tensor = QtWidgets.QPushButton("Open tensor result")
        open_tensor.clicked.connect(self._open_tensor)
        run_row.addWidget(open_tensor, 0, 1)
        open_traj = QtWidgets.QPushButton("Open trajectory")
        open_traj.clicked.connect(self._open_trajectory)
        run_row.addWidget(open_traj, 1, 0)
        save_tensor = QtWidgets.QPushButton("Save tensor result")
        save_tensor.clicked.connect(self._save)
        run_row.addWidget(save_tensor, 1, 1)
        layout.addLayout(run_row)

        field_group = QtWidgets.QGroupBox("3-D field")
        field_layout = QtWidgets.QFormLayout(field_group)
        self.field_box = QtWidgets.QComboBox()
        self.field_box.currentTextChanged.connect(self._field_changed)
        field_layout.addRow("Field", self.field_box)
        self.mu_box = QtWidgets.QComboBox()
        self.mu_box.addItems(INDEX_CHOICES)
        self.mu_box.currentIndexChanged.connect(self._field_changed)
        self.mu_box.setToolTip("First tensor index μ: 0=t, 1=x, 2=y, 3=z.")
        field_layout.addRow("First index μ", self.mu_box)
        self.nu_box = QtWidgets.QComboBox()
        self.nu_box.addItems(INDEX_CHOICES)
        self.nu_box.currentIndexChanged.connect(self._field_changed)
        self.nu_box.setToolTip("Second tensor index ν: 0=t, 1=x, 2=y, 3=z.")
        field_layout.addRow("Second index ν", self.nu_box)
        self.component_label = QtWidgets.QLabel()
        self.component_label.setWordWrap(True)
        self.component_label.setMinimumWidth(270)
        field_layout.addRow("Displayed quantity", self.component_label)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(("volume", "isosurface"))
        self.mode_box.currentTextChanged.connect(self._field_changed)
        field_layout.addRow("Mode", self.mode_box)
        layout.addWidget(field_group)

        time_group = QtWidgets.QGroupBox("Visualization timeline")
        time_layout = QtWidgets.QGridLayout(time_group)
        self.time_label = QtWidgets.QLabel("t = 0")
        time_layout.addWidget(self.time_label, 0, 0, 1, 4)

        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, self.SLIDER_STEPS)
        self.timeline_slider.sliderPressed.connect(self._pause_for_scrub)
        self.timeline_slider.valueChanged.connect(self._timeline_slider_changed)
        time_layout.addWidget(self.timeline_slider, 1, 0, 1, 4)

        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        time_layout.addWidget(self.play_button, 2, 0)

        self.rate_box = QtWidgets.QComboBox()
        self.rate_box.setEditable(True)
        self.rate_box.addItems(("0.1", "1", "10", "100", "1000", "3600", "86400"))
        self.rate_box.setCurrentText("1")
        self.rate_box.currentTextChanged.connect(self._playback_changed)
        time_layout.addWidget(QtWidgets.QLabel("Rate"), 2, 1)
        time_layout.addWidget(self.rate_box, 2, 2)

        self.loop_check = QtWidgets.QCheckBox("Loop")
        self.loop_check.toggled.connect(self._playback_changed)
        time_layout.addWidget(self.loop_check, 2, 3)

        self.trail_spin = QtWidgets.QDoubleSpinBox()
        self.trail_spin.setRange(0.0, 1e15)
        self.trail_spin.setDecimals(3)
        self.trail_spin.valueChanged.connect(self._render_time_state)
        time_layout.addWidget(QtWidgets.QLabel("Trail seconds"), 3, 0)
        time_layout.addWidget(self.trail_spin, 3, 1)

        self.follow_box = QtWidgets.QComboBox()
        self.follow_box.addItem("World")
        self.follow_box.currentTextChanged.connect(self._render_time_state)
        time_layout.addWidget(QtWidgets.QLabel("Follow"), 3, 2)
        time_layout.addWidget(self.follow_box, 3, 3)

        self.event_box = QtWidgets.QComboBox()
        time_layout.addWidget(QtWidgets.QLabel("Event"), 4, 0)
        time_layout.addWidget(self.event_box, 4, 1, 1, 2)
        jump = QtWidgets.QPushButton("Jump")
        jump.clicked.connect(self._jump_event)
        time_layout.addWidget(jump, 4, 3)
        layout.addWidget(time_group)

        camera_group = QtWidgets.QGroupBox("3-D camera / study controls")
        camera_layout = QtWidgets.QGridLayout(camera_group)
        self.parallel_check = QtWidgets.QCheckBox("Orthographic projection")
        self.parallel_check.toggled.connect(self._set_projection)
        camera_layout.addWidget(self.parallel_check, 0, 0, 1, 2)
        reset_camera = QtWidgets.QPushButton("Reset / fit")
        reset_camera.clicked.connect(self._reset_camera)
        camera_layout.addWidget(reset_camera, 0, 2)
        for index, (label, view_name) in enumerate((
            ("+X", "pos_x"), ("-X", "neg_x"),
            ("+Y", "pos_y"), ("-Y", "neg_y"),
            ("+Z", "pos_z"), ("-Z", "neg_z"),
            ("Iso", "iso"),
        )):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _checked=False, name=view_name: self._set_camera_view(name))
            camera_layout.addWidget(button, 1 + index // 4, index % 4)
        camera_hint = QtWidgets.QLabel(
            "Study mode uses trackball camera control: the scene rotates only while you drag. "
            "Use fixed axis views to return to a repeatable orientation."
        )
        camera_hint.setWordWrap(True)
        camera_layout.addWidget(camera_hint, 3, 0, 1, 4)
        layout.addWidget(camera_group)

        validation_group = QtWidgets.QGroupBox("Validation")
        validation_layout = QtWidgets.QVBoxLayout(validation_group)
        self.validation_label = QtWidgets.QLabel("No result loaded")
        self.validation_label.setWordWrap(True)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)
        validation_layout.addWidget(self.validation_label)
        validation_layout.addWidget(self.status_label)
        layout.addWidget(validation_group)
        layout.addStretch(1)

        render_container = QtWidgets.QFrame()
        render_layout = QtWidgets.QVBoxLayout(render_container)
        render_layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(render_container)
        render_layout.addWidget(self.vtk_widget)

        splitter.addWidget(controls_scroll)
        splitter.addWidget(render_container)
        splitter.setStretchFactor(1, 1)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.06, 0.07, 0.09)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.Initialize()

        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        self.orientation_marker = vtk.vtkOrientationMarkerWidget()
        self.orientation_marker.SetOrientationMarker(axes)
        self.orientation_marker.SetInteractor(self.interactor)
        self.orientation_marker.SetViewport(0.0, 0.0, 0.16, 0.16)
        self.orientation_marker.SetEnabled(1)
        self.orientation_marker.InteractiveOff()

    def _metric_changed(self, *_):
        while self.parameter_layout.rowCount():
            self.parameter_layout.removeRow(0)
        self._parameter_edits.clear()
        experiment = get_experiment(self.metric_box.currentText())
        for name, value in editable_metric_parameters(experiment.metric).items():
            edit = QtWidgets.QDoubleSpinBox()
            edit.setDecimals(9)
            edit.setRange(-1e15, 1e15)
            edit.setValue(float(value))
            self.parameter_layout.addRow(name, edit)
            self._parameter_edits[name] = edit

    def _selected_outputs(self):
        outputs = frozenset(name for name, box in self.output_checks.items() if box.isChecked())
        if not outputs:
            raise ValueError("select at least one retained field")
        return outputs

    def _build_experiment(self):
        experiment = get_experiment(self.metric_box.currentText())
        experiment = replace_metric_parameters(
            experiment, {name: edit.value() for name, edit in self._parameter_edits.items()}
        )
        experiment = configure_grid(
            experiment, points=self.points_spin.value(), extent=self.extent_spin.value()
        )
        return replace(experiment, outputs=self._selected_outputs())

    def _run(self):
        try:
            experiment = self._build_experiment()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid experiment", str(exc))
            return
        self.run_button.setEnabled(False)
        self.status_label.setText("Calculating on the validated CPU/NumPy pipeline…")

        def worker():
            try:
                result = run_experiment(experiment)
            except Exception as exc:
                self._worker_signals.failed.emit(str(exc))
                return
            self._worker_signals.finished.emit(result)

        threading.Thread(target=worker, daemon=True).start()

    @QtCore.Slot(str)
    def _worker_failed(self, message):
        self.run_button.setEnabled(True)
        self.status_label.setText("Calculation failed")
        QtWidgets.QMessageBox.critical(self, "Tensor Toolkit", message)

    def _clear_field_actor(self):
        if self._volume_state is not None:
            self.renderer.RemoveViewProp(self._volume_state["actor"])
        self._volume_state = None

    @QtCore.Slot(object)
    def _accept_result(self, result):
        self.run_button.setEnabled(True)
        self.result = result
        self.frame_cache.clear()
        self._clear_field_actor()
        self._camera_initialized = False
        self._update_field_choices()
        diagnostics = result.metadata.get("diagnostics", {})
        self.validation_label.setText(
            f"Pipeline validation status: {diagnostics.get('status', 'unknown')}"
        )
        self.status_label.setText(
            f"Loaded {result.metric_name}; grid {result.metadata.get('shape', '?')}"
        )
        self._sync_timeline_range()
        self._field_changed()

    def _update_field_choices(self):
        previous = self.field_box.currentText()
        self.field_box.blockSignals(True)
        self.field_box.clear()
        fields = [name for name in RANK2_FIELDS if self.result is None or name in self.result.fields]
        if self.result is not None and "ricci_scalar" in self.result.fields:
            fields.append("ricci_scalar")
        self.field_box.addItems(fields)
        if previous in fields:
            self.field_box.setCurrentText(previous)
        elif "stress_energy" in fields:
            self.field_box.setCurrentText("stress_energy")
        self.field_box.blockSignals(False)

    def _open_tensor(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Tensor Toolkit tensor result")
        if not path:
            return
        try:
            metadata, fields, axes = load_result(path)
            result = ExperimentResult(
                metric_name=str(metadata.get("metric_name", "unknown")),
                coordinates=tuple(metadata.get("coordinates", ("t", "x", "y", "z"))),
                axis_values=tuple(axes),
                fields=fields,
                metadata={k: v for k, v in metadata.items() if k not in {"metric_name", "coordinates", "fields"}},
            )
            self._accept_result(result)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open result", str(exc))

    def _open_trajectory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open saved classical simulation")
        if not path:
            return
        try:
            self.trajectory = load_saved_trajectory(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open trajectory", str(exc))
            return
        self._build_trajectory_scene()
        self._camera_initialized = False
        self._sync_timeline_range()
        self.status_label.setText(
            f"Loaded trajectory with {len(self.trajectory.body_names)} bodies and "
            f"{len(self.trajectory.events)} events"
        )
        self._render_time_state()

    def _save(self):
        if self.result is None:
            QtWidgets.QMessageBox.information(self, "Save result", "Run or open a tensor result first.")
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose result directory")
        if not path:
            return
        try:
            save_result(self.result, Path(path))
            self.status_label.setText(f"Saved result to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save result", str(exc))

    def _sync_timeline_range(self):
        ranges = []
        if self.result is not None:
            t = np.asarray(self.result.axis_values[0], dtype=float)
            ranges.append((float(t[0]), float(t[-1])))
        if self.trajectory is not None:
            ranges.append((float(self.trajectory.times[0]), float(self.trajectory.times[-1])))
        if not ranges:
            return
        start = min(a for a, _ in ranges)
        stop = max(b for _, b in ranges)
        self.timeline.set_range(start, stop)
        self.timeline.seek(start)
        self._set_slider_from_time(start)
        self._update_time_text()

    def _toggle_play(self):
        self._playback_changed()
        playing = self.timeline.toggle()
        self.play_button.setText("❚❚ Pause" if playing else "▶ Play")

    def _playback_changed(self, *_):
        try:
            rate = float(self.rate_box.currentText())
            if rate <= 0 or not np.isfinite(rate):
                raise ValueError
        except ValueError:
            self.status_label.setText("Playback rate must be a positive finite number")
            return
        self.timeline.playback_rate = rate
        self.timeline.loop = self.loop_check.isChecked()

    def _pause_for_scrub(self):
        self.timeline.pause()
        self.play_button.setText("▶ Play")

    def _time_from_slider(self, slider_value):
        if self.timeline.stop <= self.timeline.start:
            return self.timeline.start
        fraction = float(slider_value) / self.SLIDER_STEPS
        return self.timeline.start + fraction * (self.timeline.stop - self.timeline.start)

    def _slider_from_time(self, time_value):
        if self.timeline.stop <= self.timeline.start:
            return 0
        fraction = (float(time_value) - self.timeline.start) / (self.timeline.stop - self.timeline.start)
        return int(np.clip(round(fraction * self.SLIDER_STEPS), 0, self.SLIDER_STEPS))

    def _set_slider_from_time(self, time_value):
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(self._slider_from_time(time_value))
        self.timeline_slider.blockSignals(False)

    def _timeline_slider_changed(self, value):
        if self.timeline_slider.isSliderDown():
            self.timeline.seek(self._time_from_slider(value))
            self._render_time_state()

    def _tick(self):
        before = self.timeline.current
        now = self.timeline.advance()
        if now != before:
            self._set_slider_from_time(now)
            self._render_time_state()
        if not self.timeline.playing and self.play_button.text() != "▶ Play":
            self.play_button.setText("▶ Play")

    def _field_changed(self, *_):
        self._update_component_description()
        self.frame_cache.clear()
        self._clear_field_actor()
        self._render_time_state(force_field_rebuild=True)

    def _update_component_description(self):
        field = self.field_box.currentText()
        description = FIELD_DESCRIPTIONS.get(field, field)
        if field == "ricci_scalar":
            component = "Scalar field R — no tensor component indices are used."
            self.mu_box.setEnabled(False)
            self.nu_box.setEnabled(False)
        else:
            self.mu_box.setEnabled(True)
            self.nu_box.setEnabled(True)
            mu = self.mu_box.currentIndex()
            nu = self.nu_box.currentIndex()
            names = ("t", "x", "y", "z")
            symbol = {
                "metric": "g", "inverse_metric": "g⁻¹", "ricci": "R",
                "einstein": "G", "stress_energy": "T",
            }.get(field, field)
            component = f"Component {symbol}_{{{names[mu]}{names[nu]}}}  [{mu},{nu}]"
            if field == "stress_energy":
                if mu == 0 and nu == 0:
                    component += " — energy-density component in this coordinate basis."
                elif mu == 0 or nu == 0:
                    component += " — energy/momentum-flow component in this coordinate basis."
                else:
                    component += " — spatial stress component in this coordinate basis."
        self.component_label.setText(f"{component}\n{description}")

    def _tensor_frame_index(self):
        if self.result is None:
            return None
        return self.timeline.nearest_index(self.result.axis_values[0])

    def _cached_volume(self, index):
        key = (self.field_box.currentText(), self.mu_box.currentIndex(), self.nu_box.currentIndex(), int(index))
        return self.frame_cache.get(
            key,
            lambda: experiment_volume(
                self.result,
                self.field_box.currentText(),
                component=(self.mu_box.currentIndex(), self.nu_box.currentIndex()),
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
        image = vtk.vtkImageData()
        nx, ny, nz = volume.values.shape
        image.SetDimensions(nx, ny, nz)
        axes = (volume.x, volume.y, volume.z)
        image.SetOrigin(*(float(a[0]) for a in axes))
        image.SetSpacing(*(float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in axes))
        self._set_image_scalars(image, volume)
        return image

    def _set_image_scalars(self, image, volume):
        flat = np.ascontiguousarray(volume.values).ravel(order="F")
        vtk_values = numpy_to_vtk(flat, deep=True)
        vtk_values.SetName(volume.name)
        image.GetPointData().SetScalars(vtk_values)
        image.GetPointData().Modified()
        image.Modified()

    def _volume_range(self, volume):
        finite = volume.values[np.isfinite(volume.values)]
        if finite.size == 0:
            raise ValueError("selected volume contains no finite values")
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if np.isclose(vmin, vmax):
            vmax = vmin + max(1.0, abs(vmin)) * 1e-12
        return vmin, vmax

    def _update_field_transfer(self, state, vmin, vmax):
        middle = 0.5 * (vmin + vmax)
        if state.get("contour") is not None:
            state["contour"].SetValue(0, middle)
            state["contour"].Modified()
        color, opacity = state.get("color"), state.get("opacity")
        if color is not None and opacity is not None:
            color.RemoveAllPoints()
            color.AddRGBPoint(vmin, 0.1, 0.2, 0.8)
            color.AddRGBPoint(middle, 0.9, 0.9, 0.9)
            color.AddRGBPoint(vmax, 0.8, 0.2, 0.1)
            opacity.RemoveAllPoints()
            opacity.AddPoint(vmin, 0.0)
            opacity.AddPoint(middle, 0.08)
            opacity.AddPoint(vmax, 0.65)
            color.Modified()
            opacity.Modified()
        state["range"] = (vmin, vmax)

    def _ensure_field_actor(self, volume, force=False):
        spec = (
            self.field_box.currentText(), self.mu_box.currentIndex(),
            self.nu_box.currentIndex(), self.mode_box.currentText()
        )
        vmin, vmax = self._volume_range(volume)
        if not force and self._volume_state is not None and self._volume_state["spec"] == spec:
            self._set_image_scalars(self._volume_state["image"], volume)
            self._update_field_transfer(self._volume_state, vmin, vmax)
            return
        if self._volume_state is not None:
            self.renderer.RemoveViewProp(self._volume_state["actor"])

        image = self._vtk_image(volume)
        if self.mode_box.currentText() == "isosurface":
            contour = vtk.vtkContourFilter()
            contour.SetInputData(image)
            contour.SetValue(0, 0.5 * (vmin + vmax))
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
            state = dict(contour=contour, color=None, opacity=None)
        else:
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(image)
            color = vtk.vtkColorTransferFunction()
            opacity = vtk.vtkPiecewiseFunction()
            prop = vtk.vtkVolumeProperty()
            prop.SetColor(color)
            prop.SetScalarOpacity(opacity)
            prop.ShadeOn()
            prop.SetInterpolationTypeToLinear()
            actor = vtk.vtkVolume()
            actor.SetMapper(mapper)
            actor.SetProperty(prop)
            self.renderer.AddVolume(actor)
            state = dict(contour=None, color=color, opacity=opacity)

        self._volume_state = {
            "spec": spec, "image": image, "actor": actor, "range": (vmin, vmax), **state
        }
        self._update_field_transfer(self._volume_state, vmin, vmax)

    def _polyline_actor(self, points, width=2.0):
        vtk_points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        data = vtk.vtkPolyData()
        if len(points) >= 2:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(points))
            for i, point in enumerate(points):
                vtk_points.InsertNextPoint(*map(float, point))
                line.GetPointIds().SetId(i, i)
            cells.InsertNextCell(line)
        elif len(points) == 1:
            vtk_points.InsertNextPoint(*map(float, points[0]))
        data.SetPoints(vtk_points)
        data.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        actor = vtk.vtkActor()
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
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)
            self._body_actors[name] = actor
            trail_actor, trail_data = self._polyline_actor(np.zeros((1, 3)), width=2.0)
            self.renderer.AddActor(trail_actor)
            self._trail_actors[name] = (trail_actor, trail_data)

        event_points = trajectory_event_points(self.trajectory)
        self.event_box.clear()
        for i, event in enumerate(self.trajectory.events):
            self.event_box.addItem(f"{i}: {event.kind} @ {event.time:.6g}")
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(radius * 0.65)
            sphere.SetThetaResolution(12)
            sphere.SetPhiResolution(12)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if i < len(event_points.points):
                actor.SetPosition(*map(float, event_points.points[i]))
            self.renderer.AddActor(actor)
            self._event_actors.append(actor)

        self.follow_box.blockSignals(True)
        self.follow_box.clear()
        self.follow_box.addItems(("World", *self.trajectory.body_names))
        self.follow_box.blockSignals(False)

    def _update_polydata_line(self, polydata, points):
        vtk_points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        if len(points) >= 2:
            line = vtk.vtkPolyLine()
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
        duration = self.trail_spin.value()
        for i, name in enumerate(self.trajectory.body_names):
            self._body_actors[name].SetPosition(*map(float, positions[i]))
            trail = trajectory_trail(self.trajectory, name, t, duration=duration)
            _actor, data = self._trail_actors[name]
            self._update_polydata_line(data, trail)

    def _update_follow_camera(self):
        if self.trajectory is None or self.follow_box.currentText() == "World":
            return
        name = self.follow_box.currentText()
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

    def _set_projection(self, enabled):
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelProjection(bool(enabled))
        self.vtk_widget.GetRenderWindow().Render()

    def _reset_camera(self):
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self._camera_initialized = True
        self.vtk_widget.GetRenderWindow().Render()

    def _set_camera_view(self, name):
        camera = self.renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        bounds = self.renderer.ComputeVisiblePropBounds()
        if bounds and all(np.isfinite(bounds)):
            spans = np.array([
                max(0.0, bounds[1] - bounds[0]),
                max(0.0, bounds[3] - bounds[2]),
                max(0.0, bounds[5] - bounds[4]),
            ])
            distance = max(float(np.max(spans)) * 2.5, 1.0)
            focal = np.array([
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ])
        else:
            distance = max(float(camera.GetDistance()), 1.0)

        directions = {
            "pos_x": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "neg_x": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "pos_y": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "neg_y": (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "pos_z": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
            "neg_z": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
            "iso": (np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), np.array([0.0, 0.0, 1.0])),
        }
        direction, up = directions[name]
        camera.SetFocalPoint(*map(float, focal))
        camera.SetPosition(*map(float, focal + distance * direction))
        camera.SetViewUp(*map(float, up))
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self._camera_initialized = True
        self.vtk_widget.GetRenderWindow().Render()

    def _jump_event(self):
        if self.trajectory is None or not self.trajectory.events:
            return
        index = self.event_box.currentIndex()
        if index < 0:
            return
        event = self.trajectory.events[index]
        self.timeline.pause()
        self.play_button.setText("▶ Play")
        self.timeline.seek(event.time)
        self._set_slider_from_time(self.timeline.current)
        self._render_time_state()

    def _update_time_text(self, frame_index=None):
        text = f"t = {self.timeline.current:.6g}"
        if frame_index is not None and self.result is not None:
            frame_time = float(self.result.axis_values[0][frame_index])
            text += f"   tensor frame {frame_index} @ {frame_time:.6g}"
        self.time_label.setText(text)

    def _render_time_state(self, *_args, force_field_rebuild=False):
        frame_index = self._tensor_frame_index()
        if frame_index is not None:
            try:
                volume = self._cached_volume(frame_index)
                self._ensure_field_actor(volume, force=force_field_rebuild)
                self._prefetch_nearby(frame_index)
            except Exception as exc:
                self.status_label.setText(str(exc))
        self._update_trajectory_scene()
        self._update_follow_camera()
        self._update_time_text(frame_index)
        if not self._camera_initialized and (self.result is not None or self.trajectory is not None):
            self.renderer.ResetCamera()
            self._camera_initialized = True
        self.vtk_widget.GetRenderWindow().Render()

    def closeEvent(self, event):
        try:
            self.vtk_widget.Finalize()
        finally:
            super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TensorToolkitVTKGUI()
    window.show()
    return app.exec()


__all__ = ["TensorToolkitVTKGUI", "main"]

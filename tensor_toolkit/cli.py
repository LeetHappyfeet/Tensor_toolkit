"""Command-line entry point for Tensor Toolkit."""

import argparse
import sys
from dataclasses import replace

import numpy as np

from tensor_toolkit.backends import require_backend
from tensor_toolkit.diagnostics import field_diagnostics
from tensor_toolkit.experiment import SUPPORTED_OUTPUTS, run_experiment
from tensor_toolkit.io import load_result, save_result
from tensor_toolkit.physics import (
    run_simulation_experiment,
    save_simulation_experiment_result,
    sample_schwarzschild_trajectory,
    save_schwarzschild_trajectory_samples,
    simulation_demos,
)
from tensor_toolkit.registry import builtins, configure_grid, get_experiment


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--points", type=int, default=None, help="points per t/x/y/z axis (minimum 3)")
    parser.add_argument("--extent", type=float, default=None, help="uniform domain [-extent,+extent] on all axes")


def _add_memory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-mode",
        choices=("auto", "in_memory", "tiled"),
        default="auto",
        help="RAM strategy; auto switches to tiled slabs when needed",
    )
    parser.add_argument(
        "--tile-points",
        type=int,
        default=8,
        help="core t points per tile in tiled mode (default: 8)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensor-toolkit",
        description="Tensor Toolkit CPU reference GR experiment runner",
    )
    parser.add_argument("--version", action="version", version="tensor-toolkit 0.2.0")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="list built-in experiments")

    run = sub.add_parser("run", help="run a built-in experiment")
    run.add_argument("experiment", choices=sorted(builtins()))
    run.add_argument("--output", default=None, help="directory for result.npz and metadata.json")
    run.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))
    run.add_argument(
        "--fields",
        nargs="+",
        choices=sorted(SUPPORTED_OUTPUTS),
        default=None,
        help="persist only these outputs; intermediates are discarded unless requested",
    )
    run.add_argument(
        "--storage-mode",
        choices=("auto", "memory", "disk"),
        default="auto",
        help="persistent output storage; disk writes memory-mapped .npy fields incrementally",
    )
    _add_grid_arguments(run)
    _add_memory_arguments(run)

    inspect = sub.add_parser("inspect", help="inspect a saved experiment result")
    inspect.add_argument("path", help="saved result directory")
    inspect.add_argument("--field", default=None, help="show only one saved field")
    inspect.add_argument("--center", action="store_true", help="print the selected rank-2 field at the grid center")

    simulate = sub.add_parser("simulate", help="run a classical many-body simulation demo")
    simulate.add_argument("experiment", choices=sorted(simulation_demos()))
    simulate.add_argument("--output", default=None, help="directory for trajectory.npz and metadata.json")
    simulate.add_argument("--duration", type=float, default=None, help="override demo duration in seconds")
    simulate.add_argument("--dt", type=float, default=None, help="override integration timestep in seconds")
    simulate.add_argument("--method", choices=("rk4", "verlet"), default=None, help="override integration method")
    simulate.add_argument("--sample-every", type=int, default=None, help="save every Nth trajectory sample")
    simulate.add_argument(
        "--schwarzschild",
        nargs=2,
        metavar=("PRIMARY", "BODY"),
        default=None,
        help="sample BODY along the Newtonian trajectory in a static Schwarzschild field centered on PRIMARY",
    )
    simulate.add_argument(
        "--relativity-samples",
        type=int,
        default=5,
        help="baseline evenly spaced Schwarzschild samples; matching encounter events are also included (default: 5)",
    )
    simulate.add_argument(
        "--gr-fields",
        nargs="+",
        choices=sorted(SUPPORTED_OUTPUTS),
        default=None,
        help="optional GR tensor fields to evaluate at Schwarzschild sample events",
    )
    simulate.add_argument(
        "--gr-spacing",
        type=float,
        default=None,
        help="uniform local GR stencil spacing in metres for ct/x/y/z",
    )

    convergence = sub.add_parser("convergence", help="run a grid-resolution Einstein-symmetry study")
    convergence.add_argument("experiment", choices=sorted(builtins()))
    convergence.add_argument("--points", type=int, nargs="+", required=True, help="resolutions to test, e.g. 5 7 9")
    convergence.add_argument("--extent", type=float, default=None, help="fixed uniform domain extent")
    convergence.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))

    sub.add_parser("visualize", help="launch the desktop metric tensor simulator")
    sub.add_parser("doctor", help="show supported execution capabilities")
    return parser


def _print_validation(result) -> None:
    validation = result.metadata.get("diagnostics", {})
    fields = validation.get("fields", {})
    print("Validation:")
    for name in ("metric", "einstein", "stress_energy"):
        item = fields.get(name)
        if not item:
            continue
        finite = "PASS" if item.get("finite") else "FAIL"
        symmetry = item.get("symmetry")
        if symmetry:
            print(
                f"  {name:14s} finite={finite} "
                f"symmetry_abs={symmetry['absolute']:.6g} "
                f"symmetry_rel={symmetry['relative']:.6g}"
            )
        else:
            print(f"  {name:14s} finite={finite}")
    print(f"  status: {validation.get('status', 'UNKNOWN')}")


def _print_memory(result) -> None:
    memory = result.metadata.get("memory", {})
    if not memory:
        return
    estimate = float(memory.get("estimated_selected_bytes", 0)) / 1024**3
    print(
        f"Memory: mode={memory.get('selected_mode', 'unknown')} "
        f"estimated_peak={estimate:.2f} GiB"
    )
    if memory.get("selected_mode") == "tiled":
        print(
            f"  tile_points={memory.get('tile_points')} halo={memory.get('halo')} "
            "(t-axis slabs)"
        )
    storage = result.metadata.get("storage", {})
    if storage:
        persistent = float(memory.get("persistent_output_bytes", 0)) / 1024**3
        print(
            f"Storage: mode={storage.get('mode', 'unknown')} "
            f"format={storage.get('format', 'unknown')} persistent={persistent:.2f} GiB"
        )


def _configured_experiment(
    name,
    backend,
    points=None,
    extent=None,
    memory_mode="auto",
    tile_points=8,
    fields=None,
):
    require_backend(backend)
    experiment = get_experiment(name)
    if fields is not None:
        experiment = replace(experiment, outputs=frozenset(fields))
    experiment = replace(
        experiment,
        backend=backend,
        memory_mode=memory_mode,
        tile_points=tile_points,
    )
    return configure_grid(experiment, points=points, extent=extent)


def _run(
    name,
    output,
    backend,
    points=None,
    extent=None,
    memory_mode="auto",
    tile_points=8,
    fields=None,
    storage_mode="auto",
) -> int:
    try:
        experiment = _configured_experiment(
            name, backend, points, extent, memory_mode, tile_points, fields
        )
    except (ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Running {name} on CPU...")
    print("  grid=" + "x".join(str(axis.points) for axis in experiment.axes))
    print("  domain=" + ", ".join(f"[{axis.start:g},{axis.stop:g}]" for axis in experiment.axes))
    print("  outputs=" + ", ".join(sorted(experiment.outputs)))
    try:
        result = run_experiment(
            experiment,
            output_path=output,
            storage_mode=storage_mode,
        )
    except (MemoryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for key, value in result.fields.items():
        print(f"  {key:14s} shape={value.shape} max|value|={float(abs(value).max()):.6g}")
    _print_memory(result)
    _print_validation(result)
    if output:
        print(f"Saved: {save_result(result, output)}")
    return 0


def _inspect(path: str, field: str | None = None, center: bool = False) -> int:
    try:
        metadata, fields, axes = load_result(path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if field is not None and field not in fields:
        print(f"ERROR: field {field!r} not found; available: {', '.join(sorted(fields))}", file=sys.stderr)
        return 2

    selected = [field] if field else sorted(fields)
    print(f"Metric: {metadata.get('metric_name', 'unknown')}")
    print(f"Coordinates: {', '.join(metadata.get('coordinates', []))}")
    print(f"Backend: {metadata.get('backend', 'unknown')}")
    print("Grid: " + " x ".join(str(len(axis)) for axis in axes))
    if axes:
        print("Axes: " + "; ".join(f"{axis[0]:g}..{axis[-1]:g} ({len(axis)})" for axis in axes))
    memory = metadata.get("memory", {})
    if memory:
        print(
            f"Execution: {memory.get('selected_mode', 'unknown')} "
            f"(estimated peak {float(memory.get('estimated_selected_bytes', 0))/1024**3:.2f} GiB)"
        )
    storage = metadata.get("storage", {})
    if storage:
        print(f"Storage: {storage.get('mode', 'unknown')} ({storage.get('format', 'unknown')})")
    print("Fields:")
    for name in selected:
        value = fields[name]
        diagnostics = field_diagnostics(value)
        print(
            f"  {name:14s} shape={value.shape} min={float(value.min()):.6g} "
            f"max={float(value.max()):.6g} max|value|={diagnostics['max_abs']:.6g}"
        )
        symmetry = diagnostics.get("symmetry")
        if symmetry:
            print(f"    symmetry_abs={symmetry['absolute']:.6g} symmetry_rel={symmetry['relative']:.6g}")
        if center:
            if value.ndim < 2 or value.shape[:2] != (4, 4):
                print("    center: unavailable for non-rank-2 tensor field")
            else:
                center_index = tuple(len(axis) // 2 for axis in axes)
                matrix = value[(slice(None), slice(None), *center_index)]
                print(f"    center index={center_index}")
                print(np.array2string(matrix, precision=8, suppress_small=False))

    stored = metadata.get("diagnostics", {})
    if stored:
        print(f"Stored validation status: {stored.get('status', 'UNKNOWN')}")
    return 0


def _convergence(name, point_counts, extent, backend) -> int:
    if any(points < 3 for points in point_counts):
        print("ERROR: all convergence --points values must be at least 3", file=sys.stderr)
        return 2
    if len(set(point_counts)) != len(point_counts):
        print("ERROR: convergence --points values must be unique", file=sys.stderr)
        return 2

    try:
        require_backend(backend)
        base = replace(get_experiment(name), backend=backend, outputs=frozenset({"einstein"}))
        if extent is not None and extent <= 0:
            raise ValueError("--extent must be positive")
    except (ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Convergence study: {name} (CPU)")
    print("points  spacing       max|G|        symmetry_abs   symmetry_rel")
    for points in sorted(point_counts):
        experiment = configure_grid(base, points=points, extent=extent)
        try:
            result = run_experiment(experiment)
        except MemoryError as exc:
            print(f"{points:6d}  ERROR: {exc}")
            continue
        einstein = result.fields["einstein"]
        symmetry = field_diagnostics(einstein)["symmetry"]
        spacing = max(result.metadata["spacings"])
        print(
            f"{points:6d}  {spacing:11.6g}  {float(np.max(np.abs(einstein))):12.6g}  "
            f"{symmetry['absolute']:12.6g}  {symmetry['relative']:12.6g}"
        )
    return 0


def _simulate_classical(
    name,
    output,
    duration=None,
    dt=None,
    method=None,
    sample_every=None,
    schwarzschild=None,
    relativity_samples=5,
    gr_fields=None,
    gr_spacing=None,
) -> int:
    try:
        experiment = simulation_demos()[name]
        updates = {}
        if duration is not None:
            if duration <= 0.0:
                raise ValueError("--duration must be positive")
            updates["duration"] = float(duration)
        if dt is not None:
            if dt <= 0.0:
                raise ValueError("--dt must be positive")
            updates["dt"] = float(dt)
        if method is not None:
            updates["method"] = method
        if sample_every is not None:
            if sample_every < 1:
                raise ValueError("--sample-every must be at least 1")
            updates["sample_every"] = int(sample_every)
        if updates:
            experiment = replace(experiment, **updates)
        result = run_simulation_experiment(experiment)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Classical simulation: {result.name}")
    print(
        f"  bodies={result.metadata['body_count']} "
        f"method={result.metadata['method']} "
        f"duration={result.metadata['duration']:.6g}s "
        f"dt={result.metadata['dt']:.6g}s"
    )
    for index, body_name in enumerate(result.metadata["body_names"]):
        start = result.trajectory.positions[0, index]
        end = result.trajectory.positions[-1, index]
        print(
            f"  body[{index}] {body_name}: "
            f"start={np.array2string(start, precision=6)} "
            f"end={np.array2string(end, precision=6)}"
        )

    conservation = result.conservation
    print("System conservation:")
    print(f"  energy_relative_drift={conservation.energy_relative_drift:.6g}")
    print(f"  momentum_absolute_drift={conservation.momentum_absolute_drift:.6g}")
    print(
        "  angular_momentum_relative_drift="
        f"{conservation.angular_momentum_relative_drift:.6g}"
    )

    if result.test_particles:
        print("Test-particle invariants:")
        for name, probe in result.test_particles.items():
            h0 = np.linalg.norm(probe.specific_angular_momentum[0])
            print(
                f"  {name}: orbit={probe.orbit_class} "
                f"specific_energy={probe.specific_energy[0]:.6g} J/kg "
                f"energy_drift={probe.specific_energy_relative_drift:.6g} "
                f"specific_h={h0:.6g} m^2/s "
                f"h_drift={probe.specific_angular_momentum_relative_drift:.6g}"
            )
            print(
                f"    e={probe.eccentricity:.6g} "
                f"periapsis={probe.periapsis_distance:.6g} m"
                + (
                    f" apoapsis={probe.apoapsis_distance:.6g} m "
                    f"period={probe.orbital_period:.6g} s"
                    if probe.orbit_class == "elliptic"
                    else ""
                )
            )

    if result.encounters:
        print("Encounters:")
        for name, encounter in result.encounters.items():
            orbit_class = (
                result.test_particles[name].orbit_class
                if name in result.test_particles
                else "unknown"
            )
            angle_label = (
                "finite_deflection"
                if orbit_class == "hyperbolic"
                else "velocity_direction_change"
            )
            print(
                f"  {name}: closest={encounter.closest_approach_distance:.6g} m "
                f"at t={encounter.closest_approach_time:.6g} s "
                f"periapsis_speed={encounter.periapsis_relative_speed:.6g} m/s "
                f"{angle_label}={np.degrees(encounter.deflection_angle):.6g} deg"
            )

    if result.hyperbolic_references:
        print("Analytic hyperbolic reference:")
        for name, reference in result.hyperbolic_references.items():
            print(
                f"  {name}: e={reference.eccentricity:.6g} "
                f"v_inf={reference.v_infinity:.6g} m/s "
                f"periapsis={reference.periapsis_distance:.6g} m "
                f"periapsis_speed={reference.periapsis_speed:.6g} m/s "
                f"asymptotic_deflection={np.degrees(reference.asymptotic_deflection_angle):.6g} deg"
            )
            print(
                f"    errors: periapsis={reference.numerical_periapsis_distance_error:.6g} m "
                f"periapsis_speed={reference.numerical_periapsis_speed_error:.6g} m/s "
                f"finite_vs_asymptotic_deflection="
                f"{np.degrees(reference.finite_window_deflection_error):.6g} deg"
            )

    if schwarzschild is not None:
        primary, body = schwarzschild
        if relativity_samples < 1:
            print("ERROR: --relativity-samples must be at least 1", file=sys.stderr)
            return 2
        if gr_fields is not None and (gr_spacing is None or gr_spacing <= 0.0):
            print("ERROR: --gr-spacing must be positive when --gr-fields is used", file=sys.stderr)
            return 2
        names = result.metadata["body_names"]
        if primary not in names or body not in names:
            print(
                f"ERROR: Schwarzschild bodies must be in simulation; available: {', '.join(names)}",
                file=sys.stderr,
            )
            return 2
        masses = dict(zip(names, experiment.system.masses))
        if masses[primary] <= 0.0:
            print("ERROR: Schwarzschild primary must have positive mass", file=sys.stderr)
            return 2

        sample_times = np.linspace(
            result.trajectory.times[0],
            result.trajectory.times[-1],
            relativity_samples,
        )
        encounter_key = f"{primary}->{body}"
        if encounter_key in result.encounters:
            sample_times = np.unique(
                np.append(
                    sample_times,
                    result.encounters[encounter_key].closest_approach_time,
                )
            )
        tensor_outputs = None if gr_fields is None else frozenset(gr_fields)
        tensor_spacings = None
        if tensor_outputs is not None:
            tensor_spacings = (gr_spacing, gr_spacing, gr_spacing, gr_spacing)
        try:
            relativity = sample_schwarzschild_trajectory(
                result.trajectory,
                primary=primary,
                body=body,
                primary_mass=float(masses[primary]),
                times=sample_times,
                tensor_outputs=tensor_outputs,
                tensor_spacings=tensor_spacings,
            )
        except ValueError as exc:
            print(f"ERROR: Schwarzschild sampling failed: {exc}", file=sys.stderr)
            return 2

        radius = np.linalg.norm(relativity.coordinates[:, 1:], axis=1)
        print("Schwarzschild sampling:")
        print(
            f"  primary={primary} body={body} samples={len(relativity.times)} "
            f"coordinate_system=(ct,x,y,z)"
        )
        for i in range(len(relativity.times)):
            print(
                f"  t={relativity.times[i]:.6g}s "
                f"r_iso={radius[i]:.6g}m "
                f"dτ/dt={relativity.proper_time_rate[i]:.12g} "
                f"τ_since_start={relativity.proper_time[i]:.9g}s"
            )
        if relativity.tensor_samples is not None:
            print("  GR fields:")
            for field_name, values in relativity.tensor_samples.fields.items():
                print(
                    f"    {field_name:14s} shape={values.shape} "
                    f"max|value|={float(np.max(np.abs(values))):.6g}"
                )

    if output:
        saved = save_simulation_experiment_result(result, output)
        if schwarzschild is not None:
            save_schwarzschild_trajectory_samples(relativity, output)
        print(f"Saved: {saved}")
    return 0


def _interactive() -> int:
    names = sorted(builtins())
    print("Tensor Toolkit 0.2.0\n")
    for index, name in enumerate(names, 1):
        print(f"{index}. {name}")
    raw = input("Select experiment: ").strip()
    try:
        name = names[int(raw) - 1] if raw.isdigit() else raw
    except (IndexError, ValueError):
        raise SystemExit("invalid selection")
    return _run(name, None, "cpu")


def _visualize() -> int:
    try:
        from tensor_toolkit.gui import main as gui_main
        return gui_main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        if sys.stdin.isatty():
            return _interactive()
        parser.print_help()
        return 0
    if args.command == "list":
        for name, experiment in sorted(builtins().items()):
            print(f"{name:12s} {experiment.metric.name}")
        return 0
    if args.command == "doctor":
        print(
            "Backend: CPU/NumPy float64 (supported)\n"
            "GPU: future upgrade; explicitly unsupported in 0.2.0\n"
            "Memory: preflight + automatic t-slab tiling with 3-cell halos\n"
            "Storage: optional disk-backed NumPy memmap output for large results\n"
            "Pipeline: metric -> inverse -> Christoffel -> streamed Ricci -> scalar -> Einstein -> stress-energy\n"
            "Full Riemann is allocated only when explicitly requested."
        )
        return 0
    if args.command == "inspect":
        return _inspect(args.path, args.field, args.center)
    if args.command == "simulate":
        return _simulate_classical(
            args.experiment,
            args.output,
            args.duration,
            args.dt,
            args.method,
            args.sample_every,
            args.schwarzschild,
            args.relativity_samples,
            args.gr_fields,
            args.gr_spacing,
        )
    if args.command == "convergence":
        return _convergence(args.experiment, args.points, args.extent, args.backend)
    if args.command == "visualize":
        return _visualize()
    return _run(
        args.experiment,
        args.output,
        args.backend,
        args.points,
        args.extent,
        args.memory_mode,
        args.tile_points,
        args.fields,
        args.storage_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())

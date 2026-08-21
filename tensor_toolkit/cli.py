"""Command-line entry point for Tensor Toolkit."""

import argparse
import sys
from dataclasses import replace

import numpy as np

from tensor_toolkit.backends import require_backend
from tensor_toolkit.diagnostics import field_diagnostics
from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.io import load_result, save_result
from tensor_toolkit.registry import builtins, configure_grid, get_experiment


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        help="points per t/x/y/z axis (minimum 3)",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=None,
        help="uniform domain [-extent,+extent] on all axes",
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
    _add_grid_arguments(run)

    inspect = sub.add_parser("inspect", help="inspect a saved experiment result")
    inspect.add_argument("path", help="saved result directory")
    inspect.add_argument("--field", default=None, help="show only one saved field")
    inspect.add_argument(
        "--center",
        action="store_true",
        help="print the selected rank-2 field at the grid center",
    )

    convergence = sub.add_parser(
        "convergence", help="run a grid-resolution Einstein-symmetry study"
    )
    convergence.add_argument("experiment", choices=sorted(builtins()))
    convergence.add_argument(
        "--points",
        type=int,
        nargs="+",
        required=True,
        help="resolutions to test, e.g. 5 7 9",
    )
    convergence.add_argument(
        "--extent",
        type=float,
        default=None,
        help="fixed uniform domain extent",
    )
    convergence.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))

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


def _configured_experiment(
    name: str,
    backend: str,
    points: int | None = None,
    extent: float | None = None,
):
    require_backend(backend)
    experiment = replace(get_experiment(name), backend=backend)
    return configure_grid(experiment, points=points, extent=extent)


def _run(
    name: str,
    output: str | None,
    backend: str,
    points: int | None = None,
    extent: float | None = None,
) -> int:
    try:
        experiment = _configured_experiment(name, backend, points, extent)
    except (ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Running {name} on CPU...")
    print("  grid=" + "x".join(str(axis.points) for axis in experiment.axes))
    print(
        "  domain="
        + ", ".join(f"[{axis.start:g},{axis.stop:g}]" for axis in experiment.axes)
    )
    result = run_experiment(experiment)
    for key, value in result.fields.items():
        print(
            f"  {key:14s} shape={value.shape} "
            f"max|value|={float(abs(value).max()):.6g}"
        )
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
        print(
            f"ERROR: field {field!r} not found; available: {', '.join(sorted(fields))}",
            file=sys.stderr,
        )
        return 2

    selected = [field] if field else sorted(fields)
    print(f"Metric: {metadata.get('metric_name', 'unknown')}")
    print(f"Coordinates: {', '.join(metadata.get('coordinates', []))}")
    print(f"Backend: {metadata.get('backend', 'unknown')}")
    print("Grid: " + " x ".join(str(len(axis)) for axis in axes))
    if axes:
        print(
            "Axes: "
            + "; ".join(
                f"{axis[0]:g}..{axis[-1]:g} ({len(axis)})" for axis in axes
            )
        )
    print("Fields:")
    for name in selected:
        value = fields[name]
        diagnostics = field_diagnostics(value)
        print(
            f"  {name:14s} shape={value.shape} min={float(value.min()):.6g} "
            f"max={float(value.max()):.6g} "
            f"max|value|={diagnostics['max_abs']:.6g}"
        )
        symmetry = diagnostics.get("symmetry")
        if symmetry:
            print(
                f"    symmetry_abs={symmetry['absolute']:.6g} "
                f"symmetry_rel={symmetry['relative']:.6g}"
            )
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


def _convergence(
    name: str,
    point_counts: list[int],
    extent: float | None,
    backend: str,
) -> int:
    if any(points < 3 for points in point_counts):
        print(
            "ERROR: all convergence --points values must be at least 3",
            file=sys.stderr,
        )
        return 2
    if len(set(point_counts)) != len(point_counts):
        print(
            "ERROR: convergence --points values must be unique",
            file=sys.stderr,
        )
        return 2

    try:
        require_backend(backend)
        base = replace(
            get_experiment(name),
            backend=backend,
            outputs=frozenset({"einstein"}),
        )
        if extent is not None and extent <= 0:
            raise ValueError("--extent must be positive")
    except (ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Convergence study: {name} (CPU)")
    print("points  spacing       max|G|        symmetry_abs   symmetry_rel")
    for points in sorted(point_counts):
        experiment = configure_grid(base, points=points, extent=extent)
        result = run_experiment(experiment)
        einstein = result.fields["einstein"]
        symmetry = field_diagnostics(einstein)["symmetry"]
        spacing = max(result.metadata["spacings"])
        print(
            f"{points:6d}  {spacing:11.6g}  "
            f"{float(np.max(np.abs(einstein))):12.6g}  "
            f"{symmetry['absolute']:12.6g}  {symmetry['relative']:12.6g}"
        )
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
            "Pipeline: metric -> inverse -> Christoffel -> Riemann -> "
            "Ricci -> scalar -> Einstein -> stress-energy"
        )
        return 0
    if args.command == "inspect":
        return _inspect(args.path, args.field, args.center)
    if args.command == "convergence":
        return _convergence(args.experiment, args.points, args.extent, args.backend)
    return _run(
        args.experiment,
        args.output,
        args.backend,
        args.points,
        args.extent,
    )


if __name__ == "__main__":
    raise SystemExit(main())

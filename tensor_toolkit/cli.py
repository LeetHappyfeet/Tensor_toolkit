"""Command-line entry point for Tensor Toolkit."""

import argparse
import sys
from dataclasses import replace

from tensor_toolkit.backends import require_backend
from tensor_toolkit.experiment import run_experiment
from tensor_toolkit.io import save_result
from tensor_toolkit.registry import builtins, get_experiment


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
    sub.add_parser("doctor", help="show supported execution capabilities")
    return parser


def _run(name: str, output: str | None, backend: str) -> int:
    try:
        require_backend(backend)
    except (ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    experiment = replace(get_experiment(name), backend=backend)
    print(f"Running {name} on CPU...")
    result = run_experiment(experiment)
    for key, value in result.fields.items():
        print(
            f"  {key:14s} shape={value.shape} "
            f"max|value|={float(abs(value).max()):.6g}"
        )
    if output:
        print(f"Saved: {save_result(result, output)}")
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
    return _run(args.experiment, args.output, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())

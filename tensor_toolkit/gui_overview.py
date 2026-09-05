"""Compatibility entry point and helpers for the Tensor Toolkit visualizer."""
from __future__ import annotations

from tensor_toolkit.gui import GUI_OUTPUTS, TensorToolkitGUI


def selected_output_names(flags: dict[str, bool]) -> frozenset[str]:
    """Compatibility helper used by GUI regression tests and older callers."""
    selected = frozenset(name for name in GUI_OUTPUTS if bool(flags.get(name, False)))
    if not selected:
        raise ValueError("select at least one tensor output")
    return selected


def format_memory_plan(plan: dict[str, object]) -> str:
    """Human-readable execution summary supporting old and new memory plans."""
    gib = 1024**3
    selected = float(plan.get("estimated_selected_bytes", 0)) / gib
    in_memory = float(plan.get("estimated_in_memory_bytes", 0)) / gib
    tiled = float(plan.get("estimated_tiled_bytes", 0)) / gib
    lines = [
        f"Selected mode: {plan.get('selected_mode', 'unknown')}",
        f"Estimated peak: {selected:.2f} GiB",
        f"In-memory estimate: {in_memory:.2f} GiB",
        f"Tiled estimate: {tiled:.2f} GiB",
    ]
    available = plan.get("available_bytes")
    safe = plan.get("safe_budget_bytes")
    if available is not None:
        lines.append(f"Available RAM: {float(available) / gib:.2f} GiB")
    if safe is not None:
        lines.append(f"Safe budget: {float(safe) / gib:.2f} GiB")
    if plan.get("selected_mode") == "tiled":
        block = plan.get("block_shape")
        local = plan.get("max_local_shape")
        if block:
            lines.append("Core block: " + "x".join(str(v) for v in block))
            if local:
                lines.append("Maximum halo block: " + "x".join(str(v) for v in local))
            lines.append(f"Blocks: {plan.get('block_count', 'unknown')}")
        else:
            lines.append(
                f"Tile core: {plan.get('tile_points')} t-points; halo: {plan.get('halo')}"
            )
    return "\n".join(lines)


class TensorToolkitOverviewGUI(TensorToolkitGUI):
    """Legacy 2-D GUI retained for compatibility and regression testing."""


def main() -> int:
    """Launch the VTK-based Tensor Toolkit desktop visualizer."""
    from tensor_toolkit.vtk_gui import main as vtk_main
    return vtk_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

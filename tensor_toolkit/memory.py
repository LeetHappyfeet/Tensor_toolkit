"""Memory planning for CPU tensor calculations."""
from __future__ import annotations

import os
from math import prod

COMPONENTS = {
    "metric": 16,
    "inverse_metric": 16,
    "christoffel": 64,
    "riemann": 256,
    "ricci": 16,
    "ricci_scalar": 1,
    "einstein": 16,
    "stress_energy": 16,
}

# This deliberately exceeds the sum of the obvious persistent intermediates.
# NumPy differentiation, inversion/einsum operations and tensor contractions
# create temporary arrays that briefly coexist with metric/inverse/Gamma/Ricci.
BASE_WORKING_COMPONENTS = 320
HALO = 3


def available_memory_bytes() -> int | None:
    """Best-effort estimate of currently available physical memory."""
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return None


def output_bytes(grid_shape, outputs) -> int:
    """Bytes required by the requested persistent float64 result fields."""
    points = prod(int(n) for n in grid_shape)
    return points * 8 * sum(COMPONENTS[name] for name in outputs)


def _working_components(outputs) -> int:
    return BASE_WORKING_COMPONENTS + (256 if "riemann" in set(outputs) else 0)


def local_shape_for_block(grid_shape, block_shape, halo: int = HALO) -> tuple[int, ...]:
    """Conservative maximum local shape including both halo sides."""
    return tuple(
        min(int(global_n), max(1, int(core_n)) + 2 * int(halo))
        for global_n, core_n in zip(grid_shape, block_shape)
    )


def block_working_bytes(grid_shape, outputs, block_shape, halo: int = HALO) -> int:
    local_shape = local_shape_for_block(grid_shape, block_shape, halo)
    return int(prod(local_shape) * _working_components(outputs) * 8)


def choose_block_shape(
    grid_shape,
    outputs,
    *,
    working_budget_bytes: int,
    tile_points: int = 8,
    halo: int = HALO,
) -> tuple[int, int, int, int]:
    """Choose a safe 4-D core block that fits the working-memory budget.

    ``tile_points`` remains a compatibility/user hint for the time-core width.
    Spatial cores are automatically reduced as needed. If even a 1x1x1x1 core
    plus halos cannot fit, the calculation is rejected before allocation.
    """
    shape = tuple(int(n) for n in grid_shape)
    if len(shape) != 4:
        raise ValueError("Tensor Toolkit block planning requires a 4-D grid")
    if tile_points < 1:
        raise ValueError("tile_points must be at least 1")
    if working_budget_bytes <= 0:
        raise MemoryError("no RAM remains for a tensor working block")

    core = [min(shape[0], int(tile_points)), shape[1], shape[2], shape[3]]

    # Repeatedly split the largest current core. Halos make very tiny blocks
    # inefficient, so halving gives a robust balance between memory and block count.
    while block_working_bytes(shape, outputs, core, halo) > working_budget_bytes:
        reducible = [i for i, n in enumerate(core) if n > 1]
        if not reducible:
            minimum = block_working_bytes(shape, outputs, core, halo)
            raise MemoryError(
                "even the minimum halo block exceeds the safe working-RAM budget: "
                f"need about {minimum / 1024**3:.2f} GiB for a 1x1x1x1 core"
            )
        # Prefer spatial decomposition; time is reduced only if all larger spatial
        # dimensions have already become comparable/small.
        spatial = [i for i in reducible if i != 0]
        candidates = spatial or reducible
        axis = max(candidates, key=lambda i: core[i])
        core[axis] = max(1, (core[axis] + 1) // 2)

    return tuple(core)


def estimate_peak_bytes(
    grid_shape,
    outputs,
    *,
    mode: str,
    tile_points: int = 8,
    halo: int = HALO,
    disk_backed: bool = False,
    block_shape=None,
) -> int:
    """Conservative peak-RAM estimate for the float64 CPU pipeline."""
    shape = tuple(int(n) for n in grid_shape)
    persistent = 0 if disk_backed else output_bytes(shape, outputs)
    if mode == "in_memory":
        working = prod(shape) * _working_components(outputs) * 8
    elif mode == "tiled":
        if block_shape is None:
            block_shape = (min(shape[0], max(1, int(tile_points))), *shape[1:])
        working = block_working_bytes(shape, outputs, block_shape, halo)
    else:
        raise ValueError("mode must be 'in_memory' or 'tiled'")
    return int(persistent + working)


def select_storage_mode(
    grid_shape,
    outputs,
    *,
    requested_mode: str = "auto",
    output_path=None,
    limit_fraction: float = 0.65,
) -> str:
    """Choose memory or disk persistence before execution begins."""
    if requested_mode not in {"auto", "memory", "disk"}:
        raise ValueError("storage mode must be 'auto', 'memory', or 'disk'")
    if requested_mode == "disk" and output_path is None:
        raise ValueError("disk-backed storage requires an output directory")
    if requested_mode in {"memory", "disk"}:
        return requested_mode
    if output_path is None:
        return "memory"

    persistent = output_bytes(grid_shape, outputs)
    available = available_memory_bytes()
    if available is None:
        return "disk" if persistent >= 1024**3 else "memory"
    safe = int(available * float(limit_fraction))
    return "disk" if persistent >= max(256 * 1024**2, safe // 4) else "memory"


def memory_plan(
    grid_shape,
    outputs,
    *,
    requested_mode: str = "auto",
    tile_points: int = 8,
    limit_fraction: float = 0.65,
    disk_backed: bool = False,
) -> dict[str, object]:
    """Choose full-grid or multidimensional block execution before allocation."""
    if requested_mode not in {"auto", "in_memory", "tiled"}:
        raise ValueError("memory mode must be 'auto', 'in_memory', or 'tiled'")
    if tile_points < 1:
        raise ValueError("tile_points must be at least 1")
    if not (0.1 <= float(limit_fraction) <= 0.95):
        raise ValueError("memory_limit_fraction must be between 0.1 and 0.95")
    if disk_backed and requested_mode == "in_memory":
        raise ValueError("disk-backed output requires memory mode 'auto' or 'tiled'")

    shape = tuple(int(n) for n in grid_shape)
    available = available_memory_bytes()
    safe = int(available * float(limit_fraction)) if available is not None else 2 * 1024**3
    persistent = 0 if disk_backed else output_bytes(shape, outputs)
    working_budget = safe - persistent

    in_memory = estimate_peak_bytes(
        shape, outputs, mode="in_memory", tile_points=tile_points, disk_backed=disk_backed
    )

    if requested_mode == "in_memory":
        selected = "in_memory"
        block_shape = shape
        selected_estimate = in_memory
    else:
        should_tile = disk_backed or requested_mode == "tiled" or in_memory > safe
        if not should_tile:
            selected = "in_memory"
            block_shape = shape
            selected_estimate = in_memory
        else:
            selected = "tiled"
            block_shape = choose_block_shape(
                shape,
                outputs,
                working_budget_bytes=working_budget,
                tile_points=tile_points,
                halo=HALO,
            )
            selected_estimate = estimate_peak_bytes(
                shape,
                outputs,
                mode="tiled",
                tile_points=tile_points,
                disk_backed=disk_backed,
                block_shape=block_shape,
            )

    if selected_estimate > safe:
        extra = (
            " Use disk-backed output if persistent result arrays dominate RAM."
            if not disk_backed else ""
        )
        raise MemoryError(
            "estimated peak memory exceeds the safe RAM budget: "
            f"need about {selected_estimate / 1024**3:.2f} GiB, "
            f"safe budget is {safe / 1024**3:.2f} GiB.{extra}"
        )

    local_shape = local_shape_for_block(shape, block_shape, HALO)
    block_count = prod((shape[i] + block_shape[i] - 1) // block_shape[i] for i in range(4))
    tiled_estimate = estimate_peak_bytes(
        shape,
        outputs,
        mode="tiled",
        tile_points=tile_points,
        disk_backed=disk_backed,
        block_shape=block_shape,
    )

    return {
        "requested_mode": requested_mode,
        "selected_mode": selected,
        "tile_points": int(tile_points),
        "halo": HALO,
        "block_shape": block_shape,
        "max_local_shape": local_shape,
        "block_count": int(block_count),
        "disk_backed": bool(disk_backed),
        "persistent_output_bytes": output_bytes(shape, outputs),
        "estimated_in_memory_bytes": int(in_memory),
        "estimated_tiled_bytes": int(tiled_estimate),
        "estimated_selected_bytes": int(selected_estimate),
        "available_bytes": available,
        "safe_budget_bytes": int(safe),
        "working_budget_bytes": int(max(0, working_budget)),
    }


__all__ = [
    "available_memory_bytes", "output_bytes", "estimate_peak_bytes",
    "select_storage_mode", "memory_plan", "choose_block_shape",
    "local_shape_for_block", "block_working_bytes",
]

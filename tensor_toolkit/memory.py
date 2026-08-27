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


def estimate_peak_bytes(
    grid_shape,
    outputs,
    *,
    mode: str,
    tile_points: int = 8,
    halo: int = 3,
    disk_backed: bool = False,
) -> int:
    """Conservative peak-RAM estimate for the current float64 CPU pipeline.

    Disk-backed output excludes persistent result arrays from the resident-RAM
    estimate. The operating system may cache mapped pages, but they are evictable
    and do not require the complete result to remain resident.
    """
    shape = tuple(int(n) for n in grid_shape)
    points = prod(shape)
    requested = set(outputs)
    persistent = 0 if disk_backed else output_bytes(shape, requested)

    # Streaming curvature avoids persistent Riemann unless explicitly requested.
    # The working allowance covers coordinates, metric, inverse, metric derivatives,
    # Christoffel, Ricci/Einstein and NumPy temporaries.
    working_components = 196 + (256 if "riemann" in requested else 0)
    if mode == "in_memory":
        working_points = points
    elif mode == "tiled":
        slab = min(shape[0], max(1, int(tile_points)) + 2 * int(halo))
        working_points = slab * prod(shape[1:])
    else:
        raise ValueError("mode must be 'in_memory' or 'tiled'")
    return int(persistent + working_points * working_components * 8)


def select_storage_mode(
    grid_shape,
    outputs,
    *,
    requested_mode: str = "auto",
    output_path=None,
    limit_fraction: float = 0.65,
) -> str:
    """Choose memory or disk persistence before execution begins.

    ``auto`` keeps small results in RAM, but when an output directory is supplied
    it switches to disk if persistent arrays alone would consume a substantial
    part of the safe RAM budget (or exceed 1 GiB when RAM cannot be measured).
    """
    if requested_mode not in {"auto", "memory", "disk"}:
        raise ValueError("storage mode must be 'auto', 'memory', or 'disk'")
    if requested_mode == "disk" and output_path is None:
        raise ValueError("disk-backed storage requires --output")
    if requested_mode == "memory":
        return "memory"
    if requested_mode == "disk":
        return "disk"
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
    """Choose an execution mode before large tensor arrays are allocated."""
    if requested_mode not in {"auto", "in_memory", "tiled"}:
        raise ValueError("memory mode must be 'auto', 'in_memory', or 'tiled'")
    if tile_points < 1:
        raise ValueError("tile_points must be at least 1")
    if not (0.1 <= float(limit_fraction) <= 0.95):
        raise ValueError("memory_limit_fraction must be between 0.1 and 0.95")

    in_memory = estimate_peak_bytes(
        grid_shape, outputs, mode="in_memory", tile_points=tile_points,
        disk_backed=disk_backed,
    )
    tiled = estimate_peak_bytes(
        grid_shape, outputs, mode="tiled", tile_points=tile_points,
        disk_backed=disk_backed,
    )
    available = available_memory_bytes()
    safe = int(available * float(limit_fraction)) if available is not None else None

    # Disk persistence only reduces peak memory when calculation is tiled. An
    # in-memory solve still constructs full-grid intermediates, so force tiled
    # execution for disk-backed output unless the caller explicitly requested an
    # incompatible mode.
    if disk_backed and requested_mode == "in_memory":
        raise ValueError("disk-backed output requires memory mode 'auto' or 'tiled'")

    if requested_mode == "auto":
        if disk_backed:
            selected = "tiled"
        elif safe is None:
            selected = "tiled" if in_memory > 2 * 1024**3 else "in_memory"
        else:
            selected = "in_memory" if in_memory <= safe else "tiled"
    else:
        selected = requested_mode

    selected_estimate = in_memory if selected == "in_memory" else tiled
    if safe is not None and selected_estimate > safe:
        extra = " Use disk-backed output if persistent result arrays dominate RAM." if not disk_backed else ""
        raise MemoryError(
            "estimated peak memory exceeds the safe RAM budget even in "
            f"{selected} mode: need about {selected_estimate / 1024**3:.2f} GiB, "
            f"safe budget is {safe / 1024**3:.2f} GiB. Reduce --points, request fewer "
            f"outputs, or use a smaller tile size.{extra}"
        )

    return {
        "requested_mode": requested_mode,
        "selected_mode": selected,
        "tile_points": int(tile_points),
        "halo": 3,
        "disk_backed": bool(disk_backed),
        "persistent_output_bytes": output_bytes(grid_shape, outputs),
        "estimated_in_memory_bytes": in_memory,
        "estimated_tiled_bytes": tiled,
        "estimated_selected_bytes": selected_estimate,
        "available_bytes": available,
        "safe_budget_bytes": safe,
    }


__all__ = [
    "available_memory_bytes", "output_bytes", "estimate_peak_bytes",
    "select_storage_mode", "memory_plan",
]

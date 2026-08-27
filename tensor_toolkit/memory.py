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


def _output_bytes(grid_shape, outputs) -> int:
    points = prod(int(n) for n in grid_shape)
    return points * 8 * sum(COMPONENTS[name] for name in outputs)


def estimate_peak_bytes(grid_shape, outputs, *, mode: str, tile_points: int = 8, halo: int = 3) -> int:
    """Conservative peak-RAM estimate for the current float64 CPU pipeline."""
    shape = tuple(int(n) for n in grid_shape)
    points = prod(shape)
    requested = set(outputs)
    output_bytes = _output_bytes(shape, requested)

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
    return int(output_bytes + working_points * working_components * 8)


def memory_plan(
    grid_shape,
    outputs,
    *,
    requested_mode: str = "auto",
    tile_points: int = 8,
    limit_fraction: float = 0.65,
) -> dict[str, object]:
    """Choose an execution mode before large tensor arrays are allocated."""
    if requested_mode not in {"auto", "in_memory", "tiled"}:
        raise ValueError("memory mode must be 'auto', 'in_memory', or 'tiled'")
    if tile_points < 1:
        raise ValueError("tile_points must be at least 1")
    if not (0.1 <= float(limit_fraction) <= 0.95):
        raise ValueError("memory_limit_fraction must be between 0.1 and 0.95")

    in_memory = estimate_peak_bytes(grid_shape, outputs, mode="in_memory", tile_points=tile_points)
    tiled = estimate_peak_bytes(grid_shape, outputs, mode="tiled", tile_points=tile_points)
    available = available_memory_bytes()
    safe = int(available * float(limit_fraction)) if available is not None else None

    if requested_mode == "auto":
        if safe is None:
            selected = "tiled" if in_memory > 2 * 1024**3 else "in_memory"
        else:
            selected = "in_memory" if in_memory <= safe else "tiled"
    else:
        selected = requested_mode

    selected_estimate = in_memory if selected == "in_memory" else tiled
    if safe is not None and selected_estimate > safe:
        raise MemoryError(
            "estimated peak memory exceeds the safe RAM budget even in "
            f"{selected} mode: need about {selected_estimate / 1024**3:.2f} GiB, "
            f"safe budget is {safe / 1024**3:.2f} GiB. Reduce --points, request fewer "
            "outputs, or use a smaller tile size."
        )

    return {
        "requested_mode": requested_mode,
        "selected_mode": selected,
        "tile_points": int(tile_points),
        "halo": 3,
        "estimated_in_memory_bytes": in_memory,
        "estimated_tiled_bytes": tiled,
        "estimated_selected_bytes": selected_estimate,
        "available_bytes": available,
        "safe_budget_bytes": safe,
    }


__all__ = ["available_memory_bytes", "estimate_peak_bytes", "memory_plan"]

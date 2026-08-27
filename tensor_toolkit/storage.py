"""Disk-backed result storage for large Tensor Toolkit simulations."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np


class DiskFieldStore:
    """Allocate requested tensor fields as portable NumPy ``.npy`` memmaps."""

    def __init__(self, path):
        self.path = Path(path)
        self.fields_path = self.path / "fields"
        self.path.mkdir(parents=True, exist_ok=True)
        if self.fields_path.exists():
            shutil.rmtree(self.fields_path)
        self.fields_path.mkdir(parents=True, exist_ok=True)
        self._fields: dict[str, np.memmap] = {}

    def allocate(self, name: str, shape, dtype=np.float64) -> np.memmap:
        if name in self._fields:
            return self._fields[name]
        target = self.fields_path / f"{name}.npy"
        array = np.lib.format.open_memmap(
            target,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=tuple(int(n) for n in shape),
        )
        self._fields[name] = array
        return array

    def flush(self) -> None:
        for array in self._fields.values():
            array.flush()

    @property
    def fields(self) -> dict[str, np.memmap]:
        return self._fields


def is_disk_backed_array(value) -> bool:
    return isinstance(value, np.memmap)


def disk_field_path(path, name: str) -> Path:
    return Path(path) / "fields" / f"{name}.npy"


__all__ = ["DiskFieldStore", "is_disk_backed_array", "disk_field_path"]

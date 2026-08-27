"""Persist and reload experiment results."""

from pathlib import Path
import json
import shutil

import numpy as np

from tensor_toolkit.storage import INCOMPLETE_MARKER


def _write_metadata(result, path: Path):
    meta = {
        "metric_name": result.metric_name,
        "coordinates": result.coordinates,
        **result.metadata,
        "fields": sorted(result.fields),
    }
    (path / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=list) + "\n",
        encoding="utf-8",
    )


def save_result(result, path):
    """Save either a conventional NPZ result or finalize disk-backed memmaps."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    storage = result.metadata.get("storage", {})
    disk_backed = storage.get("mode") == "disk"

    if disk_backed:
        fields_path = path / "fields"
        if not fields_path.is_dir():
            raise ValueError(
                "disk-backed result fields are not located in the requested output directory"
            )
        for value in result.fields.values():
            flush = getattr(value, "flush", None)
            if flush is not None:
                flush()
        axes_path = path / "axes"
        if axes_path.exists():
            shutil.rmtree(axes_path)
        axes_path.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(result.axis_values):
            np.save(axes_path / f"axis_{index}.npy", np.asarray(value), allow_pickle=False)
        legacy = path / "result.npz"
        if legacy.exists():
            legacy.unlink()
    else:
        arrays = {f"field__{k}": v for k, v in result.fields.items()}
        arrays.update({f"axis__{i}": v for i, v in enumerate(result.axis_values)})
        np.savez_compressed(path / "result.npz", **arrays)

    _write_metadata(result, path)
    marker = path / INCOMPLETE_MARKER
    if marker.exists():
        marker.unlink()
    return path


def _load_disk_backed(path: Path, metadata):
    fields_path = path / "fields"
    axes_path = path / "axes"
    if not fields_path.is_dir() or not axes_path.is_dir():
        raise FileNotFoundError(
            f"{path} is marked disk-backed but is missing fields/ or axes/"
        )
    names = metadata.get("fields", [])
    fields = {}
    for name in names:
        target = fields_path / f"{name}.npy"
        if not target.is_file():
            raise FileNotFoundError(f"missing disk-backed field {target}")
        fields[name] = np.load(target, mmap_mode="r", allow_pickle=False)
    axes = []
    index = 0
    while True:
        target = axes_path / f"axis_{index}.npy"
        if not target.is_file():
            break
        axes.append(np.load(target, allow_pickle=False))
        index += 1
    if not axes:
        raise FileNotFoundError(f"no coordinate axes found in {axes_path}")
    return fields, tuple(axes)


def load_result(path):
    """Load metadata, fields, and axes without forcing large fields into RAM."""
    path = Path(path)
    if (path / INCOMPLETE_MARKER).exists():
        raise ValueError(
            f"{path} contains an incomplete disk-backed calculation; rerun or choose a different result directory"
        )

    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{path} must contain metadata.json")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    storage = metadata.get("storage", {})
    if storage.get("mode") == "disk" or storage.get("format") == "npy-memmap":
        fields, axes = _load_disk_backed(path, metadata)
        return metadata, fields, axes

    result_path = path / "result.npz"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"{path} must contain result.npz or disk-backed fields"
        )
    with np.load(result_path, allow_pickle=False) as archive:
        fields = {
            key.removeprefix("field__"): archive[key].copy()
            for key in archive.files
            if key.startswith("field__")
        }
        axes = tuple(
            archive[key].copy()
            for key in sorted(
                (name for name in archive.files if name.startswith("axis__")),
                key=lambda name: int(name.split("__", 1)[1]),
            )
        )
    return metadata, fields, axes


__all__ = ["save_result", "load_result"]

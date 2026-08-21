"""Persist and reload experiment results."""

from pathlib import Path
import json
import numpy as np


def save_result(result, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {f"field__{k}": v for k, v in result.fields.items()}
    arrays.update({f"axis__{i}": v for i, v in enumerate(result.axis_values)})
    np.savez_compressed(path / "result.npz", **arrays)
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
    return path


def load_result(path):
    """Load a saved result directory into metadata, field, and axis collections."""
    path = Path(path)
    metadata_path = path / "metadata.json"
    result_path = path / "result.npz"
    if not metadata_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"{path} must contain metadata.json and result.npz"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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

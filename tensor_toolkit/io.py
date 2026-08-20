"""Persist experiment results in a simple, inspectable format."""

from pathlib import Path
import json
import numpy as np


def save_result(result, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {f"field__{name}": value for name, value in result.fields.items()}
    arrays.update({f"axis__{i}": value for i, value in enumerate(result.axis_values)})
    np.savez_compressed(path / "result.npz", **arrays)
    metadata = {
        "metric_name": result.metric_name,
        "coordinates": result.coordinates,
        **result.metadata,
        "fields": sorted(result.fields),
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=list) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = ["save_result"]

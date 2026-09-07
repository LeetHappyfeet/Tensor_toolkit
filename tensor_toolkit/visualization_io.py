"""Visualization-side loading of saved classical trajectories.

This reconstructs public Trajectory/Event containers from the stable simulation
result format. It does not run, resample, or modify the physics engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensor_toolkit.physics.events import SimulationEvent
from tensor_toolkit.physics.trajectory import Trajectory


def load_saved_trajectory(path) -> Trajectory:
    path = Path(path)
    archive_path = path / "trajectory.npz"
    metadata_path = path / "metadata.json"
    if not archive_path.is_file():
        raise FileNotFoundError(f"{path} must contain trajectory.npz")

    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"times", "positions", "velocities", "accelerations", "body_names"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"saved trajectory is missing arrays: {sorted(missing)}")
        times = archive["times"].copy()
        positions = archive["positions"].copy()
        velocities = archive["velocities"].copy()
        accelerations = archive["accelerations"].copy()
        body_names = tuple(str(v) for v in archive["body_names"].tolist())

    events = ()
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        events = tuple(
            SimulationEvent(
                time=float(item["time"]),
                kind=str(item["kind"]),
                bodies=tuple(str(v) for v in item.get("bodies", ())),
                details={str(k): float(v) for k, v in item.get("details", {}).items()},
            )
            for item in metadata.get("events", ())
        )

    return Trajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        body_names=body_names,
        events=events,
    )


__all__ = ["load_saved_trajectory"]

import json
import numpy as np

from tensor_toolkit.visualization_io import load_saved_trajectory


def test_load_saved_trajectory_reconstructs_public_result(tmp_path):
    times = np.array([0.0, 1.0])
    positions = np.zeros((2, 1, 3))
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    np.savez_compressed(
        tmp_path / "trajectory.npz",
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        body_names=np.array(["probe"]),
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({
            "events": [{
                "time": 1.0,
                "kind": "arrival",
                "bodies": ["probe"],
                "details": {"distance": 3.0},
            }]
        }),
        encoding="utf-8",
    )

    trajectory = load_saved_trajectory(tmp_path)
    assert trajectory.body_names == ("probe",)
    assert len(trajectory.events) == 1
    assert trajectory.events[0].kind == "arrival"
    assert trajectory.events[0].details["distance"] == 3.0

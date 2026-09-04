import json
import numpy as np

from tensor_toolkit.physics import (
    Body,
    SimulationExperiment,
    System,
    run_simulation_experiment,
    save_simulation_experiment_result,
)


def test_simulation_experiment_supports_multiple_bodies(tmp_path):
    system = System([
        Body("primary", 10.0, [0, 0, 0], [0, 0, 0]),
        Body("probe_a", 0.0, [10, 2, 0], [0, 1, 0]),
        Body("probe_b", 0.0, [-8, -3, 0], [0, -1, 0]),
    ])
    experiment = SimulationExperiment(
        name="many-body-test",
        system=system,
        duration=1.0,
        dt=0.1,
        method="verlet",
        sample_every=2,
        encounters=(("primary", "probe_a"), ("primary", "probe_b")),
    )
    result = run_simulation_experiment(experiment, gravitational_constant=1.0)

    assert result.trajectory.positions.shape[1:] == (3, 3)
    assert result.metadata["body_count"] == 3
    assert set(result.encounters) == {"primary->probe_a", "primary->probe_b"}
    assert set(result.test_particles) == {"primary->probe_a", "primary->probe_b"}

    output = save_simulation_experiment_result(result, tmp_path / "run")
    saved = np.load(output / "trajectory.npz")
    assert saved["positions"].shape[1:] == (3, 3)
    assert list(saved["body_names"]) == ["primary", "probe_a", "probe_b"]

    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["body_count"] == 3
    assert set(metadata["encounters"]) == {"primary->probe_a", "primary->probe_b"}
    assert set(metadata["test_particles"]) == {"primary->probe_a", "primary->probe_b"}

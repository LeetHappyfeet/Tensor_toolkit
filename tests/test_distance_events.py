import numpy as np

from tensor_toolkit.physics import (
    Body,
    DistanceCrossingDetector,
    System,
    simulate,
)


def test_distance_crossing_detector_records_entry_event():
    system = System([
        Body("primary", 0.0, [0, 0, 0], [0, 0, 0]),
        Body("probe", 0.0, [2, 0, 0], [-1, 0, 0]),
    ])
    trajectory = simulate(
        system,
        duration=1.5,
        dt=0.25,
        gravitational_constant=1.0,
        event_detectors=(
            DistanceCrossingDetector(
                "primary",
                "probe",
                radius=1.0,
                direction="enter",
                kind="sphere_of_influence_enter",
            ),
        ),
    )
    assert len(trajectory.events) == 1
    event = trajectory.events[0]
    assert event.kind == "sphere_of_influence_enter"
    assert event.bodies == ("primary", "probe")
    assert np.isclose(event.time, 1.0)
    assert np.isclose(event.details["threshold_radius"], 1.0)

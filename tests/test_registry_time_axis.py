from tensor_toolkit.registry import configure_grid, get_experiment


def test_configure_grid_independent_time_axis():
    experiment = get_experiment("alcubierre")
    configured = configure_grid(
        experiment,
        points=7,
        extent=3.0,
        time_points=21,
        time_start=-10.0,
        time_stop=30.0,
    )
    assert configured.axes[0].points == 21
    assert configured.axes[0].start == -10.0
    assert configured.axes[0].stop == 30.0
    for axis in configured.axes[1:]:
        assert axis.points == 7
        assert axis.start == -3.0
        assert axis.stop == 3.0


def test_configure_grid_legacy_uniform_behavior_is_preserved():
    experiment = get_experiment("minkowski")
    configured = configure_grid(experiment, points=9, extent=4.0)
    for axis in configured.axes:
        assert axis.points == 9
        assert axis.start == -4.0
        assert axis.stop == 4.0


def test_configure_grid_rejects_invalid_time_interval():
    experiment = get_experiment("alcubierre")
    try:
        configure_grid(
            experiment,
            time_points=5,
            time_start=2.0,
            time_stop=2.0,
        )
    except ValueError as exc:
        assert "time_stop" in str(exc)
    else:
        raise AssertionError("invalid time interval was accepted")

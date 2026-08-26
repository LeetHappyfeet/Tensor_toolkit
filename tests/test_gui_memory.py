import pytest

from tensor_toolkit.gui_overview import format_memory_plan, selected_output_names


def test_selected_output_names_filters_known_fields():
    selected = selected_output_names(
        {
            "metric": True,
            "einstein": True,
            "stress_energy": False,
            "not_a_field": True,
        }
    )
    assert selected == frozenset({"metric", "einstein"})


def test_selected_output_names_rejects_empty_selection():
    with pytest.raises(ValueError, match="select at least one"):
        selected_output_names({name: False for name in (
            "metric", "inverse_metric", "ricci", "einstein", "stress_energy"
        )})


def test_format_memory_plan_reports_tiled_execution():
    gib = 1024**3
    text = format_memory_plan(
        {
            "selected_mode": "tiled",
            "estimated_selected_bytes": 2 * gib,
            "estimated_in_memory_bytes": 6 * gib,
            "estimated_tiled_bytes": 2 * gib,
            "available_bytes": 8 * gib,
            "safe_budget_bytes": 5 * gib,
            "tile_points": 8,
            "halo": 3,
        }
    )
    assert "Selected mode: tiled" in text
    assert "Estimated peak: 2.00 GiB" in text
    assert "In-memory estimate: 6.00 GiB" in text
    assert "Available RAM: 8.00 GiB" in text
    assert "Tile core: 8 t-points; halo: 3" in text

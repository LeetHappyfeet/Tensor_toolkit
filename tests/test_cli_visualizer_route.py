import sys
import types

from tensor_toolkit import cli


def test_visualize_routes_to_vtk_gui(monkeypatch):
    module = types.SimpleNamespace(main=lambda: 17)
    monkeypatch.setitem(sys.modules, "tensor_toolkit.vtk_gui", module)
    assert cli._visualize() == 17

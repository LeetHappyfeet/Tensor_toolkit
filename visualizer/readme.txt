LEGACY VISUALIZER DIRECTORY
===========================

The supported Tensor Toolkit visualizer has moved into the installable package:

    tensor_toolkit/gui.py
    tensor_toolkit/visualization.py

Launch it with:

    tensor-toolkit visualize

or from a repository checkout:

    python visualizer.py

The files plotTensor.py, plotThreePlusOne.py, and utils/ in this directory are retained only as legacy migration/reference material. They are not imported by the supported application and are excluded from setuptools package discovery.

The new metric tensor simulator uses the same tensor_toolkit.experiment.run_experiment pipeline as the CLI. Do not add new GR calculations to this legacy directory.

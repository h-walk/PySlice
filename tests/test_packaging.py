"""Packaging regressions: sdist ships the package source."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
if TORCH_AVAILABLE:
    from pyslice.backend import TorchBackend
from pyslice.multislice import calculators as calculators_module
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.multislice.multislice import PrismProbe, Probe, Propagate, wavelength
from pyslice.multislice.potentials import Potential
from pyslice.multislice.trajectory import Trajectory
from pyslice.io.loader import Loader
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.wf_data import WFData



from _regression_helpers import *  # shared builders (see __all__)


def test_pyproject_sdist_ships_package_source():
    # The sdist must declare the package source; otherwise hatchling shipped a
    # data-only sdist and every sdist-based install produced an empty package.
    import tomllib
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "pyproject.toml").read_text())
    build = cfg["tool"]["hatch"]["build"]
    sdist_include = build["targets"]["sdist"]["include"]
    assert any("src/pyslice" in p for p in sdist_include), sdist_include
    # no lingering global data-only include and no dead setuptools placeholder
    assert "include" not in build
    assert "setuptools" not in cfg["tool"]

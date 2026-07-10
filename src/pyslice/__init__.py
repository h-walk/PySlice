"""Public package namespace for PySlice.

PySlice exposes trajectory loading, multislice propagation, potential
generation, and postprocessing helpers from this top-level package for
interactive notebook use.
"""
# here we read out the version info set in pyproject.toml
try:
    from importlib.metadata import version
    __version__ = version("pyslice")
except Exception:
    __version__ = "dev"

from .io.loader import *
from .backend import Backend, NumpyBackend, TORCH_AVAILABLE, make_backend, to_cpu, to_numpy
from .md.molecular_dynamics import *
from .multislice.calculators import *
from .multislice.multislice import *
from .multislice.potentials import *
from .multislice.sed import *
from .multislice.trajectory import *
from .postprocessing.haadf_data import *
from .postprocessing.tacaw_data import *
from .postprocessing.testtools import *
from .postprocessing.wf_data import WFData
from .optics import OpticalColumn, WavePlane, WavePropagation, simulate_raytem_wave

"""Molecular-dynamics regressions: NPT barostat units."""
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


def test_npt_barostat_params_have_physical_units():
    from ase import units
    from pyslice.md.molecular_dynamics import MDCalculator
    externalstress, pfactor = MDCalculator._npt_barostat_params(1.01325, 100.0)
    # externalstress is ~1 atm in eV/A^3, not 1.01325 eV/A^3 (~162 GPa)
    np.testing.assert_allclose(externalstress / units.bar, 1.01325, rtol=1e-6)
    assert externalstress < 1e-6                       # ~6.3e-7, not ~1
    # pfactor = ptime^2 * B (was 75*fs**2, ~1e5x too small)
    np.testing.assert_allclose(pfactor, (75 * units.fs) ** 2 * (100.0 * units.GPa))

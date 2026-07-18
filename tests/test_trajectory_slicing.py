"""Trajectory slice_positions box-translation regressions."""
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


def test_slice_positions_translates_atoms_into_new_box():
    # slice_positions shrinks the box to the range width; the kept atoms must be
    # translated by the range lower bound so they land inside [0, width), not
    # left at their original coordinates outside the new box.
    pos = np.array([[[1.0, 1.0, 1.0], [6.0, 6.0, 6.0], [7.5, 2.0, 3.0]]], dtype=float)
    traj = Trajectory(
        atom_types=np.array([14, 14, 14]), positions=pos,
        velocities=np.zeros_like(pos), box_matrix=np.diag([10.0, 10.0, 10.0]),
        timestep=0.1)
    sliced = traj.slice_positions(x_range=(5.0, 8.0))
    assert sliced.n_atoms == 2
    assert sliced.box_matrix[0, 0] == 3.0
    xs = sliced.positions[0, :, 0]
    assert np.all((xs >= 0.0) & (xs <= 3.0))          # atoms 6.0, 7.5 -> 1.0, 2.5
    np.testing.assert_allclose(sorted(xs), [1.0, 2.5])

def test_slice_positions_applies_crop_when_every_atom_survives():
    pos = np.array([[[6.0, 1.0, 1.0], [7.5, 2.0, 3.0]]], dtype=float)
    traj = Trajectory(
        atom_types=np.array([14, 14]), positions=pos,
        velocities=np.zeros_like(pos), box_matrix=np.diag([10.0, 10.0, 10.0]),
        timestep=0.1)
    sliced = traj.slice_positions(x_range=(5.0, 8.0))
    assert sliced is not traj
    assert sliced.box_matrix[0, 0] == 3.0
    np.testing.assert_allclose(sliced.positions[0, :, 0], [1.0, 2.5])

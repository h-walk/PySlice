"""Projected-potential regressions (slice-axis wrapping, flatten, guards)."""
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


def test_potential_wraps_unwrapped_slice_axis_coordinates():
    xs = np.linspace(0.0, 4.0, 5, endpoint=False)
    ys = np.linspace(0.0, 4.0, 5, endpoint=False)
    zs = np.linspace(0.0, 4.0, 5, endpoint=False)
    atom_types = np.array([14, 14])

    unwrapped = Potential(
        xs,
        ys,
        zs,
        positions=np.array([[[1.0, 1.0, -0.1], [1.0, 1.0, 4.1]]]).reshape(2, 3),
        atom_types=atom_types,
        backend=NumpyBackend(),
        kind="gauss",
    )
    wrapped = Potential(
        xs,
        ys,
        zs,
        positions=np.array([[1.0, 1.0, 3.9], [1.0, 1.0, 0.1]]),
        atom_types=atom_types,
        backend=NumpyBackend(),
        kind="gauss",
    )

    unwrapped.build()
    wrapped.build()

    np.testing.assert_allclose(unwrapped.to_numpy(), wrapped.to_numpy())

def test_potential_flatten_collapses_slice_axis():
    potential = Potential(
        np.arange(3.0),
        np.arange(4.0),
        np.arange(2.0),
        array=np.stack(
            [
                np.ones((3, 4)),
                np.ones((3, 4)) * 3,
            ],
            axis=2,
        ),
        backend=NumpyBackend(),
    )

    potential.flatten()

    assert potential.n_slices == 1
    assert potential.nz == 1
    assert potential.to_numpy().shape == (3, 4, 1)
    np.testing.assert_allclose(potential.to_numpy()[:, :, 0], np.ones((3, 4)) * 2)

def test_slice_axis_other_than_z_is_rejected(tmp_path, monkeypatch):
    # slice_axis != 2 was silently wrong (propagation is z-locked); it must now
    # fail loudly at both entry points.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError, match="slice_axis"):
        MultisliceCalculator(force_cpu=True).setup(
            _make_tiny_trajectory(), aperture=5, voltage_eV=60e3, sampling=1.0,
            slice_thickness=1.0, slice_axis=1)
    with pytest.raises(NotImplementedError, match="slice_axis"):
        Potential(np.arange(3.0), np.arange(3.0), np.arange(3.0),
                  positions=np.array([[1.0, 1.0, 1.0]]),
                  atom_types=np.array([14]), backend=NumpyBackend(),
                  kind="gauss", slice_axis=0)

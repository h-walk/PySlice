"""Backend regressions: memmap FFT inputs, device precedence."""
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


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_torch_backend_fft_accepts_numpy_memmap_inputs(tmp_path):
    backend = TorchBackend(device="cpu")
    expected_input = np.arange(8, dtype=np.complex128).reshape(2, 4)
    memmap = np.lib.format.open_memmap(
        tmp_path / "fft_input.npy",
        dtype=np.complex128,
        mode="w+",
        shape=expected_input.shape,
    )
    memmap[:] = expected_input

    fft = backend.fft(memmap, axes=1)
    fft2 = backend.fft2(memmap)
    shifted = backend.fftshift(memmap, axes=(0, 1))

    assert isinstance(fft, np.ndarray)
    np.testing.assert_allclose(fft, np.fft.fft(expected_input, axis=1))
    np.testing.assert_allclose(fft2, np.fft.fft2(expected_input))
    np.testing.assert_allclose(shifted, np.fft.fftshift(expected_input, axes=(0, 1)))

def test_pyslice_device_explicit_wins_over_env(monkeypatch):
    from pyslice.backend import TORCH_AVAILABLE
    if not TORCH_AVAILABLE:
        import pytest
        pytest.skip("torch not available")
    from pyslice.backend import TorchBackend
    monkeypatch.setenv("PYSLICE_DEVICE", "meta")            # would previously override
    be = TorchBackend(device="cpu")                          # explicit must win
    assert be.device.type == "cpu"

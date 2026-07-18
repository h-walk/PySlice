"""WFData regressions: counts sampling and real-space padding."""
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


def test_wfdata_counts_unpacks_histogram_counts(tmp_path):
    wf = _make_wf_data(tmp_path, n_time=4)

    wf.counts(11)
    assert to_numpy(wf.array).shape == (1, 4, 2, 2, 1)
    assert np.sum(to_numpy(wf.array)) == pytest.approx(11)

    wf.counts(3)
    assert np.sum(to_numpy(wf.array)) == pytest.approx(3)

@pytest.mark.parametrize("n", [32, 33, 63])
def test_pad_real_space_preserves_field_for_odd_and_even_grids(tmp_path, n):
    # self._array is an fftshifted spectrum; pad_real_space must undo/redo the
    # shift around its FFT round trip.  Without that it is exact only for even
    # grids and smears odd-grid spectra by several percent.
    pad = 5
    xs = np.arange(n, dtype=float)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")
    psi = (np.exp(-((xg - n * 0.4) ** 2 + (yg - n * 0.6) ** 2) / (0.8 * n))
           * np.exp(1j * 0.3 * xg))  # asymmetric -> sensitive to a shift error
    spectrum = np.fft.fftshift(np.fft.fft2(psi))

    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    wf = WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
        time=np.zeros(1),
        kxs=np.fft.fftshift(np.fft.fftfreq(n)), kys=np.fft.fftshift(np.fft.fftfreq(n)),
        xs=xs.copy(), ys=xs.copy(), layer=np.array([0]),
        array=spectrum[None, None, :, :, None].astype(np.complex128),
        probe=probe, backend=NumpyBackend(), cache_dir=tmp_path,
    )
    wf.pad_real_space(add_x=pad, add_y=pad)

    recovered = np.fft.ifft2(np.fft.ifftshift(to_numpy(wf._array)[0, 0, :, :, 0]))
    reference = np.zeros((n + 2 * pad, n + 2 * pad), dtype=complex)
    reference[pad:pad + n, pad:pad + n] = psi
    rel_err = np.max(np.abs(recovered - reference)) / np.max(np.abs(psi))
    assert rel_err < 1e-10

def test_counts_samples_from_intensity_not_amplitude():
    # Shot-noise counting must draw from |psi|^2 (intensity), not |psi|. Two
    # pixels with amplitudes 1 and 3 (intensities 1 and 9) must be hit ~9:1.
    from types import SimpleNamespace
    arr = np.array([1.0, 3.0], dtype=np.complex128).reshape(1, 1, 1, 2, 1)
    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    wf = WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
        time=np.zeros(1), kxs=np.arange(1.0), kys=np.arange(2.0),
        xs=np.arange(1.0), ys=np.arange(2.0), layer=np.array([0]),
        array=arr, probe=probe, backend=NumpyBackend(), cache_dir=".")
    wf.counts(2_000_000)
    hits = to_numpy(wf._array).ravel()
    ratio = hits[1] / hits[0]
    assert 8.3 < ratio < 9.7, ratio          # intensity 9:1, not amplitude 3:1

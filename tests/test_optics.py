"""Post-exit optics regressions: free-space propagation and lenses."""
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


def test_free_space_propagates_only_the_exit_wave(tmp_path):
    wf = _make_optics_wf(tmp_path)
    before = to_numpy(wf.array).copy()
    dz = 80.0

    wf.propagate_free_space(dz)

    after = to_numpy(wf.array)
    np.testing.assert_array_equal(after[..., 0], before[..., 0])
    kx_grid, ky_grid = np.meshgrid(wf.kxs, wf.kys, indexing="ij")
    propagator = np.exp(
        -1j * np.pi * wf.probe.wavelength * dz
        * (kx_grid ** 2 + ky_grid ** 2)
    )
    np.testing.assert_allclose(
        after[..., -1], before[..., -1] * propagator[None, None, :, :]
    )

def test_lens_is_grid_centred_and_modifies_only_the_exit_wave(tmp_path):
    wf = _make_optics_wf(tmp_path)
    before = to_numpy(wf.array).copy()
    f = 500.0

    wf.propagate_through_lens(f)

    after = to_numpy(wf.array)
    np.testing.assert_array_equal(after[..., 0], before[..., 0])
    real_before = np.fft.ifft2(np.fft.ifftshift(before[0, 0, :, :, -1]))
    real_after = np.fft.ifft2(np.fft.ifftshift(after[0, 0, :, :, -1]))
    x_grid, y_grid = np.meshgrid(
        wf.xs - np.mean(wf.xs), wf.ys - np.mean(wf.ys), indexing="ij"
    )
    expected_lens = np.exp(
        -1j * (2 * np.pi / wf.probe.wavelength) / (2 * f)
        * (x_grid ** 2 + y_grid ** 2)
    )
    np.testing.assert_allclose(real_after, real_before * expected_lens, atol=1e-12)

def test_lens_accepts_an_explicit_optical_axis(tmp_path):
    wf = _make_optics_wf(tmp_path)
    before = to_numpy(wf.array).copy()
    f = 500.0
    center = (1.0, 3.0)

    wf.propagate_through_lens(f, center=center)

    real_before = np.fft.ifft2(np.fft.ifftshift(before[0, 0, :, :, -1]))
    real_after = np.fft.ifft2(
        np.fft.ifftshift(to_numpy(wf.array)[0, 0, :, :, -1])
    )
    x_grid, y_grid = np.meshgrid(
        wf.xs - center[0], wf.ys - center[1], indexing="ij"
    )
    expected_lens = np.exp(
        -1j * (2 * np.pi / wf.probe.wavelength) / (2 * f)
        * (x_grid ** 2 + y_grid ** 2)
    )
    np.testing.assert_allclose(real_after, real_before * expected_lens, atol=1e-12)

def test_downstream_optics_use_each_decoherence_copy_wavelength(tmp_path):
    n = 5
    n_copies, n_positions = 2, 2
    xs = np.arange(n, dtype=float)
    kxs = np.fft.fftshift(np.fft.fftfreq(n))
    wavelengths = np.array([0.03, 0.06])
    probe = SimpleNamespace(
        eV=1e5, wavelength=wavelengths[0], wavelengths=wavelengths,
        mrad=30.0,
        _array=np.zeros((n_copies, n_positions, n, n), dtype=np.complex128),
    )
    # Flattening is copy-major: each wavelength applies to all scan positions
    # for one decoherence copy.
    array = np.ones(
        (n_copies * n_positions, 1, n, n, 2), dtype=np.complex128
    )
    wf = WFData(
        probe_positions=[(1.0, 1.0), (3.0, 3.0)],
        probe_xs=[1.0, 3.0], probe_ys=[1.0, 3.0], time=np.zeros(1),
        kxs=kxs, kys=kxs.copy(), xs=xs, ys=xs.copy(),
        layer=np.array([1.0, 2.0]), array=array, probe=probe,
        backend=NumpyBackend(), cache_dir=tmp_path,
    )
    before = wf.array.copy()
    dz = 80.0

    wf.propagate_free_space(dz)

    kx_grid, ky_grid = np.meshgrid(kxs, kxs, indexing="ij")
    k_sq = kx_grid ** 2 + ky_grid ** 2
    for copy, wavelength_value in enumerate(wavelengths):
        expected = np.exp(-1j * np.pi * wavelength_value * dz * k_sq)
        start = copy * n_positions
        stop = start + n_positions
        np.testing.assert_allclose(
            wf.array[start:stop, 0, :, :, -1],
            np.broadcast_to(expected, (n_positions, n, n)),
        )
    np.testing.assert_array_equal(wf.array[..., 0], before[..., 0])

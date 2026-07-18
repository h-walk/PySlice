"""Spectral Energy Density (phonon dispersion) correctness tests.

SED(avg, displacements, kvec) computes Phi(k, omega) = |FFT_t{ sum_n u(n,t)
exp(i k x_n) }|^2. A single plane-wave phonon must show up as one peak at its
frequency and its wavevector; that pins the FFT axis, the k-projection, and the
positive-frequency slicing all at once.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyslice.backend import NumpyBackend
from pyslice.multislice.sed import SED


def _chain_planewave(na=16, nt=64, m_wavelengths=3, freq_bin=6, amp=1.0):
    """A 1-D atom chain carrying one travelling wave u(n,t)=cos(w0 t - k0 n).

    Returns (avg, displacements, kvec, k_index, freq_bin): kvec is an
    (na, 1, 3) grid of k_x = 2*pi*i/na, so the matching column is i=m_wavelengths.
    """
    n = np.arange(na)
    t = np.arange(nt)
    k0 = 2 * np.pi * m_wavelengths / na
    w0 = 2 * np.pi * freq_bin / nt
    avg = np.zeros((na, 3))
    avg[:, 0] = n                                   # positions x_n = n
    u = amp * np.cos(w0 * t[:, None] - k0 * n[None, :])   # (nt, na)
    displacements = np.zeros((nt, na, 3))
    displacements[:, :, 0] = u                      # polarised along x
    kx = 2 * np.pi * np.arange(na) / na
    kvec = np.zeros((na, 1, 3))
    kvec[:, 0, 0] = kx
    return avg, displacements, kvec, m_wavelengths, freq_bin


def test_sed_output_shapes():
    avg, disp, kvec, _, _ = _chain_planewave(na=8, nt=32)
    Zs, ws = SED(avg, disp, kvec, NumpyBackend(), v_xyz=0)
    assert Zs.shape == (32 // 2, 8, 1)              # (n_freq, nx, ny)
    assert ws.shape == (32 // 2,)
    np.testing.assert_allclose(ws, np.fft.fftfreq(32)[:16])


def test_sed_peaks_at_phonon_frequency_and_wavevector():
    avg, disp, kvec, k_idx, f_bin = _chain_planewave(
        na=16, nt=64, m_wavelengths=3, freq_bin=6)
    Zs, ws = SED(avg, disp, kvec, NumpyBackend(), v_xyz=0)   # (n_freq, nx, ny)

    # Global maximum sits exactly on (frequency bin, k column) of the wave.
    peak = np.unravel_index(np.argmax(Zs[:, :, 0]), Zs[:, :, 0].shape)
    assert peak == (f_bin, k_idx)
    # ...and the peak frequency reads back as freq_bin/nt on the ws axis.
    assert ws[peak[0]] == pytest.approx(6 / 64, abs=1e-9)


def test_sed_dc_displacement_has_no_spectral_power_off_zero():
    # A static (time-independent) displacement pattern lands entirely in the
    # ws=0 bin; higher-frequency bins carry ~no power.
    na, nt = 12, 32
    avg = np.zeros((na, 3)); avg[:, 0] = np.arange(na)
    disp = np.zeros((nt, na, 3)); disp[:, :, 0] = 1.0    # constant in time
    kvec = np.zeros((na, 1, 3)); kvec[:, 0, 0] = 2 * np.pi * np.arange(na) / na
    Zs, ws = SED(avg, disp, kvec, NumpyBackend(), v_xyz=0)
    assert Zs[1:, :, :].max() < 1e-9 * max(Zs[0].max(), 1e-30)


def test_sed_axis_selection_picks_the_requested_polarisation():
    # Displacement only along x: projecting on x (v_xyz=0) has power; projecting
    # on y (v_xyz=1) sees nothing.
    avg, disp, kvec, _, _ = _chain_planewave(na=16, nt=64)
    Zx, _ = SED(avg, disp, kvec, NumpyBackend(), v_xyz=0)
    Zy, _ = SED(avg, disp, kvec, NumpyBackend(), v_xyz=1)
    assert Zx.max() > 0
    assert Zy.max() < 1e-9 * Zx.max()

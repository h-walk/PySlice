"""Analytic-reference and invariant tests for PySlice's core algorithms.

Unlike the regression suite (which pins specific fixed bugs), these verify that
the *physics* is right: known constants, closed-form values, and mathematical
invariants the algorithms must satisfy (unitarity, Parseval, DC integrals,
frequency calibration). A golden-file snapshot passes even when the physics is
wrong-but-stable; an invariant does not.

Everything is self-contained (tiny in-memory objects, NumPy default; the
``backend`` fixture also exercises Torch when installed).
"""
from __future__ import annotations

import numpy as np
import pytest

from pyslice.multislice.multislice import (
    wavelength, m_effective, antialias_aperture,
    m_electron, c_light, q_electron, h_planck,
)
from pyslice.multislice.potentials import (
    Potential, load_kirkland, _FE_TO_V, get_z_from_element,
)
from pyslice.backend import NumpyBackend, to_numpy
from pyslice.postprocessing.tacaw_data import TACAWData


# ===========================================================================
# Relativistic electron optics
# ===========================================================================

@pytest.mark.parametrize("eV, lambda_A", [
    (100e3, 0.0370143),   # Kirkland App. B / de Broglie, relativistic
    (200e3, 0.0250793),
    (300e3, 0.0196875),
])
def test_relativistic_wavelength_matches_literature(eV, lambda_A):
    """lambda(E) equals the tabulated relativistic de Broglie wavelength."""
    assert wavelength(eV) == pytest.approx(lambda_A, rel=1e-4)


def test_wavelength_is_monotonic_in_voltage():
    """Higher voltage -> shorter wavelength."""
    ev = np.array([50e3, 100e3, 200e3, 300e3])
    lam = np.array([wavelength(e) for e in ev])
    assert np.all(np.diff(lam) < 0)


def test_effective_mass_is_relativistic():
    """m_eff = m_e + eV*q/c^2 exceeds the rest mass and grows with voltage."""
    assert m_effective(0.0) == pytest.approx(m_electron, rel=1e-12)
    assert m_effective(300e3) > m_effective(100e3) > m_electron
    # At 300 kV the kinetic energy is ~0.587 * rest energy.
    rest_energy_eV = m_electron * c_light**2 / q_electron
    assert m_effective(300e3) == pytest.approx(
        m_electron * (1 + 300e3 / rest_energy_eV), rel=1e-9)


def test_interaction_parameter_matches_kirkland_table():
    """sigma = (2pi/(lambda*eV)) * (E0+eV)/(2E0+eV) reproduces Kirkland's value.

    Kirkland Table (interaction parameter) gives sigma(100 kV) = 0.9244e-3
    per (V*Angstrom). This is the exact expression multislice.py builds inline.
    """
    eV = 100e3
    lam = wavelength(eV)
    E0 = m_electron * c_light**2 / q_electron
    sigma = (2 * np.pi) / (lam * eV) * (E0 + eV) / (2 * E0 + eV)
    assert sigma == pytest.approx(0.9244e-3, rel=2e-3)


# ===========================================================================
# Kirkland projected potential
# ===========================================================================

def test_kirkland_prefactor_equals_2pi_hbar2_over_me_e():
    """_FE_TO_V is 2*pi*hbar^2/(m_e*e), expressed in V*Angstrom^2."""
    hbar = h_planck / (2 * np.pi)
    prefactor_SI = 2 * np.pi * hbar**2 / (m_electron * q_electron)  # V*m^2
    assert _FE_TO_V == pytest.approx(prefactor_SI * 1e20, rel=1e-4)


def test_projected_potential_dc_integral_invariant(make_potential):
    """Exact DFT identity: integral of V_proj over the plane = f_e(0)*prefactor.

    The reciprocal potential of one atom has DC term S(0)*f_e(0) = f_e(0); the
    IFFT's spatial sum equals that DC term, so sum(V)*dx*dy = f_e(0)*_FE_TO_V.
    This ties the whole potential pipeline (form factor, IFFT, prefactor,
    grid spacing) to a single closed-form number.
    """
    be = NumpyBackend()
    box = 8.0
    pot = make_potential([[box / 2, box / 2, 0.0]], ["C"],
                         box=box, slice_thickness=2 * box, backend=be)
    V = to_numpy(pot.array[:, :, 0])

    Z = get_z_from_element("C")
    params = to_numpy(load_kirkland(be))[Z - 1]      # (3, 4): a,b,c,d
    a, b, c, d = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    f0 = float(np.sum(a / b) + np.sum(c))            # form factor at q=0

    integral = V.sum() * pot.dx * pot.dy
    assert integral == pytest.approx(f0 * _FE_TO_V, rel=1e-6)


def test_projected_potential_positive_and_peaks_at_atom(make_potential):
    """A single atom's projected potential is positive and peaks at the atom."""
    box = 8.0
    pot = make_potential([[box / 2, box / 2, 0.0]], ["Si"],
                         box=box, slice_thickness=2 * box)
    V = to_numpy(pot.array[:, :, 0])
    assert V.sum() > 0
    peak = np.unravel_index(np.argmax(V), V.shape)
    centre = (int(round(box / 2 / pot.dx)), int(round(box / 2 / pot.dy)))
    assert abs(peak[0] - centre[0]) <= 1 and abs(peak[1] - centre[1]) <= 1


def test_heavier_atom_scatters_more(make_potential):
    """Total projected potential grows with Z (more electrons to screen)."""
    box = 8.0
    totals = []
    for el in ["C", "Si", "Au"]:
        pot = make_potential([[box / 2, box / 2, 0.0]], [el],
                             box=box, slice_thickness=2 * box)
        totals.append(float(to_numpy(pot.array).sum()))
    assert totals[0] < totals[1] < totals[2]


def test_grid_conventions(make_potential):
    """nx = int(lx/sampling)+1 and the grid excludes the right endpoint."""
    box, sampling = 5.0, 0.1
    pot = make_potential([[1.0, 1.0, 0.0]], ["C"],
                         box=box, sampling=sampling, slice_thickness=2 * box)
    assert pot.nx == int(box / sampling) + 1
    assert pot.dx == pytest.approx(box / pot.nx, rel=1e-9)   # endpoint=False


# ===========================================================================
# Fresnel propagator & anti-aliasing aperture (multislice engine core)
# ===========================================================================

def _fresnel_propagator(lam, dz, kx, ky):
    kx_g, ky_g = np.meshgrid(kx, ky, indexing="ij")
    return np.exp(-1j * np.pi * lam * dz * (kx_g**2 + ky_g**2))


def test_fresnel_propagator_is_unit_modulus():
    """|P| = 1 everywhere: free-space propagation is unitary (no gain/loss)."""
    kx = np.fft.fftfreq(16, d=0.1)
    P = _fresnel_propagator(wavelength(100e3), 0.5, kx, kx)
    assert np.allclose(np.abs(P), 1.0, atol=1e-12)


def test_free_space_propagation_conserves_norm():
    """psi' = IFFT[P * FFT[psi]] preserves the L2 norm when |P| = 1 (Parseval)."""
    rng = np.random.default_rng(0)
    n = 16
    psi = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    kx = np.fft.fftfreq(n, d=0.1)
    P = _fresnel_propagator(wavelength(100e3), 0.7, kx, kx)
    out = np.fft.ifft2(P * np.fft.fft2(psi))
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(psi), rel=1e-10)


def test_antialias_aperture_cuts_outer_third():
    """Aperture is 1 at k=0, 0 beyond the 2/3-Nyquist cutoff, and in [0, 1]."""
    be = NumpyBackend()
    kx = be.asarray(np.fft.fftfreq(32, d=0.1))
    ap = to_numpy(antialias_aperture(kx, kx, be))
    assert ap.min() >= 0.0 and ap.max() <= 1.0 + 1e-12
    assert ap[0, 0] == pytest.approx(1.0)          # DC passes

    k_max = float(np.abs(to_numpy(kx)).max())
    kx_g, ky_g = np.meshgrid(to_numpy(kx), to_numpy(kx), indexing="ij")
    k_r = np.sqrt(kx_g**2 + ky_g**2)
    assert np.all(ap[k_r >= (2 / 3) * k_max] == 0.0)     # outer third killed
    assert np.all(ap[k_r < (2 / 3) * k_max - 0.02 * k_max] == 1.0)  # inner passes


def test_transmission_function_is_pure_phase():
    """t = exp(i*sigma*V) has unit modulus and advances phase for V > 0."""
    sigma = 1e-3
    V = np.linspace(0, 50, 11)          # small enough that sigma*V < pi
    t = np.exp(1j * sigma * V)
    assert np.allclose(np.abs(t), 1.0)
    phase = np.angle(t)
    assert np.all(np.diff(phase) > 0)   # positive potential -> increasing phase


# ===========================================================================
# Backend FFT conventions (a tacit assumption the whole engine relies on)
# ===========================================================================

def test_fft_ifft_roundtrip(backend):
    """ifft2(fft2(x)) == x for the active backend (norm convention sanity)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    xb = backend.asarray(x, dtype=backend.complex_dtype)
    out = to_numpy(backend.ifft2(backend.fft2(xb)))
    assert np.allclose(out, x, atol=1e-10)


def test_parseval_across_fft(backend):
    """sum|x|^2 == (1/N) sum|X|^2 for this backend's unnormalised forward FFT."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    xb = backend.asarray(x, dtype=backend.complex_dtype)
    X = to_numpy(backend.fft2(xb))
    lhs = np.sum(np.abs(x) ** 2)
    rhs = np.sum(np.abs(X) ** 2) / x.size
    assert lhs == pytest.approx(rhs, rel=1e-10)


# ===========================================================================
# TACAW spectral calibration
# ===========================================================================

def test_tacaw_frequency_axis_calibration(make_wf_data):
    """A pure tone at f0 THz produces a spectral peak at f0.

    Verifies the time->frequency calibration end to end: dt is taken from the
    time array (ps) and frequencies come out in THz with no stray 2*pi or
    save-interval factor. A miscalibration here silently mislabels every phonon.
    """
    n, dt_ps = 32, 0.02                       # Nyquist 25 THz, df = 1.5625 THz
    df = 1.0 / (n * dt_ps)
    f0 = 5 * df                                # land exactly on a bin
    t = np.arange(n) * dt_ps
    tone = np.exp(2j * np.pi * f0 * t)         # complex tone -> peak at +f0

    wf = make_wf_data(n_time=n, time=t, time_series=tone)
    tac = TACAWData(wf, force_rerun=True)

    freqs = tac.frequencies
    spec = to_numpy(tac._array)[0, :, 0, 0]    # (freq,) at one k-pixel
    assert freqs[np.argmax(spec)] == pytest.approx(f0, abs=0.5 * df)


def test_tacaw_detrend_removes_elastic_line(make_wf_data):
    """A constant (DC/elastic) time series is removed by the per-segment detrend.

    The estimator subtracts each segment's mean before the FFT, so a flat series
    carries no spectral power -- the intended suppression of the elastic peak.
    """
    wf = make_wf_data(n_time=16)               # default all-ones (DC) series
    tac = TACAWData(wf, force_rerun=True)
    spec = to_numpy(tac._array)[0, :, 0, 0]
    assert np.max(np.abs(spec)) == pytest.approx(0.0, abs=1e-12)


def test_tacaw_boxcar_matches_manual_periodogram(make_wf_data):
    """Default (boxcar, single segment) equals the detrended |FFT|^2 by hand.

    Boxcar is RMS-normalised to unity, so the only processing is the mean
    removal; matching that exactly pins the window normalisation and FFT path.
    """
    rng = np.random.default_rng(3)
    n = 16
    series = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    wf = make_wf_data(n_time=n, time_series=series)
    tac = TACAWData(wf, force_rerun=True)

    detrended = series - series.mean()          # matches the estimator's detrend
    manual = np.abs(np.fft.fftshift(np.fft.fft(detrended))) ** 2
    spec = to_numpy(tac._array)[0, :, 0, 0]
    assert np.allclose(spec, manual, rtol=1e-8, atol=1e-8)

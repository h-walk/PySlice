"""Shared builders for the split regression suites (moved verbatim from the
former test_review_regressions.py). Imported via `from _regression_helpers
import *`; every builder is listed in __all__ so underscore names come through.
"""
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



__all__ = [
    '_make_tiny_trajectory',
    '_peak_pixel',
    '_make_deferred_probe',
    '_adf_image_for',
    '_single_slice_potential',
    '_propagate',
    '_make_optics_wf',
    '_loader',
    '_make_wf_data',
    '_make_layered_wf',
    '_adf_run',
    '_tone_wf',
    '_multiprobe_wf',
    '_UNIT_XS', '_UNIT_YS', '_OFFCENTRE', '_ONCE_PIXEL',
]

# Module-level fixtures for the probe-shift/cropping regressions.
_UNIT_XS = np.linspace(0.0, 8.0, 32, endpoint=False)  # dx = 0.25, centre = 4.0
_UNIT_YS = np.linspace(0.0, 8.0, 32, endpoint=False)
_OFFCENTRE = (2.0, 2.0)          # peak pixel 2.0/0.25 = 8 when shifted once
_ONCE_PIXEL = (8, 8)             # (unshifted -> 16, double-shifted -> 0)


def _make_tiny_trajectory():
    positions = np.array([[[1.5, 1.5, 1.5]]], dtype=float)
    return Trajectory(
        atom_types=np.array([14]),
        positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([4.0, 4.0, 3.0]),
        timestep=0.1,
    )


# ---------------------------------------------------------------------------
# Regression: single off-centre probe must not be re-shifted every frame.
#
# Probe.applyShifts positions the probe with a k-space phase ramp.  It used to
# guard against re-application only via "npt > 1", which never triggers for a
# single probe position, so setup()'s shift plus run()'s per-frame shift (plus
# the trailing shift inside addTemporalDecoherence) compounded and the probe
# drifted across frames.  For TACAW this dwarfs the phonon signal.
# ---------------------------------------------------------------------------

# Grid + aperture chosen so the probe is well localised (aperture spans many
# k-pixels); on a coarse grid the probe degenerates to a plane wave whose |ψ|²
# is shift-invariant, which would hide the drift entirely.
_UNIT_XS = np.linspace(0.0, 8.0, 32, endpoint=False)  # dx = 0.25, centre = 4.0
_UNIT_YS = np.linspace(0.0, 8.0, 32, endpoint=False)
_OFFCENTRE = (2.0, 2.0)          # peak pixel 2.0/0.25 = 8 when shifted once
_ONCE_PIXEL = (8, 8)             # (unshifted -> 16, double-shifted -> 0)

def _peak_pixel(array2d):
    intensity = np.abs(to_numpy(array2d)) ** 2
    return np.unravel_index(int(np.argmax(intensity)), intensity.shape)

def _make_deferred_probe(position):
    return Probe(
        _UNIT_XS, _UNIT_YS, mrad=30.0, eV=60e3, backend=NumpyBackend(),
        probe_positions=np.asarray([position], dtype=float),
        defer_shifts=True,
    )

def _adf_image_for(probe_kwargs, tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    static = np.tile(
        np.array([[[3.1, 4.7, 1.5], [5.5, 2.2, 1.5]]], dtype=float), (2, 1, 1))
    traj = Trajectory(
        atom_types=np.array([14, 14]),
        positions=static,
        velocities=np.zeros_like(static),
        box_matrix=np.diag([8.0, 8.0, 3.0]),
        timestep=0.1,
    )
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(traj, aperture=30, voltage_eV=60e3, sampling=0.25,
               slice_thickness=1.0, ADF=(45, 150), cache_wavefunctions=False,
               **probe_kwargs)
    wf, _ = calc.run(force_rerun=True)
    return to_numpy(HAADFData(wf).calculateADF())

def _single_slice_potential(xs, ys):
    return Potential(
        xs, ys, np.array([1.5]),
        positions=np.array([[4.0, 4.0, 1.5]]),  # one atom at cell centre
        atom_types=np.array([14]),
        backend=NumpyBackend(),
        kind="kirkland",
    )

def _propagate(position, cropping, decohere=False):
    xs = np.linspace(0.0, 8.0, 32, endpoint=False)
    probe = Probe(xs, xs, mrad=30.0, eV=100e3, backend=NumpyBackend(),
                  probe_positions=np.asarray([position], dtype=float),
                  cropping=cropping, defer_shifts=True)
    if decohere:
        probe.addTemporalDecoherence(2.0, 3)
    else:
        probe.applyShifts()
    exit_wave = to_numpy(Propagate(probe, _single_slice_potential(xs, xs),
                                   NumpyBackend(), onthefly=True))
    return exit_wave, probe

def _make_optics_wf(tmp_path, n=5):
    xs = np.arange(n, dtype=float)
    kxs = np.fft.fftshift(np.fft.fftfreq(n))
    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")
    entrance = (1 + x_grid + 2 * y_grid).astype(np.complex128)
    exit_wave = np.exp(-((x_grid - 1.3) ** 2 + (y_grid - 3.1) ** 2) / 2)
    waves = np.stack(
        [np.fft.fftshift(np.fft.fft2(entrance)),
         np.fft.fftshift(np.fft.fft2(exit_wave))],
        axis=-1,
    )[None, None, :, :, :]
    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(
            np.zeros((1, 1, n, n), dtype=np.complex128)))
    return WFData(
        probe_positions=[(2.0, 2.0)], probe_xs=[2.0], probe_ys=[2.0],
        time=np.zeros(1), kxs=kxs, kys=kxs.copy(),
        xs=xs, ys=xs.copy(), layer=np.array([1.0, 2.0]),
        array=waves, probe=probe, backend=NumpyBackend(), cache_dir=tmp_path,
    )

def _loader(tmp_path, **kwargs):
    dump = tmp_path / "dummy.lammpstrj"
    dump.write_text("")  # only needs to exist; we call the resolver directly
    return Loader(filename=str(dump), **kwargs)

def _make_wf_data(tmp_path, n_time, backend=None):
    backend = NumpyBackend() if backend is None else backend
    probe = SimpleNamespace(
        eV=60e3,
        wavelength=0.05,
        mrad=5.0,
        _array=backend.asarray(
            np.zeros((1, 1, 2, 2), dtype=np.complex128),
            dtype=backend.complex_dtype,
        ),
    )
    return WFData(
        probe_positions=[(0.0, 0.0)],
        probe_xs=[0.0],
        probe_ys=[0.0],
        time=np.arange(n_time, dtype=float),
        kxs=np.arange(2, dtype=float),
        kys=np.arange(2, dtype=float),
        xs=np.arange(2, dtype=float),
        ys=np.arange(2, dtype=float),
        layer=np.array([0]),
        array=backend.asarray(
            np.ones((1, n_time, 2, 2, 1), dtype=np.complex128),
            dtype=backend.complex_dtype,
        ),
        probe=probe,
        backend=backend,
        cache_dir=tmp_path,
    )

def _make_layered_wf(cache_dir, n_layers=2, seed=0):
    nt = 16
    t = np.arange(nt) * 0.1
    arr = np.zeros((1, nt, 2, 2, n_layers), dtype=np.complex128)
    for layer in range(n_layers):
        freq = (layer + 1) * 1.0 + seed * 0.3  # distinct per layer/dataset
        arr[0, :, :, :, layer] = np.cos(2 * np.pi * freq * t)[:, None, None]
    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    return WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0], time=t,
        kxs=np.arange(2.0), kys=np.arange(2.0), xs=np.arange(2.0), ys=np.arange(2.0),
        layer=np.arange(n_layers), array=arr, probe=probe,
        backend=NumpyBackend(), cache_dir=cache_dir)

def _adf_run(tmp_path, monkeypatch, **setup_kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    s = np.array([[[4., 4., 1.0], [3., 5., 2.5]]], dtype=np.float32)
    traj = Trajectory(atom_types=np.array([14, 14]), positions=s,
                      velocities=np.zeros_like(s), box_matrix=np.diag([8., 8., 4.]),
                      timestep=0.1)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25, slice_thickness=1.0,
               probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.], ADF=(45, 150),
               cache_wavefunctions=False, **setup_kw)
    return calc.run(force_rerun=True)

def _tone_wf(tmp_path, signal):
    # WFData whose exit-wave time series is `signal` broadcast over a 2x2 k-grid.
    import pathlib
    n = len(signal)
    arr = np.zeros((1, n, 2, 2, 1), dtype=np.complex128)
    arr[0, :, :, :, 0] = np.asarray(signal)[:, None, None]
    probe = SimpleNamespace(eV=1e5, wavelength=0.037, mrad=30.0,
                            _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    return WFData(probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
                  time=np.arange(n) * 0.1, kxs=np.arange(2.0), kys=np.arange(2.0),
                  xs=np.arange(2.0), ys=np.arange(2.0), layer=np.array([0]),
                  array=arr, probe=probe, backend=NumpyBackend(),
                  cache_dir=pathlib.Path(tmp_path))

def _multiprobe_wf(tmp_path, signals):
    # signals: (n_probes, n_time) complex -> a WFData with per-probe time series.
    import pathlib
    signals = np.asarray(signals)
    nprobe, n = signals.shape
    arr = np.zeros((nprobe, n, 2, 2, 1), dtype=np.complex128)
    for p in range(nprobe):
        arr[p, :, :, :, 0] = signals[p][:, None, None]
    probe = SimpleNamespace(eV=1e5, wavelength=0.037, mrad=30.0,
                            _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    return WFData(probe_positions=[(float(p), 0.0) for p in range(nprobe)],
                  probe_xs=list(range(nprobe)), probe_ys=[0], time=np.arange(n) * 0.1,
                  kxs=np.arange(2.0), kys=np.arange(2.0), xs=np.arange(2.0), ys=np.arange(2.0),
                  layer=np.array([0]), array=arr, probe=probe, backend=NumpyBackend(),
                  cache_dir=pathlib.Path(tmp_path))

"""Shared pytest fixtures for the PySlice suite.

Everything here is self-contained: fixtures build tiny in-memory objects (a few
atoms, a 4x4 grid, a handful of frames) so the whole suite runs on one CPU core
in seconds and needs no data files, network, or GPU.

Optional backends/engines are soft: the ``backend`` fixture parametrizes over
NumPy plus Torch *only if it is installed*, and torch-only cases skip rather than
error when it is absent.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from pyslice.backend import TorchBackend


# ---------------------------------------------------------------------------
# Backend selection for the suite
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _pin_numpy_backend(monkeypatch):
    """Pin ``make_backend()`` to NumPy for the suite unless overridden.

    ``MultisliceCalculator`` (and other engine code) call ``make_backend()``,
    which selects the torch-CPU backend whenever torch is installed. Torch's
    per-FFT-call overhead makes the small-grid calculator/HAADF *logic* tests
    ~50x slower than NumPy for no extra coverage — the torch FFT path itself is
    exercised by the parity tests, which build ``TorchBackend`` directly and are
    unaffected by this env var. Run e.g. ``PYSLICE_BACKEND=torch pytest`` to
    exercise the calculators on torch instead.
    """
    if "PYSLICE_BACKEND" not in os.environ:
        monkeypatch.setenv("PYSLICE_BACKEND", "numpy")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _available_backends():
    backends = [pytest.param(NumpyBackend, id="numpy")]
    if TORCH_AVAILABLE:
        backends.append(pytest.param(TorchBackend, id="torch", marks=pytest.mark.torch))
    return backends


@pytest.fixture
def numpy_backend():
    """A plain NumPy backend — the always-available default."""
    return NumpyBackend()


@pytest.fixture(params=_available_backends())
def backend(request):
    """A backend instance, parametrized over every backend that is installed.

    Use this for numeric tests that both backends must agree on; the torch
    parametrization is marked ``torch`` and simply absent when torch is not
    installed.
    """
    return request.param()


# ---------------------------------------------------------------------------
# Factories (fixtures that return a builder, so tests choose the parameters)
# ---------------------------------------------------------------------------

@pytest.fixture
def make_wf_data(tmp_path):
    """Factory building a minimal :class:`WFData` for TACAW/postprocessing tests.

    ``time_series`` (optional) is an array broadcast over the (kx, ky) plane so a
    caller can inject a known temporal signal; default is all-ones (DC).
    """
    from pyslice.postprocessing.wf_data import WFData

    def _build(n_time=8, nk=2, backend=None, time=None, time_series=None,
               cache_dir=None):
        be = NumpyBackend() if backend is None else backend
        if time is None:
            time = np.arange(n_time, dtype=float)
        if time_series is None:
            array = np.ones((1, n_time, nk, nk, 1), dtype=np.complex128)
        else:
            ts = np.asarray(time_series, dtype=np.complex128)
            array = np.broadcast_to(
                ts[None, :, None, None, None], (1, n_time, nk, nk, 1)
            ).copy()
        probe = SimpleNamespace(
            eV=60e3, wavelength=0.05, mrad=5.0,
            _array=be.asarray(np.zeros((1, 1, nk, nk), dtype=np.complex128),
                              dtype=be.complex_dtype),
        )
        return WFData(
            probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
            time=time,
            kxs=np.arange(nk, dtype=float), kys=np.arange(nk, dtype=float),
            xs=np.arange(nk, dtype=float), ys=np.arange(nk, dtype=float),
            layer=np.array([0]),
            array=be.asarray(array, dtype=be.complex_dtype),
            probe=probe, backend=be,
            cache_dir=cache_dir if cache_dir is not None else tmp_path,
        )

    return _build


@pytest.fixture
def make_potential():
    """Factory building a :class:`Potential` for a handful of atoms on a grid.

    Returns a *built* Potential (``.array`` populated) on the given backend.
    """
    from pyslice.multislice.potentials import Potential

    def _build(positions, atom_types, box=8.0, sampling=0.1,
               slice_thickness=None, backend=None):
        be = NumpyBackend() if backend is None else backend
        st = slice_thickness if slice_thickness is not None else box
        nx = int(box / sampling) + 1
        nz = int(box / st) + 1
        xs = np.linspace(0, box, nx, endpoint=False)
        ys = np.linspace(0, box, nx, endpoint=False)
        zs = np.linspace(0, box, nz, endpoint=False)
        pot = Potential(xs, ys, zs, positions=np.asarray(positions, dtype=float),
                        atom_types=list(atom_types), backend=be)
        pot.build()
        return pot

    return _build

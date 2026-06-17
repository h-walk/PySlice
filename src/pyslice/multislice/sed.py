"""
Spectral Energy Density: phonon dispersions.
Borrowed from pySED: https://github.com/tpchuckles/pySED
"""
from __future__ import annotations

import numpy as np
from typing import Union

from pyslice.backend import Backend, to_numpy


def SED(avg: np.ndarray,
        displacements: np.ndarray,
        kvec,
        backend: Backend,
        v_xyz: Union[int, float, np.ndarray] = 0,
        bs: np.ndarray = None):
    """
    Compute Spectral Energy Density Φ(k, ω).

    THE MATH:
        Φ(k,ω) = Σb | ∫ Σn u°(n,b,t) exp(i k r̄(n,0)) exp(-iωt) dt |²
              = | FFT{ Σn u°(n,t) exp(i k x̄(n)) } |²

    Args:
        avg:           Mean atom positions, shape (n_atoms, 3).
        displacements: Time-dependent displacements, shape (n_time, n_atoms, 3).
        kvec:          k-point grid of shape (nx, ny, 3).
        backend:       Active Backend instance.
        v_xyz:         Displacement direction — 0/1/2 for x/y/z, or a 3-vector.
        bs:            Optional atom index subset for BZ-folded calculation.

    Returns:
        Tuple of (Zs, ws):
            Zs: Spectral power |FFT{…}|², shape (n_freq, nx, ny).
            ws: Frequency array (positive half), length n_freq.
    """
    nt, na, _ = displacements.shape

    if bs is None:
        bs = np.arange(na)
    else:
        na = len(bs)

    # Normalise v_xyz to a direction vector
    if isinstance(v_xyz, (int, float, np.integer)):
        v_xyz = np.roll(np.array([1.0, 0.0, 0.0]), int(v_xyz))
    else:
        v = backend.asarray(v_xyz, dtype=None)
        v_xyz = to_numpy(v / backend.sqrt(backend.sum(v ** 2)))

    nt2 = nt // 2
    ws = np.fft.fftfreq(nt)[:nt2]

    # Project displacements onto polarisation direction  →  shape (n_time, n_atoms)
    if v_xyz.sum() == 1 and v_xyz.dtype == float:
        # Pure axis: fast path via integer index
        axis = int(np.argmax(v_xyz))
        us = displacements[:, bs, axis]
    else:
        us = np.einsum("tax,x->ta", displacements[:, bs, :], v_xyz)

    # exp(i k · r̄)  →  shape (n_atoms, nx, ny)
    expo = backend.exp(
        1j * backend.einsum('aj,xyj->axy',
                            backend.asarray(avg[bs, :]),
                            backend.asarray(kvec)))

    # Σn u°(n,t) exp(i k x̄(n))  →  shape (n_time, nx, ny)
    integrands = backend.einsum('ta,axy->txy', backend.asarray(us), expo)

    # FFT along time, keep positive frequencies, take intensity
    Zs = backend.fft(integrands, axes=0)
    # Slice positive-frequency half — use numpy indexing after to_numpy
    Zs_np = to_numpy(Zs)[:nt2, :, :]
    Zs_np = np.abs(Zs_np) ** 2

    return Zs_np, ws

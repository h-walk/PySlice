"""
Core data structure for TACAW EELS calculations.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .wf_data import WFData
from ..data.pyslice_serial import PySliceSerial, Signal, Dimensions, Dimension, Metadata
from pyslice.backend import Backend, to_numpy

logger = logging.getLogger(__name__)

K_B_EV_PER_K = 8.617333262145e-5
THZ_TO_EV = 4.135667696e-3


def bose_correction_factor(frequencies_THz, temperature_K: float) -> np.ndarray:
    """Return beta E / (1 - exp(-beta E)) for TACAW gain/loss balance."""
    frequencies = np.asarray(to_numpy(frequencies_THz), dtype=np.float64)
    beta_E = frequencies * THZ_TO_EV / (K_B_EV_PER_K * temperature_K)
    beta_E = np.clip(beta_E, -500.0, 500.0)
    factor = np.ones_like(beta_E, dtype=np.float64)
    nonzero = np.abs(beta_E) > 1e-12
    factor[nonzero] = beta_E[nonzero] / (1.0 - np.exp(-beta_E[nonzero]))
    return factor


class TACAWData(PySliceSerial, Signal):
    """
    TACAW EELS results: (probe_positions, frequency, kx, ky).

    Converts a WFData wavefunction (time-domain) to spectral intensity
    |Ψ(ω,q)|² via FFT along the time axis.
    """

    _sea_config = {
        'tensor_attrs': ['_kxs', '_kys', '_xs', '_ys', '_time', '_layer',
                         '_frequencies', '_array', 'data'],
        'path_attrs': ['cache_dir'],
        'tuple_list_attrs': ['probe_positions'],
        'exclude_attrs': ['probe', '_wf_array', '_backend'],
        'force_datasets': ['_array', 'probe_positions', '_kxs', '_kys',
                           '_xs', '_ys', '_time', '_layer', '_frequencies'],
    }

    def __init__(self,
                 wf_data: WFData,
                 layer_index: Optional[int] = None,
                 keep_complex: bool = False,
                 chunkFFT: bool = False,
                 chunk_size_time: Optional[int] = None,
                 force_rerun: bool = False,
                 temperature_K: Optional[float] = None,
                 apply_bose: bool = False) -> None:

        self._backend = wf_data._backend

        # Copy coordinate metadata from WFData
        self.probe_positions = wf_data.probe_positions
        self._time  = wf_data._time
        self._kxs   = wf_data._kxs
        self._kys   = wf_data._kys
        self._xs    = wf_data._xs
        self._ys    = wf_data._ys
        self._layer = wf_data._layer
        self.probe  = wf_data.probe
        self.cache_dir   = wf_data.cache_dir
        self.keep_complex  = keep_complex
        self.chunkFFT      = chunkFFT
        self.use_memmap    = isinstance(wf_data._array, np.memmap)
        self.chunk_size_time = chunk_size_time
        self.force_rerun   = force_rerun
        self.temperature_K = temperature_K
        self.apply_bose = apply_bose

        self._wf_array   = wf_data._array
        self._array      = None
        self._frequencies = None

        self._fft_from_wf_data(layer_index)
        if self.apply_bose:
            self.apply_bose_correction(self.temperature_K)

        if Dimensions is not None:
            self.dimensions = Dimensions([
                Dimension(name='probe',     space='position',
                          values=np.arange(len(self.probe_positions))),
                Dimension(name='frequency', space='spectral', units='THz',
                          values=to_numpy(self._frequencies)),
                Dimension(name='kx',        space='scattering', units='Å⁻¹',
                          values=to_numpy(self._kxs)),
                Dimension(name='ky',        space='scattering', units='Å⁻¹',
                          values=to_numpy(self._kys)),
            ], nav_dimensions=[0, 1], sig_dimensions=[2, 3])

            self.metadata = Metadata({
                'General':    {'title': 'TACAW Intensity', 'signal_type': 'TACAW'},
                'Simulation': {
                    'voltage_eV':    float(self.probe.eV),
                    'wavelength_A':  float(self.probe.wavelength),
                    'aperture_mrad': float(self.probe.mrad),
                    'probe_positions': [list(p) for p in self.probe_positions],
                    'temperature_K': None if self.temperature_K is None else float(self.temperature_K),
                    'bose_corrected': bool(self.apply_bose),
                },
            })
            self.sea_type = "Signal"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kxs(self)         -> np.ndarray: return to_numpy(self._kxs)
    @property
    def kys(self)         -> np.ndarray: return to_numpy(self._kys)
    @property
    def xs(self)          -> np.ndarray: return to_numpy(self._xs)
    @property
    def ys(self)          -> np.ndarray: return to_numpy(self._ys)
    @property
    def frequencies(self) -> np.ndarray: return to_numpy(self._frequencies)

    @property
    def data(self):
        return to_numpy(self._array) if self._array is not None else None

    @data.setter
    def data(self, value):
        self._array = value

    @property
    def intensity(self):
        return self._array

    @intensity.setter
    def intensity(self, value):
        self._array = value

    @property
    def array(self):
        return to_numpy(self._array) if self._array is not None else None

    def apply_bose_correction(self, temperature_K: float):
        """Apply the detailed-balance Bose factor to an intensity TACAW array."""
        if temperature_K is None:
            raise ValueError("temperature_K must be provided when apply_bose=True")
        if self.keep_complex:
            raise ValueError("Bose correction expects intensity data; set keep_complex=False")

        b = self._backend
        factor = b.asarray(bose_correction_factor(self._frequencies, temperature_K), dtype=self._array.dtype)
        self._array = self._array * factor[None, :, None, None]
        self.temperature_K = temperature_K
        self.apply_bose = True
        return self

    # ------------------------------------------------------------------
    # FFT computation
    # ------------------------------------------------------------------

    def _fft_from_wf_data(self, layer_index: Optional[int] = None):
        """FFT along the time axis to convert wavefunction to TACAW data."""
        b = self._backend

        cache_tacaw = self.cache_dir / "tacaw.npy"
        cache_freq  = self.cache_dir / "tacaw_freq.npy"
        cache_meta  = self.cache_dir / "tacaw_meta.json"

        fft_len = self.chunk_size_time if self.chunk_size_time is not None else len(self._time)
        if self.chunk_size_time is None:
            self.n_chunks = 1
        else:
            if self.chunk_size_time <= 0:
                raise ValueError("chunk_size_time must be a positive integer")
            elif self.chunk_size_time > len(self._time):
                raise ValueError("chunk_size_time cannot exceed total time length")
            elif len(self._time) % self.chunk_size_time != 0:
                raise ValueError("chunk_size_time must evenly divide total time length")
            else:
                self.n_chunks = len(self._time) // self.chunk_size_time

        # Resolve the layer up front: it — together with keep_complex, the
        # chunking and the source-wavefunction identity — is part of the cache
        # identity, so a different layer / dtype / dataset sharing this cache_dir
        # is never served the wrong cached spectrum (a shape-only check was).
        if layer_index is None:
            layer_index = len(self._layer) - 1
        if not (0 <= layer_index < len(self._layer)):
            raise ValueError(
                f"layer_index {layer_index} out of range [0, {len(self._layer) - 1}]")

        meta = self._tacaw_cache_meta(layer_index, fft_len)
        if not self.force_rerun and cache_tacaw.exists() and cache_meta.exists():
            try:
                with open(cache_meta) as f:
                    cached_meta = json.load(f)
            except (OSError, ValueError):
                cached_meta = None
            if cached_meta == meta:
                cached = np.load(cache_tacaw)
                if list(cached.shape) == meta["array_shape"]:
                    self._frequencies = b.asarray(np.load(cache_freq))
                    self._array = b.asarray(cached)
                    return

        # A (re)compute invalidates any previous completion marker first, so an
        # interrupted run (partial tacaw.npy — notably the memmap accumulator)
        # is never mistaken for a complete cache on the next load.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_meta.exists():
            cache_meta.unlink()

        wf_layer = self._wf_array[:, :, :, :, layer_index]  # p,t,kx,ky

        indices = np.linspace(0, len(self._time), self.n_chunks + 1)
        dt = float(to_numpy(self._time[1] - self._time[0]))
        self._frequencies = b.fftshift(b.fftfreq(fft_len, d=dt))

        if self.chunkFFT:
            # Memory-conservative path: loop over kx
            dtype = b.complex_dtype if self.keep_complex else b.float_dtype
            shape = (wf_layer.shape[0], fft_len,
                     wf_layer.shape[2], wf_layer.shape[3])
            if self.use_memmap:
                self._array = b.memmap(shape, dtype=dtype,
                                       filename=self.cache_dir / "tacaw.npy")
            else:
                self._array = b.zeros(shape, dtype=dtype)

            for chunk_i in range(self.n_chunks):
                i1, i2 = int(to_numpy(indices[chunk_i])), int(to_numpy(indices[chunk_i + 1]))
                for kx_i in tqdm(range(len(self._kxs))):
                    sl = wf_layer[:, i1:i2, kx_i, :]
                    wf_mean = b.mean(sl, axis=1, keepdims=True)
                    wf_fft  = b.fftshift(b.fft(sl - wf_mean, axes=1), axes=1)
                    if not self.keep_complex:
                        wf_fft = b.absolute(wf_fft) ** 2
                    self._array[:, :, kx_i, :] += wf_fft
        else:
            # Standard path: FFT over full time window
            for chunk_i in range(self.n_chunks):
                i1, i2 = int(to_numpy(indices[chunk_i])), int(to_numpy(indices[chunk_i + 1]))
                sl = wf_layer[:, i1:i2, :, :]
                wf_mean = b.mean(sl, axis=1, keepdims=True)
                wf_fft  = b.fftshift(b.fft(sl - wf_mean, axes=1), axes=1)
                if not self.keep_complex:
                    wf_fft = b.absolute(wf_fft) ** 2
                self._array = wf_fft if self._array is None else self._array + wf_fft

        # Persist to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_freq,  to_numpy(self._frequencies))
        if not self.use_memmap:
            np.save(cache_tacaw, to_numpy(self._array))
        # Completion marker written LAST, so its presence guarantees a fully
        # written tacaw.npy that matches this metadata.
        with open(cache_meta, "w") as f:
            json.dump(meta, f)

    # Bump when the TACAW FFT/normalisation changes so stale tacaw.npy caches
    # are not reused across versions.
    _TACAW_CACHE_VERSION = 1

    def _tacaw_cache_meta(self, layer_index: int, fft_len: int) -> dict:
        """Identity of the cached spectrum: everything that changes its values."""
        n_probes = int(self._wf_array.shape[0])
        nkx = int(self._wf_array.shape[2])
        nky = int(self._wf_array.shape[3])
        return {
            "cache_version": self._TACAW_CACHE_VERSION,
            "layer_index": int(layer_index),
            "keep_complex": bool(self.keep_complex),
            "fft_len": int(fft_len),
            "n_chunks": int(self.n_chunks),
            "array_shape": [n_probes, int(fft_len), nkx, nky],
            "wf_dtype": str(getattr(self._wf_array, "dtype", "")),
            "wf_fingerprint": self._wf_fingerprint(),
        }

    def _wf_fingerprint(self) -> str:
        """Content fingerprint of the source wavefunction array.

        Samples up to ~1M elements strided across the whole array (so it detects
        any global difference, unlike a frame-0-style sample) and hashes them
        with the shape — enough to tell a different dataset in the same cache_dir
        apart from this one without a full multi-GB scan.
        """
        arr = self._wf_array
        flat = arr.reshape(-1)
        n = int(flat.shape[0])
        step = max(1, n // (1 << 20))
        sample = np.ascontiguousarray(to_numpy(flat[::step]))
        payload = repr(tuple(int(s) for s in arr.shape)).encode() + sample.tobytes()
        return hashlib.md5(payload).hexdigest()

    def fft_from_wf_data(self, layer_index: Optional[int] = None):
        """Public alias for backward compatibility."""
        self._fft_from_wf_data(layer_index)

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def spectrum(self, probe_index: Optional[int] = None) -> np.ndarray:
        """Spectrum for one probe (or mean over all) by summing over k-space."""
        b = self._backend
        if probe_index is None:
            spectra = [to_numpy(b.sum(self._array[i], axis=(1, 2)))
                       for i in range(len(self.probe_positions))]
            return np.mean(spectra, axis=0)
        if probe_index >= len(self.probe_positions):
            raise ValueError(f"Probe index {probe_index} out of range")
        return to_numpy(b.sum(self._array[probe_index], axis=(1, 2)))

    def spectrum_image(self, frequency: float,
                       probe_indices: Optional[List[int]] = None) -> np.ndarray:
        """Intensity at a given frequency for each probe position (real-space map)."""
        b = self._backend
        freq_idx = int(np.argmin(np.abs(self.frequencies - frequency)))
        if probe_indices is None:
            probe_indices = list(range(len(self.probe_positions)))
        return np.array([to_numpy(b.sum(self._array[p, freq_idx, :, :])) for p in probe_indices])


    def diffraction(self, probe_index: Optional[int] = None,
                    space: str = "reciprocal") -> np.ndarray:
        """Diffraction pattern (kx, ky) summed over all frequencies."""
        b = self._backend
        array_dtype = getattr(self._array, "dtype", b.complex_dtype)
        if probe_index is None:
            patterns = [to_numpy(b.sum(self._array[i], axis=0))
                        for i in range(len(self.probe_positions))]
            pattern = np.mean(patterns, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")
            pattern = to_numpy(b.sum(self._array[probe_index], axis=0))

        if space == "real":
            pattern = to_numpy(b.absolute(b.ifft2(b.asarray(pattern, dtype=array_dtype))))
        return pattern

    def spectral_diffraction(self, frequency: float,
                             probe_index: Optional[int] = None,
                             space: str = "reciprocal") -> np.ndarray:
        """Diffraction pattern at a specific frequency."""
        b = self._backend
        freq_idx = int(np.argmin(np.abs(self.frequencies - frequency)))

        if probe_index is None:
            slices = [self._array[i, freq_idx, :, :]
                      for i in range(len(self.probe_positions))]
            array_dtype = getattr(self._array, "dtype", b.complex_dtype)
            pattern = to_numpy(
                b.mean(b.stack([b.asarray(s, dtype=array_dtype) for s in slices]), axis=0)
            )
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")
            pattern = to_numpy(self._array[probe_index, freq_idx, :, :])

        if space == "real":
            array_dtype = getattr(self._array, "dtype", b.complex_dtype)
            pattern = to_numpy(b.absolute(b.ifft2(b.asarray(pattern, dtype=array_dtype))))
        return pattern

    def masked_spectrum(self, mask=None, probe_index: Optional[int] = None,
                        preview: bool = False) -> np.ndarray:
        """Spectrum with k-space masking applied."""
        b = self._backend
        kxs_np = to_numpy(self._kxs)
        kys_np = to_numpy(self._kys)

        if mask is None:
            mask = np.ones((len(kxs_np), len(kys_np)))
        elif isinstance(mask, dict):
            cx, cy = mask.get("center", (0, 0))
            if mask["shape"] == "round":
                r = mask["radius"]
                radii = np.sqrt((kxs_np[:, None] - cx) ** 2 + (kys_np[None, :] - cy) ** 2)
                mask = (radii <= r).astype(float)
        elif mask.shape != (len(kxs_np), len(kys_np)):
            raise ValueError(f"Mask shape {mask.shape} doesn't match "
                             f"k-space shape ({len(kxs_np)}, {len(kys_np)})")

        if not isinstance(self._array, (np.ndarray, np.memmap)):
            mask = b.asarray(mask, dtype=self._array.dtype)

        probe_indices = (np.arange(len(self.probe_positions))
                         if probe_index is None else [probe_index])
        spectra = []
        for i in probe_indices:
            masked = self._array[i] * mask[None, :, :]
            if preview:
                import matplotlib.pyplot as plt
                extent = (kxs_np.min(), kxs_np.max(), kys_np.min(), kys_np.max())
                fig, ax = plt.subplots()
                ax.imshow(to_numpy(b.sum(masked, axis=0)).T[::-1, :],
                          cmap="inferno", extent=extent, aspect=1)
                ax.set_xlabel("kx"); ax.set_ylabel("ky")
                ax.set_title("masked_spectrum - preview")
                plt.show()
                plt.close(fig)
                preview = False
            spectra.append(to_numpy(b.sum(masked, axis=(1, 2))))
        return np.mean(spectra, axis=0)

    def dispersion(self, kx_path: np.ndarray, ky_path: np.ndarray,
                   probe_index: Optional[int] = None,
                   space: str = "reciprocal") -> np.ndarray:
        """Extract dispersion relation along a k-path."""
        b = self._backend
        kx_np = to_numpy(self._kxs) if space != "real" else to_numpy(self._xs)
        ky_np = to_numpy(self._kys) if space != "real" else to_numpy(self._ys)

        kx_indices = np.array([np.argmin(np.abs(kx_np - v)) for v in kx_path])
        ky_indices = np.array([np.argmin(np.abs(ky_np - v)) for v in ky_path])

        probe_indices = (np.arange(len(self.probe_positions))
                         if probe_index is None else [probe_index])
        n_freq = len(self.frequencies)
        dispersion = np.zeros((n_freq, len(kx_indices)), dtype=np.complex128)

        for w in range(n_freq):
            w_slice = self._array[probe_indices, w, :, :]
            if space == "real":
                w_slice = b.ifft2(w_slice, axes=(1, 2))
            w_np = np.mean(to_numpy(w_slice), axis=0)
            for i, (ki, kj) in enumerate(zip(kx_indices, ky_indices)):
                dispersion[w, i] = w_np[ki, kj]

        return np.abs(dispersion)

    # ------------------------------------------------------------------
    # Generic heatmap plot
    # ------------------------------------------------------------------

    def plot(self, intensities, xvals, yvals,
             xlabel="kx (Å⁻¹)", ylabel="ky (Å⁻¹)",
             filename=None, title=None, extent=None):
        import matplotlib.pyplot as plt

        _AXIS_MAP = {
            "kx": ("kx (Å⁻¹)", lambda s: to_numpy(s._kxs)),
            "k":  ("kx (Å⁻¹)", lambda s: to_numpy(s._kxs)),
            "ky": ("ky (Å⁻¹)", lambda s: to_numpy(s._kys)),
            "x":  ("x (Å)",    lambda s: to_numpy(s._xs)),
            "y":  ("y (Å)",    lambda s: to_numpy(s._ys)),
            "omega": ("frequency (THz)", lambda s: s.frequencies),
        }

        if isinstance(xvals, str) and xvals in _AXIS_MAP:
            xlabel, xvals = _AXIS_MAP[xvals][0], _AXIS_MAP[xvals][1](self)
        if isinstance(yvals, str) and yvals in _AXIS_MAP:
            ylabel, yvals = _AXIS_MAP[yvals][0], _AXIS_MAP[yvals][1](self)

        if extent is None:
            extent = (np.amin(xvals), np.amax(xvals), np.amin(yvals), np.amax(yvals))
        aspect = "auto" if ylabel == "frequency (THz)" else None

        fig, ax = plt.subplots()
        ax.imshow(to_numpy(np.abs(intensities)), cmap="inferno",
                  extent=extent, aspect=aspect)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close(fig)


class SEDData(TACAWData):
    """
    SED (Spectral Energy Density) results.
    Functionally identical to TACAWData — both compute |Ψ(ω,q)|² via time-axis FFT.
    """
    def __init__(self, wf_data: WFData, layer_index: Optional[int] = None,
                 keep_complex: bool = False, force_rerun: bool = False) -> None:
        super().__init__(wf_data, layer_index, keep_complex, force_rerun=force_rerun)

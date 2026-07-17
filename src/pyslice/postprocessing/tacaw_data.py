"""
Core data structure for TACAW EELS calculations.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .wf_data import WFData
from ..data.pyslice_serial import PySliceSerial, Signal, Dimensions, Dimension, Metadata
from pyslice.backend import Backend, to_numpy, source_files_version

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
                 segment_length: Optional[int] = None,
                 overlap: float = 0.0,
                 window=None,
                 force_rerun: bool = False,
                 temperature_K: Optional[float] = None,
                 apply_bose: bool = False) -> None:
        """Convert exit-wave time series to a spectral intensity |Psi(w,q)|^2.

        The spectrum is a periodogram estimate averaged (Welch's method) over
        time segments:

            segment_length: L samples per FFT segment (None -> the whole series,
                a single segment). Generalises chunk_size_time (kept as an alias).
            overlap: fraction in [0, 1) of segment overlap; 0 -> non-overlapping
                (Bartlett), 0.5 -> Welch.
            window: taper applied per segment before the FFT -- None/'boxcar'
                (rectangular, the default), 'hann', 'hamming', 'blackman',
                'bartlett', a length-L array, or a callable L -> array. Windows
                are RMS-normalised so 'boxcar' reproduces the un-windowed result.

        Averaging only makes sense on intensities, so keep_complex=True is
        rejected when more than one segment would be averaged.
        """

        # A list/tuple of WFData -> ensemble average over independent trajectories
        # (Welch-segmented, streamed one trajectory at a time; see _compute_ensemble).
        ensemble = isinstance(wf_data, (list, tuple))
        ref = wf_data[0] if ensemble else wf_data

        self._backend = ref._backend

        # Copy coordinate metadata from the (reference) WFData
        self.probe_positions = ref.probe_positions
        self._time  = ref._time
        self._kxs   = ref._kxs
        self._kys   = ref._kys
        self._xs    = ref._xs
        self._ys    = ref._ys
        self._layer = ref._layer
        self.probe  = ref.probe
        self.cache_dir   = ref.cache_dir
        self.keep_complex  = keep_complex
        self.chunkFFT      = chunkFFT
        self.use_memmap    = isinstance(ref._array, np.memmap)
        # segment_length supersedes chunk_size_time (kept as a back-compat alias)
        if segment_length is None and chunk_size_time is not None:
            segment_length = chunk_size_time
        self.chunk_size_time = chunk_size_time
        self.segment_length = segment_length
        self.overlap = float(overlap)
        self.window = window
        self.force_rerun   = force_rerun
        self.temperature_K = temperature_K
        self.apply_bose = apply_bose

        self._wf_array   = ref._array
        self._array      = None
        self._frequencies = None

        self.n_scan_positions = len(self.probe_positions)
        if self.n_scan_positions == 0:
            raise ValueError("TACAWData requires at least one probe position")
        n_wave_rows = int(self._wf_array.shape[0])
        if n_wave_rows % self.n_scan_positions != 0:
            raise ValueError(
                "Wavefunction probe rows must be an integer multiple of the "
                f"{self.n_scan_positions} scan positions; got {n_wave_rows} rows."
            )
        self.n_copies = n_wave_rows // self.n_scan_positions
        if self.keep_complex and self.n_copies > 1:
            raise ValueError(
                "keep_complex=True cannot combine incoherent decoherence copies; "
                "use keep_complex=False to sum their spectral intensities."
            )

        if ensemble:
            self._compute_ensemble(list(wf_data), layer_index)
        else:
            self._fft_from_wf_data(layer_index)
        if self.apply_bose:
            self.apply_bose_correction(self.temperature_K)

        self._apply_signal_dimensions()

    def _apply_signal_dimensions(self):
        """Populate the PySEA Signal dimensions/metadata from the current array."""
        if Dimensions is None:
            return
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
    # Ensemble (multi-trajectory) averaging
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ensemble_compatible(ref, wf, i):
        """Trajectories must share the k-grid, scan grid and time sampling."""
        def close(a, b_):
            a, b_ = to_numpy(a), to_numpy(b_)
            return a.shape == b_.shape and np.allclose(a, b_)
        if len(wf.probe_positions) != len(ref.probe_positions):
            raise ValueError(f"trajectory {i}: probe count differs from trajectory 0")
        if not (close(wf._kxs, ref._kxs) and close(wf._kys, ref._kys)):
            raise ValueError(f"trajectory {i}: k-grid differs from trajectory 0")
        if len(wf._time) != len(ref._time) or not np.isclose(
                float(to_numpy(wf._time[1] - wf._time[0])),
                float(to_numpy(ref._time[1] - ref._time[0]))):
            raise ValueError(f"trajectory {i}: time sampling differs from trajectory 0")

    def _compute_ensemble(self, wfs, layer_index):
        """Average the Welch spectra of independent trajectories, streamed.

        Each trajectory is reduced to its (segment-averaged) spectrum one at a
        time and summed into a host accumulator weighted by its segment count,
        so peak memory is one trajectory plus the accumulator regardless of how
        many trajectories there are. Each trajectory's spectrum is itself cached
        (its own tacaw.npy), so the intermediates are reusable.
        """
        if self.keep_complex:
            raise ValueError("ensemble averaging requires intensities; use keep_complex=False")
        b = self._backend
        ref = wfs[0]
        acc, total_k = None, 0
        for i, wf in enumerate(wfs):
            self._check_ensemble_compatible(ref, wf, i)
            tac = TACAWData(wf, layer_index=layer_index,
                            segment_length=self.segment_length, overlap=self.overlap,
                            window=self.window, force_rerun=self.force_rerun)
            spec = to_numpy(tac._array)          # host; (n_scan, nfreq, nkx, nky), real
            k = int(tac.n_chunks)
            acc = spec * k if acc is None else acc + spec * k   # weighted host sum
            total_k += k
            if self._frequencies is None:
                self._frequencies = tac._frequencies
        self.n_chunks = total_k
        self._array = b.asarray(acc / total_k)

    @classmethod
    def _from_spectrum(cls, array, frequencies, meta):
        """Build a TACAWData directly from a precomputed spectrum (no FFT).

        Used by TACAWAccumulator.finalize() to wrap the reduced host array.
        """
        self = cls.__new__(cls)
        self._backend = meta['_backend']
        b = self._backend
        self.probe_positions = meta['probe_positions']
        self._time = meta['_time']; self._kxs = meta['_kxs']; self._kys = meta['_kys']
        self._xs = meta['_xs']; self._ys = meta['_ys']; self._layer = meta['_layer']
        self.probe = meta['probe']; self.cache_dir = meta['cache_dir']
        self.keep_complex = False; self.chunkFFT = False; self.use_memmap = False
        self.segment_length = meta.get('segment_length'); self.overlap = float(meta.get('overlap', 0.0))
        self.window = meta.get('window'); self.chunk_size_time = None
        self.force_rerun = False; self.temperature_K = None; self.apply_bose = False
        self.n_scan_positions = len(self.probe_positions); self.n_copies = 1
        self.n_chunks = int(meta.get('n_segments', 1))
        self._array = b.asarray(array)
        self._frequencies = b.asarray(frequencies)
        self._apply_signal_dimensions()
        return self

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

    @staticmethod
    def _make_window(window, L, b):
        """Return a length-L, RMS-normalised window as a backend array.

        RMS normalisation (w /= sqrt(mean(w**2))) makes 'boxcar' the identity and
        keeps windowed periodogram amplitudes comparable across window choices.
        """
        if window is None or (isinstance(window, str)
                              and window.lower() in ("boxcar", "rect", "rectangular", "none")):
            w = np.ones(L)
        elif isinstance(window, str):
            fn = {"hann": np.hanning, "hamming": np.hamming,
                  "blackman": np.blackman, "bartlett": np.bartlett}.get(window.lower())
            if fn is None:
                raise ValueError(
                    f"unknown window {window!r}; use 'boxcar', 'hann', 'hamming', "
                    "'blackman', 'bartlett', a length-L array, or a callable")
            w = fn(L)
        elif callable(window):
            w = np.asarray(window(L), dtype=float)
        else:
            w = np.asarray(window, dtype=float)
        w = w.astype(float).reshape(-1)
        if w.shape != (L,):
            raise ValueError(f"window must have length {L}, got {w.shape}")
        rms = np.sqrt(np.mean(w ** 2))
        if rms > 0:
            w = w / rms
        return b.asarray(w)

    def _resolve_segments(self, n_time):
        """Resolve (segment_length, list_of_starts) for Welch segmentation."""
        L = self.segment_length if self.segment_length is not None else n_time
        if not (0 < L <= n_time):
            raise ValueError(f"segment_length must be in [1, {n_time}]; got {L}")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must be a fraction in [0, 1)")
        step = L - int(round(self.overlap * L))
        if step < 1:
            raise ValueError("overlap too large: segment step < 1")
        starts = list(range(0, n_time - L + 1, step))
        if not starts:
            raise ValueError("no segments fit; reduce segment_length")
        if self.keep_complex and len(starts) > 1:
            raise ValueError(
                "keep_complex=True averages complex spectra to ~0; use a single "
                "segment (segment_length=None, overlap=0) to keep the complex spectrum")
        return L, starts

    def _fft_from_wf_data(self, layer_index: Optional[int] = None):
        """FFT along the time axis to convert wavefunction to TACAW data."""
        b = self._backend

        cache_tacaw = self.cache_dir / "tacaw.npy"
        cache_freq  = self.cache_dir / "tacaw_freq.npy"
        cache_meta  = self.cache_dir / "tacaw_meta.json"

        L, starts = self._resolve_segments(len(self._time))
        fft_len = L
        self.n_chunks = len(starts)   # number of averaged segments (Welch)

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

        dt = float(to_numpy(self._time[1] - self._time[0]))
        self._frequencies = b.fftshift(b.fftfreq(fft_len, d=dt))
        window = self._make_window(self.window, L, b)   # RMS-normalised, length L
        n_segments = len(starts)

        def _segment_periodogram(seg):
            # detrend (remove elastic/DC line) -> window -> FFT along time
            seg = seg - b.mean(seg, axis=1, keepdims=True)
            seg = seg * window.reshape((1, L) + (1,) * (seg.ndim - 2))
            out = b.fftshift(b.fft(seg, axes=1), axes=1)
            if not self.keep_complex:
                out = self._fold_incoherent_copies(b.absolute(out) ** 2)
            return out

        # Welch: average the (windowed, detrended) segment periodograms.
        if self.chunkFFT:
            # Memory-conservative path: loop over kx
            dtype = b.complex_dtype if self.keep_complex else b.float_dtype
            shape = (self.n_scan_positions, fft_len,
                     wf_layer.shape[2], wf_layer.shape[3])
            if self.use_memmap:
                self._array = b.memmap(shape, dtype=dtype,
                                       filename=self.cache_dir / "tacaw.npy")
            else:
                self._array = b.zeros(shape, dtype=dtype)

            for start in starts:
                for kx_i in tqdm(range(len(self._kxs))):
                    self._array[:, :, kx_i, :] += _segment_periodogram(
                        wf_layer[:, start:start + L, kx_i, :])
            self._array /= n_segments
        else:
            # Standard path: FFT over the full segment window
            for start in starts:
                contrib = _segment_periodogram(wf_layer[:, start:start + L, :, :])
                self._array = contrib if self._array is None else self._array + contrib
            self._array = self._array / n_segments

        # Persist to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_freq,  to_numpy(self._frequencies))
        if isinstance(self._array, np.memmap):
            self._array.flush()
        else:
            np.save(cache_tacaw, to_numpy(self._array))
        # Completion marker written LAST, so its presence guarantees a fully
        # written tacaw.npy that matches this metadata.
        with open(cache_meta, "w") as f:
            json.dump(meta, f)

    # Derived automatically from the sources that determine the TACAW spectrum
    # values, so a change to the FFT/normalisation invalidates stale tacaw.npy
    # caches without a manual bump. "v1" allows a manual bump if ever needed.
    _TACAW_CACHE_VERSION = "v1-" + source_files_version([
        os.path.join(os.path.dirname(__file__), "tacaw_data.py"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend.py"),
    ])

    def _tacaw_cache_meta(self, layer_index: int, fft_len: int) -> dict:
        """Identity of the cached spectrum: everything that changes its values."""
        n_probes = self.n_scan_positions
        nkx = int(self._wf_array.shape[2])
        nky = int(self._wf_array.shape[3])
        return {
            "cache_version": self._TACAW_CACHE_VERSION,
            "layer_index": int(layer_index),
            "keep_complex": bool(self.keep_complex),
            "fft_len": int(fft_len),
            "n_chunks": int(self.n_chunks),
            "overlap": float(self.overlap),
            "window": (self.window if isinstance(self.window, str)
                       else ("callable" if callable(self.window)
                             else ("array" if self.window is not None else None))),
            "n_copies": int(self.n_copies),
            "array_shape": [n_probes, int(fft_len), nkx, nky],
            "wf_dtype": str(getattr(self._wf_array, "dtype", "")),
            "wf_fingerprint": self._array_fingerprint(
                self._wf_array[:, :, :, :, layer_index]),
            "time_fingerprint": self._array_fingerprint(self._time),
            "kx_fingerprint": self._array_fingerprint(self._kxs),
            "ky_fingerprint": self._array_fingerprint(self._kys),
        }

    def _fold_incoherent_copies(self, intensity):
        """Sum copy-major spectral intensities onto physical scan positions."""
        if self.n_copies == 1:
            return intensity
        b = self._backend
        folded_shape = (
            self.n_copies,
            self.n_scan_positions,
        ) + tuple(int(s) for s in intensity.shape[1:])
        return b.sum(b.reshape(intensity, folded_shape), axis=0)

    @staticmethod
    def _array_fingerprint(arr) -> str:
        """Hash every value without materialising a potentially huge CPU copy."""
        flat = arr.reshape(-1)
        n = int(flat.shape[0])
        digest = hashlib.sha256()
        digest.update(repr(tuple(int(s) for s in arr.shape)).encode())
        digest.update(str(getattr(arr, "dtype", "")).encode())
        for start in range(0, n, 1 << 20):
            block = np.ascontiguousarray(to_numpy(flat[start:start + (1 << 20)]))
            digest.update(block.tobytes())
        return digest.hexdigest()

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


class TACAWAccumulator:
    """Host-resident streaming accumulator for ensemble TACAW averaging.

    Add trajectories (or probe-batch subsets) one at a time; each is reduced to
    its Welch periodogram and summed into a host array, so peak memory is one
    trajectory's working set plus the accumulator, regardless of the number of
    trajectories. ``rows`` selects which probe rows a partial fills (probe
    batching), and ``memmap_path`` puts the accumulator on disk for scans too
    large for host RAM. This is the local, single-process reducer; a multi-GPU
    driver simply sums several of these host accumulators (see the design note).

    Example
    -------
    >>> acc = TACAWAccumulator(window='hann', segment_length=L, overlap=0.5)
    >>> for wf in trajectory_wavefunctions:      # produced one at a time
    ...     acc.add(wf); del wf
    >>> tacaw = acc.finalize()                    # ensemble-averaged spectrum
    """

    def __init__(self, *, segment_length=None, overlap=0.0, window=None,
                 layer_index=None, n_probes=None, dtype=np.float64, memmap_path=None):
        self._kw = dict(segment_length=segment_length, overlap=overlap,
                        window=window, layer_index=layer_index)
        self.n_probes = n_probes
        self.dtype = dtype
        self.memmap_path = None if memmap_path is None else str(memmap_path)
        self._acc = None
        self._count = None
        self._freqs = None
        self._meta = None

    def add(self, wf_data, rows=None):
        """Reduce one trajectory (optionally a probe-batch) and add its periodogram.

        rows: probe indices this partial fills; None -> all rows (a full-grid
        trajectory). Trajectories/batches covering the same probe are averaged.
        """
        tac = TACAWData(wf_data, keep_complex=False, **self._kw)
        spec = to_numpy(tac._array)            # (n_rows, nfreq, nkx, nky), real
        k = int(tac.n_chunks)
        if self._acc is None:
            n = self.n_probes if self.n_probes is not None else spec.shape[0]
            shape = (n,) + spec.shape[1:]
            if self.memmap_path is not None:
                self._acc = np.lib.format.open_memmap(
                    self.memmap_path, mode='w+', dtype=self.dtype, shape=shape)
                self._acc[:] = 0
            else:
                self._acc = np.zeros(shape, dtype=self.dtype)
            self._count = np.zeros(n, dtype=self.dtype)
            self._freqs = to_numpy(tac._frequencies)
            self._meta = {a: getattr(tac, a) for a in
                          ('probe_positions', '_kxs', '_kys', '_xs', '_ys',
                           '_layer', '_time', 'probe', 'cache_dir', '_backend')}
        idx = slice(None) if rows is None else np.asarray(rows)
        self._acc[idx] += spec * k             # weighted by this unit's segment count
        self._count[idx] += k
        return self

    def finalize(self) -> "TACAWData":
        """Return the ensemble-averaged spectrum as a TACAWData."""
        if self._acc is None:
            raise RuntimeError("TACAWAccumulator.finalize() called before any add()")
        cnt = np.where(np.asarray(self._count) > 0, self._count, 1.0)
        avg = np.asarray(self._acc) / cnt[:, None, None, None]
        meta = dict(self._meta, segment_length=self._kw['segment_length'],
                    overlap=self._kw['overlap'], window=self._kw['window'])
        return TACAWData._from_spectrum(avg, self._freqs, meta)

    def save_partial(self, path) -> str:
        """Serialise this rank's UN-averaged partial (sum + counts + metadata).

        Written as a small .npz so a cross-rank reduce (``reduce_tacaw_partials``)
        can sum several ranks' partials into the ensemble average. Only numeric
        state is stored (no backend/probe objects), so it is portable and the
        reduce can run anywhere.
        """
        if self._acc is None:
            raise RuntimeError("save_partial() called before any add()")
        m = self._meta
        np.savez(
            str(path),
            acc=np.asarray(self._acc), count=np.asarray(self._count),
            freqs=np.asarray(self._freqs),
            kxs=to_numpy(m['_kxs']), kys=to_numpy(m['_kys']),
            xs=to_numpy(m['_xs']), ys=to_numpy(m['_ys']),
            layer=to_numpy(m['_layer']), time=to_numpy(m['_time']),
            probe_positions=np.asarray(m['probe_positions'], dtype=float),
            probe_eV=float(m['probe'].eV),
            probe_wavelength=float(m['probe'].wavelength),
            probe_mrad=float(m['probe'].mrad),
            segment_length=(-1 if self._kw['segment_length'] is None
                            else int(self._kw['segment_length'])),
            overlap=float(self._kw['overlap']),
            window=("" if self._kw['window'] is None else str(self._kw['window'])),
        )
        return str(path)


def reduce_tacaw_partials(partials, backend=None) -> "TACAWData":
    """Sum file-based TACAWAccumulator partials into an ensemble-averaged TACAWData.

    partials: a directory (globs ``partial_*.npz``) or an explicit list of .npz
    paths written by :meth:`TACAWAccumulator.save_partial`. This is the cross-rank
    (and cross-node) reduce: it sums the per-rank un-averaged periodogram sums and
    counts, then divides. Missing ranks simply do not contribute (fault tolerant).
    """
    from pyslice.backend import NumpyBackend
    if isinstance(partials, (str, Path)):
        paths = sorted(glob.glob(str(Path(partials) / "partial_*.npz")))
    else:
        paths = [str(p) for p in partials]
    if not paths:
        raise FileNotFoundError(f"no partial_*.npz found in {partials!r}")

    acc = count = ref = None
    for p in paths:
        d = np.load(p, allow_pickle=False)
        acc = d['acc'].astype(np.float64) if acc is None else acc + d['acc']
        count = d['count'].astype(np.float64) if count is None else count + d['count']
        ref = ref or {k: d[k] for k in d.files}
    b = backend or NumpyBackend()
    cnt = np.where(count > 0, count, 1.0)
    avg = acc / cnt[:, None, None, None]
    seg = int(ref['segment_length']); seg = None if seg < 0 else seg
    win = str(ref['window']); win = None if win == "" else win
    probe = SimpleNamespace(
        eV=float(ref['probe_eV']), wavelength=float(ref['probe_wavelength']),
        mrad=float(ref['probe_mrad']),
        _array=b.asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    meta = dict(
        _backend=b, probe=probe, cache_dir=Path('.'),
        probe_positions=[tuple(pp) for pp in ref['probe_positions']],
        _kxs=b.asarray(ref['kxs']), _kys=b.asarray(ref['kys']),
        _xs=b.asarray(ref['xs']), _ys=b.asarray(ref['ys']),
        _layer=b.asarray(ref['layer']), _time=b.asarray(ref['time']),
        segment_length=seg, overlap=float(ref['overlap']), window=win,
        n_segments=int(np.max(count)) if count.size else 1)
    return TACAWData._from_spectrum(avg, b.asarray(ref['freqs']), meta)

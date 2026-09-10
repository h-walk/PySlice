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
from pyslice.backend import Backend, to_numpy, source_files_version

logger = logging.getLogger(__name__)

K_B_EV_PER_K = 8.617333262145e-5
THZ_TO_EV = 4.135667696e-3


def _hash_array(array) -> str:
    """Return a content hash including dtype and shape."""
    values = np.ascontiguousarray(to_numpy(array))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(repr(values.shape).encode())
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def bose_correction_factor(frequencies_THz, temperature_K: float) -> np.ndarray:
    """Return beta E / (1 - exp(-beta E)) for TACAW gain/loss balance."""
    if temperature_K is None or temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    frequencies = np.asarray(to_numpy(frequencies_THz), dtype=np.float64)
    beta_E = frequencies * THZ_TO_EV / (K_B_EV_PER_K * temperature_K)
    beta_E = np.clip(beta_E, -500.0, 500.0)
    factor = np.ones_like(beta_E, dtype=np.float64)
    nonzero = np.abs(beta_E) > 1e-12
    factor[nonzero] = beta_E[nonzero] / (1.0 - np.exp(-beta_E[nonzero]))
    return factor


def _inversion_indices(coordinates, axis_name: str) -> np.ndarray:
    """Map a shifted FFT coordinate axis onto its inversion partner.

    Ordinary coordinate pairs are matched explicitly. For an even FFT grid,
    the lone negative Nyquist bin is its own periodic partner.
    """
    values = np.asarray(to_numpy(coordinates), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{axis_name} coordinates must be a nonempty one-dimensional array")

    scale = max(1.0, float(np.max(np.abs(values))))
    atol = 1e-10 * scale
    indices = np.empty(len(values), dtype=np.int64)
    unmatched = []
    for index, value in enumerate(values):
        matches = np.flatnonzero(np.isclose(values, -value, rtol=1e-9, atol=atol))
        if len(matches) == 1:
            indices[index] = int(matches[0])
        elif len(matches) == 0:
            unmatched.append(index)
        else:
            raise ValueError(f"{axis_name} coordinates contain duplicate inversion partners")

    # fftshift(fftfreq(N)) has one unpaired negative Nyquist coordinate when N
    # is even. It represents the same periodic sample as positive Nyquist.
    if unmatched:
        differences = np.diff(values)
        uniform_shifted_grid = (
            len(values) > 1
            and np.all(differences > 0.0)
            and np.allclose(differences, differences[0], rtol=1e-9, atol=atol)
            and np.any(np.isclose(values, 0.0, rtol=0.0, atol=atol))
            and np.isclose(
                abs(values[unmatched[0]]),
                values[-1] + differences[0],
                rtol=1e-9,
                atol=atol,
            )
        )
        if (
            len(unmatched) == 1
            and unmatched[0] == int(np.argmin(values))
            and values[unmatched[0]] < 0.0
            and uniform_shifted_grid
        ):
            indices[unmatched[0]] = unmatched[0]
        else:
            raise ValueError(
                f"{axis_name} coordinates are not closed under inversion; "
                "folding requires every coordinate q to have a -q partner"
            )

    if not np.array_equal(indices[indices], np.arange(len(values))):
        raise ValueError(f"{axis_name} inversion mapping is not self-consistent")
    return indices


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
        'default_attrs': {'gain_loss_folded': False, 'apply_bose': False,
                          'temperature_K': None},
    }

    def __init__(self,
                 wf_data: WFData,
                 layer_index: Optional[int] = None,
                 keep_complex: bool = False,
                 chunkFFT: bool = False,
                 chunk_size_time: Optional[int] = None,
                 force_rerun: bool = False,
                 temperature_K: Optional[float] = None,
                 apply_bose: bool = False,
                 fold: bool = False) -> None:
        """Transform time-domain exit waves into TACAW frequency data.

        Args:
            wf_data: Wavefunctions shaped ``(probe, time, kx, ky, layer)``.
            layer_index: Index within ``wf_data.layer`` to transform. Defaults
                to the last returned layer.
            keep_complex: Keep complex FFT amplitudes instead of converting to
                intensity ``abs(FFT)**2``.
            chunkFFT: Loop over reciprocal x values to reduce peak FFT memory.
            chunk_size_time: Optional time-window length. It must be positive
                and divide the number of saved frames exactly.
            force_rerun: Ignore a compatible ``tacaw.npy`` cache.
            temperature_K: Sample temperature used by the Bose correction.
            apply_bose: Apply signed Bose weighting after the FFT. Requires
                ``temperature_K`` and ``keep_complex=False``. Does not enable
                gain/loss folding by itself.
            fold: Average inversion-related gain/loss intensity pairs before
                any Bose weighting. Defaults to False and can be enabled
                independently of ``apply_bose``. Requires ``keep_complex=False``.

        Notes:
            Frequencies are in THz. Negative bins represent gain and positive
            bins represent loss under PySlice's FFT convention.
        """

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
        requested_bose_correction = apply_bose
        self.apply_bose = False
        self.gain_loss_folded = False
        self.source_fingerprint = getattr(wf_data, "source_fingerprint", None)

        self._wf_array   = wf_data._array
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

        self._fft_from_wf_data(layer_index)
        if requested_bose_correction:
            self.apply_bose_correction(self.temperature_K, fold=fold)
        elif fold:
            self.fold_gain_loss()

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
                    'gain_loss_folded': bool(self.gain_loss_folded),
                    'bose_corrected': bool(self.apply_bose),
                },
            })
            self.sea_type = "Signal"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kxs(self) -> np.ndarray:
        """Reciprocal x coordinates in inverse Angstroms."""
        return to_numpy(self._kxs)
    @property
    def kys(self) -> np.ndarray:
        """Reciprocal y coordinates in inverse Angstroms."""
        return to_numpy(self._kys)
    @property
    def xs(self) -> np.ndarray:
        """Real-space x coordinates in Angstroms."""
        return to_numpy(self._xs)
    @property
    def ys(self) -> np.ndarray:
        """Real-space y coordinates in Angstroms."""
        return to_numpy(self._ys)
    @property
    def frequencies(self) -> np.ndarray:
        """Signed FFT frequency bins in THz."""
        return to_numpy(self._frequencies)

    @property
    def data(self):
        """TACAW data converted to a CPU NumPy array."""
        return to_numpy(self._array) if self._array is not None else None

    @data.setter
    def data(self, value):
        self._array = value

    @property
    def intensity(self):
        """Backend-native TACAW intensity array."""
        return self._array

    @intensity.setter
    def intensity(self, value):
        self._array = value

    @property
    def array(self):
        """TACAW data converted to a CPU NumPy array."""
        return to_numpy(self._array) if self._array is not None else None

    def apply_bose_correction(self, temperature_K: float, *, fold: bool = False):
        """Apply signed Bose weighting, optionally folding gain/loss pairs first.

        By default, multiply the existing intensity by
        ``beta E / (1 - exp(-beta E))`` without imposing inversion symmetry.
        With ``fold=True``, first average the classical intensity under
        ``I(q, -frequency) = I(-q, frequency)``. Already folded data stays folded.

        Args:
            temperature_K: Sample temperature in kelvin.
            fold: Fold classical gain/loss partners before weighting. Defaults
                to False; this flag does not undo an earlier explicit fold.

        Returns:
            This object, after mutating its intensity data and metadata.
        """
        if temperature_K is None:
            raise ValueError("temperature_K must be provided when apply_bose=True")
        if temperature_K <= 0:
            raise ValueError("temperature_K must be positive")
        if self.keep_complex:
            raise ValueError("Bose correction expects intensity data; set keep_complex=False")
        if self.apply_bose:
            raise ValueError("Bose correction has already been applied to this object")

        if fold:
            self.fold_gain_loss()
        b = self._backend
        factor = b.asarray(bose_correction_factor(self._frequencies, temperature_K), dtype=self._array.dtype)
        self._array = self._array * factor[None, :, None, None]
        self.temperature_K = temperature_K
        self.apply_bose = True
        if hasattr(self, "metadata") and self.metadata is not None:
            self.metadata.Simulation.temperature_K = float(temperature_K)
            self.metadata.Simulation.gain_loss_folded = bool(self.gain_loss_folded)
            self.metadata.Simulation.bose_corrected = True
        return self

    def fold_gain_loss(self):
        """Average classical gain/loss partners before quantum correction.

        Enforces ``I(q, -frequency) = I(-q, frequency)`` by averaging each
        inversion-related pair. Both signed-frequency halves are retained so
        a subsequent Bose correction can build the gain side by detailed
        balance. Pair averaging, rather than summation, preserves total
        spectral weight and normalization.

        Returns:
            This object, after mutating its intensity data and metadata.
        """
        if self.keep_complex:
            raise ValueError("Gain/loss folding expects intensity data; set keep_complex=False")
        if self.apply_bose:
            raise ValueError("Gain/loss folding must be applied before the Bose correction")
        if getattr(self, "gain_loss_folded", False):
            return self

        expected_shape = (
            len(self._frequencies), len(self._kxs), len(self._kys)
        )
        if tuple(self._array.shape[1:]) != expected_shape:
            raise ValueError(
                "TACAW array frequency/kx/ky dimensions do not match its coordinates"
            )

        frequency_indices = _inversion_indices(self._frequencies, "frequency")
        kx_indices = _inversion_indices(self._kxs, "kx")
        ky_indices = _inversion_indices(self._kys, "ky")

        b = self._backend
        frequency_indices = b.asarray(frequency_indices, dtype=int)
        kx_indices = b.asarray(kx_indices, dtype=int)
        ky_indices = b.asarray(ky_indices, dtype=int)
        partner = self._array[:, frequency_indices, :, :]
        partner = partner[:, :, kx_indices, :]
        partner = partner[:, :, :, ky_indices]
        self._array = 0.5 * (self._array + partner)
        self.gain_loss_folded = True
        if hasattr(self, "metadata") and self.metadata is not None:
            self.metadata.Simulation.gain_loss_folded = True
        return self

    # ------------------------------------------------------------------
    # FFT computation
    # ------------------------------------------------------------------

    def _fft_from_wf_data(self, layer_index: Optional[int] = None):
        """FFT along the time axis to convert wavefunction to TACAW data."""
        b = self._backend

        if self._wf_array is None or self._wf_array.shape[-1] == 0:
            raise ValueError("TACAW requires at least one returned wavefunction layer")
        if len(self._time) < 2:
            raise ValueError("TACAW requires at least two uniformly spaced time samples")

        if layer_index is None:
            layer_index = len(self._layer) - 1
        if not (0 <= layer_index < len(self._layer)):
            raise ValueError(
                f"layer_index {layer_index} out of range [0, {len(self._layer) - 1}]")
        self.layer_index = int(layer_index)

        time_values = np.asarray(to_numpy(self._time), dtype=float)
        time_steps = np.diff(time_values)
        if time_steps[0] <= 0:
            raise ValueError(
                "TACAW requires a positive chronological frame spacing; "
                "random/frozen configurations are not a time series"
            )
        if not np.allclose(time_steps, time_steps[0], rtol=1e-7, atol=1e-12):
            raise ValueError("TACAW requires uniformly spaced time samples")

        cache_dir = None if self.cache_dir is None else Path(self.cache_dir)
        cache_tacaw = None if cache_dir is None else cache_dir / "tacaw.npy"
        cache_freq = None if cache_dir is None else cache_dir / "tacaw_freq.npy"
        cache_meta = None if cache_dir is None else cache_dir / "tacaw_manifest.json"

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
        if (not self.force_rerun and cache_dir is not None
                and cache_tacaw.exists() and cache_meta.exists()
                and cache_freq.exists()):
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
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if cache_meta.exists():
                cache_meta.unlink()

        wf_layer = self._wf_array[:, :, :, :, layer_index]  # p,t,kx,ky

        indices = np.linspace(0, len(self._time), self.n_chunks + 1)
        dt = float(time_steps[0])
        self._frequencies = b.fftshift(b.fftfreq(fft_len, d=dt))

        if self.chunkFFT:
            # Memory-conservative path: loop over kx
            dtype = b.complex_dtype if self.keep_complex else b.float_dtype
            shape = (self.n_scan_positions, fft_len,
                     wf_layer.shape[2], wf_layer.shape[3])
            if self.use_memmap:
                if cache_tacaw is None:
                    raise ValueError("Memmapped TACAW output requires wf_data.cache_dir")
                self._array = b.memmap(shape, dtype=dtype,
                                       filename=cache_tacaw)
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
                        wf_fft = self._fold_incoherent_copies(wf_fft)
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
                    wf_fft = self._fold_incoherent_copies(wf_fft)
                self._array = wf_fft if self._array is None else self._array + wf_fft

        # Completion marker is written last, after the entire array is flushed.
        if cache_dir is not None:
            np.save(cache_freq, to_numpy(self._frequencies))
            if isinstance(self._array, np.memmap):
                self._array.flush()
            else:
                np.save(cache_tacaw, to_numpy(self._array))
            metadata_tmp = cache_meta.with_suffix(".json.tmp")
            metadata_tmp.write_text(json.dumps(meta, sort_keys=True))
            metadata_tmp.replace(cache_meta)

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
        """Recompute TACAW data from the stored wavefunctions.

        Args:
            layer_index: Index within the stored layer axis. Defaults to the
                final returned layer.

        Notes:
            This compatibility method mutates the object and returns ``None``.
        """
        previous_force = self.force_rerun
        try:
            self.force_rerun = True
            self._array = None
            self.gain_loss_folded = False
            self.apply_bose = False
            self._fft_from_wf_data(layer_index)
            if hasattr(self, "metadata") and self.metadata is not None:
                self.metadata.Simulation.gain_loss_folded = False
                self.metadata.Simulation.bose_corrected = False
        finally:
            self.force_rerun = previous_force

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def spectrum(self, probe_index: Optional[int] = None) -> np.ndarray:
        """Integrate over k-space to obtain a frequency spectrum.

        Args:
            probe_index: Probe to use. ``None`` averages spectra over all probes.

        Returns:
            Array shaped ``(frequency,)`` in the same frequency order as
            :attr:`frequencies`.
        """
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
        """Integrate k-space at one frequency for each selected probe.

        Args:
            frequency: Requested signed frequency in THz. The nearest FFT bin
                is selected.
            probe_indices: Probe indices to include. Defaults to all probes.

        Returns:
            Flat array shaped ``(selected_probes,)``. Reshape it using the
            original ``probe_xs`` and ``probe_ys`` grid for a spectrum image.
        """
        b = self._backend
        freq_idx = int(np.argmin(np.abs(self.frequencies - frequency)))
        if probe_indices is None:
            probe_indices = list(range(len(self.probe_positions)))
        return np.array([to_numpy(b.sum(self._array[p, freq_idx, :, :])) for p in probe_indices])

    def nearest_frequency(self, frequency: float) -> float:
        """Return the represented FFT bin nearest ``frequency`` in THz."""
        index = int(np.argmin(np.abs(self.frequencies - frequency)))
        return float(self.frequencies[index])

    def spectrum_image_reshaped(self, frequency: float) -> np.ndarray:
        """Return a spectrum image shaped ``(probe_x, probe_y)``.

        The helper requires the original probes to form the complete Cartesian
        product of ``probe_xs`` and ``probe_ys``.
        """
        probe_xs = np.asarray(sorted(set(np.asarray(self.probe_positions)[:, 0])))
        probe_ys = np.asarray(sorted(set(np.asarray(self.probe_positions)[:, 1])))
        if len(probe_xs) * len(probe_ys) != len(self.probe_positions):
            raise ValueError("Probe positions do not form a complete Cartesian grid")
        expected = np.array([(x, y) for y in probe_ys for x in probe_xs])
        if not np.allclose(np.asarray(self.probe_positions), expected):
            raise ValueError("Probe positions are not in PySlice Cartesian-grid order")
        return self.spectrum_image(frequency).reshape(len(probe_ys), len(probe_xs)).T


    def diffraction(self, probe_index: Optional[int] = None,
                    space: str = "reciprocal") -> np.ndarray:
        """Sum over frequency to obtain a two-dimensional pattern.

        Args:
            probe_index: Probe to use. ``None`` averages patterns over all probes.
            space: ``"reciprocal"`` for a ``(kx, ky)`` pattern or ``"real"``
                for the magnitude of its inverse FFT.

        Returns:
            Two-dimensional NumPy array on the selected coordinate grid.
        """
        if space not in {"reciprocal", "real"}:
            raise ValueError("space must be 'reciprocal' or 'real'")
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
        """Return the two-dimensional pattern nearest a signed frequency.

        Args:
            frequency: Requested frequency in THz; the nearest FFT bin is used.
            probe_index: Probe to use. ``None`` averages over all probes.
            space: ``"reciprocal"`` or ``"real"`` as in :meth:`diffraction`.

        Returns:
            Two-dimensional NumPy array on the selected coordinate grid.
        """
        if space not in {"reciprocal", "real"}:
            raise ValueError("space must be 'reciprocal' or 'real'")
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
        """Integrate a reciprocal-space detector mask at every frequency.

        Args:
            mask: A ``(kx, ky)`` array, ``None`` for the full grid, or a round
                mask dictionary with ``shape``, ``center``, and ``radius``.
                Centers and radii are in inverse Angstroms.
            probe_index: Probe to use. ``None`` averages over all probes.
            preview: Display the first masked, frequency-integrated pattern.

        Returns:
            Array shaped ``(frequency,)``.
        """
        b = self._backend
        kxs_np = to_numpy(self._kxs)
        kys_np = to_numpy(self._kys)

        if mask is None:
            mask = np.ones((len(kxs_np), len(kys_np)))
        elif isinstance(mask, dict):
            cx, cy = mask.get("center", (0, 0))
            if mask.get("shape") != "round":
                raise ValueError("mask dictionary shape must be 'round'")
            r = float(mask["radius"])
            if r < 0:
                raise ValueError("mask radius must be nonnegative")
            radii = np.sqrt((kxs_np[:, None] - cx) ** 2 + (kys_np[None, :] - cy) ** 2)
            mask = (radii <= r).astype(float)
        elif mask.shape != (len(kxs_np), len(kys_np)):
            raise ValueError(f"Mask shape {mask.shape} doesn't match "
                             f"k-space shape ({len(kxs_np)}, {len(kys_np)})")

        if not isinstance(self._array, (np.ndarray, np.memmap)):
            mask = b.asarray(mask, dtype=self._array.dtype)

        if probe_index is not None and not (0 <= probe_index < len(self.probe_positions)):
            raise ValueError(f"Probe index {probe_index} out of range")
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
        """Sample TACAW magnitude along a requested two-dimensional path.

        Args:
            kx_path: Path x coordinates in inverse Angstroms, or Angstroms when
                ``space="real"``.
            ky_path: Path y coordinates with the same length and units as
                ``kx_path``.
            probe_index: Probe to use. ``None`` averages over all probes.
            space: ``"reciprocal"`` to sample k-space or ``"real"`` to sample
                inverse-FFT real space.

        Returns:
            Nonnegative array shaped ``(frequency, path_position)``.
        """
        if space not in {"reciprocal", "real"}:
            raise ValueError("space must be 'reciprocal' or 'real'")
        if len(kx_path) != len(ky_path):
            raise ValueError("kx_path and ky_path must have the same length")
        b = self._backend
        kx_np = to_numpy(self._kxs) if space != "real" else to_numpy(self._xs)
        ky_np = to_numpy(self._kys) if space != "real" else to_numpy(self._ys)

        if (
            np.any(np.asarray(kx_path) < kx_np.min())
            or np.any(np.asarray(kx_path) > kx_np.max())
            or np.any(np.asarray(ky_path) < ky_np.min())
            or np.any(np.asarray(ky_path) > ky_np.max())
        ):
            raise ValueError(
                "dispersion path extends outside the available coordinate grid"
            )

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
        """Plot a TACAW-derived heatmap.

        Args:
            intensities: Two-dimensional array to display.
            xvals: Coordinate array or one of ``"kx"``, ``"ky"``, ``"x"``,
                ``"y"``, ``"k"``, or ``"omega"``.
            yvals: Coordinate array or the same coordinate aliases as ``xvals``.
            xlabel: Label used when ``xvals`` is an explicit array.
            ylabel: Label used when ``yvals`` is an explicit array.
            filename: Save destination. If omitted, display the figure.
            title: Optional axes title.
            extent: Explicit Matplotlib image extent.
        """
        import matplotlib.pyplot as plt

        _AXIS_MAP = {
            "kx": ("kx (Å⁻¹)", lambda s: to_numpy(s._kxs)),
            "k":  ("kx (Å⁻¹)", lambda s: to_numpy(s._kxs)),
            "ky": ("ky (Å⁻¹)", lambda s: to_numpy(s._kys)),
            "x":  ("x (Å)",    lambda s: to_numpy(s._xs)),
            "y":  ("y (Å)",    lambda s: to_numpy(s._ys)),
            "omega": ("frequency (THz)", lambda s: s.frequencies),
        }

        x_alias = xvals if isinstance(xvals, str) else None
        y_alias = yvals if isinstance(yvals, str) else None
        if isinstance(xvals, str) and xvals in _AXIS_MAP:
            xlabel, xvals = _AXIS_MAP[xvals][0], _AXIS_MAP[xvals][1](self)
        if isinstance(yvals, str) and yvals in _AXIS_MAP:
            ylabel, yvals = _AXIS_MAP[yvals][0], _AXIS_MAP[yvals][1](self)

        xvals = np.asarray(xvals)
        yvals = np.asarray(yvals)
        aspect = "auto" if ylabel == "frequency (THz)" else None

        fig, ax = plt.subplots()
        values = to_numpy(np.abs(intensities))

        # TACAW reciprocal/real patterns are stored (x, y), whereas imshow
        # consumes (row=y, column=x). Dispersion is already (frequency, path).
        pattern_aliases = {("kx", "ky"), ("k", "ky"), ("x", "y")}
        if (x_alias, y_alias) in pattern_aliases:
            values = values.T
        elif values.shape == (len(xvals), len(yvals)) and values.shape != (len(yvals), len(xvals)):
            values = values.T
        elif values.shape != (len(yvals), len(xvals)):
            raise ValueError(
                "intensities must have shape (len(yvals), len(xvals)); "
                "PySlice (x, y) patterns are transposed automatically for named axes"
            )

        if extent is not None:
            xmin, xmax, ymin, ymax = extent
            xmask = (xvals >= xmin) & (xvals <= xmax)
            ymask = (yvals >= ymin) & (yvals <= ymax)
            if not np.any(xmask) or not np.any(ymask):
                raise ValueError("extent does not overlap the plotted coordinates")
            values = values[np.ix_(ymask, xmask)]
            xvals = xvals[xmask]
            yvals = yvals[ymask]

        actual_extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
        ax.imshow(values, cmap="inferno", extent=actual_extent, aspect=aspect,
                  origin="lower")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close(fig)
        return fig, ax


class SEDData(TACAWData):
    """
    SED (Spectral Energy Density) results.
    Functionally identical to TACAWData — both compute |Ψ(ω,q)|² via time-axis FFT.
    """
    def __init__(self, wf_data: WFData, layer_index: Optional[int] = None,
                 keep_complex: bool = False, force_rerun: bool = False) -> None:
        super().__init__(wf_data, layer_index, keep_complex, force_rerun=force_rerun)

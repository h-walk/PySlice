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
from ..data.seashell import adopt_signal_state
from pyslice.backend import Backend, NumpyBackend, to_numpy, source_files_version

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
    source = np.asarray(to_numpy(coordinates))
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{axis_name} coordinates must be a nonempty one-dimensional array")

    scale = max(1.0, float(np.max(np.abs(values))))
    # MPS frequency coordinates are float32. Casting to float64 cannot recover
    # precision lost when those bins were formed (notably the Nyquist spacing).
    eps = np.finfo(source.dtype).eps if np.issubdtype(source.dtype, np.floating) else 0.0
    rtol = max(1e-9, 4 * eps)
    atol = max(1e-10, 4 * eps) * scale
    indices = np.empty(len(values), dtype=np.int64)
    unmatched = []
    for index, value in enumerate(values):
        matches = np.flatnonzero(np.isclose(values, -value, rtol=rtol, atol=atol))
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
            and np.allclose(differences, differences[0], rtol=rtol, atol=atol)
            and np.any(np.isclose(values, 0.0, rtol=0.0, atol=atol))
            and np.isclose(
                abs(values[unmatched[0]]),
                values[-1] + differences[0],
                rtol=rtol,
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


def _chan_merge(n_a, s_a, m2_a, n_b, s_b, m2_b):
    """Merge two groups' (count, sum, sum of squared deviations from the mean).

    Chan et al.'s pairwise update: with means ``mu = s / n`` and
    ``delta = mu_b - mu_a``, the pooled sum of squared deviations is
    ``m2_a + m2_b + delta**2 * n_a * n_b / (n_a + n_b)``. Counts broadcast
    against the sums, so a per-probe count of shape ``(rows, 1, 1, 1)`` merges
    row-wise. A group with zero count contributes nothing. Pure NumPy, so the
    merge runs on the host in float64 whatever the backend.
    """
    n_a = np.asarray(n_a, dtype=np.float64)
    n_b = np.asarray(n_b, dtype=np.float64)
    n = n_a + n_b
    both = (n_a > 0) & (n_b > 0)
    safe_a = np.where(n_a > 0, n_a, 1.0)
    safe_b = np.where(n_b > 0, n_b, 1.0)
    safe_n = np.where(n > 0, n, 1.0)
    delta = s_b / safe_b - s_a / safe_a
    cross = np.where(both, delta ** 2 * (safe_a * safe_b / safe_n), 0.0)
    return n, s_a + s_b, m2_a + m2_b + cross


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

    # Class-level defaults: an object built without ``segment_std`` carries no
    # extra instance attributes.
    _want_segment_std = False
    _segment_m2 = None
    _segment_count = None

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
                 apply_bose: bool = False,
                 fold: bool = False,
                 fft_batch_max_bytes: Optional[int] = None,
                 segment_std: bool = False) -> None:
        """Transform time-domain exit waves into TACAW frequency data.

        Args:
            wf_data: Wavefunctions shaped ``(probe, time, kx, ky, layer)``.
            layer_index: Index within ``wf_data.layer`` to transform. Defaults
                to the last returned layer.
            keep_complex: Keep complex FFT amplitudes instead of converting to
                intensity ``abs(FFT)**2``.
            chunkFFT: Batch reciprocal x values to reduce peak FFT memory.
            chunk_size_time: Alias of ``segment_length``, kept for backward
                compatibility.
            segment_length: Samples per FFT segment. None uses the whole series
                as a single segment.
            overlap: Fraction in [0, 1) by which consecutive segments overlap;
                0 gives non-overlapping (Bartlett) segments, 0.5 Welch's method.
            window: Taper applied to each segment before the FFT: None or
                'boxcar' (rectangular, the default), 'hann', 'hamming',
                'blackman', 'bartlett', a length-L array, or a callable
                L -> array. Windows are RMS-normalised, so 'boxcar' reproduces
                the un-windowed result.
            force_rerun: Ignore a compatible ``tacaw.npy`` cache.
            temperature_K: Sample temperature used by the Bose correction.
            apply_bose: Apply signed Bose weighting after the FFT. Requires
                ``temperature_K`` and ``keep_complex=False``. Does not enable
                gain/loss folding by itself.
            fold: Average inversion-related gain/loss intensity pairs before
                any Bose weighting. Defaults to False and can be enabled
                independently of ``apply_bose``. Requires ``keep_complex=False``.
            fft_batch_max_bytes: Positive target for estimated spatial FFT
                temporaries, in bytes. Supplying it enables ``chunkFFT``.
                With ``chunkFFT=True``, None uses 16 MiB on GPUs and retains
                one-column batching on CPU. At least one kx
                column is processed; input/output storage and FFT-library
                workspace are additional. Does not change the time window.
            segment_std: Also compute the sample standard deviation of the
                segment periodograms (see :attr:`segment_std`). Off by default;
                when on, memory and, with a cache or memmap, disk use for the
                spectrum double (one extra array of the intensity's shape and
                dtype). Incompatible with ``keep_complex=True`` and with
                gain/loss folding (``fold=True``), which raise ``ValueError``.

        Notes:
            The spectrum is a periodogram estimate averaged over the segments
            (Welch's method). Averaging only makes sense on intensities, so
            keep_complex=True is rejected when more than one segment would be
            averaged.
            Frequencies are in THz. Negative bins represent gain and positive
            bins represent loss under PySlice's FFT convention.
        """

        if segment_std:
            if keep_complex:
                raise ValueError("segment_std requires intensity data; use keep_complex=False")
            if fold:
                raise ValueError(
                    "segment_std cannot be combined with gain/loss folding: the "
                    "standard deviation of a folded average needs the covariance "
                    "between each gain/loss pair, which is not kept")
            self._want_segment_std = True
            # The segment moments are not written to .sea files.
            config = dict(type(self)._sea_config)
            config['exclude_attrs'] = list(config['exclude_attrs']) + [
                '_segment_m2', '_segment_count']
            self._sea_config = config

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
        if fft_batch_max_bytes is not None and (
                isinstance(fft_batch_max_bytes, (bool, np.bool_))
                or not isinstance(fft_batch_max_bytes, (int, np.integer))
                or fft_batch_max_bytes < 1):
            raise ValueError("fft_batch_max_bytes must be a positive integer or None")
        self.chunkFFT = bool(chunkFFT or fft_batch_max_bytes is not None)
        self._fft_batch_budget_explicit = fft_batch_max_bytes is not None
        self.fft_batch_max_bytes = 16 * 1024**2 if fft_batch_max_bytes is None else int(fft_batch_max_bytes)
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
        requested_bose_correction = apply_bose
        self.apply_bose = False
        self.gain_loss_folded = False
        self.source_fingerprint = getattr(wf_data, "source_fingerprint", None)

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
        if requested_bose_correction:
            self.apply_bose_correction(self.temperature_K, fold=fold)
        elif fold:
            self.fold_gain_loss()

        self._apply_signal_dimensions()

    def _apply_signal_dimensions(self):
        """Populate the PySEA Signal dimensions/metadata from the current array."""
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
            ], nav_dimensions=[0, 1], det_dimensions=[2, 3])

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
        adopt_signal_state(self, 'TACAW')

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
        acc, m2, total_k = None, None, 0
        extra = {'segment_std': True} if self._want_segment_std else {}
        for i, wf in enumerate(wfs):
            self._check_ensemble_compatible(ref, wf, i)
            tac = TACAWData(wf, layer_index=layer_index,
                            segment_length=self.segment_length, overlap=self.overlap,
                            window=self.window, force_rerun=self.force_rerun, **extra)
            spec = to_numpy(tac._array)          # host; (n_scan, nfreq, nkx, nky), real
            k = int(tac.n_chunks)
            if self._want_segment_std:
                # Pool the segments of all trajectories (Chan merge, float64).
                m2_t = to_numpy(tac._segment_m2).astype(np.float64)
                if acc is None:
                    acc, m2 = spec * k, m2_t
                else:
                    _, acc, m2 = _chan_merge(total_k, acc, m2, k, spec * k, m2_t)
            else:
                acc = spec * k if acc is None else acc + spec * k   # weighted host sum
            total_k += k
            if self._frequencies is None:
                self._frequencies = tac._frequencies
        self.n_chunks = total_k
        self._array = b.asarray(acc / total_k)
        if self._want_segment_std:
            self._segment_m2 = b.asarray(m2, dtype=self._array.dtype)
            self._segment_count = np.full(self._array.shape[0], total_k, dtype=np.int64)

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
        self.gain_loss_folded = False; self.source_fingerprint = None; self.layer_index = None
        self._wf_array = None; self.fft_batch_max_bytes = 16 * 1024**2
        self._fft_batch_budget_explicit = False
        self.n_scan_positions = len(self.probe_positions); self.n_copies = 1
        self.n_chunks = int(meta.get('n_segments', 1))
        self._array = b.asarray(array)
        self._frequencies = b.asarray(frequencies)
        if meta.get('segment_m2') is not None:
            self._want_segment_std = True
            self._segment_m2 = b.asarray(meta['segment_m2'], dtype=self._array.dtype)
            self._segment_count = np.asarray(meta['segment_count'], dtype=np.int64)
            config = dict(type(self)._sea_config)
            config['exclude_attrs'] = list(config['exclude_attrs']) + [
                '_segment_m2', '_segment_count']
            self._sea_config = config
        self._apply_signal_dimensions()
        return self

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

    @property
    def segment_count(self):
        """Number of Welch segments behind each probe's spectrum, or None.

        An integer array shaped ``(probe,)``: the segments pooled into the
        averaged intensity of each probe row (all segments of all trajectories
        for an ensemble). Probe-batched accumulators can hold different counts
        per row. None unless the object was built with ``segment_std=True``.
        """
        return self._segment_count

    @property
    def segment_std(self):
        """Standard deviation of the segment periodograms, or None.

        Available when the object was built with ``segment_std=True`` (or
        ``TACAWAccumulator(segment_std=True)``); None otherwise. For every
        (probe, frequency, kx, ky) element ``i`` of :attr:`intensity` this is
        the sample standard deviation (ddof = 1)

            sqrt( sum_s (P_s[i] - mean_s P_s[i])**2 / (n - 1) ),

        over the ``n`` = :attr:`segment_count` segments ``s`` of that probe,
        where ``P_s`` is a segment's periodogram exactly as it enters the
        average: after mean subtraction, windowing and normalisation, summed
        over incoherent decoherence copies, before any Bose weighting. It is
        NaN where fewer than two segments exist. For an ensemble the
        segments of all trajectories are pooled. The array has the intensity's
        shape, backend and dtype; after :meth:`apply_bose_correction` (without
        folding) it is scaled by the same factor as the intensity.

        Caveats:
            * Segments overlapping by ``overlap`` are correlated, so the
              standard error of the mean intensity is not
              ``segment_std / sqrt(n)``: it needs the effective number of
              independent segments, which is smaller than ``n`` and depends on
              ``overlap``, the window and the signal. Segments from different
              trajectories are independent; segments within one trajectory are
              correlated through their shared samples and through the
              dynamics.
            * The deviation is per pixel. The standard deviation of a sum over
              pixels (a ring or aperture average, a k-integrated spectrum)
              needs the covariances between pixels, which are not kept, and
              cannot be formed from this array.
            * It is not defined after gain/loss folding, which therefore
              raises when segment statistics are present.
            * Memory (and with a cache or memmap, disk) use is twice that of
              the intensity alone. ``.sea`` files do not carry it.
        """
        m2 = self._segment_m2
        if m2 is None:
            return None
        b = self._backend
        n = np.asarray(self._segment_count, dtype=np.float64)
        denom = np.where(n >= 2, n - 1.0, np.nan).reshape((-1,) + (1,) * (m2.ndim - 1))
        denom = b.asarray(denom, dtype=m2.dtype) if not isinstance(m2, np.ndarray) \
            else denom.astype(m2.dtype, copy=False)
        return b.sqrt(b.xp.clip(m2, 0, None) / denom)

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
        if fold and self._segment_m2 is not None:
            self._raise_fold_with_segment_std()

        if fold:
            self.fold_gain_loss()
        b = self._backend
        factor = bose_correction_factor(self._frequencies, temperature_K)
        factor = (factor.astype(self._array.dtype, copy=False) if isinstance(self._array, np.ndarray)
                  else b.asarray(factor, dtype=self._array.dtype))
        self._array = self._array * factor[None, :, None, None]
        if self._segment_m2 is not None:
            # The weight is a per-frequency constant c, so the variance scales by c**2.
            self._segment_m2 = self._segment_m2 * (factor ** 2)[None, :, None, None]
        self.temperature_K = temperature_K
        self.apply_bose = True
        if hasattr(self, "metadata") and self.metadata is not None:
            self.metadata.Simulation.temperature_K = float(temperature_K)
            self.metadata.Simulation.gain_loss_folded = bool(self.gain_loss_folded)
            self.metadata.Simulation.bose_corrected = True
        return self

    @staticmethod
    def _raise_fold_with_segment_std():
        raise ValueError(
            "gain/loss folding is not available with segment_std: the standard "
            "deviation of a gain/loss average needs the covariance between the "
            "two partners' segment periodograms, which is not kept. Build the "
            "object without segment_std to fold")

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
        if self._segment_m2 is not None:
            self._raise_fold_with_segment_std()

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
        if not isinstance(self._array, np.ndarray):
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
        want_std = self._want_segment_std
        cache_m2 = None if cache_dir is None else cache_dir / "tacaw_segment_m2.npy"

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
        if (not self.force_rerun and cache_dir is not None
                and cache_tacaw.exists() and cache_meta.exists()
                and cache_freq.exists() and (not want_std or cache_m2.exists())):
            try:
                with open(cache_meta) as f:
                    cached_meta = json.load(f)
            except (OSError, ValueError):
                cached_meta = None
            if cached_meta == meta:
                cached = np.load(cache_tacaw, mmap_mode='r' if self.use_memmap else None)
                if list(cached.shape) == meta["array_shape"]:
                    self._frequencies = b.asarray(np.load(cache_freq))
                    self._array = cached if self.use_memmap else b.asarray(
                        cached, dtype=b.complex_dtype if self.keep_complex else b.float_dtype)
                    if want_std:
                        cached_m2 = np.load(cache_m2, mmap_mode='r' if self.use_memmap else None)
                        if list(cached_m2.shape) == meta["array_shape"]:
                            self._segment_m2 = cached_m2 if self.use_memmap else b.asarray(
                                cached_m2, dtype=b.float_dtype)
                            self._segment_count = np.full(
                                self._array.shape[0], self.n_chunks, dtype=np.int64)
                            return
                    else:
                        return

        # A (re)compute invalidates any previous completion marker first, so an
        # interrupted run (partial tacaw.npy — notably the memmap accumulator)
        # is never mistaken for a complete cache on the next load.
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if cache_meta.exists():
                cache_meta.unlink()

        wf_layer = self._wf_array[:, :, :, :, layer_index]  # p,t,kx,ky

        dt = float(time_steps[0])
        self._frequencies = b.fftshift(b.fftfreq(fft_len, d=dt))
        window = self._make_window(self.window, L, b)   # RMS-normalised, length L
        n_segments = len(starts)

        def _segment_periodogram(seg):
            # host (memmap) slices go to the backend first: the window lives there
            if isinstance(seg, np.ndarray) and not isinstance(b, NumpyBackend):
                seg = b.asarray(seg, dtype=b.complex_dtype)
            # detrend (remove elastic/DC line) -> window -> FFT along time
            seg = seg - b.mean(seg, axis=1, keepdims=True)
            seg = seg * window.reshape((1, L) + (1,) * (seg.ndim - 2))
            out = b.fftshift(b.fft(seg, axes=1), axes=1)
            if not self.keep_complex:
                out = self._fold_incoherent_copies(b.absolute(out) ** 2)
            return out

        # Segment variance (opt-in): the sum of squared deviations M2 of the
        # periodograms from their running mean, built with Welford's update
        #     M2_k = M2_{k-1} + (P_k - mean_{k-1}) * (P_k - mean_k),
        # where mean_{k-1} and mean_k come from the running sum that already
        # forms the Welch average, so M2 is the only extra array. Accumulating
        # sum(P) and sum(P**2) and forming (S2 - S1**2 / n) / (n - 1) would
        # cancel catastrophically wherever the segment-to-segment spread is
        # small against the mean (strong, steady Bragg or low-q elements), and
        # in float32 backends already at moderate ratios. Across trajectories
        # and partials the host merges M2 with Chan's pairwise formula in
        # float64 (see _chan_merge), which keeps the same property.
        m2 = None

        # Welch: average the (windowed, detrended) segment periodograms.
        if self.chunkFFT:
            # Spatial batching preserves the full selected time window and
            # copy-major intensity folding. Estimate four complex work arrays;
            # a single column remains the minimum indivisible batch.
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
            if want_std:
                if self.use_memmap:
                    m2 = b.memmap(shape, dtype=dtype, filename=cache_m2)
                else:
                    m2 = b.zeros(shape, dtype=dtype)

            complex_bytes = to_numpy(b.zeros(0, dtype=b.complex_dtype)).dtype.itemsize
            column_bytes = 4 * int(wf_layer.shape[0]) * fft_len * int(wf_layer.shape[3]) * complex_bytes
            width = max(1, min(len(self._kxs), max(1, self.fft_batch_max_bytes // max(1, column_bytes))))
            if str(b.device) == 'cpu' and not self._fft_batch_budget_explicit:
                width = 1
            self.fft_batch_kx = width

            for k, start in enumerate(starts, 1):
                for kx_i in tqdm(range(0, len(self._kxs), width)):
                    kx_end = min(kx_i + width, len(self._kxs))
                    contrib = _segment_periodogram(wf_layer[:, start:start + L, kx_i:kx_end, :])
                    if self.use_memmap:
                        contrib = to_numpy(contrib)
                    if want_std and k > 1:
                        prev_mean = self._array[:, :, kx_i:kx_end, :] / (k - 1)
                    self._array[:, :, kx_i:kx_end, :] += contrib
                    if want_std and k > 1:
                        m2[:, :, kx_i:kx_end, :] += (contrib - prev_mean) * (
                            contrib - self._array[:, :, kx_i:kx_end, :] / k)
            self._array /= n_segments
        else:
            # Standard path: FFT over the full segment window
            for k, start in enumerate(starts, 1):
                contrib = _segment_periodogram(wf_layer[:, start:start + L, :, :])
                if want_std:
                    if k == 1:
                        m2 = b.zeros_like(contrib)
                    else:
                        prev_mean = self._array / (k - 1)
                self._array = contrib if self._array is None else self._array + contrib
                if want_std and k > 1:
                    m2 = m2 + (contrib - prev_mean) * (contrib - self._array / k)
            self._array = self._array / n_segments
        if want_std:
            self._segment_m2 = m2
            self._segment_count = np.full(self._array.shape[0], n_segments, dtype=np.int64)

        # Completion marker is written last, after the entire array is flushed.
        if cache_dir is not None:
            np.save(cache_freq, to_numpy(self._frequencies))
            if isinstance(self._array, np.memmap):
                self._array.flush()
            else:
                np.save(cache_tacaw, to_numpy(self._array))
            if want_std:
                if isinstance(m2, np.memmap):
                    m2.flush()
                else:
                    np.save(cache_m2, to_numpy(m2))
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
        meta = {
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
        if self._want_segment_std:
            meta["segment_std"] = True
        return meta

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
            if self._want_segment_std:
                self._segment_m2 = None
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

    With ``segment_std=True`` the accumulator also pools the segment
    periodograms of every unit added, per probe row, and ``finalize()`` returns
    a TACAWData whose :attr:`TACAWData.segment_std` is their sample standard
    deviation (ddof = 1) and whose :attr:`TACAWData.segment_count` holds the
    segment count per probe row. The definition and its caveats (correlated
    overlapping segments, per-pixel values, no folding) are those of
    :attr:`TACAWData.segment_std`. The accumulator then holds one extra host
    array of the accumulator's shape and dtype (a second file next to
    ``memmap_path``, named ``<stem>_segment_m2.npy``), so memory and disk use
    double. The units are merged with Chan's pairwise formula on the sums of
    squared deviations, in the accumulator's dtype (float64 by default).
    """

    def __init__(self, *, segment_length=None, overlap=0.0, window=None,
                 layer_index=None, n_probes=None, dtype=np.float64, memmap_path=None,
                 segment_std=False):
        self._kw = dict(segment_length=segment_length, overlap=overlap,
                        window=window, layer_index=layer_index)
        self._segment_std = bool(segment_std)
        if self._segment_std:
            self._kw['segment_std'] = True
        self._m2 = None
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
            if self._segment_std:
                if self.memmap_path is not None:
                    m2_path = str(Path(self.memmap_path).with_suffix('')) + "_segment_m2.npy"
                    self._m2 = np.lib.format.open_memmap(
                        m2_path, mode='w+', dtype=self.dtype, shape=shape)
                    self._m2[:] = 0
                else:
                    self._m2 = np.zeros(shape, dtype=self.dtype)
            self._freqs = to_numpy(tac._frequencies)
            self._meta = {a: getattr(tac, a) for a in
                          ('probe_positions', '_kxs', '_kys', '_xs', '_ys',
                           '_layer', '_time', 'probe', 'cache_dir', '_backend')}
        idx = slice(None) if rows is None else np.asarray(rows)
        if self._segment_std:
            _, _, m2 = _chan_merge(
                np.asarray(self._count[idx], dtype=np.float64)[:, None, None, None],
                np.asarray(self._acc[idx], dtype=np.float64),
                np.asarray(self._m2[idx], dtype=np.float64),
                k, spec * k, to_numpy(tac._segment_m2).astype(np.float64))
            self._m2[idx] = m2
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
        if self._segment_std:
            meta['segment_m2'] = np.asarray(self._m2)
            meta['segment_count'] = np.rint(np.asarray(self._count)).astype(np.int64)
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
            **({'segment_m2': np.asarray(self._m2)} if self._segment_std else {}),
        )
        return str(path)


def reduce_tacaw_partials(partials, backend=None, segment_std=False) -> "TACAWData":
    """Sum file-based TACAWAccumulator partials into an ensemble-averaged TACAWData.

    partials: a directory (globs ``partial_*.npz``) or an explicit list of .npz
    paths written by :meth:`TACAWAccumulator.save_partial`. This is the cross-rank
    (and cross-node) reduce: it sums the per-rank un-averaged periodogram sums and
    counts, then divides. Missing ranks simply do not contribute (fault tolerant).

    With ``segment_std=True`` the partials must have been written by a
    ``TACAWAccumulator(segment_std=True)``; their sums of squared deviations are
    merged with Chan's pairwise formula in float64 and the result carries
    :attr:`TACAWData.segment_std` and :attr:`TACAWData.segment_count`, pooled
    over every segment of every partial (see :attr:`TACAWData.segment_std` for
    the definition and caveats). A partial without the second moment raises
    ``ValueError``. With the default False the second moment is ignored even if
    the partials carry it.
    """
    from pyslice.backend import NumpyBackend
    if isinstance(partials, (str, Path)):
        paths = sorted(glob.glob(str(Path(partials) / "partial_*.npz")))
    else:
        paths = [str(p) for p in partials]
    if not paths:
        raise FileNotFoundError(f"no partial_*.npz found in {partials!r}")

    acc = count = ref = m2 = None
    for p in paths:
        d = np.load(p, allow_pickle=False)
        if segment_std:
            if 'segment_m2' not in d.files:
                raise ValueError(
                    f"{p} has no segment second moment; write the partials with "
                    "TACAWAccumulator(segment_std=True)")
            m2_p = d['segment_m2'].astype(np.float64)
            if acc is None:
                m2 = m2_p
            else:
                _, _, m2 = _chan_merge(
                    count[:, None, None, None], acc, m2,
                    d['count'].astype(np.float64)[:, None, None, None],
                    d['acc'].astype(np.float64), m2_p)
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
    if segment_std:
        meta['segment_m2'] = m2
        meta['segment_count'] = np.rint(count).astype(np.int64)
    return TACAWData._from_spectrum(avg, b.asarray(ref['freqs']), meta)

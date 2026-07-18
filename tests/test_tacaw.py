"""TACAW spectral regressions: caching, folding, Welch, ensemble."""
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


def test_tacaw_nondivisible_segment_drops_partial_tail(tmp_path):
    # Welch segmentation no longer requires the segment length to divide the
    # series evenly; the trailing partial segment is dropped (2 segments of 4
    # from 10 samples), and the frequency axis follows the segment length.
    wf = _make_wf_data(tmp_path, n_time=10)
    tac = TACAWData(wf, segment_length=4, force_rerun=True)
    assert tac.n_chunks == 2
    assert len(tac.frequencies) == 4
    # chunk_size_time stays a working alias for segment_length
    tac2 = TACAWData(wf, chunk_size_time=4, force_rerun=True)
    assert tac2.n_chunks == 2

def test_tacaw_real_space_dispersion_uses_inverse_fft(tmp_path):
    tacaw = TACAWData(_make_wf_data(tmp_path, n_time=4), force_rerun=True)
    reciprocal = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    tacaw._array = reciprocal[None, None, :, :]
    tacaw._frequencies = np.array([0.0])

    dispersion = tacaw.dispersion(
        np.array([tacaw.xs[0]]),
        np.array([tacaw.ys[0]]),
        probe_index=0,
        space="real",
    )

    assert dispersion[0, 0] == pytest.approx(abs(np.fft.ifft2(reciprocal)[0, 0]))

def test_tacaw_complex_spectral_diffraction_preserves_imaginary_parts(tmp_path):
    tacaw = TACAWData(_make_wf_data(tmp_path, n_time=4), keep_complex=True, force_rerun=True)
    first = np.array([[1.0 + 2.0j, 0.0], [0.0, 0.0]], dtype=np.complex128)
    second = np.array([[3.0 + 4.0j, 0.0], [0.0, 0.0]], dtype=np.complex128)
    tacaw._array = np.stack([first, second])[:, None, :, :]
    tacaw._frequencies = np.array([0.0])
    tacaw.probe_positions = [(0.0, 0.0), (1.0, 1.0)]

    pattern = tacaw.spectral_diffraction(0.0, space="real")

    expected = np.abs(np.fft.ifft2(np.mean([first, second], axis=0)))
    np.testing.assert_allclose(pattern, expected)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_masked_spectrum_moves_generated_mask_to_torch_backend(tmp_path, monkeypatch):
    backend = TorchBackend(device="cpu")
    tacaw = TACAWData(_make_wf_data(tmp_path, n_time=4, backend=backend), force_rerun=True)

    original_asarray = backend.asarray
    mask_dtypes = []

    def spy_asarray(arraylike, dtype=None, device=None):
        if np.shape(arraylike) == (2, 2):
            mask_dtypes.append(dtype)
        return original_asarray(arraylike, dtype=dtype, device=device)

    monkeypatch.setattr(backend, "asarray", spy_asarray)

    spectrum = tacaw.masked_spectrum(mask={"shape": "round", "radius": 10.0})

    assert mask_dtypes == [tacaw._array.dtype]
    assert spectrum.shape == (4,)

def test_tacaw_cache_distinguishes_layer_dtype_and_dataset(tmp_path):
    # The tacaw.npy cache used a shape-only check, so a different layer / dtype /
    # dataset in the same cache_dir was silently served the first spectrum.
    wf = _make_layered_wf(tmp_path, seed=1)
    s0 = to_numpy(TACAWData(wf, layer_index=0)._array).copy()
    s1 = to_numpy(TACAWData(wf, layer_index=1)._array)   # same dir, other layer
    assert not np.allclose(s0, s1)
    assert np.allclose(s0, to_numpy(TACAWData(wf, layer_index=0)._array))  # hit is stable
    sc = to_numpy(TACAWData(wf, layer_index=0, keep_complex=True)._array)
    assert sc.dtype.kind == "c"                          # complex, not cached real intensity
    wf2 = _make_layered_wf(tmp_path, seed=2)             # different data, same cache_dir
    assert not np.allclose(s0, to_numpy(TACAWData(wf2, layer_index=0)._array))

def test_tacaw_cache_distinguishes_timestep_and_hashes_every_value(tmp_path):
    wf = _make_layered_wf(tmp_path, seed=1)
    first = TACAWData(wf, layer_index=0)
    frequencies = first.frequencies.copy()

    # The FFT values alone do not identify their frequency axis: identical
    # samples at a different timestep must not inherit the old frequencies.
    wf._time = wf._time * 2
    second = TACAWData(wf, layer_index=0)
    np.testing.assert_allclose(second.frequencies, frequencies / 2)

    # A single changed value must never hide between stride-sampled elements.
    large = np.zeros((1 << 20) + 3, dtype=np.complex128)
    fingerprint = TACAWData._array_fingerprint(large)
    large[(1 << 19) + 1] = 1
    assert TACAWData._array_fingerprint(large) != fingerprint

@pytest.mark.parametrize("chunk_fft", [False, True])
def test_tacaw_incoherently_folds_decoherence_copies(tmp_path, chunk_fft):
    backend = NumpyBackend()
    n_copies, n_scan, nt = 3, 2, 8
    time = np.arange(nt, dtype=float) * 0.1
    rows = []
    for copy in range(n_copies):
        for scan in range(n_scan):
            amplitude = 1.0 + copy + scan / 2
            rows.append(amplitude * np.cos(2 * np.pi * (copy + 1) * time))
    waves = np.asarray(rows, dtype=np.complex128)[:, :, None, None, None]
    probe = SimpleNamespace(
        eV=60e3, wavelength=0.05, mrad=5.0,
        _array=np.zeros((n_copies, n_scan, 1, 1), dtype=np.complex128))
    wf = WFData(
        probe_positions=[(0.0, 0.0), (1.0, 0.0)],
        probe_xs=[0.0, 1.0], probe_ys=[0.0], time=time,
        kxs=np.array([0.0]), kys=np.array([0.0]),
        xs=np.array([0.0]), ys=np.array([0.0]), layer=np.array([0]),
        array=waves, probe=probe, backend=backend,
        cache_dir=tmp_path / str(chunk_fft))

    raw = waves[:, :, :, :, 0]
    raw_fft = np.fft.fftshift(
        np.fft.fft(raw - raw.mean(axis=1, keepdims=True), axis=1), axes=1)
    expected = (np.abs(raw_fft) ** 2).reshape(
        n_copies, n_scan, nt, 1, 1).sum(axis=0)

    tacaw = TACAWData(wf, chunkFFT=chunk_fft, force_rerun=True)
    assert tacaw.array.shape == (n_scan, nt, 1, 1)
    np.testing.assert_allclose(tacaw.array, expected)
    np.testing.assert_allclose(TACAWData(wf, chunkFFT=chunk_fft).array, expected)

    with pytest.raises(ValueError, match="keep_complex"):
        TACAWData(wf, keep_complex=True, force_rerun=True)

@pytest.mark.parametrize("chunk_fft", [False, True])
def test_tacaw_memmap_writes_a_complete_reusable_cache(tmp_path, chunk_fft):
    backend = NumpyBackend()
    nt = 8
    cache_dir = tmp_path / str(chunk_fft)
    cache_dir.mkdir()
    source = backend.memmap(
        (2, nt, 1, 1, 1), dtype=np.complex128,
        filename=cache_dir / "wavefunctions.npy")
    time = np.arange(nt, dtype=float) * 0.1
    source[0, :, 0, 0, 0] = np.cos(2 * np.pi * time)
    source[1, :, 0, 0, 0] = 2 * np.cos(4 * np.pi * time)
    source.flush()
    probe = SimpleNamespace(
        eV=60e3, wavelength=0.05, mrad=5.0,
        _array=np.zeros((2, 1, 1, 1), dtype=np.complex128))
    wf = WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
        time=time, kxs=np.array([0.0]), kys=np.array([0.0]),
        xs=np.array([0.0]), ys=np.array([0.0]), layer=np.array([0]),
        array=source, probe=probe, backend=backend, cache_dir=cache_dir)

    first = TACAWData(wf, chunkFFT=chunk_fft, force_rerun=True).array.copy()
    assert (cache_dir / "tacaw.npy").exists()
    assert (cache_dir / "tacaw_meta.json").exists()
    second = TACAWData(wf, chunkFFT=chunk_fft).array
    np.testing.assert_allclose(second, first)

def test_tacaw_welch_windowing(tmp_path):
    n, dt = 64, 0.1
    t = np.arange(n) * dt
    f0 = 8 / (n * dt)                    # a tone on a frequency bin (1.25 THz)
    sig = np.cos(2 * np.pi * f0 * t) + 0.5   # +0.5 DC that detrend must remove

    peak = lambda tac: abs(tac.frequencies[np.argmax(to_numpy(tac.spectrum()))])

    # 1. default == the old un-windowed |FFT(detrend)|^2 (boxcar is RMS-identity)
    tac = TACAWData(_tone_wf(tmp_path / "a", sig))
    manual = np.abs(np.fft.fftshift(np.fft.fft(sig - sig.mean()))) ** 2 * 4  # *4: summed 2x2 k
    np.testing.assert_allclose(to_numpy(tac.spectrum()), manual, rtol=1e-6, atol=1e-6)
    assert np.isclose(peak(tac), f0)
    dc = to_numpy(tac.spectrum())[np.argmin(np.abs(tac.frequencies))]
    assert np.isclose(dc, 0.0, atol=1e-6)                      # DC/elastic removed

    # 2. windowing keeps the peak position; frequency axis follows the segment
    assert np.isclose(peak(TACAWData(_tone_wf(tmp_path / "b", sig), window="hann")), f0)
    welch = TACAWData(_tone_wf(tmp_path / "c", sig), segment_length=32, overlap=0.5, window="hann")
    assert np.isclose(peak(welch), f0)
    assert welch.n_chunks == 3 and len(welch.frequencies) == 32

    # 3. Welch averaging lowers the off-peak variance on noise
    rng = np.random.RandomState(0)
    noise = rng.randn(n) + 1j * rng.randn(n)
    s_one = to_numpy(TACAWData(_tone_wf(tmp_path / "d", noise)).spectrum())
    s_welch = to_numpy(TACAWData(_tone_wf(tmp_path / "e", noise),
                                 segment_length=16, overlap=0.5, window="hann").spectrum())
    assert np.std(s_welch) < np.std(s_one)

    # 4. complex spectra cannot be averaged over segments
    with pytest.raises(ValueError, match="keep_complex"):
        TACAWData(_tone_wf(tmp_path / "f", sig), segment_length=32, keep_complex=True)

def test_tacaw_ensemble_average_and_accumulator(tmp_path):
    from pyslice.postprocessing.tacaw_data import TACAWAccumulator
    rng = np.random.RandomState(0)
    n = 64
    A = rng.randn(2, n) + 1j * rng.randn(2, n)   # trajectory A, 2 probes
    B = rng.randn(2, n) + 1j * rng.randn(2, n)   # trajectory B
    wfA = lambda sub=None: _multiprobe_wf(tmp_path / "A", A if sub is None else A[sub])
    wfB = lambda: _multiprobe_wf(tmp_path / "B", B)

    single = to_numpy(TACAWData(wfA())._array)
    # averaging identical trajectories returns the single-trajectory spectrum
    np.testing.assert_allclose(to_numpy(TACAWData([wfA(), wfA()])._array), single, atol=1e-10)
    # averaging distinct trajectories is the mean of their spectra
    specB = to_numpy(TACAWData(wfB())._array)
    ens = to_numpy(TACAWData([wfA(), wfB()])._array)
    np.testing.assert_allclose(ens, 0.5 * (single + specB), atol=1e-10)

    # the streaming accumulator matches the list constructor
    acc = TACAWAccumulator()
    acc.add(wfA()); acc.add(wfB())
    np.testing.assert_allclose(to_numpy(acc.finalize()._array), ens, atol=1e-10)

    # probe batching: two 1-probe partials reconstruct the full 2-probe spectrum
    acc2 = TACAWAccumulator(n_probes=2)
    acc2.add(_multiprobe_wf(tmp_path / "p0", A[[0]]), rows=[0])
    acc2.add(_multiprobe_wf(tmp_path / "p1", A[[1]]), rows=[1])
    np.testing.assert_allclose(to_numpy(acc2.finalize()._array), single, atol=1e-10)

    # incompatible trajectories are rejected
    bad = wfB(); bad._kxs = np.arange(3.0)
    with pytest.raises(ValueError, match="k-grid differs"):
        TACAWData([wfA(), bad])

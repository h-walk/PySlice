"""Standard deviation of the TACAW spectrum across Welch segments (opt-in).

The reference is built in the tests with plain NumPy from the definition of the
segment periodogram P_s: segment s of length L starting at s * step, with
step = L - round(overlap * L); remove the segment's time mean; multiply by the
RMS-normalised window; |fftshift(fft(., axis=time))|**2. ``np.std(ddof=1)`` of
the list of P_s is the expected ``segment_std``.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
if TORCH_AVAILABLE:
    from pyslice.backend import TorchBackend
from pyslice.postprocessing.tacaw_data import (
    TACAWAccumulator, TACAWData, bose_correction_factor, reduce_tacaw_partials)
from pyslice.postprocessing.wf_data import WFData

L, OVERLAP = 8, 0.5
SHAPE = (2, 3, 4)   # probes, kx, ky: all different from each other and from L


def _backends():
    params = [pytest.param("numpy", id="numpy")]
    if TORCH_AVAILABLE:
        params.append(pytest.param("torch", id="torch-cpu", marks=pytest.mark.torch))
    return params


def _make_backend(name):
    return NumpyBackend() if name == "numpy" else TorchBackend(device="cpu")


def _random_series(seed, n_time, shape=SHAPE):
    rng = np.random.default_rng(seed)
    p, nkx, nky = shape
    return (rng.normal(size=(p, n_time, nkx, nky))
            + 1j * rng.normal(size=(p, n_time, nkx, nky))) + 0.3


def _wf(tmp_path, series, backend=None, array=None):
    """WFData holding ``series`` shaped (probe, time, kx, ky)."""
    backend = NumpyBackend() if backend is None else backend
    p, nt, nkx, nky = series.shape
    probe = SimpleNamespace(eV=60e3, wavelength=0.05, mrad=5.0,
                            _array=np.zeros((1, 1, nkx, nky), dtype=np.complex128))
    if array is None:
        array = backend.asarray(series[..., None], dtype=backend.complex_dtype)
    return WFData(
        probe_positions=[(float(i), 0.0) for i in range(p)],
        probe_xs=list(range(p)), probe_ys=[0.0],
        time=np.arange(nt, dtype=float) * 0.005,
        kxs=np.linspace(-1.0, 1.0, nkx), kys=np.linspace(-1.5, 1.5, nky),
        xs=np.arange(nkx, dtype=float), ys=np.arange(nky, dtype=float),
        layer=np.array([0]), array=array, probe=probe, backend=backend,
        cache_dir=tmp_path)


def _periodograms(series, seg_len=L, overlap=OVERLAP, window="hann"):
    """Per-segment periodograms (n_seg, probe, freq, kx, ky), plain NumPy."""
    nt = series.shape[1]
    step = seg_len - int(round(overlap * seg_len))
    w = np.hanning(seg_len) if window == "hann" else np.ones(seg_len)
    w = w / np.sqrt(np.mean(w ** 2))
    out = []
    for start in range(0, nt - seg_len + 1, step):
        seg = series[:, start:start + seg_len]
        seg = seg - seg.mean(axis=1, keepdims=True)
        seg = seg * w[None, :, None, None]
        out.append(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2)
    return np.array(out)


def _kw(**extra):
    return dict(segment_length=L, overlap=OVERLAP, window="hann", **extra)


# ---------------------------------------------------------------------------
# 1. one trajectory against np.std over independently built periodograms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_name", _backends())
def test_segment_std_matches_numpy_std_of_segment_periodograms(tmp_path, backend_name):
    series = _random_series(0, 40)
    periodograms = _periodograms(series)
    assert periodograms.shape[0] == 9
    tac = TACAWData(_wf(tmp_path, series, _make_backend(backend_name)),
                    segment_std=True, **_kw())

    np.testing.assert_allclose(to_numpy(tac.intensity), periodograms.mean(axis=0), rtol=1e-12)
    std = to_numpy(tac.segment_std)
    assert std.shape == tac.array.shape == (2, L, 3, 4)
    np.testing.assert_allclose(std, np.std(periodograms, axis=0, ddof=1), rtol=1e-10)
    np.testing.assert_array_equal(tac.segment_count, [9, 9])
    assert type(tac.segment_std) is type(tac.intensity)
    assert tac.segment_std.dtype == tac.intensity.dtype


def test_segment_std_sums_decoherence_copies_before_the_variance(tmp_path):
    # P_s enters the average after the incoherent sum over copies.
    series = _random_series(1, 40, shape=(4, 3, 4))          # 2 copies x 2 scan positions
    wf = _wf(tmp_path, series)
    wf.probe_positions = wf.probe_positions[:2]
    periodograms = _periodograms(series)
    folded = periodograms.reshape(periodograms.shape[0], 2, 2, *periodograms.shape[2:]).sum(axis=1)
    tac = TACAWData(wf, segment_std=True, **_kw())
    np.testing.assert_allclose(to_numpy(tac.segment_std), np.std(folded, axis=0, ddof=1),
                               rtol=1e-10)


def test_segment_std_is_stable_when_the_spread_is_tiny_against_the_mean(tmp_path):
    # A steady complex tone with a weak noise floor: the relative spread of the
    # tone bin is ~1e-7, so sum(P**2) - sum(P)**2 / n would keep no digits.
    rng = np.random.default_rng(3)
    nt = 40
    t = np.arange(nt)
    series = np.empty((2, nt, 3, 4), dtype=np.complex128)
    for p in range(2):
        tone = 1e3 * np.exp(2j * np.pi * (1.0 / L) * t)
        noise = 1e-4 * (rng.normal(size=(nt, 3, 4)) + 1j * rng.normal(size=(nt, 3, 4)))
        series[p] = tone[:, None, None] + noise
    periodograms = _periodograms(series)
    expected = np.std(periodograms, axis=0, ddof=1)

    n = periodograms.shape[0]
    naive = np.sqrt(np.clip((periodograms ** 2).sum(0) - periodograms.sum(0) ** 2 / n, 0, None)
                    / (n - 1))
    peak = np.unravel_index(np.argmax(periodograms.mean(0)), expected.shape)
    assert abs(naive[peak] / expected[peak] - 1) > 1e-2     # the stable form is needed here

    tac = TACAWData(_wf(tmp_path, series), segment_std=True, **_kw())
    got = to_numpy(tac.segment_std)
    np.testing.assert_allclose(got[peak], expected[peak], rtol=1e-6)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. accumulator and list constructor pool the segments of all trajectories
# ---------------------------------------------------------------------------

def _trajectories():
    # different lengths -> 9, 5 and 7 segments
    return [_random_series(10 + i, nt) for i, nt in enumerate((40, 24, 32))]


def test_accumulator_pools_segments_of_trajectories_with_different_counts(tmp_path):
    trajs = _trajectories()
    pooled = np.concatenate([_periodograms(s) for s in trajs], axis=0)
    assert pooled.shape[0] == 9 + 5 + 7
    expected_std = np.std(pooled, axis=0, ddof=1)

    acc = TACAWAccumulator(segment_std=True, **_kw())
    for i, s in enumerate(trajs):
        acc.add(_wf(tmp_path / f"a{i}", s))
    tac = acc.finalize()
    np.testing.assert_allclose(tac.array, pooled.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(to_numpy(tac.segment_std), expected_std, rtol=1e-10)
    np.testing.assert_array_equal(tac.segment_count, [21, 21])


@pytest.mark.parametrize("memmap", [False, True])
def test_finalized_result_is_independent_of_later_adds(tmp_path, memmap):
    a, b = _random_series(5, 40), _random_series(6, 32)
    acc = TACAWAccumulator(segment_std=True,
                           memmap_path=tmp_path / "acc.npy" if memmap else None, **_kw())
    acc.add(_wf(tmp_path / "a", a))
    first = acc.finalize()
    std, intensity = to_numpy(first.segment_std).copy(), first.array.copy()
    acc.add(_wf(tmp_path / "b", b))
    np.testing.assert_array_equal(to_numpy(first.segment_std), std)
    np.testing.assert_array_equal(first.array, intensity)
    np.testing.assert_allclose(std, np.std(_periodograms(a), axis=0, ddof=1), rtol=1e-10)
    both = np.concatenate([_periodograms(a), _periodograms(b)], axis=0)
    np.testing.assert_allclose(to_numpy(acc.finalize().segment_std),
                               np.std(both, axis=0, ddof=1), rtol=1e-10)


def test_list_constructor_pools_segments_of_trajectories(tmp_path):
    # the list constructor requires a common time axis, so the counts are equal here
    trajs = [_random_series(15 + i, 40) for i in range(3)]
    pooled = np.concatenate([_periodograms(s) for s in trajs], axis=0)
    tac = TACAWData([_wf(tmp_path / f"l{i}", s) for i, s in enumerate(trajs)],
                    segment_std=True, **_kw())
    np.testing.assert_allclose(to_numpy(tac.segment_std), np.std(pooled, axis=0, ddof=1),
                               rtol=1e-10)
    np.testing.assert_array_equal(tac.segment_count, [27, 27])
    np.testing.assert_allclose(tac.array, pooled.mean(axis=0), rtol=1e-12)
    assert getattr(TACAWData([_wf(tmp_path / "p", trajs[0]), _wf(tmp_path / "q", trajs[1])],
                             **_kw()), "segment_std", None) is None


@pytest.mark.parametrize("chunk_fft", [False, True])
def test_list_constructor_pools_memmapped_trajectories_on_a_cache_hit(tmp_path, chunk_fft):
    # The second call reads every trajectory's spectrum and moment from read-only caches.
    trajs = [_random_series(15 + i, 40) for i in range(3)]
    pooled = np.concatenate([_periodograms(s) for s in trajs], axis=0)

    def wfs():
        out = []
        for i, s in enumerate(trajs):
            d = tmp_path / f"{chunk_fft}_{i}"
            d.mkdir(exist_ok=True)
            source = NumpyBackend().memmap(s.shape + (1,), dtype=np.complex128,
                                           filename=d / "wavefunctions.npy")
            source[..., 0] = s
            source.flush()
            out.append(_wf(d, s, array=source))
        return out

    for attempt in ("first", "cache hit"):
        tac = TACAWData(wfs(), segment_std=True, chunkFFT=chunk_fft, **_kw())
        np.testing.assert_allclose(to_numpy(tac.segment_std), np.std(pooled, axis=0, ddof=1),
                                   rtol=1e-10, err_msg=attempt)
        np.testing.assert_array_equal(tac.segment_count, [27, 27])
        np.testing.assert_allclose(tac.array, pooled.mean(axis=0), rtol=1e-12)
    assert (tmp_path / f"{chunk_fft}_0" / "tacaw_segment_m2.npy").exists()


def test_accumulator_probe_batches_keep_a_count_per_row(tmp_path):
    a, b, c = (_random_series(20 + i, nt) for i, nt in enumerate((40, 24, 32)))
    acc = TACAWAccumulator(segment_std=True, n_probes=2, **_kw())
    # probe 0 sees trajectories a, b, c; probe 1 sees only b and c
    acc.add(_wf(tmp_path / "a0", a[[0]]), rows=[0])
    acc.add(_wf(tmp_path / "b0", b[[0]]), rows=[0])
    acc.add(_wf(tmp_path / "b1", b[[1]]), rows=[1])
    acc.add(_wf(tmp_path / "c01", c), rows=[0, 1])
    tac = acc.finalize()

    p0 = np.concatenate([_periodograms(x[[0]]) for x in (a, b, c)], axis=0)[:, 0]
    p1 = np.concatenate([_periodograms(x[[1]]) for x in (b, c)], axis=0)[:, 0]
    np.testing.assert_array_equal(tac.segment_count, [21, 12])
    std = to_numpy(tac.segment_std)
    np.testing.assert_allclose(std[0], np.std(p0, axis=0, ddof=1), rtol=1e-10)
    np.testing.assert_allclose(std[1], np.std(p1, axis=0, ddof=1), rtol=1e-10)


@pytest.mark.parametrize("backend_name", _backends())
def test_accumulator_on_a_backend_returns_backend_arrays(tmp_path, backend_name):
    backend = _make_backend(backend_name)
    trajs = _trajectories()[:2]
    acc = TACAWAccumulator(segment_std=True, **_kw())
    for i, s in enumerate(trajs):
        acc.add(_wf(tmp_path / f"a{i}", s, backend))
    tac = acc.finalize()
    assert type(tac.segment_std) is type(tac.intensity)
    pooled = np.concatenate([_periodograms(s) for s in trajs], axis=0)
    np.testing.assert_allclose(to_numpy(tac.segment_std), np.std(pooled, axis=0, ddof=1),
                               rtol=1e-10)


def test_accumulator_memmap_holds_the_second_moment_on_disk(tmp_path):
    trajs = _trajectories()
    pooled = np.concatenate([_periodograms(s) for s in trajs], axis=0)
    acc = TACAWAccumulator(segment_std=True, memmap_path=tmp_path / "acc.npy", **_kw())
    for i, s in enumerate(trajs):
        acc.add(_wf(tmp_path / f"m{i}", s))
    assert isinstance(acc._m2, np.memmap)
    assert (tmp_path / "acc_segment_m2.npy").exists()
    tac = acc.finalize()
    np.testing.assert_allclose(to_numpy(tac.segment_std), np.std(pooled, axis=0, ddof=1),
                               rtol=1e-10)

    plain = TACAWAccumulator(memmap_path=tmp_path / "plain.npy", **_kw())
    plain.add(_wf(tmp_path / "plain", trajs[0]))
    assert plain._m2 is None
    assert not (tmp_path / "plain_segment_m2.npy").exists()


# ---------------------------------------------------------------------------
# 3. partial files and the cross-rank reduce
# ---------------------------------------------------------------------------

def test_partials_reduce_equals_direct_accumulator(tmp_path):
    trajs = _trajectories() + [_random_series(99, 24)]
    pooled = np.concatenate([_periodograms(s) for s in trajs], axis=0)

    direct = TACAWAccumulator(segment_std=True, **_kw())
    for i, s in enumerate(trajs):
        direct.add(_wf(tmp_path / f"d{i}", s))
    direct = direct.finalize()

    out = tmp_path / "partials"
    out.mkdir()
    for rank, chunk in enumerate([trajs[:1], trajs[1:3], trajs[3:]]):    # uneven ranks
        acc = TACAWAccumulator(segment_std=True, **_kw())
        for i, s in enumerate(chunk):
            acc.add(_wf(tmp_path / f"r{rank}_{i}", s))
        acc.save_partial(out / f"partial_{rank:04d}.npz")
    reduced = reduce_tacaw_partials(out, segment_std=True)

    np.testing.assert_allclose(to_numpy(reduced.segment_std), to_numpy(direct.segment_std),
                               rtol=1e-12)
    np.testing.assert_allclose(to_numpy(reduced.segment_std), np.std(pooled, axis=0, ddof=1),
                               rtol=1e-10)
    np.testing.assert_array_equal(reduced.segment_count, direct.segment_count)
    np.testing.assert_allclose(reduced.array, pooled.mean(axis=0), rtol=1e-12)

    # without the flag the reduce ignores the stored moment and exposes none
    plain = reduce_tacaw_partials(out)
    assert plain.segment_std is None and plain.segment_count is None
    np.testing.assert_array_equal(plain.array, reduced.array)


def test_reduce_with_probe_batched_partials(tmp_path):
    a, b = _random_series(30, 40), _random_series(31, 32)
    out = tmp_path / "partials"
    out.mkdir()
    acc0 = TACAWAccumulator(segment_std=True, n_probes=2, **_kw())
    acc0.add(_wf(tmp_path / "a0", a[[0]]), rows=[0])
    acc0.add(_wf(tmp_path / "b1", b[[1]]), rows=[1])
    acc1 = TACAWAccumulator(segment_std=True, n_probes=2, **_kw())
    acc1.add(_wf(tmp_path / "b0", b[[0]]), rows=[0])
    acc0.save_partial(out / "partial_0000.npz")
    acc1.save_partial(out / "partial_0001.npz")
    reduced = reduce_tacaw_partials(out, segment_std=True)
    p0 = np.concatenate([_periodograms(a[[0]]), _periodograms(b[[0]])], axis=0)[:, 0]
    p1 = _periodograms(b[[1]])[:, 0]
    np.testing.assert_array_equal(reduced.segment_count, [9 + 7, 7])
    std = to_numpy(reduced.segment_std)
    np.testing.assert_allclose(std[0], np.std(p0, axis=0, ddof=1), rtol=1e-10)
    np.testing.assert_allclose(std[1], np.std(p1, axis=0, ddof=1), rtol=1e-10)


def test_reduce_requires_the_second_moment_in_every_partial(tmp_path):
    s = _random_series(40, 40)
    out = tmp_path / "partials"
    out.mkdir()
    acc = TACAWAccumulator(**_kw())
    acc.add(_wf(tmp_path / "x", s))
    acc.save_partial(out / "partial_0000.npz")
    with pytest.raises(ValueError, match="segment second moment"):
        reduce_tacaw_partials(out, segment_std=True)


def test_run_tacaw_ensemble_passes_segment_std_through(tmp_path):
    from pyslice.multislice.distributed import run_tacaw_ensemble
    trajs = _trajectories()
    producers = [(lambda s=s, i=i: _wf(tmp_path / f"p{i}", s)) for i, s in enumerate(trajs)]
    out = tmp_path / "out"
    for rank in range(2):
        run_tacaw_ensemble(producers, out, rank=rank, world=2, segment_std=True, **_kw())
    reduced = reduce_tacaw_partials(out, segment_std=True)
    pooled = np.concatenate([_periodograms(s) for s in trajs], axis=0)
    np.testing.assert_allclose(to_numpy(reduced.segment_std), np.std(pooled, axis=0, ddof=1),
                               rtol=1e-10)
    final = run_tacaw_ensemble(producers, tmp_path / "out2", rank=0, world=1,
                               segment_std=True, reduce=True, **_kw())
    np.testing.assert_allclose(to_numpy(final.segment_std), np.std(pooled, axis=0, ddof=1),
                               rtol=1e-10)

    # default: partials carry no second moment
    run_tacaw_ensemble(producers, tmp_path / "out3", rank=0, world=1, **_kw())
    with np.load(tmp_path / "out3" / "partial_0000.npz") as d:
        assert "segment_m2" not in d.files


# ---------------------------------------------------------------------------
# 4. batching, memmap and cache
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_name", _backends())
def test_chunked_fft_gives_the_same_segment_std(tmp_path, backend_name):
    series = _random_series(50, 40)
    backend = _make_backend(backend_name)
    base = TACAWData(_wf(tmp_path / "a", series, backend), segment_std=True, **_kw())
    # one 4096-byte column per kx; a budget of 8192 -> 2 columns per batch (batches of 2, 1)
    chunked = TACAWData(_wf(tmp_path / "b", series, backend), segment_std=True,
                        fft_batch_max_bytes=8192, **_kw())
    assert chunked.chunkFFT and chunked.fft_batch_kx == 2
    one_column = TACAWData(_wf(tmp_path / "c", series, backend), segment_std=True,
                           chunkFFT=True, **_kw())
    for other in (chunked, one_column):
        np.testing.assert_allclose(to_numpy(other.segment_std), to_numpy(base.segment_std),
                                   rtol=1e-12)
        np.testing.assert_allclose(to_numpy(other.intensity), to_numpy(base.intensity),
                                   rtol=1e-12)
    np.testing.assert_allclose(to_numpy(base.segment_std),
                               np.std(_periodograms(series), axis=0, ddof=1), rtol=1e-10)


@pytest.mark.parametrize("backend_name", _backends())
@pytest.mark.parametrize("chunk_fft", [False, True])
def test_memmap_path_gives_the_same_segment_std_and_reuses_the_cache(
        tmp_path, chunk_fft, backend_name):
    series = _random_series(60, 40)
    expected = np.std(_periodograms(series), axis=0, ddof=1)
    cache_dir = tmp_path / str(chunk_fft)
    cache_dir.mkdir()
    source = NumpyBackend().memmap(series.shape + (1,), dtype=np.complex128,
                                   filename=cache_dir / "wavefunctions.npy")
    source[..., 0] = series
    source.flush()
    wf = _wf(cache_dir, series, _make_backend(backend_name), array=source)

    first = TACAWData(wf, segment_std=True, chunkFFT=chunk_fft, force_rerun=True, **_kw())
    np.testing.assert_allclose(to_numpy(first.segment_std), expected, rtol=1e-10)
    assert (cache_dir / "tacaw_segment_m2.npy").exists()
    assert first.chunkFFT == chunk_fft and first.use_memmap

    second = TACAWData(wf, segment_std=True, chunkFFT=chunk_fft, **_kw())   # cache hit
    np.testing.assert_allclose(to_numpy(second.segment_std), expected, rtol=1e-10)
    np.testing.assert_array_equal(second.segment_count, first.segment_count)
    np.testing.assert_allclose(to_numpy(second.intensity), to_numpy(first.intensity))


def test_cache_with_segment_std_is_separate_from_the_default_cache(tmp_path):
    series = _random_series(70, 40)
    wf = _wf(tmp_path, series)
    expected = np.std(_periodograms(series), axis=0, ddof=1)

    plain = TACAWData(wf, **_kw())
    assert (tmp_path / "tacaw.npy").exists()
    assert not (tmp_path / "tacaw_segment_m2.npy").exists()

    with_std = TACAWData(wf, segment_std=True, **_kw())    # the plain cache must not serve it
    np.testing.assert_allclose(to_numpy(with_std.segment_std), expected, rtol=1e-10)
    reused = TACAWData(wf, segment_std=True, **_kw())      # now a hit, with the moment
    np.testing.assert_allclose(to_numpy(reused.segment_std), expected, rtol=1e-10)
    np.testing.assert_array_equal(reused.intensity, plain.intensity)

    # a plain request after a segment_std cache recomputes without the moment
    again = TACAWData(wf, **_kw())
    assert again.segment_std is None
    np.testing.assert_array_equal(again.intensity, plain.intensity)

    # a cache of identical meta but without the moment file is not a hit for segment_std
    TACAWData(wf, segment_std=True, **_kw())
    (tmp_path / "tacaw_segment_m2.npy").unlink()
    rebuilt = TACAWData(wf, segment_std=True, **_kw())
    assert (tmp_path / "tacaw_segment_m2.npy").exists()
    np.testing.assert_allclose(to_numpy(rebuilt.segment_std), expected, rtol=1e-10)


def test_cache_hit_with_a_wrong_shaped_moment_file_recomputes_cleanly(tmp_path):
    series = _random_series(71, 40)
    wf = _wf(tmp_path, series)
    TACAWData(wf, segment_std=True, **_kw())
    np.save(tmp_path / "tacaw_segment_m2.npy", np.zeros((1, 2, 3)))
    tac = TACAWData(wf, segment_std=True, **_kw())
    periodograms = _periodograms(series)
    np.testing.assert_allclose(tac.array, periodograms.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(to_numpy(tac.segment_std), np.std(periodograms, axis=0, ddof=1),
                               rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. default off: nothing changes
# ---------------------------------------------------------------------------

def test_default_off_leaves_attributes_cache_and_partials_unchanged(tmp_path):
    # Holds on the base commit as well: nothing about the default path moves.
    series = _random_series(80, 40)
    off = TACAWData(_wf(tmp_path / "off", series), **_kw())

    assert getattr(off, "segment_std", None) is None
    assert getattr(off, "segment_count", None) is None
    for name in ("_segment_m2", "_segment_count", "_want_segment_std", "_sea_config"):
        assert name not in vars(off)

    # cache meta and files: exactly the pre-existing set
    meta_off = json.loads((tmp_path / "off" / "tacaw_manifest.json").read_text())
    assert "segment_std" not in meta_off
    assert sorted(os.listdir(tmp_path / "off")) == [
        "tacaw.npy", "tacaw_freq.npy", "tacaw_manifest.json"]
    assert off._tacaw_cache_meta(0, L) == meta_off

    acc = TACAWAccumulator(**_kw())
    acc.add(_wf(tmp_path / "acc", series))
    assert getattr(acc, "_m2", None) is None
    acc.save_partial(tmp_path / "partial.npz")
    with np.load(tmp_path / "partial.npz") as d:
        assert sorted(d.files) == sorted([
            "acc", "count", "freqs", "kxs", "kys", "xs", "ys", "layer", "time",
            "probe_positions", "probe_eV", "probe_wavelength", "probe_mrad",
            "segment_length", "overlap", "window"])
    fin = acc.finalize()
    assert getattr(fin, "segment_std", None) is None and "_segment_m2" not in vars(fin)
    reduced = reduce_tacaw_partials([tmp_path / "partial.npz"])
    assert getattr(reduced, "segment_std", None) is None
    np.testing.assert_array_equal(reduced.array, fin.array)


def test_segment_std_on_changes_nothing_but_adds_the_moment(tmp_path):
    series = _random_series(80, 40)
    on = TACAWData(_wf(tmp_path / "on", series), segment_std=True, **_kw())
    off = TACAWData(_wf(tmp_path / "off", series), **_kw())
    np.testing.assert_array_equal(off.intensity, on.intensity)
    assert off.n_chunks == on.n_chunks
    meta_off = json.loads((tmp_path / "off" / "tacaw_manifest.json").read_text())
    assert on._tacaw_cache_meta(0, L) == dict(meta_off, segment_std=True)
    assert on._sea_config is not TACAWData._sea_config
    assert {"_segment_m2", "_segment_count"} <= set(on._sea_config["exclude_attrs"])
    assert "_segment_m2" not in TACAWData._sea_config["exclude_attrs"]


def test_segment_std_flag_validation(tmp_path):
    wf = _wf(tmp_path, _random_series(81, 40))
    with pytest.raises(ValueError, match="keep_complex"):
        TACAWData(wf, segment_std=True, keep_complex=True)


# ---------------------------------------------------------------------------
# 6. fewer than two segments
# ---------------------------------------------------------------------------

def test_fewer_than_two_segments_gives_nan(tmp_path):
    series = _random_series(90, 40)
    tac = TACAWData(_wf(tmp_path, series), segment_std=True)      # one segment, full series
    std = to_numpy(tac.segment_std)
    assert tac.segment_count.tolist() == [1, 1]
    assert std.shape == tac.array.shape
    assert np.all(np.isnan(std))
    assert np.all(np.isfinite(tac.array))

    two = TACAWData(_wf(tmp_path / "two", series), segment_length=20, segment_std=True)
    assert two.segment_count.tolist() == [2, 2]
    np.testing.assert_allclose(to_numpy(two.segment_std),
                               np.std(_periodograms(series, 20, 0.0, "boxcar"), axis=0, ddof=1),
                               rtol=1e-10)


def test_rows_with_fewer_than_two_segments_are_nan_in_the_accumulator(tmp_path):
    long_, short = _random_series(91, 40), _random_series(92, 8)
    acc = TACAWAccumulator(segment_std=True, n_probes=2, **_kw())
    acc.add(_wf(tmp_path / "l", long_[[0]]), rows=[0])
    acc.add(_wf(tmp_path / "s", short[[1]]), rows=[1])
    tac = acc.finalize()
    np.testing.assert_array_equal(tac.segment_count, [9, 1])
    std = to_numpy(tac.segment_std)
    assert np.all(np.isfinite(std[0])) and np.all(np.isnan(std[1]))


# ---------------------------------------------------------------------------
# 7. Bose weighting and folding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_name", _backends())
def test_bose_correction_scales_segment_std_like_the_intensity(tmp_path, backend_name):
    series = _random_series(100, 40)
    backend = _make_backend(backend_name)
    plain = TACAWData(_wf(tmp_path / "a", series, backend), segment_std=True, **_kw())
    raw_std = to_numpy(plain.segment_std).copy()
    raw_int = to_numpy(plain.intensity).copy()

    via_ctor = TACAWData(_wf(tmp_path / "b", series, backend), segment_std=True,
                         temperature_K=300.0, apply_bose=True, **_kw())
    plain.apply_bose_correction(300.0)
    factor = bose_correction_factor(plain.frequencies, 300.0)[None, :, None, None]
    assert not np.allclose(factor, 1.0)
    for tac in (via_ctor, plain):
        np.testing.assert_allclose(to_numpy(tac.intensity), raw_int * factor, rtol=1e-12)
        np.testing.assert_allclose(to_numpy(tac.segment_std), raw_std * factor, rtol=1e-10)
        np.testing.assert_allclose(to_numpy(tac.segment_std),
                                   np.std(_periodograms(series), axis=0, ddof=1) * factor,
                                   rtol=1e-10)


def test_bose_correction_on_a_finalized_accumulator_scales_segment_std(tmp_path):
    trajs = _trajectories()[:2]
    acc = TACAWAccumulator(segment_std=True, **_kw())
    for i, s in enumerate(trajs):
        acc.add(_wf(tmp_path / f"a{i}", s))
    tac = acc.finalize()
    raw = to_numpy(tac.segment_std).copy()
    tac.apply_bose_correction(300.0)
    factor = bose_correction_factor(tac.frequencies, 300.0)[None, :, None, None]
    np.testing.assert_allclose(to_numpy(tac.segment_std), raw * factor, rtol=1e-10)


def test_folding_with_segment_std_is_rejected_and_leaves_the_object_unchanged(tmp_path):
    series = _random_series(110, 40)
    # fold needs a q <-> -q closed grid: use symmetric k axes
    wf = _wf(tmp_path / "w", series[:, :, :3, :3])
    wf._kxs = np.array([-1.0, 0.0, 1.0])
    wf._kys = np.array([-1.0, 0.0, 1.0])

    for kwargs in (dict(fold=True), dict(fold=True, apply_bose=True, temperature_K=300.0)):
        with pytest.raises(ValueError, match="folding"):
            TACAWData(wf, segment_std=True, **_kw(), **kwargs)

    tac = TACAWData(wf, segment_std=True, **_kw())
    intensity, std = to_numpy(tac.intensity).copy(), to_numpy(tac.segment_std).copy()
    with pytest.raises(ValueError, match="folding"):
        tac.fold_gain_loss()
    with pytest.raises(ValueError, match="folding"):
        tac.apply_bose_correction(300.0, fold=True)
    assert not tac.apply_bose and not tac.gain_loss_folded
    np.testing.assert_array_equal(to_numpy(tac.intensity), intensity)
    np.testing.assert_array_equal(to_numpy(tac.segment_std), std)

    # the same grid folds normally without segment_std
    folded = TACAWData(wf, fold=True, **_kw())
    assert folded.gain_loss_folded and folded.segment_std is None

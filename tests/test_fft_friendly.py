"""FFT-friendly lateral grid (``fft_friendly``): the smooth-size helper, the grid,
the calculator's k labels, masks, crop window and cache key, and the physics.

The cells are non-square (lx != ly, so nx != ny) to catch x/y mix-ups. The main
cell, 7.3 x 6.1 A at sampling 0.1, has the default grid 74 x 61 (61 is prime) and
the smooth grid 75 x 63, so both axes change.
"""
import numpy as np
import pytest

from pyslice.backend import to_numpy
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.multislice.multislice import wavelength
from pyslice.multislice.potentials import grid_from_trajectory, next_fast_len
from pyslice.multislice.trajectory import Trajectory


def _is_smooth(n, primes=(2, 3, 5, 7)):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def _cell(lx=7.3, ly=6.1, lz=4.0, n_atoms=12, seed=1, positions=None):
    """One-frame trajectory of C and Si atoms in an lx x ly x lz box."""
    if positions is None:
        rng = np.random.RandomState(seed)
        positions = rng.rand(n_atoms, 3) * np.array([lx, ly, lz])
    positions = np.asarray(positions, dtype=float)[None]
    n = positions.shape[1]
    return Trajectory(
        atom_types=np.array([6] * (n - n // 3) + [14] * (n // 3)),
        positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([lx, ly, lz]),
        timestep=0.1,
    )


def _setup(traj, **kwargs):
    kwargs = dict(dict(aperture=20, voltage_eV=100e3, sampling=0.1, slice_thickness=1.0,
                       cache_wavefunctions=False), **kwargs)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(traj, **kwargs)
    return calc


def _k_index(labels, spacing):
    """Integer multiples m of the k pixel spacing, from k labels."""
    m = to_numpy(labels) / spacing
    np.testing.assert_allclose(m, np.round(m), atol=1e-9)
    return np.round(m).astype(int)


# ---------------------------------------------------------------------------
# next_fast_len
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, expected", [
    (1138, 1152), (1034, 1050), (903, 945), (860, 864),
    (1279, 1280), (960, 960), (1, 1),
])
def test_next_fast_len_values(n, expected):
    assert next_fast_len(n) == expected


def test_next_fast_len_is_smallest_smooth_at_least_n():
    smooth = [m for m in range(1, 2200) if _is_smooth(m)]
    for n in range(1, 2000):
        assert next_fast_len(n) == min(m for m in smooth if m >= n)


@pytest.mark.parametrize("n, primes, expected", [
    (1034, (2, 3, 5), 1080), (903, (2, 3, 5), 960), (1279, (2,), 2048),
    (7, (2,), 8), (10, (3, 7), 21),
])
def test_next_fast_len_custom_primes(n, primes, expected):
    assert next_fast_len(n, primes) == expected


@pytest.mark.parametrize("n", [0, -1, -1138])
def test_next_fast_len_rejects_n_below_one(n):
    with pytest.raises(ValueError):
        next_fast_len(n)


def test_next_fast_len_rejects_bad_arguments():
    with pytest.raises(TypeError):
        next_fast_len(1138.0)
    with pytest.raises(ValueError):
        next_fast_len(10, (1, 2))
    with pytest.raises(ValueError):
        next_fast_len(10, ())


# ---------------------------------------------------------------------------
# grid_from_trajectory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lx, ly", [(56.86, 47.2), (51.66, 45.13), (63.94, 75.81), (7.3, 6.1)])
@pytest.mark.parametrize("sampling", [0.05, 0.1, 0.0731])
def test_grid_default_matches_formula(lx, ly, sampling):
    traj = _cell(lx, ly, 10.0, positions=[[0.0, 0.0, 0.0]])
    xs, ys, zs, *lengths = grid_from_trajectory(traj, sampling=sampling, slice_thickness=0.2)
    for got, length, spacing in ((xs, lx, sampling), (ys, ly, sampling), (zs, 10.0, 0.2)):
        expected = np.linspace(0, length, int(length / spacing) + 1, endpoint=False)
        assert got.dtype == expected.dtype and np.array_equal(got, expected)
    assert lengths == [lx, ly, 10.0]


@pytest.mark.parametrize("lx, ly", [(56.86, 47.2), (51.66, 45.13), (63.94, 75.81), (7.3, 6.1)])
@pytest.mark.parametrize("sampling", [0.05, 0.1, 0.0731])
def test_grid_fft_friendly_is_smooth_and_no_coarser(lx, ly, sampling):
    traj = _cell(lx, ly, 10.0, positions=[[0.0, 0.0, 0.0]])
    xs0, ys0, zs0, *_ = grid_from_trajectory(traj, sampling=sampling, slice_thickness=0.2)
    xs, ys, zs, *_ = grid_from_trajectory(traj, sampling=sampling, slice_thickness=0.2,
                                          fft_friendly=True)
    for got, default, length in ((xs, xs0, lx), (ys, ys0, ly)):
        n = len(got)
        assert _is_smooth(n)
        assert n == next_fast_len(len(default))
        assert length / n <= sampling
        np.testing.assert_array_equal(got, np.linspace(0, length, n, endpoint=False))
        assert got[0] == 0 and got[-1] < length
    np.testing.assert_array_equal(zs, zs0)            # slices are never rounded


@pytest.mark.parametrize("flag", [None, False, np.False_])
def test_grid_fft_friendly_off_values(flag):
    traj = _cell()
    default = grid_from_trajectory(traj, sampling=0.1)
    got = grid_from_trajectory(traj, sampling=0.1, fft_friendly=flag)
    for a, b in zip(got, default):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("flag", ["yes", 1.5, 1, 0, (2, 3, 5)])
def test_grid_fft_friendly_rejects_non_bool(flag):
    with pytest.raises(TypeError, match="fft_friendly"):
        grid_from_trajectory(_cell(), sampling=0.1, fft_friendly=flag)


# ---------------------------------------------------------------------------
# MultisliceCalculator.setup: argument handling and the default mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["yes", 1.5, 1, 0])
def test_setup_fft_friendly_rejects_non_bool(tmp_path, monkeypatch, flag):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError, match="fft_friendly"):
        _setup(_cell(), fft_friendly=flag)


@pytest.mark.parametrize("flag, expected", [(None, False), (False, False), (True, True),
                                            (np.True_, True)])
def test_setup_fft_friendly_accepts_bools_and_none(tmp_path, monkeypatch, flag, expected):
    monkeypatch.chdir(tmp_path)
    calc = _setup(_cell(), fft_friendly=flag)
    assert calc.fft_friendly is expected
    assert len(calc.xs) == (75 if expected else 74)


def test_setup_fft_friendly_with_prism_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError, match="prism"):
        _setup(_cell(), prism=5, fft_friendly=True)


def test_setup_default_grid_labels_masks_and_cache_key_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traj = _cell()
    kw = dict(sampling=0.1, max_kx=2.0, max_ky=1.5, probe_positions=[(3.0, 2.5)])
    calc = _setup(traj, **kw)
    # Default grid n = int(L / sampling) + 1, labelled from the realized spacing L / n.
    assert (len(calc.xs), len(calc.ys)) == (74, 61)
    for got, n, length, kmax, keep in ((calc.kxs, 74, 7.3, 2.0, calc.keep_kxs_indices),
                                       (calc.kys, 61, 6.1, 1.5, calc.keep_kys_indices)):
        labels = np.fft.fftshift(np.fft.fftfreq(n, length / n))
        np.testing.assert_allclose(to_numpy(got), labels, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(to_numpy(keep), np.flatnonzero(np.abs(labels) <= kmax))

    key_args = (traj, calc.aperture, calc.voltage_eV, calc.slice_thickness, calc.sampling,
                calc.probe_positions, calc.base_probe.spatial_decoherence,
                calc.base_probe.temporal_decoherence, calc.base_probe._array,
                calc._cache_key_stored_layers(calc._stored_layers))
    key_kwargs = dict(output_options=calc._cache_output_options(), skip_vacuum=calc.skip_vacuum)
    assert calc.cache_key == calc._generate_cache_key(*key_args, **key_kwargs)  # no fft_friendly entry
    assert _setup(traj, fft_friendly=False, **kw).cache_key == calc.cache_key
    assert _setup(traj, fft_friendly=None, **kw).cache_key == calc.cache_key
    assert _setup(traj, fft_friendly=True, **kw).cache_key != calc.cache_key


def test_cache_key_records_fft_friendly_on_an_already_smooth_grid(tmp_path, monkeypatch):
    """A 7.4 x 6.3 A cell at sampling 0.1 has the smooth default grid 75 x 63, so
    both modes build the same grid, probe and k labels. The key still separates
    them: 'fft_friendly' is its only entry that differs."""
    monkeypatch.chdir(tmp_path)
    traj = _cell(7.4, 6.3)
    default = _setup(traj)
    ff = _setup(traj, fft_friendly=True)
    assert (len(ff.xs), len(ff.ys)) == (len(default.xs), len(default.ys)) == (75, 63)
    np.testing.assert_array_equal(to_numpy(ff.base_probe._array),
                                  to_numpy(default.base_probe._array))
    np.testing.assert_array_equal(to_numpy(ff.kxs), to_numpy(default.kxs))
    np.testing.assert_array_equal(to_numpy(ff.kys), to_numpy(default.kys))
    assert ff.cache_key != default.cache_key


# ---------------------------------------------------------------------------
# MultisliceCalculator.setup with fft_friendly=True: grid, labels and masks
# ---------------------------------------------------------------------------

def test_setup_fft_friendly_labels_k_with_actual_spacing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = _setup(_cell(), fft_friendly=True, max_kx=2.0, max_ky=1.5)
    assert (len(calc.xs), len(calc.ys)) == (75, 63)
    for labels, keep, n, length, kmax in ((calc.kxs, calc.keep_kxs_indices, 75, 7.3, 2.0),
                                          (calc.kys, calc.keep_kys_indices, 63, 6.1, 1.5)):
        labels = to_numpy(labels)
        np.testing.assert_allclose(np.diff(labels), 1 / length, rtol=1e-12)
        m = _k_index(labels, 1 / length)
        np.testing.assert_array_equal(m, np.arange(-(n // 2), n - n // 2))
        # The mask keeps exactly the pixels whose actual |k| = |m|/L is within kmax.
        np.testing.assert_array_equal(to_numpy(keep), np.flatnonzero(np.abs(m) / length <= kmax))


def test_fft_friendly_equals_default_mode_on_the_same_grid(tmp_path, monkeypatch):
    """The smooth 75 x 63 grid for sampling 0.1 is also the default grid for sampling
    0.098, so grid, k labels and exit waves must all be identical."""
    monkeypatch.chdir(tmp_path)
    traj = _cell()
    ff = _setup(traj, sampling=0.1, fft_friendly=True)
    ref = _setup(traj, sampling=0.098)
    np.testing.assert_array_equal(ff.xs, ref.xs)
    np.testing.assert_array_equal(ff.ys, ref.ys)
    wf_ff = ff.run(force_rerun=True)
    wf_ref = ref.run(force_rerun=True)
    np.testing.assert_array_equal(to_numpy(wf_ff.array), to_numpy(wf_ref.array))
    np.testing.assert_array_equal(wf_ff.kxs, wf_ref.kxs)
    np.testing.assert_array_equal(wf_ff.kys, wf_ref.kys)
    # The output records the actual grid.
    np.testing.assert_allclose(np.diff(wf_ff.xs), 7.3 / 75, rtol=1e-12)
    np.testing.assert_allclose(np.diff(wf_ff.ys), 6.1 / 63, rtol=1e-12)


# ---------------------------------------------------------------------------
# Physics: default grid against the smooth grid at the same requested sampling
# ---------------------------------------------------------------------------

def _aperture_pixels(calc):
    """k pixels strictly inside the hard aperture |k| < alpha / lambda."""
    kx = np.fft.fftfreq(len(calc.xs), float(calc.dx))
    ky = np.fft.fftfreq(len(calc.ys), float(calc.dy))
    radius = calc.aperture * 1e-3 / float(wavelength(calc.voltage_eV))
    return int(np.sum(np.hypot(kx[:, None], ky[None, :]) < radius))


def test_single_slice_total_intensity_is_aperture_pixel_count_on_both_grids(tmp_path, monkeypatch):
    """One slice is a pure phase and has no propagation step, so the exit wave
    keeps the probe's norm exactly: sum |Psi(k)|^2 = number of aperture pixels,
    which depends only on the k spacing 1/L and is the same on both grids."""
    monkeypatch.chdir(tmp_path)
    traj = _cell(lz=0.9)
    totals, pixels = [], []
    for flag in (False, True):
        calc = _setup(traj, fft_friendly=flag)
        assert len(calc.zs) == 1
        wf = calc.run(force_rerun=True)
        totals.append(np.sum(np.abs(to_numpy(wf.array)) ** 2))
        pixels.append(_aperture_pixels(calc))
    assert pixels[0] == pixels[1] == 41
    np.testing.assert_allclose(totals, 41, rtol=1e-12)


def test_single_slice_plane_wave_total_intensity_scales_with_pixel_count(tmp_path, monkeypatch):
    """A plane wave (aperture 0) has amplitude 1 on each of the N = nx * ny pixels,
    so sum |Psi(k)|^2 = N**2: the rounding changes it from (74 * 61)**2 to (75 * 63)**2."""
    monkeypatch.chdir(tmp_path)
    traj = _cell(lz=0.9)
    for flag, n_pixels in ((False, 74 * 61), (True, 75 * 63)):
        calc = _setup(traj, fft_friendly=flag, aperture=0)
        assert len(calc.xs) * len(calc.ys) == n_pixels
        total = np.sum(np.abs(to_numpy(calc.run(force_rerun=True).array)) ** 2)
        np.testing.assert_allclose(total, n_pixels ** 2, rtol=1e-12)


def test_exit_wave_intensities_agree_between_default_and_smooth_grids(tmp_path, monkeypatch):
    """Both grids share the k pixels m/L, so the cropped intensities are compared
    pixel by pixel. They differ only through the real-space sampling (dx 0.0986 ->
    0.0973 A, dy 0.1000 -> 0.0968 A): the largest difference is 0.45 % of the peak
    intensity for this 5-slice cell."""
    monkeypatch.chdir(tmp_path)
    traj = _cell()
    out, sizes = {}, []
    for flag in (False, True):
        calc = _setup(traj, fft_friendly=flag, max_kx=2.0, max_ky=2.0)
        sizes.append((len(calc.xs), len(calc.ys)))
        wf = calc.run(force_rerun=True)
        # Both modes label k in multiples of 1/L.
        mx = _k_index(wf.kxs, 1 / calc.lx)
        my = _k_index(wf.kys, 1 / calc.ly)
        out[flag] = (mx, my, np.abs(to_numpy(wf.array)[0, 0, :, :, 0]) ** 2)
    assert sizes == [(74, 61), (75, 63)]
    (mx0, my0, I0), (mx1, my1, I1) = out[False], out[True]
    np.testing.assert_array_equal(mx0, mx1)
    np.testing.assert_array_equal(my0, my1)
    assert I0.shape == (29, 25)
    rel = np.max(np.abs(I1 - I0)) / np.max(I0)
    assert rel < 1e-2
    np.testing.assert_allclose(I1.sum(), I0.sum(), rtol=1e-4)


# ---------------------------------------------------------------------------
# Cropped propagation window (min_dk) and skip_vacuum under fft_friendly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_dk", [0.3, 0.2865, 0.45])
def test_min_dk_window_is_smallest_smooth_size_meeting_min_dk(tmp_path, monkeypatch, min_dk):
    # For 0.2865, 1/(min_dk * sampling) = 34.9 rounds to 35, which is smooth but too
    # small: along y (dy = 0.0968 A) 35 pixels give dk = 0.295 > min_dk.
    monkeypatch.chdir(tmp_path)
    calc = _setup(_cell(), fft_friendly=True, min_dk=min_dk,
                  probe_positions=[(3.0, 3.0), (5.0, 2.0)])
    n = calc.probe_cropping
    dx, dy = float(calc.dx), float(calc.dy)
    assert _is_smooth(n)
    assert 1 / (n * dx) <= min_dk and 1 / (n * dy) <= min_dk
    smaller = max(m for m in range(1, n) if _is_smooth(m))
    assert 1 / (smaller * min(dx, dy)) > min_dk
    np.testing.assert_allclose(np.diff(to_numpy(calc.kxs)), 1 / (n * dx), rtol=1e-12)
    np.testing.assert_allclose(np.diff(to_numpy(calc.kys)), 1 / (n * dy), rtol=1e-12)
    arr = to_numpy(calc.run(force_rerun=True).array)
    assert arr.shape[2:4] == (n, n)
    assert np.all(np.isfinite(arr)) and np.max(np.abs(arr)) > 0


def test_min_dk_window_larger_than_grid_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="min_dk"):
        _setup(_cell(), fft_friendly=True, min_dk=0.1)


def test_skip_vacuum_keeps_probes_within_cropping_times_sampling(tmp_path, monkeypatch):
    # The window is 35 pixels of 0.0973 A: 3.41 A. The threshold stays
    # 35 * sampling = 3.5 A, so the probe 3.45 A from the atom is kept and the one
    # 3.6 A away is dropped.
    monkeypatch.chdir(tmp_path)
    atom = (1.0, 1.0)
    probes = [(1.0 + 3.0, 1.0), (1.0 + 3.45, 1.0), (1.0 + 3.6, 1.0)]
    calc = _setup(_cell(positions=[[*atom, 2.0]]), fft_friendly=True, aperture=25,
                  min_dk=0.3, skip_vacuum=True, probe_positions=probes)
    window = calc.probe_cropping * max(float(calc.dx), float(calc.dy))
    distances = [np.hypot(px - atom[0], py - atom[1]) for px, py in probes]
    assert window < distances[1] < calc.probe_cropping * calc.sampling < distances[2]
    calc.run(force_rerun=True)
    np.testing.assert_array_equal(to_numpy(calc.probe_indices), [0, 1])

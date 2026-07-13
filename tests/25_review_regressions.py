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
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.wf_data import WFData


def test_wavelength_public_helper_accepts_scalar_without_backend():
    lam = wavelength(100e3)

    assert isinstance(lam, float)
    assert lam > 0


def test_potential_wraps_unwrapped_slice_axis_coordinates():
    xs = np.linspace(0.0, 4.0, 5, endpoint=False)
    ys = np.linspace(0.0, 4.0, 5, endpoint=False)
    zs = np.linspace(0.0, 4.0, 5, endpoint=False)
    atom_types = np.array([14, 14])

    unwrapped = Potential(
        xs,
        ys,
        zs,
        positions=np.array([[[1.0, 1.0, -0.1], [1.0, 1.0, 4.1]]]).reshape(2, 3),
        atom_types=atom_types,
        backend=NumpyBackend(),
        kind="gauss",
    )
    wrapped = Potential(
        xs,
        ys,
        zs,
        positions=np.array([[1.0, 1.0, 3.9], [1.0, 1.0, 0.1]]),
        atom_types=atom_types,
        backend=NumpyBackend(),
        kind="gauss",
    )

    unwrapped.build()
    wrapped.build()

    np.testing.assert_allclose(unwrapped.to_numpy(), wrapped.to_numpy())


def test_potential_flatten_collapses_slice_axis():
    potential = Potential(
        np.arange(3.0),
        np.arange(4.0),
        np.arange(2.0),
        array=np.stack(
            [
                np.ones((3, 4)),
                np.ones((3, 4)) * 3,
            ],
            axis=2,
        ),
        backend=NumpyBackend(),
    )

    potential.flatten()

    assert potential.n_slices == 1
    assert potential.nz == 1
    assert potential.to_numpy().shape == (3, 4, 1)
    np.testing.assert_allclose(potential.to_numpy()[:, :, 0], np.ones((3, 4)) * 2)


def test_force_rerun_bypasses_frame_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_wavefunctions=True,
    )
    calc.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(calc.output_dir / "frame_0.npy", np.zeros((1, 1, 1, 1, 1), dtype=np.complex128))

    original_check_cache = calculators_module.checkCache
    cache_read_flags = []

    def spy_check_cache(cache_file, cache_wavefunctions, *args, **kwargs):
        cache_read_flags.append(cache_wavefunctions)
        return original_check_cache(cache_file, cache_wavefunctions, *args, **kwargs)

    monkeypatch.setattr(calculators_module, "checkCache", spy_check_cache)

    calc.run(force_rerun=True)

    assert cache_read_flags == [False]


def test_existing_tacaw_file_does_not_skip_returned_wavefunctions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_wavefunctions=False,
    )
    calc.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(calc.output_dir / "tacaw.npy", np.zeros((1, 1, 1, 1), dtype=np.float64))

    wave = calc.run()

    assert np.any(np.abs(to_numpy(wave.array)) > 0)


def test_loop_probes_keeps_selected_indices_integer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_xs=[1.0, 2.0],
        probe_ys=[1.0, 2.0],
        loop_probes=2,
        cache_wavefunctions=False,
    )

    wave = calc.run(force_rerun=True)

    assert to_numpy(wave.array).shape[0] == 4


def test_preview_probes_uses_flatten_compatibility(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
    )

    calc.preview_probes()


def test_tacaw_rejects_nondivisible_chunk_size(tmp_path):
    wf = _make_wf_data(tmp_path, n_time=10)

    with pytest.raises(ValueError, match="evenly divide"):
        TACAWData(wf, chunk_size_time=4, force_rerun=True)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_torch_backend_fft_accepts_numpy_memmap_inputs(tmp_path):
    backend = TorchBackend(device="cpu")
    expected_input = np.arange(8, dtype=np.complex128).reshape(2, 4)
    memmap = np.lib.format.open_memmap(
        tmp_path / "fft_input.npy",
        dtype=np.complex128,
        mode="w+",
        shape=expected_input.shape,
    )
    memmap[:] = expected_input

    fft = backend.fft(memmap, axes=1)
    fft2 = backend.fft2(memmap)
    shifted = backend.fftshift(memmap, axes=(0, 1))

    assert isinstance(fft, np.ndarray)
    np.testing.assert_allclose(fft, np.fft.fft(expected_input, axis=1))
    np.testing.assert_allclose(fft2, np.fft.fft2(expected_input))
    np.testing.assert_allclose(shifted, np.fft.fftshift(expected_input, axes=(0, 1)))


def test_wfdata_counts_unpacks_histogram_counts(tmp_path):
    wf = _make_wf_data(tmp_path, n_time=4)

    wf.counts(11)
    assert to_numpy(wf.array).shape == (1, 4, 2, 2, 1)
    assert np.sum(to_numpy(wf.array)) == pytest.approx(11)

    wf.counts(3)
    assert np.sum(to_numpy(wf.array)) == pytest.approx(3)


def test_prism_probe_copy_preserves_backend_for_chunk_shifts():
    backend = NumpyBackend()
    xs = np.linspace(0.0, 4.0, 4, endpoint=False)
    ys = np.linspace(0.0, 4.0, 4, endpoint=False)
    probe = PrismProbe(xs, ys, 5, 60e3, backend=backend, nkx=2)
    probe.applyShifts()

    copied = probe.copy(selected_probes=np.array([0, 1]))
    copied.applyShifts()

    assert copied._backend is backend
    assert to_numpy(copied._array).shape[1] == 2


def test_prism_loop_probes_smoke_preserves_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_xs=[1.0, 2.0],
        probe_ys=[1.0],
        prism=2,
        loop_probes=1,
        cache_wavefunctions=False,
    )

    wave = calc.run(force_rerun=True)

    assert to_numpy(wave.array).shape[0] == 2


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


def test_haadf_serialization_excludes_backend():
    assert "_backend" in HAADFData._sea_config["exclude_attrs"]


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


def test_apply_shifts_is_idempotent_for_single_offcentre_probe():
    probe = _make_deferred_probe(_OFFCENTRE)  # cell centre is 4.0 -> off-centre

    probe.applyShifts()
    after_first = to_numpy(probe._array).copy()
    probe.applyShifts()  # the historically buggy re-application
    probe.applyShifts()

    np.testing.assert_array_equal(to_numpy(probe._array), after_first)
    # positioned once: peak at the requested position (not centre, not doubled)
    assert _peak_pixel(probe._array[0, 0]) == _ONCE_PIXEL


def test_apply_shifts_reapplied_after_decoherence_rebuild():
    # setup() applies shifts; addTemporalDecoherence then rebuilds the template
    # from scratch and must re-position it (flag reset), then stay idempotent.
    probe = _make_deferred_probe(_OFFCENTRE)
    probe.applyShifts()
    probe.addTemporalDecoherence(sigma_eV=1.0, N=3)

    assert probe._array.shape[0] == 3  # three energy copies
    summed = np.sum(np.abs(to_numpy(probe._array[:, 0])) ** 2, axis=0)
    assert _peak_pixel(summed) == _ONCE_PIXEL  # shifted exactly once, not twice

    frozen = to_numpy(probe._array).copy()
    probe.applyShifts()  # must be a no-op now
    np.testing.assert_array_equal(to_numpy(probe._array), frozen)


def test_static_trajectory_offcentre_probe_gives_frame_invariant_exit_wave(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n_frames = 4
    static = np.tile(np.array([[[3.1, 4.7, 1.5]]], dtype=float), (n_frames, 1, 1))
    traj = Trajectory(
        atom_types=np.array([14]),
        positions=static,
        velocities=np.zeros_like(static),
        box_matrix=np.diag([8.0, 8.0, 3.0]),
        timestep=0.1,
    )
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        traj,
        aperture=30,
        voltage_eV=60e3,
        sampling=0.25,
        slice_thickness=1.0,
        probe_positions=[_OFFCENTRE],
        cache_wavefunctions=False,
    )
    arr = to_numpy(calc.run(force_rerun=True).array)  # (probe, frame, kx, ky, layer)

    # atoms are frozen -> a correctly-fixed probe reproduces the exit wave every
    # frame; the drift bug moved the probe each frame, so patterns diverged.
    reference = arr[:, 0]
    for t in range(1, n_frames):
        np.testing.assert_allclose(arr[:, t], reference, atol=1e-10, rtol=0)


def test_explicit_probe_position_list_is_simulated_verbatim(tmp_path, monkeypatch):
    # A non-grid probe_positions list (e.g. two atomic columns on a diagonal)
    # used to be silently rebuilt into the outer-product grid of its unique x/y
    # values.  That both changed the physics and desynced n_probes from the
    # number of probes actually simulated -> a shape-mismatch crash as soon as
    # wavefunctions were returned.
    monkeypatch.chdir(tmp_path)
    n_frames = 2
    static = np.tile(np.array([[[3.1, 4.7, 1.5]]], dtype=float), (n_frames, 1, 1))
    traj = Trajectory(
        atom_types=np.array([14]),
        positions=static,
        velocities=np.zeros_like(static),
        box_matrix=np.diag([8.0, 8.0, 3.0]),
        timestep=0.1,
    )
    requested = [(2.0, 2.0), (6.0, 6.0)]  # diagonal -> not a full 2x2 grid
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        traj,
        aperture=30,
        voltage_eV=60e3,
        sampling=0.25,
        slice_thickness=1.0,
        probe_positions=requested,
        cache_wavefunctions=False,
    )
    wave = calc.run(force_rerun=True)  # must not raise

    # exactly the requested probes are simulated (not expanded to a 4-point grid)
    np.testing.assert_allclose(
        np.asarray(calc.base_probe.probe_positions, dtype=float), requested)
    assert len(calc.probe_positions) == len(calc.base_probe.probe_positions)
    assert to_numpy(wave.array).shape[0] == len(requested)


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


def test_full_grid_probe_list_canonicalised_so_image_maps_correctly(tmp_path, monkeypatch):
    # A full grid handed in as an explicit list in a non-meshgrid order must
    # produce the SAME 2D image as the probe_xs/probe_ys path: WFData.reshaped()
    # assumes meshgrid order, so setup() canonicalises full grids to it.
    reference = _adf_image_for(dict(probe_xs=[2.0, 6.0], probe_ys=[2.0, 6.0]),
                               tmp_path / "ref", monkeypatch)
    scrambled = _adf_image_for(dict(probe_positions=[(6., 6.), (2., 2.), (6., 2.), (2., 6.)]),
                               tmp_path / "scr", monkeypatch)
    nested = _adf_image_for(dict(probe_positions=[(2., 2.), (2., 6.), (6., 2.), (6., 6.)]),
                            tmp_path / "nest", monkeypatch)
    np.testing.assert_allclose(scrambled, reference, atol=1e-10, rtol=0)
    np.testing.assert_allclose(nested, reference, atol=1e-10, rtol=0)


def test_malformed_probe_positions_raise_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traj = _make_tiny_trajectory()
    for bad in [(4.0, 4.0), [], [(2, 2, 4), (6, 6, 4)]]:
        calc = MultisliceCalculator(force_cpu=True)
        with pytest.raises(ValueError, match="probe_positions must be"):
            calc.setup(traj, aperture=5, voltage_eV=60e3, sampling=1.0,
                       slice_thickness=1.0, probe_positions=bad)


# ---------------------------------------------------------------------------
# Regression: probe cropping (min_dk) must (a) not crash on the b.roll call and
# (b) read the potential sub-window centred on each probe, not the grid corner,
# for every coherent (decoherence) copy.  For a single-slice potential there is
# no inter-slice propagation, so a cropped exit wave must EXACTLY equal the
# uncropped exit wave restricted to that probe's window.
# ---------------------------------------------------------------------------

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


def test_cropped_probe_reads_window_centred_on_probe_not_grid_corner():
    C = 16
    for position, decohere in [((4.0, 4.0), False),   # centred
                               ((2.0, 6.0), False),   # off-centre
                               ((4.0, 4.0), True)]:    # centred + decoherence
        full, _ = _propagate(position, 0, decohere)
        crop, probe = _propagate(position, C, decohere)
        assert crop.shape[-2:] == (C, C)
        # every coherent copy must match the uncropped window at the probe's
        # recorded offset (b.roll crash + corner-window + decoherence-row bugs)
        ox, oy = (int(v) for v in probe.offsets[0])
        window = full[:, ox:ox + C, oy:oy + C]
        np.testing.assert_allclose(crop, window, atol=1e-12, rtol=0)
    # and the centred crop must NOT be the grid-corner window (the old bug)
    full, _ = _propagate((4.0, 4.0), 0)
    crop, _ = _propagate((4.0, 4.0), C)
    assert np.max(np.abs(crop[0] - full[0, 0:C, 0:C])) > 1e-3


def test_min_dk_calculator_run_completes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    static = np.tile(np.array([[[4.0, 4.0, 1.5]]], dtype=float), (2, 1, 1))
    traj = Trajectory(
        atom_types=np.array([14]),
        positions=static,
        velocities=np.zeros_like(static),
        box_matrix=np.diag([8.0, 8.0, 3.0]),
        timestep=0.1,
    )
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25,
               slice_thickness=1.0, probe_positions=[(4.0, 4.0), (2.0, 6.0)],
               min_dk=0.25, cache_wavefunctions=False)
    arr = to_numpy(calc.run(force_rerun=True).array)  # previously TypeError in b.roll
    assert np.all(np.isfinite(arr))
    assert np.max(np.abs(arr)) > 0


@pytest.mark.parametrize("n", [32, 33, 63])
def test_pad_real_space_preserves_field_for_odd_and_even_grids(tmp_path, n):
    # self._array is an fftshifted spectrum; pad_real_space must undo/redo the
    # shift around its FFT round trip.  Without that it is exact only for even
    # grids and smears odd-grid spectra by several percent.
    pad = 5
    xs = np.arange(n, dtype=float)
    xg, yg = np.meshgrid(xs, xs, indexing="ij")
    psi = (np.exp(-((xg - n * 0.4) ** 2 + (yg - n * 0.6) ** 2) / (0.8 * n))
           * np.exp(1j * 0.3 * xg))  # asymmetric -> sensitive to a shift error
    spectrum = np.fft.fftshift(np.fft.fft2(psi))

    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    wf = WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
        time=np.zeros(1),
        kxs=np.fft.fftshift(np.fft.fftfreq(n)), kys=np.fft.fftshift(np.fft.fftfreq(n)),
        xs=xs.copy(), ys=xs.copy(), layer=np.array([0]),
        array=spectrum[None, None, :, :, None].astype(np.complex128),
        probe=probe, backend=NumpyBackend(), cache_dir=tmp_path,
    )
    wf.pad_real_space(add_x=pad, add_y=pad)

    recovered = np.fft.ifft2(np.fft.ifftshift(to_numpy(wf._array)[0, 0, :, :, 0]))
    reference = np.zeros((n + 2 * pad, n + 2 * pad), dtype=complex)
    reference[pad:pad + n, pad:pad + n] = psi
    rel_err = np.max(np.abs(recovered - reference)) / np.max(np.abs(psi))
    assert rel_err < 1e-10


def test_pyproject_sdist_ships_package_source():
    # The sdist must declare the package source; otherwise hatchling shipped a
    # data-only sdist and every sdist-based install produced an empty package.
    import tomllib
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "pyproject.toml").read_text())
    build = cfg["tool"]["hatch"]["build"]
    sdist_include = build["targets"]["sdist"]["include"]
    assert any("src/pyslice" in p for p in sdist_include), sdist_include
    # no lingering global data-only include and no dead setuptools placeholder
    assert "include" not in build
    assert "setuptools" not in cfg["tool"]


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

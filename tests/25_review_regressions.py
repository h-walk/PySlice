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


@pytest.mark.parametrize("use_memmap", [False, True])
def test_prism_adf_reconstructs_default_and_multilayer_outputs(
    tmp_path, monkeypatch, use_memmap
):
    def run(subdir, return_layers):
        work = tmp_path / subdir
        work.mkdir()
        monkeypatch.chdir(work)
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            _make_tiny_trajectory(), aperture=30, voltage_eV=60e3,
            sampling=1.0, slice_thickness=1.0,
            probe_xs=[1.0, 2.0], probe_ys=[1.0], prism=2,
            loop_probes=1, cache_wavefunctions=False, ADF=(5, 40),
            return_layers=return_layers, use_memmap=use_memmap)
        return calc, calc.run(force_rerun=True)

    _, (wave, adf) = run("exit", -1)
    assert to_numpy(wave.array).shape == (2, 1, 5, 5, 1)
    assert np.any(np.abs(to_numpy(wave.array)) > 0)
    assert to_numpy(adf.array).shape == (2, 1)

    calc, (wave_all, adf_all) = run("all", "all")
    assert to_numpy(wave_all.array).shape == (2, 1, 5, 5, calc.nz)
    assert np.all(np.any(np.abs(to_numpy(wave_all.array)) > 0, axis=(0, 1, 2, 3)))
    assert to_numpy(adf_all.array).shape == (calc.nz, 2, 1)


def test_prism_cache_validates_component_rows_not_scan_positions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def calculator():
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            _make_tiny_trajectory(), aperture=5, voltage_eV=60e3,
            sampling=1.0, slice_thickness=1.0,
            probe_xs=[1.0, 2.0], probe_ys=[1.0], prism=2,
            loop_probes=1, cache_wavefunctions=True)
        return calc

    first = calculator()
    expected = to_numpy(first.run(force_rerun=True).array).copy()

    def should_not_propagate(*args, **kwargs):
        raise AssertionError("valid PRISM component cache was not reused")

    second = calculator()
    monkeypatch.setattr(calculators_module, "Propagate", should_not_propagate)
    np.testing.assert_allclose(to_numpy(second.run().array), expected)


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


def _make_optics_wf(tmp_path, n=5):
    xs = np.arange(n, dtype=float)
    kxs = np.fft.fftshift(np.fft.fftfreq(n))
    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")
    entrance = (1 + x_grid + 2 * y_grid).astype(np.complex128)
    exit_wave = np.exp(-((x_grid - 1.3) ** 2 + (y_grid - 3.1) ** 2) / 2)
    waves = np.stack(
        [np.fft.fftshift(np.fft.fft2(entrance)),
         np.fft.fftshift(np.fft.fft2(exit_wave))],
        axis=-1,
    )[None, None, :, :, :]
    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(
            np.zeros((1, 1, n, n), dtype=np.complex128)))
    return WFData(
        probe_positions=[(2.0, 2.0)], probe_xs=[2.0], probe_ys=[2.0],
        time=np.zeros(1), kxs=kxs, kys=kxs.copy(),
        xs=xs, ys=xs.copy(), layer=np.array([1.0, 2.0]),
        array=waves, probe=probe, backend=NumpyBackend(), cache_dir=tmp_path,
    )


def test_free_space_propagates_only_the_exit_wave(tmp_path):
    wf = _make_optics_wf(tmp_path)
    before = to_numpy(wf.array).copy()
    dz = 80.0

    wf.propagate_free_space(dz)

    after = to_numpy(wf.array)
    np.testing.assert_array_equal(after[..., 0], before[..., 0])
    kx_grid, ky_grid = np.meshgrid(wf.kxs, wf.kys, indexing="ij")
    propagator = np.exp(
        -1j * np.pi * wf.probe.wavelength * dz
        * (kx_grid ** 2 + ky_grid ** 2)
    )
    np.testing.assert_allclose(
        after[..., -1], before[..., -1] * propagator[None, None, :, :]
    )


def test_lens_is_grid_centred_and_modifies_only_the_exit_wave(tmp_path):
    wf = _make_optics_wf(tmp_path)
    before = to_numpy(wf.array).copy()
    f = 500.0

    wf.propagate_through_lens(f)

    after = to_numpy(wf.array)
    np.testing.assert_array_equal(after[..., 0], before[..., 0])
    real_before = np.fft.ifft2(np.fft.ifftshift(before[0, 0, :, :, -1]))
    real_after = np.fft.ifft2(np.fft.ifftshift(after[0, 0, :, :, -1]))
    x_grid, y_grid = np.meshgrid(
        wf.xs - np.mean(wf.xs), wf.ys - np.mean(wf.ys), indexing="ij"
    )
    expected_lens = np.exp(
        -1j * (2 * np.pi / wf.probe.wavelength) / (2 * f)
        * (x_grid ** 2 + y_grid ** 2)
    )
    np.testing.assert_allclose(real_after, real_before * expected_lens, atol=1e-12)


def test_lens_accepts_an_explicit_optical_axis(tmp_path):
    wf = _make_optics_wf(tmp_path)
    before = to_numpy(wf.array).copy()
    f = 500.0
    center = (1.0, 3.0)

    wf.propagate_through_lens(f, center=center)

    real_before = np.fft.ifft2(np.fft.ifftshift(before[0, 0, :, :, -1]))
    real_after = np.fft.ifft2(
        np.fft.ifftshift(to_numpy(wf.array)[0, 0, :, :, -1])
    )
    x_grid, y_grid = np.meshgrid(
        wf.xs - center[0], wf.ys - center[1], indexing="ij"
    )
    expected_lens = np.exp(
        -1j * (2 * np.pi / wf.probe.wavelength) / (2 * f)
        * (x_grid ** 2 + y_grid ** 2)
    )
    np.testing.assert_allclose(real_after, real_before * expected_lens, atol=1e-12)


def test_downstream_optics_use_each_decoherence_copy_wavelength(tmp_path):
    n = 5
    n_copies, n_positions = 2, 2
    xs = np.arange(n, dtype=float)
    kxs = np.fft.fftshift(np.fft.fftfreq(n))
    wavelengths = np.array([0.03, 0.06])
    probe = SimpleNamespace(
        eV=1e5, wavelength=wavelengths[0], wavelengths=wavelengths,
        mrad=30.0,
        _array=np.zeros((n_copies, n_positions, n, n), dtype=np.complex128),
    )
    # Flattening is copy-major: each wavelength applies to all scan positions
    # for one decoherence copy.
    array = np.ones(
        (n_copies * n_positions, 1, n, n, 2), dtype=np.complex128
    )
    wf = WFData(
        probe_positions=[(1.0, 1.0), (3.0, 3.0)],
        probe_xs=[1.0, 3.0], probe_ys=[1.0, 3.0], time=np.zeros(1),
        kxs=kxs, kys=kxs.copy(), xs=xs, ys=xs.copy(),
        layer=np.array([1.0, 2.0]), array=array, probe=probe,
        backend=NumpyBackend(), cache_dir=tmp_path,
    )
    before = wf.array.copy()
    dz = 80.0

    wf.propagate_free_space(dz)

    kx_grid, ky_grid = np.meshgrid(kxs, kxs, indexing="ij")
    k_sq = kx_grid ** 2 + ky_grid ** 2
    for copy, wavelength_value in enumerate(wavelengths):
        expected = np.exp(-1j * np.pi * wavelength_value * dz * k_sq)
        start = copy * n_positions
        stop = start + n_positions
        np.testing.assert_allclose(
            wf.array[start:stop, 0, :, :, -1],
            np.broadcast_to(expected, (n_positions, n, n)),
        )
    np.testing.assert_array_equal(wf.array[..., 0], before[..., 0])


def test_slice_axis_other_than_z_is_rejected(tmp_path, monkeypatch):
    # slice_axis != 2 was silently wrong (propagation is z-locked); it must now
    # fail loudly at both entry points.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError, match="slice_axis"):
        MultisliceCalculator(force_cpu=True).setup(
            _make_tiny_trajectory(), aperture=5, voltage_eV=60e3, sampling=1.0,
            slice_thickness=1.0, slice_axis=1)
    with pytest.raises(NotImplementedError, match="slice_axis"):
        Potential(np.arange(3.0), np.arange(3.0), np.arange(3.0),
                  positions=np.array([[1.0, 1.0, 1.0]]),
                  atom_types=np.array([14]), backend=NumpyBackend(),
                  kind="gauss", slice_axis=0)


def _loader(tmp_path, **kwargs):
    dump = tmp_path / "dummy.lammpstrj"
    dump.write_text("")  # only needs to exist; we call the resolver directly
    return Loader(filename=str(dump), **kwargs)


def test_ovito_type_ids_are_never_used_as_atomic_numbers(tmp_path):
    # LAMMPS type IDs must not be treated as Z. Element identity comes from an
    # explicit mapping or the file's embedded element names; otherwise abort.
    ids = np.array([1, 1, 2])

    # embedded element names present -> mapped to element symbols
    resolved = _loader(tmp_path)._resolve_ovito_types(ids, {1: "Si", 2: "O"})
    assert list(resolved) == ["Si", "Si", "O"]

    # no names and no mapping -> abort asking for an exact mapping
    with pytest.raises(ValueError, match="atom_mapping"):
        _loader(tmp_path)._resolve_ovito_types(ids, {})
    # a numeric/junk type name is NOT accepted as an element
    with pytest.raises(ValueError, match="atom_mapping"):
        _loader(tmp_path)._resolve_ovito_types(np.array([1]), {1: "1"})

    # explicit mapping wins and yields atomic numbers
    resolved = _loader(tmp_path, atom_mapping={1: "Si", 2: "O"})._resolve_ovito_types(ids, {})
    assert list(resolved) == [14, 14, 8]

    # explicit mapping must be exact: a missing type aborts
    with pytest.raises(ValueError, match="missing entries"):
        _loader(tmp_path, atom_mapping={1: "Si"})._resolve_ovito_types(ids, {2: "O"})


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


def test_wavefunction_cache_key_uses_full_trajectory(tmp_path, monkeypatch):
    # The key hashes the whole trajectory, so runs differing only in a later
    # frame or in y/z (e.g. a temperature sweep from the same initial structure)
    # no longer collide on the frame-0-x fingerprint.
    monkeypatch.chdir(tmp_path)
    base = (np.random.RandomState(0).rand(5, 2, 3) * 3).astype(np.float32)

    def key(pos):
        traj = Trajectory(
            atom_types=np.array([14, 14]), positions=pos.astype(np.float32),
            velocities=np.zeros_like(pos, dtype=np.float32),
            box_matrix=np.diag([8.0, 8.0, 3.0]), timestep=0.1)
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25,
                   slice_thickness=1.0, probe_positions=[(4.0, 4.0)],
                   cache_wavefunctions=False)
        return calc.cache_key

    k = key(base.copy())
    assert key(base.copy()) == k                       # identical -> shared cache
    later = base.copy(); later[3, 0, 0] += 0.5
    assert key(later) != k                             # differs only in frame 3
    ycomp = base.copy(); ycomp[0, 0, 1] += 0.5
    assert key(ycomp) != k                             # differs only in frame-0 y


def test_wavefunction_cache_key_partitions_skip_vacuum(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traj = _make_tiny_trajectory()

    def key(skip_vacuum):
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            traj, aperture=30, voltage_eV=60e3, sampling=0.25,
            slice_thickness=1.0, probe_xs=[1.0, 2.0], probe_ys=[1.0],
            min_dk=0.25, skip_vacuum=skip_vacuum)
        return calc.cache_key

    assert key(False) != key(True)


def _make_layered_wf(cache_dir, n_layers=2, seed=0):
    nt = 16
    t = np.arange(nt) * 0.1
    arr = np.zeros((1, nt, 2, 2, n_layers), dtype=np.complex128)
    for layer in range(n_layers):
        freq = (layer + 1) * 1.0 + seed * 0.3  # distinct per layer/dataset
        arr[0, :, :, :, layer] = np.cos(2 * np.pi * freq * t)[:, None, None]
    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    return WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0], time=t,
        kxs=np.arange(2.0), kys=np.arange(2.0), xs=np.arange(2.0), ys=np.arange(2.0),
        layer=np.arange(n_layers), array=arr, probe=probe,
        backend=NumpyBackend(), cache_dir=cache_dir)


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


def test_calculator_decoherence_flows_into_folded_tacaw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = np.tile(
        np.array([[[1.5, 1.5, 1.5]]], dtype=np.float32), (4, 1, 1))
    trajectory = Trajectory(
        atom_types=np.array([14]), positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([4.0, 4.0, 3.0]), timestep=0.1)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        trajectory, aperture=20, voltage_eV=60e3, sampling=1.0,
        slice_thickness=1.0, probe_xs=[1.0, 2.0], probe_ys=[1.0],
        cache_wavefunctions=False)
    calc.base_probe.addTemporalDecoherence(2.0, 3)
    wf = calc.run(force_rerun=True)
    assert to_numpy(wf.array).shape[:2] == (6, 4)

    raw = to_numpy(wf.array[:, :, :, :, -1])
    raw_fft = np.fft.fftshift(
        np.fft.fft(raw - raw.mean(axis=1, keepdims=True), axis=1), axes=1)
    expected = (np.abs(raw_fft) ** 2).reshape(
        3, 2, *raw_fft.shape[1:]).sum(axis=0)
    tacaw = TACAWData(wf, force_rerun=True)
    assert tacaw.array.shape[0] == 2
    np.testing.assert_allclose(tacaw.array, expected)


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


def test_cache_versions_are_derived_from_source(tmp_path):
    from pyslice.backend import source_files_version
    # the helper is content-sensitive and tolerates missing files
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\n")
    v1 = source_files_version([f])
    f.write_bytes(b"x = 2\n")
    assert source_files_version([f]) != v1
    source_files_version([tmp_path / "does_not_exist.py"])  # must not raise
    # both cache tags embed a source hash (not a bare manual constant) and are
    # independent of each other
    wf_v = MultisliceCalculator._CACHE_VERSION
    tacaw_v = TACAWData._TACAW_CACHE_VERSION
    assert wf_v.startswith("v3-") and len(wf_v) > len("v3-")
    assert tacaw_v.startswith("v1-") and len(tacaw_v) > len("v1-")
    assert wf_v != tacaw_v


def test_slice_positions_translates_atoms_into_new_box():
    # slice_positions shrinks the box to the range width; the kept atoms must be
    # translated by the range lower bound so they land inside [0, width), not
    # left at their original coordinates outside the new box.
    pos = np.array([[[1.0, 1.0, 1.0], [6.0, 6.0, 6.0], [7.5, 2.0, 3.0]]], dtype=float)
    traj = Trajectory(
        atom_types=np.array([14, 14, 14]), positions=pos,
        velocities=np.zeros_like(pos), box_matrix=np.diag([10.0, 10.0, 10.0]),
        timestep=0.1)
    sliced = traj.slice_positions(x_range=(5.0, 8.0))
    assert sliced.n_atoms == 2
    assert sliced.box_matrix[0, 0] == 3.0
    xs = sliced.positions[0, :, 0]
    assert np.all((xs >= 0.0) & (xs <= 3.0))          # atoms 6.0, 7.5 -> 1.0, 2.5
    np.testing.assert_allclose(sorted(xs), [1.0, 2.5])


def test_slice_positions_applies_crop_when_every_atom_survives():
    pos = np.array([[[6.0, 1.0, 1.0], [7.5, 2.0, 3.0]]], dtype=float)
    traj = Trajectory(
        atom_types=np.array([14, 14]), positions=pos,
        velocities=np.zeros_like(pos), box_matrix=np.diag([10.0, 10.0, 10.0]),
        timestep=0.1)
    sliced = traj.slice_positions(x_range=(5.0, 8.0))
    assert sliced is not traj
    assert sliced.box_matrix[0, 0] == 3.0
    np.testing.assert_allclose(sliced.positions[0, :, 0], [1.0, 2.5])


def test_ovito_cell_matrix_transposed_to_row_convention():
    from pyslice.io.loader import _ovito_cell_to_row_convention
    a = np.array([10.0, 0.0, 0.0]); b = np.array([2.0, 10.0, 0.0])
    c = np.array([1.0, 1.5, 10.0]); origin = np.array([-5.0, 0.0, 0.0])
    ovito_matrix = np.column_stack([a, b, c, origin])  # vectors in columns, origin last
    box, got_origin = _ovito_cell_to_row_convention(ovito_matrix)
    # rows are the lattice vectors (ASE / PySlice convention), not columns
    np.testing.assert_allclose(box[0], a)
    np.testing.assert_allclose(box[1], b)
    np.testing.assert_allclose(box[2], c)
    np.testing.assert_allclose(got_origin, origin)
    # a plain orthorhombic zero-origin cell is unchanged by the transpose
    orth = np.column_stack([np.diag([8.0, 6.0, 4.0]), np.zeros(3)])
    box2, o2 = _ovito_cell_to_row_convention(orth)
    np.testing.assert_allclose(box2, np.diag([8.0, 6.0, 4.0]))
    np.testing.assert_allclose(o2, np.zeros(3))


def test_npt_barostat_params_have_physical_units():
    from ase import units
    from pyslice.md.molecular_dynamics import MDCalculator
    externalstress, pfactor = MDCalculator._npt_barostat_params(1.01325, 100.0)
    # externalstress is ~1 atm in eV/A^3, not 1.01325 eV/A^3 (~162 GPa)
    np.testing.assert_allclose(externalstress / units.bar, 1.01325, rtol=1e-6)
    assert externalstress < 1e-6                       # ~6.3e-7, not ~1
    # pfactor = ptime^2 * B (was 75*fs**2, ~1e5x too small)
    np.testing.assert_allclose(pfactor, (75 * units.fs) ** 2 * (100.0 * units.GPa))


def test_counts_samples_from_intensity_not_amplitude():
    # Shot-noise counting must draw from |psi|^2 (intensity), not |psi|. Two
    # pixels with amplitudes 1 and 3 (intensities 1 and 9) must be hit ~9:1.
    from types import SimpleNamespace
    arr = np.array([1.0, 3.0], dtype=np.complex128).reshape(1, 1, 1, 2, 1)
    probe = SimpleNamespace(
        eV=1e5, wavelength=0.037, mrad=30.0,
        _array=NumpyBackend().asarray(np.zeros((1, 1, 2, 2), dtype=np.complex128)))
    wf = WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
        time=np.zeros(1), kxs=np.arange(1.0), kys=np.arange(2.0),
        xs=np.arange(1.0), ys=np.arange(2.0), layer=np.array([0]),
        array=arr, probe=probe, backend=NumpyBackend(), cache_dir=".")
    wf.counts(2_000_000)
    hits = to_numpy(wf._array).ravel()
    ratio = hits[1] / hits[0]
    assert 8.3 < ratio < 9.7, ratio          # intensity 9:1, not amplitude 3:1


def test_loader_parse_index_forms():
    from pyslice.io.loader import _parse_index
    assert _parse_index(":") == slice(None, None, None)
    assert _parse_index(":3") == slice(None, 3, None)
    assert _parse_index("-3:") == slice(-3, None, None)
    assert _parse_index("::2") == slice(None, None, 2)
    assert _parse_index("3:5") == slice(3, 5, None)
    assert _parse_index("3-5") == slice(3, 6, None)   # inclusive dash range
    assert _parse_index("1") == 1
    assert _parse_index(-1) == -1
    import pytest as _pytest
    with _pytest.raises(ValueError):
        _parse_index("nonsense")


def test_multiframe_cif_loads_all_frames_and_index_selects(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    from ase import Atoms
    frames = [Atoms("H", positions=[[float(i), 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
              for i in range(6)]
    cif = tmp_path / "traj.cif"
    ase_io.write(str(cif), frames)
    from pyslice.io.loader import Loader

    def load(index):
        return Loader(str(cif), timestep=0.5, index=index).load()

    # default reads every image (previously only the last one survived)
    full = load(":")
    assert full.n_frames == 6
    np.testing.assert_allclose(full.positions[:, 0, 0], np.arange(6))
    # selectors
    assert load(":3").n_frames == 3
    np.testing.assert_allclose(load("-3:").positions[:, 0, 0], [3, 4, 5])
    np.testing.assert_allclose(load("3-5").positions[:, 0, 0], [3, 4, 5])  # inclusive
    strided = load("::2")
    np.testing.assert_allclose(strided.positions[:, 0, 0], [0, 2, 4])
    assert strided.timestep == 1.0                    # step rescales timestep
    assert load(1).n_frames == 1


def test_loader_cache_tracks_source_parser_and_mapping_inputs(tmp_path, monkeypatch):
    source = tmp_path / "trajectory.fake"
    source.write_text("1")
    calls = []

    def fake_parse(self):
        calls.append((source.read_text(), self.atomic_numbers))
        x = float(source.read_text())
        positions = np.array([[[x, 0.0, 0.0]]], dtype=np.float32)
        atom_type = 14 if self.atomic_numbers is None else self.atomic_numbers[1]
        return Trajectory(
            atom_types=np.array([atom_type]), positions=positions,
            velocities=np.zeros_like(positions), box_matrix=np.eye(3) * 10,
            timestep=self.timestep)

    monkeypatch.setattr(Loader, "_load_via_ovito", fake_parse)
    assert Loader(str(source)).load().positions[0, 0, 0] == 1
    assert Loader(str(source)).load().positions[0, 0, 0] == 1
    assert len(calls) == 1                              # unchanged cache hit

    source.write_text("2")
    assert Loader(str(source)).load().positions[0, 0, 0] == 2
    assert len(calls) == 2                              # source invalidated

    mapped = Loader(str(source), atom_mapping={1: "C"}).load()
    assert mapped.atom_types.tolist() == [6]
    assert len(calls) == 3                              # mapping invalidated
    assert Loader(str(source), atom_mapping={1: 6}).load().atom_types.tolist() == [6]
    assert len(calls) == 3                              # canonical mapping cache hit

    # Legacy existence-only caches have no provenance and must be reparsed.
    Loader(str(source))._get_cache_files()["metadata"].unlink()
    Loader(str(source)).load()
    assert len(calls) == 4


def _adf_run(tmp_path, monkeypatch, **setup_kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    s = np.array([[[4., 4., 1.0], [3., 5., 2.5]]], dtype=np.float32)
    traj = Trajectory(atom_types=np.array([14, 14]), positions=s,
                      velocities=np.zeros_like(s), box_matrix=np.diag([8., 8., 4.]),
                      timestep=0.1)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25, slice_thickness=1.0,
               probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.], ADF=(45, 150),
               cache_wavefunctions=False, **setup_kw)
    return calc.run(force_rerun=True)


def test_adf_depth_resolved_stack_not_layer_sum(tmp_path, monkeypatch):
    # Default: a single exit-wave ADF, kept 2D for back-compat.
    _, adf = _adf_run(tmp_path, monkeypatch)
    assert to_numpy(adf.array).shape == (3, 3)
    # return_layers='all': one ADF per thickness (a stack), NOT a sum over layers.
    _, adf_all = _adf_run(tmp_path / "all", monkeypatch, return_layers="all")
    stack = to_numpy(adf_all.array)
    assert stack.ndim == 3 and stack.shape[1:] == (3, 3)
    assert stack.shape[0] == len(adf_all.thicknesses)
    # Stored waves are captured after each slice transmission, so their depth
    # is the far boundary of that slice: first > 0 and exit == specimen depth.
    expected_depths = np.linspace(4.0 / len(stack), 4.0, len(stack))
    np.testing.assert_allclose(adf_all.thicknesses, expected_depths)
    np.testing.assert_allclose(adf.thicknesses, [4.0])
    np.testing.assert_allclose(stack[-1], to_numpy(adf.array), atol=1e-6)  # exit == default
    assert not np.allclose(stack[0], stack[-1])          # genuinely per-thickness
    assert not np.allclose(stack[-1], stack.sum(0))      # not the old layer-sum


def test_adf_depth_stack_exposes_a_layer_dimension(monkeypatch):
    import pyslice.postprocessing.haadf_data as haadf_module

    class FakeDimension:
        def __init__(self, **kwargs):
            self.name = kwargs['name']
            self.values = np.asarray(kwargs['values'])
            self.units = kwargs.get('units')

    class FakeDimensions:
        def __init__(self, dimensions, nav_dimensions, sig_dimensions):
            self.dimensions = dimensions
            self.nav_dimensions = nav_dimensions
            self.sig_dimensions = sig_dimensions

    class FakeMetadata:
        def __init__(self, values):
            self.Simulation = SimpleNamespace(**values['Simulation'])

    monkeypatch.setattr(haadf_module, 'Dimension', FakeDimension)
    monkeypatch.setattr(haadf_module, 'Dimensions', FakeDimensions)
    monkeypatch.setattr(haadf_module, 'Metadata', FakeMetadata)

    wf = _make_layered_wf(None, n_layers=2)
    adf = haadf_module.HAADFData(wf)
    adf.calculateADF(inner_mrad=0, outer_mrad=1e6)
    assert [d.name for d in adf.dimensions.dimensions] == ['layer', 'x', 'y']
    np.testing.assert_array_equal(adf.dimensions.dimensions[0].values, [0, 1])


def test_adf_sums_decoherence_copies(tmp_path, monkeypatch):
    def total(n_copies):
        s = np.array([[[4., 4., 1.0], [3., 5., 2.5]]], dtype=np.float32)
        traj = Trajectory(atom_types=np.array([14, 14]), positions=s,
                          velocities=np.zeros_like(s), box_matrix=np.diag([8., 8., 4.]),
                          timestep=0.1)
        monkeypatch.chdir(tmp_path)
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25, slice_thickness=1.0,
                   probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.], ADF=(45, 150),
                   cache_wavefunctions=False, return_layers=None)
        if n_copies:
            calc.base_probe.addTemporalDecoherence(2.0, n_copies)
        return float(np.sum(np.abs(to_numpy(calc.run(force_rerun=True)[1].array))))
    baseline = total(None)
    # Correct quadrature is independent of the number of samples. N=3 happened
    # to pass before because its central sample dominated; N=5/7 exposed the
    # missing normalisation as a dose increase.
    for n_copies in (3, 5, 7):
        ratio = total(n_copies) / baseline
        assert 0.9 < ratio < 1.1, (n_copies, ratio)


def test_decoherence_default_wavefunction_storage_keeps_every_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = np.array([[[4., 4., 1.0], [3., 5., 2.5]]], dtype=np.float32)
    traj = Trajectory(
        atom_types=np.array([14, 14]), positions=s, velocities=np.zeros_like(s),
        box_matrix=np.diag([8., 8., 4.]), timestep=0.1)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        traj, aperture=30, voltage_eV=100e3, sampling=0.25,
        slice_thickness=1.0, probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.],
        ADF=(45, 150), cache_wavefunctions=True,
        loop_probes=2)  # default return_layers=-1
    calc.base_probe.addTemporalDecoherence(2.0, 3)
    wf, adf = calc.run(force_rerun=True)
    assert wf.array.shape[0] == 3 * 9
    assert adf.array.shape == (3, 3)
    assert np.isfinite(to_numpy(wf.array)).all()

    # A second calculator must fold all cached copies into the same detector
    # result; the old cache path indexed only the first copy.
    expected = to_numpy(adf.array).copy()
    cached_calc = MultisliceCalculator(force_cpu=True)
    cached_calc.setup(
        traj, aperture=30, voltage_eV=100e3, sampling=0.25,
        slice_thickness=1.0, probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.],
        ADF=(45, 150), cache_wavefunctions=True, loop_probes=2)
    cached_calc.base_probe.addTemporalDecoherence(2.0, 3)
    cached_wf, cached_adf = cached_calc.run()
    assert cached_wf.array.shape[0] == 3 * 9
    np.testing.assert_allclose(to_numpy(cached_adf.array), expected, rtol=1e-7)


def test_cached_looped_adf_uses_full_scan_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traj = _make_tiny_trajectory()

    def calculator():
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            traj, aperture=20, voltage_eV=60e3, sampling=0.5,
            slice_thickness=1.0, probe_xs=[1.0, 2.0],
            probe_ys=[1.0, 2.0], loop_probes=2, ADF=(5, 40),
            cache_wavefunctions=True)
        return calc

    first = calculator()
    expected = to_numpy(first.run(force_rerun=True)[1].array).copy()
    second = calculator()
    np.testing.assert_allclose(to_numpy(second.run()[1].array), expected)


def test_decoherence_weights_preserve_total_probe_dose():
    baseline_probe = _make_deferred_probe((4.0, 4.0))
    baseline_probe.applyShifts()
    baseline = np.sum(np.abs(to_numpy(baseline_probe._array)) ** 2)

    for n_copies in (3, 5, 7):
        probe = _make_deferred_probe((4.0, 4.0))
        probe.addTemporalDecoherence(2.0, n_copies)
        total = np.sum(np.abs(to_numpy(probe._array)) ** 2)
        np.testing.assert_allclose(total, baseline, rtol=1e-7)

    probe = _make_deferred_probe((4.0, 4.0))
    probe.addSpatialDecoherence(50.0, 7)
    total = np.sum(np.abs(to_numpy(probe._array)) ** 2)
    np.testing.assert_allclose(total, baseline, rtol=1e-7)

    with pytest.raises(ValueError, match="positive"):
        probe.addTemporalDecoherence(0, 3)
    with pytest.raises(ValueError, match="positive integer"):
        probe.addSpatialDecoherence(1, 0)

    # One quadrature point denotes the distribution centre, not its -2 sigma
    # endpoint (numpy.linspace(start, stop, 1) returns start).
    temporal_one = _make_deferred_probe((4.0, 4.0))
    temporal_one.addTemporalDecoherence(2.0, 1)
    np.testing.assert_allclose(to_numpy(temporal_one.eVs), [temporal_one.eV])
    spatial_one = _make_deferred_probe((4.0, 4.0))
    reference = to_numpy(spatial_one._array).copy()
    spatial_one.addSpatialDecoherence(50.0, 1)
    np.testing.assert_allclose(to_numpy(spatial_one._array), reference, atol=1e-12)

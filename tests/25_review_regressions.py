from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
if TORCH_AVAILABLE:
    from pyslice.backend import TorchBackend
from pyslice.multislice import calculators as calculators_module
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.multislice.multislice import PrismProbe, wavelength
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

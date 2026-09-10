from types import SimpleNamespace

import numpy as np
import pytest

from pyslice import WFData
from pyslice.backend import NumpyBackend, TorchBackend
from pyslice.postprocessing.tacaw_data import (
    TACAWData,
    bose_correction_factor,
)


def _fake_wf_data(tmp_path, array=None):
    rng = np.random.default_rng(123)
    if array is None:
        array = (
            rng.normal(size=(2, 8, 3, 4, 1))
            + 1j * rng.normal(size=(2, 8, 3, 4, 1))
        ).astype(np.complex128)

    backend = NumpyBackend()
    return WFData(
        probe_positions=[(0.0, 0.0), (1.0, 1.0)],
        probe_xs=[0.0, 1.0],
        probe_ys=[0.0, 1.0],
        time=np.arange(array.shape[1], dtype=float) * 0.005,
        kxs=np.linspace(-1.0, 1.0, array.shape[2]),
        kys=np.linspace(-1.5, 1.5, array.shape[3]),
        xs=np.linspace(0.0, 2.0, array.shape[2]),
        ys=np.linspace(0.0, 3.0, array.shape[3]),
        layer=np.array([0]),
        probe=SimpleNamespace(eV=100e3, wavelength=0.037, mrad=0.0),
        cache_dir=tmp_path,
        array=array,
        backend=backend,
    )


def test_bose_correction_factor_has_signed_gain_loss_balance():
    frequencies = np.array([-30.0, 0.0, 30.0])

    factors = bose_correction_factor(frequencies, temperature_K=300.0)

    assert factors[0] < 1.0
    assert factors[1] == pytest.approx(1.0)
    assert factors[2] > 1.0

    beta_e = 30.0 * 4.135667696e-3 / (8.617333262145e-5 * 300.0)
    assert factors[2] / factors[0] == pytest.approx(np.exp(beta_e))


def test_tacaw_apply_bose_scales_intensity_by_frequency(tmp_path):
    array = _fake_wf_data(tmp_path / "seed")._array
    raw = TACAWData(_fake_wf_data(tmp_path / "raw", array=array))
    corrected = TACAWData(
        _fake_wf_data(tmp_path / "corrected", array=array),
        temperature_K=300.0,
        apply_bose=True,
    )

    factors = bose_correction_factor(raw.frequencies, temperature_K=300.0)
    expected = raw.array * factors[None, :, None, None]

    np.testing.assert_allclose(corrected.array, expected, rtol=1e-10, atol=1e-10)
    assert corrected.temperature_K == 300.0
    assert corrected.gain_loss_folded is False
    assert corrected.apply_bose is True


@pytest.mark.parametrize("apply_bose", [False, True])
def test_constructor_folding_is_opt_in_and_does_not_modify_raw_cache(tmp_path, apply_bose):
    wf = _fake_wf_data(tmp_path)
    raw = TACAWData(wf)
    expected_folded = TACAWData(wf)
    expected_folded.fold_gain_loss()
    expected_unfolded = raw.array.copy()
    if apply_bose:
        expected_folded.apply_bose_correction(300.0)
        factors = bose_correction_factor(raw.frequencies, 300.0)
        expected_unfolded *= factors[None, :, None, None]

    folded = TACAWData(wf, fold=True, apply_bose=apply_bose, temperature_K=300.0)
    unfolded = TACAWData(wf, fold=False, apply_bose=apply_bose, temperature_K=300.0)

    np.testing.assert_allclose(folded.array, expected_folded.array)
    np.testing.assert_allclose(unfolded.array, expected_unfolded)
    assert not np.allclose(folded.array, unfolded.array)
    assert folded.gain_loss_folded is True
    assert unfolded.gain_loss_folded is False
    assert folded.apply_bose is apply_bose
    assert unfolded.apply_bose is apply_bose
    for result in (folded, unfolded):
        if hasattr(result, "metadata") and result.metadata is not None:
            assert result.metadata.Simulation.gain_loss_folded == result.gain_loss_folded
    np.testing.assert_array_equal(TACAWData(wf).array, raw.array)


@pytest.mark.parametrize("fold", [False, True])
def test_bose_method_folding_is_opt_in(tmp_path, fold):
    result = TACAWData(_fake_wf_data(tmp_path))
    reference = TACAWData(_fake_wf_data(tmp_path))
    if fold:
        reference.fold_gain_loss()
    expected = reference.array * bose_correction_factor(reference.frequencies, 300.0)[None, :, None, None]

    kwargs = {"fold": True} if fold else {}  # Exercise the method default too.
    result.apply_bose_correction(300.0, **kwargs)

    np.testing.assert_allclose(result.array, expected)
    assert result.gain_loss_folded is fold
    if hasattr(result, "metadata") and result.metadata is not None:
        assert result.metadata.Simulation.gain_loss_folded == fold


def test_unfolded_bose_correction_accepts_asymmetric_momentum_grid(tmp_path):
    wf = _fake_wf_data(tmp_path)
    wf._kxs = np.array([-1.0, 0.0, 2.0])
    raw = TACAWData(wf)
    corrected = TACAWData(wf, apply_bose=True, temperature_K=300.0)
    factors = bose_correction_factor(raw.frequencies, 300.0)
    np.testing.assert_allclose(corrected.array, raw.array * factors[None, :, None, None])
    with pytest.raises(ValueError, match="not closed under inversion"):
        TACAWData(wf, apply_bose=True, temperature_K=300.0, fold=True)


def test_constructor_folding_rejects_complex_amplitudes(tmp_path):
    with pytest.raises(ValueError, match="intensity data"):
        TACAWData(_fake_wf_data(tmp_path), keep_complex=True, fold=True)


def test_gain_loss_fold_averages_q_omega_inversion_pairs(tmp_path):
    tacaw = TACAWData(_fake_wf_data(tmp_path))
    rng = np.random.default_rng(456)
    original = rng.random(tacaw.array.shape)
    tacaw._array = original.copy()

    tacaw.fold_gain_loss()

    kx_indices = np.array([np.argmin(np.abs(tacaw.kxs + value))
                           for value in tacaw.kxs])
    ky_indices = np.array([np.argmin(np.abs(tacaw.kys + value))
                           for value in tacaw.kys])
    for positive_index in np.flatnonzero(tacaw.frequencies > 0):
        frequency = tacaw.frequencies[positive_index]
        negative_index = int(np.argmin(np.abs(tacaw.frequencies + frequency)))
        negative = tacaw.array[:, negative_index, :, :]
        positive_at_minus_q = tacaw.array[:, positive_index, :, :]
        positive_at_minus_q = positive_at_minus_q[:, kx_indices, :][:, :, ky_indices]
        np.testing.assert_allclose(negative, positive_at_minus_q)

    assert np.sum(tacaw.array) == pytest.approx(np.sum(original))
    assert tacaw.gain_loss_folded is True


def test_bose_corrected_gain_side_obeys_detailed_balance(tmp_path):
    temperature = 300.0
    tacaw = TACAWData(
        _fake_wf_data(tmp_path),
        temperature_K=temperature,
        apply_bose=True,
        fold=True,
    )
    positive_index = int(np.flatnonzero(tacaw.frequencies > 0)[0])
    frequency = tacaw.frequencies[positive_index]
    negative_index = int(np.argmin(np.abs(tacaw.frequencies + frequency)))
    kx_indices = np.array([np.argmin(np.abs(tacaw.kxs + value))
                           for value in tacaw.kxs])
    ky_indices = np.array([np.argmin(np.abs(tacaw.kys + value))
                           for value in tacaw.kys])
    loss_at_minus_q = tacaw.array[:, positive_index, :, :]
    loss_at_minus_q = loss_at_minus_q[:, kx_indices, :][:, :, ky_indices]
    beta_e = frequency * 4.135667696e-3 / (8.617333262145e-5 * temperature)

    np.testing.assert_allclose(
        tacaw.array[:, negative_index, :, :],
        np.exp(-beta_e) * loss_at_minus_q,
        rtol=1e-10,
        atol=1e-10,
    )


def test_bose_correction_requires_intensity_data(tmp_path):
    tacaw = TACAWData(_fake_wf_data(tmp_path), keep_complex=True)

    with pytest.raises(ValueError, match="intensity data"):
        tacaw.apply_bose_correction(300.0)


def test_masked_spectrum_accepts_torch_backed_tacaw_array(tmp_path):
    torch = pytest.importorskip("torch")

    tacaw = TACAWData(_fake_wf_data(tmp_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float64
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    tacaw._backend = TorchBackend(str(device))
    tacaw._array = torch.as_tensor(tacaw.array, dtype=dtype, device=device)
    tacaw.fold_gain_loss()

    spectrum = tacaw.masked_spectrum(
        {"shape": "round", "center": (0.0, 0.0), "radius": 1.25},
        probe_index=None,
    )

    assert spectrum.shape == (len(tacaw.frequencies),)
    assert np.all(np.isfinite(spectrum))


def test_gain_loss_fold_handles_even_fft_nyquist_bins(tmp_path):
    rng = np.random.default_rng(789)
    array = (
        rng.normal(size=(2, 8, 4, 4, 1))
        + 1j * rng.normal(size=(2, 8, 4, 4, 1))
    ).astype(np.complex128)
    tacaw = TACAWData(_fake_wf_data(tmp_path, array=array))
    tacaw._kxs = np.fft.fftshift(np.fft.fftfreq(4, d=0.25))
    tacaw._kys = np.fft.fftshift(np.fft.fftfreq(4, d=0.5))
    original_sum = np.sum(tacaw.array)

    tacaw.fold_gain_loss()

    assert np.sum(tacaw.array) == pytest.approx(original_sum)
    first_result = tacaw.array.copy()
    tacaw.fold_gain_loss()
    np.testing.assert_array_equal(tacaw.array, first_result)


def test_gain_loss_fold_rejects_asymmetric_momentum_grid(tmp_path):
    tacaw = TACAWData(_fake_wf_data(tmp_path))
    tacaw._kxs = np.array([-1.0, 0.0, 2.0])

    with pytest.raises(ValueError, match="not closed under inversion"):
        tacaw.fold_gain_loss()


def test_dispersion_returns_all_frequency_bins(tmp_path):
    tacaw = TACAWData(_fake_wf_data(tmp_path))
    kx_path = np.linspace(-0.5, 0.5, 5)
    ky_path = np.zeros_like(kx_path)

    dispersion = tacaw.dispersion(kx_path, ky_path)

    assert dispersion.shape == (len(tacaw.frequencies), len(kx_path))


def test_plot_with_omega_axis_uses_frequency_coordinates(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    tacaw = TACAWData(_fake_wf_data(tmp_path))
    kx_path = np.linspace(-0.5, 0.5, 5)
    ky_path = np.zeros_like(kx_path)
    dispersion = tacaw.dispersion(kx_path, ky_path)

    tacaw.plot(
        dispersion,
        kx_path,
        "omega",
        filename=tmp_path / "dispersion.png",
    )

    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.get_ylabel() == "frequency (THz)"
    assert len(ax.collections) == 0
    assert len(ax.images) == 1
    assert ax.get_ylim() == pytest.approx((np.min(tacaw.frequencies), np.max(tacaw.frequencies)))
    assert ax.images[0].origin == "lower"
    close_figure(fig)


def test_plot_with_omega_axis_renders_positive_rows_above_zero(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    tacaw = TACAWData(_fake_wf_data(tmp_path))
    frequencies = np.asarray(tacaw.frequencies)
    kx_path = np.linspace(0.0, 1.0, 5)
    intensities = np.zeros((len(frequencies), len(kx_path)))

    positive_idx = np.flatnonzero(frequencies > 0)[0]
    negative_idx = np.flatnonzero(frequencies < 0)[-1]
    intensities[positive_idx, 2] = 1.0
    intensities[negative_idx, 2] = 0.2

    tacaw.plot(
        intensities,
        kx_path,
        "omega",
        filename=tmp_path / "orientation.png",
    )

    fig = plt.gcf()
    ax = fig.axes[0]
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    height = buffer.shape[0]

    def pixel_intensity(x, y):
        pixel_x, pixel_y = ax.transData.transform((x, y))
        row = height - int(round(pixel_y))
        col = int(round(pixel_x))
        patch = buffer[row - 2:row + 3, col - 2:col + 3, :3]
        return patch.mean()

    assert pixel_intensity(kx_path[2], frequencies[positive_idx]) > pixel_intensity(
        kx_path[2],
        frequencies[negative_idx],
    )
    assert ax.transData.transform((kx_path[2], frequencies[positive_idx]))[1] > ax.transData.transform(
        (kx_path[2], 0.0)
    )[1]
    close_figure(fig)


def test_tacaw_cache_manifest_separates_intensity_and_complex_data(tmp_path):
    wf = _fake_wf_data(tmp_path)
    intensity = TACAWData(wf, keep_complex=False)
    complex_amplitude = TACAWData(wf, keep_complex=True)

    assert not np.iscomplexobj(intensity.array)
    assert np.iscomplexobj(complex_amplitude.array)

    import json
    manifest = json.loads((tmp_path / "tacaw_manifest.json").read_text())
    assert manifest["keep_complex"] is True


def test_tacaw_constructor_reuses_compatible_cache(tmp_path, monkeypatch):
    wf = _fake_wf_data(tmp_path)
    first = TACAWData(wf)

    def fail_if_fft_runs(*_args, **_kwargs):
        raise AssertionError("compatible TACAW cache should bypass FFT")

    monkeypatch.setattr(wf._backend, "fft", fail_if_fft_runs)
    second = TACAWData(wf)

    assert np.array_equal(second.array, first.array)


def test_nearest_frequency_and_rectangular_spectrum_image(tmp_path):
    array = np.ones((6, 8, 3, 4, 1), dtype=np.complex128)
    backend = NumpyBackend()
    wf = WFData(
        probe_positions=[(x, y) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)],
        probe_xs=[0.0, 1.0, 2.0],
        probe_ys=[0.0, 1.0],
        time=np.arange(8) * 0.005,
        kxs=np.linspace(-1.0, 1.0, 3),
        kys=np.linspace(-1.5, 1.5, 4),
        xs=np.linspace(0.0, 2.0, 3),
        ys=np.linspace(0.0, 3.0, 4),
        layer=np.array([0]),
        array=array,
        probe=SimpleNamespace(eV=100e3, wavelength=0.037, mrad=0.0),
        cache_dir=tmp_path,
        backend=backend,
    )
    tacaw = TACAWData(wf, force_rerun=True)

    assert tacaw.nearest_frequency(13.0) in tacaw.frequencies
    assert tacaw.spectrum_image_reshaped(0.0).shape == (3, 2)


def test_plot_transposes_named_xy_pattern_and_crops_extent(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    tacaw = TACAWData(_fake_wf_data(tmp_path))
    values = np.arange(len(tacaw.kxs) * len(tacaw.kys)).reshape(
        len(tacaw.kxs), len(tacaw.kys)
    )

    tacaw.plot(
        values,
        "kx",
        "ky",
        extent=(-0.1, 1.1, -0.6, 1.6),
        filename=tmp_path / "cropped.png",
    )

    fig = plt.gcf()
    image = fig.axes[0].images[0]
    rendered = np.asarray(image.get_array())
    assert image.origin == "lower"
    assert rendered.shape == (3, 2)
    np.testing.assert_array_equal(rendered, values[1:, 1:].T)
    close_figure(fig)


def test_bose_correction_rejects_invalid_or_repeated_application(tmp_path):
    tacaw = TACAWData(_fake_wf_data(tmp_path))

    with pytest.raises(ValueError, match="positive"):
        tacaw.apply_bose_correction(0.0)

    tacaw.apply_bose_correction(300.0)
    with pytest.raises(ValueError, match="already"):
        tacaw.apply_bose_correction(300.0)


def test_mask_and_dispersion_validate_scientific_coordinates(tmp_path):
    tacaw = TACAWData(_fake_wf_data(tmp_path))

    with pytest.raises(ValueError, match="shape must be 'round'"):
        tacaw.masked_spectrum({"shape": "square", "radius": 1.0})
    with pytest.raises(ValueError, match="outside"):
        tacaw.dispersion(np.array([99.0]), np.array([0.0]))

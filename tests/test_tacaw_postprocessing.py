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


def test_mask_and_dispersion_validate_scientific_coordinates(tmp_path):
    tacaw = TACAWData(_fake_wf_data(tmp_path))

    with pytest.raises(ValueError, match="shape must be 'round'"):
        tacaw.masked_spectrum({"shape": "square", "radius": 1.0})
    with pytest.raises(ValueError, match="outside"):
        tacaw.dispersion(np.array([99.0]), np.array([0.0]))

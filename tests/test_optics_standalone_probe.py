"""Standalone source-to-detector wave propagation without a specimen."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyslice import simulate_raytem_wave
from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, TorchBackend
from pyslice.data import Dimension, Dimensions, Metadata, SEA_ECO_AVAILABLE
from pyslice.optics.column import OpticalColumn
from pyslice.optics.elements import (
    Aperture,
    BeamTilt,
    FreeSpace,
    Lens,
)
from pyslice.postprocessing.wf_data import WFData


def _wave_moments(wf, which_probe=0):
    reciprocal = wf.array[which_probe, 0, :, :, 0]
    real = np.fft.ifft2(np.fft.ifftshift(reciprocal))
    intensity = np.abs(real) ** 2
    intensity /= intensity.sum()
    x_grid, y_grid = np.meshgrid(wf.xs, wf.ys, indexing="ij")

    power = np.abs(reciprocal) ** 2
    power /= power.sum()
    kx_grid, ky_grid = np.meshgrid(wf.kxs, wf.kys, indexing="ij")
    wavelength_A = wf.probe.wavelength
    return np.array(
        [
            np.sum(x_grid * intensity),
            np.sum(y_grid * intensity),
            wavelength_A * np.sum(kx_grid * power),
            wavelength_A * np.sum(ky_grid * power),
        ]
    )


def _physical_norm(wf):
    reciprocal = np.asarray(wf.array[0, 0, :, :, 0])
    real = np.fft.ifft2(np.fft.ifftshift(reciprocal))
    dx = float(wf.xs[1] - wf.xs[0])
    dy = float(wf.ys[1] - wf.ys[0])
    return float(np.sum(np.abs(real) ** 2) * dx * dy)


def test_standalone_probe_has_requested_positions():
    wf = WFData.from_probe(
        extent_A=(128.0, 96.0),
        sampling=0.5,
        voltage_eV=100e3,
        aperture=5.0,
        probe_positions=[(5.0, -3.0), (-4.0, 2.0)],
        backend=NumpyBackend(),
    )

    assert wf.array.shape == (2, 1, 256, 192, 1)
    assert wf.probe_positions == [(5.0, -3.0), (-4.0, 2.0)]
    np.testing.assert_allclose(
        _wave_moments(wf, which_probe=0)[:2], [5.0, -3.0], atol=0.15, rtol=0
    )
    np.testing.assert_allclose(
        _wave_moments(wf, which_probe=1)[:2], [-4.0, 2.0], atol=0.15, rtol=0
    )


def test_standalone_probe_rounds_odd_grid_sizes_to_a_centered_even_grid():
    wf = WFData.from_probe(
        extent_A=31.0,
        sampling=1.0,
        voltage_eV=100e3,
        aperture=20.0,
        backend=NumpyBackend(),
    )

    assert wf.array.shape == (1, 1, 32, 32, 1)
    reciprocal = wf.array[0, 0, :, :, 0]
    intensity = np.abs(np.fft.ifft2(np.fft.ifftshift(reciprocal))) ** 2
    peak = np.unravel_index(np.argmax(intensity), intensity.shape)
    assert wf.xs[peak[0]] == pytest.approx(0.0)
    assert wf.ys[peak[1]] == pytest.approx(0.0)
    np.testing.assert_allclose(_wave_moments(wf)[:2], [0.0, 0.0], atol=0.05)


def test_standalone_probe_defocus_matches_free_space_propagation():
    focused = WFData.from_probe(
        extent_A=64.0,
        sampling=0.5,
        voltage_eV=100e3,
        aperture=5.0,
        backend=NumpyBackend(),
    )
    defocused = WFData.from_probe(
        extent_A=64.0,
        sampling=0.5,
        voltage_eV=100e3,
        aperture=5.0,
        defocus=125.0,
        backend=NumpyBackend(),
    )

    focused.propagate_free_space(125.0)

    np.testing.assert_allclose(defocused.array, focused.array, atol=1e-12)


def test_real_space_aperture_updates_wave_and_uses_optical_axis():
    wave = WFData.from_probe(
        extent_A=32.0,
        sampling=0.25,
        voltage_eV=100e3,
        aperture=5.0,
        probe_positions=[(3.0, 0.0)],
        backend=NumpyBackend(),
    )
    before = np.fft.ifft2(
        np.fft.ifftshift(wave.array[0, 0, :, :, 0])
    )
    x_grid, y_grid = np.meshgrid(wave.xs, wave.ys, indexing="ij")
    outside = np.hypot(x_grid, y_grid) >= 5.0

    Aperture(5.0, space="real").apply(wave)

    after = np.fft.ifft2(
        np.fft.ifftshift(wave.array[0, 0, :, :, 0])
    )
    assert np.sum(np.abs(after) ** 2) < np.sum(np.abs(before) ** 2)
    np.testing.assert_allclose(after[outside], 0.0, atol=1e-14)
    np.testing.assert_allclose(after[~outside], before[~outside], atol=1e-14)


def test_sampling_report_describes_grid_norm_and_nyquist_angle():
    wf = WFData.from_probe(
        extent_A=64.0,
        sampling=0.5,
        voltage_eV=100e3,
        aperture=5.0,
        backend=NumpyBackend(),
    )

    report = wf.sampling_report()

    assert report["shape"] == (128, 128)
    np.testing.assert_allclose(report["sampling_A"], (0.5, 0.5))
    np.testing.assert_allclose(report["extent_A"], (64.0, 64.0))
    assert report["physical_norm"] == pytest.approx(_physical_norm(wf))
    expected_nyquist_mrad = float(wf.probe.wavelength) * 1e3
    np.testing.assert_allclose(
        report["theta_nyquist_mrad"],
        (expected_nyquist_mrad, expected_nyquist_mrad),
    )
    assert report["reciprocal_edge_power_fraction"] == pytest.approx(0.0)


def test_sampling_report_uses_worst_wave_instead_of_ensemble_average():
    probe_count = 200
    wf = WFData.from_probe(
        extent_A=32.0,
        sampling=1.0,
        voltage_eV=100e3,
        aperture=0.0,
        probe_positions=[(0.0, 0.0)] * probe_count,
        backend=NumpyBackend(),
    )
    real = np.zeros((probe_count, 1, 32, 32, 1), dtype=complex)
    real[:-1, 0, 16, 16, 0] = 1.0
    real[-1, 0, 0, 16, 0] = 1.0
    wf._array = np.fft.fftshift(
        np.fft.fft2(real, axes=(2, 3)), axes=(2, 3)
    )

    report = wf.sampling_report()

    # An ensemble-wide reduction would dilute the clipped wave to 1 / 200.
    assert report["real_edge_power_fraction"] == pytest.approx(1.0)

    reciprocal = np.zeros_like(real)
    reciprocal[:-1, 0, 16, 16, 0] = 1.0
    reciprocal[-1, 0, 0, 16, 0] = 1.0
    wf._array = reciprocal

    report = wf.sampling_report()

    assert report["reciprocal_edge_power_fraction"] == pytest.approx(1.0)


def test_real_space_padding_preserves_wave_coordinates_and_norm():
    wf = WFData.from_probe(
        extent_A=(32.0, 24.0),
        sampling=0.5,
        voltage_eV=100e3,
        aperture=5.0,
        probe_positions=[(2.0, -1.0)],
        backend=NumpyBackend(),
    )
    before_real = np.fft.ifft2(
        np.fft.ifftshift(wf.array[0, 0, :, :, 0])
    )
    before_norm = _physical_norm(wf)

    # Odd pixel margins exercise the shifted-spectrum convention; omitting the
    # FFT shifts introduces an otherwise easy-to-miss checkerboard phase here.
    report = wf.pad_real_space(3.5, 2.5)

    assert report["old_shape"] == (64, 48)
    assert report["new_shape"] == (78, 58)
    np.testing.assert_allclose(report["actual_padding_A"], (3.5, 2.5))
    np.testing.assert_allclose(report["after"]["sampling_A"], (0.5, 0.5))
    np.testing.assert_allclose(report["after"]["extent_A"], (39.0, 29.0))
    assert _physical_norm(wf) == pytest.approx(before_norm, rel=1e-12)
    after_real = np.fft.ifft2(
        np.fft.ifftshift(wf.array[0, 0, :, :, 0])
    )
    np.testing.assert_allclose(after_real[7:71, 5:53], before_real, atol=1e-14)
    np.testing.assert_allclose(after_real[:7], 0.0, atol=1e-14)
    assert wf.xs[wf.array.shape[2] // 2] == pytest.approx(0.0)
    assert wf.ys[wf.array.shape[3] // 2] == pytest.approx(0.0)


def test_raytem_column_propagates_probe_without_sample_and_records_planes():
    raytem_json = {
        "Sections": [
            {
                "Section name": "column",
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "gun",
                        "kind": "Source",
                        "position": 0.0,
                        "length": 0.0,
                        "size": [2e-4, 2e-4],
                        "angle": [0.0, 0.0],
                    },
                    {
                        "Element name": "L1",
                        "kind": "Thin lens",
                        "position": 1e-4,
                        "length": 0.0,
                        "strength": 50.0,
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 5e-4,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }
    column = OpticalColumn.from_raytem(raytem_json, start="gun", stop="screen")
    wf = WFData.from_probe(
        extent_A=128.0,
        sampling=0.5,
        voltage_eV=100e3,
        aperture=5.0,
        backend=NumpyBackend(),
    )
    source_array = wf.array.copy()

    propagation = column.propagate(wf, record=True)

    assert propagation.output is wf
    assert [plane.name for plane in propagation.planes] == [
        "source",
        "to L1",
        "L1",
        "to stop",
    ]
    assert [plane.z_A for plane in propagation.planes] == [
        0.0,
        1_000.0,
        1_000.0,
        5_000.0,
    ]
    np.testing.assert_array_equal(propagation.planes[0].wave.array, source_array)
    assert not np.array_equal(propagation.output.array, source_array)


def test_simulate_raytem_wave_is_the_complete_standalone_api(monkeypatch):
    monkeypatch.setenv("PYSLICE_BACKEND", "numpy")
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "gun",
                        "kind": "Source",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "L1",
                        "kind": "Thin lens",
                        "position": 1e-5,
                        "length": 0.0,
                        "strength": 100.0,
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 5e-5,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }

    result = simulate_raytem_wave(
        raytem_json,
        extent_A=64.0,
        sampling_A=0.5,
        voltage_eV=100e3,
        convergence_mrad=4.0,
        start="gun",
        stop="screen",
    )

    assert result.column is not None
    assert result.output.array.shape == (1, 1, 128, 128, 1)
    assert result.plane("L1").z_A == pytest.approx(100.0)
    assert result.plane("to stop").z_A == pytest.approx(500.0)
    with pytest.raises(KeyError, match="Available"):
        result.plane("missing")


def test_optical_column_transforms_every_stored_wave_layer_consistently():
    wave = WFData.from_probe(
        extent_A=64.0,
        sampling=0.5,
        voltage_eV=100e3,
        aperture=4.0,
        probe_positions=[(3.0, -2.0)],
        backend=NumpyBackend(),
    )
    wave._array = np.repeat(wave.array, 2, axis=4)
    wave._layer = np.array([0, 1])

    OpticalColumn(
        [
            FreeSpace(80.0),
            Lens(900.0, rotation_rad=np.deg2rad(12.0)),
            BeamTilt(0.4e-3, -0.2e-3),
        ]
    ).apply(wave)

    np.testing.assert_allclose(
        wave.array[:, :, :, :, 0],
        wave.array[:, :, :, :, 1],
        atol=1e-12,
    )


def test_recorded_planes_own_independent_dimensions_and_metadata():
    wave = WFData.from_probe(
        extent_A=16.0,
        sampling=1.0,
        voltage_eV=100e3,
        backend=NumpyBackend(),
    )
    if SEA_ECO_AVAILABLE:
        wave.dimensions = Dimensions([
            Dimension(name="kx", space="scattering", units="Å⁻¹", values=wave.kxs.copy()),
            Dimension(name="ky", space="scattering", units="Å⁻¹", values=wave.kys.copy()),
        ], det_dimensions=[0, 1])
        wave.metadata = Metadata({"nested": {"label": "source"}})
    else:
        wave.dimensions = {
            "kx": SimpleNamespace(values=wave.kxs.copy()),
            "ky": SimpleNamespace(values=wave.kys.copy()),
        }
        wave.metadata = {"nested": {"label": "source"}}

    propagation = OpticalColumn([FreeSpace(1.0)]).propagate(wave, record=True)
    source = propagation.planes[0].wave
    propagated = propagation.planes[1].wave

    assert source.dimensions is not propagated.dimensions
    propagated.dimensions["kx"].values = np.arange(20)
    assert len(source.dimensions["kx"].values) == 16
    if SEA_ECO_AVAILABLE:
        propagated.metadata.nested.label = "propagated"
        assert source.metadata.nested.label == "source"
    else:
        propagated.metadata["nested"]["label"] = "propagated"
        assert source.metadata["nested"]["label"] == "source"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch is optional")
def test_recorded_wave_planes_support_torch_backend():
    wave = WFData.from_probe(
        extent_A=64.0,
        sampling=1.0,
        voltage_eV=100e3,
        aperture=1.0,
        backend=TorchBackend(device="cpu"),
    )

    propagation = OpticalColumn([FreeSpace(100.0)]).propagate(wave, record=True)

    assert len(propagation.planes) == 2
    assert propagation.planes[0].wave.array.device.type == "cpu"
    assert propagation.planes[0].wave.array.data_ptr() != wave.array.data_ptr()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch is optional")
def test_real_space_padding_stays_on_torch_device_and_preserves_norm():
    wave = WFData.from_probe(
        extent_A=32.0,
        sampling=0.5,
        voltage_eV=100e3,
        aperture=5.0,
        backend=TorchBackend(device="cpu"),
    )
    before_norm = wave.sampling_report()["physical_norm"]

    report = wave.pad_real_space(2.0, 3.0)

    assert wave.array.device.type == "cpu"
    assert wave._xs.device.type == "cpu"
    assert wave.array.shape == (1, 1, 72, 76, 1)
    assert report["after"]["physical_norm"] == pytest.approx(
        before_norm, rel=1e-12
    )

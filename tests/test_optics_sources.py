"""Tests for the deliberately small coherent Gaussian source."""

import numpy as np
import pytest

from pyslice import GaussianWaveSource, simulate_raytem_wave
from pyslice.backend import NumpyBackend


def _moments(values, weights):
    weights = weights / np.sum(weights)
    mean = np.sum(values * weights)
    variance = np.sum((values - mean) ** 2 * weights)
    return mean, variance


def test_gaussian_source_has_requested_size_center_tilt_and_curvature():
    source = GaussianWaveSource(
        voltage_eV=100e3,
        rms_size_A=(20.0, 15.0),
        curvature_inv_A=(2e-4, -1e-4),
        center_A=(3.0, -2.0),
        tilt_mrad=(0.05, -0.08),
    )
    wave = source.to_wave(
        extent_A=160.0,
        sampling_A=1.0,
        backend=NumpyBackend(),
    )
    reciprocal = wave.array[0, 0, :, :, 0]
    real = np.fft.ifft2(np.fft.ifftshift(reciprocal))
    power = np.abs(real) ** 2
    mean_x, variance_x = _moments(wave.xs, np.sum(power, axis=1))
    mean_y, variance_y = _moments(wave.ys, np.sum(power, axis=0))
    reciprocal_power = np.abs(reciprocal) ** 2
    mean_kx, variance_kx = _moments(
        wave.kxs, np.sum(reciprocal_power, axis=1)
    )
    mean_ky, variance_ky = _moments(
        wave.kys, np.sum(reciprocal_power, axis=0)
    )

    assert wave.array.shape[0] == 1
    assert mean_x == pytest.approx(3.0, abs=0.05)
    assert mean_y == pytest.approx(-2.0, abs=0.05)
    assert np.sqrt(variance_x) == pytest.approx(20.0, rel=2e-3)
    assert np.sqrt(variance_y) == pytest.approx(15.0, rel=2e-3)
    assert source.wavelength_A * mean_kx * 1e3 == pytest.approx(0.05, abs=0.01)
    assert source.wavelength_A * mean_ky * 1e3 == pytest.approx(-0.08, abs=0.01)
    expected_theta_x = np.hypot(
        source.wavelength_A / (4.0 * np.pi * 20.0), 2e-4 * 20.0
    )
    expected_theta_y = np.hypot(
        source.wavelength_A / (4.0 * np.pi * 15.0), 1e-4 * 15.0
    )
    assert source.wavelength_A * np.sqrt(variance_kx) == pytest.approx(
        expected_theta_x, rel=2e-3
    )
    assert source.wavelength_A * np.sqrt(variance_ky) == pytest.approx(
        expected_theta_y, rel=2e-3
    )
    assert wave.sampling_report()["physical_norm"] == pytest.approx(1.0)


def test_gaussian_source_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="voltage_eV"):
        GaussianWaveSource(voltage_eV=0.0, rms_size_A=10.0)
    with pytest.raises(ValueError, match="rms_size_A"):
        GaussianWaveSource(voltage_eV=100e3, rms_size_A=(10.0, 0.0))
    with pytest.raises(ValueError, match="curvature_inv_A"):
        GaussianWaveSource(
            voltage_eV=100e3,
            rms_size_A=10.0,
            curvature_inv_A=float("nan"),
        )


def test_high_level_raytem_simulation_accepts_explicit_gaussian_source():
    config = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "start",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 1e-5,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }
    source = GaussianWaveSource(
        voltage_eV=100e3,
        rms_size_A=10.0,
        curvature_inv_A=2e-4,
    )

    result = simulate_raytem_wave(
        config,
        source=source,
        extent_A=96.0,
        sampling_A=1.0,
        start="start",
        stop="screen",
        record=False,
    )

    assert result.output.array.shape[0] == 1
    assert result.output.probe.eV == pytest.approx(100e3)
    with pytest.raises(ValueError, match="source cannot be combined"):
        simulate_raytem_wave(
            config,
            source=source,
            voltage_eV=100e3,
            extent_A=96.0,
            sampling_A=1.0,
        )

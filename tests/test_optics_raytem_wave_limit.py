"""Cross-check PySlice wave optics against RayTEM's paraxial ray limit.

These tests are optional because RayTEM is not a core PySlice dependency. When
RayTEM is importable, they propagate a deterministic Gaussian ray ensemble and
the matching minimum-uncertainty Gaussian wavepacket through the same column.
The comparison covers the first two moments at every optical plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.hermite import hermgauss
import pytest

try:
    from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection
    from pySEA.rayTEM.elements import (
        Dipole as RayTEMDipole,
        Drift,
        Lens as RayTEMLens,
        Prism as RayTEMPrism,
        Quadrapole as RayTEMQuadrupole,
        columnByName,
        fix_ray_dims,
    )
except ImportError:
    RAYTEM_AVAILABLE = False
else:
    RAYTEM_AVAILABLE = True

if os.environ.get("PYSLICE_REQUIRE_RAYTEM") == "1" and not RAYTEM_AVAILABLE:
    raise RuntimeError(
        "PYSLICE_REQUIRE_RAYTEM=1, but pySEA.rayTEM could not be imported."
    )

pytestmark = pytest.mark.skipif(
    not RAYTEM_AVAILABLE,
    reason="RayTEM is optional for PySlice",
)

from pyslice.backend import NumpyBackend
from pyslice.multislice.multislice import wavelength
from pyslice.optics.raytem import (
    RAYTEM_MM_TO_ANGSTROM,
    optical_column_from_raytem,
)
from pyslice.postprocessing.wf_data import WFData


@dataclass(frozen=True)
class ValidationCase:
    """Parameters for one paraxial ray-versus-wave comparison."""

    energy_eV: float
    x0_A: float
    theta0_rad: float
    sigma_x_A: float
    lens_length_mm: float
    lens_strength_per_mm: float


_BACKEND = NumpyBackend()
_DX_A = 0.5
_NX = 1024
_NY = 128
_XS = (np.arange(_NX) - _NX // 2) * _DX_A
_YS = (np.arange(_NY) - _NY // 2) * _DX_A
_KXS = np.fft.fftshift(np.fft.fftfreq(_NX, d=_DX_A))
_KYS = np.fft.fftshift(np.fft.fftfreq(_NY, d=_DX_A))
_X_GRID, _Y_GRID = np.meshgrid(_XS, _YS, indexing="ij")
_D1_A = 2_000.0
_D2_A = 3_000.0
_D1_MM = _D1_A / RAYTEM_MM_TO_ANGSTROM
_D2_MM = _D2_A / RAYTEM_MM_TO_ANGSTROM
_SIGMA_Y_A = 4.0


def _gaussian_ray_ensemble(case: ValidationCase, wavelength_A: float, order: int = 8):
    """Return Gauss-Hermite rays and weights matching a Gaussian wavepacket."""
    nodes, weights = hermgauss(order)
    sigma_theta_x = wavelength_A / (4.0 * np.pi * case.sigma_x_A)
    sigma_theta_y = wavelength_A / (4.0 * np.pi * _SIGMA_Y_A)
    xs = (
        case.x0_A + np.sqrt(2.0) * case.sigma_x_A * nodes
    ) / RAYTEM_MM_TO_ANGSTROM
    theta_xs = case.theta0_rad + np.sqrt(2.0) * sigma_theta_x * nodes
    ys = np.sqrt(2.0) * _SIGMA_Y_A * nodes / RAYTEM_MM_TO_ANGSTROM
    theta_ys = np.sqrt(2.0) * sigma_theta_y * nodes
    x_grid, theta_x_grid, y_grid, theta_y_grid = np.meshgrid(
        xs, theta_xs, ys, theta_ys, indexing="ij"
    )
    quadrature_weights = (
        weights[:, None, None, None]
        * weights[None, :, None, None]
        * weights[None, None, :, None]
        * weights[None, None, None, :]
    ).ravel() / np.pi**2
    rays = fix_ray_dims(
        np.column_stack(
            [
                x_grid.ravel(),
                theta_x_grid.ravel(),
                y_grid.ravel(),
                theta_y_grid.ravel(),
            ]
        ),
        ["x", "xt", "y", "yt"],
    )
    return rays, quadrature_weights


def _ray_moments(rays, weights):
    """Return position/angle means and RMS widths in both axes."""
    moments = []
    for position_name, angle_name in (("x", "xt"), ("y", "yt")):
        position = rays[:, columnByName(position_name)]
        angle = rays[:, columnByName(angle_name)]
        mean_position = np.sum(weights * position)
        mean_angle = np.sum(weights * angle)
        sigma_position = np.sqrt(np.sum(weights * (position - mean_position) ** 2))
        sigma_angle = np.sqrt(np.sum(weights * (angle - mean_angle) ** 2))
        moments.extend(
            [
                mean_position * RAYTEM_MM_TO_ANGSTROM,
                mean_angle,
                sigma_position * RAYTEM_MM_TO_ANGSTROM,
                sigma_angle,
            ]
        )
    return np.asarray(moments)


def _gaussian_wfdata(case: ValidationCase, wavelength_A: float) -> WFData:
    """Build a normalized Gaussian packet in PySlice's shifted-k convention."""
    psi = np.exp(
        -((_X_GRID - case.x0_A) ** 2) / (4.0 * case.sigma_x_A**2)
        - _Y_GRID**2 / (4.0 * _SIGMA_Y_A**2)
    )
    psi = psi * np.exp(
        2j * np.pi * (case.theta0_rad / wavelength_A) * _X_GRID
    )
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    reciprocal = np.fft.fftshift(np.fft.fft2(psi))
    probe = SimpleNamespace(
        wavelength=wavelength_A,
        eV=case.energy_eV,
        mrad=0.0,
    )
    return WFData(
        probe_positions=[(0.0, 0.0)],
        probe_xs=[0.0],
        probe_ys=[0.0],
        time=np.array([0.0]),
        kxs=_KXS.copy(),
        kys=_KYS.copy(),
        xs=_XS.copy(),
        ys=_YS.copy(),
        layer=np.array([0]),
        array=reciprocal[None, None, :, :, None],
        probe=probe,
        backend=_BACKEND,
    )


def _wave_moments(wf: WFData, wavelength_A: float):
    """Return position/angle means and RMS widths in both axes."""
    reciprocal = wf.array[0, 0, :, :, 0]
    psi = np.fft.ifft2(np.fft.ifftshift(reciprocal))
    intensity = np.abs(psi) ** 2
    intensity_sum = np.sum(intensity)
    power = np.abs(reciprocal) ** 2
    power_sum = np.sum(power)
    moments = []
    for coordinates, frequencies, coordinate_grid, frequency_grid in (
        (_XS, _KXS, _XS[:, None], _KXS[:, None]),
        (_YS, _KYS, _YS[None, :], _KYS[None, :]),
    ):
        mean_position = np.sum(coordinate_grid * intensity) / intensity_sum
        sigma_position = np.sqrt(
            np.sum((coordinate_grid - mean_position) ** 2 * intensity) / intensity_sum
        )
        mean_frequency = np.sum(frequency_grid * power) / power_sum
        sigma_frequency = np.sqrt(
            np.sum((frequency_grid - mean_frequency) ** 2 * power) / power_sum
        )
        moments.extend(
            [
                mean_position,
                wavelength_A * mean_frequency,
                sigma_position,
                wavelength_A * sigma_frequency,
            ]
        )
    return np.asarray(moments)


def _rotate_ray_coordinates(rays, angle_rad):
    """Apply RayTEM's separately tracked Larmor rotation to physical rays."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    for first, second in (("x", "y"), ("xt", "yt")):
        a = rays[:, columnByName(first)].copy()
        b = rays[:, columnByName(second)].copy()
        rays[:, columnByName(first)] = cosine * a - sine * b
        rays[:, columnByName(second)] = sine * a + cosine * b
    return rays


def _run_case(case: ValidationCase):
    """Propagate matching RayTEM rays and a PySlice packet at every plane."""
    wavelength_A = wavelength(case.energy_eV)
    ray_elements = [
        Drift(length=_D1_MM),
        RayTEMLens(
            name="L1",
            length=case.lens_length_mm,
            strength=case.lens_strength_per_mm,
        ),
        Drift(length=_D2_MM),
    ]
    rays, weights = _gaussian_ray_ensemble(case, wavelength_A)
    ray_history = [_ray_moments(rays, weights)]
    for element in ray_elements:
        rays = element.propagate_ray(rays)
        if isinstance(element, RayTEMLens) and element.rotation:
            rays = _rotate_ray_coordinates(rays, element.rotation)
        ray_history.append(_ray_moments(rays, weights))

    section = MicroscopeSection(
        name="validation",
        elements=[
            Drift(name="start", length=0.0, position=0.0),
            RayTEMLens(
                name="L1",
                length=case.lens_length_mm,
                strength=case.lens_strength_per_mm,
                position=_D1_MM,
            ),
            Drift(
                name="screen",
                length=0.0,
                position=_D1_MM + case.lens_length_mm + _D2_MM,
            ),
        ],
    )
    column = optical_column_from_raytem(
        Microscope(name="validation", sections=[section]),
        start="start",
        stop="screen",
    )
    assert [type(element).__name__ for element in column.elements] == [
        "FreeSpace",
        "Lens",
        "FreeSpace",
    ]

    wf = _gaussian_wfdata(case, wavelength_A)
    wave_history = [_wave_moments(wf, wavelength_A)]
    for element in column.elements:
        element.apply(wf)
        wave_history.append(_wave_moments(wf, wavelength_A))

    return np.asarray(ray_history), np.asarray(wave_history)


@pytest.mark.parametrize(
    "case",
    [
        ValidationCase(
            60e3,
            -10.0,
            -1.2e-3,
            12.0,
            0.0,
            math.sqrt(RAYTEM_MM_TO_ANGSTROM / 4_000.0),
        ),
        ValidationCase(
            100e3,
            5.0,
            1.0e-3,
            20.0,
            0.0,
            math.sqrt(RAYTEM_MM_TO_ANGSTROM / 5_000.0),
        ),
        ValidationCase(
            200e3,
            15.0,
            -0.75e-3,
            30.0,
            0.0,
            math.sqrt(RAYTEM_MM_TO_ANGSTROM / 8_000.0),
        ),
    ],
)
def test_thin_lens_wavepacket_moments_follow_raytem(case):
    """Thin-lens wave moments reproduce RayTEM at every optical plane."""
    ray, wave = _run_case(case)
    np.testing.assert_allclose(wave[:, [0, 4]], ray[:, [0, 4]], atol=2e-6, rtol=0)
    np.testing.assert_allclose(wave[:, [1, 5]], ray[:, [1, 5]], atol=1e-10, rtol=0)
    np.testing.assert_allclose(wave[:, [2, 6]], ray[:, [2, 6]], atol=2e-6, rtol=0)
    np.testing.assert_allclose(wave[:, [3, 7]], ray[:, [3, 7]], atol=5e-11, rtol=0)


@pytest.mark.parametrize(
    "case",
    [
        ValidationCase(
            60e3,
            -8.0,
            1.4e-3,
            14.0,
            80.0 / RAYTEM_MM_TO_ANGSTROM,
            0.0012 * RAYTEM_MM_TO_ANGSTROM,
        ),
        ValidationCase(
            100e3,
            5.0,
            1.0e-3,
            20.0,
            100.0 / RAYTEM_MM_TO_ANGSTROM,
            0.0010 * RAYTEM_MM_TO_ANGSTROM,
        ),
        ValidationCase(
            200e3,
            12.0,
            -0.9e-3,
            26.0,
            120.0 / RAYTEM_MM_TO_ANGSTROM,
            0.0008 * RAYTEM_MM_TO_ANGSTROM,
        ),
    ],
)
def test_abcd_exact_wavepacket_moments_match_finite_raytem_lens(case):
    """Exact ABCD wave moments reproduce RayTEM's thick lens in mm units."""
    ray, wave = _run_case(case)
    np.testing.assert_allclose(wave[:, [0, 4]], ray[:, [0, 4]], atol=2e-5, rtol=0)
    np.testing.assert_allclose(wave[:, [1, 5]], ray[:, [1, 5]], atol=2e-9, rtol=0)
    np.testing.assert_allclose(wave[:, [2, 6]], ray[:, [2, 6]], atol=2e-5, rtol=0)
    np.testing.assert_allclose(wave[:, [3, 7]], ray[:, [3, 7]], atol=2e-9, rtol=0)


def _assert_final_moments_match(case, ray_elements, section_elements):
    """Compare final two-axis Gaussian moments for a RayTEM/PySlice column."""
    wavelength_A = wavelength(case.energy_eV)
    rays, weights = _gaussian_ray_ensemble(case, wavelength_A)
    for element in ray_elements:
        rays = element.propagate_ray(rays)
        if isinstance(element, RayTEMLens) and element.rotation:
            rays = _rotate_ray_coordinates(rays, element.rotation)

    column = optical_column_from_raytem(
        Microscope(
            name="validation",
            sections=[MicroscopeSection(name="validation", elements=section_elements)],
        ),
        start="start",
        stop="screen",
    )
    wf = _gaussian_wfdata(case, wavelength_A)
    column.apply(wf)

    ray = _ray_moments(rays, weights)
    wave = _wave_moments(wf, wavelength_A)
    np.testing.assert_allclose(wave[[0, 4]], ray[[0, 4]], atol=3e-5, rtol=0)
    np.testing.assert_allclose(wave[[1, 5]], ray[[1, 5]], atol=3e-9, rtol=0)
    np.testing.assert_allclose(wave[[2, 6]], ray[[2, 6]], atol=3e-5, rtol=0)
    np.testing.assert_allclose(wave[[3, 7]], ray[[3, 7]], atol=3e-9, rtol=0)


def test_thin_quadrupole_wave_moments_match_raytem():
    """Astigmatic wave power matches a physical RayTEM thin quadrupole."""
    case = ValidationCase(100e3, 6.0, 0.7e-3, 15.0, 0.0, 0.0)
    quadrupole = RayTEMQuadrupole(name="Q1", strength=50.0, length=0.0)
    _assert_final_moments_match(
        case,
        [Drift(length=_D1_MM), quadrupole, Drift(length=_D2_MM)],
        [
            Drift(name="start", length=0.0, position=0.0),
            RayTEMQuadrupole(name="Q1", strength=50.0, length=0.0, position=_D1_MM),
            Drift(name="screen", length=0.0, position=_D1_MM + _D2_MM),
        ],
    )


def test_dipole_wave_centroid_and_angle_match_raytem():
    """Wave phase ramp reproduces RayTEM's affine dipole kick."""
    case = ValidationCase(200e3, -4.0, -0.4e-3, 11.0, 0.0, 0.0)
    dipole = RayTEMDipole(name="D1", strength=0.8e-3, axis="y")
    _assert_final_moments_match(
        case,
        [Drift(length=_D1_MM), dipole, Drift(length=_D2_MM)],
        [
            Drift(name="start", length=0.0, position=0.0),
            RayTEMDipole(name="D1", strength=0.8e-3, axis="y", position=_D1_MM),
            Drift(name="screen", length=0.0, position=_D1_MM + _D2_MM),
        ],
    )


def test_calibrated_drift_wave_moments_match_raytem():
    """RayTEM drift calibration changes transverse propagation, not nominal z."""
    case = ValidationCase(100e3, 5.0, 0.8e-3, 13.0, 0.0, 0.0)
    calibrated_drift = Drift(
        name="Dcal", length=_D1_MM, calibration=0.4, position=0.0
    )
    _assert_final_moments_match(
        case,
        [calibrated_drift],
        [
            Drift(name="start", length=0.0, position=0.0),
            calibrated_drift,
            Drift(name="screen", length=0.0, position=_D1_MM),
        ],
    )


def test_prism_wave_moments_match_raytem_without_fringe_field():
    """The regular symplectic RayTEM prism has the same wave moments."""
    case = ValidationCase(100e3, 3.0, 0.3e-3, 10.0, 0.0, 0.0)
    prism = RayTEMPrism(
        name="P1", radius=1e-4, length=None, angle=10.0, strength=1e-5
    )
    _assert_final_moments_match(
        case,
        [Drift(length=_D1_MM), prism],
        [
            Drift(name="start", length=0.0, position=0.0),
            RayTEMPrism(
                name="P1",
                radius=1e-4,
                length=None,
                angle=10.0,
                strength=1e-5,
                position=_D1_MM,
            ),
            Drift(
                name="screen",
                length=0.0,
                position=_D1_MM + prism.length,
            ),
        ],
    )

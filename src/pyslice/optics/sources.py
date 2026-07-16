"""Small coherent entrance-wave models for column optics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
import math

import numpy as np

from ..backend import Backend, to_numpy
from ..data.pyslice_serial import record_pyslice_operation
from ..multislice.multislice import wavelength


Pair = float | tuple[float, float]


def _pair(value: Pair, name: str) -> tuple[float, float]:
    """Normalize a scalar or transverse pair to finite ``(x, y)`` values."""
    if isinstance(value, (int, float)):
        result = float(value), float(value)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or length-two tuple.")
        result = float(value[0]), float(value[1])
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True)
class GaussianWaveSource:
    """One coherent Gaussian electron wave at the column entrance.

    ``rms_size_A`` is the RMS width of the *intensity* in each transverse
    direction. ``curvature_inv_A`` is the wavefront slope per displacement;
    positive values describe a locally diverging wave with the phase convention
    used by :class:`WFData`. ``center_A`` and ``tilt_mrad`` translate and steer
    the same coherent wave. Partial coherence and energy distributions are
    intentionally outside this source model.
    """

    voltage_eV: float
    rms_size_A: Pair
    curvature_inv_A: Pair = 0.0
    center_A: Pair = (0.0, 0.0)
    tilt_mrad: Pair = (0.0, 0.0)

    def __post_init__(self):
        voltage = float(self.voltage_eV)
        if not math.isfinite(voltage) or voltage <= 0:
            raise ValueError("voltage_eV must be positive and finite.")
        object.__setattr__(self, "voltage_eV", voltage)
        if min(_pair(self.rms_size_A, "rms_size_A")) <= 0:
            raise ValueError("rms_size_A values must be positive.")
        _pair(self.curvature_inv_A, "curvature_inv_A")
        _pair(self.center_A, "center_A")
        _pair(self.tilt_mrad, "tilt_mrad")

    @property
    def wavelength_A(self) -> float:
        """Relativistic electron wavelength in Angstroms."""
        return float(to_numpy(wavelength(self.voltage_eV)))

    def to_wave(
        self,
        *,
        extent_A: Pair,
        sampling_A: Pair,
        backend: Optional[Backend] = None,
        device: Optional[str] = None,
    ):
        """Create a column-compatible ``WFData`` containing this one wave."""
        from ..postprocessing.wf_data import WFData

        sigma_x, sigma_y = _pair(self.rms_size_A, "rms_size_A")
        curvature_x, curvature_y = _pair(
            self.curvature_inv_A, "curvature_inv_A"
        )
        center_x, center_y = _pair(self.center_A, "center_A")
        tilt_x, tilt_y = (
            component * 1e-3 for component in _pair(self.tilt_mrad, "tilt_mrad")
        )

        template = WFData.from_probe(
            extent_A=extent_A,
            sampling=sampling_A,
            voltage_eV=self.voltage_eV,
            aperture=0.0,
            probe_positions=[(center_x, center_y)],
            backend=backend,
            device=device,
        )
        b = template._backend
        x_grid, y_grid = np.meshgrid(template.xs, template.ys, indexing="ij")
        x = x_grid - center_x
        y = y_grid - center_y
        phase = (
            np.pi
            / self.wavelength_A
            * (curvature_x * x**2 + curvature_y * y**2)
            + 2.0
            * np.pi
            / self.wavelength_A
            * (tilt_x * x + tilt_y * y)
        )
        real_wave = np.exp(
            -x**2 / (4.0 * sigma_x**2)
            - y**2 / (4.0 * sigma_y**2)
            + 1j * phase
        )
        real_wave /= np.sqrt(np.sum(np.abs(real_wave) ** 2))
        reciprocal = b.fftshift(
            b.fft2(b.asarray(real_wave, dtype=b.complex_dtype), axes=(0, 1)),
            axes=(0, 1),
        )
        template._array = reciprocal[None, None, :, :, None]
        record_pyslice_operation(
            template,
            "GaussianWaveSource.to_wave",
            parameters={
                **asdict(self),
                "extent_A": extent_A,
                "sampling_A": sampling_A,
            },
            callable_obj=self.to_wave,
        )
        return template


__all__ = ["GaussianWaveSource"]

"""Measured system-level aberrations for coherent probe-forming optics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Optional
import math
import re
import warnings

import numpy as np

from ..backend import to_numpy
from ..data.pyslice_serial import record_pyslice_operation


AberrationCoefficient = float | tuple[float, float]
ReferenceSide = Literal["entrance", "exit"]
_CNM_PATTERN = re.compile(r"^C\d\d$")


def _coefficient(value: Any, name: str) -> AberrationCoefficient:
    """Normalize one PySlice Cnm coefficient without changing its units."""
    if isinstance(value, (int, float)):
        result: AberrationCoefficient = float(value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        result = (float(value[0]), float(value[1]))
    else:
        raise TypeError(
            f"{name} must be a scalar coefficient or (coefficient, angle_rad)."
        )
    values = (result,) if isinstance(result, float) else result
    if not all(math.isfinite(component) for component in values):
        raise ValueError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True)
class ProbeAberrationModel:
    """Effective coherent aberrations measured for a probe-forming system.

    Coefficients follow PySlice's Cnm convention and are expressed in
    Angstroms. They describe the net incident-probe wave aberration at
    ``reference_plane`` rather than errors assigned to individual lenses.
    Off-axis terms use ``(coefficient_A, orientation_rad)`` values.

    ``semiangle_mrad`` applies a circular angular pupil at the same plane.
    ``replaces_upstream_element_aberrations`` defaults to true because a
    system-level measurement normally already contains the net aberrations of
    upstream elements. Downstream local aberrations remain active.

    This object intentionally describes one coherent energy. Chromatic
    averaging and partial coherence are outside the current column API.
    """

    coefficients_A: Mapping[str, AberrationCoefficient]
    semiangle_mrad: Optional[float] = None
    reference_plane: str | float = "sample"
    reference_side: ReferenceSide = "entrance"
    replaces_upstream_element_aberrations: bool = True
    name: str = "measured probe aberrations"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        coefficients = {}
        for key, value in self.coefficients_A.items():
            if not _CNM_PATTERN.fullmatch(str(key)):
                raise ValueError(
                    f"Invalid aberration key {key!r}; expected Cnm such as C30."
                )
            coefficients[str(key)] = _coefficient(value, str(key))
        object.__setattr__(self, "coefficients_A", coefficients)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not isinstance(self.replaces_upstream_element_aberrations, bool):
            raise TypeError(
                "replaces_upstream_element_aberrations must be a bool."
            )
        if self.semiangle_mrad is not None:
            semiangle = float(self.semiangle_mrad)
            if not math.isfinite(semiangle) or semiangle <= 0:
                raise ValueError("semiangle_mrad must be positive and finite.")
            object.__setattr__(self, "semiangle_mrad", semiangle)
        if self.reference_side not in {"entrance", "exit"}:
            raise ValueError("reference_side must be 'entrance' or 'exit'.")
        if isinstance(self.reference_plane, str) and not self.reference_plane:
            raise ValueError("reference_plane must not be an empty name.")
        if isinstance(self.reference_plane, (int, float)) and not math.isfinite(
            float(self.reference_plane)
        ):
            raise ValueError("reference_plane must be finite.")

    def phase_sampling_report(self, wf) -> dict[str, float]:
        """Report the largest unwrapped aberration-phase step on the grid."""
        kxs = np.asarray(wf.kxs, dtype=float)
        kys = np.asarray(wf.kys, dtype=float)
        kx, ky = np.meshgrid(kxs, kys, indexing="ij")
        radial_frequency = np.sqrt(kx**2 + ky**2)
        azimuth = np.arctan2(ky, kx)
        wavelength_A = float(to_numpy(wf.probe.wavelength))
        phase = np.zeros_like(radial_frequency)
        for key, value in self.coefficients_A.items():
            n, m = int(key[1]), int(key[2])
            coefficient, angle = (
                (float(value), 0.0)
                if isinstance(value, (int, float))
                else (float(value[0]), float(value[1]))
            )
            phase += (
                2.0
                * np.pi
                / wavelength_A
                / (n + 1)
                * coefficient
                * (radial_frequency * wavelength_A) ** (n + 1)
                * np.cos(m * (azimuth - angle))
            )

        valid = np.ones(phase.shape, dtype=bool)
        reciprocal_power = np.sum(np.abs(to_numpy(wf.array)) ** 2, axis=(0, 1, 4))
        if np.max(reciprocal_power) > 0:
            valid &= reciprocal_power > np.max(reciprocal_power) * 1e-12
        if self.semiangle_mrad is not None:
            cutoff = self.semiangle_mrad * 1e-3 / wavelength_A
            valid &= radial_frequency <= cutoff
        x_steps = np.abs(np.diff(phase, axis=0))
        y_steps = np.abs(np.diff(phase, axis=1))
        x_valid = valid[1:, :] & valid[:-1, :]
        y_valid = valid[:, 1:] & valid[:, :-1]
        maximum_step = max(
            float(np.max(x_steps[x_valid])) if np.any(x_valid) else 0.0,
            float(np.max(y_steps[y_valid])) if np.any(y_valid) else 0.0,
        )
        return {"max_phase_step_rad": maximum_step}

    def apply(self, wf):
        """Apply the coherent wave aberration and angular pupil in place."""
        phase_report = self.phase_sampling_report(wf)
        if phase_report["max_phase_step_rad"] > np.pi / 2.0:
            warnings.warn(
                "The measured aberration phase is under-sampled "
                f"(maximum adjacent-pixel step "
                f"{phase_report['max_phase_step_rad']:.3g} rad). Increase the "
                "field of view to refine reciprocal-angle sampling.",
                RuntimeWarning,
            )
        if self.coefficients_A:
            wf.aberrate(dict(self.coefficients_A))
        if self.semiangle_mrad is not None:
            wf.apply_angular_aperture(self.semiangle_mrad)
        record_pyslice_operation(
            wf,
            "ProbeAberrationModel.apply",
            parameters=asdict(self),
            callable_obj=self.apply,
        )
        return wf


__all__ = ["ProbeAberrationModel"]

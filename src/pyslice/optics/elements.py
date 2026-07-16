"""Wave-optics elements that operate on :class:`pyslice.postprocessing.wf_data.WFData`.

The classes here intentionally model PySlice wave operators, not any one ray
tracing package.  RayTEM (or another ray-code) should be adapted into these
objects before applying the optics to a wavefunction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional
from copy import deepcopy
import math
import numpy as np


class OpticalElement:
    """Base protocol for a wave-optics element.

    Implementations mutate the supplied ``WFData`` in place and return it for
    convenient chaining, matching the existing ``WFData`` propagation methods.
    """

    name: str

    def apply(self, wf):
        raise NotImplementedError


@dataclass
class FreeSpace(OpticalElement):
    """Free-space Fresnel propagation over ``dz_A`` Angstroms."""

    dz_A: float
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, wf):
        if self.dz_A != 0:
            wf.propagate_free_space(float(self.dz_A))
        return wf


@dataclass
class Lens(OpticalElement):
    """Round lens wave operator.

    ``principal_plane_drift_A`` is applied on both sides of the thin-lens phase.
    It is zero for a physical thin lens. For a finite RayTEM lens, the adapter
    chooses it from the exact symmetric ABCD factorization, making the complete
    entrance-to-exit paraxial wave operator exact up to global phase.
    ``rotation_rad`` then applies the lens's Larmor image rotation.

    ``aberrations`` are local Cnm wave-aberration coefficients owned by this
    lens. ``aberration_plane`` places their reciprocal-space phase screen at
    the entrance, principal plane, or exit. The exit default preserves the
    historical RayTEM adapter convention.
    """

    f_A: float
    name: str = ""
    z_A: Optional[float] = None
    thickness_A: float = 0.0
    principal_plane_drift_A: float = 0.0
    rotation_rad: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    aberrations: dict[str, Any] = field(default_factory=dict)
    aberration_plane: Literal["entrance", "principal", "exit"] = "exit"

    def __post_init__(self):
        if self.thickness_A < 0:
            raise ValueError("Lens thickness_A must be non-negative.")
        if self.f_A == 0:
            raise ValueError("Lens focal length f_A must be non-zero.")
        if not math.isfinite(float(self.principal_plane_drift_A)):
            raise ValueError("principal_plane_drift_A must be finite.")
        if not isinstance(self.aberrations, Mapping):
            raise TypeError("Lens aberrations must be a Cnm coefficient mapping.")
        self.aberrations = deepcopy(dict(self.aberrations))
        invalid = [
            key
            for key in self.aberrations
            if not (
                isinstance(key, str)
                and len(key) == 3
                and key.startswith("C")
                and key[1:].isdigit()
            )
        ]
        if invalid:
            raise ValueError(
                f"Invalid lens aberration keys {invalid!r}; expected Cnm labels."
            )
        if self.aberration_plane not in {"entrance", "principal", "exit"}:
            raise ValueError(
                "aberration_plane must be 'entrance', 'principal', or 'exit'."
            )

    @property
    def is_active(self) -> bool:
        return self.f_A is not None and math.isfinite(float(self.f_A))

    def apply(self, wf):
        drift_A = float(self.principal_plane_drift_A)

        def apply_aberrations_at(plane):
            if self.aberrations and self.aberration_plane == plane:
                wf.aberrate(deepcopy(self.aberrations))

        apply_aberrations_at("entrance")

        if drift_A:
            wf.propagate_free_space(drift_A)
        if self.is_active:
            wf.propagate_through_lens(float(self.f_A))
        apply_aberrations_at("principal")
        if drift_A:
            wf.propagate_free_space(drift_A)
        if self.rotation_rad:
            wf.rotate_real_space(float(self.rotation_rad))
        apply_aberrations_at("exit")
        return wf


@dataclass
class SeparableParaxialMap(OpticalElement):
    """First-order wave map with independent x and y ray matrices.

    Each matrix acts on ``(position, angle)`` and must be symplectic. The map
    is factored into an anisotropic drift, an astigmatic thin lens, and a
    second anisotropic drift, which is an exact metaplectic realization up to
    an unobservable global phase.
    """

    x_matrix: Any
    y_matrix: Any
    name: str = ""
    z_A: Optional[float] = None
    thickness_A: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.x_matrix = self._validated_matrix(self.x_matrix, "x")
        self.y_matrix = self._validated_matrix(self.y_matrix, "y")

    @staticmethod
    def _validated_matrix(matrix, axis):
        value = np.asarray(matrix, dtype=float)
        if value.shape != (2, 2):
            raise ValueError(f"{axis}_matrix must have shape (2, 2).")
        determinant = float(np.linalg.det(value))
        if not np.isclose(determinant, 1.0, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"{axis}-axis ray map is not symplectic: det={determinant:.12g}. "
                "A lossless scalar-wave operator requires det=1."
            )
        return value

    @staticmethod
    def _factor(matrix):
        a, b = matrix[0]
        c, d = matrix[1]
        if c == 0.0:
            if np.isclose(a, 1.0) and np.isclose(d, 1.0):
                return float(b), math.inf, 0.0
            raise ValueError(
                "A C=0 paraxial magnifier cannot be represented on WFData's "
                "fixed sampling grid."
            )
        return float((d - 1.0) / c), float(-1.0 / c), float((a - 1.0) / c)

    def apply(self, wf):
        dx1, fx, dx2 = self._factor(self.x_matrix)
        dy1, fy, dy2 = self._factor(self.y_matrix)
        if dx1 or dy1:
            wf.propagate_anisotropic_free_space(dx1, dy1)
        if math.isfinite(fx) or math.isfinite(fy):
            wf.propagate_through_astigmatic_lens(fx, fy)
        if dx2 or dy2:
            wf.propagate_anisotropic_free_space(dx2, dy2)
        return wf


@dataclass
class BeamTilt(OpticalElement):
    """Affine angular kick in radians."""

    theta_x_rad: float = 0.0
    theta_y_rad: float = 0.0
    name: str = ""
    z_A: Optional[float] = None
    thickness_A: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, wf):
        wf.apply_beam_tilt(float(self.theta_x_rad), float(self.theta_y_rad))
        return wf


@dataclass
class Aberration(OpticalElement):
    """Wavefront aberration phase in PySlice's Cnm convention.

    Coefficients follow :func:`pyslice.multislice.multislice.aberrationFunction`:
    values are Angstroms, and off-axis terms may be ``(value_A, phi0_rad)``.
    Examples include ``{"C10": defocus_A}``, ``{"C12": (astig_A, phi0)}``,
    and ``{"C30": spherical_A}``.
    """

    aberrations: dict[str, Any]
    name: str = ""
    z_A: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, wf):
        if self.aberrations:
            wf.aberrate(deepcopy(self.aberrations))
        return wf


@dataclass
class Aperture(OpticalElement):
    """Circular aperture in real or reciprocal space."""

    radius: float
    space: Literal["real", "reciprocal"] = "real"
    name: str = ""
    z_A: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.space not in {"real", "reciprocal"}:
            raise ValueError("Aperture space must be 'real' or 'reciprocal'.")
        if self.radius < 0:
            raise ValueError("Aperture radius must be non-negative.")

    def apply(self, wf):
        wf.applyMask(float(self.radius), realOrReciprocal=self.space)
        return wf


__all__ = [
    "OpticalElement",
    "FreeSpace",
    "Lens",
    "SeparableParaxialMap",
    "BeamTilt",
    "Aberration",
    "Aperture",
]

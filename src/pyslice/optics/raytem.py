"""Adapters from RayTEM ray configurations into PySlice wave optics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional
import json
import math
import warnings
import numpy as np

from .column import OpticalColumn, WavePropagation
from .elements import (
    Aberration,
    Aperture,
    BeamTilt,
    FreeSpace,
    Lens,
    SeparableParaxialMap,
)

RAYTEM_MM_TO_ANGSTROM = 1.0e7
AnchorSide = Literal["entrance", "exit", "center"]
RAYTEM_ABERRATION_CONTAINER_KEYS = (
    "aberrations",
    "aberration",
    "cnm",
    "Cnm",
    "C_nm",
    "aberration_coefficients",
    "aberrationCoefficients",
)


@dataclass(frozen=True)
class RayTEMElementRecord:
    """Flattened RayTEM element with absolute axial coordinates."""

    name: str
    kind: str
    z: float
    length: float
    section: str
    raw: Mapping[str, Any]

    def anchor(self, side: AnchorSide) -> float:
        if side == "entrance":
            return self.z
        if side == "exit":
            return self.z + self.length
        if side == "center":
            return self.z + self.length / 2.0
        raise ValueError("anchor side must be 'entrance', 'exit', or 'center'.")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _element_name(element: Mapping[str, Any]) -> str:
    value = element.get("Element name", element.get("name", ""))
    return "" if value is None else str(value)


def _section_name(section: Mapping[str, Any]) -> str:
    value = section.get("Section name", section.get("name", ""))
    return "" if value is None else str(value)


def _records_from_json_dict(data: Mapping[str, Any]) -> list[RayTEMElementRecord]:
    records: list[RayTEMElementRecord] = []
    sections = sorted(
        data.get("Sections", []), key=lambda section: _as_float(section.get("position"), 0.0)
    )
    for section in sections:
        section_z = _as_float(section.get("position"), 0.0)
        section_name = _section_name(section)
        for element in section.get("Elements", []):
            records.append(
                RayTEMElementRecord(
                    name=_element_name(element),
                    kind=str(element.get("kind", "")),
                    z=section_z + _as_float(element.get("position"), 0.0),
                    length=_as_float(element.get("length"), 0.0),
                    section=section_name,
                    raw=element,
                )
            )
    # Element order inside each section is the RayTEM transfer order. Preserve
    # it at shared boundaries, where floating-point positions can differ by a
    # few ulps and a global z sort can move a post-element drift ahead of the
    # element itself.
    return records


def _records_from_raytem_object(microscope: Any) -> list[RayTEMElementRecord]:
    records: list[RayTEMElementRecord] = []
    sections = sorted(
        getattr(microscope, "sections", []) or [],
        key=lambda section: _as_float(getattr(section, "position", 0.0), 0.0),
    )
    for section in sections:
        section_z = _as_float(getattr(section, "position", 0.0), 0.0)
        section_name = "" if getattr(section, "name", None) is None else str(section.name)
        for element in getattr(section, "elements", []) or []:
            raw = dict(getattr(element, "__dict__", {}))
            raw.setdefault("kind", getattr(element, "kind", ""))
            raw.setdefault("name", getattr(element, "name", ""))
            records.append(
                RayTEMElementRecord(
                    name="" if getattr(element, "name", None) is None else str(element.name),
                    kind=str(getattr(element, "kind", "")),
                    z=section_z + _as_float(getattr(element, "position", 0.0), 0.0),
                    length=_as_float(getattr(element, "length", 0.0), 0.0),
                    section=section_name,
                    raw=raw,
                )
            )
    return records


def load_raytem_records(path_or_data: str | Path | Mapping[str, Any] | Any) -> list[RayTEMElementRecord]:
    """Load RayTEM element records from JSON data, a path, or a RayTEM object.

    JSON support is dependency-free and matches ``rayTEM.Microscope.save``.
    Non-JSON paths fall back to RayTEM's own ``load_microscope`` when RayTEM is
    importable, which enables ``.sea`` files without making RayTEM a hard PySlice
    dependency.
    """
    if isinstance(path_or_data, Mapping):
        return _records_from_json_dict(path_or_data)

    if hasattr(path_or_data, "sections"):
        return _records_from_raytem_object(path_or_data)

    path = Path(path_or_data)
    if path.suffix == "":
        json_path = path.with_suffix(".json")
        if json_path.exists():
            path = json_path
    if path.suffix.lower() == ".json":
        return _records_from_json_dict(json.loads(path.read_text()))

    try:
        from pySEA.rayTEM.assemblies import load_microscope
    except Exception as exc:  # pragma: no cover - depends on optional RayTEM install
        raise ImportError(
            "RayTEM is required to load non-JSON RayTEM configurations. "
            "Use a .json file or install/import pySEA.rayTEM."
        ) from exc
    return _records_from_raytem_object(load_microscope(str(path)))


def _calibrated_lens_strength(element: Mapping[str, Any]) -> float:
    """Replicate RayTEM ``Lens.transfer_matrix`` strength calibration."""
    strength = _as_float(element.get("strength"), 0.0)
    calibration = element.get("calibration")
    if calibration is None:
        return strength
    if isinstance(calibration, (int, float)):
        return strength * float(calibration)
    if isinstance(calibration, (list, tuple)):
        # RayTEM uses A + B*x^(1/1) + C*x^(1/2) + D*x^(1/3) + ...
        values = [float(calibration[0])]
        values.extend(
            float(coef) * strength ** (1.0 / (i + 1))
            for i, coef in enumerate(calibration[1:])
        )
        return float(sum(values))
    raise TypeError(f"Unsupported RayTEM lens calibration {calibration!r}")


def _calibrated_power_strength(element: Mapping[str, Any]) -> float:
    """Replicate RayTEM quadrupole and dipole strength calibration."""
    strength = _as_float(element.get("strength"), 0.0)
    calibration = element.get("calibration")
    if calibration is None:
        return strength
    if isinstance(calibration, (int, float)):
        return strength * float(calibration)
    if isinstance(calibration, (list, tuple)) and len(calibration) == 2:
        scale, power = calibration
        return float(scale) * strength ** float(power)
    raise TypeError(f"Unsupported RayTEM power calibration {calibration!r}")


def raytem_lens_focal_length(element: Mapping[str, Any]) -> float:
    """Return the effective focal length of a RayTEM round lens.

    The returned value is in the same length units as the RayTEM configuration.
    It is derived from RayTEM's first-order lens matrix via ``f = -1 / M[xt,x]``.
    Zero-strength lenses return ``math.inf`` and are treated as drifts by the
    importer.
    """
    k = _calibrated_lens_strength(element)
    length = _as_float(element.get("length"), 0.0)
    if k == 0:
        return math.inf
    if length == 0:
        return math.copysign(1.0 / (k * k), k)
    denominator = k * math.sin(k * length)
    if denominator == 0:
        return math.inf
    return 1.0 / denominator


def raytem_lens_principal_plane_drift(element: Mapping[str, Any]) -> float:
    """Return the symmetric drift in RayTEM's exact thick-lens factorization.

    For RayTEM's matrix ``[[cos(KL), sin(KL)/K], [-K sin(KL), cos(KL)]]``,
    ``D(d) @ L(f) @ D(d)`` is identical when
    ``d = tan(KL/2)/K`` and ``f = 1/(K sin(KL))``.  The returned distance uses
    RayTEM's native millimeter units.
    """
    k = _calibrated_lens_strength(element)
    length = _as_float(element.get("length"), 0.0)
    if length == 0:
        return 0.0
    if k == 0:
        return length / 2.0
    phase = k * length
    if abs(math.cos(phase / 2.0)) < 1e-12:
        raise ValueError(
            "RayTEM lens is singular in the symmetric ABCD factorization "
            f"(K*L={phase})."
        )
    return math.tan(phase / 2.0) / k


def _resolve_anchor(
    records: Iterable[RayTEMElementRecord],
    anchor: str | float | int | None,
    *,
    side: AnchorSide,
    default: float,
) -> tuple[float, Optional[int]]:
    """Resolve a segment boundary and, for named anchors, its transfer index."""
    records = list(records)
    if anchor is None:
        return default, None
    if isinstance(anchor, (int, float)):
        return float(anchor), None
    matches = [
        (index, record)
        for index, record in enumerate(records)
        if record.name == anchor
    ]
    if not matches:
        available = sorted({record.name for record in records if record.name})
        raise ValueError(f"RayTEM element {anchor!r} not found. Named elements: {available}")
    index, record = matches[0]
    return record.anchor(side), index


def _is_lens(kind: str) -> bool:
    return kind in {"Thin lens", "QLens", "Lens"}


def _is_aperture(kind: str) -> bool:
    return kind == "Aperture"


def _is_drift(kind: str) -> bool:
    return kind == "Drift"


def _is_quadrupole(kind: str) -> bool:
    return kind in {"Thin quad", "Quad", "Quadrapole", "Quadrupole"}


def _is_dipole(kind: str) -> bool:
    return kind in {"Thin dipole", "Dipole"}


def _is_prism(kind: str) -> bool:
    return kind == "Prism"


def _z_isclose(first: float, second: float) -> bool:
    """Compare RayTEM axial coordinates across serialized float roundoff."""
    return math.isclose(float(first), float(second), rel_tol=1e-12, abs_tol=1e-15)


def _element_is_upstream(z_A: float, length_A: float, current_A: float) -> bool:
    """Return whether a complete element lies before the current plane."""
    exit_A = z_A + length_A
    if length_A == 0:
        return z_A < current_A and not _z_isclose(z_A, current_A)
    return exit_A < current_A or _z_isclose(exit_A, current_A)


def _validate_boundary_not_inside_element(
    records: Iterable[RayTEMElementRecord], boundary: float, label: str
) -> None:
    """Reject a segment boundary inside an indivisible finite ray element."""
    for record in records:
        if record.length <= 0 or not (
            _is_drift(record.kind)
            or _is_lens(record.kind)
            or _is_quadrupole(record.kind)
            or _is_dipole(record.kind)
            or _is_prism(record.kind)
        ):
            continue
        element_exit = record.z + record.length
        after_entrance = record.z < boundary and not _z_isclose(record.z, boundary)
        before_exit = boundary < element_exit and not _z_isclose(boundary, element_exit)
        if after_entrance and before_exit:
            raise ValueError(
                f"{label} z={boundary} cuts through finite RayTEM element "
                f"{(record.name or record.kind)!r} spanning "
                f"[{record.z}, {element_exit}]. Partial element "
                "transfer maps are not implemented; use its entrance or exit."
            )


def _matrix_in_angstrom(matrix_mm: np.ndarray) -> np.ndarray:
    """Convert a position-angle matrix from millimeters to Angstroms."""
    matrix_A = np.asarray(matrix_mm, dtype=float).copy()
    matrix_A[0, 1] *= RAYTEM_MM_TO_ANGSTROM
    matrix_A[1, 0] /= RAYTEM_MM_TO_ANGSTROM
    return matrix_A


def raytem_quadrupole_matrices(
    element: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return RayTEM's x and y quadrupole matrices in millimeter units."""
    k = _calibrated_power_strength(element)
    length = _as_float(element.get("length"), 0.0)
    if k == 0:
        return np.eye(2), np.eye(2)
    if length == 0:
        x_kick, y_kick = -(k * k), k * k
        if k > 0:
            x_kick, y_kick = y_kick, x_kick
        return np.array([[1.0, 0.0], [x_kick, 1.0]]), np.array(
            [[1.0, 0.0], [y_kick, 1.0]]
        )

    phase = abs(k * length)
    cosine = math.cos(phase)
    sine = math.sin(phase)
    return (
        np.array([[cosine, sine / k], [-k * sine, cosine]]),
        np.array([[cosine, sine / k], [k * sine, cosine]]),
    )


def raytem_prism_matrices(
    element: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return RayTEM's x and y prism matrices in millimeter units."""
    strength = _as_float(element.get("strength"), 0.0)
    length = _as_float(element.get("length"), 0.0)
    angle = math.radians(_as_float(element.get("angle"), 45.0))
    radius_value = element.get("radius")
    radius = _as_float(radius_value, length / angle if angle else 0.0)
    if strength == 0:
        drift = np.array([[1.0, length], [0.0, 1.0]])
        return drift, drift.copy()
    if radius == 0:
        raise ValueError("A nonzero RayTEM prism requires a nonzero radius.")

    k1 = _as_float(element.get("K1", element.get("k1")), 0.0)
    if k1 == 0:
        qx = -math.tan(strength) / radius
        focus_y = np.eye(2)
    else:
        qx = math.tan(strength) / radius
        gap = _as_float(element.get("g"), 1.0)
        psi = gap / radius * k1 * (1.0 + math.sin(strength) ** 2) / math.cos(strength)
        qy = -math.tan(strength - psi) / radius
        # This reproduces RayTEM's current slice assignment. The wave element's
        # symplectic check will reject it if it is not a physical ray map.
        focus_y = np.full((2, 2), qy)
    focus_x = np.array([[1.0, 0.0], [qx, 1.0]])

    bend_angle = strength / radius
    cosine = math.cos(bend_angle)
    sine = math.sin(bend_angle)
    bend_x = np.array(
        [[cosine, radius * sine], [-sine / radius, cosine]]
    )
    return focus_x @ bend_x @ focus_x, focus_y @ focus_y


def raytem_dipole_tilt(element: Mapping[str, Any]) -> tuple[float, float]:
    """Return RayTEM's affine dipole kick ``(theta_x, theta_y)``."""
    strength = _calibrated_power_strength(element)
    length = _as_float(element.get("length"), 0.0)
    kick = strength if length == 0 else strength * length
    if "phi" in element:
        phi = _as_float(element.get("phi"), 0.0)
    else:
        axis = element.get("axis", "x")
        if isinstance(axis, str):
            phi = 0.0 if axis.lower() == "x" else math.pi / 2.0
        elif isinstance(axis, (list, tuple)):
            phi = math.atan2(float(axis[1]), float(axis[0]))
        else:
            phi = float(axis)
    return kick * math.cos(phi), kick * math.sin(phi)


def _is_cnm_key(key: Any) -> bool:
    """Return whether ``key`` is a PySlice Cnm aberration label."""
    return isinstance(key, str) and len(key) == 3 and key[0] == "C" and key[1:].isdigit()


def _iter_aberration_items(container: Any):
    if isinstance(container, Mapping):
        yield from container.items()
    elif isinstance(container, (list, tuple)):
        for item in container:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key", item.get("name", item.get("label")))
            if key is None:
                n = item.get("n")
                m = item.get("m")
                if n is not None and m is not None:
                    key = f"C{int(n)}{int(m)}"
            value = item.get("value", item.get("coefficient", item.get("C")))
            if value is None and "magnitude" in item:
                value = item["magnitude"]
            if key is not None and value is not None:
                angle = item.get("phi0", item.get("phi", item.get("angle")))
                yield key, value if angle is None else (value, angle)


def _scale_aberration_value(value: Any, coefficient_scale_A: float):
    if isinstance(value, Mapping):
        coefficient = value.get("value", value.get("coefficient", value.get("C")))
        if coefficient is None and "magnitude" in value:
            coefficient = value["magnitude"]
        if coefficient is None:
            raise ValueError(f"Aberration mapping is missing a coefficient value: {value!r}")
        angle = value.get("phi0", value.get("phi", value.get("angle", 0.0)))
        return (float(coefficient) * coefficient_scale_A, float(angle))
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Aberration coefficient lists must not be empty.")
        if len(value) == 1:
            return float(value[0]) * coefficient_scale_A
        return (float(value[0]) * coefficient_scale_A, float(value[1]))
    return float(value) * coefficient_scale_A


def raytem_aberrations(
    element: Mapping[str, Any],
    *,
    coefficient_scale_A: float = 1.0,
) -> dict[str, Any]:
    """Extract PySlice Cnm aberrations from a RayTEM element/config mapping.

    Current RayTEM microscope JSON files do not appear to emit aberration
    coefficients, but this accepts the natural extension points:
    ``aberrations={"C30": ...}``, ``cnm={...}``, or top-level ``Cnm`` keys.
    Coefficients are scaled into Angstroms with ``coefficient_scale_A``; angular
    offsets are left in radians.
    """
    aberrations: dict[str, Any] = {}

    for container_key in RAYTEM_ABERRATION_CONTAINER_KEYS:
        if container_key not in element or element[container_key] is None:
            continue
        for key, value in _iter_aberration_items(element[container_key]):
            if _is_cnm_key(key):
                aberrations[str(key)] = _scale_aberration_value(value, coefficient_scale_A)

    for key, value in element.items():
        if _is_cnm_key(key):
            aberrations[str(key)] = _scale_aberration_value(value, coefficient_scale_A)

    return aberrations


def optical_column_from_raytem(
    path_or_data: str | Path | Mapping[str, Any] | Any,
    *,
    start: str | float | int | None = None,
    stop: str | float | int | None = None,
    start_at: AnchorSide = "exit",
    stop_at: AnchorSide = "entrance",
    include_apertures: bool = True,
    aperture_space: Literal["real", "reciprocal"] = "real",
    include_aberrations: bool = True,
    name: str = "RayTEM optical column",
) -> OpticalColumn:
    """Convert a RayTEM configuration into a PySlice ``OpticalColumn``.

    Parameters
    ----------
    path_or_data:
        RayTEM JSON path, JSON dict, RayTEM object, or optional RayTEM-loadable
        file path.
    start, stop:
        Named RayTEM element or numeric z coordinate defining the segment to
        convert. Boundaries inside finite elements are rejected because partial
        element transfer maps are not currently implemented.
    start_at, stop_at:
        Element side used for named boundaries. Numeric boundaries are absolute
        coordinates and do not change with these options; operations exactly at
        a numeric stop are excluded.
    include_aberrations:
        If true, RayTEM elements carrying Cnm coefficients in an ``aberrations``
        / ``cnm`` mapping or top-level ``C10``-style keys become PySlice
        ``Aberration`` elements. Coefficients are converted from millimeters to
        Angstroms.
    aperture_space:
        RayTEM aperture radii are physical lengths, so only ``"real"`` is
        supported. A reciprocal cutoff requires focal-plane geometry and
        cannot be inferred by changing units alone.
    Notes
    -----
    Round lenses, Larmor rotation, dipoles, quadrupoles, and prisms are
    translated from RayTEM's current first-order definitions. Quadrupole and
    prism matrices must be symplectic to admit a lossless scalar-wave operator.
    """
    records = load_raytem_records(path_or_data)
    if not records:
        raise ValueError("RayTEM configuration contained no elements.")
    if include_apertures and aperture_space != "real":
        raise ValueError(
            "RayTEM aperture radii are physical lengths in millimeters and "
            "cannot be interpreted as reciprocal-space cutoffs. Use "
            "aperture_space='real' or add a native PySlice reciprocal Aperture "
            "with a cutoff in inverse Angstroms."
        )

    min_z = min(record.z for record in records)
    max_z = max(record.z + record.length for record in records)
    start_z, start_index = _resolve_anchor(
        records, start, side=start_at, default=min_z
    )
    stop_z, stop_index = _resolve_anchor(
        records, stop, side=stop_at, default=max_z
    )
    if stop_z < start_z:
        raise ValueError(f"stop z ({stop_z}) is before start z ({start_z}).")
    _validate_boundary_not_inside_element(records, start_z, "start")
    _validate_boundary_not_inside_element(records, stop_z, "stop")

    scale = RAYTEM_MM_TO_ANGSTROM
    start_A = start_z * scale
    stop_A = stop_z * scale
    current_A = start_A
    elements = []

    for record_index, record in enumerate(records):
        # The default entrance stop ends immediately before an element whose
        # entrance is the resolved stop coordinate. Exit and center stops can
        # include an element that starts before their resolved coordinate.
        before_named_start = start_index is not None and (
            record_index < start_index
            or (record_index == start_index and start_at == "exit")
        )
        after_named_stop = stop_index is not None and (
            record_index > stop_index
            or (record_index == stop_index and stop_at == "entrance")
        )
        starts_before_start = start_index is None and (
            record.z < start_z and not _z_isclose(record.z, start_z)
        )
        starts_after_stop = stop_index is None and (
            record.z > stop_z and not _z_isclose(record.z, stop_z)
        )
        # Numeric stops are absolute coordinates, so ``stop_at`` has no
        # element-side meaning. Stop before every operation at that plane;
        # named stops continue to use their requested entrance/exit anchor.
        at_or_beyond_numeric_stop = (
            stop_index is None
            and stop is not None
            and (record.z > stop_z or _z_isclose(record.z, stop_z))
        )
        record_exit = record.z + record.length
        at_or_before_exit_start = (
            start_index is None
            and start is not None
            and start_at == "exit"
            and (record_exit < start_z or _z_isclose(record_exit, start_z))
        )
        if (
            before_named_start
            or after_named_stop
            or starts_before_start
            or starts_after_stop
            or at_or_beyond_numeric_stop
            or at_or_before_exit_start
        ):
            continue
        z_A = record.z * scale
        length_A = record.length * scale
        aberrations = raytem_aberrations(record.raw, coefficient_scale_A=scale) if include_aberrations else {}

        if _is_drift(record.kind):
            if _element_is_upstream(z_A, length_A, current_A):
                continue
            if z_A > current_A:
                elements.append(
                    FreeSpace(z_A - current_A, name=f"to {record.name or record.kind}")
                )
                current_A = z_A

            calibration = record.raw.get("calibration")
            calibrated_scale = 1.0 if calibration is None else float(calibration)
            if length_A:
                elements.append(
                    FreeSpace(
                        length_A * calibrated_scale,
                        name=record.name or "RayTEM drift",
                        metadata={
                            "raytem": dict(record.raw),
                            "section": record.section,
                            # RayTEM advances its z coordinate by the nominal
                            # length even when calibration changes transverse
                            # propagation.
                            "physical_dz_A": length_A,
                        },
                    )
                )
            if aberrations:
                elements.append(
                    Aberration(
                        aberrations,
                        name=record.name or f"{record.kind} aberrations",
                        z_A=z_A + length_A,
                        metadata={"raytem": dict(record.raw), "section": record.section},
                    )
                )
            current_A = z_A + length_A

        elif _is_lens(record.kind):
            # Skip lenses entirely upstream of the chosen start plane.
            if _element_is_upstream(z_A, length_A, current_A):
                continue
            if z_A > current_A:
                elements.append(FreeSpace(z_A - current_A, name=f"to {record.name or record.kind}"))
                current_A = z_A

            f = raytem_lens_focal_length(record.raw)
            if not math.isfinite(f):
                if length_A:
                    elements.append(
                        FreeSpace(length_A, name=f"{record.name or record.kind} thickness")
                    )
                if aberrations:
                    elements.append(
                        Aberration(
                            aberrations,
                            name=record.name or f"{record.kind} aberrations",
                            z_A=z_A + length_A,
                            metadata={"raytem": dict(record.raw), "section": record.section},
                        )
                    )
                current_A = z_A + length_A
                continue

            principal_plane_drift_A = (
                raytem_lens_principal_plane_drift(record.raw) * scale
            )
            rotation = -_calibrated_lens_strength(record.raw) * record.length
            elements.append(
                Lens(
                    f * scale,
                    name=record.name,
                    z_A=z_A,
                    thickness_A=length_A,
                    principal_plane_drift_A=principal_plane_drift_A,
                    rotation_rad=rotation,
                    metadata={"raytem": dict(record.raw), "section": record.section},
                )
            )
            if aberrations:
                elements.append(
                    Aberration(
                        aberrations,
                        name=record.name or f"{record.kind} aberrations",
                        z_A=z_A + length_A,
                        metadata={"raytem": dict(record.raw), "section": record.section},
                    )
                )
            current_A = z_A + length_A

        elif _is_quadrupole(record.kind) or _is_prism(record.kind):
            if _element_is_upstream(z_A, length_A, current_A):
                continue
            if z_A > current_A:
                elements.append(FreeSpace(z_A - current_A, name=f"to {record.name or record.kind}"))
                current_A = z_A
            if _is_quadrupole(record.kind):
                x_matrix, y_matrix = raytem_quadrupole_matrices(record.raw)
            else:
                x_matrix, y_matrix = raytem_prism_matrices(record.raw)
            elements.append(
                SeparableParaxialMap(
                    _matrix_in_angstrom(x_matrix),
                    _matrix_in_angstrom(y_matrix),
                    name=record.name or record.kind,
                    z_A=z_A,
                    thickness_A=length_A,
                    metadata={"raytem": dict(record.raw), "section": record.section},
                )
            )
            if aberrations:
                elements.append(
                    Aberration(
                        aberrations,
                        name=record.name or f"{record.kind} aberrations",
                        z_A=z_A + length_A,
                        metadata={"raytem": dict(record.raw), "section": record.section},
                    )
                )
            current_A = z_A + length_A

        elif _is_dipole(record.kind):
            if _element_is_upstream(z_A, length_A, current_A):
                continue
            if z_A > current_A:
                elements.append(FreeSpace(z_A - current_A, name=f"to {record.name or record.kind}"))
                current_A = z_A
            theta_x, theta_y = raytem_dipole_tilt(record.raw)
            elements.append(
                BeamTilt(
                    theta_x,
                    theta_y,
                    name=record.name or record.kind,
                    z_A=z_A,
                    thickness_A=length_A,
                    metadata={"raytem": dict(record.raw), "section": record.section},
                )
            )
            if aberrations:
                elements.append(
                    Aberration(
                        aberrations,
                        name=record.name or f"{record.kind} aberrations",
                        z_A=z_A + length_A,
                        metadata={"raytem": dict(record.raw), "section": record.section},
                    )
                )
            current_A = z_A + length_A

        elif include_apertures and _is_aperture(record.kind):
            if z_A > current_A:
                elements.append(FreeSpace(z_A - current_A, name=f"to {record.name or record.kind}"))
                current_A = z_A
            radius = record.raw.get("radius")
            if radius is not None:
                elements.append(
                    Aperture(
                        radius=float(radius) * scale,
                        space=aperture_space,
                        name=record.name,
                        z_A=z_A,
                        metadata={"raytem": dict(record.raw), "section": record.section},
                    )
                )
            if aberrations:
                elements.append(
                    Aberration(
                        aberrations,
                        name=record.name or f"{record.kind} aberrations",
                        z_A=z_A,
                        metadata={"raytem": dict(record.raw), "section": record.section},
                    )
                )

        elif aberrations:
            if z_A > current_A:
                elements.append(FreeSpace(z_A - current_A, name=f"to {record.name or record.kind}"))
                current_A = z_A
            elements.append(
                Aberration(
                    aberrations,
                    name=record.name or f"{record.kind} aberrations",
                    z_A=z_A,
                    metadata={"raytem": dict(record.raw), "section": record.section},
                )
            )

    if stop_A > current_A:
        elements.append(FreeSpace(stop_A - current_A, name="to stop"))

    return OpticalColumn(
        elements=elements,
        name=name,
        metadata={
            "source": "raytem",
            "start": start,
            "stop": stop,
            "start_at": start_at,
            "stop_at": stop_at,
            "start_z_raytem": start_z,
            "stop_z_raytem": stop_z,
            "length_scale_A": scale,
            "raytem_length_unit": "mm",
            "include_aberrations": include_aberrations,
        },
    )


def simulate_raytem_wave(
    path_or_data: str | Path | Mapping[str, Any] | Any,
    *,
    extent_A: float | tuple[float, float],
    sampling_A: float | tuple[float, float],
    voltage_eV: float,
    convergence_mrad: float = 0.0,
    defocus_A: float = 0.0,
    positions_A: Optional[list[tuple[float, float]]] = None,
    start: str | float | int | None = None,
    stop: str | float | int | None = None,
    start_at: AnchorSide = "exit",
    stop_at: AnchorSide = "entrance",
    record: bool = True,
    device: Optional[str] = None,
) -> "WavePropagation":
    """Propagate a standalone coherent wave through a RayTEM configuration.

    RayTEM defines the column geometry and calibrated first-order elements. The
    wave source is explicit because a ray bundle does not uniquely determine a
    coherent field. Named planes are recorded by default and can be retrieved
    with ``result.plane(name)``. Set ``record=False`` to retain only the output
    wave when the full plane history would use too much memory.
    """
    from ..postprocessing.wf_data import WFData

    wave = WFData.from_probe(
        extent_A=extent_A,
        sampling=sampling_A,
        voltage_eV=voltage_eV,
        aperture=convergence_mrad,
        defocus=defocus_A,
        probe_positions=positions_A,
        device=device,
    )
    column = optical_column_from_raytem(
        path_or_data,
        start=start,
        stop=stop,
        start_at=start_at,
        stop_at=stop_at,
    )
    result = column.propagate(wave, record=record)
    report = result.sampling_report()
    real_edge = report["max_real_edge_power_fraction"]
    reciprocal_edge = report["max_reciprocal_edge_power_fraction"]
    if real_edge > 1e-2 or reciprocal_edge > 1e-2:
        warnings.warn(
            "The RayTEM wave approaches a grid boundary "
            f"(real={real_edge:.3g}, reciprocal={reciprocal_edge:.3g}). "
            "Increase extent_A or refine sampling_A and repeat the simulation.",
            RuntimeWarning,
        )
    return result


__all__ = [
    "simulate_raytem_wave",
]

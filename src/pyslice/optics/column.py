"""Containers for applying PySlice wave-optics elements."""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .elements import OpticalElement


def _snapshot_wave(wf):
    """Copy mutable wave state without copying backend module objects."""
    snapshot = copy(wf)
    backend = getattr(wf, "_backend", None)
    for attribute in (
        "_array",
        "_kxs",
        "_kys",
        "_xs",
        "_ys",
        "_time",
        "_layer",
    ):
        value = getattr(wf, attribute, None)
        if value is None:
            continue
        if backend is not None:
            value = backend.clone(value)
        elif hasattr(value, "copy"):
            value = value.copy()
        setattr(snapshot, attribute, value)
    for attribute in (
        "dimensions",
        "metadata",
        "_original_metadata",
        "probability",
        "buckets",
        "probe_positions",
        "probe_xs",
        "probe_ys",
    ):
        if hasattr(wf, attribute):
            setattr(snapshot, attribute, deepcopy(getattr(wf, attribute)))
    return snapshot


@dataclass
class WavePlane:
    """Recorded wave immediately after one optical-column operation."""

    name: str
    wave: Any
    element: Optional[OpticalElement] = None
    z_A: Optional[float] = None


@dataclass
class WavePropagation:
    """Output wave and optional source-to-detector plane history."""

    output: Any
    planes: list[WavePlane] = field(default_factory=list)
    column: Optional["OpticalColumn"] = None

    def plane(self, name: str) -> WavePlane:
        """Return the first recorded plane named ``name``."""
        for plane in self.planes:
            if plane.name == name:
                return plane
        available = [plane.name for plane in self.planes]
        raise KeyError(f"Wave plane {name!r} was not recorded. Available: {available}")

    def sampling_report(self) -> dict[str, Any]:
        """Summarize real- and reciprocal-edge power across recorded planes."""
        planes = self.planes or [WavePlane("output", self.output)]
        reports = []
        for plane in planes:
            report = plane.wave.sampling_report()
            reports.append({"name": plane.name, "z_A": plane.z_A, **report})
        worst_real = max(reports, key=lambda item: item["real_edge_power_fraction"])
        worst_reciprocal = max(
            reports, key=lambda item: item["reciprocal_edge_power_fraction"]
        )
        return {
            "max_real_edge_power_fraction": worst_real["real_edge_power_fraction"],
            "max_real_edge_plane": worst_real["name"],
            "max_reciprocal_edge_power_fraction": worst_reciprocal[
                "reciprocal_edge_power_fraction"
            ],
            "max_reciprocal_edge_plane": worst_reciprocal["name"],
        }


@dataclass
class OpticalColumn:
    """Ordered wave-optics column applied to a ``WFData`` object.

    ``apply`` mutates the supplied wavefunction in place, mirroring PySlice's
    existing propagation methods, and returns the same object for chaining.
    """

    elements: list[OpticalElement] = field(default_factory=list)
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, element: OpticalElement) -> OpticalElement:
        self.elements.append(element)
        return element

    def extend(self, elements: Iterable[OpticalElement]) -> None:
        self.elements.extend(elements)

    def apply(self, wf):
        for element in self.elements:
            element.apply(wf)
        return wf

    def propagate(self, wf, *, record: bool = False) -> WavePropagation:
        """Propagate a wave, optionally retaining every optical plane."""
        planes = []
        scale = float(self.metadata.get("length_scale_A", 1.0))
        start_z = self.metadata.get("start_z_raytem")
        current_z_A = None if start_z is None else float(start_z) * scale
        if record:
            planes.append(
                WavePlane(
                    "source",
                    _snapshot_wave(wf),
                    z_A=current_z_A,
                )
            )
        for element in self.elements:
            element.apply(wf)
            if hasattr(element, "dz_A") and current_z_A is not None:
                current_z_A += float(
                    getattr(element, "metadata", {}).get(
                        "physical_dz_A", element.dz_A
                    )
                )
            elif getattr(element, "z_A", None) is not None:
                element_exit_A = float(element.z_A) + float(
                    getattr(element, "thickness_A", 0.0)
                )
                # Several operators may act at one physical plane. A phase
                # operation following a thick element must not move the
                # recorded coordinate back from its exit to its entrance.
                if current_z_A is None or element_exit_A > current_z_A:
                    current_z_A = element_exit_A
            if record:
                planes.append(
                    WavePlane(
                        getattr(element, "name", "") or type(element).__name__,
                        _snapshot_wave(wf),
                        element=element,
                        z_A=current_z_A,
                    )
                )
        return WavePropagation(output=wf, planes=planes, column=self)

    def summary(self) -> str:
        """Return a compact human-readable list of optical operations."""
        lines = [self.name or "OpticalColumn"]
        for i, element in enumerate(self.elements):
            label = getattr(element, "name", "") or type(element).__name__
            pieces = [f"{i:02d}", type(element).__name__, label]
            for attr in (
                "dz_A",
                "f_A",
                "thickness_A",
                "principal_plane_drift_A",
                "rotation_rad",
                "theta_x_rad",
                "theta_y_rad",
                "radius",
                "z_A",
            ):
                if hasattr(element, attr):
                    value = getattr(element, attr)
                    if value is not None:
                        pieces.append(f"{attr}={value}")
            lines.append("  " + " | ".join(pieces))
        return "\n".join(lines)

    @classmethod
    def from_raytem(cls, *args, **kwargs) -> "OpticalColumn":
        """Build an optical column from a RayTEM JSON/SEA configuration."""
        from .raytem import optical_column_from_raytem

        return optical_column_from_raytem(*args, **kwargs)


__all__ = ["WavePlane", "WavePropagation", "OpticalColumn"]

"""Attach synthetic local aberrations to every MACSTEM round lens.

The values are illustrative, not an instrument calibration. RayTEM stores
lengths in millimeters; the imported ``Lens`` objects expose Cnm coefficients
in Angstroms.

Example
-------
PYTHONPATH=src python examples/raytem_column_aberrations.py macstem.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from pyslice.optics.elements import Lens
from pyslice.optics.raytem import optical_column_from_raytem


ANGSTROM_PER_MM = 1e7

SYNTHETIC_LENS_ABERRATIONS_A = {
    "CL1": {"C10": 20_000.0, "C12": (150.0, math.radians(5.0))},
    "CL2": {"C21": (15_000.0, math.radians(-20.0))},
    "CL3": {"C23": (12_000.0, math.radians(30.0)), "C30": 800_000.0},
    "OL1": {"C12": (300.0, math.radians(15.0)), "C30": 1_500_000.0},
    "OL2": {"C12": (250.0, math.radians(-10.0)), "C30": 2_000_000.0},
    "PL1": {"C21": (40_000.0, math.radians(25.0))},
    "PL2": {"C12": (350.0, math.radians(40.0))},
    "PL3": {"C23": (35_000.0, math.radians(-20.0))},
    "PL4": {"C12": (200.0, math.radians(5.0)), "C30": 4_000_000.0},
}


def _raytem_value(value):
    """Convert an Angstrom Cnm value to RayTEM's millimeter representation."""
    if isinstance(value, tuple):
        return [value[0] / ANGSTROM_PER_MM, value[1]]
    return value / ANGSTROM_PER_MM


def aberrated_column_config(config_path: Path) -> dict:
    """Return a copied RayTEM mapping with one Cnm model on every round lens."""
    config = deepcopy(json.loads(config_path.read_text()))
    found = set()
    for section in config.get("Sections", []):
        for element in section.get("Elements", []):
            name = element.get("Element name")
            coefficients = SYNTHETIC_LENS_ABERRATIONS_A.get(name)
            if coefficients is None:
                continue
            element["aberrations"] = {
                key: _raytem_value(value) for key, value in coefficients.items()
            }
            element["aberration_plane"] = "exit"
            found.add(name)
    missing = set(SYNTHETIC_LENS_ABERRATIONS_A) - found
    if missing:
        raise ValueError(f"RayTEM config is missing expected lenses: {sorted(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="MACSTEM RayTEM JSON")
    parser.add_argument(
        "--write-config",
        type=Path,
        help="Optionally write the decorated RayTEM JSON",
    )
    args = parser.parse_args()

    config = aberrated_column_config(args.config)
    column = optical_column_from_raytem(config, start="gun", stop="CCD")
    modeled_lenses = {
        element.name
        for element in column.elements
        if isinstance(element, Lens) and element.aberrations
    }
    expected = set(SYNTHETIC_LENS_ABERRATIONS_A)
    if modeled_lenses != expected:
        raise RuntimeError(
            f"Imported lens models differ: expected {sorted(expected)}, "
            f"found {sorted(modeled_lenses)}"
        )
    if args.write_config is not None:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(json.dumps(config, indent=2) + "\n")

    print(column.summary())


if __name__ == "__main__":
    main()

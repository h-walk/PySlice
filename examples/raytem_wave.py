"""Propagate a standalone coherent wave through a RayTEM configuration.

Example
-------
PYSLICE_BACKEND=numpy python examples/raytem_wave.py microscope.json gun CCD
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyslice import simulate_raytem_wave


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="RayTEM JSON or SEA file")
    parser.add_argument("start", help="RayTEM source/start element name")
    parser.add_argument("stop", help="RayTEM observation-plane element name")
    parser.add_argument("--voltage-ev", type=float, default=100e3)
    parser.add_argument("--extent-a", type=float, default=4096.0)
    parser.add_argument("--sampling-a", type=float, default=8.0)
    parser.add_argument("--convergence-mrad", type=float, default=0.05)
    args = parser.parse_args()

    result = simulate_raytem_wave(
        args.config,
        start=args.start,
        stop=args.stop,
        voltage_eV=args.voltage_ev,
        extent_A=args.extent_a,
        sampling_A=args.sampling_a,
        convergence_mrad=args.convergence_mrad,
    )

    print(result.column.summary())
    print(result.sampling_report())
    result.output.plot_realspace()


if __name__ == "__main__":
    main()

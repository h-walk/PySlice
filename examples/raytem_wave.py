"""Propagate a coherent beam through a RayTEM column.

Example
-------
PYSLICE_BACKEND=numpy python examples/raytem_wave.py microscope.json gun CCD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyslice import (
    GaussianWaveSource,
    ProbeAberrationModel,
    simulate_raytem_wave,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="RayTEM JSON or SEA file")
    parser.add_argument("start", help="RayTEM source/start element name")
    parser.add_argument("stop", help="RayTEM observation-plane element name")
    parser.add_argument("--voltage-ev", type=float, default=100e3)
    parser.add_argument("--extent-a", type=float, default=4096.0)
    parser.add_argument("--sampling-a", type=float, default=8.0)
    parser.add_argument("--convergence-mrad", type=float, default=0.05)
    parser.add_argument("--rms-size-a", type=float)
    parser.add_argument("--curvature-inv-a", type=float, default=0.0)
    parser.add_argument(
        "--probe-aberrations-json",
        type=Path,
        help="JSON keyword arguments for ProbeAberrationModel",
    )
    args = parser.parse_args()

    source = None
    if args.rms_size_a is not None:
        source = GaussianWaveSource(
            voltage_eV=args.voltage_ev,
            rms_size_A=args.rms_size_a,
            curvature_inv_A=args.curvature_inv_a,
        )
    probe_aberrations = None
    if args.probe_aberrations_json is not None:
        probe_aberrations = ProbeAberrationModel(
            **json.loads(args.probe_aberrations_json.read_text())
        )

    result = simulate_raytem_wave(
        args.config,
        start=args.start,
        stop=args.stop,
        voltage_eV=None if source is not None else args.voltage_ev,
        source=source,
        probe_aberrations=probe_aberrations,
        extent_A=args.extent_a,
        sampling_A=args.sampling_a,
        convergence_mrad=0.0 if source is not None else args.convergence_mrad,
        record=False,
    )

    print(result.column.summary())
    print(result.sampling_report())
    result.output.plot_realspace()


if __name__ == "__main__":
    main()

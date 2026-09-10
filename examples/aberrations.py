"""
Probe Aberrations: Effect on HAADF-STEM
=======================================

Show how lens aberrations affect HAADF imaging by comparing an ideal
probe against one with spherical aberration and 2-fold astigmatism.

Aberrations are applied after setup using calc.base_probe.aberrate().
The argument is a dict of Cnm coefficients (following abTEM convention):

  Rotationally symmetric:   "C30": value           (e.g. spherical aberration in Å)
  With orientation angle:   "C12": (value, angle)   (e.g. 2-fold astigmatism)

Common coefficients:
  C10 — defocus                   C12 — 2-fold astigmatism
  C21 — axial coma                C23 — 3-fold astigmatism
  C30 — 3rd-order spherical       C32 — axial star
  C34 — 4-fold astigmatism

Input file:
    tests/inputs/hBN_monolayer.cif

Level: intermediate
Expected scale: minutes; GPU recommended

The 0.05 Å sampling resolves the full 60–200 mrad detector at 100 keV. The
example integrates that detector during propagation instead of storing a
large 4D-STEM cube.
"""

import os
from pathlib import Path
import numpy as np
from pyslice import Loader, MultisliceCalculator, HAADFData

os.makedirs("outputs", exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 1. Build an hBN frozen-phonon supercell
# ---------------------------------------------------------------------------
trajectory = Loader(PROJECT_ROOT / "tests/inputs/hBN_monolayer.cif").load()
trajectory = trajectory.fold_positions_to_orthogonal_box()
trajectory = trajectory.tile_positions([6, 6, 1])
trajectory = trajectory.generate_random_displacements(
    n_displacements=4, sigma=0.1, seed=0,
)
print(f"Supercell: {trajectory.n_atoms} atoms, {trajectory.n_frames} frozen-phonon frames")

a = 2.491   # hBN lattice parameter (Å)
b = 2.157   # inter-row spacing (= a√3/2)
probe_xs = np.linspace(a, 4 * a, 8, endpoint=False)
probe_ys = np.linspace(b, 4 * b, 8, endpoint=False)

# ---------------------------------------------------------------------------
# 2. Ideal probe — no aberrations
# ---------------------------------------------------------------------------
print("=" * 60)
print("Ideal probe (no aberrations)")
print("=" * 60)

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=30, voltage_eV=100e3, sampling=0.05, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    ADF=(60, 200), return_layers=None,
    cache_wavefunctions=False,
    loop_probes=16,
)

_, haadf_ideal = calc.run()
haadf_ideal.plot("outputs/haadf_ideal.png")
print("Saved ideal HAADF")

# ---------------------------------------------------------------------------
# 3. Aberrated probe — spherical aberration + 2-fold astigmatism
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Aberrated probe (C30 + C12)")
print("=" * 60)

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=30, voltage_eV=100e3, sampling=0.05, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    ADF=(60, 200), return_layers=None,
    cache_wavefunctions=False,
    loop_probes=16,
)

# Apply aberrations to the probe before running
calc.base_probe.aberrate({
    "C30": 1e3,              # 1000 Å spherical aberration
    "C12": (1e2, 0.0),       # 100 Å 2-fold astigmatism at 0°
})

_, haadf_aberr = calc.run()
haadf_aberr.plot("outputs/haadf_aberrated.png")
print("Saved aberrated HAADF")

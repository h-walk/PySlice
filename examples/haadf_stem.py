"""
HAADF-STEM & CBED: Frozen Phonon vs Real MD Trajectory
=======================================================

Compare two approaches to multislice simulation:

  Part A — Static structure: load a CIF, tile, add Gaussian frozen-phonon
           displacements → CBED + HAADF
  Part B — Real dynamics: load a LAMMPS MD trajectory with true thermal
           motion → CBED + HAADF

Same scan grid and microscope parameters in both cases so the results are
directly comparable. The dense scans use on-the-fly detector integration:
they do not retain a 4D-STEM diffraction cube.

Level: intermediate
Expected scale: minutes rather than a smoke test; GPU recommended

The 0.05 Å real-space sampling represents the full 60–200 mrad detector at
100 keV. Increase the scan and configuration counts only after estimating the
output and convergence cost; see docs/user-guide/stem-and-4dstem.md.

Input files:
    tests/inputs/hBN_monolayer.cif
    tests/inputs/hBN_truncated.lammpstrj
"""

import os
from pathlib import Path
import numpy as np
from pyslice import Loader, MultisliceCalculator, HAADFData

os.makedirs("outputs", exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# hBN lattice parameters (Å)
a = 2.491   # hexagonal lattice parameter
b = 2.157   # inter-row spacing along y (= a√3/2)

# Small default scan. endpoint=False avoids duplicating periodic boundaries.
probe_xs = np.linspace(a, 4 * a, 8, endpoint=False)
probe_ys = np.linspace(b, 4 * b, 8, endpoint=False)

# ============================= PART A ======================================
# Frozen phonon from CIF — static structure with random displacements
# ===========================================================================
print("=" * 60)
print("PART A: CIF + frozen phonon")
print("=" * 60)

trajectory_cif = Loader(PROJECT_ROOT / "tests/inputs/hBN_monolayer.cif").load()
trajectory_cif = trajectory_cif.fold_positions_to_orthogonal_box()
trajectory_cif = trajectory_cif.tile_positions([6, 6, 1])
trajectory_cif = trajectory_cif.generate_random_displacements(
    n_displacements=4, sigma=0.1, seed=0,
)
print(f"CIF supercell: {trajectory_cif.n_atoms} atoms, {trajectory_cif.n_frames} frozen-phonon frames")

calc = MultisliceCalculator()
calc.setup(
    trajectory_cif,
    aperture=30, voltage_eV=100e3, sampling=0.05, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    ADF=(60, 200), return_layers=None,
    cache_wavefunctions=False,
    loop_probes=16,
)

_, haadf_cif = calc.run()
haadf_cif.plot("outputs/haadf_frozen_phonon.png")
print("Saved HAADF (frozen phonon)")

# ============================= PART B ======================================
# Real MD trajectory — true correlated thermal motion
# ===========================================================================
print()
print("=" * 60)
print("PART B: LAMMPS MD trajectory")
print("=" * 60)

trajectory_md = Loader(
    PROJECT_ROOT / "tests/inputs/hBN_truncated.lammpstrj",
    timestep=0.005,
    atom_mapping={1: "B", 2: "N"},
).load()

trajectory_md = trajectory_md.fold_positions_to_orthogonal_box()
trajectory_md = trajectory_md.slice_positions([0, 6 * a], [0, 6 * b])
trajectory_md = trajectory_md.get_random_timesteps(4, seed=5)
print(f"MD trajectory: {trajectory_md.n_atoms} atoms, {trajectory_md.n_frames} frames")

calc = MultisliceCalculator()
calc.setup(
    trajectory_md,
    aperture=30, voltage_eV=100e3, sampling=0.05, slice_thickness=0.5,
    probe_xs=probe_xs, probe_ys=probe_ys,
    ADF=(60, 200), return_layers=None,
    cache_wavefunctions=False,
    loop_probes=16,
)

_, haadf_md = calc.run()
haadf_md.plot("outputs/haadf_md_trajectory.png")
print("Saved HAADF (MD trajectory)")

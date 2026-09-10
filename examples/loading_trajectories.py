"""
Loading Trajectories
====================

PySlice can load atomic structures from many file formats. This example
demonstrates the Loader API and useful Trajectory manipulation methods.

Supported formats:
  - CIF (.cif)
  - XYZ (.xyz)
  - LAMMPS dump (.lammpstrj)
  - LAMMPS positions (.positions)
  - ASE trajectory (.traj / .trj)
  - ASE Atoms objects (in-memory)

Input files:
    tests/inputs/hBN_cif.cif
    tests/inputs/silicon_xyz.xyz
    tests/inputs/hBN_truncated.lammpstrj
    tests/inputs/hBN_GAP_ase.trj
"""

from pathlib import Path
import numpy as np
from ase.build import bulk
from pyslice import Loader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "tests/inputs"

# =========================================================================
# 1. CIF file — single-frame crystal structure
# =========================================================================
traj = Loader(INPUT_DIR / "hBN_cif.cif").load()
print("--- CIF ---")
print(f"  Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")
print(f"  Box matrix (Å):\n{traj.box_matrix}")

# =========================================================================
# 2. XYZ file
# =========================================================================
traj = Loader(INPUT_DIR / "silicon_xyz.xyz").load()
print("\n--- XYZ ---")
print(f"  Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")

# =========================================================================
# 3. LAMMPS trajectory — needs atom_mapping for element identification
# =========================================================================
traj = Loader(
    INPUT_DIR / "hBN_truncated.lammpstrj",
    timestep=0.005,                    # ps
    atom_mapping={1: "B", 2: "N"},     # Map LAMMPS type IDs to elements
).load()
print("\n--- LAMMPS dump ---")
print(f"  Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")
print(f"  Timestep: {traj.timestep} ps")
# Note: atom_types are stored as atomic numbers (B=5, N=7), not strings
print(f"  Atom types (unique): {np.unique(traj.atom_types)}")

# =========================================================================
# 4. ASE trajectory file
# =========================================================================
traj = Loader(INPUT_DIR / "hBN_GAP_ase.trj").load()
print("\n--- ASE trajectory ---")
print(f"  Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")

# =========================================================================
# 5. ASE Atoms object (in-memory, no file needed)
# =========================================================================
atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (3, 3, 3)
traj = Loader(atoms=atoms).load()
print("\n--- ASE Atoms ---")
print(f"  Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")

# =========================================================================
# Trajectory manipulation methods
# =========================================================================
print("\n=== Trajectory Methods ===")

# Reload the LAMMPS trajectory for demonstration
traj = Loader(
    INPUT_DIR / "hBN_truncated.lammpstrj",
    timestep=0.005,
    atom_mapping={1: "B", 2: "N"},
).load()
# Spatial transforms require an axis-aligned box. This explicit fold replaces
# the tilted LAMMPS cell with its intended orthogonal periodic representation.
traj = traj.fold_positions_to_orthogonal_box()

# --- Spatial cropping ---
cropped = traj.slice_positions([0, 10], [0, 10])
print(f"\nslice_positions([0,10], [0,10]): {traj.n_atoms} → {cropped.n_atoms} atoms")

# --- Random timestep selection (frozen phonon sampling) ---
subset = traj.get_random_timesteps(5, seed=42)
print(f"get_random_timesteps(5): {traj.n_frames} → {subset.n_frames} frames")

# --- Tiling (supercell construction) ---
traj_cif = Loader(INPUT_DIR / "hBN_cif.cif").load()
tiled = traj_cif.tile_positions([5, 5, 1])
print(f"tile_positions([5,5,1]): {traj_cif.n_atoms} → {tiled.n_atoms} atoms")

# --- Frozen-phonon displacements from a static structure ---
phonons = tiled.generate_random_displacements(n_displacements=10, sigma=0.1, seed=0)
print(f"generate_random_displacements(10): {tiled.n_frames} → {phonons.n_frames} frames")

# --- Mean positions (for displacement analysis) ---
mean_pos = traj.get_mean_positions()
print(f"get_mean_positions(): shape = {mean_pos.shape}")

"""
Molecular Dynamics with an ML Potential
========================================

Run molecular dynamics using the ORB machine-learning interatomic potential,
producing a PySlice Trajectory ready for multislice simulation.

  Part A — Quick demo: small Si supercell, short run, full setup -> run -> inspect
  Part B — Production-quality settings: parameter choices for a real TACAW-ready
           trajectory (NVT equilibration -> NVE production)

No external input files needed — the structure is built from scratch with ASE.
"""

import os
import numpy as np
from ase.build import bulk
from pyslice import ORBMDCalculator, Trajectory, analyze_md_trajectory

os.makedirs("outputs/md_demo", exist_ok=True)
os.makedirs("outputs/md_production", exist_ok=True)

# ============================= PART A ======================================
# Quick demo — small supercell, short run, inspect the results
# ===========================================================================
print("=" * 70)
print("PART A: Quick MD Demo")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Build a silicon supercell with ASE
# ---------------------------------------------------------------------------
atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True) * (3, 3, 2)
print(f"Created Si supercell: {len(atoms)} atoms")

# ---------------------------------------------------------------------------
# 2. Set up the MD calculator with an ORB potential
# ---------------------------------------------------------------------------
md = ORBMDCalculator(model_name="orb-v3-conservative-inf-omat")

md.setup(
    atoms=atoms,
    temperature=300,              # K
    timestep=2.0,                 # fs — safe for Si (max phonon ~15 THz)
    ensemble="nvt",               # Langevin thermostat for equilibration
    friction=0.05,                # fs^-1 — moderate damping
    max_equilibration_steps=500,  # short for demo
    min_equilibration_steps=100,
    production_steps=200,         # short for demo
    save_interval=2,              # save every 2 steps -> 100 frames
    output_dir="outputs/md_demo",
    rng=np.random.default_rng(7),
)

# ---------------------------------------------------------------------------
# 3. Run the simulation
# ---------------------------------------------------------------------------
print("\nRunning equilibration + production...")
trajectory = md.run()

# ---------------------------------------------------------------------------
# 4. Inspect the trajectory
# ---------------------------------------------------------------------------
print(f"\nTrajectory summary:")
print(f"  Frames:    {trajectory.n_frames}")
print(f"  Atoms:     {trajectory.n_atoms}")
print(f"  Box (Ang): {trajectory.box_matrix[0,0]:.2f} x "
      f"{trajectory.box_matrix[1,1]:.2f} x {trajectory.box_matrix[2,2]:.2f}")
print(f"  Timestep:  {trajectory.timestep} ps")

# The returned Trajectory is directly usable for multislice:
#   calc = MultisliceCalculator()
#   calc.setup(trajectory, voltage_eV=100e3, sampling=0.1, ...)
#   wf = calc.run()

# ---------------------------------------------------------------------------
# 5. Plot thermodynamic diagnostics
# ---------------------------------------------------------------------------
print("\nGenerating analysis plots...")
analyze_md_trajectory(
    trajectory_file="outputs/md_demo/production.traj",
    log_file="outputs/md_demo/production.log",
    output_file="outputs/md_demo/md_analysis.png",
)
print("Saved outputs/md_demo/md_analysis.png")

# ============================= PART B ======================================
# Production-quality settings — TACAW-ready trajectory
# ===========================================================================
print()
print("=" * 70)
print("PART B: Production-Quality MD for TACAW")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Larger supercell for realistic phonon sampling
# ---------------------------------------------------------------------------
atoms_prod = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True) * (6, 6, 2)
print(f"Created Si supercell: {len(atoms_prod)} atoms")

# ---------------------------------------------------------------------------
# 2. Configure with production-quality parameters
# ---------------------------------------------------------------------------
# Key parameter choices for TACAW:
#
#   timestep:  2 fs — Nyquist limit is 1/(2*dt) = 250 THz, well above
#              the ~15 THz Si optical phonon.  Must be small enough
#              to resolve the fastest phonons of interest.
#
#   save_interval:  5 steps (10 fs between frames).  The effective
#              Nyquist is 1/(2*10fs) = 50 THz — still plenty for Si.
#              Fewer saved frames = smaller memory footprint.
#
#   production_steps:  5000 (total 10 ps).  Frequency resolution is
#              1 / T_total = 1 / 10 ps = 0.1 THz — good enough for
#              resolving individual phonon branches.
#
#   Equilibration (NVT + Langevin):  friction=0.05 fs^-1 gives fast
#              thermalization without over-damping.
#
#   Production (NVE):  No thermostat noise → clean phonon dynamics.
#              production_relaxation_steps lets the system forget the
#              thermostat kick before recording begins.
#
md_prod = ORBMDCalculator(model_name="orb-v3-conservative-inf-omat")

md_prod.setup(
    atoms=atoms_prod,
    temperature=300,                    # K
    timestep=2.0,                       # fs
    # --- Equilibration ---
    ensemble="nvt",                     # Langevin thermostat
    friction=0.05,                      # fs^-1
    min_equilibration_steps=500,
    max_equilibration_steps=5000,
    temp_tolerance=20.0,                # K — mean T must be within this
    temp_threshold=15.0,                # K — T fluctuations must be below this
    energy_threshold=0.01,              # relative energy stability
    # --- Production ---
    production_ensemble="nve",          # no thermostat noise
    production_relaxation_steps=200,    # 200 steps to forget thermostat kick
    production_steps=5000,              # 5000 * 2 fs = 10 ps total
    save_interval=5,                    # save every 10 fs -> 1000 frames
    output_dir="outputs/md_production",
    rng=np.random.default_rng(8),
)

print("\nParameter summary:")
print(f"  Equilibration: NVT, friction = 0.05 fs^-1, up to 5000 steps")
print(f"  Production:    NVE, 5000 steps x 2 fs = 10 ps")
print(f"  Save interval: every 5 steps -> ~1000 frames")
print(f"  Freq resolution: ~0.1 THz   |   Nyquist: 50 THz")

print("\nRunning equilibration + production (this may take a few minutes)...")
trajectory_prod = md_prod.run()

print(f"\nProduction trajectory:")
print(f"  Frames:    {trajectory_prod.n_frames}")
print(f"  Atoms:     {trajectory_prod.n_atoms}")
print(f"  Box (Ang): {trajectory_prod.box_matrix[0,0]:.2f} x "
      f"{trajectory_prod.box_matrix[1,1]:.2f} x {trajectory_prod.box_matrix[2,2]:.2f}")

# Diagnostics
analyze_md_trajectory(
    trajectory_file="outputs/md_production/production.traj",
    log_file="outputs/md_production/production.log",
    output_file="outputs/md_production/md_analysis.png",
)
print("Saved outputs/md_production/md_analysis.png")

# This trajectory is ready for multislice -> TACAW:
#
#   from pyslice import MultisliceCalculator, TACAWData
#
#   calc = MultisliceCalculator()
#   calc.setup(trajectory_prod, aperture=0, voltage_eV=100e3, sampling=0.1)
#   wf = calc.run()
#   tacaw = TACAWData(wf)
#   # -> spectral diffraction, phonon dispersion, spectrum images ...
#
# See tacaw_pipeline.py for the full multislice + TACAW workflow.

print()
print("=" * 70)
print("Done! Both trajectories are ready for multislice simulation.")
print("See tacaw_pipeline.py for the next steps (multislice -> TACAW).")
print("=" * 70)

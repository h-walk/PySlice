"""
LACBED (Large-Angle Convergent-Beam Electron Diffraction)
=========================================================

Simulate a LACBED pattern from a silicon trajectory.

The key idea: defocus the probe, propagate the exit wave through free
space, and apply a real-space selected-area aperture to produce a
large-angle convergent-beam pattern in reciprocal space.

Steps:
  1. Load a Si trajectory and tile along z for a thick specimen
  2. Run multislice with a defocused convergent probe
  3. Free-space propagation + real-space aperture
  4. Plot the LACBED pattern

Input file:
    tests/inputs/Si_truncated.lammpstrj
"""

import os
from pathlib import Path
import numpy as np
from pyslice import Loader, MultisliceCalculator

os.makedirs("outputs", exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 1. Load silicon trajectory and build a thick specimen
# ---------------------------------------------------------------------------
trajectory = Loader(
    PROJECT_ROOT / "tests/inputs/Si_truncated.lammpstrj",
    timestep=0.002,
    atom_mapping={1: "Si"},
).load()

print(f"Original: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")

# Tile along z using 10 different timesteps for thermal variation
slice_timesteps = np.arange(trajectory.n_frames)
np.random.seed(5)
np.random.shuffle(slice_timesteps)

trajectories = [trajectory.slice_timesteps(i, i + 1) for i in slice_timesteps[:10]]
trajectory = trajectories[0].tile_positions([1, 1, 10], trajectories)
print(f"After z-tiling: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")

# ---------------------------------------------------------------------------
# 2. Multislice with a defocused convergent probe
# ---------------------------------------------------------------------------
defocus_ang = -1000  # Å — large defocus spreads the probe over a wide area

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=30,               # 30 mrad convergence
    voltage_eV=100e3,
    sampling=0.1,
    slice_thickness=0.5,
)
calc.base_probe.defocus(defocus_ang)

exitwaves = calc.run()
print(f"Exit-wave shape: {exitwaves.array.shape}")

# ---------------------------------------------------------------------------
# 3. Post-processing: free-space propagation + real-space aperture
# ---------------------------------------------------------------------------
# Propagate to the detector plane (compensate for specimen thickness)
exitwaves.propagate_free_space(-defocus_ang - calc.lz)

# Apply a real-space selected-area aperture (5 Å radius)
exitwaves.applyMask(5, "real")

# ---------------------------------------------------------------------------
# 4. Plot the LACBED pattern in reciprocal space
# ---------------------------------------------------------------------------
exitwaves.plot_reciprocal("outputs/lacbed_pattern.png")
print("Saved LACBED pattern to outputs/lacbed_pattern.png")

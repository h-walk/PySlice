"""
TEM Diffraction: Structure → Potential → Diffraction Pattern
=============================================================

Level: getting started
Expected scale: seconds to a few minutes; CPU-capable; no downloads

The smallest canonical multislice workflow: illuminate one static structure
with a parallel electron beam and compute one diffraction pattern.

Along the way we visualize:
  1. The atomic structure (projected along the beam direction)
  2. The resulting electron diffraction pattern
  3. (Optional) The projected electrostatic potential — for diagnostic purposes

No external input files are needed — the structure is created from scratch.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from pyslice import Loader, MultisliceCalculator, Potential

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Build an hBN monolayer with ASE
# ---------------------------------------------------------------------------
# Orthorhombic unit cell (4 atoms) derived from the hexagonal lattice.
# This ensures a perfectly periodic orthogonal box for multislice.
a = 2.504      # hBN lattice parameter (Å)
b = a * np.sqrt(3)
c = 3.392      # z spacing (vacuum)

unit_cell = Atoms(
    symbols=["B", "N", "B", "N"],
    positions=[
        [0.0,   b / 3,     c / 2],
        [a / 2, b / 6,     c / 2],
        [a / 2, 5 * b / 6, c / 2],
        [0.0,   2 * b / 3, c / 2],
    ],
    cell=[a, b, c],
    pbc=True,
)
atoms = unit_cell * (6, 4, 2)  # 192 atoms, ~15 × 17 × 7 Å
print(f"Created hBN supercell: {len(atoms)} atoms, "
      f"box = {atoms.cell[0,0]:.1f} × {atoms.cell[1,1]:.1f} Å")

# Load one static structure. Add frozen-phonon frames only after this baseline
# calculation works and its sampling/slice-thickness convergence is understood.
trajectory = Loader(atoms=atoms).load()

# ---------------------------------------------------------------------------
# 2. Plot the atomic structure (xy projection, first layer only)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
colors = {"B": "#1f77b4", "N": "#ff7f0e"}
for atom_type in np.unique(trajectory.atom_types):
    mask = trajectory.atom_types == atom_type
    pos = trajectory.positions[0, mask]
    # Show only the first layer to avoid overlapping markers
    z_mid = pos[:, 2].min() + 0.5
    layer_mask = pos[:, 2] < z_mid
    pos = pos[layer_mask]
    label = str(atom_type)
    ax.scatter(pos[:, 0], pos[:, 1], c=colors.get(label, "gray"),
               s=60, label=label, edgecolors="k", linewidths=0.5)
ax.set_xlabel("x (Å)")
ax.set_ylabel("y (Å)")
ax.set_aspect("equal")
ax.legend()
ax.set_title("hBN monolayer (xy projection)")
fig.tight_layout()
fig.savefig("outputs/tem_structure.png", dpi=150)
plt.close(fig)
print("Saved structure plot")

# ---------------------------------------------------------------------------
# 3. Set up parallel-beam multislice
# ---------------------------------------------------------------------------
calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=0,        # Parallel beam (plane wave)
    voltage_eV=100e3,
    sampling=0.2,
    slice_thickness=0.5,
    cache_wavefunctions=False,
)

# ---------------------------------------------------------------------------
# 4. Run multislice and plot the diffraction pattern
# ---------------------------------------------------------------------------
# MultisliceCalculator builds the potential internally — you do NOT need to
# construct a Potential object yourself.  We do it below purely to visualize
# what the electrons scatter from.
wf = calc.run()
wf.plot_reciprocal(
    "outputs/tem_diffraction.png",
    nuke_zerobeam=True,
    powerscaling=0.5,
    extent=[-2, 2, -2, 2],
)
print("Saved diffraction pattern")

# ---------------------------------------------------------------------------
# 5. (Optional) Visualize the projected potential
# ---------------------------------------------------------------------------
# This is NOT part of the normal workflow — just a diagnostic visualization.
# The Potential class lets you inspect the electrostatic potential that the
# multislice algorithm scatters electrons from.
potential = Potential(
    calc.xs, calc.ys, calc.zs,
    trajectory.positions[0], list(trajectory.atom_types),
    kind="kirkland", device=calc.device, slice_axis=calc.slice_axis,
)
potential.build()
potential.plot(filename="outputs/tem_potential.png")
plt.close()
print("Saved projected potential (optional diagnostic)")

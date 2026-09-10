"""
TACAW from Trajectory: Load MD → Multislice → Spectral Diffraction
===================================================================

Run the TACAW workflow starting from a pre-existing MD trajectory.

Unlike tacaw_pipeline.py (which runs MD from scratch), this example loads
a LAMMPS dump file directly and proceeds to multislice + TACAW analysis.

Input file:
    tests/inputs/hBN_truncated.lammpstrj
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyslice import Loader, MultisliceCalculator, TACAWData

os.makedirs("outputs", exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 1. Load the MD trajectory
# ---------------------------------------------------------------------------
trajectory = Loader(
    PROJECT_ROOT / "tests/inputs/hBN_truncated.lammpstrj",
    timestep=0.005,  # ps between frames (5 fs)
    atom_mapping={1: "B", 2: "N"},
).load()
# This bundled LAMMPS cell is tilted. Multislice requires an axis-aligned box,
# so explicitly fold its periodic Cartesian representation before propagation.
trajectory = trajectory.fold_positions_to_orthogonal_box()

print(f"Trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")

# ---------------------------------------------------------------------------
# 2. Parallel-beam multislice over all frames
# ---------------------------------------------------------------------------
calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=0,            # Parallel beam (plane wave)
    voltage_eV=100e3,
    sampling=0.1,
    slice_thickness=0.5,
)

wf_data = calc.run()
print(f"Exit-wave shape: {wf_data.array.shape}")

# ---------------------------------------------------------------------------
# 3. TACAW: time → frequency domain
# ---------------------------------------------------------------------------
tacaw = TACAWData(wf_data)
freqs = tacaw.frequencies
df = freqs[1] - freqs[0]
print(f"Frequency range: {freqs[0]:.1f} – {freqs[-1]:.1f} THz  "
      f"({len(freqs)} bins, Δf = {abs(df):.1f} THz)")

# Spectral diffraction at the represented bin nearest 15 THz.
requested_THz = 15.0
selected_THz = tacaw.nearest_frequency(requested_THz)
print(f"Requested {requested_THz:.1f} THz; using {selected_THz:.1f} THz")
Z = tacaw.spectral_diffraction(selected_THz)
tacaw.plot(Z ** 0.1, "kx", "ky",
           extent=[-2, 2, -2, 2],
           filename=f"outputs/tacaw_hbn_{selected_THz:g}THz.png")
print(f"Saved spectral diffraction at {selected_THz:.1f} THz")

# ---------------------------------------------------------------------------
# 4. Phonon dispersion: Gamma → Gamma → Gamma
# ---------------------------------------------------------------------------
# The bundled cell repeats primitive in-plane vectors (a, -a/2) and
# (0, sqrt(3)a/2). Reciprocal vectors in PySlice use cycles/Å, so they are the
# rows of inv(A).T without an additional 2π factor. Here b1 = (1/a, 0), making
# -b1, 0, and +b1 reciprocal-lattice-equivalent Gamma points.
a_hbn = 2.491  # hBN lattice parameter (Å)
direct_lattice_xy = np.array([
    [a_hbn, -a_hbn / 2],
    [0.0, np.sqrt(3) * a_hbn / 2],
])
reciprocal_lattice_xy = np.linalg.inv(direct_lattice_xy).T
G_x = reciprocal_lattice_xy[0, 0]
kx_path = np.linspace(-G_x, G_x, 200)
ky_path = np.zeros_like(kx_path)
dispersion = tacaw.dispersion(kx_path, ky_path)

# Plot positive frequencies only
pos_mask = freqs >= 0
fig, ax = plt.subplots()
ax.imshow(
    np.abs(dispersion[pos_mask, :]) ** 0.125,
    cmap="inferno", aspect="auto", origin="lower",
    extent=[kx_path[0], kx_path[-1], freqs[pos_mask][0], freqs[pos_mask][-1]],
)
ax.set_xlabel("kx ($\\AA^{-1}$)")
ax.set_ylabel("frequency (THz)")
fig.savefig("outputs/tacaw_hbn_dispersion.png")
plt.close(fig)
print("Saved phonon dispersion Gamma → Gamma → Gamma")

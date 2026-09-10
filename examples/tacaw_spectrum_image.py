"""
TACAW Spectrum Imaging: Real-Space Phonon Map
==============================================

Map phonon intensity in real space using convergent-beam TACAW.

A focused STEM probe is scanned over a grid of positions. At each position
the exit wave is recorded across all MD frames, then Fourier-transformed
in time to extract intensity at a chosen phonon frequency.

This script loads the MD trajectory produced by tacaw_pipeline.py.
Run that example first to generate the trajectory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyslice import Loader, MultisliceCalculator, TACAWData

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the trajectory produced by tacaw_pipeline.py
# ---------------------------------------------------------------------------
# tacaw_pipeline.py saves every 5 steps from a 5 fs integrator: 0.025 ps/frame.
trajectory = Loader(
    "outputs/tacaw_pipeline_md/production.traj",
    timestep=0.025,
).load()
print(f"Loaded trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")

# ---------------------------------------------------------------------------
# 2. Convergent-beam multislice over a probe scan grid
# ---------------------------------------------------------------------------
a_si = 5.431  # Si lattice parameter (Å)
probe_xs = np.linspace(a_si, 4 * a_si, 18)
probe_ys = np.linspace(a_si, 4 * a_si, 18)

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=30,               # 30 mrad convergence (STEM mode)
    voltage_eV=100e3,
    sampling=0.1,
    slice_thickness=0.5,
    probe_xs=probe_xs,
    probe_ys=probe_ys,
)

wf_stem = calc.run()
print(f"Exit-wave shape: {wf_stem.array.shape}")

# ---------------------------------------------------------------------------
# 3. TACAW: extract phonon map at a chosen frequency
# ---------------------------------------------------------------------------
tacaw = TACAWData(wf_stem)

freq_THz = 10.0
selected_THz = tacaw.nearest_frequency(freq_THz)
phonon_map = tacaw.spectrum_image_reshaped(selected_THz)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(
    phonon_map,
    extent=[probe_ys[0], probe_ys[-1], probe_xs[0], probe_xs[-1]],
    cmap="inferno",
    origin="lower",
    aspect="equal",
    interpolation="bicubic",
)
ax.set_xlabel("y (Å)")
ax.set_ylabel("x (Å)")
ax.set_title(f"Phonon map at {selected_THz:g} THz")
fig.colorbar(im, ax=ax, label="Intensity (arb. u.)")
fig.tight_layout()
fig.savefig("outputs/tacaw_spectrum_image.png", dpi=150)
plt.close(fig)
print(f"Requested {freq_THz:g} THz; saved nearest bin {selected_THz:g} THz")

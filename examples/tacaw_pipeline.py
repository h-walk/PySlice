"""
TACAW Pipeline: MD → Multislice → Spectral Diffraction & Phonon Dispersion
===========================================================================

Complete PySlice workflow from scratch:
  1. Build a crystal structure with ASE
  2. Run molecular dynamics to generate a trajectory
  3. Parallel-beam multislice → exit waves
  4. TACAW: spectral diffraction at a chosen frequency + phonon dispersion

No external input files are needed — the structure is created from scratch.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from pyslice import ORBMDCalculator, MultisliceCalculator, TACAWData

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Create a silicon supercell with ASE
# ---------------------------------------------------------------------------
# Diamond-cubic Si, 10×10×2 supercell (800 atoms)
atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True) * (10, 10, 2)
print(f"Created Si supercell: {len(atoms)} atoms")

# ---------------------------------------------------------------------------
# 2. Run molecular dynamics with an ORB machine-learning potential
# ---------------------------------------------------------------------------
md = ORBMDCalculator(model_name="orb-v3-conservative-inf-omat")

md.setup(
    atoms=atoms,
    temperature=300,           # K
    timestep=5.0,              # fs
    ensemble="nvt",
    friction=0.2,              # Langevin thermostat friction (equilibration)
    production_ensemble="nvt",
    production_friction=0.01,  # Light friction for production
    production_relaxation_steps=100,
    temp_tolerance=5.0,        # K — convergence criterion
    temp_threshold=5.0,
    energy_threshold=0.05,     # dimensionless relative energy standard deviation
    production_steps=500,
    save_interval=5,           # Save every 5 steps → 100 frames
    output_dir="outputs/tacaw_pipeline_md",
    rng=np.random.default_rng(7),
)

trajectory = md.run()
print(f"MD trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")

# ---------------------------------------------------------------------------
# 3. Parallel-beam multislice
# ---------------------------------------------------------------------------
calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=0,                # Parallel beam (plane wave)
    voltage_eV=100e3,          # 100 keV electrons
    sampling=0.1,              # 0.1 Å/pixel
    slice_thickness=0.5,       # 0.5 Å slice thickness
)

wf_data = calc.run()
print(f"Parallel-beam exit-wave shape: {wf_data.array.shape}")

# Diffraction pattern (zero-beam removed, power-scaled)
wf_data.plot_reciprocal(
    "outputs/tacaw_diffraction.png",
    nuke_zerobeam=True, powerscaling=0.125,
)

# ---------------------------------------------------------------------------
# 4. TACAW: time → frequency domain
# ---------------------------------------------------------------------------
tacaw = TACAWData(wf_data)
freqs = tacaw.frequencies
print(f"Frequency range: 0 – {freqs[-1]:.1f} THz  ({len(freqs)} bins)")

# Spectral diffraction at 15 THz
Z = tacaw.spectral_diffraction(15.0)
tacaw.plot(Z ** 0.1, "kx", "ky", filename="outputs/tacaw_15THz.png")
print("Saved spectral diffraction at 15 THz")

# Phonon dispersion along X → Gamma → X. PySlice reciprocal
# coordinates are spatial frequencies in cycles/Å, so X is at 1/a here.
a_si = 5.431  # Si lattice parameter (Å)
kx_path = np.linspace(-1 / a_si, 1 / a_si, 200)
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
fig.savefig("outputs/tacaw_dispersion.png")
plt.close(fig)
print("Saved phonon dispersion X → Gamma → X")

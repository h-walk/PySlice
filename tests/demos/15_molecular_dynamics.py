"""
Molecular dynamics test using ORB force fields.

This test requires the orb-models package to be installed:
    pip install "pyslice[md]"

Set LOAD_FROM_CACHE = True to skip MD and load from previous run.
"""

import logging
import os
import sys

# Configure logging BEFORE importing other modules
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)

# Add parent directory to path for running without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import shutil

import numpy as np
from ase.build import bulk

from pyslice.io.loader import Loader
from pyslice.md import ORBMDCalculator
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.tacaw_data import TACAWData

# OPTION: Load trajectory from cache instead of running MD
# Set to True to skip MD and load from previous run
LOAD_FROM_CACHE = True
OUTPUT_DIR = "outputs/md_output"

if LOAD_FROM_CACHE and os.path.exists(f"{OUTPUT_DIR}/production.traj"):
    print("=" * 70)
    print("LOADING TRAJECTORY FROM CACHE")
    print("=" * 70)
    loader = Loader(
        filename=f"{OUTPUT_DIR}/production.traj", timestep=0.01
    )  # 5fs MD × 2 save_interval = 10fs = 0.01ps
    trajectory = loader.load()
    print(
        f"Loaded trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms"
    )
else:
    # Clean up previous MD outputs
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # CREATE TEST SYSTEM
    print("=" * 70)
    print("MOLECULAR DYNAMICS TEST")
    print("=" * 70)
    print("\nCreating test system (bulk Si [001])...")

    # Create bulk Si diamond cubic, a = 5.431 Å
    atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
    atoms = atoms * (20, 20, 1)  # 20x20x1 supercell = 3200 atoms
    print(f"System: {len(atoms)} atoms")
    print(f"Cell:\n{atoms.get_cell()}")

    # SETUP MD CALCULATOR
    # Auto-detect best device (CUDA > MPS > CPU)
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nInitializing ORBMDCalculator on {device}...")
    # Use most accurate ORB model for better phonon frequencies
    md_calc = ORBMDCalculator(
        model_name="orb-v3-conservative-inf-omat",
        device=device,
        precision="float32-highest",
    )

    # SETUP SIMULATION
    print("Setting up MD simulation...")
    md_calc.setup(
        atoms=atoms,
        temperature=300,
        timestep=5.0,
        ensemble="nvt",
        friction=0.2,  # Moderate friction for equilibration (less noise than 0.5)
        production_ensemble="nvt",  # Keep NVT for production (ORB doesn't conserve energy perfectly)
        production_friction=0.01,  # Very low friction for near-NVE dynamics with energy stabilization
        production_relaxation_steps=100,  # Let thermostat artifacts decay (100 steps = 500 fs)
        temp_tolerance=5.0,  # Tolerance for mean temperature (K)
        temp_threshold=5.0,  # Threshold for temperature std dev (K)
        energy_threshold=0.05,
        min_equilibration_steps=100,
        max_equilibration_steps=10000,
        production_steps=1000,  # 1000 steps * 5fs = 5ps total, 500 frames @ 0.01ps = 0.2 THz resolution
        check_interval=10,
        save_interval=2,
        output_dir=OUTPUT_DIR,
    )

    # RUN MD SIMULATION
    print("\nRunning MD simulation... (this may take a while)")
    trajectory = md_calc.run()

# VALIDATE TRAJECTORY
print("\n" + "=" * 70)
print("VALIDATING TRAJECTORY")
print("=" * 70)

assert trajectory.n_frames > 0
print(f"Trajectory has {trajectory.n_frames} frames")

assert trajectory.n_atoms > 0
print(f"Trajectory has {trajectory.n_atoms} atoms")

assert trajectory.positions.shape == (trajectory.n_frames, trajectory.n_atoms, 3)
print(f"Positions shape: {trajectory.positions.shape}")

assert trajectory.velocities.shape == (trajectory.n_frames, trajectory.n_atoms, 3)
print(f"Velocities shape: {trajectory.velocities.shape}")

assert trajectory.box_matrix.shape == (3, 3)
print(f"Box matrix shape: {trajectory.box_matrix.shape}")

assert trajectory.timestep > 0
print(f"Timestep: {trajectory.timestep} ps")

# CHECK TEMPERATURE FROM ASE TRAJECTORY
from ase.io import read as aseread

traj_file = f"{OUTPUT_DIR}/production.traj"
ase_traj = aseread(traj_file, index=":")
temps = [atoms_snap.get_temperature() for atoms_snap in ase_traj]
mean_temp = np.mean(temps)
print(f"Mean temperature: {mean_temp:.1f} K (target: 300 K)")

# CHECK FILES EXIST
assert os.path.exists(f"{OUTPUT_DIR}/production.traj")
print(f"Production trajectory exists")

assert os.path.exists(f"{OUTPUT_DIR}/production.log")
print(f"Production log exists")

assert os.path.exists(f"{OUTPUT_DIR}/equilibration.traj")
print(f"Equilibration trajectory exists")

assert os.path.exists(f"{OUTPUT_DIR}/equilibration.log")
print(f"Equilibration log exists")

# TEST TRAJECTORY METHODS
print("\n" + "=" * 70)
print("TESTING TRAJECTORY METHODS")
print("=" * 70)

mean_pos = trajectory.get_mean_positions()
assert mean_pos.shape == (trajectory.n_atoms, 3)
print(f"get_mean_positions: {mean_pos.shape}")

# TEST CONVERSION VIA LOADER
print("\n" + "=" * 70)
print("TESTING ASE TO PYSLICE CONVERSION")
print("=" * 70)

loader = Loader(
    filename=traj_file, timestep=0.01
)  # 5fs MD × 2 save_interval = 10fs = 0.01ps
loaded_trajectory = loader.load()
assert loaded_trajectory.n_frames == trajectory.n_frames
assert loaded_trajectory.n_atoms == trajectory.n_atoms
print(
    f"Loader conversion: {loaded_trajectory.n_frames} frames, {loaded_trajectory.n_atoms} atoms"
)

# PLOT TEMPERATURE EVOLUTION
print("\n" + "=" * 70)
print("PLOTTING TEMPERATURE EVOLUTION")
print("=" * 70)

import matplotlib.pyplot as plt

# Read equilibration log
eq_data = np.loadtxt(f"{OUTPUT_DIR}/equilibration.log", comments="#")
eq_steps = eq_data[:, 0]
eq_time = eq_data[:, 1]
eq_temps = eq_data[:, 2]
eq_epot = eq_data[:, 3]
eq_etot = eq_data[:, 5]

# Read production log
prod_data = np.loadtxt(f"{OUTPUT_DIR}/production.log", comments="#")
prod_steps = prod_data[:, 0]
prod_time = prod_data[:, 1]
prod_temps = prod_data[:, 2]
prod_epot = prod_data[:, 3]
prod_etot = prod_data[:, 5]

# Offset production time to continue from end of equilibration
# Normalize production time to start at 0, then add equilibration end time
if len(eq_time) > 0 and len(prod_time) > 0:
    prod_time_offset = prod_time - prod_time[0] + eq_time[-1]
else:
    prod_time_offset = prod_time

# Create 2-panel plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# TEMPERATURE PLOT
ax1.plot(eq_time, eq_temps, "b-", alpha=0.7, label="Equilibration")
ax1.plot(prod_time_offset, prod_temps, "r-", alpha=0.7, label="Production")
ax1.axhline(y=300, color="k", linestyle="--", label="Target (300 K)")
if len(eq_time) > 0 and len(prod_time) > 0:
    ax1.axvline(x=eq_time[-1], color="gray", linestyle=":", alpha=0.5, label="Eq->Prod")
ax1.set_xlabel("Time (ps)")
ax1.set_ylabel("Temperature (K)")
ax1.set_title("MD Temperature Evolution")
ax1.legend()
ax1.grid(alpha=0.3)

# ENERGY PLOT
ax2.plot(eq_time, eq_etot, "b-", alpha=0.7, label="Equilibration (Total)")
ax2.plot(prod_time_offset, prod_etot, "r-", alpha=0.7, label="Production (Total)")
ax2.plot(
    eq_time, eq_epot, "b--", alpha=0.5, linewidth=1, label="Equilibration (Potential)"
)
ax2.plot(
    prod_time_offset,
    prod_epot,
    "r--",
    alpha=0.5,
    linewidth=1,
    label="Production (Potential)",
)
if len(eq_time) > 0 and len(prod_time) > 0:
    ax2.axvline(x=eq_time[-1], color="gray", linestyle=":", alpha=0.5, label="Eq->Prod")
ax2.set_xlabel("Time (ps)")
ax2.set_ylabel("Energy (eV)")
ax2.set_title("MD Energy Evolution")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/md_evolution.png", dpi=150)
print(f"Saved MD evolution plot: {OUTPUT_DIR}/md_evolution.png")

# TACAW CALCULATION ON MD TRAJECTORY
print("\n" + "=" * 70)
print("TACAW CALCULATION")
print("=" * 70)

print(f"Using {trajectory.n_frames} frames for TACAW calculation")
print(f"Box matrix:\n{trajectory.box_matrix}")

# Setup multislice calculator for parallel beam (aperture=0)
calculator = MultisliceCalculator()
calculator.setup(
    trajectory,
    aperture=0,  # Parallel beam for TACAW
    voltage_eV=100e3,  # 100 keV
    sampling=0.1,  # Angstrom
    slice_thickness=0.5,  # Angstrom
)

print("Running multislice calculation...")
exitwaves = calculator.run()

exitwaves.crop([min(exitwaves.kxs)/2,max(exitwaves.kxs)/2],[min(exitwaves.kys)/2,max(exitwaves.kys)/2]) # CROP FOR LESS RAM USAGE?????

exitwaves.plot(nuke_zerobeam=True, powerscaling=0.125)
print("Computing TACAW spectrum...")
tacaw = TACAWData(exitwaves)

# Plot spectral diffraction at multiple frequencies
# Si diamond cubic (FCC lattice), a = 5.431 Å, viewed along [001]
# NOTE: fftfreq returns ordinary frequency (cycles/Å), not angular frequency (rad/Å)
# So BZ width = 2/a (not 2π/a) for FCC along [100]
# FCC reciprocal lattice is BCC, nearest G-vector along [100] is at 2/a
a_Si = 5.431
k_X = 1 / a_Si  # ≈ 0.184 Å⁻¹ (Γ to X in cycles/Å, half BZ width)
k_BZ = 2 * k_X  # ≈ 0.368 Å⁻¹ (full BZ width along [100] in cycles/Å)
n_BZ = 6
k_extent = n_BZ * k_X  # half-width for plotting ±k_extent (n_BZ zones total)
print(f"BZ width: {k_BZ:.3f} Å⁻¹, Γ→X: {k_X:.3f} Å⁻¹, showing {n_BZ} BZ (±{k_extent:.3f} Å⁻¹)")

# Get kx and ky arrays
kxs = tacaw.kxs
kys = tacaw.kys

# Find indices for the displayed k-extent region (crop to avoid interpolation blur)
kx_mask = (kxs >= -k_extent) & (kxs <= k_extent)
ky_mask = (kys >= -k_extent) & (kys <= k_extent)
kxs_crop = kxs[kx_mask]
kys_crop = kys[ky_mask]

# Plot multiple frequencies - Si phonons up to ~15 THz
frequencies_THz = [1.0, 2.0, 5.0, 6.0, 10.0, 15.0]
n_freq = len(frequencies_THz)
n_cols = 3
n_rows = (n_freq + n_cols - 1) // n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8))
axs = axs.flatten()

for i, freq in enumerate(frequencies_THz):
    Z = tacaw.spectral_diffraction(freq)

    # Nuke the zero beam (kx=0, ky=0) before cropping
    kx_zero_idx = np.argmin(np.abs(kxs))
    ky_zero_idx = np.argmin(np.abs(kys))
    Z[kx_zero_idx, ky_zero_idx] = 0

    # Crop to displayed region
    Z_crop = Z[kx_mask, :][:, ky_mask]

    axs[i].imshow(
        Z_crop.T**0.25,
        cmap="inferno",
        extent=(kxs_crop.min(), kxs_crop.max(), kys_crop.min(), kys_crop.max()),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    axs[i].set_title(f"{freq} THz")
    axs[i].set_xlabel("kx (Å⁻¹)")
    axs[i].set_ylabel("ky (Å⁻¹)")

# Hide empty subplots
for i in range(n_freq, len(axs)):
    axs[i].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tacaw_spectral_multi.png", dpi=150)
print(f"Saved TACAW spectral diffraction plots: {OUTPUT_DIR}/tacaw_spectral_multi.png")

# DISPERSION PLOT along kx at ky=0 using native resolution
print("\n" + "=" * 70)
print("DISPERSION PLOT")
print("=" * 70)

# Use native kx sampling - no resampling/interpolation
# Find ky index closest to 0
ky_zero_idx = np.argmin(np.abs(kys))
print(f"Using ky = {kys[ky_zero_idx]:.4f} Å⁻¹ (index {ky_zero_idx})")

# Get frequencies (positive only, limited to 20 THz for Si phonons)
max_freq = 20.0  # THz - Si optical phonons are ~15 THz
frequencies = tacaw.frequencies
freq_mask = (frequencies >= 0) & (frequencies <= max_freq)
frequencies_display = frequencies[freq_mask]

# Extract dispersion: intensity[probe, freq, kx, ky] -> sum over probes, select ky=0
# Use .data to get numpy array
intensity = tacaw.data  # shape: (n_probes, n_freq, n_kx, n_ky)
dispersion_full = np.mean(intensity[:, :, :, ky_zero_idx], axis=0)  # average over probes
dispersion_freq = dispersion_full[freq_mask, :]  # frequency-limited

# Crop kx to display range
kx_display_mask = (kxs >= -k_extent) & (kxs <= k_extent)
kxs_display = kxs[kx_display_mask]
dispersion_crop = dispersion_freq[:, kx_display_mask]

# Nuke zero beam (kx=0)
kx_zero_display_idx = np.argmin(np.abs(kxs_display))
dispersion_crop[:, kx_zero_display_idx] = 0

print(f"Dispersion shape: {dispersion_crop.shape} (freq × kx)")
print(f"Frequency range: {frequencies_display.min():.2f} to {frequencies_display.max():.2f} THz")
print(f"kx range: {kxs_display.min():.3f} to {kxs_display.max():.3f} Å⁻¹")

# Plot dispersion
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(
    dispersion_crop**0.25,  # power scaling for visibility
    cmap="inferno",
    extent=(kxs_display.min(), kxs_display.max(), frequencies_display.min(), frequencies_display.max()),
    origin="lower",
    aspect="auto",
    interpolation="nearest",
)
ax.set_xlabel("kx (Å⁻¹)")
ax.set_ylabel("Frequency (THz)")
ax.set_title(f"Phonon Dispersion along [100] (ky=0, {n_BZ} BZ)")

# Add BZ boundary markers at X points (±k_X, ±2k_X, ±3k_X, ...)
for i in range(-n_BZ, n_BZ + 1):
    if i != 0:
        ax.axvline(x=i * k_X, color="white", linestyle="--", alpha=0.3, linewidth=0.5)

plt.colorbar(im, ax=ax, label="Intensity$^{0.25}$")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tacaw_dispersion.png", dpi=150)
print(f"Saved dispersion plot: {OUTPUT_DIR}/tacaw_dispersion.png")

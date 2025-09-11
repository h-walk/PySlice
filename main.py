#!/usr/bin/env python3
"""TACAW Simulation Script - Load trajectory, run multislice, convert to frequency domain"""

import sys,os,numpy as np,matplotlib.pyplot as plt,time
from pathlib import Path

# Import TACAW modules
from src.io.loader import TrajectoryLoader
from src.multislice.calculators import MultisliceCalculator
from src.multislice.multislice import probe_grid
from src.postprocessing.tacaw_data import TACAWData
from src.postprocessing.haadf_data import HAADFData

def main():
    """Main TACAW simulation"""
    print("Starting TACAW simulation...")
    
    # Setup
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load trajectory
    dump = "examples/monolayer300k.lammpstrj"
    dt = 0.005  # 5 fs timestep in ps
    types = {1: "B", 2: "N"}  # Element mapping
    
    print("Loading trajectory...")
    trajectory = TrajectoryLoader(dump, timestep=dt, element_names=types).load()
    print(f"Loaded: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
    
    # Limit frames for testing
    if trajectory.n_frames > 400:
        trajectory = trajectory.slice_timesteps(list(range(400)))
        print(f"Limited to 400 frames")
    
    # Run simulation
    print("Running multislice simulation...")
    start_time = time.time()
    
    # Setup calculator
    calculator = MultisliceCalculator()
    probe_pos = [(trajectory.box_matrix[0,0]/2, trajectory.box_matrix[1,1]/2)]  # Center probe
    
    calculator.setup(
        trajectory=trajectory,
        probe_positions=probe_pos,
        aperture=0.0,
        voltage_eV=100e3,
        defocus=0.0,
        slice_thickness=0.5,
        sampling=0.1,
        cleanup_temp_files=False,
        save_path=output_dir
    )
    
    # Run simulation
    wf_data = calculator.run()
    sim_time = time.time() - start_time
    print(f"Simulation completed in {sim_time:.2f}s")
    
    # Convert to TACAW data
    print("Converting to frequency domain...")
    tacaw_data = TACAWData(wf_data)
    print(f"Conversion complete. Freq range: {tacaw_data.frequencies.min():.2f} to {tacaw_data.frequencies.max():.2f} THz")
    
    print("TACAWData conversion complete")
    
    # Create plots
    print("Creating plots...")
    
    # 1. Frequency spectrum
    spectrum = tacaw_data.spectrum(probe_index=0)
    pos_freq = tacaw_data.frequencies >= 0
    freq_pos = tacaw_data.frequencies[pos_freq]
    spec_pos = spectrum[pos_freq]
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(freq_pos, spec_pos, 'b-', linewidth=2)
    ax.fill_between(freq_pos, spec_pos, alpha=0.3)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Intensity')
    ax.set_title('TACAW Frequency Spectrum')
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrum.png', dpi=300)
    plt.close()
    
    # 2. Diffraction pattern
    diffraction = tacaw_data.diffraction(probe_index=0)
    diff_scaled = diffraction**0.25  # Fourth root for visualization
    
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(diff_scaled, extent=[tacaw_data.kxs.min(), tacaw_data.kxs.max(),
                                      tacaw_data.kys.min(), tacaw_data.kys.max()],
                   origin='lower', cmap='inferno', aspect='equal')
    plt.colorbar(im, label='Intensity^(1/4)')
    ax.set_xlabel('kx (Å⁻¹)')
    ax.set_ylabel('ky (Å⁻¹)')
    ax.set_title('TACAW Diffraction Pattern')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.tight_layout()
    plt.savefig(output_dir / 'diffraction.png', dpi=300)
    plt.close()
    
    # 3. Spectral diffraction 
    spec_diff = tacaw_data.spectral_diffraction(frequency=35, probe_index=0)
    spec_diff_scaled = spec_diff**0.25
    
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(spec_diff_scaled, extent=[tacaw_data.kxs.min(), tacaw_data.kxs.max(),
                                           tacaw_data.kys.min(), tacaw_data.kys.max()],
                   origin='lower', cmap='inferno', aspect='equal')
    plt.colorbar(im, label='Intensity^(1/4)')
    ax.set_xlabel('kx (Å⁻¹)')
    ax.set_ylabel('ky (Å⁻¹)')
    ax.set_title('Spectral Diffraction at 35 THz')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.tight_layout()
    plt.savefig(output_dir / 'spectral_diff.png', dpi=300)
    plt.close()
    
    # 4. Dispersion plot - frequency vs k-space from (0,0) to (10,0)

    
    # Create k-path from (0,0) to (10,0)
    n_samples = 1000
    kx_path = np.linspace(0, 10, n_samples)
    ky_path = np.zeros(n_samples)  # ky = 0 for the entire path
    
    # Sample intensity along the k-path for each frequency
    dispersion_data = []
    pos_freq = tacaw_data.frequencies > 0
    freq_indices = np.where(pos_freq)[0]
    
    for freq_idx in freq_indices:
        intensities = []
        for kx_val, ky_val in zip(kx_path, ky_path):
            # Find nearest k-point in the data
            kx_idx = np.argmin(np.abs(tacaw_data.kxs - kx_val))
            ky_idx = np.argmin(np.abs(tacaw_data.kys - ky_val))
            
            # Get intensity at this k-point and frequency
            intensity = tacaw_data.intensity[0, freq_idx, kx_idx, ky_idx]
            if hasattr(intensity, 'cpu'):  # Handle PyTorch tensors
                intensity = intensity.cpu().numpy()
            intensities.append(float(intensity))
        
        dispersion_data.append(intensities)
    
    dispersion_data = np.array(dispersion_data)
    freq_positive = tacaw_data.frequencies[pos_freq]
    
    # Scale for better visualization
    dispersion_data = dispersion_data**0.25
    
    # Create dispersion plot
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(dispersion_data, extent=[0, 10, freq_positive.min(), freq_positive.max()],
                   aspect='auto', cmap='inferno', origin='lower', interpolation='bilinear')
    
    plt.colorbar(im, label='Intensity^(1/4)')
    ax.set_xlabel('kx (Å⁻¹)')
    ax.set_ylabel('Frequency (THz)')
    ax.set_title('TACAW Dispersion: Frequency vs kx')
    plt.tight_layout()
    plt.savefig(output_dir / 'dispersion.png', dpi=300)
    plt.close()
    
    # 5. HAADF simulation
    print("Running HAADF simulation...")
    
    # First, create a smaller trajectory for HAADF (following unit test pattern)
    # Estimate lattice parameters from box size
    box = trajectory.box_matrix.diagonal()
    a, b = box[0]/100, box[1]/100  # Rough estimate for BN lattice (~2.5 Å)
    
    # Trim trajectory to smaller region (like unit test does)
    haadf_trajectory = trajectory.slice_positions([0, 10*a], [0, 10*b])
    print(f"HAADF trajectory trimmed to 10×10 unit cells: {haadf_trajectory.n_atoms} atoms")
    
    # Select 3 random frames (like unit test)
    slice_timesteps = np.arange(haadf_trajectory.n_frames)
    np.random.seed(5)  # For reproducibility
    np.random.shuffle(slice_timesteps)
    slice_timesteps = slice_timesteps[:3]
    haadf_trajectory = haadf_trajectory.slice_timesteps(slice_timesteps)
    print(f"HAADF using 3 random frames: {slice_timesteps}")
    
    # Create probe grid using lattice coordinates (like unit test)
    # For ~500 probes: use 20×25 = 500 probes over 3×3 unit cells
    haadf_positions = probe_grid([a, 4*a], [b, 4*b], 20, 25)
    print(f"HAADF grid: 20×25 = {len(haadf_positions)} probe positions over 3×3 unit cells")
    
    # Setup HAADF calculator with convergent beam
    haadf_calculator = MultisliceCalculator()
    haadf_calculator.setup(
        trajectory=haadf_trajectory,
        probe_positions=haadf_positions,
        aperture=30.0,  # 30 mrad convergent beam for HAADF
        voltage_eV=100e3,
        defocus=0.0,
        slice_thickness=0.5,
        sampling=0.1,
        cleanup_temp_files=False,
        save_path=output_dir
    )
    
    # Run HAADF simulation
    start_time = time.time()
    haadf_wf_data = haadf_calculator.run()
    haadf_time = time.time() - start_time
    print(f"HAADF simulation completed in {haadf_time:.2f}s")
    
    # Generate HAADF image
    print("Generating HAADF image...")
    haadf_data = HAADFData(haadf_wf_data)
    haadf_image = haadf_data.ADF(collection_angle=45, preview=False)  # 45 mrad collection angle
    
    # Plot HAADF image
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(haadf_image.T, cmap='inferno', origin='lower', 
                   extent=[a, 4*a, b, 4*b])
    plt.colorbar(im, label='HAADF Intensity')
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)') 
    ax.set_title('HAADF-STEM Image (3×3 unit cells)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'haadf.png', dpi=300)
    plt.close()
    
    # Save HAADF data
    np.save(output_dir / "haadf_image.npy", haadf_image)
    np.save(output_dir / "haadf_positions.npy", haadf_positions)
    
    # Save data
    np.save(output_dir / "frequencies.npy", tacaw_data.frequencies)
    np.save(output_dir / "spectrum.npy", spectrum)
    np.save(output_dir / "diffraction.npy", diffraction)
    
    # Summary
    print("="*60)
    print("TACAW + HAADF SIMULATION COMPLETE")
    print("="*60)
    print("TACAW Analysis:")
    print(f"  Frames: {trajectory.n_frames}, Atoms: {trajectory.n_atoms}")
    print(f"  Probe: center position")
    print(f"  k-space: {len(tacaw_data.kxs)}×{len(tacaw_data.kys)}")
    print(f"  Frequency range: {tacaw_data.frequencies.min():.2f} to {tacaw_data.frequencies.max():.2f} THz")
    print()
    print("HAADF-STEM Analysis:")
    print(f"  Frames: {haadf_trajectory.n_frames}, Atoms: {haadf_trajectory.n_atoms}")
    print(f"  Probe grid: 20×25 = {len(haadf_positions)} positions")
    print(f"  Scan area: 3×3 unit cells")
    print(f"  Convergence angle: 30 mrad")
    print(f"  Collection angle: 45 mrad")
    print(f"  HAADF image shape: {haadf_image.shape}")
    print()
    print(f"Results saved to: {output_dir}")
    print("Files generated: spectrum.png, diffraction.png, spectral_diff.png, dispersion.png, haadf.png")
    print("="*60)

if __name__ == "__main__":
    main()
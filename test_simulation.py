#!/usr/bin/env python3
"""
TACAW Simulation Test Script

This script loads a LAMMPS trajectory and runs the complete TACAW workflow:
1. Load trajectory from LAMMPS dump file
2. Run multislice simulation for each timestep
3. Convert time-domain data to frequency domain via FFT (JACR method)
4. Generate comprehensive plots of the results






Features:
- Grid-based probe positioning
- Smart caching system
- TACAW data analysis and visualization

Usage:
    python test_jacr_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our TACAW modules
try:
    from src.io.loader import TrajectoryLoader
    from src.tacaw.ms_calculator_ase import MultisliceCalculatorASE
    from src.tacaw.tacaw_data import TACAWData
except ImportError as e:
    logger.error(f"Failed to import TACAW modules: {e}")
    logger.error("Please ensure all dependencies are installed. Run: pip install -r requirements.txt")
    raise


def setup_simulation_parameters():
    """Setup default simulation parameters for the ASE-based test."""
    return {
        'aperture': 0.0,            # 0.0 mrad = plane wave (set >0 for convergent beam)
        'voltage_kv': 100.0,        # 200 kV accelerating voltage
        'pixel_size': 0.1,          # 0.1 Å/pixel
        'defocus': 0.0,             # No defocus
        'slice_thickness': 0.5,     # 1 Å slice thickness
        'sampling': 0.1,            # Real space sampling in Å
        'batch_size': 10,           # Process 10 frames at a time (adjust based on memory)
    }


def setup_probe_positions(trajectory, grid_dim="1x1", manual_positions=None):
    """
    Setup probe positions across the sample in a regular grid or using manual positions.

    Args:
        trajectory: Trajectory object with box information
        grid_dim: Grid dimension as string, e.g., "1x1", "2x2", "3x3", "4x2"
        manual_positions: Optional list of (x, y) tuples for manual probe positions

    Returns:
        List of (x, y) probe positions in Angstroms
    """
    # Get sample dimensions from trajectory
    box = trajectory.box_matrix.diagonal()

    # If manual positions are provided, use them instead of grid
    if manual_positions is not None:
        if not isinstance(manual_positions, list):
            logger.error("manual_positions must be a list of (x, y) tuples")
            return []

        # Validate manual positions
        validated_positions = []
        for i, pos in enumerate(manual_positions):
            try:
                x, y = pos
                # Convert to float and validate
                x, y = float(x), float(y)

                # Check if position is within box boundaries (with some tolerance)
                if not (0 <= x <= box[0] and 0 <= y <= box[1]):
                    logger.warning(f"Probe position {i} ({x:.2f}, {y:.2f}) is outside box boundaries "
                                 f"({box[0]:.2f} × {box[1]:.2f}), but will use it anyway")

                validated_positions.append((x, y))

            except (ValueError, TypeError) as e:
                logger.error(f"Invalid probe position {i}: {pos}. Expected (x, y) tuple with numeric values.")
                return []

        logger.info(f"Using {len(validated_positions)} manual probe positions: {validated_positions}")
        return validated_positions

    # Parse grid dimensions
    try:
        rows, cols = map(int, grid_dim.lower().split('x'))
        if rows < 1 or cols < 1:
            raise ValueError("Grid dimensions must be positive integers")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid grid dimension '{grid_dim}'. Using '1x1' as default.")
        rows, cols = 1, 1

    total_probes = rows * cols
    logger.info(f"Creating {rows}x{cols} probe grid ({total_probes} total probes)")

    # Calculate probe positions
    probe_positions = []

    # Add padding to avoid probe positions right at the boundaries
    padding_fraction = 0.1  # 10% padding from edges

    for i in range(rows):
        for j in range(cols):
            # Calculate fractional position (0 to 1) for each probe
            if rows == 1:
                # Single row - center vertically
                y_fraction = 0.5
            else:
                # Multiple rows - distribute with padding
                y_fraction = padding_fraction + (i / (rows - 1)) * (1 - 2 * padding_fraction)

            if cols == 1:
                # Single column - center horizontally
                x_fraction = 0.5
            else:
                # Multiple columns - distribute with padding
                x_fraction = padding_fraction + (j / (cols - 1)) * (1 - 2 * padding_fraction)

            # Convert to actual coordinates
            x_pos = box[0] * x_fraction
            y_pos = box[1] * y_fraction

            probe_positions.append((x_pos, y_pos))

    logger.info(f"Created {len(probe_positions)} probe positions in {rows}x{cols} grid")
    logger.info(f"Probe positions: {probe_positions}")

    return probe_positions







def plot_tacaw_results(tacaw_data, output_dir):
    """Create comprehensive plots of TACAW analysis results."""
    logger.info("Creating TACAW analysis plots...")
    
    # Create multiple figure sets for different analyses
    
    # Figure 1: Frequency spectra for different probe positions
    n_probes = len(tacaw_data.probe_positions)

    if n_probes == 1:
        # Single probe - use single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        spectrum = tacaw_data.spectrum(probe_index=0)

        # Filter to positive frequencies only
        positive_freq_mask = tacaw_data.frequency >= 0
        freq_pos = tacaw_data.frequency[positive_freq_mask]
        spectrum_pos = spectrum[positive_freq_mask]

        ax.plot(freq_pos, spectrum_pos, linewidth=2, color='darkblue')
        ax.set_xlabel('Frequency (THz)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title(f'Frequency Spectrum at Probe Position {tacaw_data.probe_positions[0]}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
    else:
        # Multiple probes - use subplot grid, handle more than 4 probes
        max_plots = min(9, n_probes)  # Show max 9 probes (3x3 grid)
        cols = min(3, max_plots)
        rows = (max_plots + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if max_plots == 1:
            axes = [axes]  # Make it iterable
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]  # Single row
        else:
            axes = axes.flatten()  # Multi-dimensional array

        colors_list = ['darkblue', 'darkgreen', 'darkred', 'darkorange', 'purple', 'brown', 'pink', 'gray', 'olive']

        for i in range(max_plots):
            ax = axes[i]
            spectrum = tacaw_data.spectrum(probe_index=i)

            # Filter to positive frequencies only
            positive_freq_mask = tacaw_data.frequency >= 0
            freq_pos = tacaw_data.frequency[positive_freq_mask]
            spectrum_pos = spectrum[positive_freq_mask]

            color = colors_list[i % len(colors_list)]
            ax.plot(freq_pos, spectrum_pos, linewidth=2, color=color)
            ax.set_xlabel('Frequency (THz)', fontsize=10)
            ax.set_ylabel('Intensity', fontsize=10)
            probe_pos = tacaw_data.probe_positions[i]
            ax.set_title(f'Probe {i} @ ({probe_pos[0]:.1f}, {probe_pos[1]:.1f})',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')

        # Hide unused subplots if any
        for i in range(max_plots, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_spectra.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Diffraction patterns
    if n_probes == 1:
        # Single probe - use single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        diffraction = tacaw_data.diffraction(probe_index=0)

        # Apply logarithmic scaling to compress dynamic range
        diffraction_log = np.log10(diffraction + 1e-10)  # Add small value to avoid log(0)

        im = ax.imshow(diffraction_log, extent=[tacaw_data.kx.min(), tacaw_data.kx.max(),
                                              tacaw_data.ky.min(), tacaw_data.ky.max()],
                       origin='lower', cmap='inferno')
        plt.colorbar(im, ax=ax, label='log₁₀(Intensity)')
        ax.set_xlabel('kx (Å⁻¹)', fontsize=12)
        ax.set_ylabel('ky (Å⁻¹)', fontsize=12)
        ax.set_title('Diffraction Pattern (Log Scale)', fontsize=14, fontweight='bold')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.grid(True, alpha=0.3)
    else:
        # Multiple probes - use subplot grid, handle more than 4 probes
        max_plots = min(9, n_probes)  # Show max 9 probes (3x3 grid)
        cols = min(3, max_plots)
        rows = (max_plots + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
        if max_plots == 1:
            axes = [axes]  # Make it iterable
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]  # Single row
        else:
            axes = axes.flatten()  # Multi-dimensional array

        for i in range(max_plots):
            ax = axes[i]
            diffraction = tacaw_data.diffraction(probe_index=i)

            # Apply logarithmic scaling to compress dynamic range
            diffraction_log = np.log10(diffraction + 1e-10)  # Add small value to avoid log(0)

            im = ax.imshow(diffraction_log, extent=[tacaw_data.kx.min(), tacaw_data.kx.max(),
                                                  tacaw_data.ky.min(), tacaw_data.ky.max()],
                           origin='lower', cmap='inferno')
            plt.colorbar(im, ax=ax, label='log₁₀(Intensity)', shrink=0.8)
            ax.set_xlabel('kx (Å⁻¹)', fontsize=10)
            ax.set_ylabel('ky (Å⁻¹)', fontsize=10)
            probe_pos = tacaw_data.probe_positions[i]
            ax.set_title(f'Probe {i} @ ({probe_pos[0]:.1f}, {probe_pos[1]:.1f})',
                        fontsize=11, fontweight='bold')
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.grid(True, alpha=0.3)

        # Hide unused subplots if any
        for i in range(max_plots, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diffraction_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Spectral diffraction at specific frequencies
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Select some interesting positive frequencies only
    positive_freqs = tacaw_data.frequency[tacaw_data.frequency > 0]
    max_freq = positive_freqs.max()
    test_frequencies = [max_freq*0.1, max_freq*0.2, max_freq*0.3, 
                       max_freq*0.4, max_freq*0.5, max_freq*0.6]
    
    for i, freq in enumerate(test_frequencies):
        ax = axes[i//3, i%3]
        spectral_diff = tacaw_data.spectral_diffraction(frequency=freq, probe_index=0)
        
        # Apply logarithmic scaling to compress dynamic range
        spectral_diff_log = np.log10(spectral_diff + 1e-10)  # Add small value to avoid log(0)
        
        im = ax.imshow(spectral_diff_log, extent=[tacaw_data.kx.min(), tacaw_data.kx.max(),
                                                tacaw_data.ky.min(), tacaw_data.ky.max()],
                       origin='lower', cmap='inferno')
        plt.colorbar(im, ax=ax, label='log₁₀(Intensity)')
        ax.set_xlabel('kx (Å⁻¹)', fontsize=11)
        ax.set_ylabel('ky (Å⁻¹)', fontsize=11)
        ax.set_title(f'Spectral Diffraction at {freq:.1f} THz (Log Scale)', fontsize=12, fontweight='bold')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectral_diffraction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
   


def main():
    """Main test function."""
    logger.info("Starting JACR TACAW simulation test...")
    
    # Setup output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load trajectory
    logger.info("Loading LAMMPS trajectory...")
    trajectory_file = "examples/monolayer300k.lammpstrj"
    timestep = 0.005  # 5 fs timestep in ps
    
    loader = TrajectoryLoader(trajectory_file, timestep=timestep)
    trajectory = loader.load()
    
    logger.info(f"Loaded trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
    
    # Limit frames for testing (remove this for full simulation)
    if trajectory.n_frames > 400:
        logger.info(f"Limiting to first 400 frames for testing (original: {trajectory.n_frames})")
        trajectory = trajectory.slice_timesteps(list(range(400)))
    
 
    # Setup simulation parameters
    sim_params = setup_simulation_parameters()

    # Grid examples: "1x1" (single center), "2x2" (4 corners), "3x3" (9 grid), "4x2" (8 rectangular)
    # Manual example: manual_positions=[(10.0, 15.0), (25.0, 30.0)]
    # Using 1x1 for testing - single probe at center for faster execution
    # For production: change to "2x2", "3x3", etc. for multiple probe positions
    probe_positions = setup_probe_positions(trajectory, grid_dim="1x1")
    
    # Initialize ASE-based calculator
    logger.info("Initializing ASE-based MultisliceCalculator...")
    calculator = MultisliceCalculatorASE()
    
    # Run JACR simulation
    logger.info("Running TACAW multislice simulation...")
    logger.info(f"Simulation parameters: {sim_params}")
    logger.info(f"Number of probe positions: {len(probe_positions)} (using single probe for testing)")
    logger.info(f"Trajectory frames: {trajectory.n_frames}, atoms: {trajectory.n_atoms}")
    
    start_time = time.time()
    
    try:
        wf_data = calculator.run_simulation(
            trajectory=trajectory,
            probe_positions=probe_positions,
            **sim_params
        )
        
        simulation_time = time.time() - start_time
        logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
        
        
        # Convert to TACAW data via FFT (JACR method) with caching
        # This avoids expensive re-computation of the FFT if the same simulation has been run before
        tacaw_cache_file = output_dir / "tacaw_data.pkl"

        # Check if cached TACAWData exists
        if tacaw_cache_file.exists():
            logger.info(f"Loading cached TACAWData from: {tacaw_cache_file.name}")
            tacaw_data = TACAWData.load(tacaw_cache_file)

            if tacaw_data is not None:
                logger.info(f"TACAWData loaded successfully. Frequency range: {tacaw_data.frequency.min():.2f} to {tacaw_data.frequency.max():.2f} THz")
            else:
                logger.warning("Failed to load cached TACAWData, performing fresh conversion...")
                tacaw_data = None
        else:
            tacaw_data = None

        # Perform conversion if no cached data or loading failed
        if tacaw_data is None:
            logger.info("Converting to frequency domain (JACR method)...")
            conversion_start = time.time()
            tacaw_data = wf_data.fft_to_tacaw_data()
            conversion_time = time.time() - conversion_start

            logger.info(f"TACAW conversion complete in {conversion_time:.2f} seconds. Frequency range: {tacaw_data.frequency.min():.2f} to {tacaw_data.frequency.max():.2f} THz")

            # Save the converted data for future use
            logger.info("Saving TACAWData for future use...")
            success = tacaw_data.save(output_dir, auto_generate_filename=True,
                                    trajectory=trajectory, sim_params=sim_params,
                                    probe_positions=probe_positions)
            if success:
                # Update cache file path for summary
                tacaw_cache_file = output_dir / f"tacaw_data_{trajectory.n_frames}f_{len(probe_positions)}p_*.pkl"
        
        # Create TACAW analysis plots
        plot_tacaw_results(tacaw_data, output_dir)

        # Get sample data for saving
        sample_spectrum = tacaw_data.spectrum(probe_index=0)
        sample_diffraction = tacaw_data.diffraction(probe_index=0)

        # Save data for further analysis
        logger.info("Saving simulation data...")
        np.save(output_dir / "tacaw_frequencies.npy", tacaw_data.frequency)
        np.save(output_dir / "tacaw_kx.npy", tacaw_data.kx)
        np.save(output_dir / "tacaw_ky.npy", tacaw_data.ky)
        np.save(output_dir / "probe_positions.npy", np.array(tacaw_data.probe_positions))
        np.save(output_dir / "sample_spectrum.npy", sample_spectrum)
        np.save(output_dir / "diffraction_pattern.npy", sample_diffraction)

        # Print comprehensive summary statistics
        logger.info("\n" + "="*60)
        logger.info("TACAW SIMULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
        logger.info(f"Probe setup: {len(probe_positions)} probe using 1x1 grid (center position for testing)")
        logger.info(f"Probe positions: {tacaw_data.probe_positions}")
        logger.info(f"k-space sampling: {len(tacaw_data.kx)} × {len(tacaw_data.ky)}")
        logger.info(f"Frequency range: {tacaw_data.frequency.min():.2f} to {tacaw_data.frequency.max():.2f} THz")
        logger.info(f"Intensity data shape: {tacaw_data.intensity.shape}")
        logger.info(f"Simulation time: {simulation_time:.2f} seconds")
        logger.info(f"Results saved to: {output_dir.absolute()}")
        logger.info(f"TACAWData cache: {tacaw_cache_file.name}")
        logger.info("="*60)

        logger.info("JACR TACAW simulation test completed successfully!")
        logger.info(f"Note: To clear cached data, delete: {tacaw_cache_file}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()

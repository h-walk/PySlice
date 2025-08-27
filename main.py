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
    from src.tacaw.ms_calculator_abtem import MultisliceCalculatorAbtem 
    from src.tacaw.tacaw_data import TACAWData
except ImportError as e:
    logger.error(f"Failed to import TACAW modules: {e}")
    logger.error("Please ensure all dependencies are installed. Run: pip install -r requirements.txt")
    raise


def setup_simulation_parameters():
    """Setup default simulation parameters for the Abtem-based test."""
    return {
        'aperture': 0.0,            # 0.0 mrad = plane wave (set >0 for convergent beam)
        'voltage_kv': 100.0,        # 100 kV accelerating voltage
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







def create_dispersion_plot(k_points=None, n_samples=200):
    """
    Create a dispersion plot with custom k-points and sampling.

    Args:
        k_points: List of (kx, ky) tuples defining the k-space path
        n_samples: Number of sampling points between each k-point

    Returns:
        kx_path, ky_path: Arrays of interpolated k-points along the path
    """
    if k_points is None:
        # Default high-symmetry path for 2D system
        k_points = [(0, 0), (2, 0), (2, 2), (0, 2), (-2, 2), (-2, 0), (-2, -2), (0, -2), (2, -2), (2, 0)]

    # Create k-space path by interpolating between k-points
    kx_path = []
    ky_path = []

    for i in range(len(k_points)):
        start_point = k_points[i]
        end_point = k_points[(i + 1) % len(k_points)]

        # Interpolate between points
        for j in range(n_samples):
            t = j / (n_samples - 1)
            kx_interp = start_point[0] + t * (end_point[0] - start_point[0])
            ky_interp = start_point[1] + t * (end_point[1] - start_point[1])
            kx_path.append(kx_interp)
            ky_path.append(ky_interp)

    return np.array(kx_path), np.array(ky_path)


def plot_tacaw_results(tacaw_data, output_dir):
    """Create comprehensive plots of TACAW analysis results with enhanced styling."""
    logger.info("Creating TACAW analysis plots...")

    # Set up professional plot style
    plt.style.use('default')  # Reset to default
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.facecolor': '#fafafa',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
    })

    # Enhanced color palette
    spectrum_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B8EA5',
                      '#7D5BA6', '#0B9A6D', '#F6C85F', '#D34F73']

    # Create multiple figure sets for different analyses

    # Figure 1: Frequency spectra for different probe positions
    n_probes = len(tacaw_data.probe_positions)

    if n_probes == 1:
        # Single probe - use single plot (square aspect)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        spectrum = tacaw_data.spectrum(probe_index=0)
        positive_freq_mask = tacaw_data.frequency >= 0
        freq_pos = tacaw_data.frequency[positive_freq_mask]
        spectrum_pos = spectrum[positive_freq_mask]

        # Enhanced line plot with gradient fill
        ax.plot(freq_pos, spectrum_pos, linewidth=3, color='#2E86AB', alpha=0.9)
        ax.fill_between(freq_pos, spectrum_pos, alpha=0.3, color='#2E86AB')

        ax.set_xlabel('Frequency (THz)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')

        probe_pos = tacaw_data.probe_positions[0]
        ax.set_title(f'TACAW Frequency Spectrum\nProbe Position: ({probe_pos[0]:.1f}, {probe_pos[1]:.1f}) Å',
                    fontsize=16, fontweight='bold', pad=20)

        # Enhanced grid and styling
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_facecolor('#fafafa')

        # Add subtle spine styling
        for spine in ax.spines.values():
            spine.set_edgecolor('#666666')
            spine.set_linewidth(1.5)

    else:
        # Multiple probes - use subplot grid
        max_plots = min(9, n_probes)
        cols = min(3, max_plots)
        rows = (max_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if max_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()

        for i in range(max_plots):
            ax = axes[i]
            spectrum = tacaw_data.spectrum(probe_index=i)
            positive_freq_mask = tacaw_data.frequency >= 0
            freq_pos = tacaw_data.frequency[positive_freq_mask]
            spectrum_pos = spectrum[positive_freq_mask]

            color = spectrum_colors[i % len(spectrum_colors)]

            # Enhanced line plot
            ax.plot(freq_pos, spectrum_pos, linewidth=2.5, color=color, alpha=0.9)
            ax.fill_between(freq_pos, spectrum_pos, alpha=0.2, color=color)

            ax.set_xlabel('Frequency (THz)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')

            probe_pos = tacaw_data.probe_positions[i]
            ax.set_title(f'Probe {i+1}\n({probe_pos[0]:.1f}, {probe_pos[1]:.1f}) Å',
                        fontsize=12, fontweight='bold', pad=10)

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')

        # Hide unused subplots
        for i in range(max_plots, len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_spectra.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Figure 2: Diffraction patterns with enhanced colormaps
    if n_probes == 1:
        # Single probe - use single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        diffraction = tacaw_data.diffraction(probe_index=0)
        diffraction_log = np.log10(diffraction + 1e-10)

        # Enhanced colormap with inferno (user preference)
        im = ax.imshow(diffraction_log, extent=[tacaw_data.kx.min(), tacaw_data.kx.max(),
                                              tacaw_data.ky.min(), tacaw_data.ky.max()],
                       origin='lower', cmap='inferno', aspect='equal')

        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('log10(Intensity)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

        ax.set_xlabel('kx (Å⁻¹)', fontsize=14, fontweight='bold')
        ax.set_ylabel('ky (Å⁻¹)', fontsize=14, fontweight='bold')

        probe_pos = tacaw_data.probe_positions[0]
        ax.set_title(f'TACAW Diffraction Pattern\nProbe Position: ({probe_pos[0]:.1f}, {probe_pos[1]:.1f}) Å',
                    fontsize=16, fontweight='bold', pad=20)

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # Add center crosshair
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.5, linewidth=1)

    else:
        # Multiple probes - use subplot grid
        max_plots = min(9, n_probes)
        cols = min(3, max_plots)
        rows = (max_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.5*rows))
        if max_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()

        for i in range(max_plots):
            ax = axes[i]
            diffraction = tacaw_data.diffraction(probe_index=i)
            diffraction_log = np.log10(diffraction + 1e-10)

            im = ax.imshow(diffraction_log, extent=[tacaw_data.kx.min(), tacaw_data.kx.max(),
                                                  tacaw_data.ky.min(), tacaw_data.ky.max()],
                           origin='lower', cmap='inferno', aspect='equal')

            cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=20)
            cbar.set_label('log10(I)', fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)

            ax.set_xlabel('kx (Å⁻¹)', fontsize=11, fontweight='bold')
            ax.set_ylabel('ky (Å⁻¹)', fontsize=11, fontweight='bold')

            probe_pos = tacaw_data.probe_positions[i]
            ax.set_title(f'Probe {i+1}\n({probe_pos[0]:.1f}, {probe_pos[1]:.1f}) Å',
                        fontsize=12, fontweight='bold', pad=10)

            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3, linestyle='--')

            # Add center crosshair
            ax.axhline(y=0, color='white', linestyle='-', alpha=0.4, linewidth=0.8)
            ax.axvline(x=0, color='white', linestyle='-', alpha=0.4, linewidth=0.8)

        # Hide unused subplots
        for i in range(max_plots, len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'diffraction_pattern.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Figure 3: Spectral diffraction at specific frequencies
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Add main title
    fig.suptitle('TACAW Spectral Diffraction Analysis', fontsize=18, fontweight='bold', y=0.98)

    # Select some interesting positive frequencies only
    positive_freqs = tacaw_data.frequency[tacaw_data.frequency > 0]
    max_freq = positive_freqs.max()
    test_frequencies = [max_freq*0.1, max_freq*0.2, max_freq*0.3,
                       max_freq*0.4, max_freq*0.5, max_freq*0.6]

    for i, freq in enumerate(test_frequencies):
        ax = axes[i//3, i%3]
        spectral_diff = tacaw_data.spectral_diffraction(frequency=freq, probe_index=0)
        spectral_diff_log = np.log10(spectral_diff + 1e-10)

        # Enhanced colormap for spectral diffraction (using inferno)
        im = ax.imshow(spectral_diff_log, extent=[tacaw_data.kx.min(), tacaw_data.kx.max(),
                                                tacaw_data.ky.min(), tacaw_data.ky.max()],
                       origin='lower', cmap='inferno', aspect='equal')

        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25)
        cbar.set_label('log10(I)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel('kx (Å⁻¹)', fontsize=12, fontweight='bold')
        ax.set_ylabel('ky (Å⁻¹)', fontsize=12, fontweight='bold')

        ax.set_title(f'{freq:.2f} THz', fontsize=14, fontweight='bold', pad=15)

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # Add center crosshair
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.6, linewidth=1)
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.6, linewidth=1)

        plt.tight_layout()
    plt.savefig(output_dir / 'spectral_diffraction.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    # Figure 4: Dispersion plot - frequency vs k-space heatmap
    logger.info("Creating dispersion plot...")

    # Create simple dispersion path from (0,0) to (5,0)
    n_samples = 200 # More samples for smoother curve
    kx_path, ky_path = create_dispersion_plot(
        k_points=[(0, 0), (5, 0)],  # Simple path along kx
        n_samples=n_samples
    )

    # Sample intensity along the k-path for each frequency
    dispersion_data = []

    for freq_idx, freq in enumerate(tacaw_data.frequency):
        if freq > 0:  # Only positive frequencies
            intensities = []

            for kx_val, ky_val in zip(kx_path, ky_path):
                # Find nearest k-point in the data
                kx_idx = np.argmin(np.abs(tacaw_data.kx - kx_val))
                ky_idx = np.argmin(np.abs(tacaw_data.ky - ky_val))

                # Get intensity at this k-point and frequency
                intensity = tacaw_data.intensity[0, freq_idx, kx_idx, ky_idx]  # Use first probe
                intensities.append(intensity)

            dispersion_data.append(intensities)

    dispersion_data = np.array(dispersion_data)
    freq_positive = tacaw_data.frequency[tacaw_data.frequency > 0]

    # Apply logarithmic scaling to the intensity data
    dispersion_data = np.log10(dispersion_data + 1e-10)  # Add small value to avoid log(0)

    # Create dispersion plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot heatmap using imshow (correct orientation: kx on x, frequency on y)
    # Note: Using imshow instead of pcolormesh to avoid dimension mismatch issues
    # extent defines the coordinate system: [left, right, bottom, top]
    im = ax.imshow(dispersion_data, extent=[0, 5, freq_positive.min(), freq_positive.max()],
                   aspect='auto', cmap='inferno', origin='lower', interpolation='bilinear')

    # Add colorbar for log intensity
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('log₁₀(Intensity)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Set labels and title with correct axis orientation
    ax.set_xlabel('k$_x$ (Å⁻¹)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (THz)', fontsize=14, fontweight='bold')
    ax.set_title('TACAW Dispersion: Frequency vs k$_x$', fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Mark k-point boundaries (vertical lines for kx coordinates)
    ax.axvline(x=0, color='white', linestyle='-', alpha=0.8, linewidth=1)
    ax.axvline(x=5, color='white', linestyle='-', alpha=0.8, linewidth=1)

    # Add numerical labels for kx coordinates (no symbols)
    k_labels = ['0', '5']
    k_positions = [0, 5]  # Actual kx coordinates

    ax.set_xticks(k_positions[:len(k_labels)])
    ax.set_xticklabels(k_labels[:len(k_labels)], fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'dispersion_plot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
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


    atomic_numbers = {1: 5, 2: 7}  # Boron=5, Nitrogen=7
    loader = TrajectoryLoader(trajectory_file, timestep=timestep, atomic_numbers=atomic_numbers)

    trajectory = loader.load()

    logger.info(f"Loaded trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
    logger.info(f"Atom types range: {trajectory.atom_types.min()} to {trajectory.atom_types.max()}")
    logger.info(f"Unique atom types: {np.unique(trajectory.atom_types)}")
    
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
    
    # Initialize Abtem-based calculator
    logger.info("Initializing Abtem-based MultisliceCalculator...")
    calculator = MultisliceCalculatorAbtem()
    
    # Run JACR simulation
    logger.info("Running TACAW multislice simulation...")
    logger.info(f"Simulation parameters: {sim_params}")
    logger.info(f"Number of probe positions: {len(probe_positions)} (using single probe for testing)")
    logger.info(f"Trajectory frames: {trajectory.n_frames}, atoms: {trajectory.n_atoms}")
    
    start_time = time.time()

    # Check for cached TACAWData first to avoid expensive wf_data generation
    tacaw_data = None
    tacaw_cache_file = output_dir / "TACAWdata.pkl"

    # Check if cached TACAWData exists with simple filename
    if tacaw_cache_file.exists():
        logger.info(f"Loading cached TACAWData from: {tacaw_cache_file.name}")
        tacaw_data = TACAWData.load(tacaw_cache_file)

        if tacaw_data is not None:
            logger.info(f"TACAWData loaded successfully. Frequency range: {tacaw_data.frequency.min():.2f} to {tacaw_data.frequency.max():.2f} THz")
            logger.info("Skipping wf_data simulation since TACAWData cache exists")
        else:
            logger.warning(f"Failed to load cached TACAWData from {tacaw_cache_file.name}, will perform fresh simulation and conversion...")
            tacaw_data = None
    else:
        logger.info("No TACAWData cache found, will perform simulation and conversion...")

    try:
        # Only run expensive simulation if no cached TACAWData exists
        if tacaw_data is None:
            logger.info("Running TACAW multislice simulation to generate wf_data...")
            simulation_start = time.time()

            wf_data = calculator.run_simulation(
                trajectory=trajectory,
                probe_positions=probe_positions,
                **sim_params
            )

            simulation_time = time.time() - simulation_start
            logger.info(f"WF data simulation completed in {simulation_time:.2f} seconds")

            # Convert wf_data to TACAW data via FFT (JACR method)
            logger.info("Converting wf_data to frequency domain (JACR method)...")
            conversion_start = time.time()
            tacaw_data = wf_data.fft_to_tacaw_data()
            conversion_time = time.time() - conversion_start

            logger.info(f"TACAW conversion complete in {conversion_time:.2f} seconds. Frequency range: {tacaw_data.frequency.min():.2f} to {tacaw_data.frequency.max():.2f} THz")

            # Save the converted data for future use
            logger.info("Saving TACAWData for future use...")
            success = tacaw_data.save(tacaw_cache_file, auto_generate_filename=False)
            if success:
                logger.info(f"TACAWData saved to: {tacaw_cache_file.name}")
        else:
            # TACAWData was loaded from cache
            simulation_time = 0.0  # No simulation was performed
        
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

        if simulation_time > 0:
            logger.info(f"Simulation time: {simulation_time:.2f} seconds")
            logger.info("Data source: Fresh simulation")
        else:
            logger.info("Simulation time: 0.0 seconds (loaded from cache)")
            logger.info("Data source: Cached TACAWData")

        logger.info(f"Results saved to: {output_dir.absolute()}")
        logger.info(f"TACAWData cache: {tacaw_cache_file.name}")
        logger.info("="*60)

        logger.info("JACR TACAW simulation test completed successfully!")
       
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()

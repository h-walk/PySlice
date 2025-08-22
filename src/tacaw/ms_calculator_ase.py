"""
ASE-based Multislice Calculator using abtem for JACR method.

This implementation follows the patterns from abeels.py and uses ASE + abtem
instead of ptyrodactyl.electrons to avoid type checking issues.
"""
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

# ASE and abtem imports
import ase
import abtem
from abtem import CustomScan

# Import our custom data structures
from .trajectory import Trajectory
from .tacaw_data import TACAWData
from .wf_data import WFData

logger = logging.getLogger(__name__)

# Configure abtem for precision
abtem.config.set({"precision": "float64"})


class MultisliceCalculatorASE:
    """
    ASE-based multislice calculator for electron microscopy simulations.

    This class uses ASE (Atomic Simulation Environment) and abtem to perform 
    multislice simulations on molecular dynamics trajectories, generating wave 
    function data for each frame using the JACR method.
    """

    def __init__(self):
        """Initialize the ASE-based multislice calculator."""
        self.element_map = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
            30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr'
        }

    def run_simulation(
        self,
        trajectory: Trajectory,
        aperture: float,
        voltage_kv: float,
        pixel_size: float,
        defocus: float = 0.0,
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        element_symbols: Optional[Dict[int, str]] = None,
        slice_thickness: float = 1.0,
        sampling: float = 0.05,
        batch_size: int = 10,
        cleanup_temp_files: bool = False,
    ) -> WFData:
        """
        Run multislice simulation for each frame in the trajectory using ASE + abtem.
        
        This method saves each frame's results to disk to minimize memory usage.

        Args:
            trajectory: Trajectory object containing atomic positions for each frame
            aperture: Probe aperture size in milliradians (0.0 for plane wave)
            voltage_kv: Accelerating voltage in kilovolts
            pixel_size: Pixel size in Angstroms
            defocus: Probe defocus in Angstroms (default: 0.0)
            probe_positions: List of (x, y) probe positions in Angstroms.
                          If None, defaults to [(0, 0)]
            element_symbols: Dictionary mapping atomic numbers to element symbols.
                           If None, uses default mapping
            slice_thickness: Thickness of each slice in Angstroms (default: 1.0)
            sampling: Real space sampling in Angstroms (default: 0.05)
            batch_size: Number of frames to process together (default: 10, set to 1 for one-by-one)
            cleanup_temp_files: Whether to delete psi files after loading (default: False - keep for debugging/restarting)

        Returns:
            WFData object containing simulation results for all frames
        """
        # Set default probe positions if not provided
        if probe_positions is None:
            probe_positions = [(0.0, 0.0)]

        # Convert voltage to energy (voltage_kv * 1000 = energy in eV)
        energy = voltage_kv * 1000.0

        # Validate inputs
        self._validate_inputs(trajectory, aperture, voltage_kv, pixel_size, probe_positions)

        # Convert trajectory to ASE atoms for each frame
        logger.info(f"Converting {trajectory.n_frames} trajectory frames to ASE atoms...")
        atoms_list = self._convert_trajectory_to_ase_atoms(trajectory, element_symbols)

        # Create output directory for psi files (organized by simulation parameters)
        sim_id = f"psi_frames_{len(atoms_list)}f_{len(probe_positions)}p_{aperture}mrad"
        output_dir = Path("psi_data") / sim_id
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Psi files will be saved to: {output_dir}")

        # Process frames in batches to balance memory and efficiency
        logger.info(f"Processing {len(atoms_list)} frames in batches of {batch_size}...")
        
        # Initialize probe/plane wave once
        if aperture > 0:
            # Convergent probe
            logger.info("Creating convergent probe...")
            probe_template = abtem.Probe(
                energy=energy,
                semiangle_cutoff=aperture,
                defocus=defocus
            )
        else:
            # Plane wave
            logger.info("Creating plane wave...")
            plane_wave_template = abtem.PlaneWave(energy=energy, sampling=sampling)

        # Process frames in batches
        n_frames = len(atoms_list)
        n_batches = (n_frames + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_frames)
            
            # Check if all frames in this batch are already computed
            all_exist = True
            for frame_idx in range(start_idx, end_idx):
                psi_file = output_dir / f"psi_t{frame_idx}.npy"
                if not psi_file.exists():
                    all_exist = False
                    break
            
            if all_exist:
                continue  # Skip this entire batch
            
            # Create batch of atoms
            batch_atoms = atoms_list[start_idx:end_idx]
            batch_ensemble = abtem.AtomsEnsemble(batch_atoms)
            
            # Create potential for this batch
            potential = abtem.Potential(
                batch_ensemble,
                sampling=sampling,
                slice_thickness=slice_thickness
            )
            
            # Run simulation for this batch
            if aperture > 0:
                # Convergent probe
                probe = probe_template.copy()
                probe.grid.match(potential)
                custom_scan = CustomScan(probe_positions)
                exit_waves = probe.multislice(potential, scan=custom_scan, max_batch=1).compute()
            else:
                # Plane wave
                plane_wave = plane_wave_template.copy()
                plane_wave.grid.match(potential)
                exit_waves = plane_wave.multislice(potential).compute()
            
            # Convert to diffraction patterns
            diffraction_patterns = self._convert_to_diffraction_patterns(exit_waves)
            
            # Extract array data
            if hasattr(diffraction_patterns, 'array'):
                pattern_array = diffraction_patterns.array
            else:
                pattern_array = np.array(diffraction_patterns)
            
            # Save each frame's pattern individually (preserving order)
            for i, frame_idx in enumerate(range(start_idx, end_idx)):
                psi_file = output_dir / f"psi_t{frame_idx}.npy"
                
                # Skip if already exists (from partial batch completion)
                if psi_file.exists():
                    continue
                
                # Extract this frame's data from the batch result
                if len(pattern_array.shape) == 4:
                    # Shape: (batch_frames, n_probes, kx, ky)
                    frame_pattern = pattern_array[i]
                elif len(pattern_array.shape) == 3:
                    # Shape: (batch_frames, kx, ky) for plane wave
                    frame_pattern = pattern_array[i]
                else:
                    raise ValueError(f"Unexpected pattern shape: {pattern_array.shape}")
                
                # Save this frame's pattern
                np.save(psi_file, frame_pattern)
            
            # Clear memory after each batch
            del exit_waves, diffraction_patterns, potential, batch_ensemble
            
        # Now load all saved files and convert to WFData
        logger.info("Loading saved psi files and converting to WFData...")
        wf_data = self._load_and_convert_to_wf_data(
            output_dir,
            len(atoms_list),
            probe_positions,
            trajectory.timestep,
            sampling,
            cleanup_temp_files
        )

        return wf_data

    def _validate_inputs(
        self,
        trajectory: Trajectory,
        aperture: float,
        voltage_kv: float,
        pixel_size: float,
        probe_positions: List[Tuple[float, float]]
    ):
        """Validate input parameters."""
        if trajectory.n_frames == 0:
            raise ValueError("Trajectory must contain at least one frame")

        if aperture < 0:
            raise ValueError("Aperture must be non-negative (0 for plane wave)")

        if voltage_kv <= 0:
            raise ValueError("Voltage must be positive")

        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")

        if not probe_positions:
            raise ValueError("At least one probe position must be provided")

    def _convert_trajectory_to_ase_atoms(
        self, 
        trajectory: Trajectory, 
        element_symbols: Optional[Dict[int, str]] = None
    ) -> List[ase.Atoms]:
        """Convert trajectory frames to ASE Atoms objects."""
        
        # Use provided element symbols or default mapping
        elem_map = element_symbols if element_symbols else self.element_map
        
        # Create element string for ASE
        element_list = []
        for atom_type in trajectory.atom_types:
            if atom_type in elem_map:
                element_list.append(elem_map[atom_type])
            else:
                logger.warning(f"Unknown atom type {atom_type}, using 'X'")
                element_list.append('X')

        # Create orthogonal cell from the original tilted cell
        original_cell = trajectory.box_matrix
        logger.info(f"Original tilted cell:\n{original_cell}")
        
        # Extract orthogonal dimensions by computing cell lengths
        # This works for any cell orientation
        cell_lengths = np.array([
            np.linalg.norm(original_cell[0, :]),  # Length of first lattice vector
            np.linalg.norm(original_cell[1, :]),  # Length of second lattice vector  
            np.linalg.norm(original_cell[2, :])   # Length of third lattice vector
        ])
        
        # Create orthogonal cell using the computed lengths
        orthogonal_cell = np.diag(cell_lengths)
        
        logger.info(f"Orthogonal cell lengths: {cell_lengths}")
        logger.info(f"Orthogonal cell:\n{orthogonal_cell}")

        atoms_list = []
        
        for frame_idx in range(trajectory.n_frames):
            # Get positions for this frame
            positions = trajectory.positions[frame_idx].copy()  # Shape: (n_atoms, 3)
            
            # Wrap negative coordinates to positive using periodic boundary conditions
            positions[:, 0] = positions[:, 0] % cell_lengths[0]  # Wrap X
            positions[:, 1] = positions[:, 1] % cell_lengths[1]  # Wrap Y  
            positions[:, 2] = positions[:, 2] % cell_lengths[2]  # Wrap Z
            
            # Create ASE Atoms object with orthogonal cell
            atoms = ase.Atoms(
                symbols=element_list,
                positions=positions,
                pbc=[True, True, True],
                cell=orthogonal_cell
            )
            
            atoms_list.append(atoms)
            
        logger.info(f"Created {len(atoms_list)} ASE Atoms objects with orthogonal cells")
        return atoms_list

    def _convert_to_diffraction_patterns(self, exit_waves):
        """Convert exit waves to diffraction patterns in k-space."""
        # Use abtem's built-in diffraction pattern calculation
        # This automatically handles the FFT to k-space
        diffraction_patterns = exit_waves.diffraction_patterns(
            max_angle='full',  # Use full detector
            return_complex=True  # Keep complex values for phase information
        )
        
        return diffraction_patterns

    def _load_and_convert_to_wf_data(
        self,
        output_dir: Path,
        n_frames: int,
        probe_positions: List[Tuple[float, float]],
        timestep: float,
        sampling: float,
        cleanup_temp_files: bool = False
    ) -> WFData:
        """Load saved psi files and convert to WFData format."""
        
        # Load first file to get dimensions
        first_file = output_dir / "psi_t0.npy"
        if not first_file.exists():
            raise RuntimeError("No saved psi files found!")
            
        first_pattern = np.load(first_file)
        
        # Determine array dimensions
        if len(first_pattern.shape) == 3:
            # Shape: (n_probes, kx, ky) for convergent beam
            n_probes, n_kx, n_ky = first_pattern.shape
        elif len(first_pattern.shape) == 2:
            # Shape: (kx, ky) for plane wave
            n_kx, n_ky = first_pattern.shape
            n_probes = 1
        else:
            raise ValueError(f"Unexpected pattern shape: {first_pattern.shape}")

        # Create time array
        time_array = np.arange(n_frames) * timestep

        # Create k-space coordinates
        # Calculate proper k-space sampling based on real space grid
        # For a real space sampling of 'sampling' Å, the k-space sampling is:
        dk = 2 * np.pi / (n_kx * sampling)  # k-space sampling interval
        k_max = dk * n_kx / 2  # Maximum k value
        
        # Create symmetric k-space arrays
        kx = np.fft.fftfreq(n_kx, d=sampling) * 2 * np.pi  # Convert to inverse Angstroms
        ky = np.fft.fftfreq(n_ky, d=sampling) * 2 * np.pi  # Convert to inverse Angstroms
        
        # Sort to have symmetric arrays from -k_max to +k_max
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)

        # Create layer indices (single layer for now)
        layer = np.array([0])

        # Initialize wavefunction data array
        # Target shape: (n_probes, n_frames, n_kx, n_ky, n_layers)
        wavefunction_data = np.zeros((n_probes, n_frames, n_kx, n_ky, 1), dtype=complex)
        
        # Load each frame's data
        logger.info(f"Loading {n_frames} saved psi files...")
        for frame_idx in range(n_frames):
            psi_file = output_dir / f"psi_t{frame_idx}.npy"
            if not psi_file.exists():
                logger.warning(f"Missing psi file for frame {frame_idx}, filling with zeros")
                continue
                
            pattern = np.load(psi_file)
            
            if len(pattern.shape) == 3:
                # Convergent beam: (n_probes, kx, ky)
                for probe_idx in range(n_probes):
                    wavefunction_data[probe_idx, frame_idx, :, :, 0] = pattern[probe_idx, :, :]
            else:
                # Plane wave: (kx, ky)
                wavefunction_data[0, frame_idx, :, :, 0] = pattern

        # Handle psi file cleanup based on user preference
        if cleanup_temp_files:
            logger.info("Cleaning up psi files...")
            files_deleted = 0
            for frame_idx in range(n_frames):
                psi_file = output_dir / f"psi_t{frame_idx}.npy"
                if psi_file.exists():
                    psi_file.unlink()
                    files_deleted += 1
            logger.info(f"Deleted {files_deleted} psi files")
            
            # Try to remove the directory if it's empty
            try:
                output_dir.rmdir()
                logger.info(f"Removed directory: {output_dir}")
            except OSError:
                logger.info(f"Directory not empty, keeping: {output_dir}")
        else:
            logger.info(f"Psi files saved for future use in: {output_dir}")
            logger.info("  - Use these files to restart interrupted simulations")
            logger.info("  - Or for debugging individual frame results")

        return WFData(
            probe_positions=probe_positions,
            time=time_array,
            kx=kx,
            ky=ky,
            layer=layer,
            wavefunction_data=wavefunction_data
        )


# Example usage:
if __name__ == "__main__":
    # Example of how to use the ASE-based MultisliceCalculator
    from .trajectory import Trajectory
    import numpy as np

    # Create a simple example trajectory
    n_atoms = 10
    n_frames = 5
    positions = np.random.rand(n_frames, n_atoms, 3) * 10  # Random positions in 10Å cube
    velocities = np.random.rand(n_frames, n_atoms, 3) * 0.1
    atom_types = np.random.randint(1, 20, n_atoms)  # Random atomic numbers 1-19
    box_matrix = np.eye(3) * 20  # 20Å cubic box
    timestep = 0.001  # 1 fs in ps

    trajectory = Trajectory(
        atom_types=atom_types,
        positions=positions,
        velocities=velocities,
        box_matrix=box_matrix,
        timestep=timestep
    )

    # Initialize ASE-based calculator
    calculator = MultisliceCalculatorASE()

    # Run simulation
    wf_data = calculator.run_simulation(
        trajectory=trajectory,
        aperture=10.0,  # 10 mrad aperture
        voltage_kv=200.0,  # 200 kV
        pixel_size=0.1,  # 0.1 Å/pixel
        defocus=100.0,  # 100 Å defocus
        probe_positions=[(0.0, 0.0), (1.0, 1.0)],  # Two probe positions
    )

    print(f"ASE-based simulation completed successfully!")
    print(f"WFData shape: probe_positions={len(wf_data.probe_positions)}, time={len(wf_data.time)}, kx={len(wf_data.kx)}, ky={len(wf_data.ky)}")

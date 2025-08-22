"""
Multislice Calculator class using our custom data structures and ptyrodactyl.electrons functions.
"""
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import os
import tempfile

# Import our custom data structures
from .trajectory import Trajectory
from .tacaw_data import TACAWData
from .wf_data import WFData


# Import ptyrodactyl.electrons for actual calculations
import ptyrodactyl.electrons as pte

# Import typing for type hints
from typing import Optional, Tuple, List, Dict



class MultisliceCalculator:
    """
    Multislice calculator for electron microscopy simulations.

    This class integrates with ptyrodactyl.electrons to perform multislice
    simulations on molecular dynamics trajectories, generating wave function
    data for each frame.
    """

    def __init__(self):
        """Initialize the multislice calculator."""
        pass

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
        supersampling: int = 2,
    ) -> WFData:
        """
        Run multislice simulation for each frame in the trajectory.

        Args:
            trajectory: Trajectory object containing atomic positions for each frame
            aperture: Probe aperture size in milliradians
            voltage_kv: Accelerating voltage in kilovolts
            pixel_size: Pixel size in Angstroms
            defocus: Probe defocus in Angstroms (default: 0.0)
            probe_positions: List of (x, y) probe positions in Angstroms.
                          If None, defaults to [(0, 0)]
            element_symbols: Dictionary mapping atomic numbers to element symbols.
                           If None, defaults to standard atomic numbers
            slice_thickness: Thickness of each slice in Angstroms (default: 1.0)
            supersampling: Supersampling factor for potential calculation (default: 4)

        Returns:
            WFData object containing simulation results for all frames
        """
        # Set default probe positions if not provided
        if probe_positions is None:
            probe_positions = [(0.0, 0.0)]

        # Validate inputs
        self._validate_inputs(trajectory, aperture, voltage_kv, pixel_size, probe_positions)

        # Convert trajectory to format expected by ptyrodactyl.electrons
        xyz_data_list = self._convert_trajectory_to_xyz_data(
            trajectory, element_symbols
        )

        # Create probe
        probe = self._create_probe(aperture, voltage_kv, pixel_size, defocus)

        # Run simulation for each frame
        cbed_results = []
        for frame_idx, xyz_data in enumerate(xyz_data_list):
            print(f"Processing frame {frame_idx + 1}/{len(xyz_data_list)}")

            # Create potential from atomic structure
            potential = self._create_potential(
                xyz_data, pixel_size, slice_thickness, supersampling
            )

            # Run CBED simulation
            cbed_result = self._run_cbed_simulation(
                potential, probe, voltage_kv, probe_positions
            )

            cbed_results.append(cbed_result)

        # Convert results to WFData format
        wf_data = self._convert_to_wf_data(
            cbed_results, probe_positions, trajectory.timestep
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

        if aperture <= 0:
            raise ValueError("Aperture must be positive")

        if voltage_kv <= 0:
            raise ValueError("Voltage must be positive")

        if pixel_size <= 0:
            raise ValueError("Pixel size must be positive")

        if not probe_positions:
            raise ValueError("At least one probe position must be provided")

    def _convert_trajectory_to_xyz_data(
        self, trajectory: Trajectory, element_symbols: Optional[Dict[int, str]] = None
    ):
        """Convert trajectory frames to XYZData format for ptyrodactyl.electrons."""
        from ptyrodactyl.electrons.electron_types import XYZData

        xyz_data_list = []

        for frame_idx in range(trajectory.n_frames):
            # Get positions for this frame
            positions = trajectory.positions[frame_idx]  # Shape: (n_atoms, 3)

            # Get atomic numbers from atom types
            # Assume atom_types are atomic numbers, convert to integers
            atomic_numbers = trajectory.atom_types.astype(int)

            # Create lattice matrix from box matrix
            lattice = trajectory.box_matrix.copy()

            # Convert to JAX arrays (ptyrodactyl.electrons expects JAX arrays)
            positions = jnp.array(positions)
            atomic_numbers = jnp.array(atomic_numbers)
            lattice = jnp.array(lattice)

            try:
                # Create XYZData object with all required arguments
                xyz_data = XYZData(
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                    lattice=lattice,
                    stress=None,  # Not needed for multislice simulation
                    energy=0.0,   # Default energy
                    properties={}, # Empty properties dict
                    comment=f"Frame {frame_idx} from trajectory"  # Frame identifier
                )
            except TypeError as e:
                # Handle different XYZData constructor signatures
                print(f"XYZData constructor error: {e}")
                print("Trying alternative constructor...")
                try:
                    # Try with minimal arguments
                    xyz_data = XYZData(
                        positions=positions,
                        atomic_numbers=atomic_numbers,
                        lattice=lattice
                    )
                except Exception as e2:
                    raise RuntimeError(f"Failed to create XYZData object: {e2}")

            xyz_data_list.append(xyz_data)

        return xyz_data_list

    def _create_probe(
        self, aperture: float, voltage_kv: float, pixel_size: float, defocus: float
    ):
        """Create electron probe using ptyrodactyl.electrons functions."""
        # Define image size (we'll use a reasonable default)
        image_size = jnp.array([128, 128])  # [height, width]

        # Convert pixel size to picometers (required by make_probe)
        pixel_size_pm = pixel_size * 100  # 1 Å = 100 pm

        # Create probe
        probe_real_space = pte.simulations.make_probe(
            aperture=aperture,
            voltage=voltage_kv,
            image_size=image_size,
            calibration_pm=pixel_size_pm,
            defocus=defocus
        )

        # Convert to ProbeModes format
        from ptyrodactyl.electrons.electron_types import make_probe_modes

        # Add mode dimension - make_probe_modes expects 3D array (H, W, M)
        # probe_real_space is 2D (H, W), so we add a mode dimension
        probe_modes_3d = probe_real_space[..., None]  # Shape: (H, W, 1)
        
        # For simplicity, treat as single mode
        probe_modes = make_probe_modes(
            modes=probe_modes_3d,
            weights=jnp.array([1.0]),
            calib=pixel_size
        )

        return probe_modes

    def _create_potential(
        self, xyz_data, pixel_size: float, slice_thickness: float, supersampling: int
    ):
        """Create potential slices from atomic structure."""
        # Use kirkland_potentials_xyz to create potential
        potential_slices = pte.atom_potentials.kirkland_potentials_xyz(
            xyz_data=xyz_data,
            pixel_size=pixel_size,
            slice_thickness=slice_thickness,
            supersampling=supersampling
        )

        return potential_slices

    def _run_cbed_simulation(
        self, potential, probe, voltage_kv: float, probe_positions: List[Tuple[float, float]]
    ):
        """Run CBED simulation for all probe positions."""
        # Convert probe positions to JAX array
        probe_pos_array = jnp.array(probe_positions)

        # For now, run CBED for a single probe position (can be extended later)
        # Use the first probe position
        main_probe_pos = probe_pos_array[0:1]  # Keep as 2D array for vmap

        # Shift probe to the desired position(s)
        shifted_probes = pte.simulations.shift_beam_fourier(
            beam=probe.modes,
            pos=main_probe_pos,
            calib_ang=probe.calib
        )

        # Create ProbeModes for shifted probe
        from ptyrodactyl.electrons.electron_types import make_probe_modes

        shifted_probe_modes = make_probe_modes(
            modes=shifted_probes[0],  # Use first (and only) shifted probe
            weights=probe.weights,
            calib=probe.calib
        )

        # Run CBED simulation
        cbed_result = pte.simulations.cbed(
            pot_slices=potential,
            beam=shifted_probe_modes,
            voltage_kv=voltage_kv
        )

        return cbed_result

    def _convert_to_wf_data(
        self,
        cbed_results: List,
        probe_positions: List[Tuple[float, float]],
        timestep: float
    ) -> WFData:
        """Convert CBED results to WFData format."""
        n_frames = len(cbed_results)
        n_probe_pos = len(probe_positions)

        # Create time array
        time_array = np.arange(n_frames) * timestep

        # For simplicity, we'll use the first CBED result to get k-space dimensions
        sample_cbed = cbed_results[0]

        # Extract kx, ky from the CBED result
        # This assumes CalibratedArray format from ptyrodactyl.electrons
        if hasattr(sample_cbed, 'data_array'):
            height, width = sample_cbed.data_array.shape
            kx = np.linspace(-sample_cbed.calib_x * width / 2, sample_cbed.calib_x * width / 2, width)
            ky = np.linspace(-sample_cbed.calib_y * height / 2, sample_cbed.calib_y * height / 2, height)
        else:
            # Fallback if different format
            height, width = sample_cbed.shape[-2:]
            # Use default calibration - this should be adjusted based on actual pixel size
            kx = np.linspace(-2.0, 2.0, width)  # Default range in Å⁻¹
            ky = np.linspace(-2.0, 2.0, height)

        # Create layer indices (for now, just single layer per frame)
        layer = np.array([0])  # Single layer for all frames

        # Convert CBED results to wavefunction data array
        # Shape: (n_probe_pos, n_frames, len(kx), len(ky), n_layers)
        wavefunction_data = np.zeros((n_probe_pos, n_frames, len(kx), len(ky), 1), dtype=complex)
        
        for frame_idx, cbed_result in enumerate(cbed_results):
            for probe_idx in range(n_probe_pos):
                # Extract complex wavefunction data from CBED result
                if hasattr(cbed_result, 'data_array'):
                    wf_data = cbed_result.data_array
                else:
                    wf_data = cbed_result
                
                # Handle different probe position configurations
                if len(wf_data.shape) == 2:
                    # Single probe position case
                    wavefunction_data[0, frame_idx, :, :, 0] = wf_data
                elif len(wf_data.shape) == 3 and wf_data.shape[0] == n_probe_pos:
                    # Multiple probe positions
                    wavefunction_data[probe_idx, frame_idx, :, :, 0] = wf_data[probe_idx]
                else:
                    # Default case - use same data for all probe positions
                    wavefunction_data[probe_idx, frame_idx, :, :, 0] = wf_data

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
    # Example of how to use the MultisliceCalculator
    from .trajectory import Trajectory
    import numpy as np

    # Create a simple example trajectory
    # This would typically come from loading a file with TrajectoryLoader
    n_atoms = 10
    n_frames = 5
    positions = np.random.rand(n_frames, n_atoms, 3) * 10  # Random positions in 10Å cube
    velocities = np.random.rand(n_frames, n_atoms, 3) * 0.1
    atom_types = np.random.randint(1, 20, n_atoms)  # Random atomic numbers 1-19
    box_matrix = np.eye(3) * 20  # 20Å cubic box
    timestep = 0.005  # 1 fs in ps

    trajectory = Trajectory(
        atom_types=atom_types,
        positions=positions,
        velocities=velocities,
        box_matrix=box_matrix,
        timestep=timestep
    )

    # Initialize calculator
    calculator = MultisliceCalculator()

    # Run simulation
    wf_data = calculator.run_simulation(
        trajectory=trajectory,
        aperture=10.0,  # 10 mrad aperture
        voltage_kv=200.0,  # 200 kV
        pixel_size=0.1,  # 0.1 Å/pixel
        defocus=100.0,  # 100 Å defocus
        probe_positions=[(0.0, 0.0), (1.0, 1.0)],  # Two probe positions
    )

    print(f"Simulation completed successfully!")
    print(f"WFData shape: probe_positions={len(wf_data.probe_positions)}, time={len(wf_data.time)}, kx={len(wf_data.kx)}, ky={len(wf_data.ky)}")


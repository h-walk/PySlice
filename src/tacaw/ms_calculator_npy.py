"""
Optimized NumPy Multislice Calculator using class-based structure.

"""

import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List
from tqdm import tqdm
import time

# Import our optimized classes
from .potential import Potential
from .multislice_npy import Probe, Propagate

# Import TACAW data structures
from .trajectory import Trajectory
from .wf_data import WFData

logger = logging.getLogger(__name__)


class MultisliceCalculatorNumpy:
    """
    Optimized multislice calculator using class-based structure.
    

    """
    
    def __init__(self):

        # Element mapping for display purposes
        self.element_map = {
            1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 14: 'Si'
        }
    
    def run_simulation(
        self,
        trajectory: Trajectory,
        aperture: float = 0.0,
        voltage_kv: float = 60.0,
        pixel_size: float = 0.1,
        defocus: float = 0.0,
        slice_thickness: float = 0.5,
        sampling: float = 0.1,
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        batch_size: int = 10,
        save_path: Optional[Path] = None
    ) -> WFData:
        """
        Run multislice simulation using optimized class-based approach.
        
        Args:
            trajectory: Input trajectory data
            aperture: Objective aperture semi-angle in mrad
            voltage_kv: Accelerating voltage in kV
            pixel_size: Pixel size in Angstroms (for output scaling)
            defocus: Defocus in Angstroms (not implemented yet)
            slice_thickness: Thickness of each slice in Angstroms
            sampling: Sampling rate in Angstroms per pixel
            probe_positions: List of (x,y) probe positions in Angstroms
            batch_size: Number of frames to process at once
            save_path: Optional path to save wave function data
            
        Returns:
            WFData: Wave function data containing complex amplitudes
        """
        
        logger.info(f"Starting optimized multislice simulation")
        logger.info(f"Trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
        logger.info(f"Simulation box: {trajectory.box_matrix[0,0]:.1f} × {trajectory.box_matrix[1,1]:.1f} × {trajectory.box_matrix[2,2]:.1f} Å")
        logger.info(f"Parameters: aperture={aperture}mrad, voltage={voltage_kv}kV, sampling={sampling}Å")
        
        # Get electron energy and wavelength
        eV = voltage_kv * 1000  # Convert kV to eV
        
        # Set up spatial grids
        lx, ly, lz = trajectory.box_matrix[0,0], trajectory.box_matrix[1,1], trajectory.box_matrix[2,2]
        
        # Create grids based on sampling
        nx = int(lx / sampling) + 1
        ny = int(ly / sampling) + 1  
        nz = int(lz / slice_thickness) + 1
        
        xs = np.linspace(0, lx, nx)
        ys = np.linspace(0, ly, ny)
        zs = np.linspace(0, lz, nz)
        
        logger.info(f"Grid dimensions: {nx} × {ny} × {nz} = {nx*ny*nz:,} points")
        logger.info(f"Grid spacing: {xs[1]-xs[0]:.3f} × {ys[1]-ys[0]:.3f} × {zs[1]-zs[0]:.3f} Å")
        
        # Set up default probe position if not provided
        if probe_positions is None:
            probe_positions = [(lx/2, ly/2)]  # Center probe
        
        # Initialize storage for results
        n_frames = trajectory.n_frames
        n_probes = len(probe_positions)
        
        # Storage: [frame, probe, x, y, z, complex]
        wavefunction_data = np.zeros((n_frames, n_probes, nx, ny, 1, 1), dtype=complex)
        
        # Process frames in batches
        total_start_time = time.time()
        
        for batch_start in range(0, n_frames, batch_size):
            batch_end = min(batch_start + batch_size, n_frames)
            batch_frames = list(range(batch_start, batch_end))
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(n_frames + batch_size - 1)//batch_size}: frames {batch_start}-{batch_end-1}")
            
            # Process each frame in the batch
            for frame_idx in tqdm(batch_frames, desc=f"Batch {batch_start//batch_size + 1}"):
                
                # Get atomic positions for this frame
                positions = trajectory.positions[frame_idx]  # Shape: (n_atoms, 3)
                atom_types = trajectory.atom_types  # Shape: (n_atoms,)
                
                # Convert atom types to element names for Potential class
                atom_type_names = []
                for atom_type in atom_types:
                    if atom_type in self.element_map:
                        atom_type_names.append(self.element_map[atom_type])
                    else:
                        # Use atomic number directly
                        atom_type_names.append(atom_type)
                
                # Create optimized potential for this frame
                potential = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland")
                
                # Create probe for this frame
                probe = Probe(xs, ys, aperture, eV)
                
                # Propagate probe through potential for each probe position
                for probe_idx, (px, py) in enumerate(probe_positions):
                    
                    # For now, use the same probe for all positions
                    # In the future, could offset the probe for different positions
                    wavefunction = Propagate(probe, potential)
                    
                    # Store result - only keep final wavefunction (z-integrated)
                    wavefunction_data[frame_idx, probe_idx, :, :, 0, 0] = wavefunction
        
        total_time = time.time() - total_start_time
        logger.info(f"Optimized simulation completed in {total_time:.2f}s")
        
        # Create metadata
        params = {
            'aperture': aperture,
            'voltage_kv': voltage_kv,
            'pixel_size': pixel_size,
            'defocus': defocus,
            'slice_thickness': slice_thickness,
            'sampling': sampling,
            'grid_shape': (nx, ny, nz),
            'box_size': (lx, ly, lz),
            'n_atoms': trajectory.n_atoms,
            'calculator': 'MultisliceCalculatorNumpy'
        }
        
        # Create coordinate arrays for output
        # Note: WFData expects (probe_positions, time, kx, ky, layer) format
        # Convert real space coordinates to k-space coordinates via FFT frequencies
        kx = np.fft.fftfreq(nx, sampling)  # k-space sampling
        ky = np.fft.fftfreq(ny, sampling)  # k-space sampling
        time_array = np.arange(n_frames) * trajectory.timestep  # Time array in ps
        layer_array = np.array([0])  # Single layer for now
        
        # Package results
        wf_data = WFData(
            probe_positions=probe_positions,
            time=time_array,
            kx=kx,
            ky=ky,
            layer=layer_array,
            wavefunction_data=wavefunction_data
        )
        
        # Save if requested
        if save_path is not None:
            wf_data.save(save_path)
            logger.info(f"Wave function data saved to {save_path}")
        
        return wf_data

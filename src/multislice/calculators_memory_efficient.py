import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List
from tqdm import tqdm
import time
import hashlib

try:
    import torch ; xp = torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if device.type == 'mps':
        complex_dtype = torch.complex64
        float_dtype = torch.float32
    else:
        complex_dtype = torch.complex128
        float_dtype = torch.float64

except ImportError:
    xp = np
    TORCH_AVAILABLE = False
    complex_dtype = np.complex128
    float_dtype = np.float64


from .potentials import gridFromTrajectory,Potential
from .multislice import Probe,Propagate,create_batched_probes
from .trajectory import Trajectory
from ..postprocessing.tacaw_data_per_probe import TACAWDataPerProbe

logger = logging.getLogger(__name__)

class MultisliceCalculatorMemoryEfficient:
    
    def __init__(self, device=None, force_cpu=False):
        """
        Initialize the PyTorch-accelerated calculator.
        
        Args:
            device: PyTorch device ('cpu', 'cuda', or None for auto-detection)
            force_cpu: Force CPU usage even if GPU is available
        """
        if not TORCH_AVAILABLE:
            if device is not None:
                logger.warning("PyTorch not available, falling back to NumPy implementation")
            self.device = None
        else:
            if force_cpu:
                self.device = torch.device('cpu')
            elif device is not None:
                self.device = torch.device(device)
            else:
                # Auto-detect best available device: CUDA > MPS > CPU
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                elif torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    self.device = torch.device('cpu')
            
            logger.info(f"Memory-efficient PyTorch calculator initialized on device: {self.device}")
        
        # Element mapping for display purposes
        self.element_map = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
            30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr'
        }
    
    def _generate_cache_key(self, trajectory, aperture, voltage_eV,
                           slice_thickness, sampling, probe_positions):
        """Generate unique cache key for simulation parameters."""
        params = {
            'n_frames': trajectory.n_frames,
            'n_atoms': trajectory.n_atoms,
            'box_matrix': trajectory.box_matrix.tolist(),
            'atom_types': trajectory.atom_types.tolist(),
            'aperture': aperture,
            'voltage_eV': voltage_eV,
            'slice_thickness': slice_thickness,
            'sampling': sampling,
            'probe_positions': probe_positions,
            'backend': 'pytorch' if TORCH_AVAILABLE else 'numpy'
        }
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    def setup(
        self,
        trajectory: Trajectory,
        aperture: float = 0.0,
        voltage_eV: float = 60e3,
        defocus: float = 0.0,
        slice_thickness: float = 0.5,
        sampling: float = 0.1,
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        batch_size: int = 10,
        save_path: Optional[Path] = None,
        cleanup_temp_files: bool = False,
        slice_axis: int = 2,
    ):
        """
        Set up multislice simulation using PyTorch acceleration.
        
        Args:
            trajectory: Input trajectory data
            aperture: Objective aperture semi-angle in mrad
            voltage_eV: Accelerating voltage in eV
            defocus: Defocus in Angstroms (not implemented yet)
            slice_thickness: Thickness of each slice in Angstroms
            sampling: Sampling rate in Angstroms per pixel
            probe_positions: List of (x,y) probe positions in Angstroms
            batch_size: Number of frames to process at once
            save_path: Optional path to save wave function data
            cleanup_temp_files: Whether to delete temp files after loading
        """
        
        self.trajectory = trajectory
        self.aperture = aperture
        self.voltage_eV = voltage_eV
        self.defocus = defocus
        self.slice_thickness = slice_thickness
        self.sampling = sampling
        self.probe_positions = probe_positions
        self.save_path = save_path
        self.cleanup_temp_files = cleanup_temp_files
        self.slice_axis = slice_axis

        # Generate cache key and setup output directory
        cache_key = self._generate_cache_key(trajectory, aperture, voltage_eV,
                                           slice_thickness, sampling, probe_positions)
        self.output_dir = Path("psi_data") / f"torch_{cache_key}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
                
        # Set up spatial grids
        xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=sampling,slice_thickness=slice_thickness)
        nx=len(xs) ; ny=len(ys) ; nz=len(zs)
        self.xs = xs ; self.ys = ys ; self.zs = zs
        self.lx = lx ; self.ly = ly ; self.lz = lz
        self.nx = nx ; self.ny = ny ; self.nz = nz
        self.dx = xs[1]-xs[0] ; self.dy = ys[1]-ys[0] ; self.dy = ys[1]-ys[0]

        # Set up default probe position if not provided
        if self.probe_positions is None:
            self.probe_positions = [(lx/2, ly/2)]  # Center probe
        self.base_probe = Probe(xs, ys, self.aperture, self.voltage_eV)

        # Initialize storage for results
        self.n_frames = trajectory.n_frames
        self.n_probes = len(self.probe_positions)
        
        # DON'T pre-allocate the huge array! This was causing memory explosion
        # self.wavefunction_data = xp.zeros((self.n_probes, self.n_frames, nx, ny, 1), dtype=complex_dtype)
        
        # Instead, we'll build the data frame by frame
        logger.info(f"Will process {self.n_probes} probes x {self.n_frames} frames without pre-allocation")
        
    def run(self) -> TACAWDataPerProbe:
        """
        Run simulation and return TACAWData directly (skip WFData intermediate).
        """
        # Process frames with caching and multiprocessing
        total_start_time = time.time()
        frames_computed = 0
        frames_cached = 0

        # Create coordinate arrays for TACAW
        kxs = xp.fft.fftshift(xp.fft.fftfreq(self.nx, self.sampling) * 2 * xp.pi)
        kys = xp.fft.fftshift(xp.fft.fftfreq(self.ny, self.sampling) * 2 * xp.pi)
        time_array = np.arange(self.n_frames) * self.trajectory.timestep

        # Compute frequencies for TACAW
        dt = self.trajectory.timestep
        frequencies = np.fft.fftfreq(self.n_frames, d=dt)
        frequencies = np.fft.fftshift(frequencies)

        # Process frames one at a time - save to cache files only (no big array!)
        with tqdm(total=self.n_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx in range(self.n_frames):
                cache_file = self.output_dir / f"frame_{frame_idx}.npy"
                positions = self.trajectory.positions[frame_idx]
                atom_types = self.trajectory.atom_types
                
                args = (frame_idx, positions, atom_types, self.xs, self.ys, self.zs, 
                       self.aperture, self.voltage_eV, self.base_probe, self.probe_positions, self.element_map, 
                       cache_file, self.slice_axis)
                
                # Process frame (this saves to cache_file automatically)
                frame_idx_result, frame_data, was_cached = _process_frame_worker_torch(args)
                
                # No need to store in memory - it's in the cache file!
                
                if was_cached:
                    frames_cached += 1
                else:
                    frames_computed += 1
                
                # Update progress bar for this frame
                pbar.update(1)
        
        # Convert coordinate arrays to numpy if needed
        if TORCH_AVAILABLE:
            kxs = kxs.cpu().numpy() if hasattr(kxs, 'cpu') else kxs
            kys = kys.cpu().numpy() if hasattr(kys, 'cpu') else kys
        
        total_time = time.time() - total_start_time
        logger.info(f"Simulation completed in {total_time:.2f}s ({frames_computed} computed, {frames_cached} cached)")
        
        # Create per-probe TACAW data - no big arrays allocated!
        tacaw_data = TACAWDataPerProbe(
            probe_positions=self.probe_positions,
            time=time_array,
            kxs=kxs,
            kys=kys,
            layer=np.array([0]),
            probe=self.base_probe,
            cache_dir=self.output_dir
        )
        
        logger.info("Per-probe TACAW ready - WFData objects created per probe as needed")
        
        # Handle cleanup
        if self.cleanup_temp_files:
            logger.info("Cleaning up cache files...")
            for frame_idx in range(self.n_frames):
                cache_file = self.output_dir / f"frame_{frame_idx}.npy"
                if cache_file.exists():
                    cache_file.unlink()
            try:
                self.output_dir.rmdir()
            except OSError:
                pass
        else:
            logger.info(f"Cache files saved in: {self.output_dir}")
        
        return tacaw_data


def _process_frame_worker_torch(args):
    frame_idx, positions, atom_types, xs, ys, zs, aperture, eV, probe, probe_positions, element_map, cache_file, slice_axis = args
    
    if cache_file.exists():
        return frame_idx, xp.asarray(np.load(cache_file)), True # if always saving as numpy, then must cast to torch array if re-reading cache file back in
    
    if TORCH_AVAILABLE:
        worker_device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        worker_device = None
    
    atom_type_names = []
    for atom_type in atom_types:
        if atom_type in element_map:
            atom_type_names.append(element_map[atom_type])
        else:
            atom_type_names.append(atom_type)
    
    #try:
    potential = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland", device=worker_device, slice_axis=slice_axis)

    n_probes = len(probe_positions)
    nx, ny = len(xs), len(ys)
    frame_data = xp.zeros((n_probes, nx, ny, 1, 1), dtype=complex_dtype)
    
    batched_probes = create_batched_probes(probe, probe_positions, worker_device)
    exit_waves_batch = Propagate(batched_probes, potential, worker_device)
        
    # Convert all exit waves to k-space
    kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}
    exit_waves_k = xp.fft.fft2(exit_waves_batch, **kwarg)
    diffraction_patterns = xp.fft.fftshift(exit_waves_k, **kwarg)
        
     # Store results
    frame_data[:, :, :, 0, 0] = diffraction_patterns #.cpu().numpy()

    np.save(cache_file, frame_data)
    return frame_idx, frame_data, False
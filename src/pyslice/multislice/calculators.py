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
from ..postprocessing.wf_data import WFData
from .sed import SED
from ..backend import zeros,expand_dims

logger = logging.getLogger(__name__)

class MultisliceCalculator:
    
    def __init__(self, device=None, force_cpu=False):
        """
        Initialize the PyTorch-accelerated calculator.

        Args:
            device: PyTorch device ('cpu', 'cuda', 'mps', or None for auto-detection)
            force_cpu: Force CPU usage even if GPU is available
        """
        if not TORCH_AVAILABLE:
            if device is not None:
                logger.warning("PyTorch not available, falling back to NumPy implementation")
            self.device = None
            self.force_cpu = False
        else:
            self.force_cpu = force_cpu
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

            logger.info(f"PyTorch calculator initialized on device: {self.device}")
        
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
        firstNAtoms = [ str(np.round(v,4)) for v in trajectory.positions[0,:100,0] ] # first timestep's first 10 atom's x positions
        params = {
            'firstNAtoms': ",".join(firstNAtoms), # WHY? prevents the same script from re-using psi_data when positions change
            'n_frames': trajectory.n_frames,
            'n_atoms': trajectory.n_atoms,
            'box_matrix': trajectory.box_matrix.tolist(),
            'atom_types': trajectory.atom_types.tolist(),
            'aperture': aperture,
            'voltage_eV': voltage_eV,
            'slice_thickness': slice_thickness,
            'sampling': sampling,
            'probe_positions': probe_positions,
            'backend': 'pytorch' if TORCH_AVAILABLE else 'numpy',
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
        probe_xs: Optional[List[float]] = None,
        probe_ys: Optional[List[float]] = None,
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        batch_size: int = 10,
        save_path: Optional[Path] = None,
        cleanup_temp_files: bool = False,
        slice_axis: int = 2,
        cache_levels: list = ["exitwaves"], # options include: exitwaves, slices, potentials (this replaces store_all_slices)
        max_kx = np.inf,
        max_ky = np.inf,
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
            store_all_slices: If True, store wavefunction at each slice for 3D visualization
        """

        self.trajectory = trajectory
        self.aperture = aperture
        self.voltage_eV = voltage_eV
        self.defocus = defocus
        self.slice_thickness = slice_thickness
        self.sampling = sampling
        self.probe_xs = probe_xs
        self.probe_ys = probe_ys
        self.probe_positions = probe_positions
        self.save_path = save_path
        self.cleanup_temp_files = cleanup_temp_files
        self.slice_axis = slice_axis
        self.cache_levels = cache_levels
        self.max_kx = max_kx
        self.max_ky = max_ky
                
        # Set up spatial grids
        xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=sampling,slice_thickness=slice_thickness)
        nx=len(xs) ; ny=len(ys) ; nz=len(zs)
        self.xs = xs ; self.ys = ys ; self.zs = zs
        self.lx = lx ; self.ly = ly ; self.lz = lz
        self.nx = nx ; self.ny = ny ; self.nz = nz
        self.dx = xs[1]-xs[0] ; self.dy = ys[1]-ys[0] ; self.dy = ys[1]-ys[0]

        # calculate kxs kys here, so we can crop them, since we'll pre-allocate wavefunction_data below
        self.kxs = xp.fft.fftshift(xp.fft.fftfreq(self.nx, self.sampling))  # k-space in 1/Å
        self.kys = xp.fft.fftshift(xp.fft.fftfreq(self.ny, self.sampling))  # k-space in 1/Å
        self.i1 = xp.argwhere(self.kxs >= -max_kx)[0][0]   # first element >=
        self.i2 = xp.argwhere(self.kxs <= max_kx)[-1][0]+1 # last element <=, +1, so i1:i2 includes i2
        self.j1 = xp.argwhere(self.kys >= -max_ky)[0][0]
        self.j2 = xp.argwhere(self.kys <= max_ky)[-1][0]+1
        self.kxs = self.kxs[self.i1:self.i2]
        self.kys = self.kys[self.j1:self.j2]
        self.nx = self.i2 - self.i1 ; self.ny = self.j2 - self.j1 ; nx = self.nx ; ny = self.ny

        # Preferred to pass probe_xs and probe_ys from which we will define a grid
        if self.probe_xs is not None and self.probe_ys is not None:
            x,y = np.meshgrid(self.probe_xs,self.probe_ys,indexing='ij')
            self.probe_positions = np.asarray(list(zip(x.flat,y.flat)))
            lx = len(self.probe_xs) ; ly = len(self.probe_ys)

        # Set up default probe position if not provided
        if self.probe_positions is None:
            self.probe_positions = [(lx/2, ly/2)]  # Center probe
            self.probe_xs = [lx/2] ; self.probe_ys = [ly/2]

        # Generate cache key and setup output directory
        cache_key = self._generate_cache_key(trajectory, aperture, voltage_eV,
                                           slice_thickness, sampling, self.probe_positions)
        #print(cache_key)
        self.output_dir = Path("psi_data/" + ("torch" if TORCH_AVAILABLE else "numpy") + "_"+cache_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)

		
        # Create probe on the correct device from the start
        self.base_probe = Probe(xs, ys, self.aperture, self.voltage_eV, device=self.device)

        # Initialize storage for results
        self.n_frames = trajectory.n_frames
        self.n_probes = len(self.probe_positions)

        # Set dtype based on the actual device we're using
        if TORCH_AVAILABLE and self.device is not None:
            if self.device.type == 'mps':
                self.complex_dtype = torch.complex64
                self.float_dtype = torch.float32
            else:
                self.complex_dtype = torch.complex128
                self.float_dtype = torch.float64
        else:
            self.complex_dtype = np.complex128
            self.float_dtype = np.float64

        # Storage: [probe, frame, x, y, layer] - matches WFData expected format
        self.n_layers = nz if "slices" in cache_levels else 1
        self.wavefunction_data = zeros((self.n_probes, self.n_frames, nx, ny, self.n_layers),
                                                   dtype=self.complex_dtype, device=self.device)
        
    def run(self) -> WFData:

        # Process frames with caching and multiprocessing
        total_start_time = time.time()
        frames_computed = 0
        frames_cached = 0

        # quality of life sanity checks: user may have set things (e.g. probe array) with the wrong data type (e.g. numpy instead of tensor). let's try to catch and correct those here
        if isinstance(self.base_probe._array,np.ndarray) and TORCH_AVAILABLE:
            self.base_probe._array = xp.tensor(self.base_probe._array)

        # Process frames one at a time with tqdm progress tracking
        with tqdm(total=self.n_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx in range(self.n_frames):
                cache_file = self.output_dir / f"frame_{frame_idx}.npy"
                # Show detailed progress for single-frame runs
                show_progress = (frame_idx == 0 and self.n_frames == 1)

                positions = self.trajectory.positions[frame_idx]
                atom_types = self.trajectory.atom_types
                atom_type_names = []
                for atom_type in atom_types:
                    if atom_type in self.element_map:
                        atom_type_names.append(self.element_map[atom_type])
                    else:
                        atom_type_names.append(atom_type)

                # frame_data should always be shaped: n_probes,nkx,nky,n_layers,1 (idk why there's a trailing 1)
                cache_exists,frame_data = checkCache(cache_file,self.cache_levels)

                if cache_exists:
                    frames_cached += 1
                else:
                    potential = Potential(self.xs, self.ys, self.zs, positions, atom_type_names, kind="kirkland", device=self.device, slice_axis=self.slice_axis, progress=show_progress, cache_dir=cache_file.parent if "potentials" in self.cache_levels else None, frame_idx = frame_idx)

                    n_probes = len(self.probe_positions)
                    nx, ny = len(self.xs), len(self.ys)
                    n_slices = len(self.zs)

                    batched_probes = create_batched_probes(self.base_probe, self.probe_positions, self.device)
                    # Propagate returns: [l,p,x,y] where l,p are both optional (if store_all_slices=True, and if n_probes>1)
                    exit_waves_batch = Propagate(batched_probes, potential, self.device, progress=show_progress, onthefly=True, store_all_slices = ("slices" in self.cache_levels) )
                    #print(exit_waves_batch.shape)
                    if n_probes==1 and "slices" not in self.cache_levels:
                        exit_waves_batch = expand_dims(exit_waves_batch,0)
                    if "slices" not in self.cache_levels:
                        exit_waves_batch = expand_dims(exit_waves_batch,0)
                    #print(exit_waves_batch.shape)
                    # frame_data is always: p,x,y,l,1 (self.wavefunction_data expects p,t,x,y,l, since we loop time. recall Propagate gave l,p,x,y)
                    frame_data = zeros((n_probes, nx, ny, self.n_layers,1), dtype=self.complex_dtype, device=self.device)
                    #print(frame_data.shape)
                    for layer_idx in range(self.n_layers):
                        kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}
                        exit_waves_k = xp.fft.fft2(exit_waves_batch[layer_idx,:,:,:], **kwarg) # l,p,x,y --> p,x,y
                        diffraction_patterns = xp.fft.fftshift(exit_waves_k, **kwarg)
                        cropped = diffraction_patterns[:,self.i1:self.i2,self.j1:self.j2]
                        frame_data[:,:,:,layer_idx,0] = cropped # load p,x,y --> p,x,y,l,1 indices

                    # Convert to CPU numpy array for saving
                    if TORCH_AVAILABLE and hasattr(frame_data, 'cpu'):
                        frame_data_cpu = frame_data.cpu().numpy()
                    else:
                        frame_data_cpu = frame_data

                    if "exitwaves" in self.cache_levels or "slices" in self.cache_levels:
                        np.save(cache_file, frame_data_cpu)
                    frames_computed += 1

                self.wavefunction_data[:, frame_idx, :, :, :] = frame_data[:, :, :, :, 0] # load p,x,y,l,1 --> p,t,x,y,l indices
                # Update progress bar for this frame
                pbar.update(1)
        
        total_time = time.time() - total_start_time
        logger.info(f"Simulation completed in {total_time:.2f}s ({frames_computed} computed, {frames_cached} cached)")
        
        # Create metadata
        params = {
            'aperture': self.aperture,
            'voltage_eV': self.voltage_eV,
            'defocus': self.defocus,
            'slice_thickness': self.slice_thickness,
            'sampling': self.sampling,
            'grid_shape': (self.nx, self.ny, self.nz),
            'box_size': (self.lx, self.ly, self.lz),
            'n_atoms': self.trajectory.n_atoms,
            'calculator': 'MultisliceCalculator'
        }
        
        # Create coordinate arrays for output
        # Note: WFData expects (probe_positions, time, kx, ky, layer) format
        # Create k-space coordinates to match expected format (same as AbTem)
        # TWP: If we're not going to also provide a shifted/etc reciprocal_array, we shouldn't shift the kxs
        #kxs = xp.fft.fftfreq(self.nx, d=self.dx)
        #kys = xp.fft.fftfreq(self.ny, d=self.dy)
        #kxs = xp.fft.fftshift(xp.fft.fftfreq(self.nx, self.sampling))  # k-space in 1/Å MOVING TO INIT SO WE CAN CROP ON-THE-FLY
        #kys = xp.fft.fftshift(xp.fft.fftfreq(self.ny, self.sampling))  # k-space in 1/Å
        time_array = np.arange(self.n_frames) * self.trajectory.timestep  # Time array in ps
        layer_array = np.arange(self.nz) if "slices" in self.cache_levels else np.array([0])  # Layer indices
        
        # Package results
        wf_data = WFData(
            probe_positions=self.probe_positions,
            probe_xs=self.probe_xs,
            probe_ys=self.probe_ys,
            time=time_array,
            kxs=self.kxs,
            kys=self.kys,
            xs=self.xs,
            ys=self.ys,
            layer=layer_array,
            array=self.wavefunction_data,
            probe=self.base_probe,
            cache_dir=self.output_dir
        )
        
        # Handle cleanup
        if self.cleanup_temp_files:
            logger.info("Cleaning up cache files...")
            for frame_idx in range(n_frames):
                cache_file = self.output_dir / f"frame_{frame_idx}.npy"
                if cache_file.exists():
                    cache_file.unlink()
            try:
                self.output_dir.rmdir()
            except OSError:
                pass
        else:
            logger.info(f"Cache files saved in: {self.output_dir}")
        
        # Save if requested - psi files already saved during processing
        
        return wf_data

logging_tracker=[]
def checkCache(cache_file,cache_levels):
    global logging_tracker
    if cache_file.exists() and ( "exitwaves" in cache_levels or "slices" in cache_levels ):
        parent = str(cache_file.parent)
        if "cache_exists-"+parent not in logging_tracker:
            logging_tracker.append("cache_exists-"+parent)
            logging.warning("One or more frames reloaded from cache: "+str(cache_file.parent))
        return True,xp.asarray(np.load(cache_file)) # if always saving as numpy, then must cast to torch array if re-reading cache file back in
    return False,0


class SEDCalculator:
    def setup(self, trajectory: Trajectory, axis:int = 2, abc:list = [1,1,1]):
        """
        Set up Spectral Energy Density calculation
        
        Args:
            trajectory: Input trajectory data
        """

        self.trajectory = trajectory
        self.axis = axis
        self.a,self.b,self.c = abc

        # Set up spatial grids
        lxyz = list( np.diag(trajectory.box_matrix) )
        nxyz = [ int(np.round(l/d)) for l,d in zip(lxyz,abc) ]
		
        del lxyz[axis]
        del nxyz[axis]
        del abc[axis]

        self.kxs=np.linspace(0,2*np.pi/abc[0],nxyz[0])
        self.kys=np.linspace(0,2*np.pi/abc[1],nxyz[1])

        self.kvec = np.zeros((len(self.kxs),len(self.kys),3))
        self.kvec[:,:,0] += self.kxs[:,None]
        self.kvec[:,:,1] += self.kys[None,:]


    def run(self) -> WFData:
        avg = self.trajectory.get_mean_positions()
        disp = self.trajectory.get_distplacements()

        # RUN SED INSTEAD OF MULTISLICE
        self.Zx,ws = SED(avg,disp,kvec=self.kvec,v_xyz=0)
        self.Zy,ws = SED(avg,disp,kvec=self.kvec,v_xyz=1)
        self.Zz,ws = SED(avg,disp,kvec=self.kvec,v_xyz=2)

        self.ws = ws/self.trajectory.timestep

    def plot(self,w,filename=None): # TODO MAYBE "RUN" SHOULD RETURN A TACAW OBJECT SO WE CAN REUSE TACAW PLOTTING/POSTPROCESSING FUNCTIONALITY??
        import matplotlib.pyplot as plt

        #fig, ax = plt.subplots()
        #extent = ( np.amin(kxs) , np.amax(kxs) , np.amin(ws) , np.amax(ws) )
        #ax.imshow((Zx[::-1,:,0]+Zy[::-1,:,0]+Zz[::-1,:,0])**.25, cmap="inferno", extent=extent,aspect="auto")
        #plt.show()

        i=np.argmin(np.absolute(self.ws-w))
        extent = ( np.amin(self.kxs) , np.amax(self.kxs) , np.amin(self.kys) , np.amax(self.kys) )

        fig, ax = plt.subplots()
        ax.imshow(np.sqrt(self.Zx[i,:,:]+self.Zy[i,:,:]+self.Zz[i,:,:]).T, cmap="inferno", extent=extent)
        ax.set_xlabel("kx ($\\AA^{-1}$)")
        ax.set_ylabel("ky ($\\AA^{-1}$)")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()



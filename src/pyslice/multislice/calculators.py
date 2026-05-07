import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List
from tqdm import tqdm
import time,os
import hashlib
#from line_profiler import profile

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
from .multislice import Probe,PrismProbe,Propagate,create_batched_probes
from .trajectory import Trajectory
from ..postprocessing.wf_data import WFData
from .sed import SED
from ..backend import zeros,expand_dims,to_cpu,memmap,ones,sum,absolute,ceil,einsum,asarray,astype

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
                           slice_thickness, sampling, probe_positions,
                           spatial_decoherence, temporal_decoherence,
                           probe_array=None):
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
        if spatial_decoherence is not None:
            params['spatial_decoherence'] = spatial_decoherence
        if temporal_decoherence is not None:
            params['temporal_decoherence'] = temporal_decoherence
        if probe_array is not None:
            probe_np = np.ascontiguousarray(to_cpu(probe_array).ravel()[:1000])
            params['probe_hash'] = hashlib.md5(probe_np.tobytes()).hexdigest()
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
        cache_layer_indices: Optional[List[int]] = None, # NEW: subset of slice indices to store; None = store all layers
        max_kx = np.inf,
        max_ky = np.inf,
        use_memmap = False,
        loop_probes = False,
        min_dk = 0,
        prism = False,
        kth=1,
        ADF=False,
        store_full=True,
        skip_vacuum=False
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
            cache_layer_indices: Optional list of slice-layer indices (0-based) to record when
                cache_levels includes "slices". If None (default), all nz layers are stored.
                Specifying a small subset (e.g. the 6 depths needed for EELS thickness series)
                can reduce disk usage by >98% without affecting propagation accuracy.
                Example: cache_layer_indices=[44, 88, 176, 264, 352, 440]
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
        self.cache_layer_indices = cache_layer_indices  # NEW: store for use in run()
        self.max_kx = max_kx
        self.max_ky = max_ky
        self.use_memmap = use_memmap   # bool: frame_data (p,x,y,l,1) and wavefunction_data (p,t,x,y,l) will be memmapped instead of held in RAM
        self.loop_probes = loop_probes # False or int: multiple probes (p,x,y) can be propagated simultaneously. this allows processing in chunks
        self.min_dk = min_dk           # float: Δk=1/L, so this will pre-crop each probe and potential slice so a smaller area is propagated
        self.prism = prism             # False or int: PRISM algorithm implementation, this denotes how many fourier components are used in kx ky
        self.kth = kth                 # int: Δk=1/L, nk = nx. huge systems waste RAM with ultra-fine Δk. this sparsifies the exitwaves via ::kth
        self.ADF = ADF                 # bool or (inner,outer): allows on-the-fly calculation of the ADF signal
        self.store_full = store_full   # bool: if ADF=True and prism=False, this skips storing of the full [t],x,y,kx,ky 5D exit data
        self.skip_vacuum = skip_vacuum # bool: if True, we skip propagation of probes in locations where there are no atoms

        # Set up spatial grids
        xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=sampling,slice_thickness=slice_thickness)
        nx=len(xs) ; ny=len(ys) ; nz=len(zs)
        self.xs = xs ; self.ys = ys ; self.zs = zs
        self.lx = lx ; self.ly = ly ; self.lz = lz
        self.nx = nx ; self.ny = ny ; self.nz = nz
        self.dx = xs[1]-xs[0] ; self.dy = ys[1]-ys[0] ; self.dy = ys[1]-ys[0]

        self.probe_cropping = 0
        if self.min_dk > 0: # dk = 1/L = 1/(nx*sampling)
            nx = int(np.round(1/(self.min_dk*self.sampling)))
            self.nx = nx ; self.ny = nx
            self.probe_cropping = nx

        self.kxs = xp.fft.fftshift(xp.fft.fftfreq(self.nx, self.sampling))  # k-space in 1/Å
        self.kys = xp.fft.fftshift(xp.fft.fftfreq(self.ny, self.sampling))  # k-space in 1/Å
        kx_mask = xp.zeros(self.nx)+1 ; ky_mask = xp.zeros(self.ny)+1
        kx_mask[ self.kxs < -max_kx ] = 0 ; kx_mask[ self.kxs > max_kx ] = 0
        ky_mask[ self.kys < -max_ky ] = 0 ; ky_mask[ self.kys > max_ky ] = 0
        self.keep_kxs_indices = xp.arange(self.nx)[kx_mask==1][::self.kth]
        self.keep_kys_indices = xp.arange(self.ny)[ky_mask==1][::self.kth]
        self.nx = len(self.keep_kxs_indices) ; self.ny = len(self.keep_kys_indices)

        # Preferred to pass probe_xs and probe_ys from which we will define a grid
        if self.probe_xs is not None and self.probe_ys is not None:
            x,y = np.meshgrid(self.probe_xs,self.probe_ys)
            self.probe_positions = np.reshape([x,y],(2,len(x.flat))).T # x,y looped indices to match what multislice.Probe does

        # If probe_positions provided but not probe_xs/probe_ys, derive them
        elif self.probe_positions is not None:
            positions = np.asarray(self.probe_positions)
            self.probe_xs = sorted(list(set(positions[:, 0])))
            self.probe_ys = sorted(list(set(positions[:, 1])))

        # Set up default probe position if not provided
        if self.probe_positions is None:
            self.probe_positions = [(lx/2, ly/2)]  # Center probe
            self.probe_xs = [lx/2] ; self.probe_ys = [ly/2]

        if self.prism:
            # Prism algorithm works by passing a series of sinusoids (fourier components shared by all probes) through the sample. "PrismProbe" will therefore give us a series of sinusoids, and there is a reconstruction step later
            self.base_probe = PrismProbe(xs, ys, self.aperture, self.voltage_eV, device=self.device, nkx=self.prism, kth=self.kth)
        else:
            # OR, we'll propagate our series of real-space probes.
            # need to make sure they're on the correct device, and defer_shifts=True means the calculator controls when to expand the probe cube (see loop_probes)
            self.base_probe = Probe(xs, ys, self.aperture, self.voltage_eV, device=self.device, probe_xs=self.probe_xs, probe_ys=self.probe_ys, probe_positions=self.probe_positions,cropping=self.probe_cropping, defer_shifts=True)

        if not self.loop_probes:
            self.base_probe.applyShifts()

        # Initialize storage for results
        self.n_frames = trajectory.n_frames

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

        # cache key is calculated TWICE: once during setup (so the user only needs to setup to infer where their cache folder will go), and again during run (just in case the user does something funky)
        # Generate cache key and setup output directory
        self.cache_key = self._generate_cache_key(self.trajectory, self.aperture, self.voltage_eV,
                                           self.slice_thickness, self.sampling, self.probe_positions,
                                           self.base_probe.spatial_decoherence, self.base_probe.temporal_decoherence,
                                           self.base_probe._array)
        self.output_dir = Path("psi_data/" + ("torch" if TORCH_AVAILABLE else "numpy") + "_"+self.cache_key)

    def preview_probes(self):
        positions = self.trajectory.positions[0]
        atom_types = self.trajectory.atom_types
        atom_type_names = []
        for atom_type in atom_types:
            if atom_type in self.element_map:
                atom_type_names.append(self.element_map[atom_type])
            else:
                atom_type_names.append(atom_type)
        potential = Potential(self.xs, self.ys, self.zs, positions, atom_type_names, kind="kirkland", device=self.device, slice_axis=self.slice_axis)
        potential.build()
        potential.flatten()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = np.absolute(to_cpu(potential.array))[:,::-1,0].T # imshow convention: y,x. our convention: x,y, and flip y (0,0 upper-left)
        xs = to_cpu(potential.xs) ; ys = to_cpu(potential.ys)
        extent = (np.amin(xs),np.amax(xs),np.amin(ys),np.amax(ys))
        ax.imshow(array, cmap="inferno", extent=extent)
        ax.set_xlabel("x ($\\AA$)") ; ax.set_ylabel("y ($\\AA$)")
        pp = np.asarray(self.base_probe.probe_positions)
        ax.scatter(pp[:,0],pp[:,1],c='r')
        plt.show()

    #@profile
    def run(self) -> WFData:

        # cache key is calculated TWICE: once during setup (so the user only needs to setup to infer where their cache folder will go), and again during run (just in case the user does something funky)
        # Generate cache key and setup output directory
        cache_key = self._generate_cache_key(self.trajectory, self.aperture, self.voltage_eV,
                                           self.slice_thickness, self.sampling, self.probe_positions,
                                           self.base_probe.spatial_decoherence, self.base_probe.temporal_decoherence,
                                           self.base_probe._array)
        if self.cache_key != cache_key:
            self.cache_key = cache_key
        self.output_dir = Path("psi_data/" + ("torch" if TORCH_AVAILABLE else "numpy") + "_"+cache_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve which layers to store ────────────────────────────────────
        # NEW: if cache_layer_indices is set, only those layers are FFT'd and
        # written to disk; the propagation itself still runs through all nz
        # slices (physically required).  cache_layer_indices=None keeps the
        # original behaviour of storing every layer.
        if "slices" in self.cache_levels and self.cache_layer_indices is not None:
            # Validate and clip indices to [0, nz-1]
            _requested = sorted(set(int(i) for i in self.cache_layer_indices))
            _dropped   = [i for i in _requested if not (0 <= i < self.nz)]
            _active_layers = [i for i in _requested if 0 <= i < self.nz]
            if _dropped:
                logger.warning(
                    f"cache_layer_indices: dropped out-of-range indices {_dropped} "
                    f"(nz={self.nz})"
                )
            if not _active_layers:
                raise ValueError(
                    "cache_layer_indices produced no valid layer indices after "
                    f"clipping to [0, {self.nz-1}]."
                )
            logger.info(
                f"Selective layer storage: recording {len(_active_layers)}/{self.nz} "
                f"layers -> {_active_layers}"
            )
            print(
                f"[MultisliceCalculator] cache_layer_indices: storing "
                f"{len(_active_layers)}/{self.nz} layers: {_active_layers}",
                flush=True
            )
        else:
            # Default: store every layer (original behaviour)
            _active_layers = list(range(self.nz)) if "slices" in self.cache_levels else [0]

        self._active_layers = _active_layers  # expose for inspection / post-processing
        # ─────────────────────────────────────────────────────────────────────

        # if probes are over vacuum (e.g. nanoparticles), we don't need to propagate them?
        self.probe_indices = xp.arange(len(self.probe_positions))
        if self.skip_vacuum and len(self.probe_positions)>1 and self.aperture>1 and self.min_dk:
            if os.path.exists(self.output_dir / f"probe_indices.npy"):
                self.probe_indices = np.load(self.output_dir / f"probe_indices.npy")
            else:
                xy_atoms = asarray(self.trajectory.positions[0,:,:2])
                self.probe_indices = []
                for i,p in enumerate(tqdm(self.probe_positions)):
                    p = asarray(p)
                    d_to_nearest_atom = xp.sqrt( xp.amin( xp.sum( (p[None,:]-xy_atoms)**2,axis=1) ) )
                    if d_to_nearest_atom < self.probe_cropping*self.sampling:
                        self.probe_indices.append(i)
                np.save(self.output_dir / f"probe_indices.npy", self.probe_indices)
            self.probe_indices = asarray(self.probe_indices)
            print("filtered to",len(self.probe_indices),"probe positions")


        nc,npt,nx,ny = self.base_probe._array.shape
        self.n_probes = nc*len(self.probe_positions)

        # Storage: [probe, frame, x, y, layer] - matches WFData expected format
        # CHANGED: n_layers is now len(_active_layers) instead of always self.nz
        self.n_layers = len(_active_layers)

        if self.store_full:
            fd_nx = self.nx ; fd_ny = self.ny ; fd_npt = self.n_probes
            if self.use_memmap:
                self.wavefunction_data = memmap((fd_npt, self.n_frames, fd_nx, fd_ny, self.n_layers),
                                                   dtype=self.complex_dtype, filename = self.output_dir / "wdf_memmap.npy" )
            else:
                self.wavefunction_data = zeros((fd_npt, self.n_frames, fd_nx, fd_ny, self.n_layers),
                                                   dtype=self.complex_dtype, device=self.device)

        # Process frames with caching and multiprocessing
        total_start_time = time.time()
        frames_computed = 0
        frames_cached = 0

        # quality of life sanity checks: user may have set things (e.g. probe array) with the wrong data type (e.g. numpy instead of tensor). let's try to catch and correct those here
        if isinstance(self.base_probe._array,np.ndarray) and TORCH_AVAILABLE:
            self.base_probe._array = xp.tensor(self.base_probe._array)

        if self.ADF: # create a dummy HAADFData object, first so we can hijack its getMask function, and later we'll load it up
            kwargs = {}
            if not isinstance(self.ADF,bool):
                kwargs["inner_mrad"],kwargs["outer_mrad"] = self.ADF
            from ..postprocessing.haadf_data import HAADFData
            array = zeros((self.n_probes,1,1,1,1),dtype=self.complex_dtype)
            array += xp.arange(self.n_probes)[:,None,None,None,None] # we'll use this as an index to map nth probe to the ADF grid coordinates i,j
            wf = WFData(probe_positions=self.probe_positions,probe_xs=self.probe_xs,probe_ys=self.probe_ys,
                time=None,kxs=self.kxs[self.keep_kxs_indices],kys=self.kys[self.keep_kys_indices],xs=self.xs,ys=self.ys,
                layer=None,array=array,probe=self.base_probe,cache_dir=self.output_dir)
            self.ADF = HAADFData(wf)
            self.ADFmask = absolute(self.ADF.getMask(**kwargs)) # HAADFData infers mask dtype from _wf_array dtype, but we'll absolute^2 later
            self.ADFindex = astype(absolute(self.ADF._wf_array[0,:,:,0,0,0,0]),int)
            self.ADF._array = zeros(self.ADFindex.shape,dtype=self.complex_dtype)

        # Process frames one at a time with tqdm progress tracking
        with tqdm(total=self.n_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx in range(self.n_frames):
                cache_file = self.output_dir / f"frame_{frame_idx}.npy"
                # Show detailed progress for single-frame runs
                show_progress = (frame_idx == 0 and self.n_frames == 1 and not self.loop_probes)

                # special case: no frames cached, but we clearly finished and got to tacaw. if so, don't bother regenerating
                # this allows cache_levels = [] to be used for disk space savings
                if os.path.exists( self.output_dir / f"tacaw.npy" ) and not os.path.exists( cache_file ):
                    pbar.update(1)
                    continue

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
                if cache_exists and not self.prism and self.ADF:
                    intensities = einsum('pxyln,xy->p',absolute(frame_data)**2,self.ADFmask)
                    self.ADF._array += intensities[self.ADFindex]

                if not os.path.exists(self.output_dir / f"kx.npy"):
                    np.save(self.output_dir / f"kx.npy",to_cpu(self.kxs[self.keep_kxs_indices]))
                    np.save(self.output_dir / f"ky.npy",to_cpu(self.kys[self.keep_kys_indices]))
                if len(self.kxs)!=self.nx and not os.path.exists(self.output_dir / f"kx_uncrop.npy"):
                    np.save(self.output_dir / f"kx_uncrop.npy",to_cpu(self.kxs))
                if len(self.kys)!=self.ny and not os.path.exists(self.output_dir / f"ky_uncrop.npy"):
                    np.save(self.output_dir / f"ky_uncrop.npy",to_cpu(self.kys))

                if cache_exists:
                    frames_cached += 1
                else:
                    potential = Potential(self.xs, self.ys, self.zs, positions, atom_type_names, kind="kirkland", device=self.device, slice_axis=self.slice_axis, progress=show_progress, cache_dir=cache_file.parent if "potentials" in self.cache_levels else None, frame_idx = frame_idx)
                    nc,npt,nx,ny = self.base_probe._array.shape ; npt = len(self.base_probe.probe_positions)
                    n_slices = len(self.zs)
                    n_waves = len(self.base_probe.probe_positions)

                    # frame_data is always: p,x,y,l,1 (self.wavefunction_data expects p,t,x,y,l, since we loop time. recall Propagate gave l,p,x,y)
                    # CHANGED: last dim is self.n_layers = len(_active_layers), not nz
                    if self.store_full or self.prism:
                        fd_nx = self.nx ; fd_ny = self.ny ; fd_npt = self.n_probes
                        if self.use_memmap:
                            frame_data = memmap((n_waves, fd_nx, fd_ny, self.n_layers,1), dtype=self.complex_dtype, filename = cache_file )
                        else:
                            frame_data = zeros((n_waves, fd_nx, fd_ny, self.n_layers,1), dtype=self.complex_dtype, device=self.device)

                    chunks = []
                    if self.loop_probes:
                        chunksize = self.loop_probes if isinstance(self.loop_probes,int) else 1
                        for i in range(10000000):
                            chunk = xp.arange(i*chunksize,(i+1)*chunksize)
                            chunk = chunk[xp.any(self.probe_indices[None,:]==chunk[:,None],axis=1)]
                            if (i+1)*chunksize>npt:
                                break
                            if len(chunk)==0:
                                continue
                            chunks.append(chunk)
                        pbar2 = tqdm(total = npt, desc = "looping probes", unit="probe")
                    else:
                        chunks.append( xp.arange(npt) )
                        pbar2 = None

                    for selected in chunks:
                        if len(selected)==npt:
                            probe = self.base_probe
                        else:
                            probe = self.base_probe.copy(selected_probes=selected)
                        probe.applyShifts()
                        # Propagate returns shape (nz, n_probes, nx, ny) when store_all_slices=True
                        exit_waves_single = Propagate(probe, potential, self.device, progress=show_progress, onthefly=True, store_all_slices = ("slices" in self.cache_levels) ) # [l],p,x,y indices

                        # expand out to fixed l,p,x,y indices
                        exit_waves_single = expand_dims(exit_waves_single,0) if len(exit_waves_single.shape)==3 else exit_waves_single

                        # FFT and load into frame_data
                        kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}

                        # CHANGED: iterate over (out_idx, real_layer_idx) pairs instead of
                        # range(self.n_layers).  When cache_layer_indices=None, _active_layers
                        # is range(nz) so out_idx == real_layer_idx and behaviour is identical
                        # to the original code.
                        for out_idx, real_layer_idx in enumerate(_active_layers):
                            exit_waves_k = xp.fft.fft2(exit_waves_single[real_layer_idx,:,:,:], **kwarg) # l,p,x,y --> p,x,y
                            diffraction_patterns = xp.fft.fftshift(exit_waves_k, **kwarg)
                            diffraction_patterns = diffraction_patterns[:,self.keep_kxs_indices,:][:,:,self.keep_kys_indices]*self.kth**2
                            if self.use_memmap:
                                diffraction_patterns = to_cpu(diffraction_patterns)
                                selected = to_cpu(selected)
                            if self.store_full or self.prism:
                                # CHANGED: write to compact slot out_idx, not real_layer_idx
                                frame_data[selected,:,:,out_idx,0] = diffraction_patterns # load p,x,y --> p,x,y,l,1 indices
                            if self.ADF and not self.prism:
                                intensities = einsum('pxy,xy->p',absolute(diffraction_patterns[:,:,:])**2,self.ADFmask)
                                for i,pp in zip(intensities,selected):
                                    self.ADF._array[self.ADFindex==pp] += i

                        if pbar2 is not None:
                            pbar2.update(int(max(selected))-pbar2.n)

                    if not self.use_memmap and ( "exitwaves" in self.cache_levels or "slices" in self.cache_levels ) and (self.store_full or self.prism):
                        # Convert to CPU numpy array for saving
                        frame_data_cpu = to_cpu(frame_data)
                        np.save(cache_file, frame_data_cpu)
                    frames_computed += 1

                if self.store_full or self.prism:
                    cropped = frame_data[:,:,:,:,0]

                if self.prism:
                    # Recall: Prism algorithm passes a series of sinusoids through the sample (fourier components shared by all real-space probes), so now for each real-space probe, we need to calculate the exitwaves from components
                    kwarg ={}
                    if self.ADF:
                        kwarg["ADF"]=(self.ADF,self.ADFmask,self.ADFindex)
                    if self.store_full:
                        kwarg["load_into"]=self.wavefunction_data[:,frame_idx,:,:,0]
                    self.base_probe.calculateProbesFromS(frame_data,self.probe_positions,**kwarg,chunksize=self.loop_probes)
                elif self.store_full:
                    if self.use_memmap:
                        cropped = to_cpu(cropped)
                    self.wavefunction_data[:, frame_idx, :, :, :] = cropped # load p,x,y,l,1 --> p,t,x,y,l indices
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
        
        time_array = np.arange(self.n_frames) * self.trajectory.timestep  # Time array in ps

        # CHANGED: layer_array now reflects the actual stored layer indices.
        # When cache_layer_indices=None, _active_layers == list(range(nz)) so
        # layer_array == np.arange(nz), identical to the original behaviour.
        layer_array = np.array(_active_layers) if "slices" in self.cache_levels else np.array([0])
        
        # Package results
        array = zeros((self.n_probes,1,1,1,1),dtype=self.complex_dtype)
        if self.store_full:
            array = self.wavefunction_data
        wf_data = WFData(
            probe_positions=self.probe_positions,
            probe_xs=self.probe_xs,
            probe_ys=self.probe_ys,
            time=time_array,
            kxs=self.kxs[self.keep_kxs_indices],
            kys=self.kys[self.keep_kys_indices],
            xs=self.xs,
            ys=self.ys,
            layer=layer_array,
            array=array,
            probe=self.base_probe,
            cache_dir=self.output_dir
        )
        
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
        
        if self.ADF:
            self.ADF._array /= self.n_frames # haadf_data divides by nc,nt,nl (from _wf_array's c,x,y,t,kx,ky,l)
            return wf_data,self.ADF

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

    def plot(self,w,filename=None):
        import matplotlib.pyplot as plt

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

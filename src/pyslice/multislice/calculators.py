import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List
from tqdm import tqdm
import time, os
import hashlib

from .potentials import grid_from_trajectory, Potential
from .multislice import Probe, PrismProbe, Propagate, create_batched_probes
from .trajectory import Trajectory
from ..postprocessing.wf_data import WFData
from .sed import SED
from pyslice.backend import make_backend, to_numpy, NumpyBackend, source_files_version

logger = logging.getLogger(__name__)

class MultisliceCalculator:
    """Configure and run multislice electron-scattering simulations.

    The calculator owns the simulation geometry, probe construction, optional
    HAADF accumulation, and wavefunction frame cache. ``return_layers``
    controls which propagated wavefunction layers are returned, while
    ``cache_wavefunctions`` and ``cache_potentials`` control disk caching.
    """

    def __init__(self, device=None, force_cpu=False):
        """
        Initialize the multislice calculator.

        Args:
            device: Device string ('cpu', 'cuda', 'mps', or None for auto-detection)
            force_cpu: Force CPU usage even if GPU is available
        """
        self.force_cpu = force_cpu
        if force_cpu:
            self._backend = make_backend('cpu')
        else:
            self._backend = make_backend(device)
        self.device = self._backend.device

        logger.info(f"Calculator initialized on device: {self.device}")

        # Element mapping for display purposes
        self.element_map = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
            30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr'
        }

    # Derived automatically from the sources whose logic determines the cached
    # wavefunction VALUES, so any change to propagation / potential / probe /
    # backend code changes the key and stale psi_data is not silently reused.
    # The "v3" prefix allows a manual bump for reasons outside these files.
    _CACHE_VERSION = "v3-" + source_files_version([
        os.path.join(os.path.dirname(__file__), "multislice.py"),
        os.path.join(os.path.dirname(__file__), "potentials.py"),
        os.path.join(os.path.dirname(__file__), "calculators.py"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend.py"),
    ])

    def _generate_cache_key(self, trajectory, aperture, voltage_eV,
                            slice_thickness, sampling, probe_positions,
                            spatial_decoherence, temporal_decoherence,
                            probe_array=None, stored_layer_indices=None):
        """Generate a short hash for parameters that affect wavefunction output.

        The trajectory is hashed in full (every frame, atom and component)
        rather than sampled from frame 0, so runs that differ only in later
        frames or in y/z — e.g. a temperature/seed sweep from the same initial
        structure — get distinct caches instead of silently sharing one.
        """
        def _hash_array(a):
            return hashlib.md5(
                np.ascontiguousarray(to_numpy(a)).tobytes()).hexdigest()

        params = {
            'cache_version': self._CACHE_VERSION,
            'traj_positions': _hash_array(trajectory.positions),
            'n_frames': trajectory.n_frames,
            'n_atoms': trajectory.n_atoms,
            'box_matrix': np.asarray(trajectory.box_matrix).tolist(),
            'atom_types': np.asarray(trajectory.atom_types).tolist(),
            'aperture': aperture,
            'voltage_eV': voltage_eV,
            'slice_thickness': slice_thickness,
            'sampling': sampling,
            'probe_positions': np.asarray(probe_positions).tolist(),
            'kth': self.kth,
            'max_kx': self.max_kx,
            'max_ky': self.max_ky,
            'min_dk': self.min_dk,
            'prism': self.prism,
            'slice_axis': self.slice_axis,
            'backend': 'torch' if not isinstance(self._backend, NumpyBackend) else 'numpy',
        }
        if stored_layer_indices is not None:
            params['stored_layer_indices'] = tuple(stored_layer_indices)
        if spatial_decoherence is not None:
            params['spatial_decoherence'] = spatial_decoherence
        if temporal_decoherence is not None:
            params['temporal_decoherence'] = temporal_decoherence
        if probe_array is not None:
            # Hashing the full probe array captures defocus, aberrations,
            # decoherence and the aperture/crop geometry baked into the probe.
            params['probe_hash'] = _hash_array(probe_array)
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:12]

    def _resolve_return_layers(self):
        """Return normalized wavefunction layer indices for the current setup."""
        if self.return_layers is None:
            return []
        if isinstance(self.return_layers, str):
            if self.return_layers == "all":
                return list(range(self.nz))
            raise ValueError(
                "return_layers must be None, -1, 'all', or a list of layer indices"
            )

        if isinstance(self.return_layers, (int, np.integer)):
            requested_layers = [int(self.return_layers)]
        else:
            requested_layers = [int(i) for i in self.return_layers]

        return_layers = sorted(set(i + self.nz if i < 0 else i for i in requested_layers))
        invalid_layers = [i for i in return_layers if i < 0 or i >= self.nz]
        if invalid_layers:
            raise ValueError(
                f"return_layers contains out-of-range layers {invalid_layers}; "
                f"valid range is [0, {self.nz - 1}]"
            )
        return return_layers

    def _stored_layers_for_return(self, return_layers):
        """Return layers that must be computed or cached for this run."""
        return list(return_layers) or [self.nz - 1]

    def _cache_key_stored_layers(self, stored_layers):
        """Return layer selection for cache-key partitioning, if needed."""
        if list(stored_layers) in ([self.nz - 1], list(range(self.nz))):
            return None
        return tuple(stored_layers)

    def _stores_exit_wave_only(self, stored_layers):
        """Return whether the stored layers are equivalent to exit-wave output."""
        return list(stored_layers) == [self.nz - 1]

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
        return_layers=-1,
        cache_wavefunctions: bool = True,
        cache_potentials: bool = False,
        max_kx=np.inf,
        max_ky=np.inf,
        use_memmap=False,
        loop_probes=False,
        min_dk=0,
        prism=False,
        kth=1,
        ADF=False,
        skip_vacuum=False,
        **kwargs,
    ):
        """
        Set up multislice simulation.

        Args:
            trajectory: Input trajectory data
            aperture: Objective aperture semi-angle in mrad
            voltage_eV: Accelerating voltage in eV
            defocus: Defocus in Angstroms applied to the probe before propagation
            slice_thickness: Thickness of each slice in Angstroms
            sampling: Sampling rate in Angstroms per pixel
            probe_positions: List of (x,y) probe positions in Angstroms
            batch_size: Number of frames to process at once
            save_path: Optional path to save wave function data
            cleanup_temp_files: Whether to delete temp files after loading
            return_layers: Wavefunction layers to include in the returned
                ``WFData``. ``-1`` (default) returns the exit wave, ``"all"``
                returns every layer, and a list such as ``[43, 87, 175]`` returns
                selected layer wavefunctions. Negative indices are resolved
                against the layer stack. ``None`` or ``[]`` suppresses returned
                wavefunction data while still allowing HAADF calculation and
                optional exit-wave caching.
            cache_wavefunctions: Whether to read/write per-frame wavefunction cache files
            cache_potentials: Whether to read/write potential-slice cache data
            skip_vacuum: Skip probe positions that are far from atoms when using probe cropping.
        """
        if kwargs:
            old_api_kwargs = {
                "cache_levels",
                "cache_layer_indices",
                "store_full",
                "output",
                "output_slice_indices",
                "cache",
                "keep_wavefunctions",
            }
            old_kwargs_used = sorted(old_api_kwargs.intersection(kwargs))
            if old_kwargs_used:
                raise TypeError(
                    "MultisliceCalculator.setup() cache/return API has changed. "
                    f"Unsupported old argument(s): {', '.join(old_kwargs_used)}.\n"
                    "Use return_layers=-1 for exit waves, return_layers='all' for all "
                    "layers, or return_layers=[...] for selected layer wavefunctions.\n"
                    "Use cache_wavefunctions=True/False for wavefunction frame caching, "
                    "cache_potentials=True/False for potential-slice caching, and "
                    "return_layers=None or return_layers=[] for HAADF-only runs that "
                    "do not need returned raw wavefunctions.\n"
                    "Old arguments were not applied."
                )
            unexpected = next(iter(kwargs))
            raise TypeError(
                "MultisliceCalculator.setup() got an unexpected keyword argument "
                f"'{unexpected}'"
            )

        b = self._backend

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
        if slice_axis != 2:
            # Propagation is hard-coded to the z axis; any other slice_axis
            # silently produces wrong results (see Potential). Fail early with
            # guidance rather than after a full run.
            raise NotImplementedError(
                "slice_axis != 2 is not supported (it would silently produce "
                "wrong results). Permute your trajectory so the beam direction "
                "is the z axis and use slice_axis=2."
            )
        self.slice_axis = slice_axis
        self.return_layers = return_layers
        self.cache_wavefunctions = cache_wavefunctions
        self.cache_potentials = cache_potentials
        self.max_kx = max_kx
        self.max_ky = max_ky
        self.use_memmap = use_memmap   # bool: frame_data (p,x,y,l,1) and wavefunction_data (p,t,x,y,l) will be memmapped instead of held in RAM
        self.loop_probes = loop_probes # False or int: multiple probes (p,x,y) can be propagated simultaneously. this allows processing in chunks
        self.min_dk = min_dk           # float: Δk=1/L, so this will pre-crop each probe and potential slice so a smaller area is propagated
        self.prism = prism             # False or int: PRISM algorithm implementation, this denotes how many fourier components are used in kx ky
        self.kth = kth                 # int: Δk=1/L, nk = nx. huge systems waste RAM with ultra-fine Δk. this sparsifies the exitwaves via ::kth
        self.ADF = ADF                 # bool or (inner,outer): allows on-the-fly calculation of the ADF signal
        self.skip_vacuum = skip_vacuum # bool: if True, we skip propagation of probes in locations where there are no atoms

        # Set up spatial grids
        xs, ys, zs, lx, ly, lz = grid_from_trajectory(trajectory, sampling=sampling, slice_thickness=slice_thickness)
        nx = len(xs); ny = len(ys); nz = len(zs)
        self.xs = xs; self.ys = ys; self.zs = zs
        self.lx = lx; self.ly = ly; self.lz = lz
        self.nx = nx; self.ny = ny; self.nz = nz
        self.dx = xs[1]-xs[0]; self.dy = ys[1]-ys[0]
        self._return_layers = self._resolve_return_layers()
        self._stored_layers = self._stored_layers_for_return(self._return_layers)
        self.returns_wavefunctions = bool(self._return_layers)

        self.probe_cropping = 0
        if self.min_dk > 0:  # dk = 1/L = 1/(nx*sampling)
            nx = int(np.round(1/(self.min_dk*self.sampling)))
            self.nx = nx; self.ny = nx      # Q: check this for non square super cells
            self.probe_cropping = nx

        self.kxs = b.fftshift(b.fftfreq(self.nx, self.sampling))  # k-space in 1/Å
        self.kys = b.fftshift(b.fftfreq(self.ny, self.sampling))  # k-space in 1/Å
        kx_mask = b.zeros(self.nx)+1; ky_mask = b.zeros(self.ny)+1
        kx_mask[self.kxs < -max_kx] = 0; kx_mask[self.kxs > max_kx] = 0
        ky_mask[self.kys < -max_ky] = 0; ky_mask[self.kys > max_ky] = 0
        self.keep_kxs_indices = b.arange(self.nx)[kx_mask==1][::self.kth]
        self.keep_kys_indices = b.arange(self.ny)[ky_mask==1][::self.kth]
        self.nx = len(self.keep_kxs_indices); self.ny = len(self.keep_kys_indices)

        # Preferred to pass probe_xs and probe_ys from which we will define a grid
        if self.probe_xs is not None and self.probe_ys is not None:
            if self.probe_positions is not None:
                logger.warning(
                    "Both probe_xs/probe_ys and probe_positions were supplied; "
                    "probe_positions is ignored in favour of the "
                    "probe_xs x probe_ys grid."
                )
            x, y = np.meshgrid(self.probe_xs, self.probe_ys)
            self.probe_positions = np.reshape([x, y], (2, len(x.flat))).T  # x,y looped indices to match what multislice.Probe does

        # If probe_positions provided but not probe_xs/probe_ys, derive the scan
        # coordinates.  probe_xs/probe_ys are the unique coordinates used to
        # reshape the flat probe axis into a 2D image (WFData.reshaped / HAADF),
        # which assumes the meshgrid flattening order (x fastest, y outer).
        elif self.probe_positions is not None:
            positions = np.asarray(self.probe_positions, dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError(
                    "probe_positions must be a sequence of (x, y) pairs with "
                    f"shape (N, 2); got array of shape {positions.shape}."
                )
            self.probe_xs = sorted(list(set(positions[:, 0])))
            self.probe_ys = sorted(list(set(positions[:, 1])))
            gx, gy = np.meshgrid(self.probe_xs, self.probe_ys)
            grid = np.reshape([gx, gy], (2, gx.size)).T
            pos_set = {(round(px, 6), round(py, 6)) for px, py in positions}
            grid_set = {(round(px, 6), round(py, 6)) for px, py in grid}
            if len(positions) == len(grid) and pos_set == grid_set:
                # The points tile a full rectangular grid: canonicalise their
                # order to the meshgrid flattening so the 2D image maps correctly
                # regardless of the order they were passed in (e.g. a nested
                # [(x, y) for x in xs for y in ys] loop).
                self.probe_positions = grid
            else:
                # Arbitrary point set (e.g. site-resolved TACAW on selected
                # columns): simulate exactly as given.  Per-probe spectra are
                # correct, but 2D image reshaping cannot apply.
                self.probe_positions = positions
                logger.warning(
                    "probe_positions (%d points) do not form a full %d x %d "
                    "grid; per-probe spectra are correct but image reshaping "
                    "(HAADFData/spectrum_image) will not apply.",
                    len(positions), len(self.probe_xs), len(self.probe_ys),
                )

        # Set up default probe position if not provided
        if self.probe_positions is None:
            self.probe_positions = [(lx/2, ly/2)]  # Center probe
            self.probe_xs = [lx/2]; self.probe_ys = [ly/2]

        if self.prism:
            # Prism algorithm works by passing a series of sinusoids (fourier components shared by all probes) through the sample. "PrismProbe" will therefore give us a series of sinusoids, and there is a reconstruction step later
            self.base_probe = PrismProbe(xs, ys, self.aperture, self.voltage_eV, backend=b, nkx=self.prism, kth=self.kth)
        else:
            # OR, we'll propagate our series of real-space probes.
            # need to make sure they're on the correct device, and defer_shifts=True means the calculator controls when to expand the probe cube (see loop_probes)
            # Pass the canonical probe_positions directly (already meshed in the
            # probe_xs/probe_ys branch above).  Passing probe_xs/probe_ys here
            # would make Probe rebuild an outer-product grid, so an explicit
            # position list would be silently replaced by that grid and desync
            # n_probes from the actually-simulated probes (shape-mismatch crash).
            self.base_probe = Probe(xs, ys, self.aperture, self.voltage_eV, backend=b, probe_positions=self.probe_positions, cropping=self.probe_cropping, defer_shifts=True)

        defocus_values = to_numpy(self.defocus)
        if np.ndim(defocus_values) != 0:
            raise TypeError("MultisliceCalculator.setup(defocus=...) expects a scalar Angstrom value.")

        defocus_value = float(defocus_values)
        if defocus_value != 0:
            if self.prism:
                raise NotImplementedError(
                    "setup(defocus=...) is only wired for the real-space Probe path; "
                    "PRISM defocus needs an explicit reconstruction-path implementation."
                )
            self.base_probe.defocus(defocus_value)

        if not self.loop_probes:
            self.base_probe.applyShifts()

        # Initialize storage for results
        self.n_frames = trajectory.n_frames

        self.float_dtype = b.float_dtype
        self.complex_dtype = b.complex_dtype

        # cache key is calculated TWICE: once during setup (so the user only needs to setup to infer where their cache folder will go), and again during run (just in case the user does something funky)
        # Generate cache key and setup output directory
        self.cache_key = self._generate_cache_key(
            self.trajectory, self.aperture, self.voltage_eV,
            self.slice_thickness, self.sampling, self.probe_positions,
            self.base_probe.spatial_decoherence, self.base_probe.temporal_decoherence,
            self.base_probe._array,
            self._cache_key_stored_layers(self._stored_layers),
        )
        self.output_dir = Path("psi_data/" + ("torch" if not isinstance(b, NumpyBackend) else "numpy") + "_"+self.cache_key)

    def preview_probes(self):
        b = self._backend
        positions = self.trajectory.positions[0]
        atom_types = self.trajectory.atom_types
        atom_type_names = []
        for atom_type in atom_types:
            if atom_type in self.element_map:
                atom_type_names.append(self.element_map[atom_type])
            else:
                atom_type_names.append(atom_type)
        potential = Potential(self.xs, self.ys, self.zs, positions, atom_type_names, backend=b, kind="kirkland", slice_axis=self.slice_axis)
        potential.build()
        potential.flatten()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = np.absolute(to_numpy(potential.array))[:, ::-1, 0].T  # imshow convention: y,x. our convention: x,y, and flip y (0,0 upper-left)
        xs = to_numpy(potential.xs); ys = to_numpy(potential.ys)
        extent = (np.amin(xs), np.amax(xs), np.amin(ys), np.amax(ys))
        ax.imshow(array, cmap="inferno", extent=extent)
        ax.set_xlabel("x ($\\AA$)"); ax.set_ylabel("y ($\\AA$)")
        pp = np.asarray(self.base_probe.probe_positions)
        ax.scatter(pp[:, 0], pp[:, 1], c='r')
        plt.show()

    #@profile
    def run(self, force_rerun: bool = False) -> WFData:
        """Run propagation and return ``WFData`` or ``(WFData, HAADFData)``.

        Returns:
            ``WFData`` for normal wavefunction-output runs. If ``ADF`` was set
            during setup, returns ``(wf_data, haadf_data)`` with HAADF
            accumulated on the fly.
        """
        b = self._backend

        # cache key is calculated TWICE: once during setup (so the user only needs to setup to infer where their cache folder will go), and again during run (just in case the user does something funky)
        # Generate cache key and setup output directory
        _return_layers = self._resolve_return_layers()
        _stored_layers = self._stored_layers_for_return(_return_layers)
        self._return_layers = _return_layers
        self._stored_layers = _stored_layers
        self.returns_wavefunctions = bool(_return_layers)
        cache_key = self._generate_cache_key(self.trajectory, self.aperture, self.voltage_eV,
                                             self.slice_thickness, self.sampling, self.probe_positions,
                                             self.base_probe.spatial_decoherence, self.base_probe.temporal_decoherence,
                                             self.base_probe._array,
                                             self._cache_key_stored_layers(_stored_layers))
        if self.cache_key != cache_key:
            self.cache_key = cache_key
        self.output_dir = Path("psi_data/" + ("torch" if b.xp is not np else "numpy") + "_"+cache_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # if probes are over vacuum (e.g. nanoparticles), we don't need to propagate them?
        self.probe_indices = b.arange(len(self.probe_positions))
        if self.skip_vacuum and len(self.probe_positions) > 1 and self.aperture > 1 and self.min_dk:
            if os.path.exists(self.output_dir / f"probe_indices.npy"):
                self.probe_indices = np.load(self.output_dir / f"probe_indices.npy")
            else:
                xy_atoms = b.asarray(self.trajectory.positions[0, :, :2])
                self.probe_indices = []
                for i, p in enumerate(tqdm(self.probe_positions)):
                    p = b.asarray(p)
                    d_to_nearest_atom = b.sqrt(b.amin(b.sum((p[None, :]-xy_atoms)**2, axis=1)))
                    if d_to_nearest_atom < self.probe_cropping*self.sampling:
                        self.probe_indices.append(i)
                np.save(self.output_dir / f"probe_indices.npy", self.probe_indices)
            self.probe_indices = b.asarray(self.probe_indices, dtype=int)
            print("filtered to", len(self.probe_indices), "probe positions")

        nc, npt, nx, ny = self.base_probe._array.shape
        self.n_probes = nc*len(self.probe_positions)
        # Storage: [probe, frame, x, y, layer] - matches WFData expected format
        self.n_layers = len(_stored_layers)
        stores_exit_wave_only = self._stores_exit_wave_only(_stored_layers)
        if self.returns_wavefunctions:
            fd_nx = self.nx; fd_ny = self.ny; fd_npt = self.n_probes
            if self.use_memmap:
                self.wavefunction_data = b.memmap((fd_npt, self.n_frames, fd_nx, fd_ny, self.n_layers),
                                                   dtype=self.complex_dtype, filename=self.output_dir / "wdf_memmap.npy")
            else:
                self.wavefunction_data = b.zeros((fd_npt, self.n_frames, fd_nx, fd_ny, self.n_layers),
                                                  dtype=self.complex_dtype)

        # Process frames with caching and multiprocessing
        total_start_time = time.time()
        frames_computed = 0
        frames_cached = 0

        # quality of life sanity checks: user may have set things (e.g. probe array) with the wrong data type (e.g. numpy instead of tensor). let's try to catch and correct those here
        if isinstance(self.base_probe._array, np.ndarray) and not isinstance(b, NumpyBackend):
            self.base_probe._array = b.asarray(self.base_probe._array)

        if self.ADF:  # create a dummy HAADFData object, first so we can hijack its getMask function, and later we'll load it up
            kwargs = {}
            if not isinstance(self.ADF, bool):
                kwargs["inner_mrad"], kwargs["outer_mrad"] = self.ADF
            from ..postprocessing.haadf_data import HAADFData
            array = b.zeros((self.n_probes, 1, 1, 1, 1), dtype=self.complex_dtype)
            array += b.arange(self.n_probes)[:, None, None, None, None]  # we'll use this as an index to map nth probe to the ADF grid coordinates i,j
            wf = WFData(probe_positions=self.probe_positions, probe_xs=self.probe_xs, probe_ys=self.probe_ys,
                        time=None, kxs=self.kxs[self.keep_kxs_indices], kys=self.kys[self.keep_kys_indices], xs=self.xs, ys=self.ys,
                        layer=None, array=array, probe=self.base_probe, backend=b, cache_dir=self.output_dir)
            self.ADF = HAADFData(wf)
            self.ADFmask = b.absolute(self.ADF.getMask(**kwargs))  # HAADFData infers mask dtype from _wf_array dtype, but we'll absolute^2 later
            self.ADFindex = b.astype(b.absolute(self.ADF._wf_array[0, :, :, 0, 0, 0, 0]), int)
            # One ADF image per stored layer (thickness). Collapsed to a plain 2D
            # image below when only one thickness is stored (the default).
            self.ADF._array = b.zeros((self.n_layers,) + tuple(self.ADFindex.shape),
                                      dtype=self.complex_dtype)

        # If tacaw.npy already exists and no per-frame cache will be written,
        # there is nothing to reload or recompute. Pass force_rerun=True to
        # override after changing probe parameters.
        skip_all_frames = (
            not force_rerun
            and os.path.exists(self.output_dir / "tacaw.npy")
            and not self.cache_wavefunctions
            and not self.returns_wavefunctions
            and not self.ADF
        )
        if skip_all_frames:
            logger.info("tacaw.npy found and no per-frame cache levels active; skipping multislice computation. Pass force_rerun=True to recompute.")

        # Process frames one at a time with tqdm progress tracking
        if not skip_all_frames:
          with tqdm(total=self.n_frames, desc="Processing frames", unit="frame") as pbar:
            for frame_idx in range(self.n_frames):
                cache_file = self.output_dir / f"frame_{frame_idx}.npy"
                # Show detailed progress for single-frame runs
                show_progress = (frame_idx == 0 and self.n_frames == 1 and not self.loop_probes)

                positions = self.trajectory.positions[frame_idx]
                atom_types = self.trajectory.atom_types
                atom_type_names = []
                for atom_type in atom_types:
                    if atom_type in self.element_map:
                        atom_type_names.append(self.element_map[atom_type])
                    else:
                        atom_type_names.append(atom_type)

                # frame_data should always be shaped: n_probes,nkx,nky,n_layers,1 (idk why there's a trailing 1)
                cache_exists, frame_data = checkCache(
                    cache_file,
                    self.cache_wavefunctions and not force_rerun,
                    b,
                    expected_n_layers=self.n_layers,
                )
                if cache_exists and not self.prism and self.ADF:
                    # Keep the layer axis so each stored thickness gets its own
                    # ADF image (previously all layers were summed together).
                    intensities = b.einsum('pxyln,xy->pl', b.absolute(frame_data)**2, self.ADFmask)
                    for out_idx in range(self.n_layers):
                        self.ADF._array[out_idx] += intensities[self.ADFindex, out_idx]

                if not os.path.exists(self.output_dir / f"kx.npy"):
                    np.save(self.output_dir / f"kx.npy", to_numpy(self.kxs[self.keep_kxs_indices]))
                    np.save(self.output_dir / f"ky.npy", to_numpy(self.kys[self.keep_kys_indices]))
                if len(self.kxs) != self.nx and not os.path.exists(self.output_dir / f"kx_uncrop.npy"):
                    np.save(self.output_dir / f"kx_uncrop.npy", to_numpy(self.kxs))
                if len(self.kys) != self.ny and not os.path.exists(self.output_dir / f"ky_uncrop.npy"):
                    np.save(self.output_dir / f"ky_uncrop.npy", to_numpy(self.kys))

                if cache_exists:
                    frames_cached += 1
                else:
                    potential = Potential(
                        self.xs, self.ys, self.zs,
                        positions, atom_type_names,
                        backend=b,
                        kind="kirkland",
                        slice_axis=self.slice_axis,
                        cache_dir=cache_file.parent if self.cache_potentials else None,
                        frame_idx=frame_idx)

                    nc, npt, nx, ny = self.base_probe._array.shape; npt = len(self.base_probe.probe_positions)
                    n_slices = len(self.zs)
                    n_waves = len(self.base_probe.probe_positions)

                    # frame_data is always: p,x,y,l,1 (self.wavefunction_data expects p,t,x,y,l, since we loop time. recall Propagate gave l,p,x,y)
                    if self.returns_wavefunctions or self.cache_wavefunctions or self.prism:
                        fd_nx = self.nx; fd_ny = self.ny; fd_npt = self.n_probes
                        if self.use_memmap:
                            frame_data = b.memmap((n_waves, fd_nx, fd_ny, self.n_layers, 1), dtype=self.complex_dtype, filename=cache_file)
                        else:
                            frame_data = b.zeros((n_waves, fd_nx, fd_ny, self.n_layers, 1), dtype=self.complex_dtype)

                    # Propagate returns: [l,p,x,y] where l,p are both optional (if store_all_slices=True, and if n_probes>1)
                    chunks = []
                    if self.loop_probes:
                        chunksize = self.loop_probes if isinstance(self.loop_probes, int) else 1
                        for start in range(0, npt, chunksize):
                            chunk = b.arange(start, min(start + chunksize, npt))
                            # only keep chunk indices if they're also in probe_indices
                            chunk_np = to_numpy(chunk)
                            indices_np = to_numpy(self.probe_indices)
                            chunk = b.asarray(
                                chunk_np[np.any(indices_np[None, :] == chunk_np[:, None], axis=1)],
                                dtype=int,
                            )
                            if len(chunk) == 0:
                                continue
                            chunks.append(chunk)
                        pbar2 = tqdm(total=npt, desc="looping probes", unit="probe")
                    else:
                        chunks.append(b.arange(npt))
                        pbar2 = None

                    for selected in chunks:
                        if len(selected) == npt:
                            probe = self.base_probe
                        else:
                            probe = self.base_probe.copy(selected_probes=selected)
                        probe.applyShifts()
                        # propagate single probe
                        exit_waves_single = Propagate(
                            probe,
                            potential,
                            b,
                            progress=show_progress,
                            onthefly=True,
                            store_all_slices=not stores_exit_wave_only,
                            stored_slice_indices=_stored_layers if not stores_exit_wave_only else None,
                        )  # [l],p,x,y indices

                        # expand out to fixed l,p,x,y indices
                        exit_waves_single = b.expand_dims(exit_waves_single, 0) if len(exit_waves_single.shape) == 3 else exit_waves_single
                        # FFT and load into frame_data - always use 'axes' and let backend handle conversion
                        for out_idx, _ in enumerate(_stored_layers):
                            exit_waves_k = b.fft2(exit_waves_single[out_idx, :, :, :], axes=(-2, -1))  # l,p,x,y --> p,x,y
                            diffraction_patterns = b.fftshift(exit_waves_k, axes=(-2, -1))
                            diffraction_patterns = diffraction_patterns[:, self.keep_kxs_indices, :][:, :, self.keep_kys_indices]*self.kth**2
                            if self.use_memmap:
                                diffraction_patterns = to_numpy(diffraction_patterns)
                                selected = to_numpy(selected)
                            if self.returns_wavefunctions or self.cache_wavefunctions or self.prism:
                                frame_data[selected, :, :, out_idx, 0] = diffraction_patterns  # load p,x,y --> p,x,y,l,1 indices
                            if self.ADF and not self.prism:
                                intensities = b.einsum('pxy,xy->p', b.absolute(diffraction_patterns[:, :, :])**2, self.ADFmask)
                                # The batch is (nc, npt) flattened as c*npt+p, so
                                # fold the decoherence copies back and sum them
                                # (the detector sees the incoherent sum). The old
                                # zip() truncated to the first copy only.
                                n_copies = intensities.shape[0] // len(selected)
                                if n_copies > 1:
                                    intensities = b.sum(
                                        b.reshape(intensities, (n_copies, len(selected))), axis=0)
                                for i, pp in zip(intensities, selected):
                                    self.ADF._array[out_idx][self.ADFindex == pp] += i
                        if pbar2 is not None:
                            pbar2.update(len(selected))

                    if not self.use_memmap and self.cache_wavefunctions:
                        # Convert to CPU numpy array for saving
                        frame_data_cpu = to_numpy(frame_data)
                        np.save(cache_file, frame_data_cpu)
                    frames_computed += 1

                if self.returns_wavefunctions or self.prism:
                    cropped = frame_data[:, :, :, :, 0]

                if self.prism:
                    # Recall: Prism algorithm passes a series of sinusoids through the sample (fourier components shared by all real-space probes), so now for each real-space probe, we need to calculate the exitwaves from components
                    kwarg = {}
                    if self.ADF:
                        kwarg["ADF"] = (self.ADF, self.ADFmask, self.ADFindex)
                    if self.returns_wavefunctions:
                        kwarg["load_into"] = self.wavefunction_data[:, frame_idx, :, :, 0]
                    self.base_probe.calculateProbesFromS(frame_data, self.probe_positions, **kwarg, chunksize=self.loop_probes)
                elif self.returns_wavefunctions:
                    if self.use_memmap:
                        cropped = to_numpy(cropped)
                    self.wavefunction_data[:, frame_idx, :, :, :] = cropped  # load p,x,y,l,1 --> p,t,x,y,l indices
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
        time_array = np.arange(self.n_frames) * self.trajectory.timestep  # Time array in ps
        layer_array = np.array(_return_layers)

        # Package results
        array = b.zeros(
            (
                self.n_probes,
                self.n_frames,
                len(self.keep_kxs_indices),
                len(self.keep_kys_indices),
                len(_return_layers),
            ),
            dtype=self.complex_dtype,
        )
        if self.returns_wavefunctions:
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
            backend=b,
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
            self.ADF._array /= self.n_frames  # per-thickness time average
            # Depth (Å) of each ADF image (the z of its stored slice), and
            # collapse to a plain 2D image when a single thickness was stored.
            self.ADF.thicknesses = to_numpy(self.zs)[list(_stored_layers)]
            if self.n_layers == 1:
                self.ADF._array = self.ADF._array[0]
            return wf_data, self.ADF

        return wf_data


logging_tracker = []

def checkCache(cache_file, cache_wavefunctions, b, expected_n_layers=None):
    global logging_tracker
    if cache_wavefunctions and cache_file.exists():
        frame_data = np.load(cache_file)
        if expected_n_layers is not None and frame_data.shape[-2] != expected_n_layers:
            logging.warning(
                "Ignoring cache with %d layers at %s; expected %d",
                frame_data.shape[-2],
                cache_file,
                expected_n_layers,
            )
            return False, 0
        parent = str(cache_file.parent)
        if "cache_exists-"+parent not in logging_tracker:
            logging_tracker.append("cache_exists-"+parent)
            logging.warning("One or more frames reloaded from cache: "+str(cache_file.parent))
        return True, b.asarray(frame_data, dtype=b.complex_dtype)  # cache is saved as numpy, so cast back to the active backend
    return False, 0


class SEDCalculator:
    def __init__(self, device=None):
        self._backend = make_backend(device)

    def setup(self, trajectory: Trajectory, axis: int = 2, abc: list = [1, 1, 1]):
        """
        Set up Spectral Energy Density calculation

        Args:
            trajectory: Input trajectory data
        """
        b = self._backend
        self.trajectory = trajectory
        self.axis = axis
        self.a, self.b_cell, self.c = abc

        # Set up spatial grids
        lxyz = list(np.diag(trajectory.box_matrix))
        nxyz = [int(np.round(l/d)) for l, d in zip(lxyz, abc)]

        del lxyz[axis]
        del nxyz[axis]
        del abc[axis]

        self.kxs = b.linspace(0, 2*np.pi/abc[0], nxyz[0])
        self.kys = b.linspace(0, 2*np.pi/abc[1], nxyz[1])

        self.kvec = b.zeros((len(self.kxs), len(self.kys), 3))
        self.kvec[:, :, 0] += self.kxs[:, None]
        self.kvec[:, :, 1] += self.kys[None, :]

    def run(self) -> WFData:
        b = self._backend
        avg = self.trajectory.get_mean_positions()
        disp = self.trajectory.get_distplacements()

        # RUN SED INSTEAD OF MULTISLICE
        self.Zx, ws = SED(avg, disp, kvec=self.kvec, backend=b, v_xyz=0)
        self.Zy, ws = SED(avg, disp, kvec=self.kvec, backend=b, v_xyz=1)
        self.Zz, ws = SED(avg, disp, kvec=self.kvec, backend=b, v_xyz=2)

        self.ws = ws / self.trajectory.timestep

    def plot(self, w, filename=None):  # TODO MAYBE "RUN" SHOULD RETURN A TACAW OBJECT SO WE CAN REUSE TACAW PLOTTING/POSTPROCESSING FUNCTIONALITY??
        import matplotlib.pyplot as plt

        i = np.argmin(np.absolute(self.ws - w))
        extent = (np.amin(to_numpy(self.kxs)), np.amax(to_numpy(self.kxs)),
                  np.amin(to_numpy(self.kys)), np.amax(to_numpy(self.kys)))

        fig, ax = plt.subplots()
        ax.imshow(np.sqrt(self.Zx[i, :, :]+self.Zy[i, :, :]+self.Zz[i, :, :]).T, cmap="inferno", extent=extent)
        ax.set_xlabel("kx ($\\AA^{-1}$)")
        ax.set_ylabel("ky ($\\AA^{-1}$)")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

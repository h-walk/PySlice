import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Union
from tqdm import tqdm
import time, os
import hashlib

from .potentials import grid_from_trajectory, Potential
from .multislice import Probe, PrismProbe, Propagate, create_batched_probes, _propagation_operators
from .trajectory import Trajectory
from ..postprocessing.wf_data import WFData
from .sed import SED
from pyslice.backend import make_backend, to_numpy, NumpyBackend, source_files_version

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 2


def _hash_array(array) -> str:
    """Return a content hash including an array's dtype and shape."""
    values = np.ascontiguousarray(to_numpy(array))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(repr(values.shape).encode())
    digest.update(values.view(np.uint8))
    return digest.hexdigest()

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
                            probe_array=None, stored_layer_indices=None,
                            output_options=None, skip_vacuum=False):
        """Hash all inputs that affect cached wavefunction output."""
        params = {
            'cache_schema': CACHE_SCHEMA_VERSION,
            'cache_version': self._CACHE_VERSION,
            'positions_hash': _hash_array(trajectory.positions),
            'n_frames': trajectory.n_frames,
            'n_atoms': trajectory.n_atoms,
            'box_hash': _hash_array(trajectory.box_matrix),
            'atom_types_hash': _hash_array(trajectory.atom_types),
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
            'skip_vacuum': bool(skip_vacuum),
            'backend': 'torch' if not isinstance(self._backend, NumpyBackend) else 'numpy',
        }
        if stored_layer_indices is not None:
            params['stored_layer_indices'] = tuple(stored_layer_indices)
        if spatial_decoherence is not None:
            params['spatial_decoherence'] = spatial_decoherence
        if temporal_decoherence is not None:
            params['temporal_decoherence'] = temporal_decoherence
        if probe_array is not None:
            params['probe_hash'] = _hash_array(probe_array)
        if output_options is not None:
            params['output_options'] = output_options
        param_str = str(sorted(params.items()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _cache_output_dir(self, cache_key: str) -> Path:
        """Return the cache directory selected by ``save_path`` and backend."""
        root = Path(self.save_path) if self.save_path is not None else Path("psi_data")
        backend_name = "numpy" if isinstance(self._backend, NumpyBackend) else "torch"
        return root / f"{backend_name}_{cache_key}"

    def _cache_output_options(self):
        """Return output-affecting settings included in cache identity."""
        return {
            "slice_axis": int(self.slice_axis),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "max_kx": float(self.max_kx),
            "max_ky": float(self.max_ky),
            "min_dk": float(self.min_dk),
            "prism": False if self.prism is False else int(self.prism),
            "kth": int(self.kth),
        }

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
        aperture: float = 0.0,                                          # units: mrad
        voltage_eV: float = 60e3,                                       # units: eV
        defocus: float = 0.0,
        slice_thickness: float = 0.5,                                   # units: Angstrom
        sampling: float = 0.1,                                          # units: Angstrom
        probe_xs: Optional[List[float]] = None,
        probe_ys: Optional[List[float]] = None,
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        probe_tilt: Optional[tuple[float]] = (0,0),                     # units: mrad
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
        potential_cache_max_bytes: int = 16 * 1024**2,
        **kwargs,
    ):
        """
        Set up multislice simulation.

        Args:
            trajectory: Atomic positions, cell, elements, and frame spacing.
            aperture: Probe convergence semi-angle in milliradians. Use zero
                for a parallel TEM beam.
            voltage_eV: Electron accelerating voltage in electron-volts.
            defocus: Scalar defocus in Angstroms applied before probe shifts.
                Nonzero setup-time defocus is not implemented for PRISM.
            slice_thickness: Projected-potential slice thickness in Angstroms.
            sampling: Real-space pixel spacing in Angstroms per pixel.
            probe_xs: STEM scan coordinates along x in Angstroms. When supplied
                with ``probe_ys``, their Cartesian product defines the scan.
            probe_ys: STEM scan coordinates along y in Angstroms.
            probe_positions: Explicit ``(x, y)`` probe coordinates in
                Angstroms. Ignored when both scan-coordinate arrays are given.
            batch_size: Reserved frame-batch size. Frame propagation currently
                proceeds one frame at a time.
            save_path: Optional cache root. By default caches are written below
                ``psi_data`` in the current working directory.
            cleanup_temp_files: Delete cached frame files after loading them.
            slice_axis: Propagation axis. Only z (2) is currently supported.
            return_layers: Wavefunction layers to include in the returned
                ``WFData``. ``-1`` (default) returns the exit wave, ``"all"``
                returns every layer, and a list such as ``[43, 87, 175]`` returns
                selected layer wavefunctions. Negative indices are resolved
                against the layer stack. ``None`` or ``[]`` suppresses returned
                wavefunction data while still allowing HAADF calculation and
                optional exit-wave caching.
            cache_wavefunctions: Whether to read/write per-frame wavefunction cache files
            cache_potentials: Whether to read/write potential-slice cache data
            potential_cache_max_bytes: Maximum retained potential-volume bytes
                for reuse across probe batches within one frame (default 16 MiB).
                Only volumes fitting completely are retained; larger volumes
                stream as before. Set zero to disable. Separate from disk caching
                and from wavefunction/FFT working memory.
            max_kx: Maximum stored absolute kx in inverse Angstroms.
            max_ky: Maximum stored absolute ky in inverse Angstroms.
            use_memmap: Store large intermediate and result arrays as NumPy
                memory maps instead of keeping them entirely in RAM.
            loop_probes: False to propagate the complete probe batch together,
                or a positive integer giving the number of probes per chunk.
            min_dk: Minimum reciprocal-space sampling in inverse Angstroms. A
                positive value crops the real-space propagation window.
            prism: False for ordinary multislice, or a positive integer giving
                the PRISM Fourier-component count in each reciprocal direction.
            kth: Keep every kth reciprocal-space sample in returned data.
            ADF: False to disable on-the-fly integration, True for the default
                detector, or ``(inner_mrad, outer_mrad)`` for detector angles.
                In ADF mode, :meth:`run` returns ``(WFData, HAADFData)``.
            skip_vacuum: Skip probe positions far from atoms when probe
                cropping is active.

        Returns:
            None. The configured simulation state is stored on the calculator.
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

        if slice_axis != 2:
            raise NotImplementedError(
                "Only z-axis (slice_axis=2) multislice propagation is currently supported"
            )
        if kth < 1 or not isinstance(kth, (int, np.integer)):
            raise ValueError("kth must be a positive integer")
        if loop_probes is not False and (
            not isinstance(loop_probes, (int, np.integer)) or loop_probes < 1
        ):
            raise ValueError("loop_probes must be False or a positive integer")
        if (isinstance(potential_cache_max_bytes, (bool, np.bool_))
                or not isinstance(potential_cache_max_bytes, (int, np.integer))
                or potential_cache_max_bytes < 0):
            raise ValueError("potential_cache_max_bytes must be a nonnegative integer; use 0 to disable")

        self.trajectory = trajectory
        self.aperture = aperture
        self.voltage_eV = voltage_eV
        self.defocus = defocus
        self.slice_thickness = slice_thickness
        self.sampling = sampling
        self.probe_xs = probe_xs
        self.probe_ys = probe_ys
        self.probe_positions = probe_positions
        self.probe_tilt = probe_tilt
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
        self.potential_cache_max_bytes = int(potential_cache_max_bytes)
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

        # Use the realized grid spacing, which can differ slightly from the
        # requested sampling after the periodic box is divided into pixels.
        self.kxs = b.fftshift(b.fftfreq(self.nx, self.dx))  # cycles/Å
        self.kys = b.fftshift(b.fftfreq(self.ny, self.dy))  # cycles/Å
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

        if self.ADF:
            positions = np.asarray(self.probe_positions, dtype=float)
            expected_order = np.array(
                [(x, y) for y in self.probe_ys for x in self.probe_xs],
                dtype=float,
            )
            if len(positions) != len(expected_order) or not np.allclose(positions, expected_order):
                raise ValueError(
                    "On-the-fly ADF requires the complete Cartesian product of "
                    "probe_xs and probe_ys in PySlice grid order"
                )

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
            self.base_probe = Probe(xs, ys, self.aperture, self.voltage_eV, backend=b, probe_positions=self.probe_positions, cropping=self.probe_cropping, defer_shifts=True, tilt = self.probe_tilt)

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
            output_options=self._cache_output_options(),
            skip_vacuum=self.skip_vacuum,
        )
        self.output_dir = self._cache_output_dir(self.cache_key)

    def preview_probes(self, filename=None):
        """Plot the first-frame projected potential and configured probe positions."""
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
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    #@profile
    def run(self, force_rerun: bool = False) -> Union[WFData, Tuple[WFData, "HAADFData"]]:
        """Run propagation and return ``WFData`` or ``(WFData, HAADFData)``.

        Args:
            force_rerun: Recompute frames even when compatible cache files exist.

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
                                             self._cache_key_stored_layers(_stored_layers),
                                             output_options=self._cache_output_options(),
                                             skip_vacuum=self.skip_vacuum)
        if self.cache_key != cache_key:
            self.cache_key = cache_key
        self.output_dir = self._cache_output_dir(cache_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # if probes are over vacuum (e.g. nanoparticles), we don't need to propagate them?
        self.probe_indices = np.arange(len(self.probe_positions))
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
            self.probe_indices = np.asarray(self.probe_indices, dtype=int)
            print("filtered to", len(self.probe_indices), "probe positions")

        nc, npt, nx, ny = self.base_probe._array.shape
        n_scan_positions = len(self.probe_positions)
        self.n_probes = nc*n_scan_positions
        # Real-space frame caches store one row per incoherent-copy/scan pair.
        # PRISM caches instead store one row per propagated Fourier component.
        expected_cache_rows = (
            len(self.base_probe.probe_positions) if self.prism
            else self.n_probes
        )
        # Storage: [probe, frame, x, y, layer] - matches WFData expected format
        self.n_layers = len(_stored_layers)
        stores_exit_wave_only = self._stores_exit_wave_only(_stored_layers)
        # CPU selection metadata is independent of frames. Only device-side
        # indexing arrays are uploaded, once, and never downloaded for copying.
        npt = len(self.base_probe.probe_positions)
        chunksize = int(self.loop_probes) if self.loop_probes else npt
        chunks = []
        allowed = to_numpy(self.probe_indices)
        for start in range(0, npt, chunksize):
            selected = np.arange(start, min(start + chunksize, npt))
            if self.loop_probes:
                selected = selected[np.isin(selected, allowed)]
            if not len(selected):
                continue
            rows = (np.arange(nc)[:, None] * npt + selected[None, :]).reshape(-1)
            copy_indices = b.asarray(selected, dtype=int) if self.prism else selected
            output_rows = rows if self.use_memmap else b.asarray(rows, dtype=int)
            chunks.append((selected, copy_indices, rows, output_rows))
        operators = None
        potential_static = None
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

        # TACAW cache validity depends on analysis settings unavailable here.
        # TACAWData owns manifest validation; a bare tacaw.npy must never cause
        # multislice propagation to be skipped.
        skip_all_frames = False

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
                    expected_n_probes=expected_cache_rows,
                    mmap=not self.prism,
                )
                if cache_exists and not self.prism and self.ADF:
                    # Keep the layer axis so each stored thickness gets its own
                    # ADF image (previously all layers were summed together).
                    for selected, _, rows, _ in chunks:
                        cached_batch = b.asarray(frame_data[rows], dtype=self.complex_dtype)
                        intensities = b.einsum('pxyln,xy->pl', b.absolute(cached_batch)**2, self.ADFmask)
                        intensities = b.sum(b.reshape(intensities, (nc, len(selected), self.n_layers)), axis=0)
                        for out_idx in range(self.n_layers):
                            for value, pp in zip(intensities[:, out_idx], selected):
                                self.ADF._array[out_idx][self.ADFindex == int(pp)] += value

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
                    if potential_static is None:
                        potential = Potential(
                            self.xs, self.ys, self.zs, positions, atom_type_names,
                            backend=b, kind="kirkland", slice_axis=self.slice_axis,
                            cache_dir=cache_file.parent if self.cache_potentials else None,
                            frame_idx=frame_idx)
                        potential_static = potential._static_state()
                    else:
                        potential = Potential._from_static(potential_static, positions, frame_idx)

                    nc, npt, nx, ny = self.base_probe._array.shape; npt = len(self.base_probe.probe_positions)
                    n_slices = len(self.zs)
                    # Real-space propagation flattens decoherence copies into
                    # the wave axis. PRISM stores its Fourier components here
                    # instead and reconstructs real-space probes afterwards.
                    n_waves = (len(self.base_probe.probe_positions)
                               if self.prism else self.n_probes)

                    # frame_data is always: p,x,y,l,1 (self.wavefunction_data expects p,t,x,y,l, since we loop time. recall Propagate gave l,p,x,y)
                    if self.returns_wavefunctions and not self.prism:
                        # Write batches directly into the final output, whether
                        # that output is a device array or a disk-backed array.
                        frame_data = self.wavefunction_data[:, frame_idx, :, :, :, None]
                    elif self.cache_wavefunctions or self.prism:
                        fd_nx = self.nx; fd_ny = self.ny; fd_npt = self.n_probes
                        if self.use_memmap:
                            frame_data = b.memmap((n_waves, fd_nx, fd_ny, self.n_layers, 1), dtype=self.complex_dtype, filename=cache_file)
                        else:
                            frame_data = b.zeros((n_waves, fd_nx, fd_ny, self.n_layers, 1), dtype=self.complex_dtype)

                    # Propagate returns: [l,p,x,y] where l,p are both optional (if store_all_slices=True, and if n_probes>1)
                    pbar2 = tqdm(total=npt, desc="looping probes", unit="probe") if self.loop_probes else None

                    # Retain a single frame's potential only when every slice
                    # fits the explicit byte budget. Large volumes keep the
                    # existing one-slice path (and optional disk cache).
                    potential_bytes = (potential.nx * potential.ny * potential.n_slices
                                       * to_numpy(b.zeros(0)).dtype.itemsize)
                    if len(chunks) > 1 and potential_bytes <= self.potential_cache_max_bytes:
                        potential.build()

                    for selected, copy_indices, rows, selected_rows in chunks:
                        if len(selected) == npt:
                            probe = self.base_probe
                        else:
                            probe = self.base_probe.copy(selected_probes=copy_indices)
                        probe.applyShifts()
                        if operators is None:
                            operators = _propagation_operators(probe, potential, b)
                        # propagate single probe
                        exit_waves_single = Propagate(
                            probe,
                            potential,
                            b,
                            progress=show_progress,
                            onthefly=True,
                            store_all_slices=not stores_exit_wave_only,
                            stored_slice_indices=_stored_layers if not stores_exit_wave_only else None,
                            _operators=operators,
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
                            if self.returns_wavefunctions or self.cache_wavefunctions or self.prism:
                                # Propagation flattens (copy, selected-probe) in
                                # copy-major order. Expand the selected position
                                # indices across copies to preserve that layout.
                                frame_data[selected_rows, :, :, out_idx, 0] = diffraction_patterns
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

                    if pbar2 is not None:
                        pbar2.close()
                    if self.cache_wavefunctions:
                        # Convert to CPU numpy array for saving
                        if isinstance(frame_data, np.memmap) and Path(frame_data.filename) == cache_file:
                            frame_data.flush()
                        else:
                            np.save(cache_file, to_numpy(frame_data))
                    frames_computed += 1

                if self.prism:
                    # Recall: Prism algorithm passes a series of sinusoids through the sample (fourier components shared by all real-space probes), so now for each real-space probe, we need to calculate the exitwaves from components
                    kwarg = {}
                    if self.ADF:
                        kwarg["ADF"] = (self.ADF, self.ADFmask, self.ADFindex)
                    if self.returns_wavefunctions:
                        kwarg["load_into"] = self.wavefunction_data[:, frame_idx, :, :, :]
                    self.base_probe.calculateProbesFromS(frame_data, self.probe_positions, **kwarg, chunksize=self.loop_probes)
                elif self.returns_wavefunctions and cache_exists:
                    # Cache files stay mapped on the host. Upload only a batch
                    # when output lives on the GPU; memmapped output stays CPU.
                    for _, _, rows, output_rows in chunks:
                        values = frame_data[rows, :, :, :, 0]
                        if not self.use_memmap:
                            values = b.asarray(values, dtype=self.complex_dtype)
                        self.wavefunction_data[output_rows, frame_idx, :, :, :] = values
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

        # Reuse populated wave storage; a zero-length layer axis allocates no
        # wave values when output is suppressed and preserves the array API.
        if self.returns_wavefunctions:
            array = self.wavefunction_data
        else:
            array = b.zeros(
                (self.n_probes, self.n_frames, len(self.keep_kxs_indices),
                 len(self.keep_kys_indices), 0), dtype=self.complex_dtype,
            )
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
        wf_data.source_fingerprint = self.cache_key

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
            # Each stored wave is captured after that slice's transmission, so
            # its depth is the far boundary (i + 1) * dz, not the slice's
            # coordinate sample zs[i]. The final layer must equal specimen lz.
            # collapse to a plain 2D image when a single thickness was stored.
            dz = float(self.lz) / len(self.zs)
            self.ADF.thicknesses = (
                np.asarray(_stored_layers, dtype=float) + 1.0) * dz
            self.ADF._set_dimensions(
                self.ADF.thicknesses if self.n_layers > 1 else None,
                layer_name='thickness', layer_units='Å')
            if self.n_layers == 1:
                self.ADF._array = self.ADF._array[0]
            return wf_data, self.ADF

        return wf_data


logging_tracker = []

def checkCache(cache_file, cache_wavefunctions, b, expected_n_layers=None,
               expected_n_probes=None, mmap=False):
    """Load a compatible cached frame, optionally leaving it mapped on the CPU.

    Args:
        mmap: Return a read-only NumPy memory map for batched cache consumption.
            Defaults to False to retain the standalone backend-array API.

    Returns:
        ``(True, frame_data)`` when a compatible cache is loaded, otherwise
        ``(False, 0)``.
    """
    global logging_tracker
    if cache_wavefunctions and cache_file.exists():
        frame_data = np.load(cache_file, mmap_mode='r' if mmap else None)
        if expected_n_layers is not None and frame_data.shape[-2] != expected_n_layers:
            logging.warning(
                "Ignoring cache with %d layers at %s; expected %d",
                frame_data.shape[-2],
                cache_file,
                expected_n_layers,
            )
            return False, 0
        if expected_n_probes is not None and frame_data.shape[0] != expected_n_probes:
            logging.warning(
                "Ignoring cache with %d probes at %s; expected %d",
                frame_data.shape[0], cache_file, expected_n_probes,
            )
            return False, 0
        parent = str(cache_file.parent)
        if "cache_exists-"+parent not in logging_tracker:
            logging_tracker.append("cache_exists-"+parent)
            logging.warning("One or more frames reloaded from cache: "+str(cache_file.parent))
        return True, frame_data if mmap else b.asarray(frame_data, dtype=b.complex_dtype)
    return False, 0


class SEDCalculator:
    """Experimental atomic spectral-energy-density calculator.

    Unlike :class:`MultisliceCalculator`, this analyzes atomic displacements
    directly and does not propagate an electron wavefunction.
    """

    def __init__(self, device=None):
        """Select the NumPy or PyTorch backend used for the SED calculation."""
        self._backend = make_backend(device)

    def setup(self, trajectory: Trajectory, axis: int = 2, abc: list = [1, 1, 1]):
        """
        Set up Spectral Energy Density calculation

        Args:
            trajectory: Input molecular-dynamics trajectory.
            axis: Cartesian axis normal to the two-dimensional reciprocal grid.
            abc: Real-space lattice spacings in Angstroms along x, y, and z.
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

    def run(self) -> None:
        """Compute x-, y-, and z-polarized SED arrays in place.

        Results are stored on ``Zx``, ``Zy``, ``Zz``, and ``ws``; ``ws`` is in
        THz when the trajectory timestep is expressed in picoseconds.
        """
        b = self._backend
        avg = self.trajectory.get_mean_positions()
        disp = self.trajectory.get_distplacements()

        # RUN SED INSTEAD OF MULTISLICE
        self.Zx, ws = SED(avg, disp, kvec=self.kvec, backend=b, v_xyz=0)
        self.Zy, ws = SED(avg, disp, kvec=self.kvec, backend=b, v_xyz=1)
        self.Zz, ws = SED(avg, disp, kvec=self.kvec, backend=b, v_xyz=2)

        self.ws = ws / self.trajectory.timestep

    def plot(self, w, filename=None):  # TODO MAYBE "RUN" SHOULD RETURN A TACAW OBJECT SO WE CAN REUSE TACAW PLOTTING/POSTPROCESSING FUNCTIONALITY??
        """Plot total SED magnitude at the frequency bin nearest ``w`` THz."""
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

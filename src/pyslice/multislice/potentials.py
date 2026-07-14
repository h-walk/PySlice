from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm import tqdm
import importlib.resources as resources

from pyslice.backend import Backend, make_backend, to_cpu, to_numpy

logger = logging.getLogger(__name__)

kirkland_file = resources.files('pyslice.data').joinpath('kirkland.txt')

# ---------------------------------------------------------------------------
# Element / Kirkland utilities
# ---------------------------------------------------------------------------

_ELEMENTS = [
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No",
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl",
    "Mc", "Lv", "Ts", "Og",
]


def get_z_from_element(element: str) -> int:
    """Return atomic number (Z) for an element symbol."""
    try:
        return _ELEMENTS.index(element) + 1
    except ValueError:
        raise ValueError(f"Unknown element symbol: {element!r}")


def _resolve_z(atom_type: Union[str, int]) -> int:
    """Resolve an atom type (string or int) to an atomic number."""
    if isinstance(atom_type, str):
        return get_z_from_element(atom_type)
    return int(atom_type)


def load_kirkland(backend: Backend) -> any:
    """
    Load and return the Kirkland scattering parameters as a backend array.

    Shape: (103, 3, 4)  — 103 elements, 3 Gaussian terms, 4 columns (a, b, c, d).

    The returned array lives on the backend's default device. This function
    no longer uses module-level global state; callers are responsible for
    caching the result if they need it.
    """
    params = []
    for i in range(103):
        skip = i * 4 + 1
        try:
            abcd = np.loadtxt(kirkland_file, skiprows=skip, max_rows=3)
            a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = abcd.flat
            # Reorder to columns (a, b, c, d) — Kirkland p. 291
            params.append([[a1, b1, c1, d1],
                            [a2, b2, c2, d2],
                            [a3, b3, c3, d3]])
        except Exception:
            logger.warning("Kirkland parameters unavailable for element %d; using zeros.", i + 1)
            params.append([[0, 0, 0, 0]] * 3)

    return backend.asarray(params)   # shape (103, 3, 4)


def kirkland_form_factor(qsq: any, Z: int, kirkland_params: any,
                         backend: Backend) -> any:
    """
    Compute the Kirkland electron scattering form factor for element Z.

    Args:
        qsq:             |q|² array, shape (nx, ny)
        Z:               Atomic number (1-based)
        kirkland_params: Parameter array, shape (103, 3, 4), on the correct device
        backend:         Active Backend instance

    Returns:
        Form factor array with the same shape as qsq
    """
    ABCDs = kirkland_params[Z - 1]      # shape (3, 4)
    a = ABCDs[:, 0]                     # shape (3,)
    b = ABCDs[:, 1]
    c = ABCDs[:, 2]
    d = ABCDs[:, 3]

    # Broadcast over spatial dimensions without storing repeated copies
    a3 = a[:, None, None]
    b3 = b[:, None, None]
    c3 = c[:, None, None]
    d3 = d[:, None, None]
    qsq3 = qsq[None, :, :]

    term1 = backend.sum(a3 / (qsq3 + b3), axis=0)
    term2 = backend.sum(c3 * backend.exp(-d3 * qsq3), axis=0)
    return term1 + term2


def grid_from_trajectory(trajectory, sampling: float = 0.1,
                         slice_thickness: float = 0.5,
                         backend: Optional[Backend] = None):
    """
    Build coordinate grids from a trajectory box matrix.

    Returns:
        xs, ys, zs, lx, ly, lz
    """
    box = trajectory.box_matrix
    lx, ly, lz = box[0, 0], box[1, 1], box[2, 2]

    nx = int(lx / sampling) + 1
    ny = int(ly / sampling) + 1
    nz = int(lz / slice_thickness) + 1

    xs = np.linspace(0, lx, nx, endpoint=False)
    ys = np.linspace(0, ly, ny, endpoint=False)
    zs = np.linspace(0, lz, nz, endpoint=False)
    if backend is not None:
        xs = backend.asarray(xs)
        ys = backend.asarray(ys)
        zs = backend.asarray(zs)

    return xs, ys, zs, lx, ly, lz


# ---------------------------------------------------------------------------
# Potential
# ---------------------------------------------------------------------------

# Prefactor: 2πℏ²/m_e in V·Å²  (Kirkland Eq. C.1)
_FE_TO_V = 47.87764737


class Potential:
    """
    Projected electrostatic potential computed via the Kirkland parameterisation.

    Parameters
    ----------
    xs, ys, zs:
        1-D coordinate arrays (Å).
    positions:
        Atom positions, shape (N, 3).
    atom_types:
        Sequence of element symbols (str) or atomic numbers (int), length N.
    backend:
        Backend instance to use for all array operations.
    kind:
        Scattering parameterisation — currently 'kirkland' or 'gauss'.
    slice_axis:
        Axis (0, 1, or 2) along which to slice the sample.
    cache_dir:
        Optional directory for per-slice potential caching.
    frame_idx:
        Frame index used to disambiguate cache filenames.
    chunk_size:
        Maximum number of atoms processed per structure-factor batch.
    """

    def __init__(
            self,
            xs, ys, zs,
            positions=None,
            atom_types=None,
            backend: Optional[Backend] = None,
            array=None,
            kind: str = "kirkland",
            slice_axis: int = 2,
            cache_dir: Optional[Path] = None,
            frame_idx: Optional[int] = None,
            chunk_size: int = 2000,
            device: Optional[str] = None,
    ):
        self._backend = make_backend(device) if backend is None else backend
        backend = self._backend

        # ----------------------------------------------------------------
        # Convert inputs to backend arrays
        # ----------------------------------------------------------------
        self.xs = backend.asarray(xs)
        self.ys = backend.asarray(ys)
        self.zs = backend.asarray(zs)

        self.nx = len(xs)
        self.ny = len(ys)
        self.nz = len(zs)
        self.dx = float(to_numpy(self.xs[1] - self.xs[0]))
        self.dy = float(to_numpy(self.ys[1] - self.ys[0]))
        self.dz = float(to_numpy(self.zs[1] - self.zs[0])) if self.nz > 1 else 0.5

        # ----------------------------------------------------------------
        # Slice-axis geometry
        # ----------------------------------------------------------------
        self.slice_axis = slice_axis
        if slice_axis != 2:
            # The propagator, slice spacing (dz), slice loop and reciprocal grid
            # in Propagate are all hard-coded to the z axis, and the (nx, ny)
            # reciprocal grid here is built from the x/y axes regardless of
            # slice_axis. A non-z value therefore lays the wrong coordinate onto
            # the grid and propagates with the wrong spacing, silently producing
            # wrong results. Permute the trajectory/grid so the beam direction
            # is z (axis 2) and keep the default slice_axis=2.
            raise NotImplementedError(
                "slice_axis != 2 is not supported (it would silently produce "
                "wrong results). Permute your trajectory so the beam direction "
                "is the z axis and use slice_axis=2."
            )
        inplane = [a for a in range(3) if a != slice_axis]
        self.inplane_axis1, self.inplane_axis2 = inplane

        coord_arrays = [self.xs, self.ys, self.zs]
        spacings = [self.dx, self.dy, self.dz]
        self.slice_coords = coord_arrays[slice_axis]
        self.slice_spacing = spacings[slice_axis]
        self.n_slices = len(self.slice_coords)
        self.slice_period = float(to_numpy(self.slice_coords[-1] + self.slice_spacing))

        # ----------------------------------------------------------------
        # k-space frequencies and |q|²
        # ----------------------------------------------------------------
        self.kxs = backend.fftfreq(self.nx, d=self.dx)
        self.kys = backend.fftfreq(self.ny, d=self.dy)
        qsq = self.kxs[:, None] ** 2 + self.kys[None, :] ** 2

        self._cache_dir = cache_dir
        self._frame_idx = frame_idx

        if array is not None:
            self.array = backend.asarray(array, dtype=backend.float_dtype)
            return

        if positions is None or atom_types is None:
            raise ValueError("positions and atom_types are required unless array is provided")

        positions = backend.asarray(positions)

        # ----------------------------------------------------------------
        # Resolve atom types to atomic numbers
        # ----------------------------------------------------------------
        unique_types = list(dict.fromkeys(atom_types))  # preserves order, deduplicates
        atom_z_list = [_resolve_z(at) for at in atom_types]
        atom_z_np = np.array(atom_z_list, dtype=np.int64)

        # ----------------------------------------------------------------
        # Precompute form factors (once per unique atom type)
        # ----------------------------------------------------------------
        if kind == "kirkland":
            kirkland_params = load_kirkland(backend)
        
        form_factors = {}
        for at in unique_types:
            Z = _resolve_z(at)
            if kind == "kirkland":
                form_factors[at] = kirkland_form_factor(qsq, Z, kirkland_params, backend)
            elif kind == "gauss":
                form_factors[at] = backend.exp(-qsq / 2.0)
            else:
                raise ValueError(f"Unknown scattering kind: {kind!r}")

        # ----------------------------------------------------------------
        # Store everything needed by _calculate_slice
        # ----------------------------------------------------------------
        self._positions = positions
        self._atom_types = atom_types
        self._atom_z_np = atom_z_np
        self._unique_types = unique_types
        self._form_factors = form_factors
        self._chunk_size = chunk_size
        self.array: Optional[any] = None

    # ------------------------------------------------------------------
    # Internal slice calculation
    # ------------------------------------------------------------------

    def _calculate_slice(self, slice_idx: int) -> any:
        """Compute the projected potential for one slice."""
        backend = self._backend
        if self.array is not None:
            return self.array[:, :, slice_idx]

        # Check cache
        cache_file = self._cache_path(slice_idx)
        if cache_file is not None and cache_file.exists():
            return backend.asarray(np.load(cache_file))

        reciprocal = backend.zeros(
            (self.nx, self.ny), dtype=backend.complex_dtype)

        slice_min, slice_max = self._slice_bounds(slice_idx)

        for at in self._unique_types:
            form_factor = self._form_factors[at]

            # Build atom-type mask (numpy, to avoid sending booleans to GPU)
            if isinstance(at, str):
                type_mask_np = np.array(
                    [t == at for t in self._atom_types], dtype=bool)
            else:
                type_mask_np = (self._atom_z_np == int(at))

            if not type_mask_np.any():
                continue

            # Pull relevant positions to numpy for masking, then back to backend
            positions_np = to_numpy(self._positions)
            type_positions_np = positions_np[type_mask_np]

            slice_coords_np = type_positions_np[:, self.slice_axis]
            if self.slice_period > 0:
                slice_coords_np = np.mod(slice_coords_np, self.slice_period)
            spatial_mask_np = (
                (slice_coords_np >= slice_min) & (slice_coords_np < slice_max))

            if not spatial_mask_np.any():
                continue

            slice_positions_np = type_positions_np[spatial_mask_np]
            atomsx = backend.asarray(slice_positions_np[:, self.inplane_axis1])
            atomsy = backend.asarray(slice_positions_np[:, self.inplane_axis2])

            shape_factor = backend.zeros(
                (self.nx, self.ny), dtype=backend.complex_dtype)

            n_atoms = len(atomsx)
            for start in range(0, n_atoms, self._chunk_size):
                end = min(start + self._chunk_size, n_atoms)
                atx = atomsx[start:end]
                aty = atomsy[start:end]

                # exp(−2πi kx x) summed over atoms  →  shape factor
                expx = backend.exp(
                    -1j * 2 * np.pi * self.kxs[None, :] * atx[:, None])
                expy = backend.exp(
                    -1j * 2 * np.pi * self.kys[None, :] * aty[:, None])
                shape_factor += backend.einsum('ax,ay->xy', expx, expy)

            reciprocal += shape_factor * form_factor

        # Transform to real space and apply physical prefactor
        real_space = backend.real(backend.ifft2(reciprocal))
        # Kirkland Eq. C.1:  V_proj = (2πℏ²/m_e) / (dx·dy) × IFFT(S · f_e)
        result = real_space * _FE_TO_V / (self.dx * self.dy)

        if cache_file is not None:
            np.save(cache_file, to_numpy(result))

        return result

    def _slice_bounds(self, slice_idx: int):
        """Return (min, max) coordinate bounds for a given slice index."""
        coords_np = to_numpy(self.slice_coords)
        half = self.slice_spacing / 2.0
        lo = coords_np[slice_idx] - half if slice_idx > 0 else 0.0
        hi = (coords_np[slice_idx] + half
              if slice_idx < self.n_slices - 1
              else coords_np[-1] + self.slice_spacing)
        return lo, hi

    def _cache_path(self, slice_idx: int) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return (self._cache_dir /
                f"potential_{self._frame_idx}_{slice_idx}.npy")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, progress: bool = False) -> None:
        """Compute the full 3-D potential array."""
        if self.array is not None:
            return

        backend = self._backend
        potential = backend.zeros(
            (self.nx, self.ny, self.n_slices), dtype=backend.float_dtype)

        iterator = range(self.n_slices)
        if progress:
            logger.info("Generating potential for %d slices", self.n_slices)
            iterator = tqdm(iterator)

        for slice_idx in iterator:
            potential[:, :, slice_idx] = self._calculate_slice(slice_idx)

        self.array = potential

    def flatten(self) -> None:
        """Collapse the slice axis into one projected potential plane."""
        if self.array is None:
            self.build()

        backend = self._backend
        self.array = backend.mean(self.array, axis=2, keepdims=True)
        self.nz = 1
        self.n_slices = 1
        self.zs = self.zs[:1]
        self.slice_coords = self.slice_coords[:1]

    def to_numpy(self) -> np.ndarray:
        """Return the potential array as a CPU NumPy array."""
        if self.array is None:
            raise RuntimeError("Call build() before to_numpy().")
        return to_numpy(self.array)

    def to_cpu(self):
        """Return the potential array moved to CPU without forcing NumPy."""
        if self.array is None:
            raise RuntimeError("Call build() before to_cpu().")
        return to_cpu(self.array)

    def plot(self, filename: Optional[str] = None) -> None:
        """Plot the summed projected potential."""
        if self.array is None:
            self.build()

        import matplotlib.pyplot as plt

        backend = self._backend
        array_np = to_numpy(
            backend.sum(backend.absolute(self.array), axis=2)).T

        extent = (
            float(to_numpy(backend.amin(self.xs))),
            float(to_numpy(backend.amax(self.xs))),
            float(to_numpy(backend.amin(self.ys))),
            float(to_numpy(backend.amax(self.ys))),
        )

        fig, ax = plt.subplots()
        ax.imshow(array_np, cmap="inferno", extent=extent)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close(fig)

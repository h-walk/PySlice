from __future__ import annotations

import os
import logging
from functools import lru_cache
from importlib.resources.abc import Traversable
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


@lru_cache(maxsize=1)
def _read_kirkland_table(source: Traversable) -> np.ndarray:
    """Read the bundled parameter table once, retaining only a small CPU array.

    Parameters
    ----------
    source : importlib.resources.abc.Traversable
        Kirkland text resource, with one header and three rows per element.

    Returns
    -------
    numpy.ndarray
        Read-only float64 coefficients shaped ``(103, 3, 4)``.
    """
    lines = source.read_text().splitlines()
    params = []
    for i in range(103):
        skip = i * 4 + 1
        try:
            abcd = np.asarray([line.split() for line in lines[skip:skip + 3]], dtype=float)
            a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = abcd.flat
            # Reorder to columns (a, b, c, d) — Kirkland p. 291
            params.append([[a1, b1, c1, d1],
                            [a2, b2, c2, d2],
                            [a3, b3, c3, d3]])
        except Exception:
            logger.warning("Kirkland parameters unavailable for element %d; using zeros.", i + 1)
            params.append([[0, 0, 0, 0]] * 3)

    table = np.asarray(params, dtype=np.float64)
    table.setflags(write=False)
    return table


def load_kirkland(backend: Backend):
    """Return an independent parameter array on the requested backend.

    Parameters
    ----------
    backend : Backend
        Backend supplying the array's device and floating-point precision.

    Returns
    -------
    array_like
        Coefficients shaped ``(103, 3, 4)``. Only the immutable CPU table is
        shared; modifying this result cannot affect another caller or device.
    """
    return backend.asarray(_read_kirkland_table(kirkland_file).copy())


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
    box = np.asarray(trajectory.box_matrix, dtype=float)
    if box.shape != (3, 3):
        raise ValueError("trajectory.box_matrix must have shape (3, 3)")
    if not np.allclose(box, np.diag(np.diag(box)), atol=1e-10):
        raise ValueError(
            "Multislice currently requires an axis-aligned orthogonal cell; "
            "orthogonalize or fold the trajectory before propagation."
        )
    lx, ly, lz = box[0, 0], box[1, 1], box[2, 2]
    if min(lx, ly, lz) <= 0:
        raise ValueError("trajectory cell lengths must be positive")
    if sampling <= 0 or slice_thickness <= 0:
        raise ValueError("sampling and slice_thickness must be positive")

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
        self._slice_coords_np = to_numpy(self.slice_coords).copy()

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
        # Keep one host snapshot for slice selection, not a device array that
        # must be downloaded afresh for every element of every slice.
        self._atom_types = atom_types
        self._atom_z_np = atom_z_np
        self._unique_types = unique_types
        self._form_factors = form_factors
        self._chunk_size = chunk_size
        self._set_frame(positions, frame_idx)

    def _set_frame(self, positions, frame_idx: Optional[int]) -> None:
        """Replace only frame-dependent state, retaining backend precision.

        Parameters
        ----------
        positions : array_like
            Atomic coordinates for this frame, in the original atom order.
        frame_idx : int or None
            Slice-cache frame identifier.
        """
        self._positions = np.array(to_numpy(self._backend.asarray(positions)), copy=True)
        self._frame_idx = frame_idx
        self.array = None
        self._index_slice_atoms()

    def _static_state(self) -> dict:
        """Extract run-invariant geometry without retaining a frame's volume.

        Returns
        -------
        dict
            Shared grid, atom identities, and element form factors. No positions,
            slice membership, or computed potential values are retained.
        """
        names = ('_backend', 'xs', 'ys', 'zs', 'nx', 'ny', 'nz', 'dx', 'dy', 'dz',
                 'slice_axis', 'inplane_axis1', 'inplane_axis2', 'slice_coords',
                 'slice_spacing', 'n_slices', 'slice_period', '_slice_coords_np',
                 'kxs', 'kys', '_cache_dir', '_atom_types', '_atom_z_np',
                 '_unique_types', '_form_factors', '_chunk_size')
        return {name: getattr(self, name) for name in names}

    @classmethod
    def _from_static(cls, state: dict, positions, frame_idx: int) -> Potential:
        """Construct another frame within a fixed-geometry calculator run.

        Parameters
        ----------
        state : dict
            State from ``_static_state`` for this run's grid, backend, and atoms.
        positions : array_like
            New atomic coordinates, with unchanged identities and ordering.
        frame_idx : int
            Slice-cache frame identifier.

        Returns
        -------
        Potential
            Frame-local potential sharing only invariant arrays with its peers.
        """
        result = cls.__new__(cls)
        result.__dict__.update(state)
        result._set_frame(positions, frame_idx)
        return result

    def _index_slice_atoms(self) -> None:
        """Index atoms once per frame while preserving the original slice rules.

        Notes
        -----
        Sorted wrapped coordinates allow binary searches for each slice's
        original half-open bounds, including first/last-slice asymmetry and
        roundoff-sized gaps or overlaps. Indices are restored to input order
        inside each group so structure-factor chunking and summation order
        remain unchanged. Storage scales with atom indices, not potential voxels.
        """
        coords = self._positions[:, self.slice_axis]
        if self.slice_period > 0:
            coords = np.mod(coords, self.slice_period)
        # Cast bounds as NumPy's comparisons against the coordinate array did.
        bounds = np.asarray([self._slice_bounds(i) for i in range(self.n_slices)],
                            dtype=coords.dtype)
        self._slice_atom_indices = [{} for _ in range(self.n_slices)]
        for at in self._unique_types:
            mask = (np.asarray([t == at for t in self._atom_types], dtype=bool)
                    if isinstance(at, str) else self._atom_z_np == int(at))
            indices = np.flatnonzero(mask)
            ordered = indices[np.argsort(coords[indices], kind='stable')]
            edges = np.searchsorted(coords[ordered], bounds, side='left')
            for slice_idx, (lo, hi) in enumerate(edges):
                self._slice_atom_indices[slice_idx][at] = np.sort(ordered[lo:hi])

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

        for at in self._unique_types:
            form_factor = self._form_factors[at]
            indices = self._slice_atom_indices[slice_idx][at]
            if not len(indices):
                continue
            slice_positions_np = self._positions[indices]
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
        coords_np = self._slice_coords_np
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

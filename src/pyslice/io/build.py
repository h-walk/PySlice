"""Build crystallographically exact, multislice-ready slabs with ASE.

Carving a rotated structure out of a box (``Trajectory.rotate_to`` +
``slice_positions``) leaves non-periodic edges, which show up as artifacts
under the periodic boundary conditions of a multislice run. This module uses
ASE — already a core PySlice dependency — to build **exactly periodic**
oriented slabs instead: ``ase.build.surface`` stacks the requested plane
along the beam axis, an integer-supercell search orthogonalizes the in-plane
cell when the surface cell is oblique (e.g. hexagonal), and the result is
returned as a diagonal-box :class:`~pyslice.multislice.trajectory.Trajectory`
plus a build record suitable for ``Metadata.build`` provenance.
"""
from __future__ import annotations

import math
from itertools import product
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

_ORTHO_TOL = 1e-5


def atom_symbols(trajectory: Any) -> list:
    """Return element symbols for a trajectory's atom types.

    Normalizes the ``atom_types`` str/int gotcha: string types pass through,
    integer types are treated as atomic numbers.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Trajectory whose ``atom_types`` to normalize.

    Returns
    -------
    list[str]
        Element symbols, one per atom.

    Raises
    ------
    ValueError
        If an integer type is not a valid atomic number (1-118); the message
        names the ``atom_mapping`` fix.

    See Also
    --------
    pyslice.io.loader.Loader : Accepts ``atom_mapping`` at load time.

    Examples
    --------
    >>> atom_symbols(trajectory)  # doctest: +SKIP
    ['C', 'C']
    """
    from ..multislice.potentials import _ELEMENTS

    symbols = []
    for atom_type in trajectory.atom_types:
        if isinstance(atom_type, (str, np.str_)):
            symbols.append(str(atom_type))
        else:
            z = int(atom_type)
            if not 1 <= z <= len(_ELEMENTS):
                raise ValueError(
                    f"atom type {z} is not a valid atomic number; pass atom_mapping when loading LAMMPS files."
                )
            symbols.append(_ELEMENTS[z - 1])
    return symbols


def trajectory_to_ase(trajectory: Any, frame: int = 0):
    """Convert one trajectory frame to ASE atoms with element-symbol types.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Source trajectory.
    frame : int, optional
        Frame index to convert, by default 0.

    Returns
    -------
    ase.Atoms
        The selected frame with periodic boundary conditions.

    Raises
    ------
    ValueError
        If atom types cannot be resolved to element symbols.

    See Also
    --------
    atom_symbols : The type normalization used here.
    """
    from ase import Atoms

    return Atoms(
        atom_symbols(trajectory),
        positions=trajectory.positions[frame],
        cell=trajectory.box_matrix,
        pbc=True,
    )


def orthogonal_supercell_matrix(cell: np.ndarray, max_index: int = 6) -> np.ndarray:
    """Find an integer in-plane supercell matrix that orthogonalizes a cell.

    Searches small integer combinations ``v1 = m1*a + m2*b`` and
    ``v2 = n1*a + n2*b`` of the first two lattice vectors for a mutually
    orthogonal, right-handed pair of minimal area, leaving the third vector
    untouched. Hexagonal cells (graphene, hBN, wurtzite basal plane) resolve
    to the standard orthorhombic supercell.

    Parameters
    ----------
    cell : numpy.ndarray
        3x3 cell matrix, lattice vectors in rows; the third vector must be
        out of plane.
    max_index : int, optional
        Largest absolute integer coefficient searched, by default 6.

    Returns
    -------
    numpy.ndarray
        3x3 integer matrix ``P`` such that ``P @ cell`` has orthogonal
        in-plane vectors (identity when the cell is already orthogonal).

    Raises
    ------
    ValueError
        If no orthogonal combination exists within ``max_index``; the message
        suggests raising the bound or using the carve fallback.

    See Also
    --------
    build_slab : Uses this to make surface cells multislice-compatible.

    Notes
    -----
    Orthogonality is exact only when the lattice admits it (rational
    vector-angle relations); a bigger ``max_index`` widens the search but
    grows the supercell.
    """
    a, b = np.asarray(cell)[0], np.asarray(cell)[1]
    if abs(np.dot(a, b)) <= _ORTHO_TOL * np.linalg.norm(a) * np.linalg.norm(b):
        return np.eye(3, dtype=int)

    best: Optional[Tuple[float, Tuple[int, int, int, int]]] = None
    indices = range(-max_index, max_index + 1)
    for m1, m2, n1, n2 in product(indices, repeat=4):
        det = m1 * n2 - m2 * n1
        if det <= 0:  # right-handed, non-degenerate
            continue
        v1 = m1 * a + m2 * b
        v2 = n1 * a + n2 * b
        n1v, n2v = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1v == 0 or n2v == 0:
            continue
        if abs(np.dot(v1, v2)) > _ORTHO_TOL * n1v * n2v:
            continue
        area = float(n1v * n2v)
        if best is None or area < best[0] - 1e-9:
            best = (area, (m1, m2, n1, n2))
    if best is None:
        raise ValueError(
            f"No orthogonal in-plane supercell found with |indices| <= {max_index}. "
            "Raise max_index, or fall back to rotate_to + slice_positions (carved, non-periodic edges)."
        )
    m1, m2, n1, n2 = best[1]
    return np.array([[m1, m2, 0], [n1, n2, 0], [0, 0, 1]], dtype=int)


def _rotate_in_plane(atoms) -> None:
    """Rotate cell and positions about z so the first lattice vector lies on +x.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose first two cell vectors lie in the xy-plane; modified
        in place.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the in-plane vectors do not end up on the Cartesian axes (the
        cell was not orthogonal).
    """
    cell = np.array(atoms.get_cell())
    angle = math.atan2(cell[0, 1], cell[0, 0])
    c, s = math.cos(-angle), math.sin(-angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    atoms.set_positions(atoms.get_positions() @ rot.T)
    atoms.set_cell(cell @ rot.T)
    cell = np.array(atoms.get_cell())
    scale = max(1.0, float(np.abs(cell).max()))
    if abs(cell[0, 1]) > _ORTHO_TOL * scale or abs(cell[1, 0]) > _ORTHO_TOL * scale or cell[1, 1] <= 0:
        raise ValueError("In-plane cell vectors are not orthogonal after rotation; orthogonalize first.")


def build_slab(
    structure: Any,
    indices: Sequence[int] = (0, 0, 1),
    thickness_A: Optional[float] = None,
    layers: Optional[int] = None,
    min_lateral_A: Optional[float] = None,
    repeats: Optional[Sequence[int]] = None,
    vacuum_A: float = 0.0,
    max_index: int = 6,
    timestep: Optional[float] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Build an exactly periodic, beam-oriented slab as a Trajectory.

    Stacks the ``indices`` plane of the input crystal along the beam (z)
    axis with ``ase.build.surface(..., periodic=True)``, orthogonalizes the
    in-plane cell when needed, squares it up on the Cartesian axes, applies
    lateral repeats and optional vacuum, and converts the result to a
    single-frame :class:`~pyslice.multislice.trajectory.Trajectory` with a
    diagonal box — the form the multislice grid requires.

    Parameters
    ----------
    structure : Trajectory | ase.Atoms
        Bulk unit cell to build from (frame 0 is used for trajectories).
    indices : Sequence[int], optional
        Miller indices of the plane stacked along the beam, by default
        (0, 0, 1). For cubic crystals this equals the [hkl] zone axis; for
        2D sheets keep (0, 0, 1) and ``layers=1``.
    thickness_A : float | None, optional
        Target slab thickness in Å; converted to whole layers (rounded up).
    layers : int | None, optional
        Explicit repeat-unit count along the beam; overrides ``thickness_A``.
        Defaults to 1 when neither is given.
    min_lateral_A : float | None, optional
        Minimum lateral extent in Å; converted to whole-cell repeats per
        axis (rounded up).
    repeats : Sequence[int] | None, optional
        Explicit lateral repeats ``(nx, ny)``; overrides ``min_lateral_A``.
    vacuum_A : float, optional
        Vacuum added along the beam, split evenly above and below, by
        default 0.
    max_index : int, optional
        Search bound for the orthogonalizing supercell, by default 6.
    timestep : float | None, optional
        Timestep stored on the returned trajectory (picoseconds).

    Returns
    -------
    tuple[Trajectory, dict]
        The slab trajectory (single frame, diagonal box) and a build record
        (method, indices, layers, layer height, orthogonalization matrix,
        repeats, vacuum, final box, atom count) for ``Metadata.build``.

    Raises
    ------
    ValueError
        If neither the surface cell nor any small integer supercell is
        orthogonal, or inputs are inconsistent.
    ImportError
        If ASE is not installed (a core dependency).

    See Also
    --------
    orthogonal_supercell_matrix : The in-plane orthogonalization search.
    pyslice.multislice.trajectory.Trajectory.rotate_to : Carve fallback for
        orientations without a periodic orthogonal cell.

    Notes
    -----
    ASE's ``surface`` treats ``indices`` as the *surface plane* (hkl); its
    normal becomes the beam axis. For cubic crystals the (hkl) normal and
    the [hkl] direction coincide, matching the "view down [hkl]" reading.

    Examples
    --------
    >>> slab, record = build_slab(diamond, (1, 1, 0), thickness_A=400,
    ...                           min_lateral_A=20)  # doctest: +SKIP
    >>> record["layers"], slab.n_atoms  # doctest: +SKIP
    (225, ...)
    """
    from ase import Atoms
    from ase.build import make_supercell, surface

    from .loader import Loader

    if isinstance(structure, Atoms):
        bulk_atoms = structure
    else:
        bulk_atoms = trajectory_to_ase(structure)

    indices = tuple(int(i) for i in indices)
    if len(indices) != 3 or not any(indices):
        raise ValueError(f"indices must be three Miller indices with at least one non-zero, got {indices}")

    # One-layer probe fixes the repeat-unit height along the beam.
    probe = surface(bulk_atoms, indices, layers=1, periodic=True)
    layer_height = float(np.array(probe.get_cell())[2, 2])
    if layers is None:
        layers = max(1, math.ceil(thickness_A / layer_height)) if thickness_A else 1
    slab = surface(bulk_atoms, indices, layers=int(layers), periodic=True)

    supercell = orthogonal_supercell_matrix(np.array(slab.get_cell()), max_index=max_index)
    if not np.array_equal(supercell, np.eye(3, dtype=int)):
        slab = make_supercell(slab, supercell)
    _rotate_in_plane(slab)
    slab.wrap(eps=1e-7)

    cell = np.array(slab.get_cell())
    lateral = np.array([cell[0, 0], cell[1, 1]])
    if repeats is None:
        if min_lateral_A:
            repeats = tuple(max(1, math.ceil(min_lateral_A / extent)) for extent in lateral)
        else:
            repeats = (1, 1)
    repeats = (int(repeats[0]), int(repeats[1]))
    if repeats != (1, 1):
        slab = slab.repeat((repeats[0], repeats[1], 1))

    if vacuum_A:
        cell = np.array(slab.get_cell())
        cell[2, 2] += float(vacuum_A)
        slab.set_cell(cell, scale_atoms=False)
        slab.positions[:, 2] += float(vacuum_A) / 2.0

    # The multislice grid reads only the box diagonal — enforce it exactly.
    cell = np.array(slab.get_cell())
    diag = np.diag(np.diag(cell))
    if not np.allclose(cell, diag, atol=_ORTHO_TOL * max(1.0, float(np.abs(cell).max()))):
        raise ValueError(f"Slab cell is not diagonal after orthogonalization:\n{cell}")
    slab.set_cell(diag, scale_atoms=False)

    trajectory = Loader(atoms=slab, timestep=timestep).load()
    record: Dict[str, Any] = {
        "method": "ase.build.surface + orthogonal supercell",
        "indices": list(indices),
        "layers": int(layers),
        "layer_height_A": round(layer_height, 6),
        "thickness_A": round(layer_height * int(layers), 4),
        "orthogonal_supercell": supercell.tolist(),
        "lateral_repeats": list(repeats),
        "vacuum_A": float(vacuum_A),
        "box_A": [round(float(v), 4) for v in np.diag(diag)],
        "n_atoms": int(trajectory.n_atoms),
    }
    return trajectory, record


def reciprocal_cell(box_matrix: np.ndarray) -> np.ndarray:
    """Return the reciprocal lattice of a cell in cycles/Å (no 2π).

    Rows ``b_i`` satisfy ``a_i · b_j = δ_ij``, matching the ``fftfreq``
    convention PySlice uses for its k-axes.

    Parameters
    ----------
    box_matrix : numpy.ndarray
        3x3 cell matrix, lattice vectors in rows (Å).

    Returns
    -------
    numpy.ndarray
        3x3 reciprocal matrix, rows in 1/Å.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the cell is singular.

    See Also
    --------
    first_bragg_g : Smallest in-plane reciprocal vector magnitude.
    """
    return np.linalg.inv(np.asarray(box_matrix, dtype=float)).T


def first_bragg_g(box_matrix: np.ndarray, max_index: int = 3, in_plane: bool = True) -> float:
    """Return the magnitude of the smallest non-zero reciprocal vector.

    This is the spatial frequency of the first Bragg reflection (1/d for the
    widest lattice spacing), the reference for probe-step and k-range rules.
    Pass a *unit-cell* box, not a tiled supercell (tiling shrinks g
    artificially). Extinction rules are ignored, which can only make the
    result smaller — conservative for sampling rules.

    Parameters
    ----------
    box_matrix : numpy.ndarray
        3x3 unit-cell matrix, lattice vectors in rows (Å).
    max_index : int, optional
        Miller-index search bound, by default 3.
    in_plane : bool, optional
        Restrict the search to in-plane reflections (l = 0), by default
        True — the beam axis of an oriented slab (often vacuum-padded) does
        not contribute Bragg peaks to the recorded pattern.

    Returns
    -------
    float
        Magnitude of the first reflection's reciprocal vector in 1/Å
        (cycles/Å, no 2π).

    Raises
    ------
    numpy.linalg.LinAlgError
        If the cell is singular.

    See Also
    --------
    reciprocal_cell : The underlying reciprocal lattice.

    Examples
    --------
    >>> import numpy as np
    >>> round(1.0 / first_bragg_g(np.diag([3.567, 3.567, 3.567])), 3)
    3.567
    """
    recip = reciprocal_cell(box_matrix)
    best = math.inf
    rng = range(-max_index, max_index + 1)
    l_rng = (0,) if in_plane else rng
    for h, k, l in product(rng, rng, l_rng):
        if h == 0 and k == 0 and l == 0:
            continue
        g = float(np.linalg.norm(np.array([h, k, l]) @ recip))
        best = min(best, g)
    return best

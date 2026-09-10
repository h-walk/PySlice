"""Resolve PySlice trajectories into SEA ``atomic-structure`` collections.

Implements sea-eco's ``signal-containers`` schema, ``atomic-structure``
profile version 1 (CONT-6) for a PySlice
:class:`~pyslice.multislice.trajectory.Trajectory`. The prescriptive source is
``sea-eco/src/pySEA/ai_wiki/sea_eco/schema/signal-containers/intents.md``; this
module is an implementation of it, and PySlice's mapping is recorded in
``docs/conformance/signal-containers.md``.

Registered by :mod:`pyslice.data.seashell` on first use, so
``seashell.resolve(trajectory)`` and ``Trajectory.sea`` both produce a
conforming, validated collection with no explicit conversion call.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

import numpy as np

from . import seashell


def trajectory_to_atomic_structure(
    trajectory: Any,
    name: str = "structure",
    kind: str = "Sample",
    source: Optional[Dict[str, Any]] = None,
    build: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> Any:
    """Build a marked ``atomic-structure`` collection from a trajectory.

    Produces a :class:`SignalCollection` holding an ``atoms``
    :class:`SignalSet` (``position``, ``element``,
    ``clamp_boundary_condition``, plus ``velocity`` when the trajectory
    carries any) and a ``cell`` SignalSet (``cell``,
    ``periodic_boundary_condition``). Coordinate and cell-vector axes are
    categorical (``x``/``y``/``z`` and ``a``/``b``/``c``); value units live on
    scalar ``SignalQuantities`` rather than on the component axes, as the
    profile requires.

    Single-frame structures use the profile's static form (no context axis);
    multi-frame trajectories carry a calibrated ``time`` context on
    ``position``. The cell stays static because a ``Trajectory`` holds one
    ``box_matrix`` for every frame.

    Parameters
    ----------
    trajectory : pyslice.multislice.trajectory.Trajectory
        Structure or trajectory to resolve.
    name : str, optional
        Collection name, by default ``"structure"``.
    kind : str, optional
        ``"Material"`` (unit cell) or ``"Sample"`` (built structure), recorded
        in ``Metadata.Material``, by default ``"Sample"``.
    source : dict | None, optional
        Database origin recorded at ``Metadata.Database``.
    build : dict | None, optional
        Build record recorded at ``Metadata.build``.
    validate : bool, optional
        Validate against the profile before marking, by default True.

    Returns
    -------
    SignalCollection
        Marked, optionally validated atomic-structure collection.

    Raises
    ------
    ImportError
        If sea-eco is not installed.
    ValueError
        If the assembled collection fails profile validation.

    See Also
    --------
    pyslice.data.seashell.resolve : Implicit entry point.
    pyslice.multislice.trajectory.Trajectory.sea : Cached property using this.

    Notes
    -----
    PySlice treats every cell as fully periodic (the multislice propagator
    assumes it), so ``periodic_boundary_condition`` is all True, and no atom
    is clamped, so ``clamp_boundary_condition`` is all False.

    Examples
    --------
    >>> structure = trajectory_to_atomic_structure(traj, kind="Material")  # doctest: +SKIP
    >>> structure["atoms"]["position"].dimensions.get_names()  # doctest: +SKIP
    ['atom', 'coordinate']
    """
    seashell.require_sea("Resolving a Trajectory to an atomic-structure collection")

    from ..io.build import atom_symbols

    mark_atomic_structure = seashell.mark_atomic_structure

    Dimension = seashell.Dimension
    Dimensions = seashell.Dimensions
    Signal = seashell.Signal
    SignalQuantities = seashell.SignalQuantities

    symbols = atom_symbols(trajectory)
    counts = Counter(symbols)
    formula = "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))
    n_atoms = int(trajectory.n_atoms)
    n_frames = int(trajectory.n_frames)
    contextual = n_frames > 1

    def atom_axis() -> Any:
        """Return a fresh ``atom`` index axis for one member."""
        return Dimension(name="atom", size=n_atoms, scale=1, offset=0)

    def coordinate_axis() -> Any:
        """Return a fresh categorical ``coordinate`` axis (x/y/z)."""
        return Dimension(name="coordinate", values=["x", "y", "z"])

    def time_axis() -> Any:
        """Return a fresh calibrated ``time`` context axis in picoseconds."""
        return Dimension(
            name="time", space="temporal", units="ps",
            values=np.arange(n_frames) * float(trajectory.timestep or 0.0),
        )

    def vector_field(array: np.ndarray, member: str, units: str) -> Any:
        """Build one (*context, atom, coordinate) member Signal."""
        axes = [atom_axis(), coordinate_axis()]
        data = np.asarray(array, dtype=float)
        if contextual:
            axes.insert(0, time_axis())
        else:
            data = data[0]
        return Signal(
            data,
            name=member,
            dimensions=Dimensions(axes),
            signal_quantities=SignalQuantities([Dimension(name=member, units=units)]),
        )

    members = [
        vector_field(trajectory.positions, "position", "Å"),
        Signal(
            np.asarray([str(symbol) for symbol in symbols]),
            name="element",
            dimensions=Dimensions([atom_axis()]),
        ),
        Signal(
            np.zeros(n_atoms, dtype=bool),
            name="clamp_boundary_condition",
            dimensions=Dimensions([atom_axis()]),
        ),
    ]
    velocities = np.asarray(trajectory.velocities, dtype=float)
    if np.any(velocities):
        members.append(vector_field(velocities, "velocity", "Å/ps"))

    cell_vector = Dimension(name="cell_vector", values=["a", "b", "c"])
    cell_members = [
        Signal(
            np.asarray(trajectory.box_matrix, dtype=float),
            name="cell",
            dimensions=Dimensions([cell_vector.deepcopy(), coordinate_axis()]),
            signal_quantities=SignalQuantities([Dimension(name="cell", units="Å")]),
        ),
        Signal(
            np.ones(3, dtype=bool),
            name="periodic_boundary_condition",
            dimensions=Dimensions([cell_vector.deepcopy()]),
        ),
    ]

    metadata: Dict[str, Any] = {
        "Material": {
            "kind": kind,
            "formula": formula,
            "elements": {element: int(n) for element, n in sorted(counts.items())},
            "n_atoms": n_atoms,
            "n_frames": n_frames,
            "timestep_ps": float(trajectory.timestep or 0.0),
        },
    }
    if source:
        metadata["Database"] = dict(source)
    if build:
        metadata["build"] = dict(build)

    structure = seashell.SignalCollection(
        [seashell.SignalSet(members, name="atoms"), seashell.SignalSet(cell_members, name="cell")],
        name=name,
        metadata=seashell.Metadata(metadata),
    )
    return mark_atomic_structure(structure, validate=validate)


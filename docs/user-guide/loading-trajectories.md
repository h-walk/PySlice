# Loading trajectories

`Loader` converts source data into a reduced `Trajectory` used by multislice.
It is not a lossless atomistic archive.

## Data contract

| Field | Shape | Unit/meaning |
|---|---|---|
| `positions` | `(frame, atom, 3)` | Å |
| `velocities` | `(frame, atom, 3)` | source-dependent; absent values become zero |
| `atom_types` | `(atom,)` | element symbols or atomic numbers |
| `box_matrix` | `(3, 3)` | one fixed cell, row lattice vectors in Å |
| `timestep` | scalar | spacing between saved frames in ps |

Atom count, identity, ordering, and cell must be constant across ASE frames.
Variable-cell ASE trajectories are rejected rather than silently collapsed to
one cell. PySlice does not preserve arbitrary ASE arrays/info, charges, tags,
constraints, atom IDs, per-frame timestamps, per-frame cells, or PBC flags.
Keep the source ASE/OVITO object or file as the archival record.

## Format guidance

| Input | Route | Required attention |
|---|---|---|
| CIF | ASE | inspect non-orthogonal cells before multislice |
| LAMMPS dump | OVITO | supply complete `atom_mapping`; confirm LAMMPS length units and saved-frame spacing |
| XYZ | OVITO | plain XYZ may lack a physical periodic cell |
| ASE `Atoms` | direct | one frame; cell required for multislice |
| iterable of ASE `Atoms` | direct | fixed atom identity/order and fixed cell required |

For an ASE trajectory, the least ambiguous pattern is:

```python
from ase.io import read
from pyslice import Loader

frames = read("production.traj", index=":")
trajectory = Loader(atoms=frames, timestep=0.01).load()
```

Here `0.01` is ps between saved frames. PySlice cannot infer it reliably from
the filename.

## Preflight

```python
import numpy as np

print(trajectory.positions.shape, trajectory.positions.dtype)
print(np.unique(trajectory.atom_types))
print(trajectory.box_matrix)
print("volume:", np.linalg.det(trajectory.box_matrix), "Å³")
print("coordinate range:",
      trajectory.positions.min(axis=(0, 1)),
      trajectory.positions.max(axis=(0, 1)))
```

Successful parsing does not prove simulation readiness. Check finite positions,
element identity, positive cell volume, coordinate bounds, beam direction, and
representative frames.

Multislice currently requires an axis-aligned orthogonal cell. For a tilted
cell whose Cartesian bounds are intentionally being folded into an orthogonal
simulation box, inspect the structure and use:

```python
trajectory = trajectory.fold_positions_to_orthogonal_box()
```

This is a representation choice, not a crystallographic no-op; verify periodic
neighbors and the resulting projected structure.

## Transform semantics

- `tile_positions((nx, ny, nz))` repeats row lattice vectors and scales the cell.
- `slice_positions(...)` currently requires an orthogonal box, selects atoms by
  their mean position, shifts retained coordinates to the new origin, and
  shrinks the corresponding cell lengths.
- `slice_timesteps(..., ith=k)` preserves chronology and multiplies frame
  spacing by `k`.
- `select_timesteps(...)`, `random_frames(...)`, and generated frozen-phonon
  configurations set `timestep=0`: they are ensembles, not uniform time series,
  and TACAW rejects them.
- `get_displacements()` is an arithmetic Cartesian displacement; unwrap atoms
  before calling it when trajectories cross periodic boundaries.
- `get_center_of_mass_drift()` reports the mass-weighted rigid translation of
  each frame relative to frame 0 (or another reference frame).
- `remove_center_of_mass_drift()` returns a sample-fixed copy with that uniform
  translation, and by default the COM velocity, removed. It preserves all
  intraframe interatomic vectors to floating-point precision. Unwrap first so
  periodic-image changes are not misdiagnosed as physical drift.

For a solid trajectory intended for TACAW:

```python
drift = trajectory.get_center_of_mass_drift()
print("final COM shift (Angstrom):", drift[-1])
print("maximum COM shift (Angstrom):", np.linalg.norm(drift, axis=1).max())

trajectory = trajectory.remove_center_of_mass_drift()
```

This changes the laboratory-frame origin, not the internal motion. Do not apply
it when whole-body translation is itself part of the phenomenon being studied.

## File-loader cache

File inputs create `.npy` sidecars and a `.cache.json` manifest beside the
source. The manifest includes source size/modification time, mapping, and OVITO
arguments. Editing the source or parser settings invalidates reuse. Delete the
sidecars when moving or auditing a dataset if provenance is uncertain.

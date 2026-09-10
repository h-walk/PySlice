# Conformance: `signal-containers`

PySlice **implements** (does not own) this schema. The prescriptive source is
sea-eco's `src/pySEA/ai_wiki/sea_eco/schema/signal-containers/` — read
`intents.md` there before changing anything below.

PySlice's only surface on this schema is the atomic structure it writes into
`SEAFile.Materials` when exporting a simulation: the unit cell as a *Material*
entry and the built structure as a *Sample* entry, both as `atomic-structure`
profile version 1 collections.

| Contract | Implementation | Verification |
|---|---|---|
| CONT-6 atomic profile v1 — marked `SignalCollection` with `atoms`/`cell` SignalSets | `PySliceService._trajectory_to_atomic_structure` (`src/pyslice/mcp/service.py`), marked via `mark_atomic_structure` | `tests/31_sea_file_export.py::test_sea_file_round_trip_materials_and_provenance` (calls `validate_atomic_structure` on the reloaded entry) |
| `atoms.position` float `(*context, atom, coordinate)`, `coordinate` = x/y/z | same; static form for single-frame structures, `time` context axis for multi-frame trajectories | same test (contextual) and `::test_database_source_flows_into_material_metadata` (static) |
| `atoms.element` string `(atom,)`; `atoms.clamp_boundary_condition` bool `(atom,)` | same; element symbols normalized by `pyslice.io.build.atom_symbols`, clamp all-False (PySlice clamps no atoms) | same test |
| `cell.cell` float `(*context, cell_vector, coordinate)`, `cell_vector` = a/b/c | same; static, because a PySlice `Trajectory` carries one `box_matrix` for all frames | same test |
| `cell.periodic_boundary_condition` bool `(*context, cell_vector)` | same; all-True — the multislice propagator assumes a fully periodic cell | same test |
| Value units on scalar `SignalQuantities`, not on component axes | `SignalQuantities([Dimension(name="position", units="Å")])`, likewise `cell` and optional `velocity` (Å/ps) | same test asserts the quantity units and that `coordinate` carries none |
| CONT-1 shared-registry member views | delegated to `SignalSet`; PySlice passes per-member `atom` axis copies and lets the set bind them | same test asserts `element.dimensions["atom"] is atoms.dimensions["atom"]` |
| CONT-2 categorical selection | delegated to sea-eco | same test asserts `position(coordinate="x")` |

## Extensions PySlice writes

Permitted by the profile ("optional typed atom properties are allowed"):

- `atoms.velocity` — float `(*context, atom, coordinate)`, units Å/ps, written
  only when the trajectory carries non-zero velocities (MD output; a
  frozen-phonon trajectory has none).

## PySlice-side metadata (outside the schema)

Provenance rides on the root collection's `Metadata` and is not part of the
profile: `Metadata.Material` (kind, formula, element counts, atom/frame
counts), `Metadata.Database` (provider, entry id, CIF path) on entries fetched
from Materials Project or COD, and `Metadata.build` (the `build_slab` /
transform record, operations JSON-encoded because `Metadata` does not persist
lists of dicts).

## Not implemented

- **Bonds.** PySlice has no bond model; the profile's bond representation is
  unused.
- **Variable atom counts across context** — outside profile v1 anyway, and
  PySlice trajectories hold a fixed atom count by construction
  (`Trajectory._validate_shapes`).
- **Reading** atomic-structure collections. PySlice only writes them; ingest
  is via `Loader` (CIF/XYZ/LAMMPS/ASE). A `.sea`-structure reader would be new
  work.

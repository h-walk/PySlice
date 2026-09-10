# Schema: `signal-containers`

PySlice **implements** this schema; sea-eco **owns** it. Structure writing is
therefore a governed surface, not a local convention.

- **Prescriptive source:**
  `sea-eco/src/pySEA/ai_wiki/sea_eco/schema/signal-containers/intents.md`
  (intents CONT-1…6) plus `fixtures/index.json`.
- **PySlice's conformance record:** `docs/conformance/signal-containers.md`.
- **Implementation:** `src/pyslice/data/atomic_structure.py`, reached
  implicitly through `Trajectory.sea` — see {doc}`resolution_layer`.

## What PySlice implements

Only **CONT-6**, the `atomic-structure` profile version 1: the structures
PySlice writes into `SEAFile.Materials`. A trajectory becomes a marked
`SignalCollection`:

```
structure                        SignalCollection, marked
├── atoms                        SignalSet
│   ├── position                 float  (*context, atom, coordinate)   units on the quantity
│   ├── element                  string (atom,)
│   ├── clamp_boundary_condition bool   (atom,)
│   └── velocity                 float  (*context, atom, coordinate)   optional
└── cell                         SignalSet
    ├── cell                     float  (*context, cell_vector, coordinate)
    └── periodic_boundary_condition  bool  (*context, cell_vector)
```

`coordinate` is categorical `x`/`y`/`z`; `cell_vector` is categorical
`a`/`b`/`c`. **Value units live on scalar `SignalQuantities`, never on the
component axes** — the profile is explicit about this, and it is the rule most
easily got wrong.

The collection is built through `mark_atomic_structure`, which validates
before marking, so an invalid structure cannot be produced in the first place.

## PySlice-specific choices inside the profile's freedom

| Choice | Why |
|---|---|
| Single-frame structures use the **static** form (no context axis) | The profile allows it, and a size-1 time axis on a static unit cell is noise |
| Multi-frame trajectories carry a `time` context on `position` | That is what the frames mean |
| The **cell stays static** even when positions are contextual | A PySlice `Trajectory` holds exactly one `box_matrix` for all frames; a per-frame cell would be fabricated |
| `periodic_boundary_condition` is all `True` | The multislice propagator assumes a fully periodic cell |
| `clamp_boundary_condition` is all `False` | PySlice has no notion of a clamped atom |
| `velocity` is written only when non-zero | Permitted as an optional typed atom property; a frozen-phonon trajectory has none |

## Metadata PySlice attaches (outside the schema)

Provenance rides on the root collection's `Metadata` and is not part of the
profile:

- `Metadata.Material` — kind (`Material`/`Sample`), formula, element counts,
  atom and frame counts, timestep.
- `Metadata.Database` — provider, entry id, CIF path, for structures fetched
  from Materials Project or COD.
- `Metadata.build` — the `build_slab`/transform record. **Operations are
  JSON-encoded strings**, because sea-eco's `Metadata` silently drops
  lists-of-dicts on HDF5 write.

## Not implemented

Stated rather than left as a gap:

- **Bonds.** PySlice has no bond model; the profile's bond representation is
  unused.
- **Variable atom counts across context** — outside profile v1 anyway, and a
  `Trajectory` holds a fixed atom count by construction
  (`Trajectory._validate_shapes`).
- **Reading.** PySlice writes these collections and never reads them; there is
  no `atomic-structure` → `Trajectory` path. Ingest is `Loader`.
- **CONT-1…5.** Shared-registry views, selection semantics, stacking,
  serialization/migration, and pipeline dispatch are sea-eco's to implement.
  PySlice relies on them (its members share an `atom` axis through the
  `SignalSet`) but implements none of them.

## Changing this

**Behaviour changes start in the schema.** Update intents and fixtures in
sea-eco first, then this implementation, then
`docs/conformance/signal-containers.md`. Do not re-derive or hand-copy schema
rules into PySlice — point at the schema, as this page does.

## Provenance and verification

| Aspect | Where |
|---|---|
| Schema (owner) | `sea-eco: src/pySEA/ai_wiki/sea_eco/schema/signal-containers/{intents.md,fixtures/index.json}` |
| Conformance record | `docs/conformance/signal-containers.md` |
| Implementation | `src/pyslice/data/atomic_structure.py` |
| Implicit entry point | `Trajectory.sea`; `pyslice.data.seashell.resolve` |
| Validation gate | sea-eco's own `validate_atomic_structure`, called on the **reloaded** entry in `tests/31_sea_file_export.py` |
| Focused tests | `tests/31_sea_file_export.py` (round trip, axis names, categorical values, quantity units, CONT-1 identity, CONT-2 selection); `tests/32_seashell_resolution.py` (implicit resolution, caching) |
| User guide | {doc}`../guides/sea_results` |
| API reference | {doc}`../api_reference` — `pyslice.data.atomic_structure` |

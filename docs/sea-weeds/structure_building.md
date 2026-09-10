# Structure building

**What exists:** `src/pyslice/io/build.py` — `build_slab` for exactly periodic
beam-oriented slabs, plus the reciprocal-lattice helpers the parameter rules
depend on. `src/pyslice/io/databases.py` — Materials Project and COD clients.

**Why:** a multislice run is periodic in the plane. A sample whose lateral
edges are not crystallographically periodic scatters off those cut edges, and
the artifact is indistinguishable from physics unless you know to look for it.

## The problem with rotate-and-carve

The obvious way to make an oriented sample is `Trajectory.rotate_to((h,k,l))`
then `slice_positions`. It produces a rotated block, and for a general
rotation that block does **not** tile its own bounding box. Under periodic
boundaries the cut faces wrap onto each other.

This is tolerable for a convergent probe kept well away from the edges. It is
damaging for parallel-beam diffraction, where the whole cell contributes.

It also leaves `box_matrix` non-diagonal, and **the multislice grid reads only
the diagonal** (`grid_from_trajectory` uses `box[0,0]`, `box[1,1]`, `box[2,2]`),
so the grid silently misdescribes the cell.

## What `build_slab` does instead

1. `ase.build.surface(bulk, indices, layers, periodic=True)` stacks the
   requested plane along the beam, giving a genuinely periodic surface cell.
2. If that cell's in-plane vectors are oblique, an integer-supercell search
   (`orthogonal_supercell_matrix`) finds the smallest right-handed pair of
   mutually orthogonal lattice combinations — hexagonal cells become the
   standard orthorhombic supercell automatically.
3. The cell is rotated so the in-plane vectors lie on the Cartesian axes.
4. Lateral repeats and optional vacuum are applied.
5. The box is asserted diagonal and set exactly so.

`thickness_A` converts to whole layers (rounded up, never truncated);
`min_lateral_A` converts to whole cell repeats. The returned build record —
layers, layer height, the orthogonalization matrix, repeats, vacuum, final box,
atom count — is what lands in `Metadata.build` (see {doc}`signal_containers`).

Verified geometry: diamond (110) comes out at exact crystal density (atom
count matches 8·V/a³), and graphene resolves to the a × a√3 orthorhombic cell.

## Two conventions worth knowing

**`indices` is the ASE surface plane (hkl).** Its normal becomes the beam
axis. For cubic crystals the (hkl) normal and the [hkl] direction coincide, so
"view down [110]" reads correctly. For non-cubic cells they are different
vectors, and "expose the (hkl) surface" is not "view down [uvw]".

**`first_bragg_g` ignores extinction rules.** It searches |hkl| ≤ 3 in-plane
and returns the smallest non-zero reciprocal vector, so diamond yields
d = 3.567 Å (the {100} spacing) rather than the first *allowed* reflection
{111} at 2.06 Å. It feeds probe-step and k-range rules, where a larger d is
the conservative direction (finer sampling, wider range). Making it
structure-factor-aware would change every parameter downstream of it, so it is
a deliberate single point of change.

Both are flagged for review rather than buried: they are judgement calls, not
derivations.

## Databases

`io/databases.py` uses the standard library only — no `requests`, no
`pymatgen`. Materials Project structures arrive as pymatgen `Structure` JSON
and are rendered as **P1 CIFs** (all sites explicit) so ASE can read them
without a pymatgen dependency; no symmetry is lost for simulation purposes.
COD CIFs are served verbatim.

Errors are actionable by contract: a missing MP key names both environment
variables and the signup URL; a network failure names the unreachable host.

**Boundary:** these clients fetch and parse. They do not judge whether a
structure is physical — partial occupancies (common in COD refinements) load
through ASE but are not simulable, and that is the caller's problem to notice.

## Failure modes

| Failure | Symptom | Guard |
|---|---|---|
| No small orthogonal periodic cell exists | `ValueError` naming `max_index` and the carve fallback | `orthogonal_supercell_matrix` search bound |
| Cell not diagonal after orthogonalization | `ValueError` showing the cell | Explicit assertion before returning |
| LAMMPS integer atom types reach `to_ase()` | `ValueError` naming `atom_mapping` | `atom_symbols` normalization |
| MP key absent | `DatabaseError` naming both env vars and the signup URL | `_resolve_mp_api_key` |
| COD id not numeric | `DatabaseError` | id check in `fetch_cif` |

## Limitations

- Orthogonalization is bounded by `max_index` (default 6). A lattice needing
  larger integers raises rather than searching forever.
- No defect, interface, or surface-reconstruction building — `build_slab`
  makes clean periodic slabs. Compose transforms, or bring a built structure.
- Cell vectors are assumed to have the third out of plane.

## Provenance and verification

| Aspect | Where |
|---|---|
| Implementation | `src/pyslice/io/build.py`, `src/pyslice/io/databases.py` |
| Focused tests | `tests/29_slab_builder.py` (density, orthogonalization, thickness rounding, vacuum, diagonal box, negative case); `tests/28_databases.py` (mocked MP/COD, URL construction, P1 CIF loadability, env-gated live COD) |
| Parameter rules that consume it | `skills/simulation-parameter-selection/SKILL.md` rules 11–12 |
| User guide | {doc}`../guides/structures` |
| AI-tool artifacts | `pyslice_build_slab`, `pyslice_search_structures`, `pyslice_fetch_structure` in `src/pyslice/mcp/server.py`; `skills/structure-retrieval/SKILL.md` |
| API reference | {doc}`../api_reference` — `pyslice.io.build`, `pyslice.io.databases` |

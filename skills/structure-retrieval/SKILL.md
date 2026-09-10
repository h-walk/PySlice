---
name: structure-retrieval
description: Use when a PySlice simulation needs a crystal structure that isn't on disk — searching the Materials Project or Crystallography Open Database by formula or elements, downloading CIF files, handling API keys, and loading the results into PySlice Trajectories ready for supercell building.
---

# Structure Retrieval

Get a starting structure from a database instead of hand-carrying CIFs:
search Materials Project (MP) or the Crystallography Open Database (COD) by
formula/elements, download the entry as a CIF, and load it as a PySlice
`Trajectory` — all via `pyslice.io.databases` (stdlib-only, no pymatgen).

## Read First

- `skills/pyslice/SKILL.md` — what happens after loading (build → simulate)
- `src/pyslice/io/databases.py` docstrings for the full parameter surface

## Core Rules

- **Providers:** `"mp"` = Materials Project (DFT-relaxed computed
  structures; **free API key required** from
  https://next-gen.materialsproject.org/api, read from `PYSLICE_MP_API_KEY`
  or `MP_API_KEY`, or passed as `api_key=`). `"cod"` = COD (experimental
  published structures; **keyless**). Prefer MP for ground-state computed
  cells, COD for a specific published refinement — and COD whenever no MP
  key is available.
- **Search before fetch:** `search_structures` returns entries whose `id`
  (`"mp-149"`, `"1010939"`) feeds `fetch_cif` /
  `load_structure_from_database`. MP results are sorted most-stable first
  (`energy_above_hull_eV`); prefer hull ≈ 0 unless a metastable phase is
  wanted.
- **MP CIFs are rendered as P1** (all sites explicit) from the structure
  JSON — no symmetry is lost for simulation purposes. COD CIFs are the
  published files verbatim.
- **Formulas:** simple element+count strings (`"SiO2"`, `"BaTiO3"`);
  parentheses are not supported — use `elements=[...]` instead. COD element
  search is strict (exactly those elements). MP accepts wildcards
  (`"Si*"`).
- **Partial occupancies** (common in COD refinements) load through ASE but
  are not physical for a multislice run — pick an ordered entry or build an
  ordered approximation before simulating.
- **After loading, verify** the structure before spending compute: check
  `n_atoms`, `atom_types`, `box_matrix`/`extent`, and render a projected
  potential (`pyslice_preview_potential`) or `traj.plot()`.
- Downloaded CIFs land where you point `output_dir` (the MCP server uses
  `<workspace>/structures/`); `Loader` caches `.npy` siblings next to them.

## Natural Language Task Routing

- "find/get/download a structure for X" → `search_structures` (COD if no MP
  key) → present top matches (id, formula, spacegroup, stability) → fetch
  the chosen one.
- "the Materials Project entry mp-NNN" → `fetch_cif("mp", "mp-NNN")` /
  `load_structure_from_database`.
- "an experimental structure of X" → provider `cod`.
- "the most stable phase of X" → MP search, take hull ≈ 0.
- "a specific polymorph/spacegroup" → search, filter entries by
  `spacegroup`, confirm with the user when ambiguous.
- Ask a concise clarification only when several materially different
  matches exist (polymorphs, hydrates, doped variants) and the choice
  changes the simulation.

## Python API Examples For Agents

```python
from pyslice.io.databases import (
    search_structures, fetch_cif, load_structure_from_database,
)

# Keyless COD search by elements (strict match), then load in one step
entries = search_structures("cod", elements=["Ti", "O"], limit=10)
traj = load_structure_from_database("cod", entries[0]["id"], output_dir="structures")

# Materials Project: most stable SiC, key from env
entries = search_structures("mp", formula="SiC", limit=5)
cif_path = fetch_cif("mp", entries[0]["id"], output_dir="structures")

from pyslice.io.loader import Loader
traj = Loader(filename=str(cif_path)).load()
traj = traj.tile_positions((4, 4, 2))        # then build as usual
```

Errors are actionable: a missing MP key raises `DatabaseError` naming the
env vars and signup URL; network failures name the unreachable host.

## MCP Tool Use

`pyslice_search_structures` (provider, formula/elements) →
`pyslice_fetch_structure` (downloads into the workspace and, by default,
returns a loaded Trajectory handle) → `pyslice_build_slab` (oriented,
exactly periodic samples) or `pyslice_transform_trajectory` → simulation.
Verify with `pyslice_describe_handle` and `pyslice_preview_potential`
before running. The fetch records the database origin, and
`pyslice_export_sea_file` later stores it under the Material entry's
`Metadata.Database` in `SEAFile.Materials`. Materials entries conform to
sea-eco's `signal-containers` schema, `atomic-structure` profile v1
(`atoms`/`cell` SignalSets inside a marked `SignalCollection`); see
`docs/conformance/signal-containers.md`.

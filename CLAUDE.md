# PySlice Agent Notes

PySlice (import `pyslice`) is the pySEA ecosystem's GPU-accelerated
multislice simulation engine: TEM/STEM/4D-STEM imaging and diffraction, plus
vibrational EELS via the TACAW method computed from MD trajectories with
universal ML potentials (ORB/FAIRChem). Simulation outputs subclass sea-eco
`Signal` and export calibrated `.sea` files via `.to_sea()`.

## Collaboration protocol

Two contributors (Ondrej and Eric), each working with Claude. PySlice does
not yet carry its own `ai_wiki/` slice, so contributor notes for PySlice work
live at the ecosystem level:
`sea-ecosystem/src/pySEA/ai_wiki/ecosystem/notes/<ondrej|eric>/`. Follow the
protocol in `sea-ecosystem/CLAUDE.md`: pull → read the other contributor's
LOG → write a TODO note with a meaningful kebab-case branch name →
`[Under Construction]` LOG entry → small commits, push and pull between them.

## Install

```bash
pip install -e .                 # core: ase, numpy, scipy, matplotlib, tqdm, h5py, ovito
pip install -e ".[fast]"         # + torch GPU acceleration
pip install -e ".[md]"           # + orb-models ML potentials (pins Python 3.12)
pip install -e ".[mcp]"          # + MCP server deps (mcp, pydantic)
pip install -e ".[sea]"          # + sea-eco for .to_sea() export
```

Note: **OVITO is not on normal PyPI** (`--find-links https://www.ovito.org/pip/`);
it is only needed for the XYZ/LAMMPS/ASE-traj loading path — CIF loading and
everything downstream work without it. `requires-python >= 3.12` (driven by
the MD stack).

## Pipeline (mental model)

```
Loader → Trajectory → (optional MD: ORBMDCalculator/FAIRChemMDCalculator)
       → MultisliceCalculator.setup(...).run() → WFData
       → {HAADFData, TACAWData} → .to_sea()
```

## Repo map

- `src/pyslice/io/loader.py` — `Loader`: CIF (via ASE), XYZ/LAMMPS/traj (via
  OVITO), in-memory ASE `Atoms`; caches `.npy` siblings next to sources.
- `src/pyslice/io/databases.py` — Materials Project + COD search →
  CIF retrieval (stdlib urllib only; MP key via `PYSLICE_MP_API_KEY`).
- `src/pyslice/io/build.py` — ASE-backed `build_slab` (exactly periodic
  zone-axis slabs, orthogonalized cells, vacuum), `first_bragg_g`,
  `atom_symbols`/`trajectory_to_ase` (canonical str/int normalization).
- `src/pyslice/multislice/trajectory.py` — `Trajectory` (positions,
  velocities, atom_types, box_matrix, timestep in **ps**) + transforms
  (tile, rotate_to, tilt, slice, frozen-phonon displacements).
- `src/pyslice/multislice/calculators.py` — `MultisliceCalculator`
  (setup/run), `SEDCalculator`.
- `src/pyslice/multislice/multislice.py` — `Probe`, `PrismProbe`,
  `Propagate`, aberrations (abTEM Cnm convention).
- `src/pyslice/multislice/potentials.py` — Kirkland potentials,
  `grid_from_trajectory`.
- `src/pyslice/md/molecular_dynamics.py` — ASE MD with ML potentials
  (timestep in **fs**; equilibration → production; no cancel hook).
- `src/pyslice/postprocessing/{wf_data,haadf_data,tacaw_data}.py` — result
  Signals with `.to_sea()`/`.load()`.
- `src/pyslice/data/seashell.py` — **the only module that imports sea-eco**.
  The resolution layer: real containers or dummy stand-ins, `resolve()`,
  `register_resolver()`, `adopt_signal_state()`.
- `src/pyslice/data/atomic_structure.py` — Trajectory → `atomic-structure`
  profile collection (the registered `Trajectory` resolver).
- `src/pyslice/data/pyslice_serial.py` — `PySliceSerial` HDF5 mixin; gets its
  sea-eco names from `seashell`, never from `pySEA` directly.
- `src/pyslice/backend.py` — numpy/torch backend seam (`make_backend`,
  `PYSLICE_DEVICE`, `PYSLICE_BACKEND=numpy`).
- `src/pyslice/mcp/{service,server}.py` — MCP surface (`python -m pyslice.mcp`).
- `skills/` — agent skills; `.claude/agents/` — subagents; `tests/` —
  numbered pytest files; `examples/` — runnable technique scripts.
- `docs/` — Sphinx tree: `guides/`, `sea-weeds/` (developer docs),
  `ai_tools/`, `conformance/`, `api_reference.rst`.

## AI tools

- **MCP:** `python -m pyslice.mcp [--workspace DIR]` — `pyslice_*` tools;
  call `pyslice_get_conventions` first. Thin server over
  `pyslice.mcp.service.PySliceService` (mirrors `pySEA.sea_eco.mcp`).
  Prompted simulations go through `pyslice_plan_simulation` (parameter
  table with supplied/derived/default origins + open questions → confirm →
  execute → `pyslice_render_signal` visual → `pyslice_export_sea_file`).
- **Skills:** `skills/pyslice` (umbrella), `multislice-imaging`,
  `tacaw-phonons`, `md-setup`, `simulation-parameter-selection` (the
  parameter physics — single source of truth), `structure-retrieval`
  (Materials Project / COD → CIF).
- **Subagents:** `.claude/agents/` — `multislice-parameter-advisor`,
  `simulation-runner`, `structure-builder`, `experiment-matcher`,
  `tacaw-analyst`, plus dev-time `pyslice-integration-reviewer` and
  `sea-data-curator`.

Keep the layering: MCP tools are the hands, skills are the know-how,
subagents are the workers. The parameter physics lives ONCE, in
`skills/simulation-parameter-selection/SKILL.md`, implemented by
`pyslice_suggest_parameters`; reference it, never duplicate it.

## Core rules

- **Units:** Å for lengths, mrad for angles (aperture/detectors), eV for
  voltage, **ps** for trajectory frame spacing, **fs** for the MD
  integration timestep, THz for TACAW frequencies, 1/Å for k-axes.
- **`aperture=0` = parallel beam (TEM); `>0` = convergent STEM probe.**
- **Atom identities:** symbol strings and atomic numbers are supported by
  `to_ase(frame=...)` and `Potential._resolve_z`. OVITO particle-type IDs
  are never assumed to be atomic numbers: resolve embedded element names
  or provide an exact `atom_mapping`.
- **MD/TACAW contracts:** NPT is equilibration-only and needs a positive
  material-specific `bulk_modulus_GPa`; production is fixed-cell NVT/NVE.
  TACAW needs chronological, uniformly spaced frames, not independent
  frozen-phonon samples. Gain/loss folding defaults to `fold=False`, even
  with Bose correction. COM diagnostics/correction require unwrapped
  coordinates; see `docs/user-guide/md-to-tacaw.md` and `tacaw-analysis.md`.
- **Structure edits are Trajectory transforms** (tile/rotate/tilt/crop/
  frozen-phonon), never `setup()` arguments; transforms return new objects.
- **Oriented slabs come from `build_slab`** (exactly periodic; ASE surface
  + orthogonalization); rotate-and-carve leaves non-periodic edges and is
  the fallback only. The multislice grid reads only the box **diagonal**.
- **SEAFile packaging contract:** results as plain calibrated `Signal`s in
  `Simulations` (readable without PySlice); Material (unit cell,
  `Metadata.Database`) and Sample (built structure, `Metadata.build`) in
  `Materials`, Sample's SEAID rooted at the Material's. Link provenance on
  the collection's own datasets — adding deep-copies and re-mints SEAIDs.
- **Materials entries follow sea-eco's `signal-containers` schema**,
  `atomic-structure` profile **v1** — a prescriptive contract, not a
  convention. Read
  `sea-eco/src/pySEA/ai_wiki/sea_eco/schema/signal-containers/intents.md`
  (CONT-6) before touching it, and keep
  `docs/conformance/signal-containers.md` in sync. Shape: a marked
  `SignalCollection` with an `atoms` SignalSet (`position` float
  `(*context, atom, coordinate)`, `element` string `(atom,)`,
  `clamp_boundary_condition` bool `(atom,)`) and a `cell` SignalSet
  (`cell` float `(*context, cell_vector, coordinate)`,
  `periodic_boundary_condition` bool). `coordinate` is categorical x/y/z,
  `cell_vector` is a/b/c, and **value units live on scalar
  `SignalQuantities`, never on the component axes**. Build it through
  `mark_atomic_structure` so it is validated on construction.
- **Every frame is propagated** — frame count = frozen-phonon configs or MD
  snapshots.
- **Blocking compute:** `MultisliceCalculator.run()` and MD `run()` block
  with tqdm and have no cancel hook; native/GPU code can hard-crash the
  process. Run them behind a subprocess/job boundary in host applications.
- **Results ARE sea-eco objects; resolution is implicit.** With sea-eco
  importable, `WFData`/`HAADFData`/`TACAWData` are first-class `Signal`s
  (name, `Provenance`, `Analysis`, dimension signature — via
  `adopt_signal_state` in each constructor) and `Trajectory.sea` resolves to
  an `atomic-structure` `SignalCollection`. Never write a conversion step;
  call `seashell.resolve(obj)` or add a `register_resolver` entry.
- **`data/seashell.py` is the ONLY sea-eco import site** (rayTEM's
  `seashells.py` pattern). Everything else imports names from it, so PySlice
  imports and simulates unchanged when sea-eco is absent — `tests/32` enforces
  this with a source scan.
- **Keep the `.sea` bridge stable:** `pyslice_serial.py` and the
  `_sea_config` dicts define the on-disk format; verify against *current*
  sea-eco (`Dimensions(..., det_dimensions=...)`) and round-trip test any
  change.
- **Caches:** wavefunctions under `psi_data/<backend>_<hash>/`, loader
  arrays as `.npy` siblings of the source file. Redirect to scratch dirs in
  managed environments.
- **Backend seam:** get array ops from `pyslice.backend.make_backend()`;
  never import torch directly in simulation code.

## Testing

```bash
python -m pytest -q tests/test_*.py tests/04_adf_calculator.py tests/18_cache_output_api.py tests/25_review_regressions.py tests/27_mcp_service.py tests/28_databases.py tests/29_slab_builder.py tests/30_plan_simulation.py tests/31_sea_file_export.py tests/32_seashell_resolution.py
```

The numbered regression files must be named explicitly for pytest discovery.
Some other numbered files are standalone simulations requiring datasets, not
unit tests; do not collect them indiscriminately. In the listed suites, `tests/27_mcp_service.py` and `tests/28_databases.py` run
CPU-only with mocked HTTP (set `PYSLICE_DB_LIVE_TESTS=1` for the live COD
round-trip). `tests/30_plan_simulation.py` encodes the reference prompted-
simulation requests (40 nm [110] diamond 4D-STEM with 10 nm slices;
graphene dispersion to ±2g) as planning tests — extend it when adding
intake rules. `tests/16_sea_eco_integration.py` is stale (imports names
`pyslice.data` no longer exports); the live bridge coverage is in
`tests/27_mcp_service.py` and `tests/31_sea_file_export.py`.

## Documentation

```bash
python -m pip install -r docs/requirements.txt
PYSLICE_DOCS_OFFLINE=1 python -m sphinx -b html -W docs docs/_build/html
```

Five sections in this order: Guides, Example Notebooks, AI Tools, Into the
SEA-weeds, API Reference. The API Reference is `autodoc` over `src/pyslice`
with the heavy optional deps in `autodoc_mock_imports`, so **building the HTML
*is* the API regeneration** — there is no stub step and no committed generated
`.rst`. Run it before calling any documentation task done.

`PYSLICE_DOCS_OFFLINE=1` drops the intersphinx mappings; without it `-W` fails
on unreachable inventories, which no `suppress_warnings` subtype silences.
Build output lands in `docs/_build/` and is gitignored — never commit it.

Every substantive `sea-weeds/` feature page carries a **Provenance and
verification** subsection linking source entry points, schemas, tests, guides,
and AI-tool artifacts. `sea-weeds/resolution_layer.md` is the reference page
for the seashell pattern; keep it in sync with `data/seashell.py`.

## Style rules

- NumPy docstring style for all new Python callables (public, private,
  dunder, properties). Documentation quality is part of code correctness.
- Prefer explicit type annotations; `Literal` over `str` for fixed value
  sets; `Sequence` over concrete containers in signatures.
- No new dataclasses (existing `Trajectory` predates the rule; do not copy
  the pattern into new code or other repos).
- Errors must be actionable: name the fix, the env var, or the install
  command.
- Prefer additive changes; `.sea` files and the packed cache formats are
  compatibility surfaces.

## Ecosystem context

For the full picture of sibling repos and connections, read
`sea-ecosystem/src/pySEA/ai_wiki/ecosystem/CLAUDE.md`. PySlice outputs feed
sea-eco Signals (`.to_sea()`), the planned PoseiTEM Simulation capability,
and PoseidonWeb workspaces. This repo is not yet registered in the ecosystem
wiki (`ai_wiki` slice pending — it requires adopting the shared layout).

## Agent file policy

`CLAUDE.md` is canonical; keep `AGENTS.md` a byte-identical copy
(`cp CLAUDE.md AGENTS.md`). Never edit `AGENTS.md` directly.

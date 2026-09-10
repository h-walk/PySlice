# MCP

`python -m pyslice.mcp` exposes PySlice's pipeline as 24 `pyslice_*` tools.
**Call `pyslice_get_conventions` first** — it states the unit system, the
workflow, and the gotchas that cause most mistakes.

For the architecture and its boundaries, see
{doc}`../sea-weeds/agentic_surface`.

## Scope and authority

The server drives simulations on the machine it runs on. It reads and writes
files in its workspace, fetches from two public structure databases, and runs
compute that can occupy a GPU for a long time. It does not authenticate
callers, so give it a workspace you are content for a client to write into.

- **Reads:** structure files you name, plus workspace artifacts.
- **Writes:** only inside the workspace; `_workspace_path` rejects escapes.
- **Network:** Materials Project (needs your key) and COD (keyless), on
  explicit search/fetch calls only.
- **Compute:** `pyslice_run_multislice` and `pyslice_run_md` block and cannot
  be cancelled. Check the estimate from `pyslice_setup_multislice` first.

## Safety

| Risk | Mitigation |
|---|---|
| Runaway memory | `pyslice_setup_multislice` reports grid size and a wavefunction-memory estimate before you run |
| Uncancellable jobs | Blocking is documented; host applications should run these behind a subprocess boundary |
| Path traversal | All artifact paths workspace-relative and validated |
| Silent parameter typos | Pydantic inputs use `extra="forbid"` |
| Lost work | Handles are in-memory only; export `.sea` to persist |
| Leaked API key | Pass via `PYSLICE_MP_API_KEY`, not in tool arguments |

## The tools

**Orientation** — `pyslice_get_conventions`, `pyslice_get_workspace`,
`pyslice_list_handles`, `pyslice_describe_handle`

**Structures** — `pyslice_search_structures`, `pyslice_fetch_structure`,
`pyslice_load_structure`, `pyslice_build_slab`,
`pyslice_transform_trajectory`, `pyslice_preview_potential`

**Planning** — `pyslice_plan_simulation` (full requests),
`pyslice_suggest_parameters` (one goal)

**Running** — `pyslice_run_md`, `pyslice_setup_multislice`,
`pyslice_run_multislice`

**Post-processing** — `pyslice_compute_haadf`, `pyslice_compute_tacaw`,
`pyslice_tacaw_spectrum`, `pyslice_spectrum_image`, `pyslice_dispersion`

**Output** — `pyslice_render_signal`, `pyslice_export_sea`,
`pyslice_export_sea_file`

Every tool takes `response_format` (`markdown` or `json`).

`pyslice_compute_tacaw` requires chronological, uniformly spaced MD frames;
independent frozen-phonon snapshots are not a time series. Gain/loss folding
is opt-in with `fold=true` and defaults to false even with `apply_bose=true`.
See {doc}`../user-guide/tacaw-analysis` for the normalization and symmetry
assumptions. `pyslice_run_md` permits NPT for equilibration only, requires a
material-specific `bulk_modulus_GPa` for it, and records fixed-cell NVT or NVE
production trajectories.

## A worked invocation

```
pyslice_get_conventions
pyslice_search_structures   provider=cod, formula=SiC
pyslice_fetch_structure     provider=cod, entry_id=<id>          → Trajectory handle
pyslice_plan_simulation     technique=haadf, structure_handle=…  → table + open questions
   ── present the plan, get confirmation ──
pyslice_build_slab          indices=[1,1,0], thickness_A=400
pyslice_transform_trajectory op=frozen_phonon, n=12, sigma_A=0.07
pyslice_setup_multislice    …                                    → check grid + GiB
pyslice_run_multislice                                           → WFData (+ HAADF)
pyslice_compute_haadf       inner_mrad=60, outer_mrad=200
pyslice_render_signal                                            → PNG to look at
pyslice_export_sea_file     + sample_handle, material_handle     → .sea with provenance
```

## Recovery

| Situation | What to do |
|---|---|
| `Unknown handle` | `pyslice_list_handles`; handles do not survive a restart |
| Wrong handle type | The error names actual vs expected — re-read the workflow order |
| Estimate too large | Coarsen `sampling_A`, crop with `max_kx`/`max_ky`, or set `return_layers=null` for HAADF-only |
| `requires sea-eco` | `pip install -e ".[sea]"` |
| MD unavailable | `pip install -e ".[md]"` (Python 3.12) |
| Database unreachable | The error names the host; COD needs no key, MP does |
| A run failed midway | Caches under `psi_data/` make a rerun cheaper; pass `force_rerun` to bypass them |

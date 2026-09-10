# The agentic surface

**What exists:** an in-process MCP server (`src/pyslice/mcp/`), six skills
(`skills/`), and seven subagents (`.claude/agents/`).

**Why three layers instead of one:** they answer different questions.
MCP tools are *hands* — atomic capabilities an LLM client can invoke. Skills
are *know-how* — when to use which capability, and the physics behind the
choice. Subagents are *workers* — multi-step roles with their own context.
Collapsing them duplicates the physics into every consumer, which is exactly
what the layering exists to prevent.

## Layering rules

1. **The parameter physics lives once**, in
   `skills/simulation-parameter-selection/SKILL.md`. `pyslice_suggest_parameters`
   and `pyslice_plan_simulation` implement those rules; the
   `multislice-parameter-advisor` subagent applies them. Nothing restates them.
2. **`server.py` holds no logic.** It registers tools and formats responses;
   every tool body is one call into `service.py`.
3. **`service.py` holds no sea-eco knowledge.** It imports SEA names from
   {doc}`the resolution layer <resolution_layer>` and delegates structure
   conversion to it. This was not always true — the atomic-structure builder
   used to live in the service, which meant only MCP callers benefited.
4. **Skills reference, never duplicate.** A skill that needs a rule links to
   the skill that owns it.

## The service layer

`PySliceService` is stateful by design. Multi-step agent work — search a
database, load a structure, build a slab, plan, run, post-process, export —
needs to pass live objects between calls without reserializing multi-gigabyte
wavefunctions. So the service keeps a **handle registry**: `Type:label`
strings mapping to live Python objects.

Consequences a caller must know, and which `pyslice_get_conventions` states:

- Handles are **live in-memory objects**. They do not survive a server
  restart. Persist with `pyslice_export_sea` / `pyslice_export_sea_file`.
- Long runs **block by contract**. `pyslice_run_multislice` and
  `pyslice_run_md` have no cancel hook, because the underlying PySlice calls
  have none. `pyslice_setup_multislice` therefore returns the grid size and a
  wavefunction-memory estimate *before* the expensive call, and the agent is
  instructed to check it.
- Artifacts are **workspace-relative**, and `_workspace_path` rejects any path
  escaping the workspace.

Inputs are pydantic models with `extra="forbid"`, so a misspelled parameter is
an error rather than a silent default.

## The planner

`pyslice_plan_simulation` is the intake step: a structured request in, a
confirmable plan out. Its contract is that **every value carries its origin**
— `supplied`, `derived`, or `default` — with a one-line justification, and
that anything guessed which materially changes the result also appears under
`open_questions`.

That structure is the point. An agent can present a table and ask, rather than
silently choosing twelve numbers. The two reference requests (40 nm [110]
diamond 4D-STEM with 10 nm slices; graphene dispersion to ±2g) are pinned as
tests in `tests/30_plan_simulation.py`, so the documented behaviour is the
enforced behaviour.

**Extension point:** adding an intake rule means editing
`skills/simulation-parameter-selection/SKILL.md`, implementing it in
`plan_simulation`, and adding its assertion to `tests/30`. Extend that test
when adding a rule — it is the record of what the planner promises.

## Compatibility notes

The server imports `FastMCP` and falls back to `MCPServer`, so it works with
mcp SDK 1.x and 2.x (2.0 renamed the class). The `mcp` extra nevertheless pins
`mcp>=1.0.0,<2.0.0` to match sea-eco's pin after the breaking 2.0 release —
the compatibility import is a safety net, not a licence to drift.

## Boundaries

- **PySlice's MCP does not own physics.** It exposes PySlice's API. A tool that
  computes something PySlice cannot belongs in PySlice first.
- **No GUI, no job queue, no scheduler.** Blocking is honest here; a host
  application is expected to put these calls behind a subprocess boundary
  (native/GPU code can hard-crash the process, and there is no cancel hook).
- **Subagent definitions are prose contracts**, not code. They are read by an
  agent client; PySlice does not execute them.

## Failure modes

| Failure | Symptom | Guard |
|---|---|---|
| Unknown handle | `KeyError` listing known handles | `_get` |
| Wrong handle type | `TypeError` naming actual vs expected | `_require_type` |
| Artifact path escapes the workspace | `ValueError` | `_workspace_path` |
| Run too large for the machine | Reported *before* running, as an estimate | `setup_multislice` returns grid + GiB |
| TACAW on a single frame | `ValueError` naming the multi-frame requirement | `compute_tacaw` |
| sea-eco absent | `ImportError` naming the extra | the resolution layer |
| MD extra absent | `RuntimeError` naming the install command | `run_md` |

## Limitations

- **Live database calls are unverified in CI.** Sandboxed environments block
  egress to `api.materialsproject.org` and `crystallography.net`; tests mock
  HTTP and assert URL construction. `PYSLICE_DB_LIVE_TESTS=1` runs the live
  COD round trip where egress exists.
- **GPU, MD, and OVITO paths are wired but not exercised** by the test suite —
  no CUDA, no ML weights, no OVITO in a bare environment.
- The memory estimate is arithmetic, not measurement.

## Provenance and verification

| Aspect | Where |
|---|---|
| Implementation | `src/pyslice/mcp/service.py` (logic), `src/pyslice/mcp/server.py` (registration), `src/pyslice/mcp/__main__.py` (entry point) |
| Pattern mirrored | `pySEA.sea_eco.mcp.{service,server}` |
| Know-how artifacts | `skills/pyslice`, `skills/multislice-imaging`, `skills/tacaw-phonons`, `skills/md-setup`, `skills/simulation-parameter-selection`, `skills/structure-retrieval` |
| Worker artifacts | `.claude/agents/{multislice-parameter-advisor,simulation-runner,structure-builder,experiment-matcher,tacaw-analyst,pyslice-integration-reviewer,sea-data-curator}.md` |
| Focused tests | `tests/27_mcp_service.py` (registry, advisor rules, workspace guard, end-to-end run); `tests/30_plan_simulation.py` (the reference requests) |
| User guides | {doc}`../guides/prompted_simulations`, {doc}`../ai_tools/mcp` |
| API reference | {doc}`../api_reference` — `pyslice.mcp.service` |

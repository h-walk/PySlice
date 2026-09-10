# Subagents

Seven subagents in `.claude/agents/`. They are the workers: multi-step roles
with their own context, which use the skills and MCP tools rather than
re-implementing physics.

## Product roles

| Subagent | Use it for | Tools |
|---|---|---|
| `multislice-parameter-advisor` | A justified parameter set for a structure and goal | Read-only (`Read`, `Grep`, `Glob`) — it advises, never runs |
| `simulation-runner` | Executing a full job: structure → build → (MD) → multislice → post-process → render → export | All |
| `structure-builder` | Producing a simulation-ready structure to spec, and verifying it | All |
| `tacaw-analyst` | Phonon work: sampling plan for a THz window, then TACAW extraction and interpretation | All |
| `experiment-matcher` | Configuring a simulation to match an experimental `.sea`, then comparing | All |

## Dev-time roles

| Subagent | Use it for |
|---|---|
| `pyslice-integration-reviewer` | Adversarial review of PySlice changes: units, the `atom_types` gotcha, serialization stability, blocking-compute boundaries, layer discipline. Read-only. |
| `sea-data-curator` | Validating that outputs carry correct Dimensions, units, metadata, and provenance before they land. Read-only. |

## How they compose

```
"simulate X"        → simulation-runner
                         ├── structure-builder            (get and shape the sample)
                         ├── multislice-parameter-advisor (decide the numbers)
                         └── tacaw-analyst                (if phonons)

"match this data"   → experiment-matcher → multislice-parameter-advisor

reviewing a change  → pyslice-integration-reviewer, then sea-data-curator
                      on any output it produces
```

The advisor is deliberately **read-only**: parameter advice should be
obtainable without the risk of something starting a GPU job. `simulation-runner`
is the only role that runs long compute, and its instructions require checking
the memory estimate before launching and reporting honestly when a run is
too large.

## Boundaries

These files are **prose contracts read by an agent client**. PySlice does not
execute them, and nothing in `src/pyslice` depends on them. They encode
working practice — reading order, what to verify, when to ask — and are
expected to be edited as that practice changes.

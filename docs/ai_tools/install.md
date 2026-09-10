# Install and discovery

PySlice ships three AI-facing surfaces: an MCP server, skills, and subagents.
Only the MCP server needs installing; the other two are files an agent client
discovers in the repository.

## MCP server

```bash
pip install -e ".[mcp]"        # mcp>=1.0.0,<2.0.0, pydantic>=2.0
pip install -e ".[mcp,sea]"    # recommended: adds sea-eco, so results are SEA objects
```

Verify:

```bash
python -m pyslice.mcp --help
```

Run it (stdio transport):

```bash
python -m pyslice.mcp --workspace /path/to/scratch
```

The workspace is where generated artifacts land — fetched CIFs under
`structures/`, `.sea` exports, PNG and HTML renders. It defaults to
`PYSLICE_MCP_WORKSPACE` or the current directory. Every artifact path a tool
accepts is workspace-relative and validated, so a tool cannot write outside it.

Register it with your client the way you would any stdio MCP server, e.g.:

```json
{
  "mcpServers": {
    "pyslice": {
      "command": "python",
      "args": ["-m", "pyslice.mcp", "--workspace", "/path/to/scratch"]
    }
  }
}
```

Confirm discovery by calling **`pyslice_get_conventions`** — it returns the
unit system, the canonical workflow, and PySlice's gotchas, and is the
intended first call in any zero-context session.

## Skills

`skills/<name>/SKILL.md`, discovered from the repository. No install step.

| Skill | Covers |
|---|---|
| `pyslice` | Umbrella: routing, units, the prompted-simulation intake workflow |
| `simulation-parameter-selection` | The parameter physics — the single source of truth |
| `multislice-imaging` | HAADF/ADF/BF, diffraction, CBED, 4D-STEM |
| `tacaw-phonons` | Vibrational EELS, dispersions, spectrum images |
| `md-setup` | ML-potential molecular dynamics |
| `structure-retrieval` | Materials Project / COD → CIF → Trajectory |

## Subagents

`.claude/agents/<name>.md`, discovered from the repository.

| Subagent | Role |
|---|---|
| `multislice-parameter-advisor` | Read-only: structure + goal → justified parameters |
| `simulation-runner` | Drives a full job end to end |
| `structure-builder` | Produces a simulation-ready structure to spec |
| `tacaw-analyst` | Phonon work: sampling plan → TACAW → interpretation |
| `experiment-matcher` | Matches a simulation to an experimental `.sea` |
| `pyslice-integration-reviewer` | Dev-time adversarial review |
| `sea-data-curator` | Dev-time: validates outputs before they land |

## Environment

| Variable | Effect |
|---|---|
| `PYSLICE_MCP_WORKSPACE` | Default artifact workspace |
| `PYSLICE_MP_API_KEY` / `MP_API_KEY` | Materials Project key (COD needs none) |
| `PYSLICE_DEVICE` | Force a compute device |
| `PYSLICE_BACKEND=numpy` | Force the numpy backend |
| `PYSLICE_DB_LIVE_TESTS=1` | Enable the live COD round-trip test |

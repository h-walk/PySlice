# Skills

Six skills in `skills/`. They carry the know-how: when to use which
capability, and the physics behind the choice.

## The one that matters most

**`simulation-parameter-selection`** is the single source of truth for the
parameter physics — 14 rules covering wavelength, antialiasing-limited
sampling, slice thickness, k-space resolution, probe wraparound, probe step,
aperture, frame counts, memory, detector angles, first-Bragg probe steps,
k-ranges as multiples of g, depth-interval slice output, and scan extent.

`pyslice_suggest_parameters` and `pyslice_plan_simulation` implement these
rules; the `multislice-parameter-advisor` subagent applies them. **Nothing
restates them.** If a rule is wrong, change it there — the change propagates.

## Triggers and non-triggers

| Skill | Triggers on | Does *not* cover |
|---|---|---|
| `pyslice` | "simulate", "multislice", "TACAW", any prompted simulation | The physics details — it routes to the others |
| `simulation-parameter-selection` | "what sampling", "how many frames", "how big a supercell", "will this fit in memory" | Running anything |
| `multislice-imaging` | HAADF/ADF/BF, diffraction, CBED, 4D-STEM, defocus/aberrations | Phonons |
| `tacaw-phonons` | Vibrational EELS, phonon dispersion, spectrum images, Bose correction | Producing the MD trajectory |
| `md-setup` | "thermalize", "run MD", ensembles, ML-potential choice | Spectroscopy analysis |
| `structure-retrieval` | "find a structure for X", Materials Project, COD, "get a CIF" | Building supercells or slabs |

## Handoffs

The umbrella `pyslice` skill routes to a specific skill, then to MCP tools:

```
prompted simulation
   → pyslice (intake workflow)
        → structure-retrieval        (get the structure)
        → simulation-parameter-selection  (choose parameters)
        → multislice-imaging | tacaw-phonons  (technique specifics)
             → md-setup              (only if phonons/thermal dynamics)
        → MCP tools                  (execute)
```

The intake workflow — decode → plan → **confirm** → execute → visualize →
persist — is documented in the `pyslice` skill and mirrored by the
`simulation-runner` subagent. The confirmation step is not optional: a
prompted simulation should show its parameter table and open questions before
spending compute. See {doc}`../guides/prompted_simulations`.

## Invocation

Skills are read by an agent client from the repository; there is no API. In a
PySlice checkout an agent should read `skills/pyslice/SKILL.md` first, since
it routes everything else.

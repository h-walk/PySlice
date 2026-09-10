# Planning a simulation from a description

You usually know what you want in words — *"an atomic-resolution 4D-STEM
simulation of a 40 nm thick, 110-oriented diamond sample with slices every
10 nm"* — not as twelve numeric arguments. PySlice can turn that into a fully
specified parameter set, show you which values you supplied and which it chose,
and wait for your confirmation before running anything expensive.

## The workflow

1. **Describe** the simulation.
2. **Plan** — `pyslice_plan_simulation` fills in everything you did not say.
3. **Confirm** — you read a parameter table and a list of open questions.
4. **Execute** — build the sample, add thermal frames, propagate, post-process.
5. **Look at it** — a rendered visual, not just file paths.
6. **Persist** — a `.sea` file with the results and the material provenance.

Steps 2–6 are MCP tools; see {doc}`../ai_tools/mcp` for the full catalog. The
`pyslice` skill routes a natural-language request through this sequence.

## What planning returns

Every parameter comes back tagged with where it came from:

| Origin | Meaning |
|---|---|
| `supplied` | You said it. Passed through verbatim. |
| `derived` | Forced by physics or by something you did say. |
| `default` | A reasonable choice PySlice made — and flags. |

Each carries a one-line justification, and anything guessed that materially
changes the result also appears under **open questions**. For the 4D-STEM
request above you would see, among others:

- `aperture_mrad = 30` *(default)* — an atomic-size probe, typical
  aberration-corrected STEM.
- `scan_step_A = d₁/20` *(derived)* — ten times the Nyquist rate of the first
  Bragg spacing, so atoms are clearly resolved.
- `return_layers` *(supplied)* — four stored depth slices, from your "every
  10 nm" through 40 nm.
- `thermal = frozen_phonon` *(default)* — thermal diffuse scattering matters
  for quantitative STEM contrast; you asked for no phonon dynamics.
- **Open questions:** lateral sample size, scan extent, and detector angles —
  none were specified, and each changes the answer.

A second example: *"a vibrational EELS dispersion of graphene out to ±2g"*
plans a **parallel beam** (`derived` — momentum resolution requires it),
`max_k = 2g₁` exactly, an NVE molecular-dynamics run of 200 frames at 16.7 fs
spacing (from the frequency window you implied), and a Γ–M–K–Γ path in 1/Å.

Both of these are pinned as tests in `tests/30_plan_simulation.py`, so the
behaviour described here is the behaviour you get.

## The physics is one document

Every rule the planner applies — sampling from the antialiasing band limit,
slice thickness dividing the cell evenly, supercell size from probe wraparound
and k-resolution, probe step from image Nyquist, frame counts from the target
frequency window — lives in one place:
`skills/simulation-parameter-selection/SKILL.md`.

If a default disagrees with how you would set the experiment up, change the
rule there and its assertion in the tests. Do not special-case it downstream;
the planner, the `pyslice_suggest_parameters` tool, and the
`multislice-parameter-advisor` subagent all read that same document.

## Doing it by hand

Planning is a convenience, not a gate. The underlying calls are ordinary
Python, and {doc}`getting_started` plus {doc}`structures` cover them. Ask for
advice without executing anything:

```python
# via MCP: pyslice_suggest_parameters, for one goal on one structure
# in Python: apply the rules from the skill document directly
```

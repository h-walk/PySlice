---
name: pyslice
description: Use when simulating electron microscopy with the PySlice Python package — multislice TEM/STEM/4D-STEM imaging, diffraction, HAADF, or TACAW vibrational-EELS/phonon spectroscopy from MD trajectories. Routes structure loading (CIF/XYZ/LAMMPS, Materials Project/COD retrieval), supercell building, MD with ML potentials, multislice execution, post-processing, and .sea export to the right PySlice APIs and pyslice_* MCP tools.
---

# PySlice

PySlice is the pySEA ecosystem's GPU-accelerated multislice simulation engine:
conventional TEM/STEM/4D-STEM imaging and diffraction, plus vibrational EELS
via TACAW (time autocorrelation of auxiliary wavefunctions) computed from MD
trajectories. Results subclass sea-eco `Signal` and export calibrated `.sea`
files the whole ecosystem reads.

## Read First

In a PySlice checkout, read:

1. `CLAUDE.md`
2. `README.md`
3. `examples/` — runnable end-to-end scripts per technique

Sub-skills (prefer the specific one when the task matches):

- `multislice-imaging` — HAADF/ADF/BF, diffraction, 4D-STEM setup
- `tacaw-phonons` — phonon spectroscopy workflow and frequency windows
- `md-setup` — ML-potential molecular dynamics for thermal trajectories
- `simulation-parameter-selection` — the parameter physics (sampling, slices,
  tiling, probes, frames); the single source of truth other skills reference
- `structure-retrieval` — Materials Project / COD search → CIF → Trajectory

## Core Rules

- **Units:** Angstroms for lengths (sampling, defocus, slice thickness, probe
  positions, sigma), mrad for aperture/detector angles, eV for voltage, THz
  for frequencies. Trajectory frame spacing is **picoseconds**; the MD
  integration timestep is **femtoseconds**. k-axes are 1/Å.
- **`aperture=0` means parallel beam** (TEM/diffraction); `aperture>0` makes a
  convergent STEM probe.
- **Structure edits are Trajectory transforms, not `setup()` arguments** —
  tiling, zone-axis rotation, tilts, cropping, and frozen-phonon displacement
  all live on `Trajectory` and return new objects.
- **The `atom_types` gotcha:** element-symbol strings on the ASE/CIF path,
  integer type-ids on the OVITO path. Pass `atom_mapping={1: "B", 2: "N"}` to
  `Loader` for LAMMPS files; `Trajectory.to_ase()` assumes strings.
- **Every frame is propagated.** Frame count = frozen-phonon configurations
  or MD snapshots; a single frame gives a static pattern with no thermal
  diffuse scattering and no TACAW time axis.
- **Runs are blocking** with tqdm progress and no cancel hook; check the grid
  and memory estimate (from `pyslice_setup_multislice` or the
  `simulation-parameter-selection` rules) before long runs. Caches land under
  `psi_data/` and next to source files as `.npy`.
- **Persist via `.to_sea(path)`** — WFData/HAADFData/TACAWData carry
  calibrated Dimensions and Metadata; do not export raw arrays when a `.sea`
  is possible. Requires sea-eco installed (`pip install 'pyslice[sea]'`).

## Prompted-Simulation Intake Workflow

When a user *prompts for a simulation* ("I want an atomic-resolution 4D-STEM
of 40 nm thick 110 diamond with slices every 10 nm", "vibrational EELS
dispersion of graphene out to ±2g"), follow this sequence:

1. **Decode** the prompt into a structured request: technique, material,
   zone axis, thickness, k-range, slice interval, optics — convert units to
   Å/mrad/eV. Get the unit cell first (`structure-retrieval`).
2. **Plan**: call `pyslice_plan_simulation` with only what was actually
   supplied. It returns a parameter table (each value marked
   supplied/derived/default with justification) and **open questions** for
   the guessed values.
3. **Confirm**: present two summary tables — the parameter table and the
   open questions ("not supplied, assumed X because Y") — and ask for
   confirmation before running anything expensive.
4. **Execute**: `pyslice_build_slab` (build plan) → thermal frames
   (`pyslice_transform_trajectory` frozen_phonon or `pyslice_run_md`) →
   `pyslice_setup_multislice` (check grid/memory) → `pyslice_run_multislice`
   → technique post-processing.
5. **Visualize**: `pyslice_render_signal` (sea-eco plotting, calibrated
   axes) — a realistic image/pattern/map the user can look at, not just
   file paths.
6. **Persist**: `pyslice_export_sea_file` — results in Simulations plus the
   Material (unit cell, database origin) and Sample (built structure,
   `Metadata.build`) in `SEAFile.Materials`, Sample rooted at Material.

## Natural Language Task Routing

Infer PySlice APIs from domain language; prefer PySlice's own functions over
re-implementations:

- "load / open a structure or trajectory" (CIF, XYZ, LAMMPS dump, ASE):
  `Loader(filename=..., atom_mapping=..., timestep=...).load()`.
- "get a structure from Materials Project / COD / a database", "find a CIF
  for X": `structure-retrieval` skill —
  `pyslice.search_structures` / `pyslice.fetch_cif` /
  `pyslice.load_structure_from_database`.
- "N nm thick sample oriented along [hkl]", "a (hkl) slab":
  `pyslice.io.build.build_slab` / `pyslice_build_slab` — exactly periodic
  ASE-built slab (orthogonalized cell, thickness in layers, lateral repeats,
  vacuum). Prefer this over rotate-and-carve; the carved path
  (`rotate_to` + `slice_positions`) leaves non-periodic edges and is the
  fallback for orientations without a small orthogonal periodic cell.
- "make a supercell / tilt the sample / crop":
  `Trajectory.tile_positions`, `.tilt_positions(alpha, beta)` (radians),
  `.slice_positions`.
- "frozen phonon / thermal snapshots without MD":
  `Trajectory.generate_random_displacements(n, sigma, seed)`.
- "run MD / thermalize / phonon trajectory": `md-setup` skill —
  `ORBMDCalculator` / `FAIRChemMDCalculator` (requires `[md]` extra).
- "what parameters should I use": `simulation-parameter-selection` skill or
  the `pyslice_suggest_parameters` MCP tool.
- "simulate an image / diffraction / 4D-STEM / HAADF": `multislice-imaging`
  skill — `MultisliceCalculator().setup(...)` then `.run()`.
- "phonon spectrum / vibrational EELS / dispersion": `tacaw-phonons` skill —
  `TACAWData(wf_data)` then `spectrum` / `spectrum_image` /
  `spectral_diffraction` / `dispersion`.
- "save / export results": `.to_sea("name.sea")`; reload with
  `WFData.load(...)` / `TACAWData.load(...)` / `HAADFData.load(...)`.

Ask a concise clarification only when the request cannot proceed safely:
missing structure source, ambiguous technique (image vs diffraction), or a
run whose memory estimate is clearly beyond the machine.

## Python API Examples For Agents

```python
from pyslice.io.loader import Loader
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.tacaw_data import TACAWData

traj = Loader(filename="structure.cif").load()
traj = traj.tile_positions((4, 4, 1)).generate_random_displacements(12, sigma=0.06, seed=0)

calc = MultisliceCalculator()          # auto-selects CUDA/MPS/CPU
calc.setup(traj, aperture=25.0, voltage_eV=100e3, sampling=0.06,
           slice_thickness=0.5, probe_xs=[0, 1, 2], probe_ys=[0, 1, 2],
           ADF=(60, 200))
wf, haadf = calc.run()                 # ADF set -> (WFData, HAADFData)

tacaw = TACAWData(wf)                  # multi-frame runs only
haadf.to_sea("haadf.sea"); tacaw.to_sea("tacaw.sea")
```

Aberrations are applied to the probe after setup, in the abTEM Cnm
convention (magnitudes in Å, azimuthal pairs as `(mag, angle_rad)`):

```python
calc.setup(traj, aperture=25.0, voltage_eV=100e3)
calc.base_probe.aberrate({"C10": 50.0, "C30": 1e4, "C12": (20.0, 0.3)})
```

## MCP Tool Use

If the PySlice MCP server is available (`python -m pyslice.mcp`), call
`pyslice_get_conventions` first, then drive the same workflow through:

- `pyslice_search_structures` / `pyslice_fetch_structure` / `pyslice_load_structure`
- `pyslice_build_slab` (periodic zone-axis slabs) /
  `pyslice_transform_trajectory` (tile, rotate_to, tilt, frozen_phonon, ...)
- `pyslice_plan_simulation` (full prompted requests — see the intake
  workflow) / `pyslice_suggest_parameters` (single-goal advice)
- `pyslice_setup_multislice` → check grid/memory → `pyslice_run_multislice`
- `pyslice_compute_haadf`, `pyslice_compute_tacaw`, `pyslice_tacaw_spectrum`,
  `pyslice_spectrum_image`, `pyslice_dispersion`, `pyslice_preview_potential`
- `pyslice_render_signal` (sea-eco-rendered visuals)
- `pyslice_export_sea` (single object) / `pyslice_export_sea_file`
  (results + Material/Sample provenance in one SEAFile)

Use direct Python when editing source files, notebooks, tests, or examples.

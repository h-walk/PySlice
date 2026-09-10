---
name: multislice-imaging
description: Use when simulating electron-microscopy images or diffraction with PySlice — HAADF/ADF/BF STEM images, CBED, SAED, 4D-STEM datasets, TEM exit waves, defocus series, or aberrated probes. Covers detector geometry, probe-grid design, on-the-fly ADF, per-slice output, and aberrations.
---

# Multislice Imaging

Configure and run PySlice `MultisliceCalculator` for imaging and diffraction:
parallel-beam TEM/diffraction, convergent-probe STEM (HAADF/ADF/BF), and
4D-STEM (full diffraction pattern per probe position).

## Read First

- `skills/pyslice/SKILL.md` — units, gotchas, workflow
- `skills/simulation-parameter-selection/SKILL.md` — the physics for every
  numeric choice below (do not re-derive it here)
- `examples/haadf_stem.py`, `examples/tem_diffraction.py`,
  `examples/aberrations.py` in a PySlice checkout

## Core Rules

- **Technique = aperture + probe positions + detector:**
  - Diffraction/SAED: `aperture=0`, single (default center) probe; the
    diffraction pattern is `|WFData|²` in k-space.
  - CBED: `aperture>0`, single probe.
  - HAADF/ADF STEM image: `aperture>0`, a probe grid (`probe_xs`/`probe_ys`),
    annular detector via `ADF=(inner_mrad, outer_mrad)` at setup (on-the-fly)
    or `HAADFData(wf).calculateADF(inner, outer)` afterwards.
  - 4D-STEM: `aperture>0`, probe grid, keep `return_layers=-1` wavefunctions;
    the dataset is `(probe, frame, kx, ky, layer)`.
- **On-the-fly ADF (`setup(ADF=(i, o))`) + `return_layers=None`** is the
  memory-cheap HAADF path: `run()` returns `(WFData, HAADFData)` and the WF
  array stays empty. Post-hoc `calculateADF` lets you re-integrate any
  detector geometry from stored wavefunctions instead.
- **Aberrations** go on the probe after setup, abTEM Cnm convention:
  `calc.base_probe.aberrate({"C10": defocus_A, "C30": Cs_A, "C12": (mag_A, angle_rad)})`.
  Plain `defocus=` in `setup()` handles pure defocus.
- **Per-slice output** for thickness series / slices-as-planes:
  `return_layers='all'` (every slice) or a list of slice indices; the last
  WFData axis becomes the z-stack. Projected potential slices come from
  `Potential.build()` instead when no propagation is needed.
- **Thermal realism:** single-frame runs have no thermal diffuse scattering.
  Add 8–16 frozen-phonon frames for quantitative HAADF (see
  `simulation-parameter-selection` rule 8).
- **Detector angles must satisfy** inner < outer, inner ≳ 3× aperture for
  dark field, and sampling must support the outer angle (rule 2).

## Natural Language Task Routing

- "simulate a HAADF/ADF/dark-field image" → probe grid + `ADF=(60, 200)` (or
  the experiment's angles) + frozen-phonon frames.
- "bright field / ABF" → post-hoc `calculateADF` with e.g. (0, aperture) or
  (10, 20) on stored wavefunctions.
- "diffraction pattern / SAED" → `aperture=0`, center probe, plot
  `|wf.data|²` at the exit layer.
- "CBED" → `aperture=experiment`, single probe.
- "4D-STEM / ptychography dataset" → probe grid, keep wavefunctions, export
  `.sea`.
- "defocus series / thickness series" → loop `defocus=` values, or
  `return_layers='all'` for thickness.
- "add aberrations / Cs / astigmatism" → `base_probe.aberrate({...})`.
- "match my experiment" → copy voltage, aperture, detector angles, pixel
  size from the experiment's metadata; see the `experiment-matcher` subagent.

## Python API Examples For Agents

```python
from pyslice.io.loader import Loader
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.haadf_data import HAADFData
import numpy as np

traj = Loader(filename="structure.cif").load()
traj = traj.tile_positions((4, 4, 1)).generate_random_displacements(12, sigma=0.06, seed=0)

scan = np.linspace(0, 10, 25)                 # 10 A scan, ~0.4 A step
calc = MultisliceCalculator()
calc.setup(traj, aperture=25.0, voltage_eV=100e3, sampling=0.06,
           slice_thickness=0.5, probe_xs=scan.tolist(), probe_ys=scan.tolist(),
           ADF=(60, 200), return_layers=None)   # on-the-fly HAADF, no WF storage
wf, haadf = calc.run()
haadf.plot(filename="haadf.png")
haadf.to_sea("haadf.sea")
```

Post-hoc detector re-integration (needs stored wavefunctions):

```python
calc.setup(traj, aperture=25.0, voltage_eV=100e3, sampling=0.06,
           probe_xs=scan.tolist(), probe_ys=scan.tolist())   # return_layers=-1
wf = calc.run()
adf_60_200 = HAADFData(wf); adf_60_200.calculateADF(60, 200)
abf_10_20  = HAADFData(wf); abf_10_20.calculateADF(10, 20)
```

## MCP Tool Use

`pyslice_setup_multislice` (probe_grid, adf, aberrations, return_layers) →
check the grid/memory report → `pyslice_run_multislice` →
`pyslice_compute_haadf` (any detector geometry, PNG render) →
`pyslice_export_sea`. `pyslice_preview_potential` sanity-checks the
structure/orientation before the expensive run.

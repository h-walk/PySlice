---
name: md-setup
description: Use when running molecular dynamics with PySlice's ML potentials (ORB, FAIRChem) to produce thermal trajectories for multislice or TACAW — choosing the potential and ensemble, equilibration vs production, timestep and save-interval choices, structure relaxation, and the frozen-phonon shortcut when full MD is unnecessary.
---

# MD Setup

PySlice integrates ASE-based molecular dynamics with universal ML potentials
to generate the thermal trajectories that frozen-phonon imaging and TACAW
spectroscopy consume.

## Read First

- `skills/pyslice/SKILL.md` — units (MD timestep is **fs**; trajectory frame
  spacing is **ps**)
- `skills/simulation-parameter-selection/SKILL.md` rule 8 — how the target
  frequency window fixes save-interval and production length
- `examples/molecular_dynamics.py`, `tests/15_molecular_dynamics.py`

## Core Rules

- **Install:** `pip install 'pyslice[md]'` — pulls `orb-models`, which pins
  **Python 3.12** (no 3.13 wheels). ML weights download on first use (large);
  GPU strongly recommended for production runs.
- **Calculators:** `ORBMDCalculator(model_name='orb-v3-direct-inf-omat',
  device=...)` (default choice) and `FAIRChemMDCalculator(model_name='uma-s-1p1',
  ...)`. Both expose `relax_structure`, `setup`, `run`.
- **Workflow:** `setup(atoms, ...)` → `run()` → `Trajectory`. Equilibration
  runs first (convergence-checked on temperature/energy), then production;
  only production frames land in the returned trajectory.
- **Ensembles:** equilibrate in `nvt` (Langevin); produce in **`nve`** for
  TACAW (thermostat-free dynamics; set `production_ensemble='nve'` and give
  `production_relaxation_steps≈100` to let thermostat artifacts decay).
  `npt` only when the cell must breathe.
- **Timestep:** 1 fs default; drop to 0.5 fs for H-containing systems or
  high temperatures. `save_interval` (steps) × timestep = frame spacing —
  this is what sets the TACAW f_max, not the integration timestep.
- **Input structure:** `Trajectory.to_ase()` assumes element-symbol
  atom_types (the str/int gotcha); normalize LAMMPS-loaded structures with
  `atom_mapping` at load time. Relax first (`relax_structure`) when the
  structure came from a database at a different level of theory.
- **Supercell before MD:** phonons need the supercell built *before* the run
  (`tile_positions`); you cannot tile dynamics afterwards.
- **No cancel hook:** `run()` blocks with tqdm; long runs belong in a
  subprocess/job you can terminate.
- **Frozen-phonon shortcut:** for HAADF/4D-STEM thermal realism (not
  spectroscopy), skip MD entirely:
  `traj.generate_random_displacements(n=12, sigma=0.05-0.1, seed=...)`.

## Natural Language Task Routing

- "thermalize / equilibrate at T" → `setup(atoms, temperature=T,
  ensemble='nvt')`, defaults handle convergence.
- "trajectory for TACAW / phonons" → plan frame spacing from the frequency
  window (rule 8) → `production_ensemble='nve'`, `save_interval` = spacing /
  timestep, `production_steps` = frames × save_interval.
- "relax / optimize the structure" → `relax_structure(atoms)` before MD.
- "thermal snapshots for a HAADF" → frozen-phonon shortcut, no MD.
- "which potential" → ORB default; FAIRChem when its coverage/accuracy for
  the chemistry is known to be better; state the model name used.

## Python API Examples For Agents

```python
from pyslice.io.loader import Loader
from pyslice.md.molecular_dynamics import ORBMDCalculator

traj0 = Loader(filename="structure.cif").load().tile_positions((4, 4, 1))
atoms = traj0.to_ase()

md = ORBMDCalculator(device="cuda")
md.setup(
    atoms,
    temperature=300,
    timestep=1.0,               # fs
    ensemble="nvt",             # equilibration
    production_ensemble="nve",  # clean dynamics for TACAW
    production_relaxation_steps=100,
    production_steps=200 * 17,  # 200 frames...
    save_interval=17,           # ...every ~16.7 fs -> f_max ~30 THz
    output_dir="md_out",
)
traj = md.run()                 # Trajectory, timestep in ps
```

## MCP Tool Use

`pyslice_run_md` wraps this flow (calculator, model_name, device,
temperature_K, timestep_fs, ensembles, production_steps, save_interval) and
returns a trajectory handle for `pyslice_setup_multislice`. It is blocking —
get the plan from `pyslice_suggest_parameters` (goal `tacaw`) first.

# Molecular dynamics to TACAW

ML force fields are broad models, not automatically validated phonon models.
Check the selected model against the material, structure, forces, phonon
frequencies, and stress before production.

## Units

| Quantity | Unit |
|---|---|
| `MDCalculator.setup(timestep=...)` | fs per integrator step |
| `friction` | fs⁻¹ |
| `save_interval` | integrator steps |
| returned `Trajectory.timestep` | ps between saved frames |
| `Loader(..., timestep=...)` | ps between saved frames |
| TACAW frequency | THz = cycles/ps |

```python
frame_dt_ps = timestep_fs * save_interval / 1000
n_frames = production_steps // save_interval
nyquist_THz = 0.5 / frame_dt_ps
delta_f_THz = 1 / (n_frames * frame_dt_ps)
```

Integrator stability and saved-frame Nyquist are different convergence tests.

## Ensemble policy

Use NVT to equilibrate. Test an NVE pilot and quantify total-energy drift. If
NVE is stable with respect to timestep/model/precision, prefer it for spectral
production. If weak thermostatting is necessary, report friction and show that
peaks and widths are insensitive to it.

PySlice initializes MD with zero total linear momentum, and ASE Langevin is
constructed with ``fixcm=True`` so thermostat noise cannot drive a rigid random
walk. Diagnose the recorded result rather than assuming this is sufficient for
every calculator: finite-precision residual forces can still accumulate. For a
solid spectrum in the sample frame, inspect ``get_center_of_mass_drift()`` and,
if necessary, use ``remove_center_of_mass_drift()`` before multislice. The
source positions must be unwrapped before either operation.

NPT is supported for **ASE equilibration only**. It may be used to equilibrate
the volume at a target pressure, after which PySlice freezes the final cell and
switches to NVT (the default) or NVE for recorded production. NPT production is
rejected because a PySlice `Trajectory` has one fixed `box_matrix`; silently
propagating a variable-cell trajectory would assign the wrong cell to most
frames.

ASE's NPT barostat needs a material-dependent bulk modulus. PySlice accepts
pressure in bar and the bulk modulus in GPa, converts both to ASE units, and
constructs the barostat factor from the requested timescale:

```python
md.setup(
    atoms,
    ensemble="npt",
    pressure=1.01325,          # bar; positive means compression
    bulk_modulus_GPa=98.0,     # use a value appropriate to this material/model
    barostat_timescale=75.0,   # fs
    production_ensemble="nve",
    production_relaxation_steps=200,
)
```

The saved `equilibration.traj` remains an ASE trajectory and retains its
per-frame cells. It is suitable for inspecting NPT convergence, but `Loader`
will deliberately reject it as multislice/TACAW input. Use the returned
fixed-cell production trajectory instead. Monitor pressure, volume, cell shape,
and temperature; the generic equilibration heuristic currently checks
temperature and energy, not pressure convergence.

`md.run()` now stops if equilibration criteria are not met. Those criteria are
heuristics, not proof of stationarity; inspect `equilibration.log` and compare
independent seeds.

## Reproducibility

```python
import numpy as np

seed = 7
md.setup(..., rng=np.random.default_rng(seed),
         output_dir=f"outputs/si_300K_seed{seed}")
```

Record structure/input hash, model and weights, commit, environment, device,
precision, relaxation, ensemble, timestep, friction, seed, actual equilibration
steps/result, production length, save interval, energy drift, and temperature
statistics.

Output directories are not exact checkpoints, and integrator/RNG state is not
restored. Existing equilibration or production logs/trajectories are protected
by default; use a new directory for a new realization. Pass `overwrite=True`
only when replacement is deliberate.

When reloading `production.traj`, supply `frame_dt_ps` explicitly. Omitting it
uses Loader's generic 1 ps default and rescales the TACAW frequency axis.

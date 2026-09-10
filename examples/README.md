# PySlice examples

Run these scripts from the repository root after an editable installation.
The settings are demonstrations unless a script explicitly supplies a
convergence study.

| I want to… | Script | Inputs/extras | Cost | Status |
|---|---|---|---|---|
| Verify structure and trajectory loading | `loading_trajectories.py` | bundled inputs; OVITO/ASE | light | canonical |
| Compute one static TEM pattern | `tem_diffraction.py` | no external input | light | canonical first run |
| Compare frozen-phonon and MD HAADF | `haadf_stem.py` | bundled inputs | intermediate; GPU recommended | canonical |
| Inspect probe aberrations in HAADF | `aberrations.py` | bundled CIF | intermediate; GPU recommended | canonical |
| Run TACAW from an existing trajectory | `tacaw_from_trajectory.py` | bundled LAMMPS trajectory | substantial | canonical |
| Generate MD and run TACAW | `tacaw_pipeline.py` | `md` extra; model download | substantial | advanced |
| Generate ORB molecular dynamics | `molecular_dynamics.py` | `md` extra; model download | substantial | advanced |
| Build a TACAW spectrum image | `tacaw_spectrum_image.py` | output from `tacaw_pipeline.py` | substantial | advanced |
| Simulate LACBED | `lacbed.py` | bundled trajectory | substantial | advanced |

Plots and primary results normally go below `outputs/`, but caches are separate:

- Multislice caches default to `psi_data/` unless `save_path` selects another
  cache root.
- File-based `Loader` examples may create `.npy` sidecars and a `.cache.json`
  manifest beside the source file, including under `tests/inputs/` for bundled
  inputs.

## Reproduction scale

`k_space_tmdc_showcase_pub.py` and `real_space_phonon_showcase_pub.py` are
publication-oriented workflows, not tutorials. They require deliberate
hardware, storage, model, seed, input, cache, and convergence choices. Treat
their parameters as provenance for a particular study rather than defaults.

## Tests are not tutorials

The numbered files in `tests/` are scientific regressions, compatibility
programs, or stress tests. Some are intentionally expensive. Use the scripts
listed above for user workflows.

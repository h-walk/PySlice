# Smoke-testing the PySlice examples on Delta

Run every script in `examples/` on NCSA Delta to confirm they still work after
code changes. Each example runs as its own SLURM **job-array** task on a single
GPU, so failures are isolated and every example gets its own log.

## Files
- `submit_delta.sbatch` — job array (one task per example).
- `run_example.sh` — runs one example headless from `examples/`, returns its exit
  code (also usable interactively / on a login node).
- `logs/` — per-task logs land here.

## Quick start
```bash
cd examples/delta_smoke
mkdir -p logs

# edit submit_delta.sbatch first: set --account and your module/conda env
sbatch submit_delta.sbatch                # all 11 examples (array 0-10)

# subsets:
sbatch --array=0-5   submit_delta.sbatch  # the fast multislice-only examples
sbatch --array=6-10  submit_delta.sbatch  # the MD-based examples
sbatch --array=3     submit_delta.sbatch  # just haadf_stem.py
```

Check results:
```bash
grep -H 'rc=' logs/pyslice-ex-*.log        # one "[done] <ex> rc=0" per example
ls ../outputs                              # figures / .npy / .traj artifacts
```
`rc=0` means the example ran to completion. The multi-GPU ensemble example lives
separately in `examples/delta_tacaw_ensemble/` (it needs 4 GPUs) — submit that
one on its own.

## What each array index runs
| # | example | stages exercised | notes |
|---|---------|------------------|-------|
| 0 | `tem_diffraction.py` | Loader (in-memory), Potential, multislice | fast |
| 1 | `loading_trajectories.py` | Loader (cif/xyz/lammpstrj/traj) | fast, no multislice |
| 2 | `aberrations.py` | multislice, HAADF, aberrations | fast |
| 3 | `haadf_stem.py` | multislice, CBED, HAADF (frozen-phonon + MD dump) | moderate |
| 4 | `lacbed.py` | multislice, free-space propagation, real-space aperture | moderate |
| 5 | `tacaw_from_trajectory.py` | multislice, TACAW | moderate |
| 6 | `molecular_dynamics.py` | ORB MD (NVT/NVE) | **MD on CPU** |
| 7 | `tacaw_pipeline.py` | ORB MD → multislice → TACAW | **MD on CPU**; writes `outputs/tacaw_pipeline_md/production.traj` |
| 8 | `tacaw_spectrum_image.py` | STEM-scan TACAW map | needs #7's trajectory (auto-bootstrapped if missing) |
| 9 | `k_space_tmdc_showcase_pub.py` | ORB MD → multislice → TACAW (TMDC) | **MD on CPU**, longest |
| 10 | `real_space_phonon_showcase_pub.py` | ORB MD → multislice → TACAW (Si/graphene) | **MD on CPU**, longest |

## Environment (what to install in your conda env)
```bash
pip install -e '.[fast]'   # torch — GPU multislice/TACAW
pip install -e '.[io]'     # ovito — required to read .lammpstrj (examples 1, 3, 4, 5)
pip install orb-models     # ORB ML potential (examples 6, 7, 9, 10)
```
Without `ovito`, examples 1/3/4/5 fail with a clear "OVITO is required to read
this trajectory" error (they load LAMMPS dumps). `ovito` is served from its own
package index — see the OVITO docs if pip can't resolve it.

## Three things to know
1. **MD runs on CPU.** The examples call `ORBMDCalculator(...)` with no `device`,
   so ORB defaults to CPU; only the multislice/TACAW stages use the GPU. Fine for
   a smoke test, slow for examples 7/9/10. To use the GPU for MD, add
   `device="cuda"` to those `ORBMDCalculator(...)` calls (or lower
   `production_steps`). Bump `--time` if a run is truncated.
2. **ORB weights need to be present.** Compute nodes usually have no internet.
   Pre-fetch once on a login node (populates `$HOME/.cache`, shared with compute
   nodes) by running e.g. `python molecular_dynamics.py` there, **or** set
   `ORB_WEIGHTS_PATH` in the sbatch — the two `*_showcase_pub.py` scripts read it.
3. **Headless plotting.** `run_example.sh` sets `MPLBACKEND=Agg`, so figures are
   written to `../outputs/` and nothing tries to open a display.

## Running one example interactively (fastest debug loop)
```bash
salloc --account=YOUR_ALLOCATION --partition=gpuA40x4 --gpus-per-node=1 \
       --cpus-per-task=16 --mem=64g --time=01:00:00
cd examples/delta_smoke
./run_example.sh haadf_stem.py
```

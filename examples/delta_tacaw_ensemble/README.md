# Multi-GPU ensemble TACAW on Delta

Run **16 independent MD trajectories** across **4 A100 GPUs** on one NCSA Delta
node, average their **Welch** spectra (9 overlapping segments each), and write a
single ensemble-averaged TACAW spectrum.

## Files
- `tacaw_ensemble_delta.py` — the rank-aware driver (map: per-rank streaming
  accumulate + write a partial; reduce: rank 0 sums the partials).
- `submit_delta.sbatch` — SLURM script (1 node, 4 tasks, 4 GPUs).

## How it maps to hardware
- **One process per GPU.** `srun --ntasks=4` launches 4 ranks; each reads its
  identity from SLURM via `pyslice.dist_env()` and pins `cuda:$SLURM_LOCALID`.
  (The explicit device wins over `PYSLICE_DEVICE` — needed for per-rank pinning.)
- **Trajectory parallelism (Level 1).** Rank `r` owns trajectories `r, r+4, …`
  (4 each). Each is a distinct MD realisation (distinct seed) of the same
  structure/probe grid.
- **Streaming, host-resident accumulator.** Within a rank, trajectories are
  produced and reduced to a Welch spectrum **one at a time** and summed into a
  host accumulator — peak memory is *one trajectory + one accumulator*,
  independent of how many trajectories a rank owns. Exit waves never touch disk.
- **File-based reduce.** Each rank writes `partial_<rank>.npz` (a small,
  un-averaged periodogram sum + counts). After a barrier, rank 0 sums them into
  the ensemble average. Robust across nodes on the shared filesystem, and fault
  tolerant — a died rank just drops its trajectories.

## Run
```bash
# edit submit_delta.sbatch: set --account and your module/conda env
sbatch submit_delta.sbatch
# result:
#   tacaw_ensemble_out/tacaw_ensemble.npy   (probe, frequency, kx, ky)
#   tacaw_ensemble_out/frequencies_THz.npy
```

## Tuning
- `SEGMENT_LENGTH`, `OVERLAP`, `WINDOW` in the driver set the Welch estimate;
  `PRODUCTION_STEPS = 5 * SEGMENT_LENGTH` gives 9 segments at 50 % overlap.
- **More GPUs / nodes:** raise `--ntasks`/`--nodes`; the strided assignment and
  file-based reduce scale unchanged. With `M` trajectories on `G` GPUs each does
  `ceil(M/G)`.
- **Many beam positions** (a full STEM scan whose spectrum exceeds host RAM):
  add a probe-batch axis — make each unit a `(trajectory, probe-batch)` and pass
  `n_probes=<total>` and `rows_of=<i -> probe rows>` to `run_tacaw_ensemble`; the
  accumulator can be an on-disk memmap (`memmap_path=`). See
  `/workspaces/run/pyslice_tacaw_averaging_design.md` ("Many beam positions").

## Verifying the logic without a cluster
The whole map-reduce is exercised on CPU in
`tests/25_review_regressions.py::test_distributed_ensemble_equals_serial_on_cpu`
(a simulated 4-rank run reproduces the serial ensemble bit-for-bit) — so the
distribution/reduce is validated without any GPU.

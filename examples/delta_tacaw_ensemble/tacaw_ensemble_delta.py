#!/usr/bin/env python
"""Multi-GPU ensemble TACAW on NCSA Delta (or any SLURM/torchrun cluster).

Scenario: N_TRAJ independent MD trajectories, one process per GPU, Welch-averaged
spectra streamed into a host accumulator per rank, then a file-based reduce.

Map–reduce:
  * each rank owns a stride of trajectories (rank r -> r, r+world, ...),
    runs MD -> multislice -> a per-trajectory Welch spectrum, streaming them into
    a TACAWAccumulator (peak memory = one trajectory + the accumulator), and
    writes ONE partial_<rank>.npz;
  * after a barrier, rank 0 sums the partials into the ensemble-averaged TACAWData.

Launch with `srun` (4 tasks / 4 GPUs on one Delta gpuA100x4 node) -- see
submit_delta.sbatch. Each task is pinned to cuda:$SLURM_LOCALID.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from ase.build import bulk

from pyslice.multislice.distributed import dist_env, run_tacaw_ensemble
from pyslice.postprocessing.tacaw_data import reduce_tacaw_partials

# ---- ensemble / Welch parameters (the "16 trajectories, 9 segments" scenario) ----
N_TRAJ = 16
SEGMENT_LENGTH = 64          # L samples per FFT segment
OVERLAP = 0.5               # 50% -> Welch; 9 segments needs a 5*L-long trajectory
WINDOW = "hann"
PRODUCTION_STEPS = 5 * SEGMENT_LENGTH   # 5 segment-lengths -> K = 2*(N/L)-1 = 9
SAVE_INTERVAL = 1
TEMPERATURE = 300.0


def make_trajectory_producer(traj_idx, device, workdir):
    """Return a zero-arg callable that produces trajectory `traj_idx`'s WFData.

    Deferred (a callable) so exit waves for a trajectory are produced and freed
    one at a time inside the accumulator. Each trajectory is a distinct MD
    realisation (distinct seed) of the same structure.
    """
    def _produce():
        # Local imports so a rank only pays for the ML potential it actually uses.
        from pyslice import ORBMDCalculator, MultisliceCalculator

        atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (4, 4, 2)
        md = ORBMDCalculator(model_name="orb-v3-direct-inf-omat", device=device)
        md.setup(atoms, temperature=TEMPERATURE, timestep=2.0,
                 production_steps=PRODUCTION_STEPS, save_interval=SAVE_INTERVAL,
                 output_dir=Path(workdir) / f"md_{traj_idx}",
                 rng=np.random.default_rng(1000 + traj_idx))   # distinct realisation
        trajectory = md.run()

        calc = MultisliceCalculator(device=device)             # explicit device wins
        calc.setup(trajectory, aperture=0, voltage_eV=100e3,
                   sampling=0.1, slice_thickness=0.5,
                   probe_positions=[(atoms.cell[0, 0] / 2, atoms.cell[1, 1] / 2)],
                   cache_wavefunctions=False)
        return calc.run()
    return _produce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tacaw_ensemble_out",
                    help="shared output dir (partials + final spectrum)")
    ap.add_argument("--n-traj", type=int, default=N_TRAJ)
    args = ap.parse_args()

    rank, world, local_rank = dist_env()
    device = f"cuda:{local_rank}"          # one process per GPU; explicit pin
    out_dir = Path(args.out)
    workdir = out_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[rank {rank}/{world}] device={device} "
          f"trajectories={list(range(rank, args.n_traj, world))}", flush=True)

    producers = [make_trajectory_producer(i, device, workdir)
                 for i in range(args.n_traj)]

    # map: accumulate this rank's trajectories, write partial_<rank>.npz
    run_tacaw_ensemble(
        producers, out_dir,
        rank=rank, world=world,
        segment_length=SEGMENT_LENGTH, overlap=OVERLAP, window=WINDOW,
    )

    # barrier so every partial exists before the reduce
    _barrier(rank, world, out_dir)

    # reduce: rank 0 sums the partials into the ensemble-averaged spectrum
    if rank == 0:
        tacaw = reduce_tacaw_partials(out_dir)
        np.save(out_dir / "tacaw_ensemble.npy", tacaw.array)
        np.save(out_dir / "frequencies_THz.npy", tacaw.frequencies)
        print(f"[rank 0] reduced {args.n_traj} trajectories -> "
              f"{out_dir/'tacaw_ensemble.npy'}  shape={tacaw.array.shape}", flush=True)


def _barrier(rank, world, out_dir):
    """torch.distributed barrier if available, else wait for all partials."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier(); return
    except Exception:
        pass
    import time
    for _ in range(6000):                              # ~10 min budget
        n = len(list(Path(out_dir).glob("partial_*.npz")))
        if n >= world:
            return
        time.sleep(0.1)


if __name__ == "__main__":
    main()

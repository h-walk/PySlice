#!/bin/bash
#SBATCH --job-name=pyslice-tacaw-ensemble
#SBATCH --partition=gpuA100x4          # NCSA Delta 4x A100 GPU node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4            # one task (rank) per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16             # host cores for the CPU-side accumulate/reduce
#SBATCH --mem=200g
#SBATCH --time=02:00:00
#SBATCH --account=YOUR_ALLOCATION      # <-- set your Delta allocation
#SBATCH --output=tacaw_ensemble_%j.log

# --- environment (adjust to your Delta setup) ---
module purge
module load anaconda3_gpu             # or your Python/CUDA modules
# source activate pyslice             # your env with pyslice[fast] + orb-models installed

set -euo pipefail

OUT=${OUT:-tacaw_ensemble_out}

# One rank per GPU. SLURM sets SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID,
# which pyslice.dist_env() reads; each rank pins cuda:$SLURM_LOCALID.
# --gpu-bind=closest keeps each task on its local GPU.
srun --ntasks=4 --gpu-bind=closest \
     python tacaw_ensemble_delta.py --out "$OUT" --n-traj 16

echo "Ensemble spectrum written to ${OUT}/tacaw_ensemble.npy"

#!/bin/bash
#SBATCH --job-name=kikuchi-mgpu-test
#SBATCH --partition=gpu8
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

run_root='/emmadrive/Staff/Eric_Hoglund_e81/Pfeifer/qwe-dev-scratch/kikuchi-mgpu-test'
cd "$run_root/tests"
source /emmadrive/Shared_Code/torchenv/bin/activate
export PYTHONPATH="$run_root/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$run_root/matplotlib"
unset PYSLICE_DEVICE
unset PYSLICE_BACKEND
mkdir -p "$MPLCONFIGDIR"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
srun --unbuffered python 33_kikuchi_mgpu.py 149 tilts=6 workers_per_gpu=2 --no-gui

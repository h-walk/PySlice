#!/bin/bash
# Smoke-test every PySlice example on NCSA Delta as a SLURM job array:
# one array task per example, each on its own single-GPU allocation, so a
# failure in one example is isolated and every example gets its own log.
#
#   cd examples/delta_smoke
#   mkdir -p logs
#   sbatch submit_delta.sbatch                 # run them all (array 0-10)
#   sbatch --array=3 submit_delta.sbatch       # run just example #3 (haadf_stem)
#   sbatch --array=6-10 submit_delta.sbatch    # run just the MD-based ones
#
# Results: logs/pyslice-ex-<jobid>_<taskid>.log  and artifacts in ../outputs/.
#
#SBATCH --job-name=pyslice-ex
#SBATCH --partition=gpuA40x4          # A40 is plenty for a smoke test; swap to
                                      # gpuA100x4 if you want it faster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1             # multislice/TACAW use the GPU
#SBATCH --cpus-per-task=16            # ORB MD runs on CPU here (see NOTE below)
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --account=YOUR_ALLOCATION     # <-- set your Delta allocation
#SBATCH --array=0-10
#SBATCH --output=logs/pyslice-ex-%A_%a.log

set -uo pipefail

# --- environment (adjust to your Delta setup) --------------------------------
module purge
module load anaconda3_gpu             # or your Python/CUDA modules
# source activate pyslice             # your env, built with:
#   pip install -e '.[fast]'          # torch (GPU multislice/TACAW)
#   pip install -e '.[io]'            # ovito  -> reads .lammpstrj (examples 1,3,4,5)
#   pip install orb-models            # ORB ML potential (examples 6,7,9,10)
# (ovito comes from its own index; see the OVITO install docs if pip can't find it.)

# ORB weights: compute nodes usually have no internet. Either pre-fetch once on a
# login node (populates $HOME/.cache, which compute nodes share) OR point the two
# showcase scripts at a local copy:
# export ORB_WEIGHTS_PATH=/path/to/orb-v3-conservative-inf-omat.ckpt

# --- example manifest: array index -> script. Keep tacaw_pipeline before ------
# tacaw_spectrum_image so the trajectory exists when the latter runs.
EXAMPLES=(
    tem_diffraction.py                 # 0  multislice, in-memory structure
    loading_trajectories.py            # 1  Loader API (cif/xyz/lammpstrj/traj)
    aberrations.py                     # 2  HAADF with lens aberrations
    haadf_stem.py                      # 3  CBED + HAADF (frozen phonon vs MD)
    lacbed.py                          # 4  LACBED (defocus + free space + aperture)
    tacaw_from_trajectory.py           # 5  multislice + TACAW from a dump
    molecular_dynamics.py              # 6  ORB MD (no multislice)
    tacaw_pipeline.py                  # 7  MD -> multislice -> TACAW (writes traj)
    tacaw_spectrum_image.py            # 8  STEM TACAW map (needs #7's traj)
    k_space_tmdc_showcase_pub.py       # 9  TMDC dispersion (MD + multislice + TACAW)
    real_space_phonon_showcase_pub.py  # 10 real-space phonon maps (MD + multislice)
)

EX="${EXAMPLES[$SLURM_ARRAY_TASK_ID]}"
echo "[array ${SLURM_ARRAY_TASK_ID}] -> ${EX}"

srun bash "${SLURM_SUBMIT_DIR}/run_example.sh" "${EX}"
rc=$?
echo "[array ${SLURM_ARRAY_TASK_ID}] ${EX} finished rc=${rc}"
exit $rc

# NOTE — MD runs on CPU. The example scripts call ORBMDCalculator(...) without a
# device, so ORB defaults to CPU (only the multislice/TACAW stages use the GPU).
# That is fine for a "does it still run" smoke test but slow for the larger MD
# examples (7, 9, 10). To run ORB on the GPU, add device="cuda" to the
# ORBMDCalculator(...) call in those scripts, or reduce production_steps.

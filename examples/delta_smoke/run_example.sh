#!/bin/bash
# Run ONE PySlice example as a smoke test.
#
# Usage:   ./run_example.sh <example_basename.py>
# Example: ./run_example.sh haadf_stem.py
#
# Runs from the examples/ directory (examples use paths like ../tests/inputs/...
# and write to ./outputs), forces a headless matplotlib backend, and returns the
# example's own exit code so SLURM/callers can tell pass from fail.
set -uo pipefail

EX="${1:?usage: run_example.sh <example.py>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # examples/delta_smoke
EXDIR="$(cd "$HERE/.." && pwd)"                         # examples/
cd "$EXDIR"

export MPLBACKEND=Agg                                  # never open a display
mkdir -p outputs "$HERE/logs"

# tacaw_spectrum_image.py consumes the trajectory produced by tacaw_pipeline.py.
# If it isn't there yet (e.g. the array element ran first), produce it now so the
# example is self-contained.
if [[ "$EX" == "tacaw_spectrum_image.py" && ! -f outputs/tacaw_pipeline_md/production.traj ]]; then
    echo "[bootstrap] outputs/tacaw_pipeline_md/production.traj missing -> running tacaw_pipeline.py first"
    python tacaw_pipeline.py || { echo "[bootstrap FAILED] tacaw_pipeline.py rc=$?"; exit 1; }
fi

echo "======================================================================"
echo "[run ] $EX"
echo "[host] $(hostname)   [cwd] $(pwd)   [gpu] ${CUDA_VISIBLE_DEVICES:-none}"
echo "[time] start $(date)"
echo "======================================================================"

t0=$SECONDS
python "$EX"
rc=$?
echo "======================================================================"
echo "[done] $EX  rc=$rc  elapsed $((SECONDS - t0))s  $(date)"
echo "======================================================================"
exit $rc

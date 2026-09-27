#!/bin/bash
#SBATCH --job-name=se-randall-measure
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --dependency=afterok:6740040:6740049
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586/measure-%j.log
set -euo pipefail
root=/oscar/scratch/sboughan/se-semantic-recovery/randall-renderer-636bc586
run=$root/measure-$SLURM_JOB_ID
mkdir -p "$run/source" "$run/oracle"
tar -xzf "$root/source-v1.tar.gz" -C "$run/source"
tar -xzf /oscar/scratch/sboughan/se-semantic-recovery/state-bench-gpu-8d153702/oracle.tar.gz -C "$run/oracle"
tar -xzf "$root/randall-harness-v12.tar.gz" -C "$run/source"
export SWM_ROOT="$run/oracle"
export PYTHONPATH="$run/source:$run/oracle:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$run/source"
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python -m scripts.benchmarks.renderer_table.export "$root/randall-selection-remote.json" "$run/evidence"
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python -m scripts.benchmarks.renderer_table.assemble "$run/evidence/selection.json" "$run/results"
tar -czf "$run/compact-evidence.tar.gz" -C "$run" evidence results
printf '{"status":"EXPORT_AND_SUMMARY_VERIFIED"}\n' > "$run/exit.json"

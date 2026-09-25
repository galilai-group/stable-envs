#!/bin/bash
#SBATCH --job-name=se-state-cold-start
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
bundle="$PWD"
sha256sum -c SHA256SUMS
run="$bundle/run-$SLURM_JOB_ID"
mkdir "$run"
mkdir "$run/source" "$run/oracle" "$run/cache" "$run/results"
tar -xzf source.tar.gz -C "$run/source"
tar -xzf oracle.tar.gz -C "$run/oracle"
export PYTHONPATH="$run/source:$run/oracle" SWM_ROOT="$run/oracle"
export STATE_BENCH_SOURCE_COMMIT=8dc26634406bbff836d0a73079b3921b54d57378
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$run/cache" JAX_COMPILATION_CACHE_DIR="$run/cache/jax"
export JAX_PLATFORMS=cuda CRAFTAX_JAX_BACKEND=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=false
nvidia-smi > "$run/nvidia-smi.txt"
py=/oscar/scratch/sboughan/se-semantic-recovery/stable-envs-craftax-b6a8242f-gpu/run-6679082/venv/bin/python
cd "$run/source"
status=0
for name in CraftaxClassicPixels CraftaxClassicSymbolic CraftaxPixels CraftaxSymbolic; do
 for kind in public oracle; do
  # Separate disk cache as well as process: do not reuse compiled executables.
  export JAX_COMPILATION_CACHE_DIR="$run/cache/$name-$kind"
  "$py" scripts/benchmarks/cold_craftax.py --registration "swm/$name-v1" --device cuda --kind "$kind" --output "$run/results/$name-$kind" > "$run/results/$name-$kind.log" 2>&1 || status=$?
 done
done
printf '{"exit_code":%s,"job":"%s"}\n' "$status" "$SLURM_JOB_ID" > "$run/exit.json"
cd "$run"
find results -type f -print0 | sort -z | xargs -0 sha256sum > RESULT-SHA256SUMS
tar -czf "$bundle/evidence-$SLURM_JOB_ID.tar.gz" results nvidia-smi.txt exit.json RESULT-SHA256SUMS
exit "$status"

#!/bin/bash
#SBATCH --job-name=se-rendered-state
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
bundle="$PWD"
sha256sum -c UPLOAD-SHA256SUMS
run="$bundle/run-$SLURM_JOB_ID"
mkdir "$run" "$run/source" "$run/oracle" "$run/cache"
tar -xzf source.tar.gz -C "$run/source"
tar -xzf /oscar/scratch/sboughan/se-semantic-recovery/state-bench-gpu-8d153702/oracle.tar.gz -C "$run/oracle"
export PYTHONPATH="$run/source:$run/oracle" SWM_ROOT="$run/oracle"
export STATE_BENCH_SOURCE_COMMIT=f66d8d2f844441ea9ddc9ae15c7a4460ea7a398a
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$run/cache" WARP_CACHE_PATH="$run/cache/warp" TORCH_EXTENSIONS_DIR="$run/cache/torch"
export JAX_PLATFORMS=cuda CRAFTAX_JAX_BACKEND=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=false
nvidia-smi > "$run/nvidia-smi.txt"
cat > "$run/normal-python" <<'EOF'
#!/bin/bash
export PYTHONPATH="$PYTHONPATH:/oscar/scratch/sboughan/venvs/se-vis311/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step6/lib/python3.12/site-packages:/oscar/scratch/sboughan/venvs/se-step4-swm/lib/python3.12/site-packages"
exec /oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python "$@"
EOF
chmod +x "$run/normal-python"
cd "$run/source"
status=0
/oscar/scratch/sboughan/venvs/se6-verify-vis311/bin/python scripts/benchmarks/run_rendered_matrix.py --device cuda --output "$run/results" \
 --profile "normal=$run/normal-python" --profile "ogbench=$run/normal-python" \
 --profile fetch=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python \
 --profile manipulation=/oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete/run-6669909/profile/bin/python \
 --profile craftax=/oscar/scratch/sboughan/se-semantic-recovery/stable-envs-craftax-b6a8242f-gpu/run-6679082/venv/bin/python \
 > "$run/driver.log" 2>&1 || status=$?
printf '{"exit_code":%s,"job":"%s"}\n' "$status" "$SLURM_JOB_ID" > "$run/exit.json"
cd "$run"
find results -type f -print0 | sort -z | xargs -0 sha256sum > RESULT-SHA256SUMS
tar -czf "$bundle/evidence-$SLURM_JOB_ID.tar.gz" results driver.log nvidia-smi.txt exit.json RESULT-SHA256SUMS
exit "$status"

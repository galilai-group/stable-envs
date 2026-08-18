#!/bin/bash
#SBATCH --job-name=se-torch213
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_torch213_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
UV=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/uv
VENV=/mnt/vast/home/randall.balestriero/venvs/se-all
PY=$VENV/bin/python
SRC=/mnt/vast/home/randall.balestriero/src
export UV_CACHE_DIR=/mnt/vast/home/randall.balestriero/.cache/uv
echo "### node=$(hostname) gpus=$(nvidia-smi -L | wc -l)"

echo "-- before --"; "$UV" pip freeze --python "$PY" 2>/dev/null | grep -iE "^(torch|torchvision|triton|warp-lang|nvdiffrast)==" | sed 's/^/   /'

echo; echo "######## 1. torch 2.13.0 + torchvision 0.28.0 ########"
"$UV" pip install --python "$PY" torch==2.13.0 torchvision==0.28.0 2>&1 | tail -8

echo; echo "######## 2. rebuild nvdiffrast against the NEW torch ########"
# Its .so links against libtorch; a torch upgrade leaves a stale extension that imports and then
# fails at a symbol, so this is a rebuild rather than an upgrade check.
NV=$VENV/lib/python3.11/site-packages/nvidia
CU=$(ls -d $NV/cu13 $NV/cu12 2>/dev/null | head -1)
export CPATH="$CU/include:${CPATH:-}" LIBRARY_PATH="$CU/lib:${LIBRARY_PATH:-}"
export CUDA_HOME=/mnt/vast/home/randall.balestriero/miniconda3/envs/test
export PATH="$CUDA_HOME/bin:$PATH"
echo "   headers: $CU/include (cusparse.h $([ -f $CU/include/cusparse.h ] && echo present || echo MISSING))"
"$UV" pip install --python "$PY" --no-deps --no-build-isolation --reinstall "$SRC/nvdiffrast" 2>&1 | grep -vE "^\s+\[|nvcc" | tail -10

echo; echo "-- after --"; "$UV" pip freeze --python "$PY" 2>/dev/null | grep -iE "^(torch|torchvision|triton|warp-lang|nvdiffrast|mujoco|numpy|jax)==" | sed 's/^/   /'

echo; echo "######## 3. does everything still import? ########"
PYTHONPATH=. "$PY" - <<'EOF' 2>&1 | grep -vE "^Gym has|^Please upgrade|^Users of|^See the migration|WARNING:jax"
import json, torch, stable_envs as se
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| triton", __import__("triton").__version__)
print("REGISTRY:", len(se.REGISTRY), "of 196")
print("modules that FAILED:", {k: str(v)[:70] for k, v in se.envs.import_errors().items()})
cover = json.load(open("scratch/census/assignment.json"))
want = {e for es in cover.values() for e in es if not e.startswith("Overcooked")}
missing = sorted(want - set(se.REGISTRY))
print(f"cover minus Overcooked = {len(want)}; unreachable: {len(missing)} -> {missing}")
EOF

echo; echo "######## 4. THE ACTUAL GATE: full suite, 2 GPUs, one interpreter ########"
# Overcooked deliberately out of the assignment for this run; everything else must be reachable or
# SE_REQUIRE_ENVS fails the run rather than skipping it.
export SE_REQUIRE_ENVS=$("$PY" -c "
import json
c=json.load(open('scratch/census/assignment.json'))
print(','.join(sorted(e for es in c.values() for e in es if not e.startswith('Overcooked'))))")
echo "SE_REQUIRE_ENVS covers $(tr ',' '\n' <<<"$SE_REQUIRE_ENVS" | grep -c .) envs"
RUN=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_torch213_${SLURM_JOB_ID}_suite
mkdir -p "$RUN"
"$PY" -u -m stable_envs.testing.supervisor --rundir "$RUN" --json "$RUN/result.json" \
    -- tests/ -q -rs --durations=25 > "$RUN/pytest.log" 2>&1
rc=$?
echo "--- last 80 lines (full log: $RUN/pytest.log) ---"
tail -80 "$RUN/pytest.log"
echo "--- failure names ---"
grep -E "^(FAILED|ERROR) " "$RUN/pytest.log" | sed "s/ - .*//" | sort | uniq -c | sort -rn | head -40
echo "### done rc=$?"

#!/bin/bash
#SBATCH --job-name=ogb-cube-scene-gate
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

# Submit exactly once with --chdir and --output pointing at the immutable scratch bundle.
set -euo pipefail
bundle="$PWD"
run="$bundle/run-$SLURM_JOB_ID"
mkdir "$run"
mkdir "$run/source" "$run/oracle" "$run/evidence" "$run/cache"
export UV_CACHE_DIR="$run/cache/uv"
export XDG_CACHE_HOME="$run/cache"
export WARP_CACHE_PATH="$run/cache/warp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
sha256sum -c SHA256SUMS
cp bundle.json SHA256SUMS gate.sbatch "$run/evidence/"
tar -xzf source.tar.gz -C "$run/source"
tar -xzf oracle.tar.gz -C "$run/oracle"
base=/oscar/scratch/sboughan/venvs/se7-ogb
"$base/bin/python" -m venv --without-pip "$run/profile"
/oscar/home/sboughan/.local/bin/uv pip install --python "$run/profile/bin/python" --no-deps numpy==2.2.6 gymnasium==1.2.3 dm-control==1.0.41 > "$run/evidence/profile-install.log" 2>&1
printf '%s\n' "$base/lib/python3.12/site-packages" > "$run/profile/lib/python3.12/site-packages/gpu-dependencies.pth"
export PYTHONPATH="$run/source:$run/oracle"
python="$run/profile/bin/python"
cd "$run/source"
nvidia-smi > "$run/evidence/nvidia-smi.txt"
"$python" -m pip --version > "$run/evidence/pip.txt" 2>&1 || true
"$python" - <<'PY' > "$run/evidence/profile.json"
import importlib.metadata as m
import json, platform, sys
from pathlib import Path
import stable_worldmodel, stable_envs
required={'mujoco':'3.12.0','mujoco-warp':'3.12.0','warp-lang':'1.17.0','ogbench':'1.2.1','numpy':'2.2.6','gymnasium':'1.2.3','dm-control':'1.0.41','torch':'2.14.0+cu126','tensordict':'0.14.2'}
actual={k:m.version(k) for k in required}
assert actual==required,actual
assert Path(stable_worldmodel.__file__).is_relative_to(Path(sys.path[2])) or '/oracle/' in stable_worldmodel.__file__
print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'required':required,'actual':actual,'oracle':stable_worldmodel.__file__,'candidate':stable_envs.__file__,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())},indent=2))
PY
status=0
for family in Cube Scene; do
    mkdir "$run/evidence/$family"
    cpu=0
    MUJOCO_GL=disable "$python" -m pytest -q tests/unit/test_ogbench_manipulation_cpu.py tests/unit/test_ogbench_render_state.py -k "$family" --junitxml="$run/evidence/$family/cpu.xml" > "$run/evidence/$family/cpu.log" 2>&1 || cpu=$?
    if [ "$cpu" -eq 0 ]; then
        "$python" - "$run/evidence/$family/cpu.xml" <<'PY' || cpu=$?
import sys,xml.etree.ElementTree as ET
root=ET.parse(sys.argv[1]).getroot()
cases=list(root.iter('testcase'))
assert len(cases)==46,len(cases)
assert not any(list(c) for c in cases), 'CPU gate contains skips/failures/errors'
print('OGB_FAMILY_CPU_GATE_PASS',sys.argv[1],flush=True)
PY
    fi
    gpu=99
    if [ "$cpu" -eq 0 ]; then
        gpu=0
        MUJOCO_GL=egl "$python" scripts/recovery/ogbench_renderer_gate.py --family "$family" --fixtures docs/research/semantic_recovery_evidence/ogb_renderer_closeout/fixtures --output "$run/evidence/$family" > "$run/evidence/$family/gpu.log" 2>&1 || gpu=$?
    fi
    printf '{"cpu_exit":%s,"gpu_exit":%s}\n' "$cpu" "$gpu" > "$run/evidence/$family/exits.json"
    if [ "$cpu" -ne 0 ] || [ "$gpu" -ne 0 ] || ! grep -qx "OGB_${family^^}_GPU_GATE_PASS" "$run/evidence/$family/gpu.log"; then
        status=1
    fi
done
if [ "$status" -eq 0 ]; then
    printf '%s\n' OGB_CUBE_SCENE_GPU_GATE_PASS | tee "$run/evidence/terminal.txt"
else
    printf '%s\n' OGB_CUBE_SCENE_GPU_GATE_FAIL | tee "$run/evidence/terminal.txt"
fi
cd "$run/evidence"
find . -type f ! -name SHA256SUMS.evidence -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.evidence
cd "$run"
tar -czf evidence.tar.gz evidence
sha256sum evidence.tar.gz > evidence.tar.gz.sha256
exit "$status"

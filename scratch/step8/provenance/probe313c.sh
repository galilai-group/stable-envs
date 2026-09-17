#!/bin/bash
# Round 3: the two OGB cells under 3.13. They need stable_worldmodel (the cell builds the
# swm-registered env to compile the scene), which se8-mjwarp313 was not built with -- added here
# with --no-deps so no pin moves. The 3.12 comparison is the COMMITTED closure2 shard
# examples/results/step7/closure2/gate_a/cuda_ogbmanip.json, measured in the same SLURM allocation
# on the same card; its interpreter recorded stable_worldmodel only as "present", so the swm version
# is a provenance gap on the 3.12 side and is reported as one.
set -uo pipefail
: "${SCR:?set SCR to the scratch root before running, e.g. SCR=$SCR bash "$0"}"
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=$SCR/se8/reference-provenance
OUT=$CODE/examples/results/step8/provenance/mjwarp313
VENV=$SCR/venvs/se8-mjwarp313
mkdir -p "$OUT"; cd "$CODE"; export PYTHONPATH=$CODE
echo "### node $(hostname) job ${SLURM_JOB_ID:-none}"
$(command -v uv) pip install --python "$VENV/bin/python" --no-deps \
  'stable-worldmodel==0.1.1' 'loguru' 'lancedb' 'pylance' 'pyarrow' 'rich' 'typer' 'pydantic==2.13.5' 'pydantic-core==2.46.5' 'annotated-types==0.8.0' 'annotated-doc==0.0.5' 'typing-inspection==0.4.4' 'markdown-it-py==4.2.0' 'mdurl==0.1.2' 'shellingham==1.5.4' 'six==1.17.0' 'python-dateutil==2.9.0.post0' 'deprecation==2.1.0' 'lance-namespace==0.8.6' 'lance-namespace-urllib3-client==0.8.6' 'pygments' 'requests' 'overrides' 2>&1 | tail -4
"$VENV/bin/python" scratch/step8/provenance/texgap_probe.py --out "$OUT/texgap_ogb_313.json" 2>&1 | tail -25
"$VENV/bin/python" scratch/step8/provenance/digest_replay.py --device cuda \
  --cells OGBCube,OGBScene --out "$OUT/digest_ogb_313.json" 2>&1 | tail -10
echo "### done"

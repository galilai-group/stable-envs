#!/bin/bash
set -euo pipefail
while kill -0 1609 2>/dev/null; do sleep 10; done
cd /private/tmp/stable-envs-semantic-recovery
out=docs/research/semantic_recovery_evidence/state_performance_cold_cpu
mkdir "$out"
export PYTHONPATH=.:/Users/samib/stable-worldmodel SWM_ROOT=/Users/samib/stable-worldmodel
export SDL_VIDEODRIVER=dummy MUJOCO_GL=disable
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu CRAFTAX_JAX_BACKEND=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=/private/tmp/state-performance-cold-cpu-cache
for name in CraftaxClassicPixels CraftaxClassicSymbolic CraftaxPixels CraftaxSymbolic; do
 for kind in public oracle; do
  export JAX_COMPILATION_CACHE_DIR="$XDG_CACHE_HOME/$name-$kind"
  /private/tmp/craftax-recovery-312/bin/python scripts/benchmarks/cold_craftax.py --registration "swm/$name-v1" --device cpu --kind "$kind" --output "$out/$name-$kind" > "$out/$name-$kind.log" 2>&1
  printf '%s %s COMPLETE\n' "$name" "$kind"
 done
done

# Common environment for every Step-6 integration-verification job. Sourced, never executed.
#
# HOME is at 100% of quota (AGENT_COMMON), so every cache lives under $SCR. MUJOCO_GL=egl because
# the reference dm_control/MuJoCo renderers are called headless on the compute nodes.
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export WARP_CACHE_PATH=$SCR/.cache/warp TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1

# The tree under measurement. Overridable so the coordinator's post-merge re-run can point this at a
# freshly synced copy without editing a script.
TREE=${TREE:-$SCR/se6/verify}
BASE=${BASE:-$SCR/se6/baseline}
OUTROOT=${OUTROOT:-$TREE/examples/results/step6/verify}

# rsync drops .git, so the SHA under verification reaches every artifact through the environment.
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$TREE/GIT_COMMIT" 2>/dev/null || true)}

# One interpreter per frozen shard, chosen to REPRODUCE the interpreter each shard was frozen in
# (examples/results/step4/conformance/<shard>.json -> provenance.reference_packages). Reproducing a
# digest in a different dependency set does not answer the question this job asks.
#
#   se-step3        dm_control + gym_pusht + ogbench + mujoco 3.12 + mujoco_warp 3.12, no SWM
#                   -> exactly the dmc/ogb/raster shards' frozen interpreter (jobs 6176998/6178140)
#   se-step4-swm    craftax + stable_worldmodel, NO mujoco at all -> the swm shard's (job 6180921)
#   se-s3b-fetch311 gymnasium_robotics 1.4.2 + mujoco/mujoco_warp 3.11 -> the fetch shard's (6176999)
#   se6-verify      created by this workstream, installs NOTHING: a --without-pip venv whose single
#                   .pth adds se-step6's site-packages (pytest, dm_control 1.0.45, ogbench, pygame,
#                   pymunk, triton), then se-step3's (mujoco_warp, warp, gym_pusht), then the
#                   stable_worldmodel checkout. It is the only interpreter on the cluster that can
#                   run the WHOLE GPU pytest suite in one process; se-step6 has pytest but no
#                   mujoco_warp and se-step5 has mujoco_warp but no pytest.
venv_for () {
  case "$1" in
    dmc|ogb|raster) echo "$SCR/venvs/se-step3" ;;
    swm)            echo "$SCR/venvs/se-step4-swm" ;;
    fetch)          echo "$SCR/venvs/se-s3b-fetch311" ;;
    verify)         echo "$SCR/venvs/se6-verify" ;;
    *)              echo "$SCR/venvs/se6-verify" ;;
  esac
}

banner () {
  echo "=============================================================================="
  echo "job=${SLURM_JOB_ID:-none} node=$(hostname) tree=$TREE sha=${STABLE_ENVS_GIT_COMMIT:-unknown}"
  echo "date=$(date -u +%FT%TZ) partition=${SLURM_JOB_PARTITION:-?}"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no gpu"
  echo "=============================================================================="
}

# Raw pytest output NEVER lands in the repository: it carries absolute $SCR interpreter paths, and
# tests/unit/test_repo_contracts.py's citation scanner reads every .txt under examples/, so a raw log
# fails a real check on a file that is not documentation. Borrowed verbatim from
# scratch/step6/reacher-correctness/gpu_tests.sbatch, which is where that was learned.
keep () {  # keep <results-file> ; reads stdin
  local dest="$1" full="$SCR/logs/${SLURM_JOB_ID:-adhoc}_$(basename "$1")"
  mkdir -p "$(dirname "$dest")" "$SCR/logs"
  tee "$full" | grep -E "^(tests/\S+::\S+ (PASSED|FAILED|SKIPPED|XFAIL|XPASS)|=+ .*(passed|failed|error|skipped).*|[0-9]+ (passed|failed).*|FAILED |ERROR )" \
    > "$dest" || true
  echo "--- tail of $dest ---"; tail -5 "$dest"
}

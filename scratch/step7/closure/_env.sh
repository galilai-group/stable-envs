# Common environment for the Step-7 closure verification jobs. Sourced, never executed.
#
# Descends from scratch/step6/verify/_env.sh: same HOME-quota rule (everything under $SCR), same
# "one interpreter per frozen shard" discipline, same `keep` filter that stops raw pytest output --
# which carries absolute $SCR paths -- from ever landing in the repository, because
# tests/unit/test_repo_contracts.py's citation scanner reads every .txt under examples/.
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export WARP_CACHE_PATH=$SCR/.cache/warp TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1

TREE=${TREE:-$SCR/se7/closure}
BASE=${BASE:-$SCR/se7/baseline}
OUTROOT=${OUTROOT:-$TREE/examples/results/step7/closure}
SWMSRC=${SWMSRC:-$SCR/se7/swm-src}

# rsync drops .git, so the SHA under verification reaches every artifact through a file the sync
# writes on purpose (.git_sha) rather than through an inference.
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$TREE/.git_sha" 2>/dev/null || true)}

# INTERPRETERS.
#
# The frozen Step-4 shards keep the Step-6 rule: each shard replays in the interpreter it was frozen
# in (examples/results/step4/conformance/<shard>.json -> provenance.reference_packages). Reproducing
# a digest in a different dependency set answers a different question.
#   se-step3        dm_control 1.0.45 + gym_pusht + ogbench + mujoco/mujoco_warp 3.12  -> dmc/ogb/raster
#   se-step4-swm    craftax 1.6.1 + jax 0.11.1 + stable_worldmodel, NO mujoco           -> swm
#   se-s3b-fetch311 gymnasium_robotics 1.4.2 + mujoco/mujoco_warp 3.11                  -> fetch
#
# The seven Step-7 cells each name the interpreter their declaration was measured in:
#   se7-craftax     jax 0.11.1 (cpu backend) + craftax 1.6.1   -> CraftaxClassic, CraftaxFull
#   se-vis311       mujoco 3.11.0 + dm_control 1.0.44          -> the declared DM Control pair
#   se7-ogb         ogbench 1.2.1 + mujoco_warp 3.12 (+ swm source on PYTHONPATH) -> OGBCube/OGBScene
#   se7-rocket      nvdiffrast 0.4.0 + PyFlyt 0.29.0 + pybullet -> PFRocketLanding, the six nvdiff cells
#   se6-verify      installs nothing: a --without-pip venv chaining se-step6 + se-step3 + se-step5
#                   + the swm checkout; the only interpreter that can run tests/gpu over the MJWarp
#                   families in one process (se-step6 has pytest and no mujoco_warp).
venv_for () {
  case "$1" in
    dmc|ogb|raster)      echo "$SCR/venvs/se-step3" ;;
    swm)                 echo "$SCR/venvs/se-step4-swm" ;;
    fetch)               echo "$SCR/venvs/se-s3b-fetch311" ;;
    craftax)             echo "$SCR/venvs/se7-craftax" ;;
    vis311)              echo "$SCR/venvs/se-vis311" ;;
    ogbmanip)            echo "$SCR/venvs/se7-ogb" ;;
    rocket)              echo "$SCR/venvs/se7-rocket" ;;
    verify)              echo "$SCR/venvs/se6-verify" ;;
    step6)               echo "$SCR/venvs/se-step6" ;;
    *)                   echo "$SCR/venvs/se6-verify" ;;
  esac
}

banner () {
  echo "=============================================================================="
  echo "job=${SLURM_JOB_ID:-none} node=$(hostname) tree=$TREE sha=${STABLE_ENVS_GIT_COMMIT:-unknown}"
  echo "date=$(date -u +%FT%TZ) partition=${SLURM_JOB_PARTITION:-?} load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no gpu"
  echo "=============================================================================="
}

stage () { echo; echo "############################## STAGE $* :: $(date -u +%FT%TZ)"; }

# Raw pytest output NEVER lands in the repository (absolute $SCR interpreter paths fail the citation
# scanner in tests/unit/test_repo_contracts.py). Borrowed verbatim from scratch/step6/verify/_env.sh.
keep () {
  local dest="$1" full="$SCR/logs/${SLURM_JOB_ID:-adhoc}_$(basename "$1")"
  mkdir -p "$(dirname "$dest")" "$SCR/logs"
  tee "$full" | grep -E "^(tests/\S+::\S+ (PASSED|FAILED|SKIPPED|XFAIL|XPASS)|=+ .*(passed|failed|error|skipped).*|[0-9]+ (passed|failed).*|FAILED |ERROR )" \
    > "$dest" || true
  echo "--- tail of $dest ---"; tail -4 "$dest"
}

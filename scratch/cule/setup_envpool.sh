#!/bin/bash
set -x
VENV=/mnt/vast/home/randall.balestriero/venvs/envpool
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
source "$SCRATCH/make_venv.sh" "$VENV" || exit 1
case "$(command -v pip)" in "$VENV"/*) ;; *) echo "FATAL pip=$(command -v pip)"; exit 1;; esac

pip install -q --upgrade pip wheel setuptools
echo "=== pip index versions envpool ==="
pip index versions envpool 2>&1 | head -n 5
echo "=== pip install envpool ==="
pip install envpool 2>&1 | tail -n 25
echo "ENVPOOL_INSTALL_EXIT=$?"
pip install -q numpy "ale-py>=0.10" gymnasium opencv-python-headless
pip list 2>/dev/null | grep -i -E "envpool|ale-py|gymnasium|numpy"
python - <<'PY'
import sys
try:
    import envpool
    print("envpool", envpool.__version__)
    print("n_atari_tasks", len([t for t in envpool.list_all_envs() if t.endswith('-v5')]))
except Exception as e:
    print("ENVPOOL IMPORT FAILED:", repr(e))
import ale_py, gymnasium
print("ale_py", ale_py.__version__, "gymnasium", gymnasium.__version__, "py", sys.version)
PY
python "$SCRATCH/bench_envpool.py" "$SCRATCH/envpool_bench.json"
echo "BENCH_EXIT=$?"

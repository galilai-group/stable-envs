#!/bin/bash
# make_venv.sh <venv_path>
# Create an isolated python3.12 venv.  The image ships /usr/bin/python3.12
# without ensurepip (python3.12-venv is not installed), so the venv is created
# --without-pip and pip is bootstrapped from a staged get-pip.py.
#
# Hard-asserts that python/pip resolve INSIDE the venv afterwards: without the
# assert a failed creation silently falls through to whatever conda env happens
# to be on PATH, which must never be written to.
set -euo pipefail
VENV="$1"
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule

rm -rf "$VENV"
/usr/bin/python3.12 -m venv --without-pip "$VENV"
"$VENV/bin/python" "$SCRATCH/get-pip.py" -q
# shellcheck disable=SC1091
source "$VENV/bin/activate"

case "$(command -v python)" in "$VENV"/*) ;; *) echo "FATAL: python is $(command -v python), not in $VENV"; exit 1;; esac
case "$(command -v pip)"    in "$VENV"/*) ;; *) echo "FATAL: pip is $(command -v pip), not in $VENV"; exit 1;; esac
python -c "import sys; assert sys.prefix.startswith('$VENV'), sys.prefix; print('venv ok', sys.prefix, sys.version)"

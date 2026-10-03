#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/navigation-online-profile-v1-20261002
test ! -e "$BASE/matplotlib3109"
export UV_CACHE_DIR="$BASE/uv-cache"
"$HOME/.local/bin/uv" pip install --python /oscar/scratch/sboughan/venvs/se-step6/bin/python --target "$BASE/matplotlib3109" --no-deps matplotlib==3.10.9 contourpy cycler fonttools kiwisolver pyparsing python-dateutil six
PYTHONPATH="$BASE/matplotlib3109" /oscar/scratch/sboughan/venvs/se-step6/bin/python -c 'import matplotlib; from matplotlib.backends.backend_agg import FigureCanvasAgg; from matplotlib.figure import Figure; import importlib.metadata as m; print({p:m.version(p) for p in ("matplotlib","numpy","torch","contourpy","cycler","fonttools","kiwisolver","pyparsing","python-dateutil","six")}); assert matplotlib.__version__ == "3.10.9"'

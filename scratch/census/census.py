"""What does THIS interpreter register? Printed as JSON, one line, for merging across venvs.

The registry is not a single number, because no interpreter can hold every family: robocasa pins
mujoco 3.3.1 while mujoco_warp needs >= 3.11, JAX and torch CUDA builds disagree, and several families
ship their own fork of a reference package. The honest count is the union over interpreters, with the
per-interpreter counts kept so a reader can see which venv contributes what.
"""
from __future__ import annotations

import json
import sys

out = {"python": sys.executable}
try:
    import stable_envs as se
    out["registry"] = sorted(se.REGISTRY)
    out["n"] = len(se.REGISTRY)
    out["import_errors"] = se.envs.import_errors()
    out["file"] = se.__file__
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"
print("CENSUS " + json.dumps(out))

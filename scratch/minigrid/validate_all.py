"""Run the exactness check over every instantiable MiniGrid/BabyAI env id."""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mg_gpu import MiniGridGPU  # noqa: E402
from validate import check_env  # noqa: E402

envs_json, out_path = sys.argv[1], sys.argv[2]
tile = int(sys.argv[3]) if len(sys.argv) > 3 else 32
rows = json.load(open(envs_json))["rows"]
ids = [r["id"] for r in rows]

here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, f"atlas_{tile}.npz"))
gpu = MiniGridGPU(z["atlas"], tile, device="cuda")

out = []
for i, eid in enumerate(ids):
    try:
        r = check_env(gpu, eid, n_ep=2, ep_len=10, seed0=777, device="cuda", tile=tile)
    except Exception as ex:  # noqa: BLE001
        r = {"env_id": eid, "error": f"{type(ex).__name__}: {ex}"}
    out.append(r)
    if "error" in r:
        print(f"[{i+1}/{len(ids)}] {eid:44s} ERROR {r['error']}", flush=True)
    else:
        bad = (r["n_frames"] * 3
               - r["full_hl"]["exact_frames"] - r["full_nohl"]["exact_frames"]
               - r["pov"]["exact_frames"]) + r["state_sufficiency_mismatches"]
        print(f"[{i+1}/{len(ids)}] {eid:44s} n={r['n_frames']:3d} bad={bad}", flush=True)

ok = [r for r in out if "error" not in r]
agg = {
    "n_envs": len(out),
    "n_env_errors": len(out) - len(ok),
    "total_frames": sum(r["n_frames"] for r in ok),
    "state_sufficiency_mismatches": sum(r["state_sufficiency_mismatches"] for r in ok),
}
for k in ("full_hl", "full_nohl", "pov"):
    agg[k + "_exact"] = sum(r[k]["exact_frames"] for r in ok)
    agg[k + "_max_abs_diff"] = max([r[k]["max_abs_diff"] for r in ok] or [0])
agg["vis_triton_matches_torch_everywhere"] = all(r["vis_triton_matches_torch"] for r in ok)
agg["triton_matches_torch_everywhere"] = all(
    r[k]["triton_matches_torch"] for r in ok for k in ("full_hl", "full_nohl", "pov")
)
agg["errors"] = [r for r in out if "error" in r]
print("\nAGG " + json.dumps(agg, indent=2))
with open(out_path, "w") as f:
    json.dump({"agg": agg, "per_env": out}, f, indent=2)

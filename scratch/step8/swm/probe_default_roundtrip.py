"""Cheap, CPU-only probe: does ``default_style() == style_from_variation(env, {})`` for every
stable-envs class that carries a ``VariationMap``.

This is the "registered default is reproducible through the mapping" check plan section 23 asks
for, run directly rather than inferred: construct the class at a tiny size, read its own default
style row, run the SAME empty-variation config through ``style_from_variation``, and compare with
``np.array_equal`` (not ``allclose`` -- the round trip is claimed exact).

A class that cannot be constructed under this interpreter (GPU-only backend: Fetch, OGBCube,
OGBScene, PFRocketLanding; or a missing optional dependency: OGBench's Pointmaze pair) is recorded
as ``SKIPPED`` with the reason, not silently dropped -- ``build_gate_f.py`` fills those rows from
the existing committed GPU evidence
(``examples/results/step8/variations-fetch/gpu_evidence.json``,
``examples/results/step8/variations-newcells/gpu/default_{ogb,rocket}_{baseline,branch}.json``)
instead of leaving them blank.

Run once per available CPU interpreter and merge the outputs (this script never overwrites a class
another run already recorded, so `se6-venv` + `se7-ogb-venv` together cover everything constructible
without a GPU):

    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step8/swm/probe_default_roundtrip.py
    MUJOCO_GL=glfw /Users/samib/se7-ogb-venv/bin/python scratch/step8/swm/probe_default_roundtrip.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "examples" / "results" / "step8" / "swm" / "default_roundtrip_cpu.json"

# Optional extra registrations some classes need explicit imports for (gated on a missing GPU
# backend inside the module itself; importing here just lets us SEE the "no backend" reason
# instead of a bare KeyError from se.make).
for mod in ("stable_envs.envs.rocket_landing",):
    try:
        __import__(mod)
    except Exception:
        pass


def _sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    import numpy as np
    import stable_envs as se
    from stable_envs.compat.worldmodel import VARIATION_MAPS, style_from_variation

    existing = json.loads(OUT.read_text()) if OUT.is_file() else {
        "what_this_is": __doc__, "baseline_sha": _sha(), "runs": {},
    }
    host = platform.node()
    interpreter = sys.executable
    run_key = f"{interpreter}"
    rec = existing["runs"].setdefault(run_key, {"host": host, "python": sys.version.split()[0],
                                                 "classes": {}})
    for name in sorted(VARIATION_MAPS):
        try:
            env = se.make(name, batch=1, size=32, device="cpu")
        except Exception as ex:
            rec["classes"][name] = {"status": "SKIPPED", "reason": f"{type(ex).__name__}: {ex}"[:300]}
            continue
        try:
            base = np.asarray(env.default_style(), np.float32).reshape(-1)
            rt = np.asarray(style_from_variation(env, {}), np.float32).reshape(-1)
            eq = bool(np.array_equal(base, rt))
            rec["classes"][name] = {
                "status": "MEASURED", "default_equals_style_from_empty_variation": eq,
                "style_dim": int(base.shape[0]),
            }
        except Exception as ex:
            rec["classes"][name] = {"status": "ERROR", "reason": f"{type(ex).__name__}: {ex}"[:300]}
        print(name, rec["classes"][name])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

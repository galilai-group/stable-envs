"""Gate A for a cell that is not yet in the frozen Step-4 universe: ``examples/fidelity_step4.measure``
called directly with the configuration dict the universe would carry once the Step-8 freeze adds it.

Same checks, same corpus protocol (``corpus_for``: 64 of a 256-state pool at ``STATE_SEED``, written
to ``--states-dir`` as ``<Config>.npz`` -- the candidate frozen corpus), same raw-shard shape, so
``scratch/step4/certify.py`` can label it once the cell has a universe entry. The verdicts printed
here are read off the raw row for the MANDATORY group (bounds ``== 0`` / ``identical``) and the
visual check's observation; the reference-judged checks are meaningful only under the declared
3.11 pair (``se-vis311`` on Oscar) and are recorded, not judged, under 3.12.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="DMCQuadrupedWalk")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--pool", type=int, default=256)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("fidelity_step4", ROOT / "examples" / "fidelity_step4.py")
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    import stable_envs as se
    cls = se.REGISTRY[a.config]
    config = {"config_id": a.config, "env": a.config, "swm_ids": ["swm/QuadrupedDMControl-v0"],
              "role": "registered_default", "siblings": [], "registered_default_coverage": "full",
              "fidelity_tier": cls.fidelity_tier, "backends": ["mjwarp", "cpu3d", "raycast3d"],
              "default_size": cls.default_size, "reference_id": cls.reference_id,
              "parity_interior_min": None, "parity_all_exact_min": None, "parity_note": "",
              "render_decoder": None, "module": cls.__module__}
    row = F.measure(config, Path(a.states_dir), a.n_states, a.pool, a.device, None)
    doc = {"schema_version": 1, "step": 7, "shard": "dmc-quadruped", "provenance": F.provenance(a.device),
           "configs": {a.config: row}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print("status:", row.get("status"), row.get("why", ""))
    for group in ("mandatory", "family_specific"):
        for k, v in sorted((row.get(group) or {}).items()):
            obs = {kk: vv for kk, vv in v.items() if kk not in ("traceback",)}
            print(f"  [{group}] {k}: {json.dumps(obs, default=str)[:220]}")
    print("seconds", row.get("seconds"))


if __name__ == "__main__":
    main()

"""The plate bake, exercised on the CPU where the cells themselves refuse to run.

WHY THIS EXISTS. ``OGBCube``/``OGBScene`` have no CPU backend by declaration, so nothing on a Mac can
render them — but the PLATE is drawn by ``mujoco.Renderer`` and not by ``mujoco_warp``, so the half
of the pipeline the Step-9 texel deduplication touched (``_grid_texture`` -> ``texel_slots``,
``_apply_floor`` -> ``substitute``, ``_floor_key`` -> ``texel_key``) CAN be exercised here. This
drives it through the unbound methods on a stub, in BOTH trees, and compares the baked plates byte
for byte. It is the evidence that "the shared helper is the same operation" is a measurement rather
than a reading of two sources.

Run it against this tree and against a detached worktree of the baseline; the digests must match.

    MUJOCO_GL=glfw PYTHONPATH=$PWD /Users/samib/se7-ogb-venv/bin/python \
        scratch/step9/envs-ogbench/plate_probe.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import torch


def main() -> int:
    import stable_envs.envs.ogb_manip as ogb

    out = {}
    for cell in ("Cube", "Scene"):
        b = ogb._bundle(cell)
        # An INSTANCE with only what the plate path needs -- no CUDA, no mujoco_warp, no renderer of
        # our own. ``__init__`` refuses cpu outright, which is the declaration, so the object is made
        # with ``object.__new__`` exactly as ``tests/unit/test_ogb_manip.py`` makes its own stub.
        cls = type(f"_Plate{cell}", (ogb.OGBManipBase,), {"CELL": cell})
        stub = object.__new__(cls)
        stub._b = b
        stub.device = torch.device("cpu")
        rec = {"grid_texels": int(b.grid_tex["texels"]) if "texels" in b.grid_tex else None,
               "grid_covered": int(b.grid_tex.get("covered", -1))}
        # the cache key: the default collapses to None by whatever route it is asked for
        default = b.var_defaults["floor.color"]
        rec["key_default_is_none"] = stub._floor_key(default) is None
        rec["key_none_is_none"] = stub._floor_key(None) is None
        rec["key_nearby_is_none"] = stub._floor_key(default + 1e-4) is None  # same bytes
        custom = np.array([[0.90, 0.10, 0.20], [0.30, 0.70, 0.40]])
        rec["key_custom"] = list(stub._floor_key(custom))
        # the bakes themselves, at a small resolution
        for tag, floor in (("default", None), ("default_explicit", default), ("custom", custom)):
            plate = stub._bake_plate(96, floor=floor)
            a = plate.detach().cpu().numpy()
            rec[f"plate_{tag}"] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            rec[f"plate_{tag}_mean"] = round(float(a.mean()), 6)
        rec["default_matches_explicit"] = rec["plate_default"] == rec["plate_default_explicit"]
        rec["custom_differs"] = rec["plate_custom"] != rec["plate_default"]
        # and going back to the default must return the ORIGINAL plate, not a stale context
        again = stub._bake_plate(96, floor=None).detach().cpu().numpy()
        rec["default_restored_after_custom"] = (
            hashlib.sha256(np.ascontiguousarray(again).tobytes()).hexdigest() == rec["plate_default"])
        out[cell] = rec
    print("@@JSON@@" + json.dumps(out, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))
    ok = all(r["default_matches_explicit"] and r["custom_differs"]
             and r["default_restored_after_custom"] and r["key_default_is_none"]
             and r["key_none_is_none"] and r["key_nearby_is_none"] for r in out.values())
    print("PLATE PROBE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

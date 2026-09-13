"""What actually makes ``OGBenchMaze._bake`` non-reproducible?

Ticket 1 attributed it to ``qpos``: ``_bake`` renders the shared ``MjData`` without resetting the
pose, so the plate would carry whatever ``_fk`` left behind. Resetting to ``qpos0`` inside ``_bake``
(RAYCAST-2c) changed NOTHING -- the same two ant mazes still fail, byte for byte the same pixels --
so the attribution was wrong and this finds the real carrier instead of guessing again.

It separates four candidates by construction:

  1. **The bake is simply nondeterministic.** Bake the same key N times in a row with nothing in
     between. If the plate moves, no amount of resetting state will help.
  2. **A RENDER changes the next bake.** Bake, render, bake, render, bake. If only the bakes that
     follow a render move, something the render does reaches the rasterizer.
  3. **The FIRST bake of a process is special.** The construction-time bake happens before any CUDA
     context exists; every later one does not.
  4. **It is not the plate.** Every other scene field a bake produces -- ``plate_t``, the camera, the
     lights, the agent's base colours -- is digested too, so "the frame moved" can be located.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# $PYTHONPATH DECIDES WHICH stable_envs IS MEASURED. A before/after run puts the baseline copy on
# PYTHONPATH for one leg and this branch for the other, so this file's own repo root must NOT be
# inserted ahead of it -- doing that made both legs import the branch and report a 1.00x speed-up as
# a real one. Appended, so the helper imports still resolve when PYTHONPATH is unset.
sys.path.append(str(ROOT))


def dig(x) -> str:
    import numpy as np
    import torch
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]


def scene_digests(sc) -> dict:
    import numpy as np
    return {
        "plate": dig(sc.plate), "plate_t": dig(sc.plate_t),
        "cam": dig(np.asarray(list(sc.cam.pos) + list(sc.cam.forward) + list(sc.cam.up)
                              + [sc.cam.near, sc.cam.top, sc.cam.bottom, sc.cam.center], np.float64)),
        "lights": dig(np.asarray(sc.lights.dirs, np.float64)),
        "agent_base": dig(np.asarray(sc.agent_base, np.float64)),
    }


def px_diff(a, b):
    import numpy as np
    A = np.asarray(a, np.int32)
    B = np.asarray(b, np.int32)
    d = np.abs(A - B)
    m = d.max(-1)                       # (H,W) for a plate, (B,H,W) for a frame stack
    flat = m if m.ndim == 2 else m.max(tuple(range(m.ndim - 2)))     # collapse to one (H,W) mask
    ys, xs = np.nonzero(flat)
    return {"max": int(d.max()), "n_px": int((m > 0).sum()),
            "bbox_y0x0y1x1": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
            if len(ys) else None}


def probe(cfg, device, size):
    import numpy as np
    import stable_envs as se
    import stable_envs.envs.ogbench as OGB

    rep = {"config": cfg, "stable_envs_from": se.__file__}
    env = se.make(cfg, batch=4, size=size, device=device)
    npz = ROOT / "examples/results/step4/states" / f"{cfg}.npz"
    pool = np.load(npz)["states"] if npz.is_file() else env.sample_states(8)
    stA = np.asarray(pool[:4], np.float32)
    stB = np.asarray(pool[4:8], np.float32)

    # 0. the bake the constructor already did, before any render and before any CUDA work here
    sc = env._scene(env.size)
    rep["bake_at_construction"] = scene_digests(sc)
    plate0 = sc.plate.cpu().numpy()

    def rebake(tag, plates):
        OGB._SCENE_CACHE.clear()
        s = env._scene(env.size)
        d = scene_digests(s)
        p = s.plate.cpu().numpy()
        d["plate_vs_construction"] = px_diff(plate0, p)
        plates[tag] = p
        return d

    plates: dict = {}
    seq = {}
    # 1. N bakes in a row, nothing in between
    for i in range(3):
        seq[f"rebake_{i}_no_render_between"] = rebake(f"n{i}", plates)
    # 2. alternate render / bake
    env.render(stA)
    seq["rebake_after_render_A"] = rebake("rA", plates)
    env.render(stB)
    seq["rebake_after_render_B"] = rebake("rB", plates)
    env.render(stA)
    seq["rebake_after_render_A_again"] = rebake("rA2", plates)
    # 3. a bake with NO render since the previous bake, but after renders have happened at all
    seq["rebake_quiet_after_renders"] = rebake("q", plates)
    rep["sequence"] = seq

    # which pairs differ
    keys = list(plates)
    rep["pairwise_plate_max_abs"] = {
        f"{a}|{b}": px_diff(plates[a], plates[b])["max"]
        for i, a in enumerate(keys) for b in keys[i + 1:]}
    rep["distinct_plates"] = len({dig(v) for v in plates.values()})
    rep["verdict"] = {
        "bake_is_nondeterministic_with_nothing_between":
            len({dig(plates[k]) for k in ("n0", "n1", "n2")}) > 1,
        "a_render_changes_the_next_bake":
            dig(plates["n2"]) != dig(plates["rA"]) or dig(plates["rA"]) != dig(plates["rB"]),
        "construction_bake_is_special": any(
            v["plate_vs_construction"]["max"] for v in seq.values()),
    }

    # 4. and the public symptom, for the same env
    a = env.render(stA).cpu().numpy()
    OGB._SCENE_CACHE.clear()
    b = env.render(stA).cpu().numpy()
    rep["public_render_after_cache_clear"] = px_diff(a, b)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    rep = {"host": socket.gethostname(), "device": args.device,
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "code_root": os.environ.get("PYTHONPATH", ""), "configs": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for cfg in args.configs:
        try:
            rep["configs"][cfg] = probe(cfg, args.device, args.size)
        except Exception as exc:                                            # noqa: BLE001
            import traceback
            rep["configs"][cfg] = {"error": repr(exc)[:300],
                                   "tb": traceback.format_exc()[-1200:]}
        print(json.dumps({cfg: rep["configs"][cfg]}, indent=2)[:5000], flush=True)
        args.out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

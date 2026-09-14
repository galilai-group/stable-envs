"""Why does the same bake key produce two plates, and which one did Step 4 freeze?

``bake_probe.py`` established THAT the plate alternates. Its digest sequences then turned out to be
byte-identical between two processes on two different code trees and, for the teleport key, between
two nodes -- so the output is a deterministic function of the CALL HISTORY, not a coin flip:

    AntmazeGiantNavigate       A B A B A B A B     (strict alternation)
    AntmazeTeleportExplore     A A A B A A A A
    HumanoidmazeMediumNavigate A A A A A A A A

``OGBenchMaze._bake`` keeps one ``mujoco.Renderer`` per texture state forever and drives it
``update_scene -> render -> update_scene`` per bake -- two scene updates and one render, so successive
bakes do not start from the same renderer state. This asks two questions that can be answered without
changing the renderer:

  (i)  **Which plate did Step 4 freeze?** Bake the key repeatedly and, for each distinct plate,
       replay the frozen states and digest the frames. If one of them reproduces the frozen
       ``digest_ours`` then the Step-4 record is internally consistent and what it froze was simply a
       NON-FIRST bake -- which is what a session that ran the whole check suite before the oracle
       would produce.
  (ii) **Is the renderer's lifetime the mechanism?** Replay the shot sequence four ways -- as shipped,
       with a fresh Renderer per bake, with a discarded warm-up render, and without the trailing
       ``update_scene`` -- and report which one makes the sequence constant.

REPORT ONLY. Nothing here edits ``stable_envs``; the variants drive MuJoCo directly through the same
cached objects ``_bake`` uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))


def _root_tag(p: str) -> str:
    return re.sub(r"^/oscar/scratch/([^/]+)", r"$SCR", p)


def dig(x) -> str:
    import numpy as np
    import torch
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]


def letters(seq):
    m, out = {}, []
    for s in seq:
        m.setdefault(s, chr(65 + len(m)))
        out.append(m[s])
    return "".join(out), m


# ------------------------------------------------------------------ (i) which plate did Step 4 freeze?

def which_plate_step4_froze(cfg, device, n_bakes):
    import numpy as np
    import stable_envs as se
    import stable_envs.envs.ogbench as OGB
    from stable_envs.testing import conformance as C

    step4 = json.loads((ROOT / "examples/results/step4/conformance/ogb.json").read_text())
    frozen = step4["configs"][cfg]["oracle"]["digest_ours"]
    states = np.asarray(np.load(ROOT / "examples/results/step4/states" / f"{cfg}.npz")["states"],
                        np.float32)
    env = se.make(cfg, batch=len(states), size=None, device=device)

    rows = []
    for i in range(n_bakes):
        OGB._SCENE_CACHE.clear()
        sc = env._scene(env.size)                       # bake number i
        pd = dig(sc.plate)
        fd = C.frames_digest(env.render(states))        # the cached scene, so THIS plate
        rows.append({"bake": i, "plate": pd, "frames_digest": fd,
                     "matches_frozen": fd == frozen})
        print(f"    bake {i}: plate {pd}  frames {fd[:16]}  frozen_match={fd == frozen}", flush=True)
    seq, _ = letters([r["plate"] for r in rows])
    return {"config": cfg, "frozen_digest_ours": frozen, "plate_sequence": seq, "bakes": rows,
            "any_bake_reproduces_frozen": any(r["matches_frozen"] for r in rows),
            "distinct_plates": len({r["plate"] for r in rows}),
            "distinct_frame_digests": len({r["frames_digest"] for r in rows})}


# ------------------------------------------------------------------ (ii) is it the renderer lifetime?

def _shot(m, d, r, cam, agent, mode):
    """One bake's shot sequence. ``mode`` selects the variant under test."""
    import numpy as np
    keep = np.array(m.geom_rgba[agent], copy=True)
    m.geom_rgba[agent, 3] = 0.0
    if mode == "warmup_render":
        r.update_scene(d, camera=cam)
        r.render()                                      # discarded
    r.update_scene(d, camera=cam)
    plate = r.render().copy()
    m.geom_rgba[agent] = keep
    if mode != "no_trailing_update":
        r.update_scene(d, camera=cam)                   # what _bake does, for the camera/lights
    return plate


def renderer_lifetime(loco, maze, size, n_bakes, max_fresh):
    import mujoco
    import stable_envs.envs.ogbench as OGB

    env, u, pr = OGB._scene_env(loco, maze, size)
    m, d = u.model, u.data
    cam = u.custom_camera
    agent = pr["agent"]
    d.qpos[:] = m.qpos0
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    out = {}
    kept = pr["renderers"].get(0)
    if kept is None:
        kept = pr["renderers"][0] = mujoco.Renderer(m, width=size, height=size)

    for mode in ("as_shipped", "warmup_render", "no_trailing_update"):
        seq = [dig(_shot(m, d, kept, cam, agent, mode)) for _ in range(n_bakes)]
        out[mode] = {"sequence": letters(seq)[0], "digests": seq,
                     "distinct": len(set(seq)), "renderer": "kept (as _bake does)"}
        print(f"    {mode:22s} {out[mode]['sequence']}  distinct={out[mode]['distinct']}", flush=True)

    # A FRESH Renderer per bake LEAKS an EGL context each time -- the module note above OGBenchMaze
    # records that releasing one corrupts every other live renderer in the process, so they are never
    # closed. Capped, and the cost is reported rather than hidden.
    fresh_seq = []
    for _ in range(min(n_bakes, max_fresh)):
        rr = mujoco.Renderer(m, width=size, height=size)
        fresh_seq.append(dig(_shot(m, d, rr, cam, agent, "as_shipped")))
    out["fresh_renderer_per_bake"] = {
        "sequence": letters(fresh_seq)[0], "digests": fresh_seq, "distinct": len(set(fresh_seq)),
        "renderer": f"a new mujoco.Renderer per bake ({len(fresh_seq)} EGL contexts leaked)"}
    print(f"    {'fresh_renderer_per_bake':22s} {out['fresh_renderer_per_bake']['sequence']}  "
          f"distinct={out['fresh_renderer_per_bake']['distinct']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", nargs="+", default=["AntmazeTeleportExplore",
                                                     "AntmazeGiantNavigate"])
    ap.add_argument("--scenes", nargs="+", default=["ant/teleport", "ant/giant"])
    ap.add_argument("--bakes", type=int, default=8)
    ap.add_argument("--max-fresh", type=int, default=6)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import stable_envs as se                                             # noqa: F401
    rep = {"host": socket.gethostname(), "device": args.device,
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "stable_envs_from": _root_tag(se.__file__),
           "which_plate_step4_froze": {}, "renderer_lifetime": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for cfg in args.configs:
        print(f"(i) {cfg}", flush=True)
        try:
            rep["which_plate_step4_froze"][cfg] = which_plate_step4_froze(
                cfg, args.device, args.bakes)
        except Exception as exc:                                         # noqa: BLE001
            import traceback
            rep["which_plate_step4_froze"][cfg] = {"error": repr(exc)[:300],
                                                   "tb": traceback.format_exc()[-1000:]}
            print("    ERROR", repr(exc)[:160], flush=True)
        args.out.write_text(json.dumps(rep, indent=2))

    for s in args.scenes:
        loco, maze = s.split("/")
        print(f"(ii) {s}", flush=True)
        try:
            rep["renderer_lifetime"][s] = renderer_lifetime(
                loco, maze, args.size, args.bakes, args.max_fresh)
        except Exception as exc:                                         # noqa: BLE001
            import traceback
            rep["renderer_lifetime"][s] = {"error": repr(exc)[:300],
                                           "tb": traceback.format_exc()[-1000:]}
            print("    ERROR", repr(exc)[:160], flush=True)
        args.out.write_text(json.dumps(rep, indent=2))

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

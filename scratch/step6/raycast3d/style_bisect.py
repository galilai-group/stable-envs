"""Step 6C ticket 1, deliverable 4 — why ``check_style_is_state`` observes 1 on two OGBench cells.

REPORT ONLY. Nothing here patches anything. It runs the frozen check from
``stable_envs/testing/conformance.py`` and then walks a ladder of controlled variants until the
difference appears, so the cause is demonstrated rather than argued.

The frozen check is:

    a = env.render(st, style=sty)                      # the env that has already been used
    fresh = type(env)(...); fresh.set_state(st, style=sty); b = fresh.render()

and it asserts ``max|a-b| == 0``. Two things differ between the two sides and only one of them is
about style: WHICH ENV OBJECT renders, and WHETHER the style arrived through ``render(style=)`` or
through ``set_state(style=)``. Sections 2-5 separate them, and then separate both from the
module-level ``_SCENE_CACHE``, which ``type(env)(...)`` touches on construction and which clears
itself WHOLESALE at ``_SCENE_CACHE_MAX`` entries.

HYPOTHESIS UNDER TEST: the baked plate is not a pure function of (scene, style).
``OGBenchMaze._bake`` renders the SHARED ``MjData`` without resetting ``qpos``, so a plate baked
after a render carries whatever pose the last ``_fk`` left in the model. The plate is then cached,
so the impurity is invisible until the cache evicts and the scene is baked a second time from a
different leftover pose. ``DMControlSuite._bake`` DOES reset (``d.qpos[:] = m.qpos0`` then
``mj_forward``) — which is the control, and is why the DM Control shard passes this check.

Section ``plate_purity`` is the decisive one and is deliberately CHEAP: it uses the default style,
so one scene is baked rather than one per sampled appearance.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def dig(x):
    import hashlib
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def maxdiff(a, b):
    import numpy as np
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


def where_diff(a, b):
    import numpy as np
    d = np.abs(a.astype(np.int32) - b.astype(np.int32)).max(-1)
    flat = d.reshape(len(d), -1)
    rows = [int(i) for i in range(len(flat)) if flat[i].max() > 0]
    out = {"rows_differing": len(rows), "rows": rows[:8],
           "px_differing_total": int((d > 0).sum()), "max": int(d.max())}
    if rows:
        r = rows[0]
        ys, xs = np.nonzero(d[r])
        out["first_row"] = {"row": r, "n_px": int(len(ys)),
                            "bbox_y0x0y1x1": [int(ys.min()), int(xs.min()),
                                              int(ys.max()), int(xs.max())],
                            "max": int(d[r].max())}
    return out


def guard(rep, name, fn):
    """Run one probe; a probe that raises records its exception instead of losing the ladder."""
    try:
        rep[name] = fn()
    except Exception as exc:                                            # noqa: BLE001
        rep[name] = {"error": repr(exc)[:300]}
        rep.setdefault("_tracebacks", {})[name] = traceback.format_exc()[-1200:]
    return rep.get(name)


def plate_purity(cfg, device, size):
    """Is the baked plate a pure function of (scene, style)?  Default style, so ONE scene.

    The public path is used to disturb the shared ``MjData``: ``env.render(states)`` runs ``_fk``,
    which writes ``qpos`` and runs ``mj_kinematics`` on ``sc.data``. Then the module cache is
    evicted the way it evicts itself at ``_SCENE_CACHE_MAX``, and the SAME key is baked again.
    """
    import mujoco
    import numpy as np
    import stable_envs as se
    import stable_envs.envs.ogbench as OGB

    env = se.make(cfg, batch=8, size=size, device=device)
    npz = ROOT / "examples/results/step4/states" / f"{cfg}.npz"
    pool = np.load(npz)["states"] if npz.is_file() else env.sample_states(64)
    stA = np.asarray(pool[np.arange(8) % len(pool)], np.float32)
    stB = np.asarray(pool[np.arange(8, 16) % len(pool)], np.float32)

    OGB._SCENE_CACHE.clear()
    sc0 = env._scene(env.size)
    p0, t0 = sc0.plate.cpu().numpy(), sc0.plate_t.cpu().numpy()
    qpos_at_first_bake = np.array(sc0.data.qpos, copy=True)

    env.render(stA)                                    # the public path leaves a pose in sc.data
    qpos_after_A = np.array(sc0.data.qpos, copy=True)
    OGB._SCENE_CACHE.clear()
    scA = env._scene(env.size)
    pA, tA = scA.plate.cpu().numpy(), scA.plate_t.cpu().numpy()

    env.render(stB)
    qpos_after_B = np.array(sc0.data.qpos, copy=True)
    OGB._SCENE_CACHE.clear()
    scB = env._scene(env.size)
    pB, tB = scB.plate.cpu().numpy(), scB.plate_t.cpu().numpy()

    # control: reset the shared data to the model's own reference pose and bake a third time
    scB.data.qpos[:] = scB.model.qpos0
    mujoco.mj_forward(scB.model, scB.data)
    OGB._SCENE_CACHE.clear()
    sc0b = env._scene(env.size)
    p0b = sc0b.plate.cpu().numpy()

    return {
        "qpos_at_first_bake": qpos_at_first_bake[:7].tolist(),
        "qpos_after_render_A": qpos_after_A[:7].tolist(),
        "qpos_after_render_B": qpos_after_B[:7].tolist(),
        "plate_digest_first": dig(p0), "plate_digest_after_A": dig(pA),
        "plate_digest_after_B": dig(pB), "plate_digest_reset_to_qpos0": dig(p0b),
        "plate_A_vs_first": where_diff(pA[None], p0[None]),
        "plate_B_vs_A": where_diff(pB[None], pA[None]),
        "plate_reset_vs_first": where_diff(p0b[None], p0[None]),
        "plate_t_max_abs_A_vs_first": float(abs(tA - t0).max()),
        "plate_t_max_abs_B_vs_A": float(abs(tB - tA).max()),
        "verdict_plate_depends_on_history": dig(pA) != dig(p0) or dig(pB) != dig(pA),
        "verdict_reset_restores_plate": dig(p0b) == dig(p0),
    }


def frozen_check(cfg, device, batch, size):
    """The frozen check plus the four-way separation of env-object from style-path."""
    import stable_envs as se
    import stable_envs.envs.ogbench as OGB

    rep: dict = {}
    env = se.make(cfg, batch=batch, size=size, device=device)
    st = env.sample_states(env.B)
    sty = env.sample_style(env.B)
    rep["style_dim"] = int(env.style_dim)
    rep["plate_paths"] = list(env.PLATE_PATHS)
    rep["scene_cache_len_before"] = len(OGB._SCENE_CACHE)
    rep["scene_cache_max"] = OGB._SCENE_CACHE_MAX

    a = env.render(st, style=sty).cpu().numpy()
    rep["a_digest"] = dig(a)

    def fresh_check():
        f = type(env)(batch=env.B, size=env.size, supersample=env.ss, device=str(env.device))
        f.set_state(st, style=sty)
        b = f.render().cpu().numpy()
        return {"max_abs_diff": maxdiff(a, b), "where": where_diff(a, b)}
    guard(rep, "check_style_is_state", fresh_check)
    rep["scene_cache_len_after_check"] = len(OGB._SCENE_CACHE)

    guard(rep, "same_env_rendered_twice", lambda: {
        "max_abs_diff": maxdiff(a, env.render(st, style=sty).cpu().numpy())})
    guard(rep, "fresh_env_style_via_render", lambda: {
        "max_abs_diff": maxdiff(a, type(env)(batch=env.B, size=env.size, supersample=env.ss,
                                             device=str(env.device))
                                .render(st, style=sty).cpu().numpy())})
    guard(rep, "same_env_style_via_set_state", lambda: {
        "max_abs_diff": maxdiff(a, env.set_state(st, style=sty).render().cpu().numpy())})

    def after_clear():
        OGB._SCENE_CACHE.clear()
        c = env.render(st, style=sty).cpu().numpy()
        return {"max_abs_diff": maxdiff(a, c), "where": where_diff(a, c)}
    guard(rep, "same_env_after_scene_cache_clear", after_clear)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--skip-frozen-check", action="store_true",
                    help="run only the cheap plate-purity probe")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    report = {"host": socket.gethostname(), "device": args.device, "batch": args.batch,
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "configs": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for cfg in args.configs:
        rec: dict = {"config": cfg}
        guard(rec, "plate_purity", lambda c=cfg: plate_purity(c, args.device, args.size))
        if not args.skip_frozen_check:
            guard(rec, "frozen_check",
                  lambda c=cfg: frozen_check(c, args.device, args.batch, args.size))
        report["configs"][cfg] = rec
        print(json.dumps({cfg: rec}, indent=2)[:6000], flush=True)
        args.out.write_text(json.dumps(report, indent=2))       # written after EVERY config
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

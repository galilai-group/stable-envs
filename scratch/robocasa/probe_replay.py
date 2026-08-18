"""How compact can the *scene* half of the state be, before the frame stops matching?

The pose half is settled: with a fixed compiled model, restoring qpos/qvel reproduces the frame
bit-for-bit. The open question is the other half -- what has to be carried alongside qpos so the
model can be rebuilt. RoboCasa offers two candidates and its own code picks one of them, which is
itself evidence:

  A. ``ep_meta``          -- a ~8 KB JSON (layout_id, style_id, object_cfgs, fixture_refs,
                             gen_textures, cam_configs). This is what a "scene id + variation"
                             would look like if it existed.
  B. ``model_file``       -- the full expanded MJCF, ~400-700 KB per episode. This is what
                             robomimic's ``reset_to`` actually restores, *after* also applying
                             ep_meta -- so the authors did not consider A sufficient.

Both are replayed here against a reference episode and scored on identical pixels, so "A is not
enough" is a measurement rather than an inference from the fact that B exists.

Also asks the batching question that decides the whole thesis: across episodes, does the compiled
model stay fixed (one kernel, shared meshes/textures, thousands of envs) or change shape?
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def sha(s):
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PickPlaceCounterToSink")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--n-fixed", type=int, default=4, help="resets with layout+style pinned")
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import imageio.v2 as imageio
    from robocasa.utils.env_utils import create_env

    env = create_env(env_name=args.env_name, robots="PandaOmron",
                     camera_names=["robot0_agentview_left"],
                     camera_widths=args.res, camera_heights=args.res,
                     seed=0, render_onscreen=False)
    cam = "robot0_agentview_left"

    def render():
        return env.sim.render(width=args.res, height=args.res, camera_name=cam)[::-1].copy()

    rec = {}

    # ---- reference episode -------------------------------------------------------------------
    env.reset()
    ep_meta = env.get_ep_meta()
    ref_xml = env.sim.model.get_xml()
    ref_state = env.sim.get_state().flatten().copy()
    ref_frame = render()
    ref = {"xml_sha": sha(ref_xml), "nq": int(env.sim.model._model.nq),
           "ngeom": int(env.sim.model._model.ngeom),
           "layout": int(env.layout_id), "style": int(env.style_id)}
    rec["reference"] = ref
    imageio.imwrite(os.path.join(args.out, "ref_frame.png"), ref_frame)

    ep_json = json.dumps(ep_meta, default=str)
    rec["descriptor_sizes"] = {
        "ep_meta_json_bytes": len(ep_json),
        "ep_meta_gzip_bytes": len(gzip.compress(ep_json.encode())),
        "model_xml_bytes": len(ref_xml),
        "model_xml_gzip_bytes": len(gzip.compress(ref_xml.encode())),
        "qpos_float32_bytes_per_step": int(env.sim.model._model.nq) * 4,
    }

    def score(tag, frame, nq, xml):
        d = (np.abs(frame.astype(np.int16) - ref_frame.astype(np.int16))
             if frame is not None and frame.shape == ref_frame.shape else None)
        r = {"nq": nq, "nq_matches": nq == ref["nq"], "xml_matches": sha(xml) == ref["xml_sha"]}
        if d is not None:
            r.update({"px_maxdiff": int(d.max()), "px_meandiff": float(d.mean()),
                      "px_exact_frac": float((d.max(-1) == 0).mean())})
        else:
            r.update({"px_maxdiff": None, "note": "shape mismatch / no frame"})
        rec[tag] = r
        print(f"{tag}: {json.dumps(r)}")
        return frame

    # ---- A: ep_meta only ---------------------------------------------------------------------
    env.set_ep_meta(ep_meta)
    env.reset()
    xa = env.sim.model.get_xml()
    fa = None
    if env.sim.model._model.nq == ref["nq"]:
        env.sim.set_state_from_flattened(ref_state)
        env.sim.forward()
        fa = render()
    fa = score("A_ep_meta_only", fa, int(env.sim.model._model.nq), xa)
    if fa is not None:
        imageio.imwrite(os.path.join(args.out, "A_ep_meta_only.png"), fa)

    # ---- B: official reset_to path (ep_meta + full model xml) --------------------------------
    env.set_ep_meta(ep_meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(ref_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(ref_state)
    env.sim.forward()
    fb = render()
    score("B_ep_meta_plus_xml", fb, int(env.sim.model._model.nq), env.sim.model.get_xml())
    imageio.imwrite(os.path.join(args.out, "B_ep_meta_plus_xml.png"), fb)
    imageio.imwrite(os.path.join(args.out, "B_diff_x8.png"),
                    np.clip(np.abs(fb.astype(np.int16) - ref_frame.astype(np.int16)) * 8, 0, 255).astype(np.uint8))

    # ---- batching question: is the model shape stable if layout+style are pinned? -------------
    env.unset_ep_meta()
    env2 = create_env(env_name=args.env_name, robots="PandaOmron",
                      camera_names=[cam], camera_widths=args.res, camera_heights=args.res,
                      seed=0, render_onscreen=False,
                      layout_ids=ref["layout"], style_ids=ref["style"])
    fixed = []
    for _ in range(args.n_fixed):
        env2.reset()
        mm = env2.sim.model._model
        fixed.append({"nq": int(mm.nq), "ngeom": int(mm.ngeom), "nmesh": int(mm.nmesh),
                      "nmeshface": int(mm.nmeshface), "ntex": int(mm.ntex),
                      "texdata_MB": float(mm.tex_data.size / 1e6),
                      "xml_sha": sha(env2.sim.model.get_xml()),
                      "layout": int(env2.layout_id), "style": int(env2.style_id)})
        print("pinned-layout reset:", fixed[-1])
    rec["pinned_layout_style_resets"] = fixed
    rec["pinned_model_shape_stable"] = len({(f["nq"], f["ngeom"], f["nmesh"]) for f in fixed}) == 1
    rec["pinned_model_identical"] = len({f["xml_sha"] for f in fixed}) == 1

    print("\n" + json.dumps({k: rec[k] for k in
          ("reference", "descriptor_sizes", "pinned_model_shape_stable", "pinned_model_identical")},
          indent=2))
    with open(os.path.join(args.out, "replay.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)


if __name__ == "__main__":
    main()

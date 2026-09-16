"""GPU evidence for the Fetch VariationMaps: the default must not move, a mapped leaf must.

Run inside a SLURM GPU job. Writes ``examples/results/step8/variations-fetch/gpu_evidence.json``.

Three claims, at the PIXEL level rather than at the style-row level (the style-row versions live in
``tests/unit/test_variation_maps_fetch.py`` and run without a GPU):

1. ``render(style=style_from_variation(env, {}))`` is byte-identical to ``render()`` — the registered
   default is exactly reproducible through the mapping.
2. every leaf we declare unmapped, pushed through the mapping, is byte-identical to the default —
   an unsupported leaf cannot leak into a frame.
3. every mapped leaf moves pixels, and the swm->ours unit conversion lands where it should:
   ``light.intensity = 0.8`` (the model's own diffuse) must reproduce the default frame exactly,
   which is the whole justification for the 1.25 factor.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch

import stable_envs as se
from stable_envs.compat import worldmodel as WM
from stable_envs.compat.variation_maps import fetch as FM

SCENES = ("Reach", "Push", "Slide", "PickAndPlace")
SIZE = int(os.environ.get("SE8_SIZE", "128"))
NSTATES = int(os.environ.get("SE8_STATES", "8"))
OUT = os.path.join("examples", "results", "step8", "variations-fetch")


def _cmp(a, b):
    d = (a.to(torch.int16) - b.to(torch.int16)).abs()
    return dict(maxdiff=int(d.max()), frac_changed=float((d.max(-1).values > 0).to(torch.float32).mean()),
                mean=float(d.to(torch.float32).mean()))


def provenance():
    def sh(c):
        try:
            return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    import mujoco
    import mujoco_warp
    # rsync drops .git, so on Oscar `git rev-parse` returns nothing and the artifact came back with
    # a null SHA (job 6451948). The sync writes the SHA to .git_sha on purpose and _env.sh exports
    # it; read those FIRST and fall back to git only in a real checkout.
    sha = (os.environ.get("STABLE_ENVS_GIT_COMMIT")
           or (open(".git_sha").read().strip() if os.path.exists(".git_sha") else None)
           or sh("git rev-parse HEAD"))
    return dict(
        git_sha=sha, git_branch=sh("git rev-parse --abbrev-ref HEAD"),
        host=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_nodelist=os.environ.get("SLURM_JOB_NODELIST"),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        cuda=torch.version.cuda, torch=torch.__version__, python=sys.version.split()[0],
        mujoco=mujoco.__version__, mujoco_warp=getattr(mujoco_warp, "__version__", "unknown"),
        warp=__import__("warp").config.version, numpy=np.__version__,
        size=SIZE, n_states=NSTATES)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"provenance": provenance(), "scenes": {}}
    unmapped_values = {
        "camera.angle_delta": [[10.0, 10.0]], "rendering.transparent_arm": 1,
        "block.mass": [40.0], "agent.start_position": [1.45, 0.9],
        "goal.start_position": [1.45, 0.9, 0.4247], "block.start_position": [1.45, 0.9],
        "block.angle": [1.5], "object.color": [0.05, 0.95, 0.2]}
    mapped_values = {"table.color": [0.9, 0.2, 0.05], "background.color": [0.9, 0.05, 0.9],
                     "object.color": [0.05, 0.95, 0.2], "light.intensity": [0.35]}
    for scene in SCENES:
        env = se.make(f"Fetch{scene}", batch=NSTATES, size=SIZE, device="cuda")
        m = WM.variation_map(f"Fetch{scene}")
        st = env.sample_states(NSTATES, rng=np.random.default_rng(0))
        default = np.asarray(env.default_style(), np.float32)
        base = env.render(st, style=default[None].repeat(NSTATES, 0)).clone()
        row = {"swm_id": m.swm_id, "our_leaves": sorted(env.variation_space.names())}

        through = WM.style_from_variation(env, {})
        row["default_through_empty_mapping"] = dict(
            style_byte_identical=bool(np.array_equal(through, default)),
            pixels=_cmp(base, env.render(st, style=through[None].repeat(NSTATES, 0))))

        cfg = WM.variation_from_style(env, default)
        row["round_trip"] = dict(
            light_intensity=float(np.asarray(cfg["light.intensity"]).reshape(-1)[0]),
            style_byte_identical=bool(np.array_equal(
                WM.style_from_variation(env, cfg), default)),
            pixels=_cmp(base, env.render(st, style=WM.style_from_variation(env, cfg)[None]
                                         .repeat(NSTATES, 0))))

        # swm's light.intensity at the model's OWN diffuse must reproduce the default frame.
        s = WM.style_from_variation(env, {"light.intensity": [FM.LIGHT_DIFFUSE_0]})
        row["light_intensity_at_model_diffuse"] = dict(
            style_byte_identical=bool(np.array_equal(s, default)),
            pixels=_cmp(base, env.render(st, style=s[None].repeat(NSTATES, 0))))

        row["unmapped_leaves_move_nothing"] = {}
        for leaf in sorted(m.swm_only):
            if leaf not in unmapped_values:
                continue
            s = WM.style_from_variation(env, {leaf: unmapped_values[leaf]})
            row["unmapped_leaves_move_nothing"][leaf] = dict(
                style_byte_identical=bool(np.array_equal(s, default)),
                pixels=_cmp(base, env.render(st, style=s[None].repeat(NSTATES, 0))))

        row["mapped_leaves_move_pixels"] = {}
        inv = {t: o for o, (t, _) in m.pairs().items()}
        layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
        for leaf, val in mapped_values.items():
            if leaf not in inv:
                continue
            s = WM.style_from_variation(env, {leaf: val})
            o, w = layout[inv[leaf]]
            rest = np.ones(s.shape, bool)
            rest[o:o + w] = False
            row["mapped_leaves_move_pixels"][leaf] = dict(
                ours=inv[leaf], only_its_own_slot_moved=bool(np.array_equal(s[rest], default[rest])),
                slot_moved=bool(not np.array_equal(s[o:o + w], default[o:o + w])),
                pixels=_cmp(base, env.render(st, style=s[None].repeat(NSTATES, 0))))
        res["scenes"][scene] = row
        print(scene, json.dumps(row)[:400], flush=True)

    ok = all(
        r["default_through_empty_mapping"]["style_byte_identical"]
        and r["default_through_empty_mapping"]["pixels"]["maxdiff"] == 0
        and r["round_trip"]["style_byte_identical"]
        and r["light_intensity_at_model_diffuse"]["pixels"]["maxdiff"] == 0
        and all(v["pixels"]["maxdiff"] == 0 for v in r["unmapped_leaves_move_nothing"].values())
        and all(v["pixels"]["maxdiff"] > 0 and v["only_its_own_slot_moved"]
                for v in r["mapped_leaves_move_pixels"].values())
        for r in res["scenes"].values())
    res["verdict"] = "PASS" if ok else "FAIL"
    with open(os.path.join(OUT, "gpu_evidence.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("VERDICT", res["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

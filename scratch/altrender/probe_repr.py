"""Would a model notice? Measured in a frozen pretrained encoder, against a scale and two controls.

"A model would not care" is a hypothesis, and the honest version of this test in this repo is a
pretrained world model run with its own evaluation unchanged and only the pixels swapped. **No such
model exists for RoboCasa or HumanoidBench** -- the PushT precedent (``project_pusht_showcase``) does
not transfer, and training one to answer this question would cost more than the whole investigation.
So this is the cheap proxy, and it is labelled as a proxy: the frames go through a frozen pretrained
encoder and what is reported is how far the approximation moves the representation **relative to how
far a different state moves it**.

That ratio is the quantity that matters. A representation that is 2 % perturbed by our renderer and
100 % perturbed by the next state in the trajectory is one a downstream head cannot plausibly be
confused by; one that is 60 % perturbed is. The scale is measured, not assumed.

Two controls, because a test that cannot fail proves nothing:

* **blank** -- a mid-grey frame. If a blank frame does not move the representation much further than
  our renderer does, the encoder is not reading the image and the whole test is void.
* **jpeg** -- JPEG q95 of the official frame, i.e. the storage baseline. This is the comparison the
  argument rests on: if our renderer perturbs the representation LESS than the stored dataset does,
  then a model trained on stored frames has already accepted a larger perturbation than the one being
  proposed.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from probe_budget_frontier import config_list, env_module, jpeg_roundtrip, patched, upsample

#: Cached locally under ~/.cache/huggingface/hub, so this runs offline on a compute node.
ENCODERS = ("vit_base_patch14_dinov2.lvd142m", "resnet18.a1_in1k",
            "vit_base_patch16_dinov3.lvd1689m")


def embed(model, cfg, frames: np.ndarray, bs: int = 32) -> torch.Tensor:
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    size = cfg["input_size"][1]
    outs = []
    for i in range(0, len(frames), bs):
        x = torch.as_tensor(frames[i:i + bs], device="cuda").permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic",
                                            align_corners=False, antialias=True)
        x = (x - mean) / std
        with torch.no_grad():
            outs.append(model(x).float())
    z = torch.cat(outs)
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--configs", default="shipped,r128")
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()

    import timm
    import stable_envs as se
    cls = se.REGISTRY[a.env]
    mod = env_module(a.env)
    family = "RoboCasa" if a.env.startswith("RoboCasa") else "HumanoidBench"

    ref = cls.reference_env(a.size)
    probe = se.make(a.env, batch=1, size=a.size, device="cuda")
    states = probe.sample_states(a.states)
    official = cls.reference_render(ref, states)
    del probe
    mod._CTX_CACHE.clear()
    torch.cuda.empty_cache()

    variants = {"jpeg_q95": jpeg_roundtrip(official, 95)[0],
                "blank": np.full_like(official, 128)}
    for name in a.configs.split(","):
        r, m, ov, lod = config_list(family)[name]
        orig = patched(mod, ov, lod)
        try:
            env = se.make(a.env, batch=1, size=a.size, device="cuda")
            variants[name] = upsample(env.render(states, size=r), a.size, m).cpu().numpy()
            del env
        finally:
            mod.MW.create_render_context = orig
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()

    out = {"env": a.env, "n_states": a.states, "encoders": {}}
    for enc in ENCODERS:
        try:
            model = timm.create_model(enc, pretrained=True, num_classes=0).eval().cuda()
            cfg = timm.data.resolve_data_config({}, model=model)
            z0 = embed(model, cfg, official)
            # The SCALE: how far apart are two different states in this representation?
            perm = torch.randperm(len(z0))
            scale = float((1.0 - (z0 * z0[perm]).sum(-1)).mean())
            row = {"state_to_state_cosine_dist": scale, "input_size": cfg["input_size"][1]}
            for k, v in variants.items():
                d = float((1.0 - (z0 * embed(model, cfg, v)).sum(-1)).mean())
                row[k] = {"cosine_dist": d, "frac_of_state_change": d / scale if scale else None}
            out["encoders"][enc] = row
        except Exception as e:                                        # noqa: BLE001
            out["encoders"][enc] = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps({enc: out["encoders"][enc]}), flush=True)
        torch.cuda.empty_cache()

    p = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/repr_{a.env}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()

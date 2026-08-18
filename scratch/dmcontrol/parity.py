"""Parity of ``stable_envs.envs.dmcontrol`` against ``dm_control``'s own renderer.

Reports the same two numbers the conformance suite asserts on -- interior-pixel and all-pixel
agreement -- plus a breakdown of WHERE the disagreement is, because on a plate renderer the aggregate
is dominated by pixels that are exact by construction and says nothing about the part that is
actually computed. Pixels are split with MuJoCo's own segmentation buffer into: static (the plate),
mover (the ray-cast), and the mover's silhouette boundary.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def interior_mask(frames):
    import scipy.ndimage as ndi
    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    import mujoco

    import stable_envs as se
    from stable_envs.envs import dmcontrol as DM
    from stable_envs.testing import conformance as C

    names = args.envs.split(",") if args.envs else sorted(DM.DOMAINS)
    out = {"size": args.size, "ss": args.ss, "n": args.n, "envs": {}}
    for name in names:
        env = se.make(name, batch=8, size=args.size, supersample=args.ss, device="cuda")
        cls = type(env)
        ref = cls.reference_env(size=args.size)
        states = env.sample_states(args.n, rng=np.random.default_rng(0))
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, official)

        # Where the disagreement is. The split that matters for a plate renderer is "pixels the plate
        # already gives" versus "pixels the mover pass must produce", and the env's own baked plate
        # draws the line: it is state-independent by construction, so the pixels where the reference
        # differs from it are exactly the second set -- movers, their shadows and their reflections.
        # That is a stronger question than a segmentation mask, which counts only the silhouettes.
        sc = env._scene()
        plate = sc.plate.cpu().numpy()
        mov = np.abs(plate[None].astype(np.int32) - official.astype(np.int32)).max(-1) > 0
        import scipy.ndimage as ndi
        grow = np.stack([ndi.binary_dilation(x, iterations=1) for x in mov])
        edge = grow & ~np.stack([ndi.binary_erosion(x, iterations=1) for x in mov])

        diff = np.abs(ours.astype(np.int32) - official.astype(np.int32)).max(-1)
        bad = diff > 0
        rec = {**r,
               "mover_frac": float(mov.mean()),
               "bad_frac": float(bad.mean()),
               "bad_on_mover": float((bad & grow).sum() / max(1, bad.sum())),
               "bad_on_mover_edge": float((bad & edge).sum() / max(1, bad.sum())),
               "bad_off_mover": float((bad & ~grow).sum() / max(1, bad.sum())),
               "static_exact_frac": float((diff[~grow] == 0).mean()),
               "mover_exact_frac": float((diff[mov] == 0).mean()) if mov.any() else None,
               "mover_mean_abs": float(diff[mov].mean()) if mov.any() else None,
               }
        im = interior_mask(official)
        di = diff[:, :-1, :-1]
        rec["interior_bad_on_mover"] = float(((di > 0) & im & grow[:, :-1, :-1]).sum()
                                             / max(1, ((di > 0) & im).sum()))
        out["envs"][name] = rec
        print(f"{name:22s} interior={r['interior_exact_frac']} all={r['exact_frac']:.5f} "
              f"mean={r['mean_abs']:.4f} max={r['max_abs']:3d} | mover={rec['mover_frac']:.4f} "
              f"bad={rec['bad_frac']:.4f} on_mover={rec['bad_on_mover']:.3f} "
              f"off={rec['bad_off_mover']:.3f} static_exact={rec['static_exact_frac']:.5f}",
              flush=True)
        if args.png:
            import imageio.v2 as iio
            os.makedirs(args.png, exist_ok=True)
            k = int(np.argmax(diff.reshape(len(diff), -1).sum(1)))
            strip = np.concatenate([official[k], ours[k],
                                    np.clip(8 * np.abs(official[k].astype(np.int16)
                                                       - ours[k].astype(np.int16)), 0,
                                            255).astype(np.uint8)], 1)
            iio.imwrite(os.path.join(args.png, f"{name}_strip.png"), strip)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

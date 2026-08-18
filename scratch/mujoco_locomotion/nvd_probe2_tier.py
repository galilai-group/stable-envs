"""What the VISUAL tier has to declare, measured through the env's own surface. One env per process.

Everything here is measured with the SAME function the conformance suite asserts against
(``stable_envs.testing.conformance.check_visual_fidelity``), so the declared numbers and the tested
numbers cannot be two different experiments. Reported per env:

  * ``mean_abs`` / ``within8_frac`` / ``ssim`` over several INDEPENDENT draws from the env's own
    ``sample_states`` — the spread is what the thresholds are set from, because a threshold taken from
    one draw flakes on the next.
  * batch and chunk invariance, which for this backend is a real question rather than a formality:
    nvdiffrast's instanced path takes the whole ``(B,V,4)`` clip-space buffer at once and the env
    chunks it, so a batch-dependent reduction would make ``render(x)[i] != render(x[i])``.
  * ``check_resolution_consistency``, for ``tests/gpu/test_reproducibility.py``'s per-env bound.
  * throughput at the env's own surface (states in, uint8 frames out) against the official
    rasterizer, as a curve rather than a peak.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdloco"


def emit(tag, obj):
    print(f"@@{tag} " + json.dumps(obj, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="MujocoHalfCheetah")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch

    from stable_envs.envs import mujoco_locomotion as M
    from stable_envs.testing import conformance as C

    env_id = M.VISUAL_ENVS[args.cls]
    cls = type(args.cls, (M.GymMujocoVisual,), dict(
        ENV_ID=env_id, name=M._hyphen_name(env_id), reference_id=f"gymnasium/{env_id}",
        __doc__="VISUAL TIER probe build"))
    env = cls(batch=8, size=args.size, device="cuda")
    c = env._nvd()
    emit("env", {"cls": args.cls, "state_dim": env.state_dim, "size": env.size, "ss": env.ss,
                 "chunk": c.chunk, "geoms": len(c.geoms), "verts": int(c.br.scene.V),
                 "tris": int(c.br.scene.face.shape[0]),
                 "lights": {"n": len(c.lights.pos),
                            "headlight": [bool(x) for x in c.lights.headlight],
                            "directional": [bool(x) for x in c.lights.directional],
                            "ambient": c.lights.ambient.tolist(),
                            "diffuse": c.lights.diffuse.tolist(),
                            "specular": c.lights.specular.tolist(),
                            "dir": c.lights.dir.tolist(), "pos": c.lights.pos.tolist()}})

    # ---- fidelity, several independent draws
    draws = []
    for k in range(int(args.draws)):
        r = C.check_visual_fidelity(env, n=int(args.states))
        draws.append(r)
        emit(f"fidelity.draw{k}", r)
    agg = {k: [min(d[k] for d in draws), max(d[k] for d in draws)]
           for k in ("mean_abs", "within8_frac", "ssim", "exact_frac", "interior_exact_frac",
                     "interior_mean_abs")}
    emit("fidelity.spread", agg)

    # ---- invariance
    st = env.sample_states(64)
    full = env.render(st).cpu().numpy()
    worst = 0
    for b in (1, 3, 7, 8, 64):
        got = np.concatenate([env.render(st[i:i + b]).cpu().numpy() for i in range(0, 64, b)])
        worst = max(worst, int(np.abs(got.astype(np.int16) - full.astype(np.int16)).max()))
    perm = np.random.default_rng(0).permutation(64)
    pm = int(np.abs(env.render(st[perm]).cpu().numpy().astype(np.int16)
                    - full[perm].astype(np.int16)).max())
    det = int(np.abs(env.render(st).cpu().numpy().astype(np.int16) - full.astype(np.int16)).max())
    dflt = int(np.abs(env.render(st, style=env.default_style()).cpu().numpy().astype(np.int16)
                      - full.astype(np.int16)).max())
    emit("invariance", {"batch_size_max_abs": worst, "permutation_max_abs": pm,
                        "determinism_max_abs": det, "default_style_max_abs": dflt})

    # ---- per-env variation: two halves of the batch, two pictures
    coll = []
    for path in env.variation_space.names():
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        lo = np.broadcast_to(node.low, node.shape).astype(np.float64)
        hi = np.broadcast_to(node.high, node.shape).astype(np.float64)
        half = np.zeros((8, 1))
        half[4:] = 1.0
        ramp = (half * (hi - lo).reshape(1, -1) + lo.reshape(1, -1)).astype(np.float32)
        f = env.render(st[:8], variation={path: ramp}).cpu().numpy()
        if len(np.unique(f.reshape(8, -1), axis=0)) < 2:
            coll.append(path)
    emit("variation", {"collapsed": coll})

    # ---- resolution consistency (for DOWNSAMPLE_EXACT_FRAC)
    rc = C.check_resolution_consistency(env, st[:8])
    env.size = int(args.size)
    emit("resolution", rc)

    # ---- where a small-batch frame goes, since batch 1 is the number a caller feels first
    if args.profile:
        import stable_envs.render.nvdiff as ND
        for b in (1, 64):
            x = st[:b]
            q = x.astype(np.float64)
            def timeit(fn, reps=30):
                fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    fn()
                torch.cuda.synchronize()
                return 1e3 * (time.perf_counter() - t0) / reps
            xp, xm, cp, cm = env._pose(c, q)
            xp_t = torch.as_tensor(xp, dtype=torch.float32, device="cuda")
            xm_t = torch.as_tensor(xm, dtype=torch.float32, device="cuda")
            cam = ND.Camera(cp, cm[:1], c.fovy, c.znear, c.zfar)
            campos = torch.as_tensor(np.atleast_2d(cam.pos), dtype=torch.float32, device="cuda")
            camfwd = torch.as_tensor(-np.asarray(cam.mat).reshape(-1, 3, 3)[:, :, 2],
                                     dtype=torch.float32, device="cuda")
            view, proj = cam.matrices(1.0)
            vt = torch.as_tensor(view, dtype=torch.float32, device="cuda")
            pt = torch.as_tensor(proj, dtype=torch.float32, device="cuda")
            emit(f"profile.b{b}", {
                "env_render": timeit(lambda: env.render(x)),
                "br_render": timeit(lambda: c.br.render(xp_t, xm_t, cam)),
                "pose": timeit(lambda: env._pose(c, q)),
                "matrices": timeit(lambda: cam.matrices(1.0)),
                "ray_dirs": timeit(lambda: c.br._ray_dirs(cam, env.size, b)),
                "background": timeit(lambda: c.br._background(cam, env.size, b, campos, camfwd)),
                "clip": timeit(lambda: c.br.scene.clip(xp_t, xm_t, vt, pt))})

    # ---- throughput at the env's own surface
    if args.bench:
        pool = np.tile(st, (16, 1))
        rows = {}
        for b in (1, 8, 64, 256, 512, 1024):
            x = pool[:b] if b <= len(pool) else np.tile(pool, (b // len(pool) + 1, 1))[:b]
            env.render(x[:min(b, 8)])
            torch.cuda.synchronize()
            reps = 20 if b <= 8 else (5 if b <= 256 else 3)
            t0 = time.perf_counter()
            for _ in range(reps):
                env.render(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
            rows[b] = {"fps": b / dt, "ms": 1e3 * dt,
                       "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            emit(f"bench.b{b}", rows[b])
            torch.cuda.reset_peak_memory_stats()
        # the official rasterizer, same states, same size
        ref = C.reference_env(cls, env.size)
        t0 = time.perf_counter()
        cls.reference_render(ref, st[:32])
        dt = time.perf_counter() - t0
        rows["official"] = {"fps": 32 / dt}
        emit("bench.official", rows["official"])
        # and the kinematics half of our own frame, so the split is visible
        t0 = time.perf_counter()
        for _ in range(5):
            env._pose(c, st[:256].astype(np.float64))
        rows["kinematics_only"] = {"fps": 256 / ((time.perf_counter() - t0) / 5)}
        emit("bench.kinematics", rows["kinematics_only"])
    else:
        rows = {}

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/tier_{args.cls}.json", "w") as f:
        json.dump({"cls": args.cls, "size": args.size, "draws": draws, "spread": agg,
                   "resolution": rc, "bench": rows}, f, indent=1, default=float)


if __name__ == "__main__":
    main()

"""Parity and Gate A for ``CraftaxFull``, against the UN-JITTED official renderer.

The un-jitted reference is the ruling W1 established and this env inherits: under ``jax.jit`` XLA
fuses craftax's arithmetic differently and it disagrees with its own un-jitted self by up to 1.53e-05
in float32, which straddles a uint8 rounding boundary wherever the picture is compressed towards one
value -- and the ``light_map`` multiply, the night blend and the sleep halving all do that.

    PYTHONPATH=. /Users/samib/se7-craftax-venv/bin/python scripts/regression/craftax/certify.py \
        --device cpu --tag mac
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "examples", "results", "step7", "craftax-full", "certify")


def band(row, env):
    from stable_envs.envs import craftax_full as CF

    j = 5 * CF.N_TILE
    lvl = int(round(float(row[j + 2]))) % CF.N_LEVEL
    light = float(CF._tent_np(np.asarray(row[j + 3])))
    day = light if lvl == 0 else 1.0
    lit = bool(np.all(CF._tent_np(row[4 * CF.N_TILE : 5 * CF.N_TILE]) == 1.0))
    if day == 1.0 and lit:
        return "fast_uint8"
    if day >= CF.NIGHT_THRESHOLD:
        return "lit_no_prng"
    return "night_prng"


def parity(cls, states, size, device):
    from stable_envs.testing.conformance import frames_digest

    env = cls(batch=len(states), device=device, size=size)
    ours = env.render(states, size=size).cpu().numpy()
    ref = cls.reference_env(size=size)
    t0 = time.perf_counter()
    off = cls.reference_render(ref, states)
    dt = time.perf_counter() - t0
    d = np.abs(ours.astype(np.int16) - off.astype(np.int16))
    wh = CF_OBS_H * size
    rec = {
        "size": size,
        "n": int(len(states)),
        "shape": list(ours.shape),
        "exact_frac": float((d.max(-1) == 0).mean()),
        "mean_abs": float(d.mean()),
        "max_abs": int(d.max()),
        "world_exact_frac": float((d[:, :wh].max(-1) == 0).mean()),
        "strip_exact_frac": float((d[:, wh:].max(-1) == 0).mean()),
        "digest_ours": frames_digest(ours),
        "digest_official": frames_digest(off),
        "reference_seconds_per_frame": dt / max(len(states), 1),
    }
    bands = {}
    for i, row in enumerate(states):
        b = band(row, env)
        bands.setdefault(b, []).append(i)
    rec["per_band"] = {
        b: {"n": len(ix), "exact_frac": float((d[ix].max(-1) == 0).mean()), "max_abs": int(d[ix].max())}
        for b, ix in sorted(bands.items())
    }
    return rec


def gate_a(cls, device):
    from stable_envs.testing import conformance as C

    env = cls(batch=16, device=device)
    st = env.sample_states(env.B)
    out = {}
    out["check_state_roundtrip"] = C.check_state_roundtrip(env)
    out["check_state_dims_matter"] = C.check_state_dims_matter(env)
    out["check_batch_invariance"] = C.check_batch_invariance(env)
    out["check_determinism"] = C.check_determinism(env)
    out["check_render_is_pure"] = C.check_render_is_pure(env, st)
    out["check_order_independence"] = C.check_order_independence(env, st)
    out["check_batch_size_invariance"] = C.check_batch_size_invariance(env, st)
    out["check_chunked_render_matches"] = C.check_chunked_render_matches(env, st)
    out["check_input_form_invariance"] = C.check_input_form_invariance(env, st)
    out["check_set_state_copies"] = C.check_set_state_copies(env)
    out["check_style_is_state"] = C.check_style_is_state(env)
    out["check_default_style_is_official"] = C.check_default_style_is_official(env)
    out["check_variations_do_something"] = C.check_variations_do_something(env)
    out["check_per_env_variation"] = C.check_per_env_variation(env)
    out["check_task_object_visibility"] = C.check_task_object_visibility(env, st)
    # check_reference_renderer_is_stable builds the env with device="cuda" hard-coded, so it is a
    # CUDA-host check by construction. Recorded as not-applicable rather than as an error on a CPU
    # host; the shared check is not this workstream's to change.
    if torch.cuda.is_available():
        out["check_reference_renderer_is_stable"] = C.check_reference_renderer_is_stable(cls, 10)
    else:
        out["check_reference_renderer_is_stable"] = {
            "skipped": "conformance.check_reference_renderer_is_stable hard-codes device='cuda'"
        }
    out["check_colors_apply_exactly"] = C.check_colors_apply_exactly(env)
    out["check_resolution_consistency"] = C.check_resolution_consistency(env, C.downsample_protocol_states(env))
    out["variations_by_batch"] = {
        str(b): C.check_variations_do_something(cls(batch=b, device=device)) for b in (1, 2, 4, 16)
    }
    out["dims_by_batch"] = {}
    for b in (1, 2, 3, 4, 8, 16):
        bad = []
        for seed in range(12):
            e = cls(batch=b, device=device)
            rng = np.random.default_rng(seed)
            e.sample_states = lambda n, _r=rng, _e=e: cls.sample_states(_e, n, _r)
            r = C.check_state_dims_matter(e)
            if r["dead_dims"]:
                bad.append({"seed": seed, "dead": r["dead_dims"]})
        out["dims_by_batch"][str(b)] = bad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--skip-gate-a", action="store_true")
    a = ap.parse_args()
    from stable_envs.envs.craftax_full import OBS_H, CraftaxFull

    global CF_OBS_H
    CF_OBS_H = OBS_H
    torch.set_num_threads(1)

    if a.corpus:
        states = np.load(a.corpus)["states"].astype(np.float32)
    else:
        env = CraftaxFull(batch=a.n, device=a.device)
        states = env.sample_states(a.n, np.random.default_rng(a.seed))
    rep = {"tag": a.tag, "device": a.device, "n": int(len(states)), "seed": a.seed, "torch": torch.__version__}
    try:
        from importlib import metadata

        import jax

        rep["reference"] = {
            "craftax": metadata.version("craftax"),
            "jax": jax.__version__,
            "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        }
    except Exception as e:  # noqa: BLE001
        rep["reference"] = {"error": f"{type(e).__name__}: {e}"}
    rep["parity"] = {str(s): parity(CraftaxFull, states, s, a.device) for s in (10, 16)}
    if not a.skip_gate_a:
        rep["gate_a"] = gate_a(CraftaxFull, a.device)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"certify_{a.device}_{a.tag}.json")
    with open(p, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str)[:7000])
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())

"""Step-7 W1 certification harness: parity, Gate A, and the resolution-consistency mechanism.

Runs wherever a jax-bearing interpreter exists (the local reference venv, or ``$SCR/venvs/se7-craftax``
on Oscar once SSH returns). Writes one JSON per device under
``examples/results/step7/craftax-classic-full/certify/``.

Three things it does that a plain conformance run does not:

* it reports parity **per lighting band**, because a single all-pixel number over a mixed pool hides
  which band fails;
* it reports ``check_resolution_consistency`` per band as well, because the frozen
  ``DOWNSAMPLE_EXACT_FRAC`` is met above ``light_level = 0.5`` and cannot be met below it -- the noise
  field is drawn at the frame's own resolution, so 2S is a different field and not a supersample of S.
  That is a property of the REFERENCE and no port can design it away without changing pixels;
* it replays the world rows of a Step-4-shaped state, which is the frozen corpus's replay obligation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from stable_envs.envs.craftax import CraftaxClassic, N_STRIP, OBS_H, OBS_W, STRIP_H
from stable_envs.testing import conformance as cf

OUT = "examples/results/step7/craftax-classic-full/certify"
LIGHT = OBS_H * OBS_W * 2 + 2 + N_STRIP


def bands(st):
    light = st[:, LIGHT]
    return {"night": light < 0.5, "dusk": (light >= 0.5) & (light < 1.0), "day": light == 1.0}


def parity(env, st, size):
    ref = CraftaxClassic.reference_env(size=size)
    off = CraftaxClassic.reference_render(ref, st)
    ours = env.render(st, size=size).cpu().numpy()
    assert off.shape == ours.shape, (off.shape, ours.shape)
    d = np.abs(off.astype(np.int16) - ours.astype(np.int16))
    row = {"shape": list(ours.shape), "exact_frac": float((d.max(-1) == 0).mean()),
           "mean_abs": float(d.mean()), "max_abs": int(d.max()),
           "digest_ours": hashlib.sha256(ours.tobytes()).hexdigest()[:16],
           "digest_official": hashlib.sha256(off.tobytes()).hexdigest()[:16]}
    w = OBS_H * size
    row["world_rows_exact_frac"] = float((d[:, :w].max(-1) == 0).mean())
    row["strip_rows_exact_frac"] = float((d[:, w:].max(-1) == 0).mean())
    for name, m in bands(st).items():
        if not m.any():
            continue
        dd = d[m]
        row[f"{name}_n"] = int(m.sum())
        row[f"{name}_exact_frac"] = float((dd.max(-1) == 0).mean())
        row[f"{name}_max_abs"] = int(dd.max())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    env = CraftaxClassic(device=a.device)
    st = env.sample_states(a.states, np.random.default_rng(a.seed))
    res = {"device": a.device, "torch": torch.__version__, "n_states": a.states, "seed": a.seed,
           "host": os.uname().nodename, "machine": os.uname().machine,
           "declared": {"parity_all_exact_min": CraftaxClassic.parity_all_exact_min,
                        "parity_interior_min": CraftaxClassic.parity_interior_min,
                        "reference_versions": dict(CraftaxClassic.reference_versions)}}
    try:
        import jax
        res["jax"] = jax.__version__
        res["jax_threefry_partitionable"] = bool(jax.config.jax_threefry_partitionable)
    except Exception as e:                                                  # noqa: BLE001
        res["jax"] = f"absent: {type(e).__name__}"

    res["band_counts"] = {k: int(v.sum()) for k, v in bands(st).items()}
    res["n_sleeping"] = int(st[:, OBS_H * OBS_W * 2 + 1 + N_STRIP].sum())

    if "absent" not in str(res["jax"]):
        res["parity"] = {str(s): parity(env, st, s) for s in (7, 16)}

    # Gate A, the mandatory checks that need no reference
    res["gate_a"] = {}
    for name in ("check_state_roundtrip", "check_state_dims_matter", "check_batch_invariance",
                 "check_batch_size_invariance", "check_order_independence", "check_determinism",
                 "check_render_is_pure", "check_chunked_render_matches",
                 "check_input_form_invariance", "check_set_state_copies", "check_style_is_state",
                 "check_default_style_is_official", "check_per_env_variation",
                 "check_variations_do_something", "check_colors_apply_exactly"):
        fn = getattr(cf, name)
        try:
            import inspect
            p = inspect.signature(fn).parameters
            res["gate_a"][name] = fn(env, st) if "states" in p else fn(env)
        except Exception as e:                                              # noqa: BLE001
            res["gate_a"][name] = {"ERROR": f"{type(e).__name__}: {e}"}

    # resolution consistency, per band, because the verdict differs by band and only by band
    res["resolution_consistency"] = {"mixed_pool": cf.check_resolution_consistency(env, st)}
    for name, v in (("all_daylight", 1.0), ("dusk_0.75", 0.75), ("dusk_0.50", 0.5),
                    ("night_0.45", 0.45), ("night_0.10", 0.1)):
        s2 = np.array(st, copy=True)
        s2[:, LIGHT] = v
        res["resolution_consistency"][name] = cf.check_resolution_consistency(env, s2)

    # the Step-4 replay obligation: a 127-slot row zero-extended, daylight, awake, empty-handed
    old = np.zeros((a.states, env.state_dim), np.float32)
    old[:, :127] = st[:, :127]
    old[:, LIGHT] = 1.0
    rep = {}
    for size in (7, 16):
        f = env.render(old, size=size).cpu().numpy()
        w = OBS_H * size
        rep[str(size)] = {
            "world_rows_digest": hashlib.sha256(f[:, :w].tobytes()).hexdigest(),
            "strip_is_black": bool(f[:, w:].max() == 0),
            "frames_shape": list(f.shape),
            "world_mean_pixel": float(f[:, :w].mean()),
            "world_distinct_colors": int(len(np.unique(f[:, :w].reshape(-1, 3), axis=0)))}
    res["step4_world_row_replay"] = rep

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"certify_{a.device}{('_' + a.tag) if a.tag else ''}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "gate_a"}, indent=1, default=str))
    print("gate_a:", json.dumps(res["gate_a"], default=str)[:2000])
    print("wrote", p)


if __name__ == "__main__":
    main()

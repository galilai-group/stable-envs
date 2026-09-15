"""Is the OFFICIAL Craftax renderer resolution-consistent under night? An oracle experiment.

``check_resolution_consistency`` renders at S and at 2S, box-downsamples the 2S frame, and reports
what fraction of pixels agree. Under night this renderer scores 0.28 where it scores 1.000 by day,
and the claim made about that was a claim about the *reference*: its noise field is drawn at the
frame's own resolution, so 2S is a different field rather than a supersample of S. **That claim had
been argued from our own implementation, which is not evidence about the reference.** This measures
it on the reference itself.

Protocol, deliberately the check's own and not a new one:

* the same states, per lighting band, from the frozen Step-7 corpus, plus the check's own protocol
  draw (``render_worker.sample_inputs(env, 8)`` at ``STATE_SEED``, default appearance) so the number
  the frozen bound is judged on is covered too;
* the same reference as parity -- craftax's renderer, **un-jitted**, with each state's own
  ``state_rng`` written back, so the two resolutions see the SAME PRNG state;
* the same rounding ``reference_render`` uses, then the check's exact ``_box_downsample``;
* the same statistic the check returns.

The 32 px official textures come from craftax's own ``load_all_textures_given_size(32)`` -- the same
function that builds the 16 px set it ships. craftax caches only 7, 16 and 64, so the renderer is
built through a patched lookup rather than a reimplementation: every pixel is still craftax's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from stable_envs.envs.craftax import CraftaxClassic, N_STRIP, OBS_H, OBS_W
from stable_envs.testing import conformance as cf

OUT = "examples/results/step7/craftax-classic-full/oracle"
LIGHT = OBS_H * OBS_W * 2 + 2 + N_STRIP


def official_renderer(size: int):
    """craftax's own renderer at ``size`` px, including sizes it does not pre-cache."""
    import craftax.craftax_classic.constants as C
    import craftax.craftax_classic.renderer as R
    cached = C.load_all_textures()
    if size in cached:
        tex = cached
    else:
        tex = dict(cached)
        tex[size] = C.load_all_textures_given_size(size)
    orig = R.load_all_textures
    R.load_all_textures = lambda: tex
    try:
        return R.make_craftax_pixel_renderer(size)          # un-jitted, as reference_env returns
    finally:
        R.load_all_textures = orig


def official_frames(states, size):
    import jax
    _, tmpl, _ = CraftaxClassic.reference_env(size=16)       # template + the flag assertion
    return CraftaxClassic.reference_render((official_renderer(size), tmpl, size), states)


def _st(d):
    return {"exact_frac": float((d.max(-1) == 0).mean()), "mean_abs": float(d.mean()),
            "max_abs": int(d.max())}


def stat(a, b, S):
    """The check's statistic, and the same split into world rows and strip rows.

    The split is the whole point: the two regions can fail for different reasons, and a single
    number over the frame cannot tell a noise field drawn at the frame's own resolution from a strip
    whose icon size is ``int(0.8*t)`` and therefore does not scale by two.
    """
    d = np.abs(cf._box_downsample(b, 2).astype(np.int16) - a.astype(np.int16))
    w = OBS_H * S
    return {"frame": _st(d), "world_rows": _st(d[:, :w]), "strip_rows": _st(d[:, w:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    a = ap.parse_args()
    S, F = a.size, 2
    env = CraftaxClassic(device="cpu")

    corpus = np.load("examples/results/step7/craftax-classic-full/corpus/states.npz")["states"]
    L = corpus[:, LIGHT]
    pools = {"corpus_night": corpus[L < 0.5],
             "corpus_dusk": corpus[(L >= 0.5) & (L < 1.0)],
             "corpus_daylight": corpus[L == 1.0],
             "downsample_protocol_draw": cf.downsample_protocol_states(env)}
    pl = pools["downsample_protocol_draw"][:, LIGHT]
    pools["protocol_draw_night_only"] = pools["downsample_protocol_draw"][pl < 0.5]

    import jax
    res = {"what": ("check_resolution_consistency's exact protocol applied to the OFFICIAL craftax "
                    "renderer, to decide whether night resolution-inconsistency is a property of the "
                    "reference or of this port"),
           "size_S": S, "factor": F, "reference": "craftax 1.6.1, un-jitted, per-state state_rng",
           "jax": jax.__version__,
           "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
           "official_textures_at_2S": ("craftax.craftax_classic.constants."
                                       f"load_all_textures_given_size({S * F})"),
           "downsample": "stable_envs.testing.conformance._box_downsample (the check's own)",
           "corpus": "examples/results/step7/craftax-classic-full/corpus/states.npz",
           "protocol": cf.downsample_protocol(pools["downsample_protocol_draw"]),
           "bands": {}}

    for name, st in pools.items():
        if len(st) == 0:
            continue
        off_s, off_2s = official_frames(st, S), official_frames(st, S * F)
        our_s = env.render(st, size=S).numpy()
        our_2s = env.render(st, size=S * F).numpy()
        row = {"n_states": int(len(st)),
               "light_min": float(st[:, LIGHT].min()), "light_max": float(st[:, LIGHT].max()),
               "official": stat(off_s, off_2s, S), "ours": stat(our_s, our_2s, S),
               "parity_at_S": _st(np.abs(off_s.astype(np.int16) - our_s.astype(np.int16))),
               "states_sha256": hashlib.sha256(
                   np.ascontiguousarray(st, np.float32).tobytes()).hexdigest()[:16]}
        row["official_is_resolution_consistent"] = {
            k: row["official"][k]["exact_frac"] >= 1.0 - 0.005
            for k in ("frame", "world_rows", "strip_rows")}
        res["bands"][name] = row
        print(f"{name:26s} official world {row['official']['world_rows']['exact_frac']:.4f} "
              f"strip {row['official']['strip_rows']['exact_frac']:.4f} | "
              f"ours world {row['ours']['world_rows']['exact_frac']:.4f} "
              f"strip {row['ours']['strip_rows']['exact_frac']:.4f} | "
              f"parity@S exact {row['parity_at_S']['exact_frac']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"resolution_oracle_S{S}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()

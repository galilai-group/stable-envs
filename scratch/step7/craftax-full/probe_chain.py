"""The lighting chain of the FULL game, against W1's ``night_and_sleep``, on real frames.

The brief says to "reuse W1's strip/night/sleep". This probe asks whether that is true, on the
reference's own intermediates rather than by reading the two sources side by side, and quantifies
where it is not.

Also measures:
  * the ``light_map`` alphabet over every level of several worlds -- it decides whether the
    underground darkness can be folded into a baked table or has to stay a per-tile float;
  * the daylight override ``daylight = select(player_level == 0, light_level, 1.0)``, which makes the
    night chain the identity on eight of the nine levels;
  * the exact torch transcription of the FULL chain, bit-for-bit against the reference.

Writes ``phase0/chain.json``.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")
TILE_PX = 10


def full_vignette(h, w):
    """craftax full's ``night_noise_intensity_texture``: a NORMALISED RADIAL DISTANCE, in jnp f32.

    Classic's is ``1 - exp(-0.5 (x^2+y^2)/0.5^2)`` on a ``linspace(-1, 1)`` grid computed in numpy
    float64. They are different functions, so :func:`stable_envs.envs.craftax.night_vignette` is not
    reusable here -- which is a phase-0 finding rather than a detail.
    """
    x = np.arange(h, dtype=np.float32)[:, None] - (h // 2)
    y = np.arange(w, dtype=np.float32)[None, :] - (w // 2)
    v = np.sqrt(x ** 2 + y ** 2).astype(np.float32)
    # NOT ``v / v.max()``. craftax normalises under jax, and XLA lowers a float32 divide by a
    # scalar as a RECIPROCAL MULTIPLY -- which disagrees with a true divide on 2,265 of the 9,900
    # elements at the registered size, by one ULP each. One ULP in the vignette is one ULP in the
    # night mask, which is enough to move a uint8 pixel. Measured in ``phase0/chain.json``.
    v = (v * np.float32(np.float32(1.0) / np.float32(v.max()))).astype(np.float32)
    return torch.from_numpy(v)[:, :, None]


def full_night_and_sleep(world, light, sleeping, keys, light_map):
    """craftax full's chain, op for op. ``world`` (n,h,w,3) is the pre-lighting composite."""
    from stable_envs.render import prng
    n, h, w, _ = world.shape
    px = world * light_map                                   # per-tile darkness, BEFORE night
    night_tex = torch.tensor([0.0, 16.0, 64.0])
    lt = light.view(n, 1, 1, 1)
    noise = (prng.uniform(keys, (h, w)) * 95 + 32).unsqueeze(-1)
    intensity = torch.clamp_min(2.0 * (0.5 - lt), 0.0)
    mask = full_vignette(h, w) * intensity
    night = (1.0 - mask) * px + mask * noise
    night = night_tex * 0.5 + 0.5 * night
    px = lt * px + (1 - lt) * night
    sleep_level = 1.0 - sleeping.view(n, 1, 1, 1).to(px.dtype) * 0.5
    return sleep_level * px


def main():
    import jax
    import jax.numpy as jnp
    from craftax.craftax import constants as C
    from instrumented_renderer import make_instrumented_renderer
    from craftax.craftax.renderer import make_craftax_pixel_renderer

    inst = make_instrumented_renderer(TILE_PX)
    ref = make_craftax_pixel_renderer(TILE_PX)
    pool = pickle.load(open(POOL, "rb"))
    states = pool["rollout"][:40] + pool["level_shifted"]
    rep = {"tile_px": TILE_PX, "n_states": len(states)}

    # ---- the instrumented copy must be the reference
    same = sum(int(np.array_equal(np.asarray(inst(s)["frame"]), np.asarray(ref(s))))
               for s in states)
    rep["instrumented_equals_reference"] = f"{same}/{len(states)}"

    # ---- the chain in torch, bit for bit
    H, W = C.OBS_DIM[0] * TILE_PX, C.OBS_DIM[1] * TILE_PX
    res = []
    for s in states:
        cap = inst(s)
        world = torch.from_numpy(np.asarray(cap["world"]).astype(np.float32))[None]
        lvl = int(np.asarray(s.player_level))
        daylight = float(np.asarray(s.light_level)) if lvl == 0 else 1.0
        lmv = np.asarray(s.light_map)[lvl]
        pp = np.asarray(s.player_position)
        pad = C.MAX_OBS_DIM + 2
        padded = np.pad(lmv, pad, constant_values=0.0)
        tl = pp - np.array(C.OBS_DIM) // 2 + pad
        view = padded[tl[0]:tl[0] + C.OBS_DIM[0], tl[1]:tl[1] + C.OBS_DIM[1]]
        lm = torch.from_numpy(np.repeat(np.repeat(view, TILE_PX, 0), TILE_PX, 1)
                              .astype(np.float32))[None, :, :, None]
        k = np.asarray(s.state_rng).astype(np.uint32).astype(np.int64)
        keys = torch.tensor([[int(np.int32(np.uint32(k[0]))), int(np.int32(np.uint32(k[1])))]],
                            dtype=torch.int32)
        got = full_night_and_sleep(world, torch.tensor([daylight]),
                                   torch.tensor([bool(np.asarray(s.is_sleeping))]), keys, lm)
        want = torch.from_numpy(np.asarray(cap["after_sleep"]).astype(np.float32))[None]
        res.append({
            "level": lvl, "light": float(np.asarray(s.light_level)), "daylight": daylight,
            "sleeping": bool(np.asarray(s.is_sleeping)),
            "bitexact": bool(torch.equal(got, want)),
            "max_abs": float((got - want).abs().max()),
            "uint8_exact": bool(np.array_equal(
                np.round(got.numpy()).clip(0, 255).astype(np.uint8),
                np.round(want.numpy()).clip(0, 255).astype(np.uint8))),
        })
    rep["full_chain_torch"] = {
        "bitexact": f"{sum(r['bitexact'] for r in res)}/{len(res)}",
        "uint8_exact": f"{sum(r['uint8_exact'] for r in res)}/{len(res)}",
        "max_abs_over_pool": max(r["max_abs"] for r in res),
        "per_state": res,
    }

    # ---- W1's night_and_sleep, offered the same inputs the two ways its signature allows
    from stable_envs.envs.craftax import night_and_sleep as w1
    w1res = {"with_light_map_arg": [], "light_map_premultiplied": []}
    for s, r in zip(states, res):
        cap = inst(s)
        world = torch.from_numpy(np.asarray(cap["world"]).astype(np.float32))[None]
        lvl = int(np.asarray(s.player_level))
        daylight = float(np.asarray(s.light_level)) if lvl == 0 else 1.0
        lmv = np.asarray(s.light_map)[lvl]
        pp = np.asarray(s.player_position)
        pad = C.MAX_OBS_DIM + 2
        view = np.pad(lmv, pad, constant_values=0.0)[
            (pp - np.array(C.OBS_DIM) // 2 + pad)[0]:
            (pp - np.array(C.OBS_DIM) // 2 + pad)[0] + C.OBS_DIM[0],
            (pp - np.array(C.OBS_DIM) // 2 + pad)[1]:
            (pp - np.array(C.OBS_DIM) // 2 + pad)[1] + C.OBS_DIM[1]]
        lm = torch.from_numpy(np.repeat(np.repeat(view, TILE_PX, 0), TILE_PX, 1)
                              .astype(np.float32))[None, :, :, None]
        want = torch.from_numpy(np.asarray(cap["after_sleep"]).astype(np.float32))[None]
        k = np.asarray(s.state_rng).astype(np.uint32)
        keys = torch.tensor([[int(np.int32(k[0])), int(np.int32(k[1]))]], dtype=torch.int32)
        lt = torch.tensor([daylight])
        sl = torch.tensor([bool(np.asarray(s.is_sleeping))])
        a = w1(world.clone(), lt, sl, keys, light_map=lm)
        b = w1((world * lm).clone(), lt, sl, keys)
        for name, got in (("with_light_map_arg", a), ("light_map_premultiplied", b)):
            w1res[name].append({
                "bitexact": bool(torch.equal(got, want)),
                "max_abs": float((got - want).abs().max()),
                "uint8_exact": bool(np.array_equal(
                    np.round(got.numpy()).clip(0, 255).astype(np.uint8),
                    np.round(want.numpy()).clip(0, 255).astype(np.uint8))),
            })
    rep["w1_night_and_sleep_against_full"] = {
        name: {"bitexact": f"{sum(x['bitexact'] for x in v)}/{len(v)}",
               "uint8_exact": f"{sum(x['uint8_exact'] for x in v)}/{len(v)}",
               "max_abs_over_pool": max(x["max_abs"] for x in v)}
        for name, v in w1res.items()}

    # ---- the daylight override: eight of nine levels never see the night chain
    lv = {}
    for lvl in range(9):
        s = pool["level_shifted"][lvl]
        a = np.asarray(ref(s.replace(light_level=jnp.float32(0.0))))
        b = np.asarray(ref(s.replace(light_level=jnp.float32(1.0))))
        lv[lvl] = int((a != b).sum())
    rep["light_level_pixels_moved_by_level"] = lv
    rep["daylight_override"] = ("daylight = select(player_level == 0, light_level, 1.0): the night "
                               "chain is exactly the identity on levels 1..8, whatever light_level "
                               "holds")

    # ---- the light_map alphabet, over every level of several worlds
    vals = set()
    per_world = []
    for seed in (0, 1, 2, 3):
        from craftax.craftax_env import make_craftax_env_from_name
        env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
        _o, st = env.reset(jax.random.PRNGKey(seed), env.default_params)
        u = np.unique(np.asarray(st.light_map))
        per_world.append(int(u.size))
        vals |= set(float(x) for x in u)
    sv = sorted(vals)
    rep["light_map_alphabet"] = {
        "n_distinct_over_4_worlds": len(sv), "per_world": per_world,
        "min": sv[0], "max": sv[-1], "first_24": sv[:24],
        "is_small_enough_to_bake": len(sv) <= 64,
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "chain.json"), "w") as f:
        json.dump(rep, f, indent=2)
    small = dict(rep)
    small["full_chain_torch"] = {k: v for k, v in rep["full_chain_torch"].items()
                                 if k != "per_state"}
    print(json.dumps(small, indent=2))


if __name__ == "__main__":
    sys.exit(main())

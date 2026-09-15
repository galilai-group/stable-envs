"""Parity on a pool built from the EDGES, not from the sampler.

The frozen corpus comes from ``sample_states``, so it exercises what the sampler draws. This builds
the states the sampler will not: ``light_level`` exactly at and either side of the 0.5 threshold, key
halves at 0 and 65535, every count at both ends of its range, every level, the fully-opaque mob slot
on a tile, every projectile orientation, an empty window and a maximally crowded one. It is the
independent-pool check W1's reviewer ran by hand, written down so it can be re-run.

    JAX_PLATFORMS=cpu PYTHONPATH=. /Users/samib/se7-craftax-venv/bin/python \
        scratch/step7/craftax-full/probe_boundary.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "examples", "results", "step7", "craftax-full", "certify")

from stable_envs.envs import craftax_full as CF                       # noqa: E402
from stable_envs.envs.craftax_full import CraftaxFull                 # noqa: E402

J = 5 * CF.N_TILE
S0 = J + 4 + CF.N_KEY_HALVES


def rows():
    """Each entry is ``(name, mutate)``; ``mutate`` edits a zeroed state row in place."""
    k = CF.N_TILE
    centre = (CF.OBS_H // 2) * CF.OBS_W + (CF.OBS_W // 2)
    half = float(np.float32(0.5))
    just_under = float(np.nextafter(np.float32(0.5), np.float32(0.0)))
    just_over = float(np.nextafter(np.float32(0.5), np.float32(1.0)))
    just_under_one = float(np.nextafter(np.float32(1.0), np.float32(0.0)))

    def base(st):
        st[0:k] = 2                                   # GRASS everywhere
        st[4 * k:5 * k] = 1.0                         # fully lit
        st[J + 3] = 1.0                               # full daylight

    out = []

    def add(name, fn):
        def m(st):
            base(st)
            fn(st)
        out.append((name, m))

    for lv, tag in ((0.0, "light_0"), (just_under, "light_just_under_half"),
                    (half, "light_exactly_half"), (just_over, "light_just_over_half"),
                    (just_under_one, "light_just_under_one"), (1.0, "light_one")):
        add(tag, lambda st, v=lv: st.__setitem__(J + 3, v))
    for h, tag in (((0, 0, 0, 0), "key_all_zero"), ((65535,) * 4, "key_all_max"),
                   ((65535, 0, 0, 65535), "key_mixed"), ((32768, 32768, 32768, 32768), "key_signbit")):
        def m(st, hh=h):
            st[J + 3] = 0.1
            st[J + 4:J + 8] = hh
        add(f"night_{tag}", m)
    for lvl in range(CF.N_LEVEL):
        add(f"level_{lvl}", lambda st, v=lvl: st.__setitem__(J + 2, v))
        add(f"level_{lvl}_dark", lambda st, v=lvl: (st.__setitem__(J + 2, v),
                                                    st.__setitem__(slice(4 * k, 5 * k), 0.2)))
    add("sleeping", lambda st: st.__setitem__(J + 1, 1.0))
    add("sleeping_at_night", lambda st: (st.__setitem__(J + 1, 1.0), st.__setitem__(J + 3, 0.05)))
    for f in range(CF.N_FACING):
        add(f"facing_{f}", lambda st, v=f: st.__setitem__(J, v))
    add("all_counts_zero", lambda st: st.__setitem__(slice(S0, None), 0))
    add("all_counts_max", lambda st: st.__setitem__(
        slice(S0, None), np.asarray(CF.STRIP_MOD, np.float32) - 1))
    add("counts_at_nine", lambda st: st.__setitem__(
        slice(S0, None), np.minimum(np.asarray(CF.STRIP_MOD, np.float32) - 1, 9)))
    add("counts_at_ten", lambda st: st.__setitem__(
        slice(S0, None), np.minimum(np.asarray(CF.STRIP_MOD, np.float32) - 1, 10)))
    add("empty_window", lambda st: None)
    add("out_of_bounds_window", lambda st: st.__setitem__(slice(0, k), CF.B_OUT_OF_BOUNDS))
    add("darkness_window", lambda st: st.__setitem__(slice(0, k), CF.B_DARKNESS))
    add("every_block_once", lambda st: st.__setitem__(
        slice(0, k), np.arange(k, dtype=np.float32) % CF.N_BLOCK))
    add("necromancer_and_vulnerable", lambda st: (
        st.__setitem__(0, CF.B_NECROMANCER), st.__setitem__(1, CF.B_NECROMANCER_VULNERABLE)))
    add("every_item_once", lambda st: st.__setitem__(
        slice(k, 2 * k), np.arange(k, dtype=np.float32) % CF.N_ITEM))
    add("ladders_open_and_blocked", lambda st: (
        st.__setitem__(k + 0, CF.I_LADDER_DOWN), st.__setitem__(k + 1, CF.I_LADDER_DOWN_BLOCKED)))

    def crowded(st):
        # the simulator's full budget, including the one fully opaque mob sprite
        slots = ([CF.MOB_BASE["melee"] + i for i in (0, 5, 7)]
                 + [CF.MOB_BASE["passive"] + i for i in (0, 1, 2)]
                 + [CF.MOB_BASE["ranged"] + i for i in (5, 3)])
        pos = [p for p in range(k) if p != centre][:len(slots)]
        for p, v in zip(pos, slots):
            st[2 * k + p] = v
        free = [p for p in range(k) if p != centre and st[2 * k + p] == 0]
        for i, p in enumerate(free[:6]):
            st[3 * k + p] = 1 + (i % CF.N_PROJ_TYPE) * 4 + (i % 4)
    add("full_mob_and_projectile_budget", crowded)

    def orientations(st):
        # the simulator carries at most MAX_MOB_PROJ + MAX_PLAYER_PROJ projectiles, so all four
        # orientations of one type plus two of another is the most a REPRESENTABLE row can show
        for o in range(4):
            st[3 * k + o] = 1 + 0 * 4 + o
        for o in range(2):
            st[3 * k + 4 + o] = 1 + 5 * 4 + o
    add("every_projectile_orientation", orientations)

    def over_budget(st):
        """DELIBERATELY unrepresentable: eight projectiles against a budget of six.

        Kept in the pool and expected to differ, because the alternative is a pool that quietly
        avoids the boundary. ``reference_render`` can place only the first six, so the last two tiles
        disagree -- and that is the documented limit of the render window, not a renderer defect.
        """
        for i in range(8):
            st[3 * k + i] = 1 + (i % CF.N_PROJ_TYPE) * 4 + (i % 4)
    add("OVER_BUDGET_eight_projectiles", over_budget)
    add("opaque_mob_on_a_ladder", lambda st: (
        st.__setitem__(2 * k + 0, CF.MOB_BASE["ranged"] + 5),
        st.__setitem__(k + 0, CF.I_LADDER_DOWN)))
    return out


def main():
    from stable_envs.testing.conformance import frames_digest
    specs = rows()
    st = np.zeros((len(specs), CraftaxFull.state_dim), np.float32)
    for i, (_name, fn) in enumerate(specs):
        fn(st[i])
    env = CraftaxFull(batch=len(specs), device="cpu")
    rep = {"n": len(specs), "names": [n for n, _ in specs], "sizes": {}}
    try:
        from importlib import metadata
        import jax
        rep["reference"] = {"craftax": metadata.version("craftax"), "jax": jax.__version__,
                            "partitionable": bool(jax.config.jax_threefry_partitionable)}
    except Exception as e:                                             # noqa: BLE001
        rep["reference"] = {"error": f"{type(e).__name__}: {e}"}
    for size in (10, 16):
        ours = env.render(st, size=size).cpu().numpy()
        ref = CraftaxFull.reference_env(size=size)
        off = CraftaxFull.reference_render(ref, st)
        d = np.abs(ours.astype(np.int16) - off.astype(np.int16))
        worst = [{"name": specs[i][0], "exact_frac": float((d[i].max(-1) == 0).mean()),
                  "max_abs": int(d[i].max())}
                 for i in range(len(specs)) if d[i].max() > 0]
        # the deliberately-unrepresentable row is scored apart: a pool that avoided the boundary
        # would report a better number and mean less
        keep = [i for i, (nm, _f) in enumerate(specs) if not nm.startswith("OVER_BUDGET")]
        dk = d[keep]
        rep["sizes"][str(size)] = {
            "representable_rows": len(keep),
            "exact_frac": float((dk.max(-1) == 0).mean()), "max_abs": int(dk.max()),
            "digest_ours": frames_digest(ours[keep]),
            "digest_official": frames_digest(off[keep]),
            "rows_that_differ": worst,
            "expected_to_differ": [nm for nm, _f in specs if nm.startswith("OVER_BUDGET")]}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "boundary_pool.json")
    with open(p, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "names"}, indent=2)[:3000])
    print("n rows", len(specs), "->", p)


if __name__ == "__main__":
    sys.exit(main())

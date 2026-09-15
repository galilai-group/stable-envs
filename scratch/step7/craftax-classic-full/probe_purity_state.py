"""Phase 0 (2b)(4) — is ``state_rng`` observable, and what must the render window grow by?

(2b) A pure ``f(state)`` renderer needs every input the reference reads. ``state_rng`` is a field of
``EnvState`` (``envs/craftax_state.py``), so the reference IS pure — but only if the key is part of
the state we store. This measures whether two states that agree on map/window/player/mobs and differ
only in ``state_rng`` render differently, at night and in daylight.

(4) The byte accounting for the render window, today and with each component added.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.constants import INVENTORY_OBS_HEIGHT, OBS_DIM
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

OUT = "examples/results/step7/craftax-classic-full/phase0"
STATES = "scratch/step7/craftax-classic-full/states.pkl"


def u8(a):
    return np.round(np.asarray(a, np.float64)).clip(0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    states = pickle.load(open(STATES, "rb"))
    res = {}

    for bps in (7, 16):
        render = jax.jit(make_craftax_pixel_renderer(bps))
        Hw = OBS_DIM[0] * bps
        rows = []
        for st in states:
            light = float(np.asarray(jax.device_get(st.light_level)))
            base = u8(jax.device_get(render(st)))
            alt = u8(jax.device_get(render(st.replace(
                state_rng=jax.random.PRNGKey(123456)))))
            world_diff = int((base[:Hw] != alt[:Hw]).sum())
            strip_diff = int((base[Hw:] != alt[Hw:]).sum())
            rows.append(dict(light=light, world_px_changed=world_diff,
                             strip_px_changed=strip_diff))
        night = [r for r in rows if r["light"] < 0.5]
        day = [r for r in rows if r["light"] >= 0.5]
        res[f"bps{bps}_state_rng_observability"] = dict(
            n_states=len(rows),
            n_light_below_half=len(night),
            night_states_whose_world_changed=sum(1 for r in night if r["world_px_changed"]),
            night_mean_world_channels_changed=(
                float(np.mean([r["world_px_changed"] for r in night])) if night else None),
            night_max_world_channels_changed=(
                max(r["world_px_changed"] for r in night) if night else None),
            world_channels_total=Hw * OBS_DIM[1] * bps * 3,
            day_states_whose_world_changed=sum(1 for r in day if r["world_px_changed"]),
            any_strip_changed=sum(1 for r in rows if r["strip_px_changed"]),
        )
        print(json.dumps(res[f"bps{bps}_state_rng_observability"], indent=2))

    # ---- (4) render-window byte accounting
    nb = OBS_DIM[0] * OBS_DIM[1]
    today = nb * 2 + 1                      # blocks + mobs + facing, float32
    add = {
        "inventory_counts_12": 12,
        "intrinsics_4": 4,
        "light_level_1": 1,
        "is_sleeping_1": 1,
        "state_rng_key_2_uint32": 2,
    }
    grown = today + sum(add.values())
    frame7 = (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * 7 * OBS_DIM[1] * 7 * 3
    frame16 = (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * 16 * OBS_DIM[1] * 16 * 3
    res["state_window"] = dict(
        today_floats=today, today_bytes=today * 4,
        added_floats=add, added_bytes={k: v * 4 for k, v in add.items()},
        grown_floats=grown, grown_bytes=grown * 4,
        growth_bytes=(grown - today) * 4,
        growth_pct=round(100.0 * (grown - today) / today, 1),
        frame_bytes_7px_world_only=OBS_DIM[0] * 7 * OBS_DIM[1] * 7 * 3,
        frame_bytes_7px_full=frame7, frame_bytes_16px_full=frame16,
        ratio_frame_over_state_7px=round(frame7 / (grown * 4), 2),
        ratio_frame_over_state_16px=round(frame16 / (grown * 4), 2),
        note=("the PRNG key is 2 uint32 exactly representable in float32? NO: 2**32 > 2**24, so a "
              "float32 state row cannot carry a uint32 key losslessly. See findings.md."))

    # is a uint32 exactly representable in float32?
    rng = np.random.default_rng(0)
    k = rng.integers(0, 2 ** 32, size=100000, dtype=np.uint64).astype(np.uint32)
    rt = k.astype(np.float32).astype(np.uint64).astype(np.uint32)
    res["state_window"]["uint32_float32_roundtrip_exact_frac"] = float((rt == k).mean())
    # split into two 16-bit halves instead
    hi, lo = (k >> 16).astype(np.float32), (k & 0xFFFF).astype(np.float32)
    rt2 = ((hi.astype(np.uint64) << 16) | lo.astype(np.uint64)).astype(np.uint32)
    res["state_window"]["uint32_as_two_float32_halves_exact_frac"] = float((rt2 == k).mean())
    print(json.dumps(res["state_window"], indent=2))

    with open(os.path.join(OUT, "purity_and_state_window.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

"""Freeze facts from craftax itself that the full-Craftax unit tests check without importing jax.

Run in a venv with craftax 1.6.1 and jax:

    JAX_PLATFORMS=cpu python scripts/regression/craftax/gen_golden.py
"""

import hashlib
import json
import os
import sys

import numpy as np

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "tests",
    "fixtures",
    "regression",
    "fidelity",
    "craftax_full_golden.json",
)


def main():
    from importlib import metadata

    import jax
    from craftax.craftax import constants as C

    rec = {
        "craftax": metadata.version("craftax"),
        "jax": jax.__version__,
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "what": "craftax's own night_noise_intensity_texture and a few alphabet facts, frozen so the "
        "unit tests can assert them without importing jax",
        "vignette_sha256": {},
        "vignette_probe": {},
        "naive_divide_differs": {},
        "alpha_alphabets": {},
        "opaque_mob_slots": {},
    }
    for t in (10, 16):
        tex = C.load_all_textures_given_size(t)
        v = np.asarray(tex["night_noise_intensity_texture"])[..., 0].astype(np.float32)
        rec["vignette_sha256"][str(t)] = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        # a few exact values, so a failure says where the vignette differs
        h, w = v.shape
        rec["vignette_probe"][str(t)] = {
            f"{r},{c}": float(v[r, c])
            for r, c in ((0, 0), (h // 2, w // 2), (h // 2, 0), (0, w // 2), (h - 1, w - 1), (3, 7))
        }
        # how many elements a true float32 divide would get wrong
        x = (np.arange(h, dtype=np.float32) - (h // 2))[:, None]
        y = (np.arange(w, dtype=np.float32) - (w // 2))[None, :]
        raw = np.sqrt(x**2 + y**2).astype(np.float32)
        naive = (raw / raw.max()).astype(np.float32)
        rec["naive_divide_differs"][str(t)] = int((naive != v).sum())
        rec["alpha_alphabets"][str(t)] = {
            k: sorted(float(z) for z in np.unique(np.asarray(tex[k])))
            for k in (
                "melee_mob_texture_alphas",
                "passive_mob_texture_alphas",
                "ranged_mob_texture_alphas",
                "projectile_texture_alphas",
                "full_map_player_textures_alpha",
            )
        }
        rec["alpha_alphabets"][str(t)]["full_map_item_textures[a]"] = sorted(
            float(z) for z in np.unique(np.asarray(tex["full_map_item_textures"])[..., 3])
        )
    rec["geometry"] = {
        "OBS_DIM": list(C.OBS_DIM),
        "INVENTORY_OBS_HEIGHT": int(C.INVENTORY_OBS_HEIGHT),
        "n_block": len(C.BlockType),
        "n_item": len(C.ItemType),
        "registered_tile_px": int(C.BLOCK_PIXEL_SIZE_AGENT),
        "inventory_cap": 99,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", OUT)
    print(json.dumps(rec, indent=2)[:1500])


if __name__ == "__main__":
    sys.exit(main())

"""Enumerate every registered MiniGrid/BabyAI env and record what the tile model needs.

Instantiating each id is the only honest way to find the ones that break the model:
a grid that is not a grid, a view size that is not odd, an object outside
OBJECT_TO_IDX, a render path that is not Grid.render.
"""
from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.minigrid_env import MiniGridEnv

ids = sorted(k for k in gym.registry
             if k.startswith("MiniGrid-") or k.startswith("BabyAI-"))

rows, fails = [], []
by_shape: dict = {}
for eid in ids:
    try:
        env = gym.make(eid)
        u = env.unwrapped
        env.reset(seed=0)
        row = {
            "id": eid,
            "is_minigrid_env": isinstance(u, MiniGridEnv),
            "grid": [u.width, u.height],
            "view": u.agent_view_size,
            "stw": bool(u.see_through_walls),
            "render_is_default": type(u).get_full_render is MiniGridEnv.get_full_render
            and type(u).get_pov_render is MiniGridEnv.get_pov_render
            and type(u).get_frame is MiniGridEnv.get_frame,
            "grid_cls": type(u.grid).__name__,
        }
        # every object present in a fresh episode must be inside the encode alphabet
        types = sorted({int(x) for x in u.grid.encode()[:, :, 0].ravel()})
        row["types_present"] = types
        rows.append(row)
        by_shape.setdefault(tuple(row["grid"]), []).append(eid)
        env.close()
    except Exception as ex:  # noqa: BLE001
        fails.append({"id": eid, "error": f"{type(ex).__name__}: {ex}"})

summary = {
    "n_registered": len(ids),
    "n_minigrid": sum(1 for r in rows if r["id"].startswith("MiniGrid-")),
    "n_babyai": sum(1 for r in rows if r["id"].startswith("BabyAI-")),
    "n_instantiated": len(rows),
    "n_failed_to_build": len(fails),
    "all_are_MiniGridEnv": all(r["is_minigrid_env"] for r in rows),
    "all_default_render": all(r["render_is_default"] for r in rows),
    "all_grid_cls_Grid": sorted({r["grid_cls"] for r in rows}),
    "view_sizes": sorted({r["view"] for r in rows}),
    "see_through_walls_values": sorted({r["stw"] for r in rows}),
    "grid_dims_min": [min(r["grid"][0] for r in rows), min(r["grid"][1] for r in rows)],
    "grid_dims_max": [max(r["grid"][0] for r in rows), max(r["grid"][1] for r in rows)],
    "n_distinct_grid_shapes": len(by_shape),
    "failures": fails,
}
print(json.dumps(summary, indent=2))
with open(sys.argv[1] if len(sys.argv) > 1 else "envs.json", "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=2)

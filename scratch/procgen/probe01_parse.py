"""Does the transcribed schema actually describe the blob? Round-trip every game, byte for byte.

Also dumps, per game: the render-relevant fields, so the compact state can be sized from
measurement rather than from the C++ header.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state_io as S  # noqa: E402

GAMES = ["maze", "heist", "miner"]
ALL = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball",
       "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]


def main():
    from procgen import ProcgenGym3Env
    out = {}
    games = ALL if "--all" in sys.argv else GAMES
    for name in games:
        env = ProcgenGym3Env(num=2, env_name=name, distribution_mode="hard", rand_seed=0,
                             start_level=0, num_levels=0)
        rows = []
        rng = np.random.default_rng(0)
        for step in range(12):
            env.act(rng.integers(0, 15, size=2).astype(np.int32))
            blobs = env.get_state()
            for b in blobs:
                st = S.parse(b)
                ok = S.dump(st) == b
                rows.append({
                    "roundtrip": bool(ok),
                    "total": len(b),
                    "rng_text": S.rng_text_bytes(st),
                    "n_ent": len(st["entities"]),
                    "grid": [st["grid_w"], st["grid_h"]],
                    "rest": len(st["_rest"]),
                    "center_agent": st["center_agent"],
                    "use_backgrounds": st["use_backgrounds"],
                    "paint_vel_info": st["paint_vel_info"],
                    "bg_index": st["background_index"],
                    "bg_pct_x": st["bg_pct_x"],
                    "bg_tile_ratio": st["bg_tile_ratio"],
                    "unit": st["unit"], "view_dim": st["view_dim"],
                    "x_off": st["x_off"], "y_off": st["y_off"],
                    "visibility": st["visibility"],
                    "main": [st["main_width"], st["main_height"]],
                    "grid_types": sorted(set(np.unique(st["grid"]).tolist())),
                    "ent": [[e["type"], e["image_type"], e["image_theme"], e["render_z"],
                             round(e["x"], 4), round(e["y"], 4), round(e["rx"], 4),
                             round(e["ry"], 4), round(e["rotation"], 4),
                             e["is_reflected"], e["use_abs_coords"], round(e["alpha"], 3)]
                            for e in st["entities"]],
                })
        bad = [r for r in rows if not r["roundtrip"]]
        out[name] = {"n": len(rows), "roundtrip_fail": len(bad), "sample": rows[0],
                     "last": rows[-1],
                     "rest_bytes": sorted({r["rest"] for r in rows})}
        print(f"{name:10s} roundtrip {len(rows)-len(bad)}/{len(rows)}  "
              f"total={rows[0]['total']} rng={rows[0]['rng_text']} rest={rows[0]['rest']} "
              f"grid={rows[0]['grid']} ents={rows[0]['n_ent']} "
              f"unit={rows[0]['unit']:.6f} view_dim={rows[0]['view_dim']:.6f} "
              f"xoff={rows[0]['x_off']:.4f} yoff={rows[0]['y_off']:.4f} "
              f"center_agent={rows[0]['center_agent']} types={rows[0]['grid_types']}")
        for e in rows[0]["ent"]:
            print(f"    ent {e}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "parse_report.json"),
              "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()

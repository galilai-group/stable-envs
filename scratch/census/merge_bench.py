"""Merge the per-interpreter decode sweeps into one table: default vs ours vs stored.

Each env is measured in exactly one interpreter (the sweep is a set cover), so merging is a union with
no reconciliation to do. What it does have to preserve is WHERE each row came from, and how much that
matters is now a per-row measurement rather than a blanket claim. The official-renderer column never
travelled. The "ours" column was described as reproducing across nodes to under a percent, and that is
true of the fused rasterizer and FALSE of the ray-cast families: those renders are host-bound (cpu% at
or above 100 in the table below), so DMCBallInCupCatch at 84 px batch 4096 measured 3,664 frames/s on
an idle node and 271-978 on contended ones. Read a row's cpu% before comparing it with anything.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

RES = "/mnt/vast/home/randall.balestriero/code/stable-envs/examples/results/decode"

FAMILY = [
    ("MetaWorld", "Meta-World"), ("HumanoidBench", "HumanoidBench"), ("DMC", "DM Control"),
    ("RoboCasa", "RoboCasa"), ("Griddly", "Griddly"), ("Overcooked", "Overcooked"),
    ("Jumanji", "Jumanji"), ("Pgx", "PGX"), ("Procgen", "Procgen"), ("MiniHack", "MiniHack"),
    ("Craftax", "Craftax"), ("Crafter", "Crafter"), ("PushT", "PushT"), ("Fetch", "Fetch"),
    ("Mujoco", "MuJoCo locomotion"), ("BabyAI", "MiniGrid / BabyAI"),
    ("Antmaze", "OGBench"), ("Humanoidmaze", "OGBench"), ("Pointmaze", "OGBench"),
    ("PointMaze", "D4RL PointMaze"), ("SimplePointMaze", "swm nav"), ("TwoRoom", "swm nav"),
    ("Piecewise", "swm nav"), ("CarRacing", "CarRacing"), ("Sokoban", "Sokoban"),
    ("LunarLander", "LunarLander"),
]
MINIGRID = {"Empty", "FourRooms", "DoorKey", "MultiRoom", "LavaGap", "Crossing"}
HIGHWAY = {"Intersection", "Merge", "Parking", "Racetrack", "Roundabout", "Highway"}
CLASSIC = {"CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"}


def family(name: str) -> str:
    for pre, fam in FAMILY:
        if name.startswith(pre):
            return fam
    if name in MINIGRID or name.startswith("MiniGrid"):
        return "MiniGrid / BabyAI"
    if name in HIGHWAY:
        return "highway-env"
    if name in CLASSIC:
        return "classic control"
    return "other"


def cpu_fps(e: dict):
    c = e.get("cpu")
    if not isinstance(c, dict):
        return None
    v = c.get("set_state_render")
    return v.get("fps") if isinstance(v, dict) else None


rows, errors, meta = [], {}, {}
for path in sorted(glob.glob(f"{RES}/bench_vs_decode_*.json")):
    tag = re.sub(r"^bench_vs_decode_|\.json$", "", os.path.basename(path))
    d = json.load(open(path))
    meta[tag] = {"device": d.get("device"), "cpu": d.get("cpu"), "load": d.get("loadavg")}
    for name, e in d["envs"].items():
        if "error" in e:
            errors[name] = (tag, e["error"])
            continue
        g, dec = e.get("gpu_best"), e.get("decode")
        if not g or not dec:
            continue
        rows.append({
            "name": name, "family": family(name), "tag": tag,
            "shape": f"{dec['h']}x{dec['w']}", "gpu": g["fps"], "batch": g["batch"],
            "dec": dec["decode_fps_per_core"], "file": dec.get("load_decode_fps_per_core"),
            "cpu": cpu_fps(e), "state_b": e.get("state_bytes"), "jpeg_b": dec["jpeg_bytes"],
            "raw_b": dec["raw_bytes"],
            # The supersample the env RESOLVED to and how much of the timed loop was spent on CPU.
            # Both belong in the table: 44 classes declare default_supersample = 2 and were swept at
            # 1 until this was fixed, and the ray-cast families are host-bound (cpu% at or above 100),
            # so their throughput is a property of the node's processor and its load rather than of
            # the GPU. Rows measured before that instrumentation carry None and print as "?".
            "ss": g.get("supersample"), "cpu_frac": g.get("cpu_frac"),
        })

# Dedupe. The Meta-World chunk was re-split after a bad node, so a handful of envs appear under both
# the abandoned tag and its replacement, and a straight union counted them twice (203 rows for 196
# envs). Keep the row that has the official-renderer column, then the more recent file.
best_row: dict[str, dict] = {}
for r in rows:
    prev = best_row.get(r["name"])
    if prev is None or (r["cpu"] is not None and prev["cpu"] is None):
        best_row[r["name"]] = r
dupes = len(rows) - len(best_row)
rows = list(best_row.values())

rows.sort(key=lambda r: -r["gpu"] / r["dec"])

L = [f"# One frame, three ways — {len(rows)} environments", "",
     "Every column is frames/s for the same transformation, `state -> uint8 frame`, at the env's own",
     "native resolution.", "",
     "| column | what it is |", "|---|---|",
     "| **default** | the env's own official renderer, one frame at a time — the only entry point it "
     "offers. Blank where the reference package is not installed in the interpreter that owns that env. |",
     "| **ours** | the GPU path at its best batch. |",
     "| **stored** | one CPU core decoding *this env's own frame*, JPEG q90, already in memory — no "
     "filesystem, no network. The case most favourable to a stored dataset. |", "",
     "`cores` is **ours / stored**: how many CPU cores of decoding one GPU replaces.", "",
     "`ss` is the supersample the env resolved to (its own `default_supersample`, not a flag), and",
     "`cpu%` is process CPU seconds per wall second of the timed loop: at or above 100% the host is",
     "busy for the whole render and that row is a measurement of the node's processor under its load,",
     "not of the GPU. The ray-cast families sit there.", "",
     "| env | family | frame | ss | default | ours (batch) | cpu% | stored /core | **cores** |"
     " vs default |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    cpu = f"{r['cpu']:,.0f}" if r["cpu"] else "—"
    vs = f"{r['gpu'] / r['cpu']:,.0f}x" if r["cpu"] else "—"
    cf = r.get("cpu_frac")
    L.append(f"| {r['name']} | {r['family']} | {r['shape']} | {r.get('ss') or '?'} | {cpu} | "
             f"{r['gpu']:,.0f} ({r['batch']:,}) | "
             f"{'?' if cf is None else format(100 * cf, '.0f') + '%'} | {r['dec']:,.0f} | "
             f"**{r['gpu'] / r['dec']:,.1f}x** | {vs} |")

byfam = defaultdict(list)
for r in rows:
    byfam[r["family"]].append(r)
L += ["", "## By family", "",
      "| family | envs | frame | default f/s | ours f/s | stored f/s/core | **cores replaced** |",
      "|---|---|---|---|---|---|---|"]


def rng(vals, fmt=",.0f"):
    vals = [v for v in vals if v]
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    return format(lo, fmt) if abs(hi - lo) < 0.005 * max(hi, 1) else \
        f"{format(lo, fmt)}–{format(hi, fmt)}"


for fam in sorted(byfam, key=lambda f: -max(r["gpu"] / r["dec"] for r in byfam[f])):
    rs = byfam[fam]
    shapes = sorted({r["shape"] for r in rs})
    L.append(f"| {fam} | {len(rs)} | {shapes[0] if len(shapes) == 1 else f'{shapes[0]}…'} | "
             f"{rng([r['cpu'] for r in rs])} | {rng([r['gpu'] for r in rs])} | "
             f"{rng([r['dec'] for r in rs])} | "
             f"**{rng([r['gpu'] / r['dec'] for r in rs], ',.1f')}x** |")

fl = [r for r in rows if r["file"]]
if fl:
    ratios = sorted(r["file"] / r["dec"] for r in fl)
    L += ["", "## Does the *load* half of load-and-decode matter?", "",
          f"Across {len(fl)} envs, reading a warm local JPEG file and decoding it runs at "
          f"{ratios[0]:.2f}–{ratios[-1]:.2f}x the speed of decoding the same bytes already in memory "
          f"(median {ratios[len(ratios) // 2]:.2f}x).",
          "A stored-frame pipeline is decode-bound, not IO-bound, so the in-memory column is both the",
          "favourable bar and very nearly the only one. This says nothing about a dataset that has to",
          "cross a network — that is out of scope here and would do worse."]

nocpu = [r["name"] for r in rows if not r["cpu"]]
L += ["", "## Provenance and caveats", "",
      f"* Measured across {len(meta)} interpreters, since no single one can import every family.",
      "* **Neither column travels unconditionally.** A row whose `cpu%` is near 0 is one kernel on a",
      "  device the job owns and repeats across nodes to under a percent. A row at or above 100% is",
      "  host-bound — the ray-cast families are, at 100-206% — and moves with the node and its load:",
      "  DMCBallInCupCatch measured 3,664 f/s idle and 271-978 contended, a 13x spread. The",
      "  official-renderer column is a single-threaded Python renderer sharing a node and has moved by",
      "  more than 100x with load; read it as an order of magnitude.",
      f"* {len(nocpu)} envs have no `default` number because no reference package was installed in the",
      "  interpreter that owns them. That is an absence, not a zero.",
      "* One GPU against one CPU core is the right comparison for \"store or regenerate on the node I",
      "  have\", and the wrong one for a per-watt or per-dollar argument."]
if errors:
    L += ["", f"* {len(errors)} envs did not measure:"]
    L += [f"  * `{k}` ({v[0]}) — {v[1][:160]}" for k, v in sorted(errors.items())]

out = f"{RES}/ALL.md"
open(out, "w").write("\n".join(L) + "\n")
json.dump({"rows": rows, "meta": meta, "errors": errors}, open(f"{RES}/ALL.json", "w"), indent=2)
print("\n".join(L[:14]))
print(f"\n... wrote {out} ({len(rows)} envs, {len(errors)} errors)")

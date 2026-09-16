"""Generate examples/results/step8/variations-dmc-control/leaf_classification.json.

One row per (swm id, leaf): the swm declaration read live, the Step-8 class, the stable-envs control
it reaches (or the reason it reaches none), and the measurement behind the class.
"""
import argparse, json, os, numpy as np
from stable_envs.core.env import make
from stable_envs.compat.worldmodel import variation_map, check_mapping, swm_variation_space
from stable_envs.compat.variation_maps.dmc_control import MAPS, EXPECTED_COUNTS, LIGHT_RANGE

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--measure", required=True)
a = ap.parse_args()
M = lambda n: json.load(open(os.path.join(a.measure, n)))
dmc8 = M("dmc_leaf_measure2.json"); reach = M("model_reach.json")
classic = M("classic_measure.json"); phys = M("classic_phys.json"); defaults = M("defaults.json")

import gymnasium, stable_worldmodel  # noqa: F401

def leafspec(sp):
    d = {"type": type(sp).__name__}
    if hasattr(sp, "n"): d["n"] = int(sp.n)
    if hasattr(sp, "low"):
        d["low"] = np.asarray(sp.low).tolist(); d["high"] = np.asarray(sp.high).tolist()
        d["shape"] = list(sp.shape)
    d["default"] = np.asarray(sp.init_value).tolist()
    return d

rows, per_id, seen = [], {}, {}
for cell in sorted(MAPS):
    env = make(cell, batch=1, device="cpu")
    m = variation_map(env); chk = check_mapping(env)
    ref = gymnasium.make(m.swm_id); space = swm_variation_space(ref)
    ours_for = {}
    for o, (t, _) in m.pairs().items(): ours_for.setdefault(t, []).append(o)
    for o, t in m.rgb255.items(): ours_for.setdefault(t, []).append(o)
    for o, (t, _) in m.enum.items(): ours_for.setdefault(t, []).append(o)
    for name in sorted(space.names()):
        node = space
        for p in name.split("."): node = node[p]
        cls = chk["classification"][name]
        ev = {}
        if m.swm_id in dmc8:
            h = dmc8[m.swm_id]["hidden"].get(name, {}); v = dmc8[m.swm_id].get("visible", {}).get(name)
            if h: ev["px_changed_8_states"] = h["px"]
            if v: ev["px_changed_8_states_target_visible"] = v["px"]
            r = reach.get(m.swm_id, {}).get(name)
            if r is not None: ev["mjmodel_arrays_changed"] = r["model_arrays_changed"]
        if m.swm_id in phys and name in phys[m.swm_id]:
            ev["px_changed_8_states_official"] = phys[m.swm_id][name]["px"]
            ev["px_total"] = phys[m.swm_id][name]["px_total"]
        if m.swm_id in classic and name in classic[m.swm_id]["cases"]:
            c = classic[m.swm_id]["cases"][name]
            ev["swm_vs_ours_px_at_probe"] = c["px_diff"]; ev["px_total"] = c["px_total"]
            ev["swm_vs_ours_px_at_default"] = classic[m.swm_id]["cases"]["default"]["px_diff"]
        rows.append({"swm_id": m.swm_id, "stable_envs_cell": cell, "leaf": name,
                     "swm": leafspec(node), "classification": cls,
                     "stable_envs_control": sorted(ours_for.get(name, [])) or None,
                     "reason": chk["reasons"].get(name), "note": chk["notes"].get(name),
                     "evidence": ev})
    ref.close()
    if m.swm_id not in seen:
        seen[m.swm_id] = cell
        per_id[m.swm_id] = {"cells": [cell], "counts": EXPECTED_COUNTS[m.swm_id],
                            "registered_size": defaults[cell]["size"],
                            "default_style_slots_moved_to_reach_the_swm_default":
                                sorted(defaults[cell]["style_slots_differing"]),
                            "px_default_vs_mapped_default_8_states":
                                defaults[cell]["px_default_vs_mapped"],
                            "px_total": defaults[cell]["px_total"]}
    else:
        per_id[m.swm_id]["cells"].append(cell)
    if m.swm_id in LIGHT_RANGE.get(cell, ()) or cell in LIGHT_RANGE:
        b, lo, hi, _ = LIGHT_RANGE[cell]
        per_id[m.swm_id]["light_intensity_reachable_range"] = [lo, hi]
        per_id[m.swm_id]["light_diffuse_base"] = b

tot = {k: 0 for k in ("mapped", "unsupported", "dead_upstream", "not_applicable")}
for sid, r in per_id.items():
    for k in tot: tot[k] += r["counts"][k]
json.dump({"what_this_is": "Step-8 Workstream A: every stable_worldmodel variation leaf of the five "
                           "classic-control and twelve DM Control ids, classified, with the "
                           "measurement behind the class.",
           "schema_version": 1, "step": 8, "workstream": "variations-dmc-control",
           "baseline_sha": "43ac95238f7472137619f3e69e7636e11dd088b1",
           "swm_source": "galilai-group/stable-worldmodel @ 176d545",
           "venv": "/Users/samib/se8-vm-venv (python 3.12, torch 2.14.0 cpu, mujoco 3.12.0, "
                   "dm_control 1.0.45, gymnasium 1.3.0, MUJOCO_GL=glfw)",
           "totals": {**tot, "leaves": sum(tot.values()), "ids": len(per_id), "cells": len(MAPS)},
           "per_id": per_id, "rows": rows}, open(a.out, "w"), indent=1)
print("wrote", a.out, tot, sum(tot.values()))

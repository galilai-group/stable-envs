"""Render the per-id leaf tables of docs/research/step8_variations_dmc_control.md from the artifact."""
import argparse, json, textwrap
ap=argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); a=ap.parse_args()
d=json.load(open(a.src))
by={}
for r in d["rows"]:
    by.setdefault(r["swm_id"],{}).setdefault(r["leaf"],r)
ORDER=["swm/AcrobotControl-v1","swm/CartPoleControl-v1","swm/PendulumControl-v1",
 "swm/MountainCarControl-v0","swm/MountainCarContinuousControl-v0","swm/AcrobotDMControl-v0",
 "swm/BallInCupDMControl-v0","swm/CartpoleDMControl-v0","swm/CheetahDMControl-v0",
 "swm/FingerDMControl-v0","swm/HopperDMControl-v0","swm/HumanoidDMControl-v0",
 "swm/ManipulatorDMControl-v0","swm/PendulumDMControl-v0","swm/QuadrupedDMControl-v0",
 "swm/ReacherDMControl-v0","swm/WalkerDMControl-v0"]
def rng(s):
    if "n" in s: return f"Discrete({s['n']})"
    lo,hi=s["low"],s["high"]
    f=lambda x: x if not isinstance(x,list) else (f(x[0]) if len(set(map(str,x)))==1 else x)
    def flat(x):
        while isinstance(x,list): x=x[0]
        return x
    return f"Box[{flat(lo):g}, {flat(hi):g}]{'' if s['shape']==[1] else ' shape '+str(tuple(s['shape']))}"
def dflt(s):
    v=s["default"]
    if isinstance(v,list) and len(v)==1: return f"{v[0]:g}"
    return str(v)
def ev(r):
    e=r["evidence"]; bits=[]
    if "mjmodel_arrays_changed" in e:
        c=e["mjmodel_arrays_changed"]
        bits.append("no MjModel array moves" if not c else "moves "+", ".join(f"`{x}`" for x in c[:3])+("..." if len(c)>3 else ""))
    if "px_changed_8_states" in e: bits.append(f"{e['px_changed_8_states']:,} px / 8 states")
    if "px_changed_8_states_target_visible" in e and e["px_changed_8_states_target_visible"]!=e.get("px_changed_8_states"):
        bits.append(f"{e['px_changed_8_states_target_visible']:,} px with the target visible")
    if "px_changed_8_states_official" in e: bits.append(f"{e['px_changed_8_states_official']:,} px / {e['px_total']:,} in the official frame")
    if "swm_vs_ours_px_at_default" in e:
        bits.append(f"swm-vs-ours {e['swm_vs_ours_px_at_default']} px at the default, "
                    f"{e['swm_vs_ours_px_at_probe']:,} / {e['px_total']:,} at a probe")
    return "; ".join(bits) or "—"
def one(s):
    return " ".join(textwrap.wrap(s.replace("\n"," "),10**6))
out=[]
for sid in ORDER:
    rows=by[sid]; pid=d["per_id"][sid]
    c=pid["counts"]
    out.append(f"### `{sid}`\n")
    out.append(f"cell(s): {', '.join('`'+x+'`' for x in pid['cells'])} — registered size {pid['registered_size']} px — "
               f"**{c['total']} leaves: {c['mapped']} mapped, {c['unsupported']} unsupported, "
               f"{c['dead_upstream']} dead upstream, {c['not_applicable']} not applicable**\n")
    if pid["default_style_slots_moved_to_reach_the_swm_default"]:
        out.append(f"The swm REGISTERED DEFAULT is a different appearance from the compiled model's and is reached "
                   f"through {', '.join('`'+x+'`' for x in pid['default_style_slots_moved_to_reach_the_swm_default'])} "
                   f"({pid['px_default_vs_mapped_default_8_states']:,} of {pid['px_total']:,} px differ from the cell's "
                   f"default style over 8 states).\n")
    else:
        out.append(f"The swm registered default IS the cell's default style: **0 of {pid['px_total']:,} px** differ "
                   f"over 8 states.\n")
    out.append("| leaf | swm type / range / default | class | stable-envs control | evidence |")
    out.append("|---|---|---|---|---|")
    for name in sorted(rows):
        r=rows[name]; s=r["swm"]
        ctl = ", ".join("`"+x+"`" for x in r["stable_envs_control"]) if r["stable_envs_control"] else "—"
        out.append(f"| `{name}` | {rng(s)}, default {dflt(s)} | **{r['classification']}** | {ctl} | {ev(r)} |")
    notes=[(n,rows[n]["note"]) for n in sorted(rows) if rows[n]["note"]]
    reasons=[(n,rows[n]["reason"]) for n in sorted(rows) if rows[n]["reason"]]
    if notes or reasons:
        out.append("")
        for n,t in notes: out.append(f"* `{n}` — {one(t)}")
        for n,t in reasons: out.append(f"* `{n}` — {one(t)}")
    out.append("")
open(a.out,"w").write("\n".join(out))
print("wrote",a.out,len(out),"lines")

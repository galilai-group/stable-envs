"""Generate the paper's throughput tables from the measurement JSONs. Never hand-transcribe a number.

Emits, into ``examples/paper/tables/``:

  ``family.tex``     one row per family, the body's headline table. A rule is drawn where the median
                     crosses the storage baseline, so a family that gains a kernel moves across it by
                     itself rather than by an edit.
  ``summary.tex``    a handful of recognisable environments plus the aggregate over all of them. The
                     named rows are chosen to SPAN the range rather than to flatter it: the best
                     family, a mid one, and the two that lose to reading.
  ``full_N.tex``     the complete registry, split into appendix pages.
  ``macros.tex``     every quantity the paper states in PROSE, as ``\newcommand``. main.tex inputs it
                     and writes ``\seStoredMedian`` rather than a digit, so re-running this script
                     updates the sentences as well as the tables. A paper whose tables regenerate and
                     whose paragraphs do not is a paper that will eventually contradict itself.
  ``stats.json``     the same aggregates as data, including the per-family roll-up.

Three columns, one operating point for every row --- 256 px, batch 1024 --- so the table compares
environments rather than operating points:

  ``default``   the environment's own official renderer, one frame at a time.
  ``ours``      the GPU path.
  ``stored``    JPEG-in-Lance, the layout ``stable_worldmodel`` writes, read a chunk at a time and
                batch-decoded through libjpeg-turbo. Measured for every environment in ONE process on
                ONE node, so the only thing that varies down the column is compressed bytes per frame.

The two speedup columns are ``ours/default`` and ``ours/stored``, and they answer different questions:
the first is "is this faster than calling the simulator's renderer", the second is "is this faster than
loading the dataset you would otherwise have built".
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "examples/results/storage"
OUT = Path(__file__).resolve().parent / "tables"

#: Rows for the body table. Ordered to span the range, not to flatter it.
FEATURED = [
    ("GriddlySokoban", "Griddly Sokoban"),
    ("OvercookedCrampedRoom", "Overcooked"),
    ("MiniGridEmpty", "MiniGrid Empty"),
    ("JumanjiMaze", "Jumanji Maze"),
    ("CraftaxClassic", "Craftax"),
    ("PushT", "PushT"),
    ("ProcgenMaze", "Procgen Maze"),
    ("Crafter", "Crafter"),
    ("CartPole", "CartPole"),
    ("PointmazeLargeNavigate", "OGBench pointmaze-L"),
    ("AntmazeLargeNavigate", "OGBench antmaze-L"),
    ("FetchPickAndPlace", "Fetch Pick-and-Place"),
    ("HumanoidBenchH1Walk", "HumanoidBench H1-walk"),
    ("MujocoHalfCheetah", "Gym MuJoCo HalfCheetah"),
    ("MetaWorldReach", "Meta-World reach"),
    ("RoboCasaCounterToSink", "RoboCasa counter-to-sink"),
    ("DMCCheetahRun", "DM Control cheetah-run"),
    ("CarRacing", "CarRacing"),
]


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def num(v, dp=0, dash="---"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return dash
    return f"{v:,.{dp}f}"


def load() -> dict:
    """Join the three measurements per env: GPU at 256/1024, official renderer, Lance read."""
    envs: dict[str, dict] = {}

    for p in sorted(glob.glob(str(STORE / "storage_*.json"))):
        d = json.load(open(p))
        for k, v in d["envs"].items():
            if "error" in v:
                continue
            e = envs.setdefault(k, {})
            e["size"] = v.get("size")
            e["size_note"] = v.get("size_note")
            e["shape"] = "x".join(str(x) for x in v.get("frame_shape", []))
            e["state_bytes"] = v.get("state_bytes")
            if isinstance(v.get("gpu"), dict) and v["gpu"].get("fps"):
                e["gpu"] = v["gpu"]["fps"]
                e["batch"] = v["gpu"].get("batch")
                # The resolved supersample and the host fraction belong to the number they were taken
                # with, so they come from THIS sweep -- the 256/1024 operating point the table uses --
                # rather than from the native-resolution one, which is a different measurement.
                e["ss"] = v["gpu"].get("supersample")
                e["cpu_frac"] = v["gpu"].get("cpu_frac")

    lance = STORE / "lance.json"
    if lance.exists():
        for k, v in json.load(open(lance))["envs"].items():
            e = envs.setdefault(k, {})
            e["jpeg_b"] = v.get("jpeg_bytes_per_frame")
            e["raw_b"] = v.get("raw_bytes_per_frame")
            for tag, key in (("local_1", "st_local"), ("shared_1", "st_shared"),
                             ("local_n", "st_par"), ("local_gpu", "st_gpu")):
                if isinstance(v.get(tag), dict) and v[tag].get("fps"):
                    e[key] = v[tag]["fps"]

    # The official-renderer column comes from the native-resolution sweep. It is a CPU number that does
    # not move with resolution the way the GPU one does, and re-measuring it at 256 for 196 envs buys
    # nothing; the caveat is recorded in the caption rather than hidden.
    allj = REPO / "examples/results/decode/ALL.json"
    if allj.exists():
        for r in json.load(open(allj))["rows"]:
            e = envs.setdefault(r["name"], {})
            e.setdefault("family", r["family"])
            if r.get("cpu"):
                e["cpu"] = r["cpu"]
                e["cpu_shape"] = r["shape"]
            # Bytes per frame and per state come from the native-resolution sweep, which is where the
            # picture the environment actually produces is measured. The storage claim is computed from
            # these rather than asserted.
            e["state_b"], e["raw_b"], e["jpeg_b"] = r.get("state_b"), r.get("raw_b"), r.get("jpeg_b")
            # Fall back to the native-resolution sweep for the two provenance fields only when the
            # 256/1024 sweep did not record them. A GPU number measured at the wrong supersample
            # describes a different picture, and one with cpu_frac near 1 is a property of the node
            # rather than of the device, so a row without either is a row that cannot be read.
            e.setdefault("ss", r.get("ss"))
            e.setdefault("cpu_frac", r.get("cpu_frac"))
            if e.get("ss") is None:
                e["ss"] = r.get("ss")
            if e.get("cpu_frac") is None:
                e["cpu_frac"] = r.get("cpu_frac")
    return envs


#: The headline "stored" number is the fastest disk-based path that uses only the CPU -- NOT the fastest
#: path measured, which is nvJPEG on the GPU and which this file now reports beside it.
#:
#: The distinction was previously collapsed, and collapsing it chose the conclusion. Measured over the
#: same 195 environments at the same operating point:
#:
#:     1 core            median 7.77x   geomean 13.37x     0/195 below 1.0
#:     8 workers         median 1.39x   geomean  2.26x    81/195 below 1.0
#:     nvJPEG on GPU     median 0.66x   geomean  1.31x   107/195 below 1.0
#:
#: So against the strongest baseline the MEDIAN environment loses and the geometric mean is a wash.
#: That is the honest headline and it does not damage the thesis: the storage claim -- 1,929x against
#: raw pixels, 90x against JPEG -- is what the paper rests on, and it holds against every path here,
#: because nvJPEG still requires the frames to have been written down.
#:
#: Both ratios are emitted, and every claim in the prose is a macro so the two cannot drift.
STORED = "st_par"

#: The fastest path measured, full stop. Reported beside STORED rather than instead of it: an 8-worker
#: CPU decode is what most people actually run, and nvJPEG is what the baseline BECOMES if it is
#: optimised as hard as ours is. A comparison that omits either is answering a different question.
STORED_GPU = "st_gpu"


def row(name: str, e: dict, label: str | None = None) -> str:
    ours, cpu = e.get("gpu"), e.get("cpu")
    sto = e.get(STORED)
    vs_def = ours / cpu if ours and cpu else None
    vs_sto = ours / sto if ours and sto else None
    star = "$^{*}$" if e.get("size_note") else ""
    return (f"{tex(label or name)}{star} & {e.get('shape', '---')} & {num(cpu)} & "
            f"{num(ours)} & {num(sto)} & "
            f"{num(vs_def, 0) if (vs_def or 0) >= 10 else num(vs_def, 1)}$\\times$ & "
            f"\\textbf{{{num(vs_sto, 2) if (vs_sto or 0) < 10 else num(vs_sto, 0)}$\\times$}} \\\\")


#: Order for the per-family table: the families whose median clears the storage baseline first, then the
#: ones that do not, each block sorted by that median. Written by ``family_rows`` rather than by hand so
#: that a family which crosses the line moves on its own.
def family_rows(usable: dict) -> list[dict]:
    """One row per family: counts, median throughputs and the two median ratios, best ratio first."""
    byfam: dict[str, list[dict]] = {}
    for v in usable.values():
        byfam.setdefault(v.get("family", "unknown"), []).append(v)
    out = []
    for fam, vs in byfam.items():
        rs = [v["gpu"] / v[STORED] for v in vs if v.get(STORED)]
        gs = [v["gpu"] / v[STORED_GPU] for v in vs if v.get(STORED_GPU)]
        ds = [v["gpu"] / v["cpu"] for v in vs if v.get("cpu")]
        if not rs:
            continue
        # State bytes against raw frame bytes, the storage claim, per family. Generated for the same
        # reason as the ratios: the hand-written version of this column had drifted by a factor of two.
        comp = sorted(v["raw_b"] / v["state_b"] for v in vs if v.get("raw_b") and v.get("state_b"))
        out.append({"family": fam, "n": len(vs),
                    "ours": st.median([v["gpu"] for v in vs]),
                    "stored": st.median([v[STORED] for v in vs if v.get(STORED)]),
                    "compress": (comp[0], comp[-1]) if comp else None,
                    "vs_default": st.median(ds) if ds else float("nan"),
                    "vs_stored": st.median(rs), "min": min(rs), "max": max(rs),
                    "vs_stored_gpu": st.median(gs) if gs else float("nan")})
    return sorted(out, key=lambda r: -r["vs_stored"])


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-page", type=int, default=58)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    envs = load()
    usable = {k: v for k, v in envs.items() if v.get("gpu")}
    print(f"{len(envs)} envs joined, {len(usable)} with a GPU number, "
          f"{sum(1 for v in usable.values() if v.get('st_local'))} with a Lance number, "
          f"{sum(1 for v in usable.values() if v.get('cpu'))} with an official-renderer number")

    ratios = [v["gpu"] / v[STORED] for v in usable.values() if v.get(STORED)]
    gratios = [v["gpu"] / v[STORED_GPU] for v in usable.values() if v.get(STORED_GPU)]
    vsdef = [v["gpu"] / v["cpu"] for v in usable.values() if v.get("cpu")]

    # ---- body summary ----
    body = [r"% GENERATED by examples/paper/make_tables.py -- do not edit by hand",
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Environment & Frame & Default & Ours & Stored & $\times$default & $\times$stored \\",
            r"\midrule"]
    for name, label in FEATURED:
        if name in usable:
            body.append(row(name, usable[name], label))
    body += [r"\midrule",
             (f"\\textbf{{All {len(usable)} environments}} & 256$^{{2}}$ & "
              f"median {num(st.median([v['cpu'] for v in usable.values() if v.get('cpu')]))} & "
              f"median {num(st.median([v['gpu'] for v in usable.values()]))} & "
              f"median {num(st.median([v[STORED] for v in usable.values() if v.get(STORED)]))} & "
              f"{num(st.median(vsdef))}$\\times$ & "
              f"\\textbf{{{num(st.median(ratios), 1)}$\\times$}} \\\\"),
             (f"\\quad arithmetic mean & & & & & {num(st.mean(vsdef))}$\\times$ & "
              f"\\textbf{{{num(st.mean(ratios), 1)}$\\times$}} \\\\"),
             (f"\\quad geometric mean & & & & & {num(st.geometric_mean(vsdef))}$\\times$ & "
              f"\\textbf{{{num(st.geometric_mean(ratios), 2)}$\\times$}} \\\\"),
             (f"\\quad range & & & & & "
              f"{num(min(vsdef), 1)}--{num(max(vsdef))}$\\times$ & "
              f"{num(min(ratios), 2)}--{num(max(ratios))}$\\times$ \\\\"),
             r"\bottomrule", r"\end{tabular}"]
    (OUT / "summary.tex").write_text("\n".join(body) + "\n")

    # ---- per-family roll-up ----
    # The paper's thesis is now two-sided -- storage everywhere, speed for most families and not all --
    # and this is the table that has to carry it, so it is generated rather than asserted. A rule is
    # drawn where the median crosses 1.0, which is the only place the verdict changes.
    fams = family_rows(usable)
    above = [f for f in fams if f["vs_stored"] >= 1.0]
    fl = [r"% GENERATED by examples/paper/make_tables.py -- do not edit by hand",
          r"\begin{tabular}{lrlrrrrr}", r"\toprule",
          (r"Family & Envs & State\,:\,frame & Ours f/s & $\times$default & $\times$stored"
           r"$_{\mathrm{cpu}}$ & $\times$stored$_{\mathrm{gpu}}$ & Range $\times$stored$_{\mathrm{cpu}}$ \\"),
          r"\midrule"]
    for i, f in enumerate(fams):
        if i == len(above):
            fl.append(r"\midrule")
        # One precision for both ends of a range, chosen from the wider one, so a family does not read
        # as "74.39--74". A family of one has no range to report.
        hi, lo, n = f["max"], f["min"], f["n"]
        dp = 2 if hi < 10 else (1 if hi < 100 else 0)
        rng = "---" if n == 1 else f"{num(lo, dp)}--{num(hi, dp)}"
        c = f["compress"]
        comp = "---" if not c else (f"1/{num(c[0])}" if c[0] == c[1]
                                    else f"1/{num(c[0])}--1/{num(c[1])}")
        md, ms, mg = f["vs_default"], f["vs_stored"], f["vs_stored_gpu"]
        gtxt = "---" if mg != mg else (num(mg, 2) if mg < 10 else num(mg, 0)) + r"$\times$"
        fl.append(f"{tex(f['family'])} & {n} & {comp} & {num(f['ours'])} & "
                  f"{num(md, 0) if (md or 0) >= 10 else num(md, 1)}$\\times$ & "
                  f"\\textbf{{{num(ms, 2) if ms < 10 else num(ms, 0)}$\\times$}} & {gtxt} & {rng} \\\\")
    fl += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "family.tex").write_text("\n".join(fl) + "\n")

    # ---- appendix, paginated ----
    ordered = sorted(usable.items(), key=lambda kv: -(kv[1]["gpu"] / kv[1].get(STORED, math.inf)
                                                      if kv[1].get(STORED) else 0))
    pages = [ordered[i:i + a.per_page] for i in range(0, len(ordered), a.per_page)]
    for i, page in enumerate(pages):
        L = [r"% GENERATED by examples/paper/make_tables.py -- do not edit by hand",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Environment & Frame & Default & Ours & Stored & $\times$default & $\times$stored \\",
             r"\midrule"]
        L += [row(k, v) for k, v in page]
        L += [r"\bottomrule", r"\end{tabular}"]
        (OUT / f"full_{i}.tex").write_text("\n".join(L) + "\n")
    print(f"wrote summary.tex and {len(pages)} appendix pages "
          f"({a.per_page} rows each) to {OUT}")

    # ---- appendix driver: one longtable-free page per chunk, each in its own table environment ----
    drv = [r"% GENERATED by examples/paper/make_tables.py -- do not edit by hand"]
    for i in range(len(pages)):
        drv += [r"\begin{table}[t]", r"\centering",
                (f"\\caption{{Per-environment throughput at $256\\times256$, batch 1024 "
                 f"(page {i + 1} of {len(pages)}), ordered by $\\times$stored.}}"),
                f"\\label{{tab:full{i}}}", r"\vskip 0.06in", r"\scriptsize",
                r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{0.92}",
                f"\\input{{tables/full_{i}}}", r"\end{table}", r"\clearpage"]
    (OUT / "appendix_pages.tex").write_text("\n".join(drv) + "\n")

    extra: list[str] = []

    # ---- the other storage paths ----
    lj = STORE / "lance.json"
    if lj.exists():
        d = json.load(open(lj))
        cols = {"local_1": [], "shared_1": [], "local_n": [], "local_gpu": []}
        for v in d["envs"].values():
            for k in cols:
                if isinstance(v.get(k), dict) and v[k].get("fps"):
                    cols[k].append(v[k]["fps"])
        names = {"local_1": "node-local NVMe, 1 core",
                 "shared_1": "shared filesystem, 1 core",
                 "local_n": f"node-local, {d.get('workers', 8)} worker processes",
                 "local_gpu": "node-local, nvJPEG decode on the GPU"}
        L = [r"% GENERATED by examples/paper/make_tables.py",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Read path & Median f/s & Min & Max \\", r"\midrule"]
        for k, label in names.items():
            if cols[k]:
                L.append(f"{label} & {num(st.median(cols[k]))} & {num(min(cols[k]))} & "
                         f"{num(max(cols[k]))} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "storage_paths.tex").write_text("\n".join(L) + "\n")

    # ---- what a frozen encoder sees ----
    # Reported SPLIT BY TIER and never pooled. The visual tier is a different render of the same scene
    # by construction, so a pooled mean would both flatter the visual tier and slander the parity one.
    dj = REPO / "examples/results/dino/dino_agreement.json"
    if dj.exists():
        d = json.load(open(dj))
        ok = {k: v for k, v in d["envs"].items() if "ratio_ours_over_jpeg95" in v}
        L = [r"% GENERATED by examples/paper/make_tables.py -- do not edit by hand",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             (r"Tier & Envs & Ours & JPEG q95 & 2 states & Blank \\"), r"\midrule"]
        dm = {}
        for tier in ("parity", "visual"):
            g = [v for v in ok.values() if v.get("tier") == tier]
            if not g:
                continue
            def med(key):
                return st.median([v[key] for v in g])
            L.append(f"{tier} & {len(g)} & \\textbf{{{med('dino_cos_ours_vs_official'):.4f}}} & "
                     f"{med('dino_cos_jpeg95_vs_official'):.4f} & "
                     f"{med('dino_cos_state_vs_state'):.4f} & "
                     f"{med('dino_cos_blank_vs_official'):.3f} \\\\")
            closer = sum(1 for v in g if v["ratio_ours_over_jpeg95"] < 1.0)
            dm[tier] = {"n": len(g), "ours": med("dino_cos_ours_vs_official"),
                        "jpeg": med("dino_cos_jpeg95_vs_official"),
                        "scale": med("dino_cos_state_vs_state"),
                        "blank": med("dino_cos_blank_vs_official"), "closer": closer}
        L += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "dino.tex").write_text("\n".join(L) + "\n")
        for tier, m in dm.items():
            t = tier.capitalize()
            extra.append(macro(f"seDino{t}N", str(m["n"])))
            extra.append(macro(f"seDino{t}Ours", f"{m['ours']:.4f}"))
            extra.append(macro(f"seDino{t}Jpeg", f"{m['jpeg']:.4f}"))
            extra.append(macro(f"seDino{t}Scale", f"{m['scale']:.4f}"))
            extra.append(macro(f"seDino{t}Blank", f"{m['blank']:.3f}"))
            extra.append(macro(f"seDino{t}Closer", str(m["closer"])))
            extra.append(macro(f"seDino{t}PctScale",
                               f"{100 * m['ours'] / max(m['scale'], 1e-12):.0f}"))
        extra.append(macro("seDinoModel", r"DINOv2 ViT-B/14"))
        extra.append(macro("seDinoN", str(d.get("n", 32))))

    # ---- DDP ----
    rows_ddp = []
    for w in (1, 2, 4, 8):
        p = REPO / f"examples/results/ddp_w{w}/ddp_train_from_states.json"
        if p.exists():
            rows_ddp.append(json.load(open(p)))
    if rows_ddp:
        L = [r"% GENERATED by examples/paper/make_tables.py",
             r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Ranks & Frames/s & Render f/s per rank & Render \% of step & Peak GB/rank \\",
             r"\midrule"]
        for d in rows_ddp:
            L.append(f"{d['world_size']} & {num(d['frames_per_s'])} & "
                     f"{num(st.mean(d['render_fps_per_rank']))} & "
                     f"{100 * d['render_frac_of_step']:.0f}\\% & "
                     f"{d['per_rank'][0]['peak_gb']:.2f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "ddp.tex").write_text("\n".join(L) + "\n")

    # ---- the storage claim ----
    # Bytes weighted by bytes: the state that determines every frame in the registry against the raw
    # pixels of those frames. The per-env median is more than twice as large, so the summed ratio is the
    # conservative statement and the one the prose makes.
    have = [v for v in envs.values() if v.get("state_b") and v.get("raw_b")]
    tot_state, tot_raw = sum(v["state_b"] for v in have), sum(v["raw_b"] for v in have)
    tot_jpeg = sum(v["jpeg_b"] for v in have if v.get("jpeg_b"))
    per_env_raw = sorted(v["raw_b"] / v["state_b"] for v in have)
    per_env_med = st.median(per_env_raw)

    # ---- what the GPU rows are actually measuring ----
    cpu_fracs = [v["cpu_frac"] for v in usable.values() if v.get("cpu_frac") is not None]
    ss_two = [k for k, v in usable.items() if v.get("ss") == 2]

    stats = {"n_envs": len(usable),
             "n_families": len({v.get("family") for v in usable.values() if v.get("family")}),
             "vs_stored": {"median": st.median(ratios), "geomean": st.geometric_mean(ratios),
                           "min": min(ratios), "max": max(ratios),
                           "below_1": sum(1 for r in ratios if r < 1)},
             "vs_default": {"median": st.median(vsdef), "geomean": st.geometric_mean(vsdef),
                            "min": min(vsdef), "max": max(vsdef)},
             "families": fams,
             "storage": {"n": len(have), "state_bytes": tot_state, "raw_bytes": tot_raw,
                         "jpeg_bytes": tot_jpeg, "raw_over_state": tot_raw / tot_state,
                         "jpeg_over_state": tot_jpeg / tot_state,
                         "per_env_median": per_env_med,
                         "per_env_min": per_env_raw[0], "per_env_max": per_env_raw[-1]},
             "host_bound": {"n": len(cpu_fracs),
                            "median": st.median(cpu_fracs) if cpu_fracs else None,
                            "max": max(cpu_fracs) if cpu_fracs else None,
                            "at_or_above_1": sum(1 for c in cpu_fracs if c >= 1.0),
                            "n_supersample_2": len(ss_two)}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))

    # ---- macros, so the PROSE cannot drift from the tables either ----
    # Every quantity the body states in words is defined here and used as a macro in main.tex. Rerunning
    # this script therefore updates the sentences as well as the tables, which is the only way a claim
    # like "most families clear the bar" stays true when a kernel lands.
    n_above = len(above)
    M = [r"% GENERATED by examples/paper/make_tables.py -- do not edit by hand.",
         r"% main.tex states these in prose; defining them here is what keeps the sentences and the",
         r"% tables from disagreeing after a re-measurement.",
         macro("seNenvs", f"{len(usable):,}"),
         macro("seNfamilies", str(stats["n_families"])),
         macro("seNfamRated", str(len(fams))),
         macro("seNfamAbove", str(n_above)),
         macro("seNfamBelow", str(len(fams) - n_above)),
         # The ENVIRONMENT count below the line, not just the family count. main.tex used to state
         # both by hand and had both wrong at once -- "one of the six below the line, and those six
         # hold 98 environments" against a table showing five families and 82 envs. That sentence is
         # the exact failure this file's header says the macros exist to prevent, so it gets a macro
         # rather than a correction.
         macro("seNenvBelow", str(sum(f["n"] for f in fams[n_above:]))),
         macro("seNenvAbove", str(sum(f["n"] for f in fams[:n_above]))),
         macro("seNabove", str(len(ratios) - stats["vs_stored"]["below_1"])),
         macro("seNbelow", str(stats["vs_stored"]["below_1"])),
         # The SAME quantities against the fastest path measured, so the prose can state both and a
         # reader never has to reconstruct the weaker comparison from an appendix. These were the
         # numbers a referee recomputed by hand to show the headline had been divided by the fastest
         # CPU path while a faster GPU path sat two pages later in the same document.
         macro("seGpuMedian", num(st.median(gratios), 2)),
         macro("seGpuGeomean", num(st.geometric_mean(gratios), 2)),
         macro("seNaboveGpu", str(sum(1 for r in gratios if r >= 1.0))),
         macro("seNbelowGpu", str(sum(1 for r in gratios if r < 1.0))),
         macro("seNratedGpu", str(len(gratios))),
         macro("seGpuFps", f"{st.median([v[STORED_GPU] for v in usable.values() if v.get(STORED_GPU)]):,.0f}"),
         macro("seNrated", str(len(ratios))),
         macro("seStoredMedian", num(st.median(ratios), 1)),
         macro("seStoredGeomean", num(st.geometric_mean(ratios), 2)),
         macro("seStoredMax", num(max(ratios))),
         macro("seStoredMin", num(min(ratios), 3)),
         macro("seDefaultMedian", num(st.median(vsdef))),
         macro("seDefaultGeomean", num(st.geometric_mean(vsdef))),
         macro("seDefaultMax", num(max(vsdef))),
         macro("seStorageRatio", num(tot_raw / tot_state)),
         macro("seStorageMedian", num(per_env_med)),
         macro("seStorageMax", num(per_env_raw[-1])),
         macro("seStorageStateB", num(tot_state)),
         macro("seStorageRawB", num(tot_raw)),
         macro("seJpegRatio", num(tot_jpeg / tot_state)),
         macro("seNhostRows", str(len(cpu_fracs))),
         macro("seHostMedian", num(st.median(cpu_fracs) * 100) if cpu_fracs else "---"),
         macro("seHostMax", num(max(cpu_fracs) * 100) if cpu_fracs else "---"),
         macro("seNhostAbove", str(stats["host_bound"]["at_or_above_1"])),
         macro("seNssTwo", str(len(ss_two))),
         macro("seStoredMedianFps", num(st.median([v[STORED] for v in usable.values()
                                                   if v.get(STORED)]))),
         # How much kinder the one-core bar would have been. Stated so that choosing the eight-worker
         # column reads as a choice with a size, rather than as the only column that existed.
         macro("seStoredWorkerFactor",
               num(st.median([v[STORED] for v in usable.values() if v.get(STORED)])
                   / st.median([v["st_local"] for v in usable.values() if v.get("st_local")]), 1)),
         macro("seOursMedianFps", num(st.median([v["gpu"] for v in usable.values()]))),
         macro("seDefaultMedianFps", num(st.median([v["cpu"] for v in usable.values()
                                                    if v.get("cpu")])))]
    (OUT / "macros.tex").write_text("\n".join(M + extra) + "\n")

    print(json.dumps({k: v for k, v in stats.items() if k != "families"}, indent=2))
    print(f"\n{'family':22s} {'n':>3s} {'xdefault':>9s} {'xstored':>8s}   range")
    for f in fams:
        print(f"{f['family']:22s} {f['n']:3d} {f['vs_default']:9,.0f} {f['vs_stored']:8.2f}   "
              f"{f['min']:.2f}-{f['max']:,.2f}")
    print(f"\n{n_above} of {len(fams)} families clear the storage baseline; "
          f"{len(ratios) - stats['vs_stored']['below_1']} of {len(ratios)} environments do.")


if __name__ == "__main__":
    main()

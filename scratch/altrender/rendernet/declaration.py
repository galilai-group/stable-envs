"""Per cell, the numbers a sealed checkpoint has to carry, in the shape ``render/rendernet.seal`` takes.

``stable_envs/render/rendernet.py`` will not seal a corrector that does not beat the render it
replaces, and it decides that from three blocks that have to be MEASURED and travel with the weights:

* ``visual_fidelity`` -- the declared draw (``sample_states`` at four seeds, worst of the four) at the
  DEFAULT appearance, plus the plain class's numbers on the same draws under ``plain_*`` so the two
  are never compared across state sets.
* ``variation_fidelity`` -- the same corrector under the env's OWN declared variation space. RoboCasa
  §3 is why this block exists: a corrector fitted at one camera pose scores 3.25 at the default and
  9.06 under +-8 degrees of declared pan, which is worse than not correcting and worse than the render
  it replaces, and only this block sees it.
* ``baseline`` -- the shipped render and the bilinear resize on the held-out split, so a reader can
  see what the corrector is a departure FROM without opening another file.

This does not seal anything. Sealing writes into ``~/.cache/stable-envs/rendernet`` and is the
library's job; this is the measurement, emitted as JSON so it can be handed over.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

FIVE = ("mean_abs", "within8_frac", "ssim", "interior_mean_abs", "interior_exact_frac")


def _split(res, name):
    for r in res["splits"]:
        if r["split"] == name:
            return r
    return None


def block(path_default: str, path_sty: str | None) -> dict:
    res = json.load(open(path_default))
    ho = _split(res, "heldout")
    dc = _split(res, "decl")
    out = {"net": res["net"], "env": res["env"],
           "render": {"render_size": res["render_size"], "tex_lod": res["tex_lod"],
                      "backface_cull": False, "out_size": 256},
           "throughput": {"shipped_256_fps": res["throughput"]["shipped_256_fps"],
                          "render_only_fps": res["throughput"]["render_only_fps"],
                          "corrected_fps": res["throughput"]["render_plus_corrector"]["fps"]},
           "params": res.get("params"), "ckpt": res["ckpt"]}
    if dc and "worst_of_draws" in dc:
        w = dc["worst_of_draws"]
        vf = dict(w["corrector"])
        vf.update({f"plain_{k}": w["shipped_256"][k] for k in FIVE})
        vf["measured_at"] = ("256 px, 64 states x 4 independent draws from sample_states, "
                             "worst draw, one NVIDIA H200")
        out["visual_fidelity"] = vf
        out["jpeg_q95_of_official"] = w.get("jpeg_q95")
    if ho:
        v = ho["variants"]
        out["heldout"] = {k: {m: v[k][m] for m in FIVE + ("max_abs", "p999", "frac_over_64",
                                                          "mean_abs_dynamic", "mean_abs_static")}
                          for k in v}
        out["baseline"] = {"shipped_256": {m: v["shipped_256"][m] for m in FIVE},
                           "bilinear": {m: v["bilinear"][m] for m in FIVE}}
    if path_sty and os.path.exists(path_sty):
        rs = json.load(open(path_sty))
        hs = _split(rs, "heldout")
        if hs:
            v = hs["variants"]
            out["variation_fidelity"] = {
                **{k: v["corrector"][k] for k in FIVE},
                **{f"plain_{k}": v["shipped_256"][k] for k in FIVE},
                "measured_at": ("held-out states under the env's own declared variation space, "
                                "256 px, blocks of 64"),
                "style_leaves": rs.get("style_leaves")}
    out["blank_frame_control"] = res.get("hallucination", {}).get("blank_input", {}).get("grey")
    return out


def gates(b: dict) -> dict:
    """The three the loader's seal enforces, computed here so a hand-off carries the verdict too."""
    g = {}
    for name, key in (("declared", "visual_fidelity"), ("variation", "variation_fidelity")):
        blk = b.get(key)
        if not blk:
            g[name] = None
            continue
        g[name] = bool(blk["mean_abs"] < blk["plain_mean_abs"]
                       and blk["within8_frac"] > blk["plain_within8_frac"]
                       and blk["ssim"] > blk["plain_ssim"])
    ho = b.get("heldout")
    if ho and "shipped_256" in ho:
        c, s = ho["corrector"], ho["shipped_256"]
        g["heldout_all_five"] = bool(c["mean_abs"] < s["mean_abs"]
                                     and c["interior_mean_abs"] < s["interior_mean_abs"]
                                     and c["within8_frac"] > s["within8_frac"]
                                     and c["ssim"] > s["ssim"]
                                     and c["interior_exact_frac"] > s["interior_exact_frac"])
    g["faster"] = b["throughput"]["corrected_fps"] > b["throughput"]["shipped_256_fps"]
    g["ship"] = all(v for v in (g.get("declared"), g.get("variation"),
                                g.get("heldout_all_five"), g.get("faster")) if v is not None) \
        and g.get("declared") is not None and g.get("variation") is not None
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "eval_*_on_default.json"))):
        stem = os.path.basename(p)[len("eval_"):-len("_on_default.json")]
        sty = os.path.join(a.dir, f"eval_{stem}_on_sty.json")
        try:
            b = block(p, sty)
        except Exception as e:                                        # noqa: BLE001
            print(f"skip {p}: {type(e).__name__}: {e}")
            continue
        b["gates"] = gates(b)
        rows[stem] = b
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1, default=float)
    print(f"wrote {a.out}  {len(rows)} checkpoints\n")
    hdr = (f"{'checkpoint':60s} {'decl mean':>9s} {'plain':>7s} {'var mean':>9s} {'plain':>7s} "
           f"{'f/s':>8s} {'shipped':>8s}  gates")
    print(hdr)
    for k, b in sorted(rows.items()):
        vf, vr, g = b.get("visual_fidelity"), b.get("variation_fidelity"), b["gates"]
        f = lambda d, key: (f"{d[key]:9.2f}" if d else f"{'--':>9s}")
        print(f"{k:60s} {f(vf,'mean_abs')} {f(vf,'plain_mean_abs')[2:]} "
              f"{f(vr,'mean_abs')} {f(vr,'plain_mean_abs')[2:]} "
              f"{b['throughput']['corrected_fps']:8.0f} {b['throughput']['shipped_256_fps']:8.0f}  "
              f"decl={g.get('declared')} var={g.get('variation')} "
              f"held5={g.get('heldout_all_five')} fast={g.get('faster')} "
              f"{'SHIP' if g['ship'] else 'NO'}")


if __name__ == "__main__":
    main()

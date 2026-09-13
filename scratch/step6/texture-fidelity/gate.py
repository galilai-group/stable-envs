"""Apply the FROZEN Step-4 bounds (examples/results/step4/certification.json) to a replay shard.

No bound is written here: every threshold is read from the certification artifact's own
``checks[...]["bound"]`` record, so a replay under another MuJoCo is judged by exactly the bar Step 4
applied. Prints one row per config per check for each arm and writes a JSON table.

    python scratch/step6/texture-fidelity/gate.py --arm 311=<dmc.json> --arm 312=<dmc.json> --out gate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CERT = ROOT / "examples" / "results" / "step4" / "certification.json"
CONFIGS = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidStand")


def judge_visual(bound: dict, obs: dict) -> list[str]:
    v = []
    if obs["mean_abs"] > bound["mean_abs_max"]:
        v.append(f"mean_abs {obs['mean_abs']:.3f} > {bound['mean_abs_max']}")
    if obs["within8_frac"] < bound["within8_min"]:
        v.append(f"within8_frac {obs['within8_frac']:.4f} < {bound['within8_min']}")
    if obs["ssim"] < bound["ssim_min"]:
        v.append(f"ssim {obs['ssim']:.4f} < {bound['ssim_min']}")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="tag=path/to/dmc.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cert = json.loads(CERT.read_text())["configs"]
    table = {"frozen_step4": {}, "arms": {}}
    for cfg in CONFIGS:
        ch = cert[cfg]["checks"]
        table["frozen_step4"][cfg] = {
            "gate_a": cert[cfg]["gate_a"], "failures": cert[cfg]["failures"],
            "visual": ch["check_visual_fidelity"], "resolution": ch["check_resolution_consistency"],
            "digest_official": None, "digest_ours": None}
    for arm in a.arm:
        tag, path = arm.split("=", 1)
        doc = json.loads(Path(path).read_text())
        prov = doc["provenance"]
        rows = {"provenance": {k: prov.get(k) for k in ("gpu", "driver", "host", "slurm_job_id", "torch", "mujoco_gl")},
                "packages": prov.get("reference_packages"), "configs": {}}
        for cfg in CONFIGS:
            c = doc["configs"].get(cfg)
            if not c or c.get("status") != "measured":
                rows["configs"][cfg] = {"status": (c or {}).get("status"), "why": (c or {}).get("why")}
                continue
            fs = c["family_specific"]
            vb = cert[cfg]["checks"]["check_visual_fidelity"]["bound"]
            vo = fs["check_visual_fidelity"].get("evidence")
            rb = cert[cfg]["checks"]["check_resolution_consistency"]["bound"]      # ">= x - 0.005"
            floor = eval(rb.split(">=")[1])                                         # numeric literal
            ro = fs["check_resolution_consistency"].get("evidence")
            vv = judge_visual(vb, vo) if vo else ["not measured: " + str(fs["check_visual_fidelity"].get("why"))]
            rv = ([] if ro and ro["exact_frac"] >= floor else
                  [f"exact_frac {ro['exact_frac']:.4f} < {rb}" if ro else "not measured"])
            rows["configs"][cfg] = {
                "check_visual_fidelity": {"observed": {k: vo.get(k) for k in ("mean_abs", "within8_frac", "ssim", "exact_frac", "interior_exact_frac")} if vo else None,
                                          "bound": {k: vb[k] for k in ("mean_abs_max", "within8_min", "ssim_min")},
                                          "declared": {k: vb[k] for k in ("mean_abs", "within8_frac", "ssim")},
                                          "status": "pass" if not vv else "fail", "violations": vv},
                "check_resolution_consistency": {"observed": ro, "bound": rb,
                                                 "status": "pass" if not rv else "fail", "violations": rv},
                "gate_a_on_these_two_checks": "PASS" if not vv and not rv else "FAIL",
                "oracle": {k: c["oracle"].get(k) for k in ("digest_ours", "digest_official", "mean_pixel", "distinct_colors")},
                "reference_stable": c["oracle"].get("reference_stable", {}).get("evidence"),
            }
        table["arms"][tag] = rows
    Path(a.out).write_text(json.dumps(table, indent=1, sort_keys=True))
    # a compact console table
    print(f"{'config':18s} {'arm':6s} {'visual mean/w8/ssim':28s} {'bound':22s} {'res exact':10s} {'bound':16s} verdict")
    for cfg in CONFIGS:
        f = table["frozen_step4"][cfg]
        vo, ro = f["visual"]["observed"], f["resolution"]["observed"]
        print(f"{cfg:18s} {'step4':6s} {vo['mean_abs']:.2f}/{vo['within8_frac']:.3f}/{vo['ssim']:.3f}{'':10s} "
              f"{f['visual']['bound']['mean_abs_max']}/{f['visual']['bound']['within8_min']}/{f['visual']['bound']['ssim_min']:<8} "
              f"{ro['exact_frac']:.4f}     {f['resolution']['bound']:16s} {f['gate_a']} {f['failures']}")
        for tag, rows in table["arms"].items():
            r = rows["configs"][cfg]
            if "check_visual_fidelity" not in r:
                print(f"{'':18s} {tag:6s} {r}")
                continue
            vo = r["check_visual_fidelity"]["observed"]; ro = r["check_resolution_consistency"]["observed"]
            print(f"{'':18s} {tag:6s} {vo['mean_abs']:.2f}/{vo['within8_frac']:.3f}/{vo['ssim']:.3f}{'':10s} {'(same)':22s} "
                  f"{ro['exact_frac']:.4f}     {'(same)':16s} {r['gate_a_on_these_two_checks']} "
                  f"{r['check_visual_fidelity']['violations'] + r['check_resolution_consistency']['violations']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""M5: write PFRocketLanding.visual_fidelity from the certified tier JSON, by the DMC family's convention.

    python scratch/step7/rocket-landing/fill_declaration.py examples/results/step7/rocket-landing/tier/<host>.json

Measured values = the worst of the four 64-state draws over the REACHABLE rows (``visual_fidelity.worst``),
the spread, the under-variation row and the reference-unreachable share are copied; the thresholds are
the family's margin convention -- ceiling ~1.3x the worst draw, floors ~0.93-0.95x of it, rounded to
round numbers -- and are printed for the author to paste, never tuned to the measurement. The script
prints the block; it does not edit the module (the edit is reviewed by hand and by the unit test).
"""
import json, sys


def main():
    R = json.load(open(sys.argv[1]))
    vf = R["visual_fidelity"]
    w, sp, var = vf["worst"], vf["spread"], R["variation_fidelity"]
    draws = vf["draws"]
    ro = [d["reachable_only"] for d in draws]
    prov = R["provenance"]
    ceil = round(w["mean_abs"] * 1.3, 0)
    fl8 = round(w["within8_frac"] * 0.95, 2)
    fls = round(w["ssim"] * 0.94, 2)
    block = {
        "mean_abs": round(w["mean_abs"], 2), "within8_frac": round(w["within8_frac"], 3), "ssim": round(w["ssim"], 3),
        "exact_frac": round(min(d["exact_frac"] for d in ro), 3),
        "interior_exact_frac": round(min(d["interior_exact_frac"] for d in ro), 3),
        "draw_spread": {k: [round(v[0], 3), round(v[1], 3)] for k, v in sp.items()},
        "under_variation": {k: round(var[k], 3) for k in ("mean_abs", "within8_frac", "ssim", "exact_frac")},
        "reference_unreachable_share": 1.0 / 960.0,
        "reference_unreachable_rows_in_draws": w["unreachable_rows_total"],
        "mean_abs_max": ceil, "within8_min": fl8, "ssim_min": fls,
        "measured_at": (f"480 px, supersample 1, default appearance, worst of 4 independent 64-state draws of "
                        f"sample_states (default_rng(1000..1003)) over the rows the reference bridge reaches, "
                        f"against PyFlyt 0.29.0 / pybullet 3.2.7 TinyRenderer frames of swm/PFRocketLanding-v0 at the "
                        f"lagged pose; {R['gpu']} {R['host'].split('.')[0]}, SLURM job {R['slurm_job']}, tree "
                        f"{prov['sha'][:7]}, torch {R['versions']['torch']}, nvdiffrast {R['versions']['nvdiffrast']} "
                        f"(scratch/step7/rocket-landing/measure_tier.py, examples/results/step7/rocket-landing/tier/)"),
        "reference_versions": {"PyFlyt": "0.29.0", "pybullet": "3.2.7"},
        "measured_with": {"nvdiffrast": R["versions"]["nvdiffrast"], "torch": R["versions"]["torch"]},
    }
    print(json.dumps(block, indent=4))


if __name__ == "__main__":
    main()

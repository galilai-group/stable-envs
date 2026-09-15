"""M4 probe: localise the batch-size dependence job 6406061 measured (row 0 alone vs row 0 of 64: max 255).

check_batch_invariance (B=16 vs 1) and chunked render_states (chunk 1 vs 64 on 8 states) agree, so the
dependence appears somewhere above 16 rows. This renders the SAME 64 states (default_rng(11), as the
gate) at every prefix size, compares row 0 against the single render and against the full-64 render,
reports which rows of the full render differ from their single renders, the renderer's chunk, whether
the 64-render is deterministic, and where the differing pixels sit. MujocoHopper is the control for
nvdiff's unchanged base path. Diagnostic only; no code is changed.
"""
import json, os, sys, platform
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
import stable_envs as se
OUT = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "tier", "probe_batch")


def cmp(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max": int(d.max()), "px": int((d.max(-1) > 0).sum())}


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "slurm_job": os.environ.get("SLURM_JOB_ID"),
         "sha": open(os.path.join(ROOT, ".git_sha")).read().strip() if os.path.exists(os.path.join(ROOT, ".git_sha")) else "unknown"}
    for name in ("PFRocketLanding", "MujocoHopper"):
        if name not in se.REGISTRY:
            R[name] = "not registered"; continue
        env = se.make(name, batch=16, device="cuda")
        st = env.sample_states(64, rng=np.random.default_rng(11))
        full = env.render(st).cpu().numpy()
        full2 = env.render(st).cpu().numpy()
        singles = np.stack([env.render(st[i:i + 1]).cpu().numpy()[0] for i in range(64)])
        r = {"deterministic_at_64": bool(np.array_equal(full, full2)),
             "rows_where_full64_differs_from_single": [i for i in range(64) if cmp(full[i], singles[i])["max"] > 0],
             "row0_vs_prefix": {}}
        if name == "PFRocketLanding":
            c = env._nvd(env.size, env.ss)
            r["chunk"] = int(c.chunk)
            r["step_at_64"] = int(-(-64 // max(1, -(-64 // c.chunk))))
        for b in (1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 32, 33, 40, 48, 63, 64):
            sub = env.render(st[:b]).cpu().numpy()
            r["row0_vs_prefix"][str(b)] = {"vs_single": cmp(sub[0], singles[0]), "vs_full64": cmp(sub[0], full[0])}
        # the offending pixels of row 0 (single vs full-64): where are they?
        d = np.abs(full[0].astype(np.int16) - singles[0].astype(np.int16)).max(-1)
        ys, xs = np.nonzero(d)
        r["row0_diff_px"] = int(len(ys))
        if len(ys):
            r["row0_diff_bbox"] = [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
            r["row0_diff_row_hist"] = np.histogram(ys, bins=8, range=(0, full.shape[1]))[0].tolist()
        if name == "PFRocketLanding":
            from PIL import Image
            Image.fromarray(singles[0]).save(os.path.join(OUT, "row0_single.png"))
            Image.fromarray(full[0]).save(os.path.join(OUT, "row0_in64.png"))
            Image.fromarray((d > 0).astype(np.uint8) * 255).save(os.path.join(OUT, "row0_diffmask.png"))
            # is it the chunk loop? render the first 32 alone and the last 32 alone, compare to full
            for lo, hi in ((0, 32), (32, 64)):
                part = env.render(st[lo:hi]).cpu().numpy()
                r[f"rows{lo}_{hi}_alone_vs_full64"] = cmp(part, full[lo:hi])
            # a 1024-batch, as the pytest check: row 0 vs single
            st_big = env.sample_states(1024, rng=np.random.default_rng(11))
            big = env.render(st_big).cpu().numpy()
            r["row0_of_1024_vs_single"] = cmp(big[0], env.render(st_big[0:1]).cpu().numpy()[0])
            r["rows_of_1024_differing_from_single_first64"] = [i for i in range(64) if cmp(big[i], env.render(st_big[i:i + 1]).cpu().numpy()[0])["max"] > 0]
        R[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "row0_vs_prefix"}), flush=True)
        print(name, "row0_vs_prefix", json.dumps(r["row0_vs_prefix"]), flush=True)
    with open(os.path.join(OUT, "probe_batch_size.json"), "w") as fh:
        json.dump(R, fh, indent=1)


if __name__ == "__main__":
    main()

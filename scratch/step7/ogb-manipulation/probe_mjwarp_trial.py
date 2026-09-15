"""Step 7 / W5 phase 0 -- put the OGBench manipulation scene through the EXISTING MJWarp backend.

No new renderer and no env class: this builds the swm-registered `MjModel` and hands it to
`stable_envs.render.mjwarp.BatchRenderer` exactly as `stable_envs/envs/fetch.py` does, at the
registered resolution and the registered `front_pixels` camera.  What it answers:

* **Does it build at all** at 193k visible triangles (6.1x Fetch's 31,720), and what does the BVH
  cost in device memory?
* **BATCH_SCALES.** Step 6 closed `fetch-gpu` because `fps(4096)/fps(1)` was 7.6-11.7x against a
  required 20x AT THE FROZEN 480 px, and recorded the same scene PASSING at <=240 px (26-99x).
  These envs render at 224 px and 200 px, so the Fetch verdict does not transfer; measure it.
* **A first look at fidelity**, so the tier conversation starts from pixels: MJWarp's frame against
  MuJoCo's own EGL frame at the same state, with `gl_rounding` on (the Fetch correction) -- whole
  frame, interior, and the share owned by the one translucent goal marker MJWarp cannot draw.

Nothing is written outside the given --out.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, statistics, sys, time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256, 1024])
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import gymnasium as gym
    import mujoco
    import torch
    import stable_worldmodel  # noqa: F401
    from stable_envs.render import mjwarp as MJ
    from stable_envs.testing.conformance import interior_mask

    for eid in a.envs:
        tag = eid.split("/")[-1].replace("-v0", "")
        rec = {"env_id": eid, "host": socket.gethostname(), "platform": platform.platform(),
               "mujoco": mujoco.__version__, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "mujoco_gl": os.environ.get("MUJOCO_GL")}
        try:
            import importlib.metadata as md
            rec["mujoco_warp"] = md.version("mujoco-warp")
            rec["warp_lang"] = md.version("warp-lang")
        except Exception:
            pass

        env = gym.make(eid); u = env.unwrapped
        env.reset(seed=0); m, d = u.model, u.data
        res = int(u._render_height)
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
        rec["res"], rec["camera_id"], rec["nq"] = res, int(cam), int(m.nq)

        # a pool of real states + the official EGL frames for them
        pool, offs = [], []
        nb = getattr(u, "_num_buttons", 0)
        kw = {f"button_state_{i}": 0 for i in range(nb)} if nb else {}
        for s in range(a.states):
            env.reset(seed=500 + s); m, d = u.model, u.data
            for _ in range(int(np.random.default_rng(s).integers(0, 20))):
                env.step(env.action_space.sample())
            pool.append(d.qpos.copy()); offs.append(u.render().copy())
        qpos = np.stack(pool).astype(np.float32)
        offs = np.stack(offs)
        np.savez_compressed(out / f"official_{tag}.npz", qpos=qpos, frames=offs)

        rec["texture_gaps"] = MJ.texture_gaps(m)
        ea = MJ.effective_alpha(m, "geom")
        gg = [0, 1, 2]
        rec["partial_alpha_visible_geoms"] = [
            [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i), float(ea[i])]
            for i in range(m.ngeom) if int(m.geom_group[i]) in gg and 0 < ea[i] < 1]

        # ---- fidelity at B=len(pool) ---------------------------------------
        try:
            with MJ.gl_rounding():
                br = MJ.BatchRenderer(m, batch=len(pool), res=res, camera=int(cam))
                fr = br.render(torch.as_tensor(qpos, device="cuda")).cpu().numpy()
            rec["built"] = True
            rec["vram_mib_after_build"] = int(torch.cuda.max_memory_allocated() / 2**20)
            di = np.abs(fr.astype(np.int16) - offs.astype(np.int16)).max(-1)
            im = interior_mask(offs)
            rec["fidelity"] = {
                "all_exact_frac": float((di == 0).mean()),
                "all_within8_frac": float((di <= 8).mean()),
                "mean_abs": float(di.mean()),
                "interior_frac": float(im.mean()),
                "interior_exact_frac": float((di[:, :-1, :-1][im] == 0).mean()) if im.any() else None,
                "interior_mean_abs": float(di[:, :-1, :-1][im].mean()) if im.any() else None,
            }
            np.savez_compressed(out / f"mjwarp_{tag}.npz", mjwarp=fr[:4], official=offs[:4])
            del br
            torch.cuda.empty_cache()
        except Exception as e:
            rec["built"] = False
            rec["build_error"] = f"{type(e).__name__}: {e}"[:1200]

        # ---- BATCH_SCALES ---------------------------------------------------
        rec["batch_rows"] = []
        if rec.get("built"):
            for B in a.batches:
                try:
                    torch.cuda.reset_peak_memory_stats()
                    with MJ.gl_rounding():
                        br = MJ.BatchRenderer(m, batch=B, res=res, camera=int(cam))
                    q = torch.as_tensor(
                        np.repeat(qpos, int(np.ceil(B / len(qpos))), axis=0)[:B], device="cuda")
                    for _ in range(3):
                        br.render(q)
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(a.repeat):
                        t0 = time.perf_counter()
                        br.render(q)
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                    med = statistics.median(ts)
                    rec["batch_rows"].append({
                        "batch": B, "ms": med * 1e3, "fps": B / med,
                        "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 2**20)})
                    print(f"{eid} B={B}: {med*1e3:.2f} ms -> {B/med:.0f} f/s", flush=True)
                    del br, q
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec["batch_rows"].append({"batch": B, "error": f"{type(e).__name__}: {e}"[:400]})
                    print(f"{eid} B={B}: FAILED {type(e).__name__}", flush=True)
                    torch.cuda.empty_cache()
            ok = [r for r in rec["batch_rows"] if "fps" in r]
            if len(ok) >= 2:
                rec["batch_scales_max_over_1"] = ok[-1]["fps"] / ok[0]["fps"]
                rec["batch_scales_note"] = (
                    f"fps({ok[-1]['batch']})/fps({ok[0]['batch']}); Step 2 asks for 20x at the "
                    f"registered resolution. Fetch measured 7.6-11.7x at 480 px and 26-99x at <=240 px.")

        (out / f"mjwarp_trial_{tag}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(json.dumps({k: v for k, v in rec.items() if k != "batch_rows"}, indent=1)[:2000], flush=True)
        env.close()


if __name__ == "__main__":
    main()

"""Step 7 / W5 phase 0 -- how much of an OGB frame is even MEASURABLE as parity?

Two numbers that decide what a fidelity tier can mean here:

* **Interior fraction.** `stable_envs.testing.conformance.interior_mask` keeps pixels
  whose 3x3 neighbourhood is one flat colour -- the pixels where a disagreement is a real
  error rather than a sub-pixel coverage rule.  With 62 % of the visible triangles under
  one pixel across and the official renderer running 4x MSAA, most of an OGB frame may be
  edge.  If the interior fraction is small, an `interior_exact_frac` of 1.0 certifies very
  little and the honest tier is `visual`, not `parity`.
* **The translucent goal marker.** `object_target_*` renders at alpha 0.2, and
  `stable_envs/envs/fetch.py` records that MJWarp never reads alpha.  Measure the pixel
  share the marker owns, so the cost of not drawing it (or of adding alpha) is priced.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np


def main():
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from stable_envs.testing.conformance import interior_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    ap.add_argument("--states", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for eid in a.envs:
        env = gym.make(eid); u = env.unwrapped
        env.reset(seed=9); m, d = u.model, u.data
        h, w = u._render_height, u._render_width
        rec = {"env_id": eid, "h": h, "w": w, "offsamples_msaa": int(m.vis.quality.offsamples)}

        frames, states = [], []
        for s in range(a.states):
            env.reset(seed=400 + s); m, d = u.model, u.data
            for _ in range(int(np.random.default_rng(s).integers(0, 20))):
                env.step(env.action_space.sample())
            frames.append(u.render().copy())
            states.append(np.concatenate([d.qpos, d.qvel, d.mocap_pos.ravel(), d.mocap_quat.ravel()]))
        F = np.stack(frames)
        im = interior_mask(F)
        rec["interior_frac_mean"] = float(im.mean())
        rec["interior_frac_per_state"] = [float(x.mean()) for x in im]

        # MSAA: how much does 4x supersampling cost against 1 sample/pixel?
        saved = int(m.vis.quality.offsamples)
        m.vis.quality.offsamples = 0
        u._renderer.close(); u._initialize_renderer()
        f1 = []
        for s in range(a.states):
            pass
        env.reset(seed=400); m, d = u.model, u.data
        m.vis.quality.offsamples = 0
        u._renderer.close(); u._initialize_renderer()
        f_noaa = u.render().copy()
        m.vis.quality.offsamples = saved
        u._renderer.close(); u._initialize_renderer()
        f_aa = u.render().copy()
        rec["msaa"] = {
            "offsamples": saved,
            "frac_pixels_differing_1x_vs_4x": float((f_noaa != f_aa).any(-1).mean()),
            "max_abs_diff": int(np.abs(f_noaa.astype(np.int16) - f_aa.astype(np.int16)).max()),
            "mean_abs_diff": float(np.abs(f_noaa.astype(np.int16) - f_aa.astype(np.int16)).mean()),
        }
        imn = interior_mask(np.stack([f_aa]))[0]
        di = np.abs(f_noaa.astype(np.int16) - f_aa.astype(np.int16)).max(-1)[:-1, :-1]
        rec["msaa"]["interior_exact_frac_1x_vs_4x"] = float((di[imn] == 0).mean()) if imn.any() else None
        rec["msaa"]["interior_frac"] = float(imn.mean())

        # translucent goal marker share
        gg = [i for i, v in enumerate(u._scene_option.geomgroup) if v]
        tg = [i for i in range(m.ngeom)
              if int(m.geom_group[i]) in gg and 0.0 < float(m.geom_rgba[i][3]) < 1.0]
        rec["translucent_geom_ids"] = [
            {"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
             "alpha": float(m.geom_rgba[i][3])} for i in tg]
        if tg:
            env.reset(seed=400); m, d = u.model, u.data
            tg = [i for i in range(m.ngeom)
                  if int(m.geom_group[i]) in gg and 0.0 < float(m.geom_rgba[i][3]) < 1.0]
            f_with = u.render().copy()
            sv = m.geom_rgba.copy()
            for i in tg:
                m.geom_rgba[i, 3] = 0.0
            f_without = u.render().copy()
            m.geom_rgba[:] = sv
            rec["translucent_px_share"] = float((f_with != f_without).any(-1).mean())
            rec["translucent_mean_abs"] = float(
                np.abs(f_with.astype(np.int16) - f_without.astype(np.int16)).mean())
            # and if drawn OPAQUE instead (what an alpha-blind ray-caster does)
            for i in tg:
                m.geom_rgba[i, 3] = 1.0
            f_opaque = u.render().copy()
            m.geom_rgba[:] = sv
            rec["translucent_as_opaque_px_share"] = float((f_with != f_opaque).any(-1).mean())
            rec["translucent_as_opaque_mean_abs"] = float(
                np.abs(f_with.astype(np.int16) - f_opaque.astype(np.int16)).mean())

        # bytes/state vs bytes/frame with the full render-relevant state
        rec["render_state_floats"] = int(m.nq + m.nv + 7 * m.nmocap) + int(getattr(u, "_num_buttons", 0))
        rec["render_state_bytes_f64"] = rec["render_state_floats"] * 8
        rec["frame_bytes"] = int(h * w * 3)
        rec["frame_over_state"] = rec["frame_bytes"] / rec["render_state_bytes_f64"]

        tag = eid.split("/")[-1].replace("-v0", "")
        np.savez_compressed(out / f"interior_{tag}.npz", frames=F[:2], states=np.stack(states)[:2])
        (out / f"interior_{tag}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{eid}: interior_frac {rec['interior_frac_mean']*100:.2f}%  "
              f"MSAA 1x-vs-4x differs {rec['msaa']['frac_pixels_differing_1x_vs_4x']*100:.1f}% "
              f"(mean {rec['msaa']['mean_abs_diff']:.2f}, interior_exact {rec['msaa']['interior_exact_frac_1x_vs_4x']})  "
              f"translucent share {rec.get('translucent_px_share', 0)*100:.2f}% "
              f"(as-opaque {rec.get('translucent_as_opaque_px_share', 0)*100:.2f}%)", flush=True)
        env.close()


if __name__ == "__main__":
    main()

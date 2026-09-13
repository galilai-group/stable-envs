"""Step 6A: WHERE does a trackcom DM Control frame disagree with the official one, per surface?

Runs once per interpreter (one MuJoCo/mujoco_warp version = one arm) and writes, per config:

  * whole-frame agreement in the three quantities Gate A asserts (``mean_abs``, ``within8_frac``,
    ``ssim``), computed by the SAME functions ``check_visual_fidelity`` uses, on two state sets:
    the frozen Step-4 corpus, and the draw ``check_visual_fidelity`` itself makes (``env.rng``
    re-seeded to ``fidelity_step4.CHECK_SEED``), so the number the certifier judged is localised too;
  * the same agreement split into skybox / floor / bodies off MuJoCo's own segmentation buffer,
    rendered by the REFERENCE at the identical states, plus each region's share of the residual;
  * ``check_resolution_consistency`` on the frozen corpus (our renderer against itself at 2x), and
    the same 2x residual split per region -- that check never touches the reference, so its failure
    can only be our renderer or the texture bytes it samples;
  * the compiled texture bytes (sha256 of ``tex_data`` per texture) and the material table, so
    "same declaration" is a digest rather than a belief;
  * the exact versions: mujoco, dm_control, mujoco_warp, warp, torch, driver, card, node, job.

Frames are saved (``frames/<config>.npz``) so ``cross.py`` can compare the two arms pixel for pixel.

    python scratch/step6/texture-fidelity/localize.py --configs DMCCheetahRun,... --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

CHECK_SEED = 4041                 # fidelity_step4.CHECK_SEED, restated so this file is standalone


def _ver(mod: str) -> str | None:
    try:
        return md.version(mod)
    except Exception:
        try:
            return getattr(__import__(mod), "__version__", "present")
        except Exception:
            return None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def provenance() -> dict:
    import torch
    p = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
         "python": platform.python_version(), "torch": torch.__version__,
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
         "mujoco_gl": os.environ.get("MUJOCO_GL"), "venv": sys.prefix,
         "packages": {m: _ver(m) for m in ("mujoco", "dm_control", "mujoco_warp", "mujoco-warp", "warp-lang", "numpy", "scipy")}}
    if torch.cuda.is_available():
        p["gpu"] = torch.cuda.get_device_name(0)
        p["driver"] = _run(["bash", "-c", "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"])
    return p


def texture_table(m) -> dict:
    import mujoco
    tex, mats = [], []
    for t in range(m.ntex):
        adr, w, h = int(m.tex_adr[t]), int(m.tex_width[t]), int(m.tex_height[t])
        nch = int(m.tex_nchannel[t]) if hasattr(m, "tex_nchannel") else 3
        data = np.asarray(m.tex_data[adr:adr + w * h * nch])
        tex.append({"id": t, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t),
                    "type": int(m.tex_type[t]), "w": w, "h": h, "nch": nch,
                    "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                    "mean": round(float(data.mean()), 4),
                    "distinct": int(len(np.unique(data.reshape(-1, nch), axis=0)))})
    for i in range(m.nmat):
        mats.append({"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i),
                     "texid": np.asarray(m.mat_texid[i]).reshape(-1).tolist(),
                     "texrepeat": m.mat_texrepeat[i].tolist(), "texuniform": int(m.mat_texuniform[i]),
                     "rgba": np.round(m.mat_rgba[i], 4).tolist(), "reflectance": float(m.mat_reflectance[i]),
                     "shininess": float(m.mat_shininess[i]), "specular": float(m.mat_specular[i])})
    return {"ntex": int(m.ntex), "nmat": int(m.nmat), "textures": tex, "materials": mats,
            "offsamples": int(m.vis.quality.offsamples), "shadowsize": int(m.vis.quality.shadowsize),
            "geom_matid": np.asarray(m.geom_matid).reshape(-1).tolist()}


def segmentation(cls, ref, states) -> np.ndarray:
    """``(N,H,W,2)`` int32 ``[objid, objtype]`` from the reference at exactly the states it renders.

    The state-setting is ``reference_render``'s, line for line, so the mask belongs to the frame.
    """
    p = ref.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    size = int(getattr(ref, "_se_size", 0) or cls.default_size)
    slots, nq = cls._slots(), int(m.nq)
    out = []
    for st in np.asarray(states, np.float64):
        for (arr, k), v in zip(slots, st[nq:]):
            getattr(m, arr).reshape(-1)[k] = v
        with p.reset_context():
            p.data.qpos[:] = st[:nq]
            p.data.qvel[:] = 0
        out.append(p.render(height=size, width=size, camera_id=cls.CAMERA, segmentation=True).copy())
    return np.stack(out).astype(np.int32)


def masks(m, seg) -> dict:
    import mujoco
    objid, objtype = seg[..., 0], seg[..., 1]
    is_geom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    floor_ids = [g for g in range(m.ngeom) if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    sky = objid < 0
    floor = is_geom & np.isin(objid, floor_ids)
    body = ~sky & ~floor
    return {"skybox": sky, "floor": floor, "bodies": body}


def agree(a, b, mask=None) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    if mask is not None:
        d = d[mask]
    if d.size == 0:
        return {"px": 0}
    return {"px": int(d.size), "mean_abs": float(d.mean()), "exact_frac": float((d == 0).mean()),
            "within8_frac": float((d <= 8).mean()), "max_abs": int(d.max()),
            "p50": float(np.median(d)), "p90": float(np.percentile(d, 90))}


def by_region(a, b, mk) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    total = float(d.sum()) or 1.0
    out = {}
    for name, mask in mk.items():
        r = agree(a, b, mask)
        r["share"] = float(mask.mean())
        r["residual_share"] = float(d[mask].sum() / total)      # this region's part of sum |diff|
        out[name] = r
    return out


def box_down(x, f):
    n, h, w, c = x.shape
    return x.reshape(n, h // f, f, w // f, f, c).astype(np.float64).mean((2, 4)).round().astype(np.uint8)


def sheet(path: Path, official, ours, seg_masks, idx):
    try:
        from PIL import Image
    except ImportError:
        return None
    rows = []
    for i in idx:
        d = np.abs(ours[i].astype(np.int16) - official[i].astype(np.int16))
        amp = np.clip(d * 4, 0, 255).astype(np.uint8)
        segimg = np.zeros_like(official[i])
        segimg[seg_masks["skybox"][i]] = (60, 60, 200)
        segimg[seg_masks["floor"][i]] = (60, 160, 60)
        segimg[seg_masks["bodies"][i]] = (220, 80, 80)
        rows.append(np.concatenate([official[i], ours[i], amp, segimg], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    return str(path)


def one(config: str, states_dir: Path, out: Path, n: int) -> dict:
    import torch
    import stable_envs as se
    from stable_envs.testing import conformance as C

    t0 = time.perf_counter()
    env = se.make(config, batch=n, device="cuda")
    cls = type(env)
    m = cls._model()
    import mujoco
    row = {"config": config, "size": int(env.size), "textures": texture_table(m),
           "reference_id": cls.reference_id,
           "floor_geom_ids": [g for g in range(m.ngeom)
                              if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]}
    z = np.load(states_dir / f"{config}.npz", allow_pickle=False)
    frozen = np.ascontiguousarray(z["states"], np.float32)
    row["corpus_sha256"] = C.frames_digest(frozen)
    ref = C.reference_env(cls, env.size)
    # the draw check_visual_fidelity makes, reproduced exactly (fidelity_step4.reseed + the check)
    env.rng = np.random.default_rng(CHECK_SEED)
    if env.style_dim:
        env.set_style(None)
    check = np.ascontiguousarray(env.sample_states(n), np.float32)
    row["check_draw_sha256"] = C.frames_digest(check)

    saved = {}
    for tag, states in (("frozen", frozen), ("check", check)):
        env.rng = np.random.default_rng(CHECK_SEED)
        if env.style_dim:
            env.set_style(None)
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        seg = segmentation(cls, ref, states)
        mk = masks(m, seg)
        r = C.pixel_agreement(ours, official)
        r["ssim"] = C.ssim(ours, official)
        r["digest_ours"] = C.frames_digest(ours)
        r["digest_official"] = C.frames_digest(official)
        r["by_region"] = by_region(ours, official, mk)
        # SSIM restricted to a region: SSIM is windowed, so mask the per-pixel map instead
        r["ssim_by_region"] = {k: ssim_masked(ours, official, v) for k, v in mk.items()}
        r["region_share"] = {k: float(v.mean()) for k, v in mk.items()}
        row[tag] = r
        saved[f"{tag}_official"] = official
        saved[f"{tag}_ours"] = ours
        saved[f"{tag}_seg"] = seg.astype(np.int16)
        sheets = out / "sheets"
        sheets.mkdir(parents=True, exist_ok=True)
        r["sheet"] = sheet(sheets / f"{config}_{tag}.png", official, ours, mk,
                           np.linspace(0, len(states) - 1, 4).astype(int))

    # resolution consistency, the way the certifier measured it (frozen corpus, default style)
    env.rng = np.random.default_rng(CHECK_SEED)
    if env.style_dim:
        env.set_style(None)
    rc = C.check_resolution_consistency(env, frozen)
    S = int(env.size)
    a = env.render(frozen, size=S).cpu().numpy()
    big = env.render(frozen, size=2 * S).cpu().numpy()
    env.render(frozen[:1], size=S)
    down = box_down(big, 2)
    mk = masks(m, saved["frozen_seg"].astype(np.int32))
    d = np.abs(down.astype(np.int16) - a.astype(np.int16))
    rc["by_region"] = {}
    for k, mask in mk.items():
        dm = d[mask]
        rc["by_region"][k] = {"share": float(mask.mean()), "px": int(dm.shape[0]),
                              "exact_frac": float((dm.max(-1) == 0).mean()) if dm.size else None,
                              "mean_abs": float(dm.mean()) if dm.size else None,
                              "inexact_share": float((dm.max(-1) != 0).sum() / max(1, (d.max(-1) != 0).sum()))}
    rc["digest_S"] = C.frames_digest(a)
    rc["digest_2S_down"] = C.frames_digest(down)
    row["resolution"] = rc
    saved["frozen_ours_2S_down"] = down

    (out / "frames").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames" / f"{config}.npz", **saved)
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def ssim_masked(a, b, mask, win: int = 7) -> float | None:
    import scipy.ndimage as ndi
    f = lambda x: ndi.uniform_filter(x, size=win, mode="nearest")
    x = a.astype(np.float64) @ [0.299, 0.587, 0.114]
    y = b.astype(np.float64) @ [0.299, 0.587, 0.114]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ux, uy = f(x), f(y)
    vx, vy = f(x * x) - ux * ux, f(y * y) - uy * uy
    vxy = f(x * y) - ux * uy
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return float(s[mask].mean()) if mask.any() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--states-dir", default=str(ROOT / "examples" / "results" / "step4" / "states"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"what_this_is": "Step 6A per-surface localisation of the trackcom residual, one arm",
           "provenance": provenance(), "configs": {}}
    for cfg in a.configs.split(","):
        try:
            doc["configs"][cfg] = one(cfg, Path(a.states_dir), out, a.n)
            r = doc["configs"][cfg]
            print(f"[{cfg}] frozen mean={r['frozen']['mean_abs']:.3f} within8={r['frozen']['within8_frac']:.4f} "
                  f"ssim={r['frozen']['ssim']:.4f} | check mean={r['check']['mean_abs']:.3f} "
                  f"within8={r['check']['within8_frac']:.4f} ssim={r['check']['ssim']:.4f} | "
                  f"res exact={r['resolution']['exact_frac']:.4f}  ({r['seconds']}s)", flush=True)
        except Exception as e:                                          # noqa: BLE001
            import traceback
            doc["configs"][cfg] = {"config": cfg, "error": f"{type(e).__name__}: {e}",
                                   "traceback": traceback.format_exc()[-2000:]}
            print(f"[{cfg}] ERROR {type(e).__name__}: {e}", flush=True)
        (out / "localize.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print("wrote", out / "localize.json")


if __name__ == "__main__":
    main()

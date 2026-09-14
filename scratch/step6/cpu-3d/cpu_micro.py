"""Step 6G cpu-3d, Phase 1: the numbers a CPU cost model needs, measured on the certification host.

Run on an Oscar BATCH node (no GPU), one torch thread, in the se-step6 venv (dm_control, no warp):

  1. the same-host official renderer: dm_control physics.render at 84 px, camera 0, on THIS node
     (Step 3's DMC official numbers were taken on GPU nodes through hardware EGL; a batch node has
     Mesa only, and S_cpu is defined against the official renderer on the same host);
  2. the one-core JPEG-q90 decode bar for an 84 px DMC frame and for a 480 px frame;
  3. torch elementwise throughput at 1 thread: per-op cost against element count, i.e. the two
     constants (dispatch floor, bytes/s) behind every "vectorised torch" architecture;
  4. raycast3d's CPU cast (`cast_nearest_torch`, `any_blocking_torch`) on a trackcom-shaped scene
     (K capsules over the full 84x84 grid) -- the cost architecture (a)/(e) would pay per ray-primitive
     test, measured on this host rather than borrowed from the raycast3d workstream's note;
  5. MuJoCo's own FK stages per state on this host.

Writes JSON under --out. No renderer code is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = {"DMCCheetahRun": ("cheetah", "run"), "DMCHopperHop": ("hopper", "hop"),
           "DMCWalkerWalk": ("walker", "walk"), "DMCHumanoidStand": ("humanoid", "stand")}


def tim(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    s = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); s.append(time.perf_counter() - t0)
    return statistics.median(s)


def official(states_dir: Path, sizes=(84,)) -> dict:
    import mujoco
    from dm_control import suite
    out = {"mujoco_gl": os.environ.get("MUJOCO_GL")}
    try:
        from OpenGL import GL
    except Exception:
        GL = None
    for cname, (dom, task) in DOMAINS.items():
        env = suite.load(dom, task, task_kwargs={"random": 0})
        p = env.physics
        st = np.load(states_dir / f"{cname}.npz")["states"].astype(np.float64)
        rec = {}
        for size in sizes:
            def frame(i):
                with p.reset_context():
                    p.data.qpos[:] = st[i % len(st), :p.model.nq]
                return p.render(size, size, camera_id=0)
            f0 = frame(0)
            if GL is not None and "gl_renderer" not in out:
                try:
                    out["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
                    out["gl_version"] = GL.glGetString(GL.GL_VERSION).decode()
                except Exception as e:
                    out["gl_renderer"] = f"unreadable: {e}"
            n = 200 if size <= 128 else 40
            t0 = time.perf_counter()
            for i in range(n):
                frame(i)
            dt = (time.perf_counter() - t0) / n
            rec[str(size)] = {"fps_render_only": 1.0 / dt, "us_per_frame": dt * 1e6, "n": n}
            if size == 84:
                np.save(states_dir.parent.parent / "step6" / "cpu-3d" / "characterize" / f"{cname}_official_84.npy", f0)
        out[cname] = rec
        print(f"official {cname}: " + ", ".join(f"{k}px {v['fps_render_only']:.1f} f/s" for k, v in rec.items()), flush=True)
    return out


def decode_bar(frame: np.ndarray) -> dict:
    import io
    from PIL import Image
    h, w = frame.shape[:2]
    buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="JPEG", quality=90); blob = buf.getvalue()
    n = max(40, int(4e7 / (h * w)))
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    dt = (time.perf_counter() - t0) / n
    return {"h": h, "w": w, "jpeg_bytes": len(blob), "decode_fps_per_core": 1.0 / dt, "n": n}


def torch_throughput() -> dict:
    import torch
    out = {"threads": torch.get_num_threads(), "rows": []}
    for N in (7056, 16 * 7056, 64 * 7056, 230400, 4 * 230400, 16 * 230400):
        a = torch.rand(N); b = torch.rand(N); c = torch.rand(N)
        def chain():
            x = a * b
            for _ in range(9):
                x = x * b + c
            return x
        t = tim(chain, reps=10) / 10
        out["rows"].append({"N": N, "us_per_op": t * 1e6, "GB_per_s": 3 * 4 * N / t / 1e9,
                            "ns_per_element_op": t / N * 1e9})
        print(f"torch N={N:8d}: {t*1e6:8.1f} us/op  {3*4*N/t/1e9:6.2f} GB/s  {t/N*1e9:6.3f} ns/elem", flush=True)
    # dispatch floor: a 1-element op
    a1 = torch.rand(1)
    out["dispatch_floor_us"] = tim(lambda: a1 * a1, reps=200) * 1e6
    return out


def raycast_cost() -> dict:
    import torch
    from stable_envs.render import raycast3d as R3
    out = {"rows": []}
    H = W = 84
    cam = R3.Camera(pos=(0.0, -3.0, 0.9), forward=(0.0, 1.0, -0.15), up=(0.0, 0.0, 1.0),
                    near=0.1, top=0.0414, bottom=-0.0414)
    o, d = cam.rays(H, W, "cpu")
    rng = np.random.default_rng(0)
    for K in (1, 8, 19):
        kinds = [R3.CAPSULE] * K
        pos = rng.uniform([-0.6, -0.3, 0.1], [0.6, 0.3, 1.2], (K, 3))
        q = rng.normal(size=(K, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        R = np.zeros((K, 3, 3))
        for i in range(K):
            import mujoco
            mujoco.mju_quat2Mat(R[i].reshape(-1), q[i])
        size = np.column_stack([rng.uniform(0.04, 0.09, K), rng.uniform(0.05, 0.2, K), np.zeros(K)])
        for B in (1, 16, 64):
            N = H * W
            oo = o.expand(B, N, 3); dd = d.expand(B, N, 3)
            sz = torch.tensor(size, dtype=torch.float32).expand(B, K, 3)
            xp = torch.tensor(pos, dtype=torch.float32).expand(B, K, 3)
            xm = torch.tensor(R, dtype=torch.float32).expand(B, K, 3, 3)
            t_cast = tim(lambda: R3.cast_nearest_torch(kinds, sz, xp, xm, oo, dd), reps=5)
            lim = torch.full((B, N), R3.INF)
            t_shad = tim(lambda: R3.any_blocking_torch(kinds, sz, xp, xm, oo, dd, lim), reps=5)
            tests = B * N * K
            out["rows"].append({"K": K, "B": B, "cast_ms": t_cast * 1e3, "shadow_ms": t_shad * 1e3,
                                "cast_ns_per_ray_prim": t_cast / tests * 1e9,
                                "shadow_ns_per_ray_prim": t_shad / tests * 1e9,
                                "cast_fps_equiv": B / t_cast})
            print(f"raycast3d torch K={K:2d} B={B:2d}: cast {t_cast*1e3:8.2f} ms ({t_cast/tests*1e9:6.1f} ns/ray-prim), "
                  f"shadow {t_shad*1e3:8.2f} ms ({t_shad/tests*1e9:6.1f} ns)", flush=True)
    # one full-frame plane intersection + shade pass, B=64: the floor/sky half of a frame in torch
    B = 64
    N = H * W
    oo = o.expand(B, N, 3); dd = d.expand(B, N, 3)
    ppos = torch.zeros(B, 1, 3); pR = torch.eye(3).expand(B, 1, 3, 3)
    psz = torch.tensor([[100.0, 100.0, 0.1]]).expand(B, 1, 3)
    t_plane = tim(lambda: R3.intersect(R3.PLANE, psz[:, 0:1], ppos, pR, oo, dd), reps=5)
    lights = R3.Lights(dirs=np.array([[0, 0, 1.0], [0.3, 0.3, 0.9]]), ambient=np.array([0.4, 0.0]),
                       diffuse=np.array([0.8, 0.7]), specular=np.array([0.1, 0.3]), exponent=64.0)
    ldir, lamb, ldif, lspec = lights.to("cpu")
    n = torch.rand(B, N, 3); v = -dd; mat = torch.rand(B, N, 3); tex = torch.ones(B, N, 3); spec = torch.full((B, N), 0.5)
    t_shade = tim(lambda: R3.shade(n, v, ldir, lamb, ldif, lspec, 64.0, mat, tex, spec), reps=5)
    out["plane_pass_ms_B64"] = t_plane * 1e3
    out["shade_pass_ms_B64_2lights"] = t_shade * 1e3
    print(f"plane pass B=64: {t_plane*1e3:.2f} ms; shade pass (2 lights) B=64: {t_shade*1e3:.2f} ms", flush=True)
    return out


def fk_cost(states_dir: Path) -> dict:
    import mujoco
    from dm_control import suite
    out = {}
    for cname, (dom, task) in DOMAINS.items():
        m = suite.load(dom, task, task_kwargs={"random": 0}).physics.model.ptr
        d = mujoco.MjData(m)
        st = np.load(states_dir / f"{cname}.npz")["states"].astype(np.float64)
        d.qpos[:] = st[0, :m.nq]
        rec = {}
        for fn, name in ((mujoco.mj_kinematics, "mj_kinematics"), (mujoco.mj_comPos, "mj_comPos"),
                         (mujoco.mj_camlight, "mj_camlight"), (mujoco.mj_forward, "mj_forward")):
            fn(m, d)
            t0 = time.perf_counter()
            for _ in range(2000):
                fn(m, d)
            rec[name + "_us"] = (time.perf_counter() - t0) / 2000 * 1e6
        out[cname] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states-dir", type=Path, default=ROOT / "examples/results/step4/states")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(a.threads)
    host = socket.gethostname().split(".")[0]
    note = {"host": host, "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__}
    try:
        import cpuinfo  # noqa
    except Exception:
        try:
            note["cpu_model"] = [l for l in open("/proc/cpuinfo") if "model name" in l][0].split(":")[1].strip()
        except Exception:
            pass
    note["torch_throughput"] = torch_throughput()
    note["raycast3d_cpu"] = raycast_cost()
    note["fk"] = fk_cost(a.states_dir)
    try:
        note["official"] = official(a.states_dir, sizes=(84, 480))
        f84 = np.load(a.out / "DMCHumanoidStand_official_84.npy")
        note["decode_84"] = decode_bar(f84)
        from dm_control import suite
        p = suite.load("humanoid", "stand", task_kwargs={"random": 0}).physics
        note["decode_480_dmc_frame"] = decode_bar(p.render(480, 480, camera_id=0))
        print("decode 84:", note["decode_84"]["decode_fps_per_core"], "decode 480 (dmc):", note["decode_480_dmc_frame"]["decode_fps_per_core"], flush=True)
    except Exception as e:
        note["official"] = {"error": f"{type(e).__name__}: {e}"}
        print("official/decode failed:", note["official"], flush=True)
    (a.out / f"cpu_micro_{host}.json").write_text(json.dumps(note, indent=1))
    print("wrote", a.out / f"cpu_micro_{host}.json")


if __name__ == "__main__":
    main()

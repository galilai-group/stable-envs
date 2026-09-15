"""M4 probe 2: WHICH operator makes PFRocketLanding's frame depend on the batch size (job 6407151:
row 0 identical for B<=24 and for B>=31, the two families 10 px apart on the rocket's silhouette,
interior edges and 1/255 truncation steps -- ulp-level vertex/normal changes)?

Replicates render()'s preamble exactly and bitwise-compares, for row 0, every intermediate of
``Scene.clip`` across B in (1, 24, 31, 32, 64): ``mvp`` (dense batched 4x4 GEMM), ``pos``/``nrm``/``wp``
(the three ``torch.sparse.mm`` calls, whose dense operand has 4B / 3B columns). Then isolates the SpMM
by feeding it the B=1 ``mvp`` replicated (dense stage identical by construction), and the rasterizer by
feeding ``dr.rasterize`` the B=1 clip positions replicated. Diagnostic only; no code is changed.
"""
import json, os, sys, platform
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
import stable_envs as se
from stable_envs.envs import rocket_landing as RL
from stable_envs.render import nvdiff as ND
import nvdiffrast.torch as dr
OUT = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "tier", "probe_batch")


def diff(a, b):
    a, b = a.detach().float().cpu().numpy(), b.detach().float().cpu().numpy()
    ne = a != b
    ulp = np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))
    return {"elements_differing": int(ne.sum()), "max_abs": float(np.abs(a - b).max()), "max_ulp": int(ulp.max())}


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "slurm_job": os.environ.get("SLURM_JOB_ID"),
         "sha": open(os.path.join(ROOT, ".git_sha")).read().strip() if os.path.exists(os.path.join(ROOT, ".git_sha")) else "unknown",
         "torch": torch.__version__, "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
         "float32_matmul_precision": torch.get_float32_matmul_precision()}
    env = se.make("PFRocketLanding", batch=16, device="cuda")
    c = env._nvd(env.size, env.ss)
    S = c.br.scene
    st_all = env.sample_states(64, rng=np.random.default_rng(11))
    dev = "cuda"

    def stages(B):
        st = st_all[:B].astype(np.float64)
        xp, xm = RL.geom_poses(st)
        _, _, pos_prev, q_prev, _, _ = RL.lagged_pose(st)
        eye, cmat = RL.camera_pose(pos_prev, q_prev)
        xp_t = torch.as_tensor(xp, dtype=torch.float32, device=dev)
        xm_t = torch.as_tensor(xm, dtype=torch.float32, device=dev)
        cam = ND.Camera(eye, cmat, RL.CAM_FOV, RL.CAM_NEAR, RL.CAM_FAR)
        view_np, proj_np = cam.matrices(aspect=1.0)
        view = torch.as_tensor(view_np, dtype=torch.float32, device=dev)
        proj = torch.as_tensor(proj_np, dtype=torch.float32, device=dev)
        G = S.G
        M = torch.zeros(B, G, 4, 4, device=dev, dtype=torch.float32)
        M[:, :, :3, :3] = xm_t
        M[:, :, :3, 3] = xp_t
        M[:, :, 3, 3] = 1.0
        mvp = (proj @ view)[:, None] @ M if view.ndim == 3 else (proj @ view)[None, None] @ M
        pos, nrm, wp = S.clip(xp_t, xm_t, view, proj)
        return {"inputs_xp": xp_t, "inputs_xm": xm_t, "view": view, "mvp": mvp, "pos": pos, "nrm": nrm, "wp": wp}

    ref = stages(1)
    R["row0_vs_B1"] = {}
    for B in (24, 31, 32, 64):
        s = stages(B)
        R["row0_vs_B1"][str(B)] = {k: diff(s[k][0], ref[k][0]) for k in ("inputs_xp", "inputs_xm", "view", "mvp", "pos", "nrm", "wp")}
        print("B", B, json.dumps(R["row0_vs_B1"][str(B)]), flush=True)

    # SpMM alone: the B=1 mvp replicated B times -> the dense stage is identical by construction
    G = S.G
    R["spmm_alone_vs_B1"] = {}
    X1 = ref["mvp"]                                               # (1, G, 4, 4)
    p1 = torch.sparse.mm(S.S4, X1.permute(1, 3, 0, 2).reshape(4 * G, 4)).reshape(S.V, 1, 4)[:, 0]
    for B in (24, 31, 32, 64):
        XB = X1.expand(B, -1, -1, -1).contiguous()
        pB = torch.sparse.mm(S.S4, XB.permute(1, 3, 0, 2).reshape(4 * G, 4 * B)).reshape(S.V, B, 4)[:, 0]
        R["spmm_alone_vs_B1"][str(B)] = diff(pB, p1)
    print("spmm_alone", json.dumps(R["spmm_alone_vs_B1"]), flush=True)
    # dense GEMM alone: the same view/proj with M replicated
    R["dense_gemm_alone_vs_B1"] = {}
    view1, proj1 = ref["view"], None
    for B in (24, 31, 32, 64):
        s = stages(B)
        # replicate row 0 of M through the batched GEMM
        M0 = torch.zeros(B, G, 4, 4, device=dev)
        M0[:, :, :3, :3] = ref["inputs_xm"][0]
        M0[:, :, :3, 3] = ref["inputs_xp"][0]
        M0[:, :, 3, 3] = 1.0
        pv = (s["mvp"].new_zeros(0))  # placeholder
        PV1 = (ref["mvp"][0:1] * 0)  # not used
        # (proj @ view) for row 0 replicated:
        vp = (torch.as_tensor(ND.Camera(RL.camera_pose(*RL.lagged_pose(st_all[:1].astype(np.float64))[2:4])[0],
                                        RL.camera_pose(*RL.lagged_pose(st_all[:1].astype(np.float64))[2:4])[1],
                                        RL.CAM_FOV, RL.CAM_NEAR, RL.CAM_FAR).matrices(aspect=1.0)[1], dtype=torch.float32, device=dev)
              @ view1)                                            # (1,4,4)
        mvpB = vp.expand(B, -1, -1)[:, None] @ M0
        R["dense_gemm_alone_vs_B1"][str(B)] = diff(mvpB[0], ref["mvp"][0])
    print("dense_gemm_alone", json.dumps(R["dense_gemm_alone_vs_B1"]), flush=True)
    # rasterizer alone: identical clip positions replicated
    R["rasterize_alone_vs_B1"] = {}
    Rr = c.br.res * c.br.ss
    pos1 = ND.offset_clip(ref["pos"], Rr, c.br.sample_offset)
    r1, _ = dr.rasterize(c.br.ctx, pos1, S.face, resolution=[Rr, Rr], grad_db=False)
    for B in (24, 32, 64):
        rB, _ = dr.rasterize(c.br.ctx, pos1.expand(B, -1, -1).contiguous(), S.face, resolution=[Rr, Rr], grad_db=False)
        R["rasterize_alone_vs_B1"][str(B)] = diff(rB[0], r1[0])
    print("rasterize_alone", json.dumps(R["rasterize_alone_vs_B1"]), flush=True)
    with open(os.path.join(OUT, "probe_clip_ulps.json"), "w") as fh:
        json.dump(R, fh, indent=1)


if __name__ == "__main__":
    main()

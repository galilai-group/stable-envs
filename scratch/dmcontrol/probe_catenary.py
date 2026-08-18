"""Probe: reproduce MuJoCo's mjv_catenary (spatial-tendon "string" rendering) in pure numpy.

Ground truth is obtained two independent ways:
  (a) ctypes on the exported symbol mjv_catenary in libmujoco.so (float64),
  (b) the capsule geoms MuJoCo itself puts in an mjvScene for real ball_in_cup
      states (float32, so ~1e-7 accurate).

Run under srun; see the sbatch/srun line in the task notes.
"""

import ctypes
import glob
import math
import os

import numpy as np

# ----------------------------------------------------------------------------- ctypes ground truth

MUJOCO_PKG = os.path.dirname(__import__("mujoco").__file__)
LIBPATH = sorted(glob.glob(os.path.join(MUJOCO_PKG, "libmujoco.so.*")))[-1]
_lib = ctypes.CDLL(LIBPATH)

_D3 = ctypes.c_double * 3
_lib.mjv_catenary.restype = ctypes.c_int
_lib.mjv_catenary.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # const mjtNum x0[3]
    ctypes.POINTER(ctypes.c_double),  # const mjtNum x1[3]
    ctypes.POINTER(ctypes.c_double),  # const mjtNum gravity[3]
    ctypes.c_double,                  # mjtNum length
    ctypes.POINTER(ctypes.c_double),  # mjtNum* catenary  (out, >= 3*ncatenary)
    ctypes.c_int,                     # int ncatenary
]


def catenary_c(x0, x1, gravity, length, ncatenary):
    """Ground truth: call the C symbol directly."""
    a0 = np.ascontiguousarray(x0, dtype=np.float64)
    a1 = np.ascontiguousarray(x1, dtype=np.float64)
    ag = np.ascontiguousarray(gravity, dtype=np.float64)
    out = np.zeros(3 * max(ncatenary, 8) + 32, dtype=np.float64)
    p = ctypes.POINTER(ctypes.c_double)
    n = _lib.mjv_catenary(
        a0.ctypes.data_as(p), a1.ctypes.data_as(p), ag.ctypes.data_as(p),
        ctypes.c_double(float(length)), out.ctypes.data_as(p), ctypes.c_int(int(ncatenary)),
    )
    return out[: 3 * n].reshape(n, 3).copy()


# ------------------------------------------------------------------------- pure-numpy reproduction

_MJMINVAL = 1e-15
_TOL = 1e-9


def _cosh_sinh(x):
    e = math.exp(x)
    return 0.5 * (e + 1.0 / e), 0.5 * (e - 1.0 / e)


def _residual(b, intercept, want_grad):
    a = 0.5 / b
    ch, sh = _cosh_sinh(a)
    res = 1.0 / math.sqrt(2 * b * sh - 1.0) - intercept
    grad = (a * ch - sh) * math.pow(2 * b * sh - 1.0, -1.5) if want_grad else None
    return res, grad


def _solve_catenary(v, h, length):
    intercept = 1.0 / math.sqrt(math.sqrt(length * length - v * v) / h - 1.0)
    b = intercept / math.sqrt(24.0)
    for _ in range(50):
        res, grad = _residual(b, intercept, True)
        if abs(res) < _TOL:
            break
        step = -res / grad
        for _ in range(10):
            if abs(_residual(b + step, intercept, False)[0]) < abs(res):
                break
            step *= 0.5
        b += step
    return b


def catenary(x0, x1, gravity, length, npoints):
    """Pure-numpy port of MuJoCo mjv_catenary. Returns (N, 3) float64, N <= npoints."""
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    d = x0 - x1
    if math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) > length:   # taut: straight line
        return np.stack([x0, x1])
    up = np.asarray(gravity, dtype=np.float64) * -1.0
    n = math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2])
    up = np.array([1.0, 0.0, 0.0]) if n < _MJMINVAL else up * (1.0 / n)
    x01 = x1 - x0
    ac = x01 - up * (up[0] * x01[0] + up[1] * x01[1] + up[2] * x01[2])
    n = math.sqrt(ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2])
    ac = np.zeros(3) if n < _MJMINVAL else ac * (1.0 / n)
    h = x01[0] * ac[0] + x01[1] * ac[1] + x01[2] * ac[2]   # horizontal extent
    v = x01[0] * up[0] + x01[1] * up[1] + x01[2] * up[2]   # vertical rise
    if length > 100 * h:                                   # near-vertical: hanging bead, 3 points
        d_up = -0.5 * (math.sqrt(length * length - h * h) - v)
        return np.stack([x0, x0 + up * d_up + ac * (h * d_up / (2 * d_up - v)), x1])
    bh = _solve_catenary(v, h, length) * h                 # scaled catenary flatness
    h_off = -0.5 * (math.log((length + v) / (length - v)) * bh - h)
    v_off = -_cosh_sinh(h_off / bh)[0] * bh
    pts = np.empty((npoints, 3))
    pts[0], pts[npoints - 1] = x0, x1
    for i in range(1, npoints - 1):
        hor = i * h / npoints                              # note: /npoints, not /(npoints-1)
        pts[i] = x0 + ac * hor + up * (bh * _cosh_sinh((hor - h_off) / bh)[0] + v_off)
    return pts


# ----------------------------------------------------------------------------------------- testing

def bits(a, b):
    """max |a-b| and max ULP difference."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    ia = a.view(np.int64).copy()
    ib = b.view(np.int64).copy()
    ia[ia < 0] = np.int64(np.uint64(1) << np.uint64(63)) - ia[ia < 0]
    ib[ib < 0] = np.int64(np.uint64(1) << np.uint64(63)) - ib[ib < 0]
    return float(np.max(np.abs(a - b))), int(np.max(np.abs(ia - ib)))


def test_random(rng, n=200, ncat=29):
    maxerr = maxulp = 0
    counts = {2: 0, 3: 0, ncat: 0}
    for _ in range(n):
        x0 = rng.uniform(-2, 2, 3)
        mode = rng.integers(0, 4)
        if mode == 0:      # generic
            x1 = x0 + rng.uniform(-2, 2, 3)
        elif mode == 1:    # near-vertical (triggers the bead branch)
            x1 = x0 + np.array([rng.normal(0, 1e-4), rng.normal(0, 1e-4), rng.uniform(-1, 1)])
        elif mode == 2:    # near-horizontal
            x1 = x0 + np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.normal(0, 1e-3)])
        else:              # nearly degenerate (x0 ~ x1)
            x1 = x0 + rng.normal(0, 1e-9, 3)
        dist = float(np.linalg.norm(x1 - x0))
        length = dist * float(rng.uniform(0.5, 3.0))
        g = np.array([0.0, 0.0, -9.81]) if rng.integers(0, 2) else rng.normal(0, 5, 3)
        if np.linalg.norm(g) < 1e-12:
            g = np.array([0.0, 0.0, -9.81])
        ref = catenary_c(x0, x1, g, length, ncat)
        got = catenary(x0, x1, g, length, ncat)
        assert got.shape == ref.shape, (got.shape, ref.shape, mode, length, dist)
        counts[ref.shape[0]] = counts.get(ref.shape[0], 0) + 1
        e, u = bits(ref, got)
        maxerr, maxulp = max(maxerr, e), max(maxulp, u)
    return maxerr, maxulp, counts


def main():
    import mujoco
    from dm_control import suite

    print("mujoco", mujoco.__version__, "lib", LIBPATH)

    # --- random-input validation vs the C symbol -----------------------------------------------
    rng = np.random.default_rng(0)
    e, u, counts = test_random(rng, 200, 29)
    print(f"[random x200] max|err| = {e:.3e}   max ULP = {u}   npoints histogram = {counts}")
    for nc in (3, 5, 29, 100):
        e2, u2, _ = test_random(np.random.default_rng(1), 60, nc)
        print(f"[random x60, ncatenary={nc:3d}] max|err| = {e2:.3e}  max ULP = {u2}")

    # --- ball_in_cup ---------------------------------------------------------------------------
    env = suite.load("ball_in_cup", "catch", task_kwargs={"random": 0})
    physics = env.physics
    m, d = physics.model.ptr, physics.data.ptr
    i = 0  # single tendon

    print("\n--- ball_in_cup model facts ---")
    print("ntendon", m.ntendon, "name", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, 0))
    print("tendon_num", m.tendon_num[i], "tendon_limited", m.tendon_limited[i],
          "range", m.tendon_range[i], "width", m.tendon_width[i])
    print("tendon_stiffness", m.tendon_stiffness[i], "damping", m.tendon_damping[i],
          "frictionloss", m.tendon_frictionloss[i], "lengthspring", m.tendon_lengthspring[i])
    print("tendon_matid", m.tendon_matid[i], "tendon_rgba", m.tendon_rgba[i],
          "tendon_group", m.tendon_group[i])
    print("opt.gravity", m.opt.gravity, "disableflags", m.opt.disableflags)
    print("vis.quality.numslices", m.vis.quality.numslices,
          "-> ncatenary =", min(m.vis.quality.numslices + 1, 100))
    print("actuator trntype", [int(m.actuator_trntype[j]) for j in range(m.nu)],
          "(mjTRN_TENDON =", int(mujoco.mjtTrn.mjTRN_TENDON), ")")
    print("vis.rgba.constraint", m.vis.rgba.constraint)
    if m.tendon_matid[i] >= 0:
        print("material rgba", m.mat_rgba[m.tendon_matid[i]])

    ncat = min(m.vis.quality.numslices + 1, 100)
    vopt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    scn = mujoco.MjvScene(m, maxgeom=2000)

    # replicate mjv_isCatenary in python for reporting
    def is_catenary():
        has_stiff = bool(m.tendon_stiffness[i]) or bool(np.any(m.tendon_stiffnesspoly[i]))
        lspring = has_stiff and m.tendon_lengthspring[i][0] == 0 and m.tendon_lengthspring[i][1] > 0
        lconstr = ((not has_stiff) and m.tendon_limited[i] == 1
                   and m.tendon_range[i][0] == 0 and d.ten_length[i] < m.tendon_range[i][1])
        has_damp = bool(m.tendon_damping[i]) or bool(np.any(m.tendon_dampingpoly[i]))
        ok = (not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
              and np.linalg.norm(m.opt.gravity) > _MJMINVAL and m.tendon_num[i] == 2
              and (bool(lspring) != bool(lconstr)) and not has_damp
              and m.tendon_frictionloss[i] == 0)
        if ok:
            for j in range(m.nu):
                if (m.actuator_trntype[j] == mujoco.mjtTrn.mjTRN_TENDON
                        and m.actuator_trnid[j][0] == i):
                    ok = False
        L = (m.tendon_range[i][1] if lconstr else m.tendon_lengthspring[i][1]) if ok else None
        return ok, L

    env.reset()
    arng = np.random.default_rng(3)
    max_scene_err = 0.0
    max_c_err = 0
    nstates = ncap_seen = ntaut = 0
    lengths_seen = set()
    ten_len_min, ten_len_max = 1e9, -1e9
    rgba_seen = set()
    radius_seen = set()
    branch_counts = {}

    for step in range(400):
        env.step(arng.uniform(-1, 1, 2))
        if step % 6:
            continue
        ok, L = is_catenary()
        ten_len_min = min(ten_len_min, float(d.ten_length[i]))
        ten_len_max = max(ten_len_max, float(d.ten_length[i]))
        if not ok:
            ntaut += 1
            continue
        lengths_seen.add(L)
        # NOTE: mjData.wrap_xpos is exposed to python as (nwrap, 6); ten_wrapadr counts
        # 3-vectors, so reshape to (nwrap*2, 3) first.
        adr = d.ten_wrapadr[i]
        wx = np.asarray(d.wrap_xpos, dtype=np.float64).reshape(-1, 3)
        x0 = wx[adr].copy()
        x1 = wx[adr + 1].copy()

        # (a) C ground truth vs numpy
        ref = catenary_c(x0, x1, m.opt.gravity, L, ncat)
        got = catenary(x0, x1, m.opt.gravity, L, ncat)
        max_c_err = max(max_c_err, bits(ref, got)[1])

        # (b) independent: read the capsules MuJoCo actually put in the scene
        mujoco.mjv_updateScene(m, d, vopt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        segs = [scn.geoms[k] for k in range(scn.ngeom)
                if scn.geoms[k].objtype == mujoco.mjtObj.mjOBJ_TENDON]
        ncap_seen = len(segs)
        ends = []
        for g in segs:
            axis = np.array(g.mat, dtype=np.float64).reshape(3, 3)[:, 2]
            p = np.array(g.pos, dtype=np.float64)
            ends.append((p - axis * g.size[2], p + axis * g.size[2]))
            rgba_seen.add(tuple(np.round(np.array(g.rgba, np.float64), 6)))
            radius_seen.add(round(float(g.size[0]), 9))
        chain = np.array([ends[0][0]] + [e[1] for e in ends])
        assert len(segs) == got.shape[0] - 1, (len(segs), got.shape)
        max_scene_err = max(max_scene_err, float(np.max(np.abs(chain - got))))
        branch_counts[got.shape[0]] = branch_counts.get(got.shape[0], 0) + 1
        nstates += 1
        if nstates >= 64:
            break

    print("\n--- ball_in_cup validation ---")
    print(f"states checked (catenary drawn): {nstates}; states with catenary OFF (taut): {ntaut}")
    print(f"ten_length range over rollout: [{ten_len_min:.6f}, {ten_len_max:.6f}]  "
          f"(tendon_range upper = {m.tendon_range[i][1]})")
    print(f"catenary length used: {sorted(lengths_seen)}")
    print(f"capsule segments in scene (last state): {ncap_seen}  (npoints = {ncap_seen + 1})")
    print(f"npoints histogram over states: {branch_counts}  (3 = near-vertical bead branch)")
    print(f"capsule radius (geom.size[0]): {sorted(radius_seen)}")
    print(f"segment rgba: {sorted(rgba_seen)}")
    print(f"numpy vs C   : max ULP = {max_c_err}  (0 == bit-for-bit identical)")
    print(f"numpy vs scene capsules (float32 mjvGeom): max|err| = {max_scene_err:.3e}")

    # --- does a normal rollout ever go taut? ----------------------------------------------------
    print("\n--- taut frequency over long rollouts ---")
    for tag, act in (("uniform[-1,1]", lambda r: r.uniform(-1, 1, 2)),
                     ("bang-bang +-1", lambda r: r.choice([-1.0, 1.0], 2)),
                     ("bang-bang, switch every 5", None)):
        r = np.random.default_rng(7)
        env.reset()
        mx, ntot, noff, nbead = 0.0, 0, 0, 0
        a = r.choice([-1.0, 1.0], 2)
        for t in range(4000):
            if act is None:
                if t % 5 == 0:
                    a = r.choice([-1.0, 1.0], 2)
            else:
                a = act(r)
            ts = env.step(a)
            if ts.last():
                env.reset()
            tl = float(d.ten_length[i])
            mx = max(mx, tl)
            ntot += 1
            ok2, L2 = is_catenary()
            if not ok2:
                noff += 1
            else:
                wx2 = np.asarray(d.wrap_xpos, dtype=np.float64).reshape(-1, 3)
                if catenary(wx2[d.ten_wrapadr[i]], wx2[d.ten_wrapadr[i] + 1],
                            m.opt.gravity, L2, ncat).shape[0] == 3:
                    nbead += 1
        print(f"  {tag:26s} steps={ntot} max ten_length={mx:.6f} "
              f"taut(catenary off)={noff} ({100*noff/ntot:.2f}%)  bead-branch={nbead}")

    # --- forced taut state: what does MuJoCo draw? ----------------------------------------------
    print("\n--- forced taut state ---")
    with physics.reset_context():
        physics.named.data.qpos["ball_x"] = 0.4
        physics.named.data.qpos["ball_z"] = 0.4
    physics.forward()
    ok, L = is_catenary()
    mujoco.mjv_updateScene(m, d, vopt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    segs = [scn.geoms[k] for k in range(scn.ngeom)
            if scn.geoms[k].objtype == mujoco.mjtObj.mjOBJ_TENDON]
    print(f"ten_length = {float(d.ten_length[i]):.6f} > range upper {m.tendon_range[i][1]}; "
          f"is_catenary = {ok}; tendon capsules in scene = {len(segs)}")
    if segs:
        print("taut segment rgba =", np.array(segs[0].rgba), "radius =", segs[0].size[0])
    # and mjv_catenary's own taut branch, for completeness
    print("mjv_catenary(dist>length) returns:", catenary_c([0, 0, 0], [1, 0, 0], [0, 0, -9.81],
                                                           0.5, 29).shape)

    # taut-case rgba/geometry statistics over a rollout
    r = np.random.default_rng(11)
    env.reset()
    taut_rgba, taut_err, taut_ncap = [], 0.0, set()
    for t in range(2000):
        ts = env.step(r.choice([-1.0, 1.0], 2) if t % 5 == 0 else r.uniform(-1, 1, 2))
        if ts.last():
            env.reset()
        if is_catenary()[0]:
            continue
        mujoco.mjv_updateScene(m, d, vopt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        sg = [scn.geoms[k] for k in range(scn.ngeom)
              if scn.geoms[k].objtype == mujoco.mjtObj.mjOBJ_TENDON]
        taut_ncap.add(len(sg))
        taut_rgba.append(np.array(sg[0].rgba, np.float64))
        wx2 = np.asarray(d.wrap_xpos, np.float64).reshape(-1, 3)
        ax = np.array(sg[0].mat, np.float64).reshape(3, 3)[:, 2]
        p = np.array(sg[0].pos, np.float64)
        taut_err = max(taut_err, float(np.max(np.abs(
            np.array([p - ax * sg[0].size[2], p + ax * sg[0].size[2]])
            - wx2[d.ten_wrapadr[i]:d.ten_wrapadr[i] + 2]))))
    taut_rgba = np.array(taut_rgba)
    print(f"taut states sampled: {len(taut_rgba)}; capsule counts seen: {sorted(taut_ncap)}")
    print(f"taut rgba min {taut_rgba.min(0)}  max {taut_rgba.max(0)}")
    print(f"taut capsule endpoints vs wrap_xpos: max|err| = {taut_err:.3e}")


if __name__ == "__main__":
    main()

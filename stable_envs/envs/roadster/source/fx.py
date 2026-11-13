"""Exhaust smoke: vapor puffs of the cold-gas jets, drawn as a pure function of the exhaust history.

The sim records, for each 50 Hz step, where the car was and how hard each jet fired (`RoadsterSim` keeps
the last `HISTORY` steps, 1.5 s, the longest a puff lives). `Smoke.draw` regenerates every puff from that
history: the puffs a step emitted (count, jitter, life and size) come from a random generator seeded by
the step index, and each puff's motion is the closed-form solution of its equation of
motion over its age (velocity relaxing to the wind, a little buoyancy, growth and fade). Nothing carries
over between frames, so a frame depends on the simulator state alone and a restored state draws the same
smoke as the uninterrupted run. The pool is a fixed set of sphere geoms on the world body; when there are
more puffs than geoms the youngest are kept.
"""

from __future__ import annotations

import mujoco
import numpy as np

N_SMOKE = 1600
HISTORY = 75  # exhaust history entries kept by the env: 1.5 s at 50 Hz
STEP_DT = 0.02  # s between history entries (the env's control step)
LIFE = (0.7, 1.5)  # s
RATE = 300.0  # puffs per second per nozzle at full throttle
MAX_PER_STEP = 16  # puffs a nozzle can emit per step (poisson(6) at full throttle; 16 is beyond its tail)
RELAX = 3.2  # 1/s: velocity relaxes to the wind
BUOYANCY = 0.9  # m/s^2 upward
GROUND_SPREAD = 2.0  # m/s^2: puffs on the ground spread sideways
MIN_THROTTLE = 0.03
VIS = 'contype="0" conaffinity="0"'

# layout of one history entry: step index, time, chassis position, quaternion, velocity, jet ctrl, wind
ENTRY = {
    "step": (0, 1),
    "time": (1, 1),
    "pos": (2, 3),
    "quat": (5, 4),
    "vel": (9, 3),
    "ctrl": (12, 9),
    "wind": (21, 3),
}
ENTRY_DIM = 24
GREY = np.array([0.8, 0.82, 0.85])


def xml() -> str:
    return "\n".join(
        f'<geom name="smoke{i}" type="sphere" size="0.1" pos="0 0 -80" rgba="1 1 1 0" '
        f'material="smoke" group="1" {VIS}/>'
        for i in range(N_SMOKE)
    )


ASSET = '<material name="smoke" specular="0" shininess="0" emission="0.45"/>'


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(n, 4) unit quaternions [w, x, y, z] -> (n, 3, 3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


class Smoke:
    """Draws the puffs of an exhaust history into the smoke geom pool (visual only).

    `nozzles` is a list of (jet index, sign, position in the chassis frame, exhaust direction) for every
    flame; a bidirectional jet has two nozzles with opposite signs.
    """

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"smoke{i}") for i in range(N_SMOKE)]
        self.ids = np.array(ids)
        self.available = bool((self.ids >= 0).all())
        self.nozzles: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        self.tint = np.ones(3)
        self._draws: dict = {}  # memoized per-step random draws (see _emit)
        self._hide()

    def _hide(self):
        if self.available:
            self.m.geom_rgba[self.ids, 3] = 0.0
            self.m.geom_pos[self.ids] = (0.0, 0.0, -80.0)

    # ------------------------------------------------------------------ emission
    def _emit(self, hist: np.ndarray):
        """Puffs emitted by the history entries: (p0, v0, r0, life, t_emit, wind) arrays, youngest last."""
        n_noz = len(self.nozzles)
        if n_noz == 0 or len(hist) == 0:
            return None
        jet = np.array([k for k, _, _, _ in self.nozzles])
        sign = np.array([s for _, s, _, _ in self.nozzles], float)
        p_noz = np.stack([p for _, _, p, _ in self.nozzles])  # (n_noz, 3)
        ex_noz = np.stack([e for _, _, _, e in self.nozzles])
        E = len(hist)
        ctrl = hist[:, ENTRY["ctrl"][0] : ENTRY["ctrl"][0] + 9][:, jet]  # (E, n_noz)
        u = np.where(sign[None] == 0, np.maximum(ctrl, 0.0), np.maximum(sign[None] * ctrl, 0.0))
        u = np.where(u > MIN_THROTTLE, u, 0.0)
        K = MAX_PER_STEP
        count = np.zeros((E, n_noz), int)
        jitter = np.empty((E, n_noz, K, 3))
        dist = np.empty((E, n_noz, K))
        speed = np.empty((E, n_noz, K))
        life = np.empty((E, n_noz, K))
        r0 = np.empty((E, n_noz, K))
        cache = {}
        for i in range(E):
            # One generator per step, seeded by the step index only. The draws are a pure function of the
            # entry, so they are memoized per (step, jet commands) while the entry stays in the history.
            key = (int(hist[i, 0]), hist[i, ENTRY["ctrl"][0] : ENTRY["ctrl"][0] + 9].tobytes())
            draws = self._draws.get(key)
            if draws is None:
                rng = np.random.default_rng(int(hist[i, 0]))
                draws = (
                    np.minimum(rng.poisson(u[i] * RATE * STEP_DT), K),
                    rng.normal(0.0, 1.0, (n_noz, K, 3)),
                    rng.uniform(0.05, 0.25, (n_noz, K)),
                    rng.uniform(0.8, 1.15, (n_noz, K)),
                    rng.uniform(LIFE[0], LIFE[1], (n_noz, K)),
                    rng.uniform(0.04, 0.08, (n_noz, K)),
                )
            cache[key] = draws
            count[i], jitter[i], dist[i], speed[i], life[i], r0[i] = draws
        self._draws = cache
        live = np.arange(K)[None, None, :] < count[:, :, None]  # (E, n_noz, K)
        if not live.any():
            return None
        R = _quat_to_mat(hist[:, ENTRY["quat"][0] : ENTRY["quat"][0] + 4])  # (E, 3, 3)
        x0 = hist[:, ENTRY["pos"][0] : ENTRY["pos"][0] + 3]
        car_vel = hist[:, ENTRY["vel"][0] : ENTRY["vel"][0] + 3]
        p_w = x0[:, None] + np.einsum("eij,nj->eni", R, p_noz)  # (E, n_noz, 3)
        d_w = np.einsum("eij,nj->eni", R, ex_noz)
        e, n, k = np.nonzero(live)
        d = d_w[e, n]
        j = jitter[e, n, k]
        uu = u[e, n]
        p0 = p_w[e, n] + d * dist[e, n, k][:, None] + 0.03 * j
        v0 = d * ((10 + 9 * uu) * speed[e, n, k])[:, None] + 1.4 * j + 0.6 * car_vel[e]
        return (
            p0,
            v0,
            r0[e, n, k],
            life[e, n, k] * (0.6 + 0.4 * uu),
            hist[e, ENTRY["time"][0]],
            hist[e, ENTRY["wind"][0] : ENTRY["wind"][0] + 3],
        )

    # ------------------------------------------------------------------ motion
    @staticmethod
    def _advance(p0, v0, r0, life, age, wind):
        """Position and radius of each puff at `age`.

        Velocity relaxes to the wind (rate RELAX) under a constant buoyancy; puffs that reach the ground stay
        on it and spread sideways.
        """
        k = RELAX
        decay = np.exp(-k * age)[:, None]
        v_inf = wind.copy()
        v_inf[:, 2] += BUOYANCY / k  # steady-state rise of the warm-ish vapor
        p = p0 + v_inf * age[:, None] + (v0 - v_inf) * (1 - decay) / k
        frac = np.clip(age / life, 0, 1)
        r = r0 * (1 + 7.0 * frac**0.7)
        low = p[:, 2] < r
        if low.any():
            # the age at which the puff met the ground (linear estimate), then a sideways spread from there
            z0 = np.maximum(p0[low, 2] - r[low], 1e-6)
            drop = z0 - (p[low, 2] - r[low])
            a_g = age[low] * np.clip(z0 / np.maximum(drop, 1e-9), 0.0, 1.0)
            on_ground = np.maximum(age[low] - a_g, 0.0)
            v_g = v_inf[low] + (v0[low] - v_inf[low]) * np.exp(-k * a_g)[:, None]
            heading = v_g[:, :2] / (np.linalg.norm(v_g[:, :2], axis=1, keepdims=True) + 1e-9)
            p[low, :2] += heading * (0.5 * GROUND_SPREAD * on_ground**2)[:, None]
            p[low, 2] = r[low]
        return p, r, frac

    # ------------------------------------------------------------------ drawing
    def puffs(self, hist: np.ndarray, t_now: float):
        """Live puffs at time `t_now`: (positions, radii, colors, alphas), youngest first, at most N_SMOKE."""
        emitted = self._emit(np.asarray(hist, float))
        if emitted is None:
            return None
        p0, v0, r0, life, t_emit, wind = emitted
        age = t_now - t_emit
        keep = (age >= 0) & (age < life)
        if not keep.any():
            return None
        p0, v0, r0, life, age, wind = p0[keep], v0[keep], r0[keep], life[keep], age[keep], wind[keep]
        order = np.argsort(age, kind="stable")[:N_SMOKE]  # youngest first; the oldest go when over budget
        p0, v0, r0, life, age, wind = p0[order], v0[order], r0[order], life[order], age[order], wind[order]
        p, r, frac = self._advance(p0, v0, r0, life, age, wind)
        alpha = 0.13 * (1 - frac) ** 1.3 * np.clip(age / 0.05, 0.3, 1)
        hot = np.clip(1 - age / 0.12, 0, 1)[:, None]  # bright, tinted for the first ~0.1 s
        col = self.tint * hot + GREY * (1 - hot)
        return p, r, col, alpha

    def draw(self, hist: np.ndarray, t_now: float, data: mujoco.MjData):
        """Write the puffs of `hist` at time `t_now` into the geom pool (model and the data's world poses)."""
        if not self.available:
            return
        m = self.m
        result = self.puffs(hist, t_now)
        n = 0 if result is None else len(result[0])
        if n:
            p, r, col, alpha = result
            ids = self.ids[:n]
            m.geom_size[ids, 0] = r
            m.geom_rgba[ids, :3] = col
            m.geom_rgba[ids, 3] = alpha
            m.geom_pos[ids] = p
            data.geom_xpos[ids] = p  # world-body geoms: show this frame, not after the next kinematics
        rest = self.ids[n:]
        m.geom_rgba[rest, 3] = 0.0
        m.geom_pos[rest] = (0.0, 0.0, -80.0)
        data.geom_xpos[rest] = (0.0, 0.0, -80.0)

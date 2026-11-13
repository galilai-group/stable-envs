"""One Triton kernel per classic-control domain: the step of every lane, its outputs and its render rows.

Each kernel repeats the operations of the Torch kernel in ``batched.py`` in the same order, without
fused multiply-adds, so each operation rounds as the Torch operation does. Constants
that float32 cannot hold exactly come from a float64 table in device memory, because Triton types a
Python float literal as float32.
"""

from __future__ import annotations

import math

import torch

from .batched import _F32_GOAL, _F32_LEFT, _F32_RIGHT

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

#: Float64 constants of each kernel, read by index.
CONSTANTS = {
    "cartpole": (2.4, 12 * 2 * math.pi / 360, 4.0 / 3.0),
    "pendulum": (math.pi, 2 * math.pi, 0.001, 0.1),
    "acrobot": (
        math.pi,
        math.pi - (-math.pi),
        math.pi / 2.0,
        9.8,
        0.2 / 2.0,
        0.2,
        0.2 / 6.0,
        4 * math.pi,
        9 * math.pi,
    ),
    "mountain_car": (1.2, 0.6),
    "mountain_car_continuous": (0.0025, 0.6, 1.2, 0.45, 0.1, _F32_RIGHT, _F32_LEFT, _F32_GOAL),
}

#: Kernel argument -> State field or output, per domain.
COLUMNS = {
    "cartpole": {
        "X": "x",
        "XD": "x_dot",
        "TH": "theta",
        "THD": "theta_dot",
        "GRAV": ("physics", "gravity"),
        "MCART": ("physics", "masscart"),
        "MPOLE": ("physics", "masspole"),
        "LEN": ("physics", "length"),
        "FMAG": ("physics", "force_mag"),
        "TAU": ("physics", "tau"),
        "BEY": ("task", "steps_beyond_terminated"),
        "HAS": ("task", "has_steps_beyond_terminated"),
    },
    "pendulum": {
        "TH": "theta",
        "THD": "theta_dot",
        "TQ": "torque",
        "HAS": ("task", "has_last_u"),
        "GRAV": ("physics", "g"),
        "MASS": ("physics", "m"),
        "LEN": ("physics", "l"),
        "MAXT": ("physics", "max_torque"),
        "MAXS": ("physics", "max_speed"),
        "DT": ("physics", "dt"),
    },
    "acrobot": {
        "T1": "theta1",
        "T2": "theta2",
        "D1": "dtheta1",
        "D2": "dtheta2",
        "L1": ("physics", "link_length_1"),
        "M1": ("physics", "link_mass_1"),
        "M2": ("physics", "link_mass_2"),
        "LC1": ("physics", "link_com_pos_1"),
        "LC2": ("physics", "link_com_pos_2"),
        "MOI": ("physics", "link_moi"),
    },
    "mountain_car": {
        "POS": "position",
        "VEL": "velocity",
        "GRAV": ("physics", "gravity"),
        "FORCE": ("physics", "force"),
        "MAXS": ("physics", "max_speed"),
    },
    "mountain_car_continuous": {
        "POS": "position",
        "VEL": "velocity",
        "IS32": ("task", "state_is_float32"),
        "POWER": ("physics", "power"),
        "MAXS": ("physics", "max_speed"),
    },
}

if HAVE_TRITON:

    @triton.jit
    def _r32(x):
        return x.to(tl.float32).to(tl.float64)

    @triton.jit
    def _keep(new, old, live, MASKED: tl.constexpr):
        if MASKED:
            return tl.where(live, new, old)
        return new

    @triton.jit
    def _finish(
        I, O, Bo, rows, ok, live, reward, term, F,
        WF: tl.constexpr, WI: tl.constexpr, WO: tl.constexpr, WB: tl.constexpr, REW: tl.constexpr,
        EL: tl.constexpr, TERM: tl.constexpr, TRUNC: tl.constexpr, MASKED: tl.constexpr, MAX_STEPS: tl.constexpr,
    ):  # fmt: skip
        """Write the reward, the episode flags and the elapsed steps."""
        elapsed = tl.load(I + rows * WI + EL, mask=ok, other=0)
        if MASKED:
            elapsed = elapsed + live.to(tl.int64)
            truncated = live & (elapsed >= MAX_STEPS)
            reward = tl.where(live, reward, 0.0)
            term = live & term
        else:
            elapsed = elapsed + 1
            truncated = elapsed >= MAX_STEPS
        tl.store(I + rows * WI + EL, elapsed, mask=ok)
        tl.store(F + rows * WF + REW, reward, mask=ok)
        tl.store(Bo + rows * WB + TERM, term, mask=ok)
        tl.store(Bo + rows * WB + TRUNC, truncated, mask=ok)

    @triton.jit
    def _observe(
        O, rows, ok, live, k: tl.constexpr, value, WO: tl.constexpr, OBS: tl.constexpr, INFO: tl.constexpr,
        MASKED: tl.constexpr,
    ):  # fmt: skip
        if MASKED:
            value = tl.where(live, value, 0.0)
        value = value.to(tl.float32)
        tl.store(O + rows * WO + OBS + k, value, mask=ok)
        tl.store(O + rows * WO + INFO + k, value, mask=ok)

    @triton.jit
    def _cartpole_kernel(
        F, I, O, Bo, A, M, SIM, R, K, n,
        WF: tl.constexpr, WI: tl.constexpr, WO: tl.constexpr, WB: tl.constexpr,
        X: tl.constexpr, XD: tl.constexpr, TH: tl.constexpr, THD: tl.constexpr, GRAV: tl.constexpr,
        MCART: tl.constexpr, MPOLE: tl.constexpr, LEN: tl.constexpr, FMAG: tl.constexpr, TAU: tl.constexpr,
        BEY: tl.constexpr, HAS: tl.constexpr, REW: tl.constexpr, EL: tl.constexpr, TERM: tl.constexpr,
        TRUNC: tl.constexpr, OBS: tl.constexpr, INFO: tl.constexpr,
        MASKED: tl.constexpr, NARROW: tl.constexpr, MAX_STEPS: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = rows < n
        f = F + rows * WF
        live = tl.load(M + rows, mask=ok, other=0) != 0 if MASKED else ok
        x0, xd0 = tl.load(f + X, mask=ok, other=0.0), tl.load(f + XD, mask=ok, other=0.0)
        th0, thd0 = tl.load(f + TH, mask=ok, other=0.0), tl.load(f + THD, mask=ok, other=0.0)
        gravity, masscart = tl.load(f + GRAV, mask=ok, other=1.0), tl.load(f + MCART, mask=ok, other=1.0)
        masspole, length = tl.load(f + MPOLE, mask=ok, other=1.0), tl.load(f + LEN, mask=ok, other=1.0)
        force_mag, tau = tl.load(f + FMAG, mask=ok, other=1.0), tl.load(f + TAU, mask=ok, other=1.0)
        bound, limit, four_thirds = tl.load(K), tl.load(K + 1), tl.load(K + 2)
        a = tl.load(A + rows, mask=ok, other=0)
        total_mass = masspole + masscart
        polemass_length = masspole * length
        force = tl.where(a == 1, force_mag, -force_mag)
        costheta, sintheta = tl.cos(th0), tl.sin(th0)
        temp = (force + polemass_length * (thd0 * thd0) * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (four_thirds - masspole * (costheta * costheta) / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        x, x_dot = x0 + tau * xd0, xd0 + tau * xacc
        theta, theta_dot = th0 + tau * thd0, thd0 + tau * thetaacc
        term = (x < -bound) | (x > bound) | (theta < -limit) | (theta > limit)
        beyond = tl.load(I + rows * WI + BEY, mask=ok, other=0)
        has = tl.load(Bo + rows * WB + HAS, mask=ok, other=0) != 0
        again = term & has
        reward = tl.where(again, 0.0, 1.0).to(tl.float64)
        new_beyond = tl.where(again, beyond + 1, tl.where(term, 0, beyond))
        x, x_dot = _keep(x, x0, live, MASKED), _keep(x_dot, xd0, live, MASKED)
        theta, theta_dot = _keep(theta, th0, live, MASKED), _keep(theta_dot, thd0, live, MASKED)
        tl.store(f + X, x, mask=ok)
        tl.store(f + XD, x_dot, mask=ok)
        tl.store(f + TH, theta, mask=ok)
        tl.store(f + THD, theta_dot, mask=ok)
        tl.store(I + rows * WI + BEY, _keep(new_beyond, beyond, live, MASKED), mask=ok)
        tl.store(Bo + rows * WB + HAS, _keep(has | term, has, live, MASKED), mask=ok)
        _finish(I, O, Bo, rows, ok, live, reward, term, F, WF, WI, WO, WB, REW, EL, TERM, TRUNC, MASKED, MAX_STEPS)
        _observe(O, rows, ok, live, 0, x, WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 1, x_dot, WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 2, theta, WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 3, theta_dot, WO, OBS, INFO, MASKED)
        tl.store(SIM + rows * 4, x, mask=ok)
        tl.store(SIM + rows * 4 + 1, x_dot, mask=ok)
        tl.store(SIM + rows * 4 + 2, theta, mask=ok)
        tl.store(SIM + rows * 4 + 3, theta_dot, mask=ok)
        tl.store(R + rows * 2, x.to(tl.float32), mask=ok)
        tl.store(R + rows * 2 + 1, theta.to(tl.float32), mask=ok)

    @triton.jit
    def _pendulum_kernel(
        F, I, O, Bo, A, M, SIM, R, K, n,
        WF: tl.constexpr, WI: tl.constexpr, WO: tl.constexpr, WB: tl.constexpr,
        TH: tl.constexpr, THD: tl.constexpr, TQ: tl.constexpr, HAS: tl.constexpr, GRAV: tl.constexpr,
        MASS: tl.constexpr, LEN: tl.constexpr, MAXT: tl.constexpr, MAXS: tl.constexpr, DT: tl.constexpr,
        REW: tl.constexpr, EL: tl.constexpr, TERM: tl.constexpr, TRUNC: tl.constexpr, OBS: tl.constexpr,
        INFO: tl.constexpr, MASKED: tl.constexpr, NARROW: tl.constexpr, MAX_STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):  # fmt: skip
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = rows < n
        f = F + rows * WF
        live = tl.load(M + rows, mask=ok, other=0) != 0 if MASKED else ok
        th, thdot = tl.load(f + TH, mask=ok, other=0.0), tl.load(f + THD, mask=ok, other=0.0)
        torque0 = tl.load(f + TQ, mask=ok, other=0.0)
        has0 = tl.load(Bo + rows * WB + HAS, mask=ok, other=0) != 0
        g, m, length = (
            tl.load(f + GRAV, mask=ok, other=1.0),
            tl.load(f + MASS, mask=ok, other=1.0),
            tl.load(f + LEN, mask=ok, other=1.0),
        )
        max_torque, max_speed = tl.load(f + MAXT, mask=ok, other=1.0), tl.load(f + MAXS, mask=ok, other=1.0)
        dt = tl.load(f + DT, mask=ok, other=1.0)
        pi, turn, torque_weight, speed_weight = tl.load(K), tl.load(K + 1), tl.load(K + 2), tl.load(K + 3)
        u = tl.load(A + rows, mask=ok, other=0.0).to(tl.float64)
        bound = _r32(max_torque) if NARROW else max_torque
        u = tl.minimum(tl.maximum(u, -bound), bound)
        mod = libdevice.fmod(th + pi, turn)
        wrapped = tl.where((mod != 0) & (mod < 0), mod + turn, mod) - pi
        if NARROW:
            torque_cost = _r32(_r32(torque_weight) * _r32(u * u))
            drive = _r32(_r32(3.0 / (m * (length * length))) * u)
        else:
            torque_cost = torque_weight * (u * u)
            drive = 3.0 / (m * (length * length)) * u
        costs = wrapped * wrapped + speed_weight * (thdot * thdot) + torque_cost
        newthdot = thdot + (3 * g / (2 * length) * tl.sin(th) + drive) * dt
        newthdot = tl.minimum(tl.maximum(newthdot, -max_speed), max_speed)
        newth = th + newthdot * dt
        newth, newthdot = _keep(newth, th, live, MASKED), _keep(newthdot, thdot, live, MASKED)
        tl.store(f + TH, newth, mask=ok)
        tl.store(f + THD, newthdot, mask=ok)
        torque = _keep(u, torque0, live, MASKED)
        tl.store(f + TQ, torque, mask=ok)
        tl.store(Bo + rows * WB + HAS, _keep(ok, has0, live, MASKED), mask=ok)
        _finish(I, O, Bo, rows, ok, live, -costs, ok & ~ok, F, WF, WI, WO, WB, REW, EL, TERM, TRUNC, MASKED, MAX_STEPS)
        _observe(O, rows, ok, live, 0, tl.cos(newth), WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 1, tl.sin(newth), WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 2, newthdot, WO, OBS, INFO, MASKED)
        tl.store(SIM + rows * 3, newth, mask=ok)
        tl.store(SIM + rows * 3 + 1, newthdot, mask=ok)
        tl.store(SIM + rows * 3 + 2, torque, mask=ok)
        tl.store(R + rows * 2, newth.to(tl.float32), mask=ok)
        tl.store(R + rows * 2 + 1, torque.to(tl.float32), mask=ok)

    @triton.jit
    def _acrobot_dsdt(theta1, theta2, dtheta1, dtheta2, torque, m2, moi, half_pi, c):
        (d1_base, d1_lengths, d1_cross, d2_base, d2_cross, phi2_scale, phi1_v2, phi1_v12, phi1_g, dd2_v1, dd2_i) = c
        cos2, sin2 = tl.cos(theta2), tl.sin(theta2)
        d1 = d1_base + m2 * (d1_lengths + d1_cross * cos2) + moi + moi
        d2 = m2 * (d2_base + d2_cross * cos2) + moi
        phi2 = phi2_scale * tl.cos(theta1 + theta2 - half_pi)
        phi1 = (
            phi1_v2 * (dtheta2 * dtheta2) * sin2
            - phi1_v12 * dtheta2 * dtheta1 * sin2
            + phi1_g * tl.cos(theta1 - half_pi)
            + phi2
        )
        ddtheta2 = (torque + d2 / d1 * phi1 - dd2_v1 * (dtheta1 * dtheta1) * sin2 - phi2) / (dd2_i - (d2 * d2) / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2

    @triton.jit
    def _acrobot_wrap(x, pi, turn):
        for _ in tl.static_range(4):
            x = tl.where(x > pi, x - turn, x)
        for _ in tl.static_range(4):
            x = tl.where(x < -pi, x + turn, x)
        return x

    @triton.jit
    def _acrobot_kernel(
        F, I, O, Bo, A, M, SIM, R, K, n,
        WF: tl.constexpr, WI: tl.constexpr, WO: tl.constexpr, WB: tl.constexpr,
        T1: tl.constexpr, T2: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, L1: tl.constexpr,
        M1: tl.constexpr, M2: tl.constexpr, LC1: tl.constexpr, LC2: tl.constexpr, MOI: tl.constexpr,
        REW: tl.constexpr, EL: tl.constexpr, TERM: tl.constexpr, TRUNC: tl.constexpr, OBS: tl.constexpr,
        INFO: tl.constexpr, MASKED: tl.constexpr, NARROW: tl.constexpr, MAX_STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):  # fmt: skip
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = rows < n
        f = F + rows * WF
        live = tl.load(M + rows, mask=ok, other=0) != 0 if MASKED else ok
        y1, y2 = tl.load(f + T1, mask=ok, other=0.0), tl.load(f + T2, mask=ok, other=0.0)
        y3, y4 = tl.load(f + D1, mask=ok, other=0.0), tl.load(f + D2, mask=ok, other=0.0)
        l1, m1, m2 = (
            tl.load(f + L1, mask=ok, other=1.0),
            tl.load(f + M1, mask=ok, other=1.0),
            tl.load(f + M2, mask=ok, other=1.0),
        )
        lc1, lc2, moi = (
            tl.load(f + LC1, mask=ok, other=1.0),
            tl.load(f + LC2, mask=ok, other=1.0),
            tl.load(f + MOI, mask=ok, other=1.0),
        )
        pi, turn, half_pi, g = tl.load(K), tl.load(K + 1), tl.load(K + 2), tl.load(K + 3)
        half_dt, dt, sixth_dt = tl.load(K + 4), tl.load(K + 5), tl.load(K + 6)
        max_vel_1, max_vel_2 = tl.load(K + 7), tl.load(K + 8)
        a = tl.load(A + rows, mask=ok, other=0)
        # AVAIL_TORQUE[a] = [-1.0, 0.0, 1.0][a], including Python's negative indices.
        torque = ((a + 3) % 3 - 1).to(tl.float64)
        c = (
            m1 * (lc1 * lc1),
            l1 * l1 + lc2 * lc2,
            2 * l1 * lc2,
            lc2 * lc2,
            l1 * lc2,
            m2 * lc2 * g,
            -m2 * l1 * lc2,
            2 * m2 * l1 * lc2,
            (m1 * lc1 + m2 * l1) * g,
            m2 * l1 * lc2,
            m2 * (lc2 * lc2) + moi,
        )
        k11, k12, k13, k14 = _acrobot_dsdt(y1, y2, y3, y4, torque, m2, moi, half_pi, c)
        k21, k22, k23, k24 = _acrobot_dsdt(
            y1 + half_dt * k11, y2 + half_dt * k12, y3 + half_dt * k13, y4 + half_dt * k14, torque, m2, moi, half_pi, c
        )
        k31, k32, k33, k34 = _acrobot_dsdt(
            y1 + half_dt * k21, y2 + half_dt * k22, y3 + half_dt * k23, y4 + half_dt * k24, torque, m2, moi, half_pi, c
        )
        k41, k42, k43, k44 = _acrobot_dsdt(
            y1 + dt * k31, y2 + dt * k32, y3 + dt * k33, y4 + dt * k34, torque, m2, moi, half_pi, c
        )
        theta1 = _acrobot_wrap(y1 + sixth_dt * (k11 + 2 * k21 + 2 * k31 + k41), pi, turn)
        theta2 = _acrobot_wrap(y2 + sixth_dt * (k12 + 2 * k22 + 2 * k32 + k42), pi, turn)
        dtheta1 = y3 + sixth_dt * (k13 + 2 * k23 + 2 * k33 + k43)
        dtheta2 = y4 + sixth_dt * (k14 + 2 * k24 + 2 * k34 + k44)
        dtheta1 = tl.minimum(tl.maximum(dtheta1, -max_vel_1), max_vel_1)
        dtheta2 = tl.minimum(tl.maximum(dtheta2, -max_vel_2), max_vel_2)
        cos1 = tl.cos(theta1)
        term = -cos1 - tl.cos(theta2 + theta1) > 1.0
        reward = tl.where(term, 0.0, -1.0).to(tl.float64)
        theta1, theta2 = _keep(theta1, y1, live, MASKED), _keep(theta2, y2, live, MASKED)
        dtheta1, dtheta2 = _keep(dtheta1, y3, live, MASKED), _keep(dtheta2, y4, live, MASKED)
        tl.store(f + T1, theta1, mask=ok)
        tl.store(f + T2, theta2, mask=ok)
        tl.store(f + D1, dtheta1, mask=ok)
        tl.store(f + D2, dtheta2, mask=ok)
        _finish(I, O, Bo, rows, ok, live, reward, term, F, WF, WI, WO, WB, REW, EL, TERM, TRUNC, MASKED, MAX_STEPS)
        _observe(O, rows, ok, live, 0, tl.cos(theta1), WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 1, tl.sin(theta1), WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 2, tl.cos(theta2), WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 3, tl.sin(theta2), WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 4, dtheta1, WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 5, dtheta2, WO, OBS, INFO, MASKED)
        tl.store(SIM + rows * 4, theta1, mask=ok)
        tl.store(SIM + rows * 4 + 1, theta2, mask=ok)
        tl.store(SIM + rows * 4 + 2, dtheta1, mask=ok)
        tl.store(SIM + rows * 4 + 3, dtheta2, mask=ok)
        tl.store(R + rows * 2, theta1.to(tl.float32), mask=ok)
        tl.store(R + rows * 2 + 1, theta2.to(tl.float32), mask=ok)

    @triton.jit
    def _mountain_car_kernel(
        F, I, O, Bo, A, M, SIM, R, K, n,
        WF: tl.constexpr, WI: tl.constexpr, WO: tl.constexpr, WB: tl.constexpr,
        POS: tl.constexpr, VEL: tl.constexpr, GRAV: tl.constexpr, FORCE: tl.constexpr, MAXS: tl.constexpr,
        REW: tl.constexpr, EL: tl.constexpr, TERM: tl.constexpr, TRUNC: tl.constexpr, OBS: tl.constexpr,
        INFO: tl.constexpr, MASKED: tl.constexpr, NARROW: tl.constexpr, MAX_STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):  # fmt: skip
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = rows < n
        f = F + rows * WF
        live = tl.load(M + rows, mask=ok, other=0) != 0 if MASKED else ok
        position0, velocity0 = tl.load(f + POS, mask=ok, other=0.0), tl.load(f + VEL, mask=ok, other=0.0)
        gravity, force = tl.load(f + GRAV, mask=ok, other=0.0), tl.load(f + FORCE, mask=ok, other=0.0)
        max_speed = tl.load(f + MAXS, mask=ok, other=0.0)
        left, right = -tl.load(K), tl.load(K + 1)
        a = tl.load(A + rows, mask=ok, other=0)
        velocity = velocity0 + ((a - 1).to(tl.float64) * force + tl.cos(3 * position0) * (-gravity))
        velocity = tl.minimum(tl.maximum(velocity, -max_speed), max_speed)
        position = tl.minimum(tl.maximum(position0 + velocity, left), right)
        velocity = tl.where((position == left) & (velocity < 0), 0.0, velocity)
        term = (position >= 0.5) & (velocity >= 0)
        position, velocity = _keep(position, position0, live, MASKED), _keep(velocity, velocity0, live, MASKED)
        tl.store(f + POS, position, mask=ok)
        tl.store(f + VEL, velocity, mask=ok)
        reward = tl.full([BLOCK], -1.0, tl.float64)
        _finish(I, O, Bo, rows, ok, live, reward, term, F, WF, WI, WO, WB, REW, EL, TERM, TRUNC, MASKED, MAX_STEPS)
        _observe(O, rows, ok, live, 0, position, WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 1, velocity, WO, OBS, INFO, MASKED)
        tl.store(SIM + rows * 2, position, mask=ok)
        tl.store(SIM + rows * 2 + 1, velocity, mask=ok)
        tl.store(R + rows, position.to(tl.float32), mask=ok)

    @triton.jit
    def _mountain_car_continuous_kernel(
        F, I, O, Bo, A, M, SIM, R, K, n,
        WF: tl.constexpr, WI: tl.constexpr, WO: tl.constexpr, WB: tl.constexpr,
        POS: tl.constexpr, VEL: tl.constexpr, IS32: tl.constexpr, POWER: tl.constexpr, MAXS: tl.constexpr,
        REW: tl.constexpr, EL: tl.constexpr, TERM: tl.constexpr, TRUNC: tl.constexpr, OBS: tl.constexpr,
        INFO: tl.constexpr, MASKED: tl.constexpr, NARROW: tl.constexpr, MAX_STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):  # fmt: skip
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = rows < n
        f = F + rows * WF
        live = tl.load(M + rows, mask=ok, other=0) != 0 if MASKED else ok
        position0, velocity0 = tl.load(f + POS, mask=ok, other=0.0), tl.load(f + VEL, mask=ok, other=0.0)
        is32 = tl.load(Bo + rows * WB + IS32, mask=ok, other=0) != 0
        power, max_speed = tl.load(f + POWER, mask=ok, other=0.0), tl.load(f + MAXS, mask=ok, other=0.0)
        hill, right_bound, left_bound, goal, weight = (
            tl.load(K),
            tl.load(K + 1),
            -tl.load(K + 2),
            tl.load(K + 3),
            tl.load(K + 4),
        )
        right32, left32, goal32 = tl.load(K + 5), tl.load(K + 6), tl.load(K + 7)
        a = tl.load(A + rows, mask=ok, other=0.0).to(tl.float64)
        cosine = hill * tl.cos(tl.where(is32, _r32(3 * position0), 3 * position0))
        low, high = a < -1.0, a > 1.0
        clipped = low | high
        clipped_term = tl.where(low, -power, power) - cosine
        if NARROW:
            product = _r32(a * _r32(power))
            action_term = _r32(product - _r32(cosine))
            wide = ~is32
        else:
            action_term = a * power - cosine
            wide = ~is32 | ~clipped
        narrow_velocity = _r32(velocity0 + tl.where(clipped, _r32(clipped_term), action_term))
        velocity = tl.where(wide, velocity0 + tl.where(clipped, clipped_term, action_term), narrow_velocity)
        limit32 = _r32(max_speed)
        over = tl.where(wide, velocity > max_speed, velocity > limit32)
        velocity = tl.where(over, max_speed, velocity)
        under = ~over & tl.where(wide, velocity < -max_speed, velocity < -limit32)
        velocity = tl.where(under, -max_speed, velocity)
        wide = ~is32 | (wide & ~over & ~under)
        position = tl.where(wide, position0 + velocity, _r32(position0 + _r32(velocity)))
        right = tl.where(wide, position > right_bound, position > right32)
        position = tl.where(right, right_bound, position)
        left = ~right & tl.where(wide, position < left_bound, position < left32)
        position = tl.where(left, left_bound, position)
        wide = wide | right | left
        at_wall = tl.where(wide, position == left_bound, position == left32)
        velocity = tl.where(at_wall & (velocity < 0), 0.0, velocity)
        term = tl.where(wide, position >= goal, position >= goal32) & (velocity >= 0)
        reward = tl.where(term, 100.0, 0.0).to(tl.float64) - (a * a) * weight
        position, velocity = _r32(position), _r32(velocity)
        position, velocity = _keep(position, position0, live, MASKED), _keep(velocity, velocity0, live, MASKED)
        tl.store(f + POS, position, mask=ok)
        tl.store(f + VEL, velocity, mask=ok)
        tl.store(Bo + rows * WB + IS32, _keep(ok, is32, live, MASKED), mask=ok)
        _finish(I, O, Bo, rows, ok, live, reward, term, F, WF, WI, WO, WB, REW, EL, TERM, TRUNC, MASKED, MAX_STEPS)
        _observe(O, rows, ok, live, 0, position, WO, OBS, INFO, MASKED)
        _observe(O, rows, ok, live, 1, velocity, WO, OBS, INFO, MASKED)
        tl.store(SIM + rows * 2, position, mask=ok)
        tl.store(SIM + rows * 2 + 1, velocity, mask=ok)
        tl.store(R + rows, position.to(tl.float32), mask=ok)

    KERNELS = {
        "cartpole": _cartpole_kernel,
        "pendulum": _pendulum_kernel,
        "acrobot": _acrobot_kernel,
        "mountain_car": _mountain_car_kernel,
        "mountain_car_continuous": _mountain_car_continuous_kernel,
    }


def launcher(domain: str, columns: dict, widths, num_envs: int, device: torch.device, max_steps: int):
    """Return ``launch(blocks, actions, active, sim, render, masked, narrow)`` for the domain kernel."""
    constants = torch.tensor(CONSTANTS[domain], dtype=torch.float64, device=device)
    fixed = {name: columns[key][1] for name, key in COLUMNS[domain].items()}
    fixed.update(
        REW=columns["reward"][1],
        EL=columns[("lifecycle", "elapsed_steps")][1],
        TERM=columns["terminated"][1],
        TRUNC=columns["truncated"][1],
        OBS=columns["observation"][1],
        INFO=columns["info_state"][1],
        WF=widths[0],
        WI=widths[1],
        WO=widths[2],
        WB=widths[3],
        MAX_STEPS=max_steps,
        BLOCK=128,
    )
    kernel = KERNELS[domain]
    grid = (triton.cdiv(num_envs, 128),)

    def launch(blocks, actions, active, sim, render, masked: bool, narrow: bool):
        kernel[grid](
            *blocks, actions, active, sim, render, constants, num_envs,
            MASKED=masked, NARROW=narrow, enable_fp_fusion=False, **fixed,
        )  # fmt: skip

    return launch

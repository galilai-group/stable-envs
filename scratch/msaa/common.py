"""Shared pieces for the anti-aliasing question: can a small CNN turn an ss=1 render into an ss=2 one?

The problem is NOT the one ``scratch/altrender/rendernet`` solved. There the input was a cheap render
at a lower resolution with a texture missing, and the corrector had to invent detail. Here the input
and the target are the SAME renderer, at the SAME resolution, on the SAME state -- ``env.render`` with
``supersample=1`` against ``env.render`` with ``supersample=2``. The only difference is how many rays
were averaged into each pixel, so nothing is absent from the input in the way a texture was, and the
residual is edge-localised by construction.

That changes three things about the harness, and they are why this module exists rather than importing
``rendernet`` wholesale:

* **The bilinear skip becomes an identity.** ``F.interpolate(x, size=x.shape[-2:])`` is a copy, so a
  zero-initialised head starts the model EXACTLY at the ss=1 frame, byte for byte -- a stronger anchor
  than rendernet's, where step 0 was already an approximation.
* **The metric that decides the question is interior-exactness against the OFFICIAL frame.** These are
  parity cells: ``test_render_parity_with_official_renderer`` asserts every interior pixel is
  bit-identical to ``mujoco.Renderer``. A corrector that lowers mean |diff| and moves one interior
  pixel has broken the claim. So every evaluation here carries the official frame as a third tensor,
  and ``pixel_agreement`` is called with the OFFICIAL frame as the reference (its ``interior_mask`` is
  computed on the second argument, and computing it on ours would be measuring our own flatness).
* **The states are drawn, not pooled.** ``sample_states`` on both families rolls out fresh episodes
  from a generator, so two different seeds give two genuinely disjoint state sets and the held-out
  split is a split of the generator rather than of a fixed pool.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ROOT = "/mnt/vast/home/randall.balestriero/data/msaanet"
LOGS = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/msaanet"
CKPT = os.path.join(LOGS, "ckpt")

#: The seed ``tests/gpu/test_conformance.py`` draws its parity states from. The ``decl`` split here is
#: that exact draw, so a number reported on it is comparable to the assertion the cell must pass.
PARITY_STATE_SEED = 20260811
#: Training and held-out draws. Disjoint from each other and from the parity seed; because
#: ``sample_states`` rolls out fresh episodes with random actions, two seeds give two continuous state
#: sets whose intersection is empty with probability one -- asserted rather than assumed by
#: ``gen_triples.py``, which checks that no held-out state matches a training state to 1e-6.
SEED_TRAIN = 11
SEED_HELDOUT = 4001

#: The cells. Three DM Control parity cells spanning the family's floor/kernel-share range, and the
#: two gym MuJoCo parity cells furthest from their stored-frame bar.
CELLS = ("DMCCartpoleBalance", "DMCReacherEasy", "DMCFingerSpin", "MujocoPusher", "MujocoReacher")


def make(name: str, batch: int = 1, size: int = 256, seed: int = 0):
    import stable_envs as se
    return se.make(name, batch=batch, size=size, device="cuda", seed=seed)


def cls_of(name: str):
    import stable_envs as se
    return se.REGISTRY[name]


def sample_states(name: str, n: int, seed: int, size: int = 256) -> np.ndarray:
    """``(n, state_dim)`` from the env's own visited distribution at a named seed.

    One env instance, one generator. ``sample_states`` loads a fresh official env per call and rolls
    out ``n`` episodes, so this is the slow half of data generation on the DM Control cells and it is
    a pure function of ``seed``.
    """
    env = make(name, batch=1, size=size, seed=seed)
    st = env.sample_states(int(n), rng=np.random.default_rng(seed))
    return np.asarray(st, np.float32)


def official_frames(name: str, states: np.ndarray, size: int, chunk: int = 256,
                    log_every: int = 1024) -> np.ndarray:
    """``mujoco.Renderer``'s own frames, through the env class's ``reference_render``.

    Not reimplemented here: ``reference_render`` asserts the state round-trip, and that assertion is
    the only thing standing between "the reference did not reach this state" and a plausible-looking
    parity number computed from two different scenes.
    """
    import time
    cls = cls_of(name)
    ref = cls.reference_env(int(size))
    out = np.empty((len(states), int(size), int(size), 3), np.uint8)
    t0 = time.perf_counter()
    for lo in range(0, len(states), chunk):
        hi = min(lo + chunk, len(states))
        out[lo:hi] = cls.reference_render(ref, states[lo:hi])
        if log_every and (hi % log_every == 0 or hi == len(states)):
            print(f"  official {hi}/{len(states)}  {hi / (time.perf_counter() - t0):.1f} f/s",
                  flush=True)
    return out


def our_frames(name: str, states: np.ndarray, size: int, ss: int, chunk: int = 256,
               env=None) -> np.ndarray:
    """Our renderer at a chosen supersample. ``chunk`` bounds host memory, not device memory."""
    import torch
    e = env or make(name, batch=1, size=size)
    out = np.empty((len(states), int(size), int(size), 3), np.uint8)
    for lo in range(0, len(states), chunk):
        hi = min(lo + chunk, len(states))
        f = e.render(torch.as_tensor(states[lo:hi], device="cuda"), size=int(size), supersample=int(ss))
        out[lo:hi] = f.cpu().numpy()
    return out


# ------------------------------------------------------------------------------------- metrics

def agreement(ours: np.ndarray, official: np.ndarray, block: int = 64) -> dict:
    """The repo's own ``pixel_agreement`` plus SSIM and the error tail, in blocks of ``block``.

    Blocked for the reason ``rendernet``'s README records: ``testing.conformance.ssim`` runs a 3-D
    uniform filter over the whole ``(N, H, W)`` stack and therefore smooths along the FRAME axis, so
    SSIM over 2048 frames is a different quantity from SSIM over the 64 a declaration is measured at.
    ``pixel_agreement`` is per-pixel and blocking it changes nothing; it is blocked anyway so that
    every column of a row comes from the same partition of the same frames.
    """
    from stable_envs.testing import conformance as C
    keys = ("mean_abs", "exact_frac", "within8_frac", "interior_exact_frac", "interior_mean_abs",
            "interior_frac")
    acc = {k: [] for k in keys}
    acc["ssim"] = []
    n_int = n_bad = 0
    mx = 0
    for lo in range(0, len(ours), block):
        a, b = ours[lo:lo + block], official[lo:lo + block]
        r = C.pixel_agreement(a, b)
        for k in keys:
            acc[k].append(r[k])
        acc["ssim"].append(C.ssim(a, b))
        mx = max(mx, int(r["max_abs"]))
        n_int += int(r["interior_px"] or 0)
        n_bad += int(r["interior_bad_px"] or 0)
    out = {k: (None if any(v is None for v in acc[k]) else float(np.mean(acc[k]))) for k in acc}
    out["max_abs"] = mx
    out["interior_px"] = n_int
    out["interior_bad_px"] = n_bad
    # Interior-exactness over the WHOLE evaluation, not the mean of per-block fractions: the decision
    # a cell has to make is "did any interior pixel move", and that is a count.
    out["interior_exact_frac_pooled"] = None if n_int == 0 else 1.0 - n_bad / n_int
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    out["p999"] = float(np.percentile(d, 99.9))
    out["p9999"] = float(np.percentile(d, 99.99))
    out["n_frames"] = int(len(ours))
    return out


def loadavg() -> list[float]:
    return list(os.getloadavg())

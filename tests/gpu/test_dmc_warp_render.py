"""DM Control drawn with MuJoCo Warp's ray tracer (``backend="mjwarp"``) on CUDA.

The frames are finite and deterministic, stay under a per-cell MSE bound against MuJoCo OpenGL (run
with ``MUJOCO_GL=egl``), draw every look leaf of the appearance gate where the analytic renderer draws
it, and equal the frames of the State when they come from the live MuJoCo Warp physics.
``SE_DMC_WARP_REPORT=path`` appends one JSON line of raw MSE per cell to ``path``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.testing.conformance import reference_env
from tests.gpu._shared import family_params, resolve_family

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]
DMC = [
    name
    for name, cls in se.registry.REGISTRY.items()
    if cls.family == "dmcontrol" and getattr(cls, "DOMAIN", None) != "manipulator"
]
#: Bound on the mean frame MSE (0-255 scale) against MuJoCo OpenGL; cells not listed use DEFAULT_BOUND.
MSE_BOUND: dict[str, float] = {}
DEFAULT_BOUND = 40.0
FRAMES = 16


def _make(name, n, **kw):
    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MuJoCo Warp rendering")
    return se.make(name, num_envs=n, device="cuda", **kw)


def _rows(env, n, seed=7):
    return torch.as_tensor(env.sample_states(n, np.random.default_rng(seed)), device="cuda")


def _mse(a, b) -> float:
    a = torch.as_tensor(a).double().cpu()
    b = torch.as_tensor(b).double().cpu()
    return float(((a - b) ** 2).mean())


@pytest.mark.parametrize("name", family_params(DMC, family="dmc"))
def test_frames_are_deterministic_and_semantics_finite(name):
    with _make(name, 4, backend="mjwarp") as env:
        rows = _rows(env, 4)
        first = env._render(rows).clone()
        second = env._render(rows)
        assert first.dtype == torch.uint8 and first.shape == (4, env.size, env.size, 3)
        assert torch.equal(first, second)
        assert float(first.float().std()) > 0
        seg, depth = env.render_semantic(rows)
        assert seg.shape == (4, env.size, env.size, 2) and depth.shape == (4, env.size, env.size)
        hit = seg[..., 0] >= 0
        assert bool(hit.any()) and bool(torch.isfinite(depth[hit]).all()) and bool((depth[hit] > 0).all())


@pytest.mark.parametrize("name", family_params(DMC, family="dmc"))
def test_mse_against_mujoco_opengl(name):
    with _make(name, FRAMES, backend="mjwarp") as env:
        rows = _rows(env, FRAMES)
        ours = env._render(rows).cpu().numpy()
        cls = type(env)
        ref = cls.reference_render(reference_env(cls, env.size), rows.double().cpu().numpy())
    with _make(name, FRAMES) as current:
        default = current._render(rows).cpu().numpy()
    got = {"cell": name, "mse_warp": _mse(ours, ref), "mse_default": _mse(default, ref), "frames": FRAMES}
    per_frame = ((ours.astype(np.float64) - ref.astype(np.float64)) ** 2).reshape(FRAMES, -1).mean(1)
    got["mse_warp_max"] = float(per_frame.max())
    if os.environ.get("SE_DMC_WARP_REPORT"):
        with open(os.environ["SE_DMC_WARP_REPORT"], "a") as f:
            f.write(json.dumps(got) + "\n")
    assert got["mse_warp"] <= MSE_BOUND.get(name, DEFAULT_BOUND), got


def _leaf_cases():
    from tests.helpers.reference.gates.dmc_appearance import CASES, FLOOR_CASES

    out = []
    for cases in (CASES, FLOOR_CASES):
        for cell, rows in cases.items():
            out += [pytest.param(cell, case, id=f"{cell}-{'-'.join(case)}-{i}") for i, case in enumerate(rows)]
    out += [
        pytest.param("DMCCartpoleSwingup", {"floor.color": [[0.9, 0.1, 0.1], [0.1, 0.8, 0.2]]}, id="cartpole-floor")
    ]
    out += [pytest.param("DMCCartpoleSwingup", {"camera.fovy_scale": 1.3}, id="cartpole-fovy")]
    return out


def _style(env, case):
    """One style row: our own leaves directly, the others through the stable_worldmodel variation map."""
    from tests.helpers.reference import style_from_variation

    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    theirs = {k: v for k, v in case.items() if k not in layout}
    row = np.asarray(env.default_style(), np.float32).reshape(-1).copy()
    if theirs:
        row = style_from_variation(env, theirs, base=row)
    for k, v in case.items():
        if k in layout:
            o, w = layout[k]
            row[o : o + w] = np.asarray(v, np.float32).reshape(-1)[:w]
    return torch.as_tensor(row, device="cuda")


@pytest.mark.parametrize(("name", "case"), _leaf_cases())
def test_every_look_leaf_is_drawn(name, case):
    """Where the analytic renderer's frames change with the leaf, MuJoCo Warp's change on the same pixels."""
    with _make(name, 8, backend="mjwarp") as warp, _make(name, 8) as analytic:
        rows = _rows(warp, 8, seed=11)
        style = _style(warp, case)
        if torch.equal(style, torch.as_tensor(warp.default_style(), device="cuda")):
            pytest.skip(f"{case} is the cell's default appearance")
        moved = {}
        for label, env in (("warp", warp), ("analytic", analytic)):
            base = env._render(rows).clone()
            leaf = env._render(rows, style=style)
            assert torch.isfinite(leaf.float()).all()
            moved[label] = (base != leaf).any(-1)
    w, a = moved["warp"], moved["analytic"]
    if bool(a.any()):
        iou = float((w & a).sum()) / float((w | a).sum())
        assert iou >= 0.3, (case, iou, int(w.sum()), int(a.sum()))
    else:
        assert float(w.float().mean()) <= 0.005, (case, int(w.sum()))


LIVE = ["DMCCartpoleSwingup", "DMCBallInCupCatch", "DMCFingerTurnHard", "DMCReacher", "DMCCheetahRun"]


@pytest.mark.parametrize("name", family_params([n for n in LIVE if n in DMC], family="dmc"))
def test_live_frames_equal_state_frames(name):
    with _make(name, 6, backend="mjwarp", simulation_backend="mjwarp") as env:
        assert env._render_live is not None
        env.reset(seed=3)
        rng = np.random.default_rng(5)
        for _ in range(3):
            actions = torch.as_tensor(rng.uniform(-1, 1, (6, env.action_dim)), device="cuda")
            out = env.step(actions, states=None, frames=slice(-1, None))
            live = out["frames"][:, 0]
            state = env.get_state()
            drawn = env.render(state)
            again = env.render()
            for frames in (live, again):
                diff = (frames.int() - drawn.int()).abs()
                assert float((diff > 2).float().mean()) <= 0.005, (name, int(diff.max()))


def test_speed_options_keep_the_frames_and_fidelity_options_draw():
    name = "DMCCartpoleSwingup"
    if name not in DMC:
        pytest.skip(f"{name} is not registered here")
    with _make(name, 5, backend="mjwarp") as env:
        rows = _rows(env, 5)
        reference = env._render(rows).clone()
    for speed in ({"graph": False}, {"bvh": "rebuild"}, {"worlds": 2}):
        with _make(name, 5, backend="mjwarp", render_options=speed) as env:
            assert torch.equal(env._render(rows), reference), speed
    for fidelity in ({"shadows": "one"}, {"shadows": "none"}, {"samples": 2}, {"reflection": False}, {"sky": False}):
        with _make(name, 5, backend="mjwarp", render_options=fidelity) as env:
            frames = env._render(rows)
            assert frames.shape == reference.shape and float(frames.float().std()) > 0, fidelity

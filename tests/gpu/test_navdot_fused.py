"""The fused swm_nav kernel draws what the torch path draws, on the envs themselves.

The pattern is ``tests/gpu/test_fast_paths.py``'s: a fast path is only safe to switch on by default
if something renders the same scene BOTH ways and diffs them, on the registered envs rather than on
a synthetic case. Here the two ways are ``navdot.blend_dots`` and the ``_alpha_blend``/
``_gaussian_dot`` composition it replaces, and the switch is ``navdot.available``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs import swm_nav as SN
from stable_envs.render import navdot

CONFIGS = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not navdot.HAVE_TRITON,
                                reason="the fused path needs a CUDA device and Triton")


def _torch_path(env, base, s, sl, k):
    img = base
    pos, sig, col = env._dots(s, sl, k)
    for d in range(pos.shape[1]):
        img = SN._alpha_blend(img, SN._gaussian_dot(env._gy, env._gx, pos[:, d], sig[:, d]),
                              col[:, d])
    return img


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("B", [1, 7, 64])
def test_fused_matches_the_torch_path_on_sampled_states(cfg, B):
    env = se.make(cfg, batch=B, device="cuda")
    env.rng = np.random.default_rng(0)
    st = torch.as_tensor(np.asarray(env.sample_states(B), np.float32), device="cuda")
    sty = env.resolve_style(None, None, batch=B)
    sl = lambda p: sty[p][0:B]                                             # noqa: E731
    base = env._base_image(sty, 0, B).expand(B, SN.IMG, SN.IMG, 3).contiguous()
    want = _torch_path(env, base, st, sl, B)
    pos, sig, col = env._dots(st, sl, B)
    got = navdot.blend_dots(base.clone(), pos, sig, col)
    assert torch.equal(want, got), f"{cfg} B={B}: the fused kernel and the torch path disagree"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_fused_matches_on_states_the_env_would_never_sample(cfg):
    """Dots off the frame, on and between pixel centres, and stacked on each other."""
    env = se.make(cfg, batch=1, device="cuda")
    pos = [(112.0, 112.0), (112.5, 112.5), (0.0, 0.0), (223.0, 223.0), (-60.0, 112.0),
           (112.0, -60.0), (280.0, 280.0), (-3000.0, 4000.0)]
    dim = env.state_dim
    st = np.zeros((len(pos), dim), np.float32)
    for i, (x, y) in enumerate(pos):
        st[i, 0], st[i, 1] = x, y
        if dim >= 4:
            st[i, 2], st[i, 3] = (x, y) if i % 2 else (223.0 - x, 223.0 - y)
    e = se.make(cfg, batch=len(pos), device="cuda")
    s = torch.as_tensor(st, device="cuda")
    sty = e.resolve_style(None, None, batch=len(pos))
    sl = lambda p: sty[p][0:len(pos)]                                      # noqa: E731
    base = e._base_image(sty, 0, len(pos)).expand(len(pos), SN.IMG, SN.IMG, 3).contiguous()
    want = _torch_path(e, base, s, sl, len(pos))
    p_, sg, cl = e._dots(s, sl, len(pos))
    assert torch.equal(want, navdot.blend_dots(base.clone(), p_, sg, cl)), cfg


@pytest.mark.parametrize("cfg", CONFIGS)
def test_render_is_the_fused_path_and_is_still_invariant(cfg):
    """``render`` takes the kernel here, and the contracts hold across it."""
    env = se.make(cfg, batch=8, device="cuda")
    assert navdot.available(env.device), "this test is not exercising the fused path"
    env.rng = np.random.default_rng(1)
    st = np.asarray(env.sample_states(8), np.float32)
    first = env.render(st).clone()
    assert torch.equal(env.render(st), first), f"{cfg}: render is not pure"
    perm = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    assert torch.equal(env.render(np.ascontiguousarray(st[perm]))[np.argsort(perm)], first), (
        f"{cfg}: render is not order invariant")
    solo = se.make(cfg, batch=1, device="cuda")
    for i in (0, 3, 7):
        assert torch.equal(solo.render(st[i:i + 1])[0], first[i]), f"{cfg}: batch-1 row {i}"

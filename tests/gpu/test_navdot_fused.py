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
from stable_envs.envs.navigation import envs as SN
from stable_envs.envs.navigation import navdot

CONFIGS = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not navdot.HAVE_TRITON, reason="the fused path needs a CUDA device and Triton"
)


def _torch_path(env, base, s, sl, k):
    img = base
    pos, sig, col = env._dots(s, sl, k)
    for d in range(pos.shape[1]):
        img = SN._alpha_blend(img, SN._gaussian_dot(env._gy, env._gx, pos[:, d], sig[:, d]), col[:, d])
    return img


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("B", [1, 7, 64])
def test_fused_matches_the_torch_path_on_sampled_states(cfg, B):
    env = se.make(cfg, num_envs=B, device="cuda")
    env.rng = np.random.default_rng(0)
    st = torch.as_tensor(np.asarray(env.sample_states(B), np.float32), device="cuda")
    sty = env.resolve_style(None, None, batch=B)
    sl = lambda p: sty[p][0:B]  # noqa: E731
    base = env._base_image(sty, 0, B).expand(B, SN.IMG, SN.IMG, 3).contiguous()
    want = _torch_path(env, base, st, sl, B)
    pos, sig, col = env._dots(st, sl, B)
    got = navdot.blend_dots(base.clone(), pos, sig, col)
    assert torch.equal(want, got), f"{cfg} B={B}: the fused kernel and the torch path disagree"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_fused_matches_on_states_the_env_would_never_sample(cfg):
    """Dots off the frame, on and between pixel centres, and stacked on each other."""
    env = se.make(cfg, num_envs=1, device="cuda")
    pos = [
        (112.0, 112.0),
        (112.5, 112.5),
        (0.0, 0.0),
        (223.0, 223.0),
        (-60.0, 112.0),
        (112.0, -60.0),
        (280.0, 280.0),
        (-3000.0, 4000.0),
    ]
    dim = env.state_dim
    st = np.zeros((len(pos), dim), np.float32)
    for i, (x, y) in enumerate(pos):
        st[i, 0], st[i, 1] = x, y
        if dim >= 4:
            st[i, 2], st[i, 3] = (x, y) if i % 2 else (223.0 - x, 223.0 - y)
    e = se.make(cfg, num_envs=len(pos), device="cuda")
    s = torch.as_tensor(st, device="cuda")
    sty = e.resolve_style(None, None, batch=len(pos))
    sl = lambda p: sty[p][0 : len(pos)]  # noqa: E731
    base = e._base_image(sty, 0, len(pos)).expand(len(pos), SN.IMG, SN.IMG, 3).contiguous()
    want = _torch_path(e, base, s, sl, len(pos))
    p_, sg, cl = e._dots(s, sl, len(pos))
    assert torch.equal(want, navdot.blend_dots(base.clone(), p_, sg, cl)), cfg


@pytest.mark.parametrize("cfg", CONFIGS)
def test_per_row_styles_against_the_class_s_own_torch_path(cfg):
    """Every row a different dot colour and radius, checked against ``env.blend_dots``.

    Two things this asserts that the tests above do not. First the REFERENCE is the class's own
    method rather than a composition spelled out here, so if a subclass ever changes the order it
    blends its dots in, or adds one, this fails instead of quietly comparing the kernel against a
    copy of the old order. TwoRoomTarget blends target then agent and each blend quantises to uint8,
    so a swap is a different picture wherever the two overlap — the frozen replay pins it and the
    scratch fuzz pins it, and neither is run by pytest.

    Second, per-row styles: the kernel indexes ``pos``, ``den``, ``max`` and ``colour`` by row, and a
    kernel that read row 0 for the whole batch would pass every uniform-style test in this file.
    """
    B = 6
    env = se.make(cfg, num_envs=B, device="cuda")
    env.rng = np.random.default_rng(5)
    st = torch.as_tensor(np.asarray(env.sample_states(B), np.float32), device="cuda")

    rows = np.tile(np.asarray(env.default_style(), np.float32), (B, 1))
    lay = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    rng = np.random.default_rng(6)
    for path in ("agent.color", "agent.radius", "target.color", "target.radius"):
        if path not in lay:
            continue
        o, w = lay[path]
        rows[:, o : o + w] = rng.uniform(0, 255, (B, w)) if path.endswith("color") else rng.uniform(7.0, 14.0, (B, w))
    sty = env.resolve_style(rows, None, batch=B)
    sl = lambda p: sty[p][0:B]  # noqa: E731
    base = env._base_image(sty, 0, B).expand(B, SN.IMG, SN.IMG, 3).contiguous()

    want = env.blend_dots(base, st, sl, B)  # the class's own torch path, in its order
    pos, sig, col = env._dots(st, sl, B)
    got = navdot.blend_dots(base.clone(), pos, sig, col)
    assert torch.equal(want, got), f"{cfg}: per-row styles through the fused kernel"

    # no two rows may have collapsed onto one another's style
    flat = got.reshape(B, -1)
    assert len({tuple(flat[i].tolist()) for i in range(B)}) == B, (
        f"{cfg}: {B} different dot styles produced fewer than {B} different frames"
    )

    # and through render(), which is what a caller reaches
    assert torch.equal(env._render(st.cpu().numpy(), style=rows), want), (
        f"{cfg}: render() with per-row styles disagrees with the torch path"
    )


@pytest.mark.parametrize("cfg", CONFIGS)
def test_render_is_the_fused_path_and_is_still_invariant(cfg):
    """``render`` takes the kernel here, and the contracts hold across it."""
    env = se.make(cfg, num_envs=8, device="cuda")
    assert navdot.available(env.device), "this test is not exercising the fused path"
    st = env.reset(seed=1)
    first = env.render(st).clone()
    assert torch.equal(env.render(st), first), f"{cfg}: render is not pure"
    perm = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    assert torch.equal(env.render(st[perm])[np.argsort(perm)], first), (
        f"{cfg}: render is not order invariant"
    )
    solo = se.make(cfg, num_envs=1, device="cuda")
    for i in (0, 3, 7):
        assert torch.equal(solo.render(st[i : i + 1])[0], first[i]), f"{cfg}: batch-1 row {i}"


def test_blend_addresses_frames_beyond_two_gib():
    # Launch only the first 8x8 tile of each frame; the last row starts beyond int32.
    batch, height, width = 684, 1024, 1024
    if torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        pytest.skip("64-bit address regression needs 5 GiB of free GPU memory")
    base = torch.empty((batch, height, width, 3), dtype=torch.uint8, device="cuda")
    out = torch.empty_like(base)
    colours = (torch.arange(batch, device="cuda") % 251).to(torch.uint8)
    base[:, :8, :8] = colours[:, None, None, None]
    dummy = torch.empty(1, device="cuda")
    navdot._blend_kernel[(batch, 1)](
        base,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        out,
        H=height,
        W=width,
        ND=0,
        NTX=width // 8,
        BH=8,
        BW=8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    assert (batch - 1) * height * width * 3 > 2**31
    assert torch.equal(out[:, :8, :8], base[:, :8, :8])

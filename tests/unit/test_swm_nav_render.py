"""swm_nav still draws what it drew before Step 6 baked its background and cropped its dots.

Two optimisations live behind ``render()`` for TwoRoom, TwoRoomTarget, Piecewise and PiecewiseZones,
and each is a claim that some work could not have reached a pixel:

  * ``_base_image`` builds the background, walls, doors, zones and border ONCE per distinct style row
    and expands or gathers it back, instead of once per row of the batch. The claim is that the base
    image is a function of the style and of nothing else -- not of the state, not of what was
    rendered before.
  * ``_blend_dot`` evaluates the Gaussian and the alpha blend only inside ``_dot_window``. The claim
    is that outside it the normalised alpha is below ``_ALPHA_FLOOR``, where the blend is the
    identity.

Both are checked here against the arithmetic they replaced -- the full-frame
``_alpha_blend(img, _gaussian_dot(whole grid), colour)`` composition -- on the cases that make the
claims hard rather than on the ones the envs happen to produce. The adversarial one is an agent
OUTSIDE the frame: ``_gaussian_dot`` divides by the frame's own maximum, so a dot whose centre is off
the picture is rescaled until its largest in-frame sample is 1.0, and a window derived from the
un-normalised tail alone would crop a dot that is painting at full strength.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs import navigation as SN

CONFIGS = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")


def _reference(env, states, sty):
    """The pre-Step-6 render: base image per row, both dots over the whole frame."""
    s = torch.as_tensor(np.asarray(states, np.float32), device=env.device).reshape(-1, env.state_dim)
    k = int(s.shape[0])
    sl = lambda p: sty[p][0:k]  # noqa: E731
    img = env.base_image(sl, k)
    if hasattr(env, "_target_dot_is_drawn") or "Target" in type(env).__name__:
        img = SN._alpha_blend(
            img, SN._gaussian_dot(env._gy, env._gx, s[:, 2:4], sl("target.radius")), sl("target.color")
        )
    return SN._alpha_blend(img, SN._gaussian_dot(env._gy, env._gx, s[:, 0:2], sl("agent.radius")), sl("agent.color"))


def _check(env, states):
    sty = env.resolve_style(None, None, batch=len(states))
    want = _reference(env, states, sty)
    got = env.render(np.asarray(states, np.float32))
    if torch.equal(want, got):
        return
    d = (want.to(torch.int16) - got.to(torch.int16)).abs()
    raise AssertionError(
        f"{type(env).__name__}: {int((d.amax(-1) > 0).sum())} pixels differ, max {int(d.max())}. "
        f"Either the baked base image is not a function of the style alone, or _dot_window cropped "
        f"a dot that could still move a pixel."
    )


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("B", [1, 5, 16])
def test_matches_the_full_frame_composition_on_sampled_states(cfg, B):
    env = se.make(cfg, num_envs=B, device="cpu")
    env.rng = np.random.default_rng(0)
    _check(env, env.sample_states(B))


@pytest.mark.parametrize("cfg", CONFIGS)
def test_matches_on_states_the_env_would_never_sample(cfg):
    """Agents outside the frame, on the border, on and between pixel centres.

    ``render(states)`` takes whatever it is given -- only ``step`` clamps -- so a state outside the
    picture is a legal input and is exactly where the frame-maximum normalisation stops being a
    detail.
    """
    env = se.make(cfg, num_envs=1, device="cpu")
    # The MODERATE offsets are the ones that bite. A centre 400 px away underflows exp() to zero
    # before the normalisation can rescale anything, so every windowing agrees on a blank dot; a
    # centre 20-60 px outside still paints at full strength on the frame edge, and there the window
    # has to carry the distance from the centre to the nearest in-frame sample or it crops a live
    # dot. Dropping that term passes every extreme case and fails these.
    pos = [
        (112.0, 112.0),
        (112.5, 112.5),
        (0.0, 0.0),
        (223.0, 223.0),
        (-50.0, 112.0),
        (-20.0, 112.0),
        (230.0, 112.0),
        (112.0, -30.0),
        (112.0, 250.0),
        (-35.0, -35.0),
        (250.0, 250.0),
        (-5.0, 229.0),
        (-400.0, 112.0),
        (112.0, -400.0),
        (600.0, 600.0),
        (1e4, 1e4),
        (-1e4, 112.0),
    ]
    dim = env.state_dim
    states = np.zeros((len(pos), dim), np.float32)
    for i, (x, y) in enumerate(pos):
        states[i, 0], states[i, 1] = x, y
        if dim >= 4:  # TwoRoomTarget carries the target too
            states[i, 2], states[i, 3] = 223.0 - x, 223.0 - y
    e = se.make(cfg, num_envs=len(pos), device="cpu")
    _check(e, states)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("radius", [7.0, 9.5, 14.0])
def test_matches_across_the_declared_radius_range(cfg, radius):
    """The window scales with sigma, so both ends of ``agent.radius`` have to be checked."""
    env = se.make(cfg, num_envs=4, device="cpu")
    env.rng = np.random.default_rng(1)
    st = env.sample_states(4)
    style = np.tile(np.asarray(env.default_style(), np.float32), (4, 1))
    sty = env.resolve_style(style, {"agent.radius": [radius]}, batch=4)
    want = _reference(env, st, sty)
    got = env.render(np.asarray(st, np.float32), style=sty.values)
    assert torch.equal(want, got), f"{cfg} at agent.radius={radius}"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_the_bake_is_not_a_function_of_call_history(cfg):
    """Purity, order invariance and batch-size invariance, with the cache deliberately warm.

    The bake is the one piece of state Step 6 added to this path, and the failure it can have is a
    plate that was mutated after it was cached -- which happened once, when the agent was painted
    into the plate at batch 1 and every later render of that style came back with a stale dot on it.
    """
    env = se.make(cfg, num_envs=8, device="cpu")
    env.rng = np.random.default_rng(2)
    st = np.asarray(env.sample_states(8), np.float32)
    first = env.render(st).clone()
    solo = se.make(cfg, num_envs=1, device="cpu")  # warms and could corrupt the shared cache
    for i in range(8):
        assert torch.equal(solo.render(st[i : i + 1])[0], first[i]), f"{cfg}: row {i} at batch 1"
    assert torch.equal(env.render(st), first), f"{cfg}: render is not pure"
    perm = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    assert torch.equal(env.render(np.ascontiguousarray(st[perm]))[np.argsort(perm)], first), (
        f"{cfg}: render is not order invariant"
    )
    dup = np.concatenate([st[:1]] * 4 + [st[1:2]] * 4)
    out = env.render(dup)
    assert torch.equal(out[0], out[3]) and torch.equal(out[4], out[7]), f"{cfg}: duplicate rows"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_a_per_row_style_is_not_collapsed_to_one_plate(cfg):
    """Every row a different appearance: the bake must group by row and not paint row 0 everywhere.

    This is the failure mode the style tensor's own docstring calls the three shipped collapses, and
    a cache keyed on one row of a batch is the fourth way to have it.
    """
    env = se.make(cfg, num_envs=6, device="cpu")
    env.rng = np.random.default_rng(3)
    st = np.asarray(env.sample_states(6), np.float32)
    base = np.asarray(env.default_style(), np.float32)
    rows = np.tile(base, (6, 1))
    off = dict((p, o) for p, o, _ in env.variation_space.flat_layout())["background.color"]
    for i in range(6):
        rows[i, off : off + 3] = [10 + 40 * i, 20, 200 - 30 * i]
    sty = env.resolve_style(rows, None, batch=6)
    want = _reference(env, st, sty)
    got = env.render(st, style=rows)
    assert torch.equal(want, got), f"{cfg}: per-row style"
    flat = got.reshape(6, -1)
    assert len({tuple(flat[i].tolist()) for i in range(6)}) == 6, (
        f"{cfg}: six different backgrounds produced fewer than six different frames"
    )


@pytest.mark.parametrize("cfg", CONFIGS)
def test_two_styles_with_repeats_take_the_gather_branch(cfg):
    """The branch between "one plate, expanded" and "every row different, no bake".

    ``_base_image`` has three outcomes: all rows identical (one plate, expanded), all rows distinct
    (no bake at all), and the case in between -- a few distinct styles with repeats, which builds one
    plate each and GATHERS them back with an index. Only the third exercises ``torch.cat(plates)[pick]``,
    and until this test nothing did: the suite's per-row-style case has six distinct rows and takes the
    fallback, and every other case has one. An index built in the wrong order there would hand row i
    the wrong style's background and every digest above would still pass.

    Two styles across six rows, interleaved rather than blocked, so a gather that merely preserved
    grouping order would be caught.
    """
    env = se.make(cfg, num_envs=6, device="cpu")
    env.rng = np.random.default_rng(11)
    st = np.asarray(env.sample_states(6), np.float32)
    base = np.asarray(env.default_style(), np.float32)
    off = dict((p, o) for p, o, _ in env.variation_space.flat_layout())["background.color"]
    rows = np.tile(base, (6, 1))
    which = [0, 1, 0, 1, 1, 0]  # interleaved, three of each
    for i, w in enumerate(which):
        rows[i, off : off + 3] = [20.0, 200.0, 40.0] if w == 0 else [180.0, 30.0, 90.0]
    sty = env.resolve_style(rows, None, batch=6)
    want = _reference(env, st, sty)
    got = env.render(st, style=rows)
    assert torch.equal(want, got), f"{cfg}: two styles with repeats"

    # the gather really was the branch taken, not the expand and not the fallback
    plates = {}
    for i in range(6):
        plates.setdefault(tuple(rows[i, off : off + 3]), []).append(i)
    assert len(plates) == 2 and all(len(v) == 3 for v in plates.values())

    # rows sharing a style must share a background, and the two styles must differ
    a, b = list(plates.values())
    assert not torch.equal(got[a[0]][0, 0], got[b[0]][0, 0]), (
        f"{cfg}: the two styles produced the same corner pixel, so the gather collapsed them"
    )
    for i in a[1:]:
        assert torch.equal(got[i][0, 0], got[a[0]][0, 0]), f"{cfg}: row {i} got the wrong plate"
    for i in b[1:]:
        assert torch.equal(got[i][0, 0], got[b[0]][0, 0]), f"{cfg}: row {i} got the wrong plate"

    # and the invariants, on the branch that has an index in it
    assert torch.equal(env.render(st, style=rows), got), f"{cfg}: gather branch is not pure"
    perm = np.array([4, 0, 5, 2, 1, 3])
    assert torch.equal(
        env.render(np.ascontiguousarray(st[perm]), style=np.ascontiguousarray(rows[perm]))[np.argsort(perm)], got
    ), f"{cfg}: gather branch is not order invariant"
    solo = se.make(cfg, num_envs=1, device="cpu")
    for i in range(6):
        assert torch.equal(solo.render(st[i : i + 1], style=rows[i : i + 1])[0], got[i]), (
            f"{cfg}: row {i} differs when rendered alone, so a plate is stale or misindexed"
        )


def test_the_alpha_floor_is_the_measured_one():
    """The crop rests on this number; if somebody widens it the crop stops being a bound.

    Re-measured here rather than asserted: the smallest alpha ``_alpha_blend`` can act on over every
    uint8 contrast, found by bisection against that function itself.
    """
    smallest = 1.0
    for img_v in (0, 1, 127, 254, 255):
        for col_v in (0, 1, 127, 254, 255):
            if img_v == col_v:
                continue
            img = torch.full((1, 1, 1, 3), img_v, dtype=torch.uint8)
            rgb = torch.full((1, 3), float(col_v))
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                a = torch.full((1, 1, 1), mid)
                lo, hi = (lo, mid) if not torch.equal(SN._alpha_blend(img, a, rgb), img) else (mid, hi)
            smallest = min(smallest, hi)
    assert SN._ALPHA_FLOOR <= smallest, (
        f"_ALPHA_FLOOR is {SN._ALPHA_FLOOR:.3e} but _alpha_blend acts on alphas as small as "
        f"{smallest:.3e}; the dot window is no longer a bound"
    )

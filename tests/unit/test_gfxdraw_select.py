"""``render/gfxdraw.select_`` takes a different code path per device. They must agree on the bit.

The opaque write of every gfxdraw primitive is ``mask ? colour : surface``. ``torch.where`` is the
obvious spelling and is what CUDA gets; on the CPU it takes a scalar path over a broadcast bool and
the bitwise identity ``d ^= (d ^ s) * m`` is 3.7x faster, so the CPU gets that instead. The two
compute the same integers -- that is the whole justification for choosing between them by device --
and this is where that claim is checked rather than asserted, on a CPU host where only one of the
two branches ever runs in anger.

Without this, a divergence would surface as a GPU-only pixel difference in a renderer whose entire
claim is that its frames are bit-identical to pygame's.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stable_envs as se
from stable_envs.render import gfxdraw as G
from stable_envs.testing import conformance as C

CONFIGS = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")


def _where_branch(dst, mask, src):
    """The branch ``select_`` takes on CUDA, forced onto whatever device the test is running on."""
    torch.where(mask, src, dst, out=dst)


@pytest.mark.parametrize("name", CONFIGS)
def test_both_select_branches_draw_the_same_frame(name, monkeypatch):
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter")
    env = se.make(name, num_envs=8, device="cpu")
    states = np.asarray(env.sample_states(8, np.random.default_rng(0)), np.float32)
    style = C.uniform_style(env, 8, groups=2, seed=0)

    native = env._render(states).clone()
    native_styled = env._render(states, style=style).clone() if style is not None else None

    monkeypatch.setattr(G, "select_", _where_branch)
    assert torch.equal(env._render(states), native), (
        f"{name}: the two select_ branches disagree on the default appearance"
    )
    if native_styled is not None:
        assert torch.equal(env._render(states, style=style), native_styled), (
            f"{name}: the two select_ branches disagree under a per-environment style"
        )


@pytest.mark.parametrize("shape", [(3, 2, 5, 7), (3, 1, 1, 4)])
def test_select_is_a_plain_integer_choice(shape):
    """The identity itself, over every uint8 value rather than the ones a frame happens to hold."""
    g = torch.Generator().manual_seed(0)
    dst = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    src = torch.randint(0, 256, (shape[0], shape[1], 1, 1), dtype=torch.uint8, generator=g)
    mask = torch.rand(shape[1:], generator=g).gt(0.5).unsqueeze(0)
    want = torch.where(mask, src, dst)
    got = dst.clone()
    G.select_(got, mask, src)
    assert torch.equal(got, want)

"""``blend_at`` is an overridable seam, and its weight argument has a shape contract.

``Canvas.blend_at`` blends a weight per write. It would be cheaper for a primitive to hand over ONE
element when every write shares it — ``aaline``'s solid initial pixel and the two axis-aligned runs
all write at ``pixelColor`` — and ``blend`` broadcasts, so the whole render still draws the right
picture. But ``blend_at`` is overridden outside this module: ``LunarLander._AARecorder`` records
what ``aaline`` WOULD write by indexing the weight with its own mask, so it bakes the flag fringe
once at construction. A one-element weight against a ``B*(kmax+1)`` mask is an ``IndexError``
there — raised inside ``LunarLander.__init__``, so ``se.make("LunarLander")`` fails outright on any
host that has Box2D.

That regression shipped once. It survived a hand-written recorder check because the quad that check
used had four GENERAL edges, and a general edge never reaches the vline/hline scatters — so the
offending weight was passed and never indexed. The axis-aligned case is the one that matters and is
the one the flag triangle has.

So: drive the recorder with the exact shapes ``aaline_emit`` emits, over an edge set that reaches
every one of the five scatters, and assert the contract directly (``len(w) == len(valid)``) as well
as the behaviour.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stable_envs.render import gfxdraw as G

H = W = 32


class _Recorder(G.Canvas):
    """``LunarLander._AARecorder``, verbatim in shape: index the weight by the write mask."""

    def __init__(self, gfx):
        super().__init__(1, gfx.H, gfx.W, gfx.device, gfx.sh, gfx.ys, gfx.xs)
        self.writes: list[tuple] = []
        self.shapes: list[tuple] = []

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
        self.shapes.append((tuple(w.shape), tuple(valid.shape), tuple(y.shape)))
        ok = valid & (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        if bool(ok.any()):
            self.writes.append((y[ok].clone(), x[ok].clone(), w.expand(ok.shape)[ok].clone()))


#: One quad with two axis-aligned edges (so the vline and hline short-circuits both fire) and two
#: general ones (so the Wu pair fires), plus one that leaves the surface (so ``clip_line`` engages).
QUADS = {
    "axis-aligned-and-general": [[4, 4], [4, 20], [18, 20], [18, 4]],
    "all-general": [[2, 2], [12, 3], [10, 12], [3, 11]],
    "off-surface": [[-9, 6], [40, 9], [37, 26], [-6, 22]],
    "degenerate-row": [[5, 7], [25, 7], [25, 7], [5, 7]],
}


def _run(verts, kmax=64):
    gfx = G.Gfx(H, W, "cpu")
    rec = _Recorder(gfx)
    vi = torch.tensor([verts], dtype=torch.int64)
    bidx = torch.zeros(1, dtype=torch.long)
    for e in range(4):
        G.aaline(rec, vi[:, e, 0], vi[:, e, 1], vi[:, (e + 1) % 4, 0], vi[:, (e + 1) % 4, 1], None, kmax, bidx)
    return rec


@pytest.mark.parametrize("case", sorted(QUADS))
def test_a_blend_at_override_may_index_the_weight_by_the_write_mask(case):
    """The recorder must survive every scatter ``aaline`` issues, not only the general ones."""
    rec = _run(QUADS[case])
    assert rec.shapes, "aaline issued no scatters at all"
    for w_shape, valid_shape, y_shape in rec.shapes:
        assert w_shape == valid_shape == y_shape, (
            f"aaline handed blend_at a weight of shape {w_shape} against a mask of {valid_shape} "
            f"and coordinates of {y_shape}. An override that indexes the weight by the mask -- "
            f"LunarLander._AARecorder, which runs inside __init__ -- needs all three to agree."
        )


def test_the_axis_aligned_scatters_are_actually_reached():
    """The guard on the guard: a test that never fires the vline/hline path cannot catch this.

    The all-general quad is the shape the original check used; it reaches the solid-initial and Wu
    scatters only. The axis-aligned quad must reach strictly more writes, which is what makes it the
    case worth pinning.
    """
    axis = _run(QUADS["axis-aligned-and-general"])
    general = _run(QUADS["all-general"])
    n_axis = sum(int(y.numel()) for y, _, _ in axis.writes)
    n_general = sum(int(y.numel()) for y, _, _ in general.writes)
    assert n_axis > n_general, (
        f"the axis-aligned quad recorded {n_axis} writes and the all-general one {n_general}; the "
        f"vline/hline short-circuit is not being exercised, so this file is not testing what it "
        f"says it tests"
    )


#: The write list each quad class produces in the frozen baseline. Positions and
#: weights, in order. LunarLander bakes this into its flag fringe, so the CONTENT is a contract and
#: not only the shapes: a throughput edit is not allowed to change what the flags look like.
BASELINE_DIGESTS = {
    "all-general": ("66998ca141c31551", 66),
    "axis-aligned-and-general": ("17e088394cd240cf", 68),
    "degenerate-row": ("f8584a32e0bb9326", 46),
    "off-surface": ("d6a79b41d375dee3", 122),
}


@pytest.mark.parametrize("case", sorted(QUADS))
def test_the_recorded_fringe_matches_the_baseline(case):
    import hashlib

    rec = _run(QUADS[case])
    h = hashlib.sha256()
    for y, x, w in rec.writes:
        for t in (y, x, w):
            h.update(t.numpy().tobytes())
    want, n_want = BASELINE_DIGESTS[case]
    n = sum(int(y.numel()) for y, _, _ in rec.writes)
    assert (h.hexdigest()[:16], n) == (want, n_want), (
        f"the recorded aaline write list for the {case!r} quad changed against 89fa5ca "
        f"({n} writes, digest {h.hexdigest()[:16]}); LunarLander bakes this into its flag fringe"
    )

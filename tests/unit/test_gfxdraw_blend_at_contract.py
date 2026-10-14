"""The writes ``aaline`` hands ``Canvas.blend_at``, recorded per quad.

The quads below reach all five scatters, including the axis-aligned vline/hline runs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stable_envs.render import gfxdraw as G

H = W = 32


class _Recorder(G.Canvas):
    """A ``blend_at`` override that records each in-bounds write and its weight."""

    def __init__(self, gfx):
        super().__init__(1, gfx.H, gfx.W, gfx.device, gfx.sh, gfx.ys, gfx.xs)
        self.writes: list[tuple] = []

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
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


def test_the_axis_aligned_scatters_are_actually_reached():
    """The axis-aligned quad records more writes than the all-general one, so vline/hline run.

    The all-general quad reaches only the solid-initial and Wu scatters.
    """
    axis = _run(QUADS["axis-aligned-and-general"])
    general = _run(QUADS["all-general"])
    n_axis = sum(int(y.numel()) for y, _, _ in axis.writes)
    n_general = sum(int(y.numel()) for y, _, _ in general.writes)
    assert n_axis > n_general, (
        f"the axis-aligned quad recorded {n_axis} writes and the all-general one {n_general}; the "
        f"vline/hline short-circuit is not exercised"
    )


#: Digest prefix and write count of each quad's recorded positions and weights, in order.
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
        f"the recorded aaline write list for the {case!r} quad changed ({n} writes, digest {h.hexdigest()[:16]})"
    )

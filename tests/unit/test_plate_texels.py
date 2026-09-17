"""``stable_envs.envs._plate_texels`` — the shared per-value texel substitution.

These run on a CPU stub, with no MuJoCo model and no GL context, because every property worth
asserting about this module is arithmetic or bookkeeping and none of it should need a GPU to check:

``quantise``          MuJoCo TRUNCATES ``int(255 * v)``; the half-levels are where it shows.
``texel_slots``       masks are taken from the PRISTINE texture, and a mask that disagrees with the
                      texture it is handed is refused rather than silently painting the wrong texels.
``substitute``        writes every masked texel, leaves everything else alone, and ALWAYS uploads.
``texel_key``         the quantised bytes, and ``None`` at the registered default.

The GL half — that the upload goes to the renderer's own context — cannot be asserted here and is
asserted where it is observable: ``tests/unit/test_variation_maps_dmc_control.py`` renders, and the
byte-equality evidence against a full recompile is in
``examples/results/step8/capability/bake_vs_recompile_probe*.json``.
"""
from __future__ import annotations

import numpy as np
import pytest

from stable_envs.envs._plate_texels import quantise, substitute, texel_key, texel_slots

A = [0.1, 0.2, 0.3]
B = [0.2, 0.3, 0.4]
MARK = [0.5, 0.5, 0.5]


class _Ctx:
    def __init__(self):
        self.current = 0

    def make_current(self):
        self.current += 1


class _Renderer:
    """The two attributes ``substitute`` touches, and a count of what it did."""

    def __init__(self):
        self._gl_context = _Ctx()
        self._mjr_context = object()


class _Model:
    """A ``MjModel`` stub with one texture: enough for addressing, masks and the write."""

    def __init__(self, texels, nch=3, lead=7):
        self.tex_adr = np.array([lead], np.int64)
        self.tex_width = np.array([len(texels)], np.int64)
        self.tex_height = np.array([1], np.int64)
        self.tex_nchannel = np.array([nch], np.int64)
        body = np.asarray(texels, np.uint8).reshape(-1, nch)
        # deliberate leading padding: a helper that ignored ``tex_adr`` would still pass without it
        self.tex_data = np.concatenate([np.full(lead, 199, np.uint8), body.reshape(-1)])
        self.lead = lead


def _checker(n_a=4, n_b=3, n_mark=2):
    return _Model([quantise(A)] * n_a + [quantise(B)] * n_b + [quantise(MARK)] * n_mark)


@pytest.fixture()
def uploads(monkeypatch):
    """``mujoco.mjr_uploadTexture`` calls, recorded. The import is inside ``substitute``."""
    seen = []
    import types
    fake = types.SimpleNamespace(mjr_uploadTexture=lambda m, con, t: seen.append(t))
    monkeypatch.setitem(__import__("sys").modules, "mujoco", fake)
    return seen


# ----------------------------------------------------------------------------- quantisation

@pytest.mark.parametrize("v,want", [
    (0.1, 25),        # 25.5 -> 25, where rounding gives 26: the DM Control grid's own rgb1[0]
    (0.3, 76),        # 76.5 -> 76
    (0.16, 40),       # the OGB floor's own value, 40.8 -> 40
    (0.0, 0), (1.0, 255), (0.5, 127),
])
def test_quantise_truncates_it_does_not_round(v, want):
    assert int(quantise([v, v, v])[0]) == want


def test_quantise_clips_rather_than_wrapping():
    assert list(quantise([-1.0, 2.0, 0.5])) == [0, 255, 127]


def test_quantise_is_what_swm_writes():
    """swm's Fetch skybox write is ``(bg * 255).astype(np.uint8)`` — the same truncation, by a
    different route. If these ever disagree, our texture is not swm's texture."""
    v = np.linspace(0.0, 1.0, 101)
    assert np.array_equal(quantise(np.stack([v, v, v], 1)),
                          (np.stack([v, v, v], 1) * 255).astype(np.uint8))


# ----------------------------------------------------------------------------------- slots

def test_slots_are_none_when_the_texture_is_absent():
    assert texel_slots(_checker(), -1, [A, B]) is None


def test_masks_default_to_equality_and_report_their_coverage():
    s = texel_slots(_checker(), 0, [A, B])
    assert [int(m.sum()) for m in s["masks"]] == [4, 3]
    assert (s["covered"], s["texels"]) == (7, 9)     # the two mark texels are NOT covered


def test_an_empty_mask_is_refused():
    """A colour the texture does not carry means the leaf would silently do nothing."""
    with pytest.raises(AssertionError, match="none of default colour"):
        texel_slots(_checker(), 0, [A, [0.77, 0.77, 0.77]])


def test_explicit_masks_are_checked_against_the_texture():
    """The DM Control case: masks supplied because markrgb == rgb2 makes equality ambiguous.

    A mask that selects a texel not holding its colour is the failure mode that matters — it would
    paint texels swm's recompile leaves alone — so it is refused rather than trusted.
    """
    m = _checker()
    good = [np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], bool), np.array([0, 0, 0, 0, 1, 1, 1, 0, 0], bool)]
    s = texel_slots(m, 0, [A, B], masks=good)
    assert [int(x.sum()) for x in s["masks"]] == [4, 3]
    bad = [good[0], np.array([0, 0, 0, 0, 1, 1, 1, 1, 0], bool)]   # reaches into the marks
    with pytest.raises(AssertionError, match="do not hold default colour"):
        texel_slots(m, 0, [A, B], masks=bad)


def test_a_mask_of_the_wrong_length_is_refused():
    with pytest.raises(AssertionError, match="texels"):
        texel_slots(_checker(), 0, [A, B], masks=[np.ones(3, bool), np.ones(3, bool)])


# ------------------------------------------------------------------------------ substitute

def test_substitute_writes_the_masked_texels_and_nothing_else(uploads):
    m, r = _checker(), _Renderer()
    s = texel_slots(m, 0, [A, B])
    before = m.tex_data.copy()
    substitute(m, r, s, [[0.9, 0.1, 0.1], [0.0, 0.0, 1.0]])
    body = m.tex_data[m.lead:].reshape(-1, 3)
    assert np.array_equal(body[:4], np.tile(quantise([0.9, 0.1, 0.1]), (4, 1)))
    assert np.array_equal(body[4:7], np.tile(quantise([0.0, 0.0, 1.0]), (3, 1)))
    assert np.array_equal(body[7:], np.tile(quantise(MARK), (2, 1))), "markrgb texels were painted"
    assert np.array_equal(m.tex_data[:m.lead], before[:m.lead]), "wrote outside tex_adr"


def test_substitute_makes_the_context_current_before_every_upload(uploads):
    """The load-bearing line. ``mujoco.Renderer`` makes its context current in its constructor and
    in ``render()`` and nowhere else, and GL texture names are per-context, so an upload under
    another rig's context overwrites an unrelated texture object in it."""
    m, r = _checker(), _Renderer()
    s = texel_slots(m, 0, [A, B])
    substitute(m, r, s, [A, B])
    substitute(m, r, s, [A, B])
    assert (r._gl_context.current, uploads) == (2, [0, 0])


def test_substitute_is_unconditional_even_at_the_default(uploads):
    """No "it already looks like that" shortcut: the rig's texture and context outlive any one bake
    and carry whatever the last one left, so skipping the default write is what leaks a previous
    render's floor into this one."""
    m, r = _checker(), _Renderer()
    s = texel_slots(m, 0, [A, B])
    substitute(m, r, s, [[0.9, 0.1, 0.1], B])         # a custom bake
    substitute(m, r, s, s["default"])                 # then the default
    body = m.tex_data[m.lead:].reshape(-1, 3)
    assert np.array_equal(body, _checker().tex_data[7:].reshape(-1, 3))
    assert len(uploads) == 2


def test_substitute_restores_exactly_after_any_sequence(uploads):
    """Absolute assignment, so bake B after A is bake B: the masks are positional and never depend
    on what the texture currently holds."""
    m, r = _checker(), _Renderer()
    s = texel_slots(m, 0, [A, B])
    pristine = m.tex_data.copy()
    for c in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.33, 0.66, 0.99], B):
        substitute(m, r, s, [c, c])
    substitute(m, r, s, s["default"])
    assert np.array_equal(m.tex_data, pristine)


def test_two_colours_colliding_cannot_corrupt_the_masks(uploads):
    """The reason the masks come from the pristine texture. Paint both sets the SAME colour, then
    ask for two different ones: a helper that re-derived its masks by equality would now have one
    mask of seven texels and one empty, and the second request would be wrong."""
    m, r = _checker(), _Renderer()
    s = texel_slots(m, 0, [A, B])
    substitute(m, r, s, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    substitute(m, r, s, [A, B])
    assert np.array_equal(m.tex_data, _checker().tex_data)


def test_a_wrong_number_of_colours_is_refused(uploads):
    m, r = _checker(), _Renderer()
    s = texel_slots(m, 0, [A, B])
    with pytest.raises(AssertionError, match="colours for a texture"):
        substitute(m, r, s, [A, B, MARK])


def test_a_four_channel_texture_keeps_its_fourth_channel(uploads):
    m = _Model([list(quantise(A)) + [11], list(quantise(B)) + [22]], nch=4)
    r = _Renderer()
    s = texel_slots(m, 0, [A, B])
    substitute(m, r, s, [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    body = m.tex_data[m.lead:].reshape(-1, 4)
    assert list(body[:, 3]) == [11, 22]


# -------------------------------------------------------------------------------- cache key

def test_the_key_is_none_at_the_default_by_whatever_route():
    s = texel_slots(_checker(), 0, [A, B])
    assert texel_key(s, None) is None
    assert texel_key(s, [A, B]) is None
    # 0.1003 * 255 = 25.58 -> 25, the same byte as 0.1: the same texture, so the same plate
    assert texel_key(s, [[0.1003, 0.2003, 0.3003], B]) is None
    assert texel_key(None, [A, B]) is None


def test_the_key_is_the_bytes_not_the_floats():
    s = texel_slots(_checker(), 0, [A, B])
    k = texel_key(s, [[0.9, 0.1, 0.1], B])
    assert k == (229, 25, 25, 51, 76, 102)
    assert texel_key(s, [[0.9004, 0.1004, 0.1004], B]) == k
    assert texel_key(s, [[0.9, 0.1, 0.1], [0.9, 0.1, 0.1]]) != k


def test_the_key_is_hashable_and_orders_the_colours():
    s = texel_slots(_checker(), 0, [A, B])
    assert {texel_key(s, [A, [0.9, 0.9, 0.9]]), texel_key(s, [[0.9, 0.9, 0.9], B])} != {None}
    assert texel_key(s, [A, [0.9, 0.9, 0.9]]) != texel_key(s, [[0.9, 0.9, 0.9], B])

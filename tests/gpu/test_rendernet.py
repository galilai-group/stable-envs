"""The LEARNED-decoder render path, on the real checkpoints: what it composes, and what it refuses.

``test_conformance.py`` already covers these classes for everything every other env is covered for —
they are ordinary registry entries, so parity/visual fidelity, determinism, batch invariance, style,
dead dimensions and the reference bridge all apply to them without a line being added there. This file
is for the properties that only exist because a network is in the path:

  composition      the delivered frame IS the cheap render put through the corrector, and nothing
                   else — no second resize, no supersample, no stray copy. Modelled on
                   ``test_fast_paths.py``: assert the consequence, not the argument.
  it is applied    the frame differs from a bilinear upsample of the same cheap render. A corrector
                   that silently failed to load, or whose head had collapsed to zero, would still
                   produce a perfectly reasonable picture and pass every other test in the suite.
  shape discipline the network sees ONE batch shape, so cuDNN and Inductor make one choice; a
                   different choice is a different sum order and lands on the far side of a rounding.
  the identity     a corrector keyed to one cell REFUSES every other, loudly. On the wrong camera it
                   is measured at 2.6x worse than not correcting at all, and it fails by painting
                   texture that is not there, which the pixel mean does not catch.
  the opt-in       no checkpoint, no class. The absence of weights removes an env; it never changes
                   one.

If no corrector is cached on this machine, everything here skips with the command that builds one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import EMPTY, family_params

import stable_envs as se
from stable_envs.render import rendernet as RN

LEARNED = sorted(n for n, c in se.registry.REGISTRY.items() if getattr(c, "render_decoder", "renderer") == "learned")


#: The module whose absence EXPLAINS an empty family. RoboCasa is the only family that declares a
#: learned decoder, and it is the reason this file can legitimately have nothing to run: the kitchen is
#: BUILT by an interpreter pinned to mujoco==3.3.1 and RENDERED by one needing mujoco_warp>=3.11, so
#: most interpreters hold neither end of it.
LEARNED_MODULES = ("robocasa",)


@pytest.fixture(scope="module", params=family_params(LEARNED, family="learned-decoder"))
def env(request):
    """The learned-decoder envs, or a STATED reason there are none — and they are not one reason.

    This fixture used to be ``params=LEARNED or ["_none"]`` with a body that skipped: the one green
    skip that ``tests/gpu/_shared.py`` exists to abolish, still here after the sweep that removed it
    from four other files. It collapsed three situations with three different remedies into one
    sentence, so the two that are faults read as the one that is not:

      no ``robocasa`` here      a dependency fact, and nothing to do — the cover in
                                ``tests/fixtures/regression/census/assignment.json`` hands these envs to the
                                interpreter that has them. SKIP, with the import error attached.
      no sealed corrector       an ARTIFACT fact. The class is gated on the checkpoint ("no
                                checkpoint, no class", above), so this empties the family too — but
                                the remedy is a command, not a different interpreter. SKIP, saying so.
      neither of those          the family is empty for no reason anybody stated. FAIL: a sweep that
                                collects nothing is green and proves nothing.
    """
    if request.param == EMPTY:
        errs = se.import_errors()
        blamed = {m: errs[m] for m in LEARNED_MODULES if m in errs}
        if blamed:
            pytest.skip(
                "learned-decoder: no env declares one here because "
                + "; ".join(f"{m} did not import ({e})" for m, e in blamed.items())
                + " — the corrector is checked in the interpreter that owns RoboCasa in "
                "tests/fixtures/regression/census/assignment.json"
            )
        if not RN.available():
            pytest.skip(
                f"learned-decoder: {', '.join(LEARNED_MODULES)} imported cleanly, so this is a "
                f"missing ARTIFACT rather than a missing dependency — a learned class is gated on "
                f"its checkpoint, so no sealed corrector means no class to sweep. Build one:\n    "
                + RN.BUILD_CMD.format(env="RoboCasaCounterToSink")
                + f"\n(the cache looked in is {RN.CACHE_ROOT})"
            )
        pytest.fail(
            f"learned-decoder: {', '.join(LEARNED_MODULES)} imported cleanly AND this machine has "
            f"sealed correctors for {sorted(RN.available())}, yet NO registered class declares "
            f"render_decoder='learned'. Nothing explains that, so it is a registration or naming "
            f"change rather than a missing dependency — and every test in this file would otherwise "
            f"have collected zero cases and reported green."
        )
    return se.make(request.param, num_envs=8, device="cuda")


def plain_sibling(env):
    """The same cell drawn by the renderer alone. Every learned class is derived from one."""
    name = type(env).__name__.removesuffix("Corrected")
    if name not in se.registry.REGISTRY:
        pytest.fail(
            f"{type(env).__name__} has no plain sibling {name} in the registry; a learned "
            f"decoder is a second way to draw a cell that already exists, and without the "
            f"first one there is nothing to compare it against"
        )
    return se.make(name, num_envs=env.B, size=env.size, device="cuda")


def cheap_render(env, states):
    """The cheap frame, produced independently of the corrected class, through its OWN profile."""
    corr = env.corrector()
    base = se.registry.REGISTRY[type(env).__name__.removesuffix("Corrected")]
    cls = type("CheapProbe", (base,), dict(TEX_LOD=corr.tex_lod, BACKFACE_CULL=corr.backface_cull))
    e = cls(num_envs=env.B, size=env.size, device="cuda")
    return e.render(states, size=corr.render_size)


def test_the_frame_is_exactly_the_cheap_render_put_through_the_corrector(env):
    """Composition, byte for byte, against the two halves assembled by hand.

    The failure this catches is a render profile that drifted from the one the checkpoint was fitted
    on — a texture LOD, a resolution, a culling flag. None of them raises: they hand the network an
    input outside its training distribution and it answers confidently, which is the same silent
    failure as a mismatched cell with none of its visibility.
    """
    corr = env.corrector()
    states = env.sample_states(16, rng=np.random.default_rng(0))
    got = env._render(states)
    low = cheap_render(env, states)
    want = corr(low)
    d = (got.to(torch.int16) - want.to(torch.int16)).abs()
    assert int(d.max()) == 0, (
        f"{type(env).__name__}.render is not render({corr.render_size} px, tex_lod {corr.tex_lod}) "
        f"followed by the corrector: they differ by max {int(d.max())}/255 on "
        f"{int((d > 0).any(-1).sum())} pixels. Something in the render path is not the configuration "
        f"the weights were fitted on."
    )


def test_the_corrector_is_actually_applied(env):
    """The frame must differ from a plain resize of the same cheap render, and from the plain class.

    A corrector that failed to load, loaded as its zero-initialised initialisation, or was applied to
    the wrong tensor would produce a completely reasonable picture — the bilinear anchor is a real
    frame — and would pass composition, determinism, batch invariance and the fidelity gate's ceiling
    while delivering none of what the class claims. This is the only test that looks at whether the
    network did anything.
    """
    import torch.nn.functional as F

    corr = env.corrector()
    states = env.sample_states(8, rng=np.random.default_rng(1))
    got = env._render(states)
    low = cheap_render(env, states)
    anchor = (
        F.interpolate(
            low.permute(0, 3, 1, 2).float().div(255.0),
            size=(corr.out_size, corr.out_size),
            mode="bilinear",
            align_corners=False,
        )
        .mul_(255.0)
        .round_()
        .clamp_(0, 255)
    )
    anchor = anchor.to(torch.uint8).permute(0, 2, 3, 1)
    moved = float(((got != anchor).any(-1)).float().mean())
    assert moved > 0.05, (
        f"the corrected frame differs from a bilinear upsample of its own input on only "
        f"{100 * moved:.2f}% of pixels. The residual head is zero-initialised, so 'exactly the "
        f"bilinear baseline' is what an untrained, unloaded or misapplied corrector looks like."
    )
    plain = plain_sibling(env).render(states)
    assert not torch.equal(got, plain), (
        "the learned frame is byte-identical to the plain class's frame, which cannot happen if a network ran on it"
    )


def test_the_same_states_twice_give_the_same_bytes(env):
    """Determinism, at the size and batch a consumer uses, after the graph has been replayed.

    Compilation, autotuning and CUDA-graph capture are exactly the machinery that can make a frame
    depend on how many times the process has rendered. The static input buffer's tail is refilled on
    every call for the same reason.
    """
    states = env.sample_states(16, rng=np.random.default_rng(2))
    a = env._render(states).clone()
    env._render(env.sample_states(16, rng=np.random.default_rng(3)))  # something else in between
    b = env._render(states)
    d = (a.to(torch.int16) - b.to(torch.int16)).abs()
    assert int(d.max()) == 0, (
        f"two renders of the same states differ by max {int(d.max())}/255 on {int((d > 0).any(-1).sum())} pixels"
    )


@pytest.mark.parametrize("sizes", [(1, 3, 7, 8, 16, 129)])
def test_a_frame_does_not_depend_on_how_many_frames_were_asked_for(env, sizes):
    """Including the odd ones, and including one over the network's fixed batch.

    The network is always handed exactly ``NET_BATCH`` rows and the padding is discarded, so this is
    supposed to hold by construction — which is precisely why it is asserted: the measured
    alternative is that it does not. The same checkpoint compiled at a network batch of 64 and at 128
    produces DIFFERENT pixels (digests fe684973 and e18c6b84 on the same states), because
    max-autotune benchmarks per shape and two convolution tile configurations sum a 3x3xC window in
    different orders. A caller's batch size must not be able to reach that decision.
    """
    n = max(sizes)
    states = env.sample_states(n, rng=np.random.default_rng(4))
    ref = env._render(states).cpu().numpy()
    for b in sizes:
        got = env._render(states[:b]).cpu().numpy()
        d = np.abs(got.astype(np.int16) - ref[:b].astype(np.int16))
        assert int(d.max()) == 0, (
            f"rendering {b} of the same states gives different pixels from rendering {n} "
            f"(max {int(d.max())}/255 on {int((d > 0).any(-1).sum())} pixels)"
        )


def test_a_size_the_checkpoint_was_not_fitted_for_is_refused(env):
    """Not serviced by resizing. A corrector is a fixed map between two resolutions."""
    states = env.sample_states(4, rng=np.random.default_rng(5))
    with pytest.raises(ValueError, match="delivers"):
        env._render(states, size=env.size // 2)
    with pytest.raises(ValueError, match="delivers"):
        env._render(states, supersample=2)


def test_this_corrector_refuses_every_other_registered_cell(env):
    """The identity lock, against every sibling this machine actually has.

    Not a synthetic key (``tests/unit/test_rendernet_identity.py`` does that): the real checkpoint against the
    real other cells, because the mistake this prevents is the plausible one — the cells are siblings,
    they share a kitchen or a robot, and their frames look alike.
    """
    corr = env.corrector()
    mine = type(env).corrector_cell_key()
    others = [
        c
        for n, c in sorted(se.registry.REGISTRY.items())
        if getattr(c, "render_decoder", "renderer") == "renderer"
        and hasattr(c, "corrector_cell_key")
        and c.corrector_cell_key().get("family") == mine["family"]
        and c.__name__ != type(env).__name__.removesuffix("Corrected")
    ]
    if not others:
        pytest.skip("no sibling cells of this family are registered in this interpreter")
    for cls in others:
        key = dict(cls.corrector_cell_key(), env_class=f"{cls.__name__}Corrected")
        with pytest.raises(RN.CellMismatch):
            RN.load(key, device="cuda", path=corr.path)


def test_no_checkpoint_means_no_class(monkeypatch):
    """Point the loader at an empty directory and the corrected classes stop being built.

    The opt-in is the presence of a file, and this is the assertion that says so. It also says the
    other half: nothing about the plain classes is conditional on it.
    """
    RC = pytest.importorskip("stable_envs.envs.robocasa")
    monkeypatch.setattr(RN, "CACHE_ROOT", "/nonexistent/stable-envs-rendernet")
    assert RN.available() == {}
    assert RC._make_corrected_classes() == {}
    plain = [n for n in se.registry.REGISTRY if n.startswith("RoboCasa") and not n.endswith("Corrected")]
    assert plain, "the plain RoboCasa classes must be registered whether or not weights exist"
    for n in plain:
        assert se.registry.REGISTRY[n].render_decoder == "renderer"
        assert se.registry.REGISTRY[n].TEX_LOD == 1 and se.registry.REGISTRY[n].BACKFACE_CULL is None, (
            f"{n} is the plain class and its MJWarp render profile must be the one every number in "
            f"FIDELITY was measured through"
        )


def test_the_declaration_travels_with_the_weights(env):
    """The class's declared numbers ARE the sealed checkpoint's, field for field.

    ``visual_fidelity`` for a learned decoder describes a (renderer, checkpoint) pair, so it cannot
    live on the class independently of the file — a checkpoint replaced without its declaration would
    otherwise be described by the previous one's numbers.
    """
    corr = env.corrector()
    with open(corr.path[:-3] + ".json") as f:
        hdr = json.load(f)
    assert env.visual_fidelity == hdr["visual_fidelity"]
    assert env.variation_fidelity == hdr["variation_fidelity"]
    assert env.describe()["render_decoder"] == "learned"
    assert "LEARNED" in (type(env).__doc__ or "").upper()
    assert "LEARNED" in se.format_registry()


def test_the_declared_numbers_beat_the_render_they_replace(env):
    """A learned decoder that is not better than the renderer it replaces must not exist.

    Asserted on the declaration as well as on live frames (``test_conformance.py``) because this one
    is free and catches the case where a checkpoint was sealed by something other than
    ``train_corrector.py``, whose gate is the same three comparisons.
    """
    corr = env.corrector()
    for label, decl in (
        ("at the default appearance", corr.visual_fidelity),
        ("under the declared variation", corr.variation_fidelity),
    ):
        assert decl, f"{label}: no measurement was declared"
        assert decl["mean_abs"] < decl["plain_mean_abs"], f"{label}: mean |diff| is not better"
        assert decl["within8_frac"] > decl["plain_within8_frac"], f"{label}: within-8 is not better"
        assert decl["ssim"] > decl["plain_ssim"], f"{label}: SSIM is not better"


def test_the_compiled_configuration_is_the_declared_one(env):
    """The numbers were measured through a specific compile mode and network batch. Say which.

    Inductor fusion moves ~11% of pixels by 1/255 against eager, and the network batch changes the
    autotuner's choice; both are inside the declaration's error bars and neither is inside its
    reproducibility. A dataset generated here is reproducible by someone who sets the same two
    environment variables, and this is where they find out what they were.
    """
    corr = env.corrector()
    decl = corr.visual_fidelity.get("decoder", "")
    assert RN.COMPILE_MODE in decl and str(RN.NET_BATCH) in decl, (
        f"this process is running the corrector at mode={RN.COMPILE_MODE!r} batch={RN.NET_BATCH} and "
        f"the declaration says {decl!r}. Those are different arithmetic; re-measure with "
        f"examples/rendernet/train_corrector.py or put the settings back."
    )

"""The env contract, asserted over the registry. Add an env, re-run pytest, get covered.

These checks are the repo's whole warrant. The claim "you never need to store frames again" is only
worth making if the regenerated frame is the frame you would have stored, and that is an empirical
question with a number attached, not a design intention. Each check below corresponds to a way the claim
can be false:

  render parity        the GPU frame must match the OFFICIAL renderer on states the env actually
                       reaches. Reported as interior-pixel agreement (pixels whose neighbourhood in the
                       reference is uniform) separately from all-pixel agreement, because edge pixels
                       are anti-aliasing artifacts of two different rasterizers and neither is ground
                       truth there — folding them together lets a real geometry error hide inside a
                       plausible-looking aggregate.
  state round-trip     set_state(get_state()) must be the identity, and rendering must depend on every
                       state dimension. A dimension the render ignores means the state is not minimal or
                       the renderer is wrong; both are silent.
  batch invariance     frame i of a batch must equal the frame rendered alone. Batching is the entire
                       performance story, so a batch-dependent render would make every fast number a lie.
  backend agreement    the fused kernel must agree with the readable torch implementation. The kernel is
                       the thing that is fast; the torch path is the thing that is obviously correct.
  variation coverage   every declared variation must actually change the picture. A knob wired into the
                       space but not into ``scene()`` looks supported and does nothing.
  determinism          the same states must render identically twice.
  simulator agreement  where an env ALSO ships a batched GPU ``step``, how far it drifts from the
                       official simulator on identical actions from identical states. This is a
                       different claim from render parity and its strength differs by env: exact for a
                       finite discrete transition (MiniGrid), a few ulps for a published closed form
                       (classic control), a distribution for a contact solver (PushT).
                       ``check_step_matches_reference`` reports all three in the same shape so they can
                       be compared — and never in the same words.

Helpers here are plain functions returning numbers so they can be used from a benchmark or a report,
not only from pytest.

ONE RULE FOR ENV AUTHORS, learned three times at increasing cost:

    In this process an EGL context is created and NEVER released. Anything that wants to bound how
    many exist must bound how many are CREATED.

Releasing one corrupts rendering in every other live context, and the corruption is silent — it shows
up as pixel differences in unrelated environments, which reads exactly like a state-sufficiency
violation or a broken renderer. It has appeared as: a reference-env pool that closed the oldest entry;
this suite building two reference envs per env for `test_state_is_sufficient` (now one, driven through
two histories); and a bake path that constructed and closed a ``mujoco.Renderer`` per baked plate,
~175 releases across 25 envs. Cache renderers by content and keep them; do not close them.

WHAT THAT RULE IS AND IS NOT, now measured rather than inferred (H200, driver 590.48.01, MuJoCo 3.9,
``scratch``-run probe): **96 simultaneous live EGL contexts were held with zero pixel difference, and
closing one of 96 changed nothing (0/255) in the survivors.** So the "~22 contexts for the whole
package" figure that used to appear in this file was an assumption, not a measurement, and it was low
by more than 4x. Two things follow, and they point in opposite directions:

  * a test is not obliged to keep the context count in the low tens, so bounding distinct static-scene
    style values is a COST argument (each distinct value re-bakes a plate through MuJoCo) rather than a
    correctness one;
  * the failures that produced the rule were real, and on a healthy node they do not reproduce. Every
    instance traced back to a node whose EGL rasterizer was already returning different pixels for the
    same scene. ``check_reference_renderer_is_stable`` below turns that into a one-line diagnosis
    instead of 25 confusing parity failures followed by SIGABRT.

The rule is therefore kept as written — do not close contexts, do not scale creation with batch — but
it is now a cheap invariant rather than a scarce budget, and a suite that trips it should suspect its
NODE first.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

#: ``(env class, output size)`` -> the OFFICIAL reference env. Built once per process, never dropped.
_REFERENCE_ENVS: dict = {}


def reference_env(cls, size: int):
    """The reference env for ``cls`` at ``size``, built once per process and kept alive forever.

    Not a speed optimisation, though it is worth about two minutes on the full run. ``reference_env()``
    returns a LIVE renderer, and for the MuJoCo families that renderer owns an EGL context. A caller
    that binds one to a local drops it the moment the function returns, and CPython frees it there and
    then — so the conformance suite was RELEASING contexts while other tests were still using theirs,
    which is the one thing the module docstring says must never happen. Measured on the full 44-env
    run with ``GLContext.__init__``/``free`` instrumented: **83 contexts created and 50 released
    mid-run**, most of them a parity test's reference env being collected while the next env's was
    being built.

    Keying on the CLASS and not on ``reference_id`` is deliberate. MiniGrid registers a top-down and a
    POV class against the same official id, and they drive it differently (``get_full_render`` versus
    ``get_pov_render``, and a different ``_se_tile``), so sharing one env between them would compare a
    frame against the wrong reference and the mistake would look like a renderer bug.

    Failures are not cached: a missing optional dependency should keep raising ``ImportError`` so the
    caller can skip with the reason, rather than being turned into a ``KeyError`` on the second call.
    """
    key = (cls, int(size))
    hit = _REFERENCE_ENVS.get(key)
    if hit is None:
        hit = _REFERENCE_ENVS[key] = cls.reference_env(size=int(size))
    return hit


def frames_digest(frames) -> str:
    """sha256 of a uint8 frame stack. The unit of comparison across a process boundary.

    Shipping pixels between processes to compare them costs more than the render does; a digest is 32
    bytes and any difference at all changes it. ``.contiguous()`` first, because a strided view hashes
    its buffer rather than its logical contents and two equal frames would then digest differently.
    """
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(frames)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def check_reference_renderer_is_stable(cls, size: int, n: int = 4) -> dict:
    """Is THIS MACHINE's offscreen rasterizer deterministic at all? Ask before blaming the code.

    Every EGL incident recorded in this repo — frames 69-90/255 apart on identical states, then
    ``SIGABRT`` — was first diagnosed as a state-sufficiency violation or a context-budget problem, and
    at least one of them was neither: the node's own rasterizer was returning different pixels for the
    same scene. The measured context budget (96 live, and closing one harmless) does not support the
    scarcity reading, and the same suite passes on a healthy node and fails on a flagged one.

    So this asks the cheapest possible question first: render the same states twice through the SAME
    reference env and diff. It costs two frames and it converts "25 envs failed and then the
    interpreter aborted" into one line naming the node.
    """
    ref = reference_env(cls, size)
    env = cls(num_envs=1, size=size, device="cuda")
    st = env.sample_states(n, rng=np.random.default_rng(0))
    a = cls.reference_render(ref, st)
    b = cls.reference_render(ref, st)
    return {"max_abs_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}


def interior_mask(frames: np.ndarray) -> np.ndarray:
    """(N,H-1,W-1) bool: pixels whose 3x3 neighbourhood in ``frames`` is a single flat colour.

    A disagreement at an interior pixel is a real error — wrong colour, wrong shape, wrong place. A
    disagreement at an edge pixel is a sub-pixel coverage rule landing on a different side of a
    rounding, which two independent rasterizers are entitled to disagree about.

    WHERE THIS PREMISE FAILS, measured rather than assumed. "Flat 3x3 in the reference" implies "no
    rounding ambiguity here" only for a renderer that fills flat-coloured primitives. For a renderer
    whose output is a smooth per-pixel function it is false: the quantisation contour of a slow ramp
    can sit arbitrarily far from any 3x3 non-flatness, so a pixel on the contour is classified
    interior while being exactly where two implementations may round apart. The navigation family is the
    worked example — a normalised Gaussian alpha with a 41 px tail, where the reference is constant
    over a 7x7 patch around pixels whose underlying float is still moving — and it costs those envs a
    declared ``parity_interior_min`` of 0.999995 (about 2 pixels in 2.3 million, one in eight draws).
    See ``stable_envs.envs.navigation``.

    The mask is deliberately NOT widened to compensate. Doing so would relax the strong claim for
    every flat-shaded env in the registry — where the 3x3 test is exactly right — in order to
    accommodate one family, and this repo's rule is that such a family declares a floor and names its
    mechanism instead.
    """
    import scipy.ndimage as ndi

    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1] + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def pixel_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement between two uint8 frame stacks, split into all-pixel and interior-pixel."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    res = {
        "mean_abs": float(d.mean()),
        "exact_frac": float((d == 0).mean()),
        "within8_frac": float((d <= 8).mean()),
        "max_abs": int(d.max()),
    }
    try:
        m = interior_mask(b)
        di = d[:, :-1, :-1][m]
        if di.size == 0:
            # No flat 3x3 neighbourhood anywhere, so "interior agreement" is undefined — NOT zero.
            # Reporting 0.0 (or the NaN that ``empty.mean()`` yields) makes a BIT-EXACT renderer fail
            # the parity assertion, because NaN >= floor is False and no tolerance can rescue it. This
            # is reachable: Craftax's dithered pixel art has no flat neighbourhood at 7px or 16px, and
            # it reported interior_frac 0.000 alongside exact_frac 1.0. None is the honest value, and
            # the test skips the interior check on it and falls back to all-pixel agreement.
            res.update(
                interior_frac=float(m.mean()),
                interior_exact_frac=None,
                interior_mean_abs=None,
                interior_px=0,
                interior_bad_px=0,
            )
        else:
            # The COUNTS, not only the fraction. An interior disagreement is reported as a number
            # like 0.9999996, and nobody can act on that: the decision an env author has to make is
            # whether this is a geometry error or a handful of pixels on a quantisation boundary, and
            # that is a question about how many pixels, not about how many nines. Measured on the two
            # families where it came up, MujocoPusher and the navigation envs, the answer was 1-2 pixels
            # of ~3 million, on about a fifth of draws — which reads as a rounding boundary at a
            # glance and as a passing test with 0.9999996 in it.
            res.update(
                interior_frac=float(m.mean()),
                interior_exact_frac=float((di == 0).mean()),
                interior_mean_abs=float(di.mean()),
                interior_px=int(di.size),
                interior_bad_px=int((di != 0).sum()),
            )
    except ImportError:  # scipy is optional; the coarse numbers still stand
        res.update(
            interior_frac=None,
            interior_exact_frac=None,
            interior_mean_abs=None,
            interior_px=None,
            interior_bad_px=None,
        )
    return res


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """Mean SSIM over a uint8 frame stack, on luminance, with the standard 8-bit constants.

    Written out rather than taken from ``skimage``, which is not a dependency of this repo and would
    be one for eight lines of uniform filtering. Luminance rather than per-channel because the visual
    tier's residual is a colour-and-texture error that lands in luma, and averaging three channels
    would report the same disagreement three times.

    It exists for the VISUAL tier and for nothing else. A parity env is scored by counting pixels that
    are bit-identical, which is a statement no perceptual metric can make; SSIM is here because
    "6.6/255 mean" does not by itself say whether a picture reads as the same picture, and the visual
    tier's whole claim is that it does.
    """
    import scipy.ndimage as ndi

    f = lambda x: ndi.uniform_filter(x, size=win, mode="nearest")
    x = a.astype(np.float64) @ [0.299, 0.587, 0.114]
    y = b.astype(np.float64) @ [0.299, 0.587, 0.114]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ux, uy = f(x), f(y)
    vx, vy = f(x * x) - ux * ux, f(y * y) - uy * uy
    vxy = f(x * y) - ux * uy
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    return float(s.mean())


def reference_accepts_style(cls) -> bool:
    """Can this family's official renderer be driven at a NON-DEFAULT appearance?

    ``mujoco.Renderer`` takes no appearance argument, so an env that wants to be measured under its
    own declared variation has to write the drawn camera and lights into the reference MODEL, and
    whether it does is a per-family fact. It matters because measuring one side at a drawn camera and
    the other at the default reports mean 41.6/255 and SSIM 0.22 — a number about the harness, which
    looks exactly like a catastrophic renderer regression.
    """
    import inspect

    fn = getattr(cls, "reference_render", None)
    if fn is None:
        return False
    try:
        return "style" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def check_visual_fidelity_under_variation(env, n: int = 64, seed: int = 0) -> dict:
    """``check_visual_fidelity`` with a DRAWN appearance per state, from the env's own space.

    The declaration every visual-tier env carries is measured at the DEFAULT appearance, which is the
    right thing for a renderer: a rasterizer that agrees with the official one at one camera agrees at
    another, because it is the same arithmetic pointed somewhere else. That reasoning does not survive
    a LEARNED decoder, and the difference is measured: a corrector fitted at one camera pose scores
    3.25/255 there and 9.06 under the env's own declared camera variation — worse than not correcting
    at all (8.91) and worse than the render it replaces (7.93). A default-appearance declaration would
    have said 3.25 and the frames a consumer got, using the variation knobs this family exists for,
    would have been the worst of the three.

    So this is a separate measurement and it is asserted separately. It is meaningful for any visual
    env whose reference can be driven at an appearance, and it is where a corrector's training
    distribution shows up.
    """
    cls = type(env)
    if not reference_accepts_style(cls):
        return {
            "skipped": f"{cls.__name__}.reference_render does not take a style, so the official "
            f"renderer cannot be driven at this env's declared appearance and the two "
            f"sides would be pictures of different scenes"
        }
    ref = reference_env(cls, env.size)
    rng = np.random.default_rng(int(seed))
    states = env.sample_states(int(n), rng=rng)
    style = env.sample_style(int(n), rng=rng)
    official = cls.reference_render(ref, states, style=style)
    ours = env._render(states, style=style).cpu().numpy()
    if ours.shape != official.shape:
        raise AssertionError(f"shape {ours.shape} vs official {official.shape}")
    out = pixel_agreement(ours, official)
    out["ssim"] = ssim(ours, official)
    out["n_states"] = int(n)
    out["seed"] = int(seed)
    return out


def check_visual_fidelity(env, n: int = 64) -> dict:
    """How close is a VISUAL-tier env's frame to the official one? Measured the way the env declares.

    Deliberately a different function from ``pixel_agreement``, and deliberately reporting different
    headline numbers. A parity env is judged by how many interior pixels are bit-identical, because
    that is a claim about identity. A visual env is a different render of the same scene and will
    never be bit-identical, so it is judged by how far off it is (``mean_abs``), how much of the frame
    is close (``within8_frac``) and whether it reads as the same picture (``ssim``) -- and the
    interior-exact number is still reported, because it is the honest reminder of what the tier gave
    up rather than a threshold anyone should pass.
    """
    cls = type(env)
    ref = reference_env(cls, env.size)
    states = env.sample_states(int(n))
    official = cls.reference_render(ref, states)
    ours = env._render(states).cpu().numpy()
    if ours.shape != official.shape:
        raise AssertionError(f"shape {ours.shape} vs official {official.shape}")
    out = pixel_agreement(ours, official)
    out["ssim"] = ssim(ours, official)
    out["n_states"] = int(n)
    return out


def check_state_roundtrip(env, n: int = 64) -> dict:
    """set_state(get_state()) is the identity, to float32."""
    st = env.sample_states(env.B)
    env._set_render_state(st)
    back = env._get_render_state().cpu().numpy()
    return {"max_abs_err": float(np.abs(back - st).max())}


def check_state_dims_matter(env, n: int = 8, eps: float = 25.0, donors: int = 4) -> dict:
    """Every state dimension must change the frame. A dead dimension is a wrong contract.

    Substitution first, additive nudge second, and a dimension is only declared dead if EVERY attempt
    leaves the frame bit-identical. A single fixed nudge is not a sound probe: it was ``+25.0``, and any
    integer dimension whose renderer wraps it modulo a period dividing 25 is invariant to exactly that
    step. Measured on a 25x25 MiniGrid: ``agent_x``/``agent_y`` were reported dead while being
    demonstrably live, and a 22x22 grid of the same env family reported nothing — the probe's verdict
    depended on the grid width, which is a property of the probe, not of the env.

    Substituting the value from ANOTHER sampled state is the robust form, because it lands somewhere the
    env genuinely reaches: it respects integer alphabets, bounded ranges and wrap-around without knowing
    anything about them. Several donors are tried since one may coincide with the original value.

    THE ONE ESCAPE, AND WHY IT IS NOT A RELAXATION
    ----------------------------------------------

    A dimension can be *mathematically incapable* of changing the image while still being a real part of
    the state, and then "dead" is the wrong word for it. The worked case is ``DMCManipulatorBringBall``:
    ``ball_y`` and the two ``target_ball`` quaternion slots do exactly one thing, which is to ORIENT AN
    UNTEXTURED SPHERE. A sphere is a surface of revolution about every axis through its centre, so an
    analytic per-pixel intersector's output is invariant under rotating it -- not approximately, exactly.
    (MuJoCo's own renderer is not: it draws a tessellated mesh shaded per vertex, so rotating it moves
    the Gouraud interpolation by 734-896 px at <= 55/255. That is the same class of disagreement this
    family already declares as "a one-level rim on curved silhouettes".)

    So a dimension may be EXCLUDED, and the shape of the escape is the whole point:

      * the ENV DECLARES candidates, by name, in ``render_invariant_dims``. A declaration alone excludes
        nothing.
      * THE CHECK VALIDATES each one against the compiled model, through ``render_scene_signature``:
        perturbing the dimension must change ONLY ``geom_xmat`` of geoms the env marks SPHERE, with
        ``geom_xpos``, ``geom_size``, ``geom_type``, every non-sphere ``geom_xmat``, and every material,
        light and camera array bit-unchanged. Anything else and the exclusion is REFUSED and the
        dimension is reported dead exactly as before.
      * an excluded dimension is REPORTED, in ``excluded_dims`` with its reason. It never becomes a
        silent pass, and ``dead_dims`` keeps its old meaning so no existing caller changes behaviour.

    That is why this is a render-contract exception rather than a weaker check. The escape cannot be
    taken by a dimension the renderer merely failed to draw: such a dimension moves a position, a size,
    a colour, or a non-sphere orientation, and every one of those is compared. An env that declares a
    candidate it cannot justify gets a REFUSAL, which is louder than the silence it replaces. And an env
    that offers no hooks is unaffected -- both are optional and absent everywhere but the one cell that
    needs them.

    Measured guard against the obvious objection, that this hides a renderer that ignores rotation
    generally: the same three dimensions are NOT excluded when the target is a BOX
    (``target_ball.shape = 0``), because the box's ``geom_type`` is not SPHERE, the validation refuses,
    and the check finds them live. ``tests/unit/test_dmcontrol_episode_quat.py`` asserts both directions.
    """
    st = env.sample_states(env.B)
    base = env._render(st).cpu().numpy()
    pool = [env.sample_states(env.B) for _ in range(max(1, donors))]
    candidates = set(_render_invariant_candidates(env))
    dead, excluded, refused = [], {}, {}
    for d in range(env.state_dim):
        alive = False
        for donor in pool:
            pert = st.copy()
            pert[:, d] = donor[:, d]
            if np.array_equal(pert[:, d], st[:, d]):
                continue  # donor happened to match; proves nothing either way
            if not np.array_equal(env._render(pert).cpu().numpy(), base):
                alive = True
                break
        if not alive:  # fall back to an additive nudge
            for step in (eps, -eps, 1.0, 0.5):
                pert = st.copy()
                pert[:, d] = pert[:, d] + step
                if not np.array_equal(env._render(pert).cpu().numpy(), base):
                    alive = True
                    break
        if alive:
            continue
        name = env.state_names[d] if env.state_names else str(d)
        if name not in candidates:
            dead.append(name)
            continue
        ok, why = _validate_render_invariant(env, st, d)
        if ok:
            excluded[name] = why
        else:
            refused[name] = why
            dead.append(name)
    out = {"dead_dims": dead}
    if excluded:
        out["excluded_dims"] = excluded
    if refused:
        out["refused_exclusions"] = refused
    return out


def _render_invariant_candidates(env) -> tuple:
    """Dimension names the env OFFERS for exclusion. Optional; absent on every env but one."""
    f = getattr(env, "render_invariant_dims", None)
    if f is None:
        return ()
    return tuple(f() if callable(f) else f)


#: Arrays a render-invariance claim must leave bit-unchanged. Everything a renderer can read off the
#: scene except the one thing the claim is about (``geom_xmat`` on spheres), so the escape is bounded by
#: an enumeration rather than by a promise.
_RENDER_INVARIANT_FROZEN = (
    "geom_xpos",
    "geom_size",
    "geom_type",
    "geom_rgba",
    "geom_matid",
    "geom_names",
    "mat_rgba",
    "mat_reflectance",
    "mat_specular",
    "mat_shininess",
    "mat_texid",
    "light_pos",
    "light_dir",
    "light_diffuse",
    "light_specular",
    "light_ambient",
    "light_castshadow",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
)

#: Frozen fields an env may legitimately omit, with the reason it may. EMPTY, deliberately: this tuple
#: exists so that a future omission has to be argued here, in the rule, rather than being granted by a
#: ``continue``. The enumeration in ``_RENDER_INVARIANT_FROZEN`` is the whole bound on the exclusion,
#: so a field the env does not emit is a field the rule is not checking -- and review demonstrated the
#: false positive that produces: ``thumbtip`` rotates only the two fingertip SPHERES and TRANSLATES
#: them, so with ``geom_xpos`` absent the rule grants an exclusion for a dimension that moves 60 px at
#: max 206/255. A missing key is now a REFUSAL that names the key.
_RENDER_INVARIANT_OPTIONAL: dict[str, str] = {}

#: ``mjtGeom.mjGEOM_SPHERE``. The rule derives sphere-ness from ``geom_type`` ITSELF rather than
#: trusting a flag the env under test supplies, because the flag is an assertion by the party being
#: judged: review showed that ``geom_is_sphere = all True`` grants a BOX target. It matters beyond a
#: hostile case -- ``mjGEOM_ELLIPSOID`` (4) is the sphere-LIKE kind that is emphatically NOT
#: rotation-invariant, and it is now in the registry through the quadruped cell.
_MJGEOM_SPHERE = 2


def _validate_render_invariant(env, states, d: int, variation=None) -> tuple:
    """``(ok, reason)``: may dimension ``d`` be excluded as render-invariant sphere orientation?

    Asks the ENV for a signature of the render-side scene at two states that differ only in ``d``, and
    decides HERE. The env supplies model facts; this function owns the rule, so there is exactly one
    definition of what the exclusion means and it lives with the check that grants it.

    TWO THINGS THE RULE TAKES FROM ITSELF RATHER THAN FROM THE ENV, both because review produced a
    working false positive for each:

      * a frozen field that is MISSING is a refusal, not a skip. The enumeration is the whole bound on
        the exclusion, so an unreported field is an unchecked one -- with ``geom_xpos`` absent, the
        rule granted ``thumbtip``, which rotates only the two fingertip SPHERES but also TRANSLATES
        them and moves 60 px at max 206/255.
      * sphere-ness is DERIVED from ``geom_type``, never read from a ``geom_is_sphere`` the env
        supplies, because that flag is an assertion by the party being judged -- set to all-True it
        granted a BOX target. ``mjGEOM_ELLIPSOID`` is the sphere-like kind that is emphatically not
        rotation-invariant and is now in the registry.

    WHAT THE SIGNATURE DOES NOT YET COVER, recorded so the next cell does not assume it does. The
    enumeration describes geoms, materials, lights and the camera. It does NOT describe (1) SPATIAL
    TENDON geometry, which MuJoCo synthesises per frame from wrap points rather than from any model
    array -- a domain whose excluded dimension could move a tendon needs it added first; nor (2) a
    non-FIXED camera POSE, since ``cam_pos``/``cam_quat``/``cam_fovy`` are model arrays and a
    ``trackcom`` camera's actual pose lives in ``data.cam_xpos``. Both are out of reach of every
    dimension any current cell declares, and both must be added before one that is not.
    """
    sig = getattr(env, "render_scene_signature", None)
    if sig is None:
        return False, (
            "the env declares render_invariant_dims but provides no "
            "render_scene_signature, so the claim cannot be validated"
        )
    row = np.asarray(states[:1]).copy()
    kw = {} if variation is None else {"variation": variation}
    a = sig(row, **kw)
    moved = row.copy()
    # A rotation big enough to be unambiguous, applied to the ONE dimension under test. The state may
    # hold a quaternion component, so the nudge is deliberately small and the check is on the SCENE,
    # not on the state's own algebra.
    moved[0, d] = moved[0, d] + 0.2
    b = sig(moved, **kw)
    if set(a) != set(b):
        return False, "render_scene_signature returned different keys for the two states"
    # EVERY frozen field must be PRESENT, not merely unchanged. A field the env does not emit is a
    # field this rule is not checking, and the enumeration is the entire bound on the exclusion.
    missing = [k for k in _RENDER_INVARIANT_FROZEN if k not in a and k not in _RENDER_INVARIANT_OPTIONAL]
    if missing:
        return False, (
            f"render_scene_signature does not emit {missing}, and the frozen enumeration "
            f"is the whole bound on this exclusion -- a field that is not reported is a "
            f"field that is not checked. Emit it, or add it to "
            f"_RENDER_INVARIANT_OPTIONAL with the reason it cannot change a pixel"
        )
    if "geom_type" not in a:
        return False, "render_scene_signature must emit geom_type; sphere-ness is derived from it"
    # DERIVED HERE, never taken from the env. ``geom_is_sphere`` would otherwise be an assertion by
    # the party being judged, and a wrong one grants an exclusion for a BOX. If the env still emits
    # the flag it must AGREE, so a disagreement is a loud refusal rather than a silently ignored field.
    kind = np.asarray(a["geom_type"]).reshape(-1)
    sphere = kind == _MJGEOM_SPHERE
    claimed = a.get("geom_is_sphere")
    if claimed is not None:
        claimed = np.asarray(claimed, bool).reshape(-1)
        if claimed.shape != sphere.shape or not np.array_equal(claimed, sphere):
            return False, (
                "render_scene_signature's geom_is_sphere disagrees with the geom_type it "
                "reports; sphere-ness is derived from geom_type here and a disagreement "
                "means one of the two is wrong"
            )
    if sphere.size == 0 or not sphere.any():
        return False, "the model has no sphere geoms, so no dimension can be sphere-orientation-only"
    for k in _RENDER_INVARIANT_FROZEN:
        if k not in a:
            continue  # only reachable via _RENDER_INVARIANT_OPTIONAL
        if not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
            return False, (
                f"perturbing this dimension changed {k}, which a claim of sphere-orientation-only invariance forbids"
            )
    xa, xb = np.asarray(a["geom_xmat"]), np.asarray(b["geom_xmat"])
    if xa.shape != xb.shape:
        return False, "geom_xmat changed shape between the two states"
    changed = ~np.all(np.isclose(xa, xb, rtol=0, atol=0), axis=tuple(range(1, xa.ndim)))
    if not changed.any():
        return False, (
            "perturbing this dimension changed nothing in the scene at all, so it is dead "
            "in the MODEL and not merely invisible to the renderer"
        )
    bad = np.flatnonzero(changed & ~sphere)
    if bad.size:
        names = a.get("geom_names") or []
        which = [names[i] if i < len(names) else int(i) for i in bad]
        return False, (
            f"perturbing this dimension rotated NON-sphere geoms {which}, whose orientation "
            f"a correct renderer must draw"
        )
    names = a.get("geom_names") or []
    which = [names[i] if i < len(names) else int(i) for i in np.flatnonzero(changed)]
    return True, (
        f"render-invariant: sphere orientation. Perturbing it rotates only the sphere "
        f"geom(s) {which} and changes nothing else in the render-side scene (geom_xpos, "
        f"geom_size, geom_type, every non-sphere geom_xmat, and every material, light and "
        f"camera array are bit-identical). An analytic per-pixel sphere is exactly invariant "
        f"under rotation about its centre, so no renderer of this design can make this "
        f"dimension visible. MuJoCo's tessellated, per-vertex-shaded sphere is not invariant "
        f"and does move a few hundred pixels at <= 55/255; that difference is declared in the "
        f"env's state_dims_note."
    )


def check_batch_invariance(env, n: int = 8) -> dict:
    """Frame i of a batch equals the frame rendered on its own."""
    st = env.sample_states(env.B)
    batched = env._render(st).cpu().numpy()
    worst = 0
    for i in range(min(n, env.B)):
        one = env._render(st[i : i + 1]).cpu().numpy()[0]
        worst = max(worst, int(np.abs(one.astype(np.int16) - batched[i].astype(np.int16)).max()))
    return {"max_abs_diff": worst}


def check_backend_agreement(env, n: int = 4, max_size: int = 96) -> dict:
    """The fused kernel agrees with the readable torch reference implementation.

    Deliberately run on a few environments at a reduced resolution. ``render_torch`` allocates a
    ``(B, 3, R, R)`` float canvas and touches it once per primitive, so on highway-env's 604-primitive
    roundabout at 600x600 it took minutes and pushed the whole suite past 17. What this check is for is
    whether the two implementations draw the same picture, and that is a property of the arithmetic, not
    of the resolution — a disagreement in an edge rule or a normal direction shows up at 96px on four
    scenes just as surely as at 600px on sixteen, and shows up in seconds.

    **There are two backends only where there is a CUDA device.** ``HAVE_TRITON`` says the
    interpreter can import Triton; it says nothing about where the scene lives, and the fused kernel
    is CUDA-only. Guarding on the import alone and then calling ``render_triton`` unconditionally is
    the same defect ``raster2d.render`` used to have -- on a Triton host a ``device="cpu"``
    env reached ``with torch.cuda.device(scene.kind.device)`` and this check died with
    ``ValueError: Expected a cuda device, but got: cpu``, taking the whole conformance row with it.
    It is a defect here rather than a slow path there: a CPU env has one backend, so the question the
    check asks does not arise, and the honest answer is that it does not apply -- which is what the
    tilemap branch below already returns for the other reason a Scene2D comparison cannot be made.
    The predicate is therefore the scene's device, read the same way ``raster2d.render`` reads it.
    """
    from ..render.raster2d import HAVE_TRITON, render_torch, render_triton

    if not HAVE_TRITON:
        return {"skipped": "triton unavailable"}
    b_n = min(int(n), env.B)
    st = env.sample_states(env.B)[:b_n]
    try:
        sc = env.scene(torch.as_tensor(st, device=env.device))
    except NotImplementedError:
        return {"skipped": "env renders without a Scene2D (tilemap path)"}
    if sc.kind.device.type != "cuda":
        return {
            "skipped": f"the fused kernel is CUDA-only and this scene is on "
            f"{sc.kind.device}; on a CPU device there is one backend, not two"
        }
    size = min(int(env.size), int(max_size))
    a = render_triton(sc, size, env.ss).cpu().numpy()
    b = render_torch(sc, size, env.ss).cpu().numpy()
    return pixel_agreement(a, b)


def _frame_variation_paths(env):
    """Variation leaves that are required to change a frame.

    Dynamics/reset/task leaves (``env.dynamics_variation_leaves``) configure ``g(state, action,
    variation)`` and are not required to move pixels.
    """
    skip = set(getattr(env, "dynamics_variation_leaves", ()) or ())
    for path in env.variation_space.names():
        if path not in skip:
            yield path


def check_variations_do_something(env) -> dict:
    """Every declared *frame* variation leaf must change the picture when moved off its default."""
    st = env.sample_states(env.B)
    base = env._render(st).cpu().numpy()
    inert = []
    for path in _frame_variation_paths(env):
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        default = node.value
        try:
            if hasattr(node, "n"):  # Discrete: flip it
                alt = 0 if int(default) != 0 else 1
            else:  # Box/RGBBox: go to the far end of the range
                lo = np.broadcast_to(node.low, node.shape).astype(np.float64)
                hi = np.broadcast_to(node.high, node.shape).astype(np.float64)
                d = np.asarray(default, np.float64)
                alt = np.where(np.abs(d - lo) > np.abs(d - hi), lo, hi).astype(node.dtype)
            frames = env._render(st, variation={path: alt}).cpu().numpy()
            if np.array_equal(frames, base):
                inert.append(path)
        except Exception as e:  # a variation that cannot be set is inert too
            inert.append(f"{path} ({type(e).__name__}: {e})")
    return {"inert_variations": inert}


def check_per_env_variation(env) -> dict:
    """A variation given one value PER ENVIRONMENT must produce that many different frames.

    Separate from ``check_variations_do_something``, which only moves a knob for the whole batch. Both
    passed while per-env values were silently collapsing to environment 0: a batch of 8 different
    backgrounds rendered as 1 unique frame. That is the repo's headline capability — domain
    randomisation as a tensor rather than a loop — so it gets its own check.
    """
    if env.B < 2:
        return {"skipped": "batch too small"}
    st = env.sample_states(env.B)
    collapsed = []
    for path in _frame_variation_paths(env):
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        if not hasattr(node, "low"):
            continue  # Discrete: nothing continuous to spread
        lo = np.broadcast_to(node.low, node.shape).astype(np.float64)
        hi = np.broadcast_to(node.high, node.shape).astype(np.float64)
        # TWO distinct values, not a B-wide ramp. What is being tested is whether per-environment
        # values stay per-environment, and two groups already falsify a collapse to environment 0.
        # A ramp of B distinct values is not merely wasteful: envs that bake a static plate per
        # distinct value (the OGBench mazes) then run B MuJoCo bakes for ONE knob and thrash the plate
        # cache, which is minutes of wall time per env. It used to be justified by an EGL context
        # budget; that budget was never measured and the measurement does not support it (96 live
        # contexts held, and closing one of them harmless — see the module docstring). The cost
        # argument stands on its own and is the reason this stays at two values.
        half = np.zeros((env.B, 1))
        half[env.B // 2 :] = 1.0
        ramp = half * (hi - lo).reshape(1, -1) + lo.reshape(1, -1)
        try:
            f = env._render(st, variation={path: ramp.astype(node.dtype)}).cpu().numpy()
        except Exception:
            continue  # env may not accept a per-env form for this leaf
        if len(np.unique(f.reshape(env.B, -1), axis=0)) < 2:
            collapsed.append(path)
    return {"collapsed_variations": collapsed}


def check_style_is_state(env) -> dict:
    """``(state, style)`` must fully determine the frame — the claim that lets a dataset store states.

    A fresh env handed the stored pair must reproduce the frame bit-for-bit. If it does not, the frame
    depends on something the dataset did not record, and "regenerate live from states" is false: it
    would reproduce whatever appearance happened to be configured at render time, which is a
    coincidence dressed as reproduction.
    """
    if env.style_dim == 0:
        return {"skipped": "env declares no style"}
    st = env.sample_states(env.B)
    sty = env.sample_style(env.B)
    a = env._render(st, style=sty).cpu().numpy()
    fresh = type(env)(num_envs=env.B, size=env.size, supersample=env.ss, device=str(env.device))
    fresh._set_render_state(st, style=sty)
    b = fresh._render().cpu().numpy()
    return {"max_abs_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}


def check_default_style_is_official(env) -> dict:
    """Rendering with ``default_style()`` must equal rendering with no style at all.

    The default is what parity against the official renderer is measured under, so if the two paths
    disagree then every parity number describes an appearance nobody gets by default.
    """
    if env.style_dim == 0:
        return {"skipped": "env declares no style"}
    st = env.sample_states(env.B)
    a = env._render(st).cpu().numpy()
    b = env._render(st, style=env.default_style()).cpu().numpy()
    return {"max_abs_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}


#: What a ``color_exactness_applicability`` declaration must carry before the harness will honour it.
#: Every one of these is a condition Sami attached to the 2026-09-16 ruling, and the reader refuses a
#: declaration that is missing any of them rather than narrowing the check on free text: an opt-out
#: that costs nothing to write is a way to silence a check, which is the failure mode the ruling names.
#:
#:  ``leaves``      the EXACT leaf paths, listed one by one. No wildcard, no prefix, no "all".
#:  ``mechanism``   the lighting/material law, in words, so a reader knows what is being claimed.
#:  ``max_shading_factor``  the law's MEASURED ceiling, and it must be < 1 -- the number that makes
#:                  literal equality unreachable. A law that reaches 1 has no case to make here.
#:  ``official_renderer_fails_the_same_check``  the demonstration AGAINST the official renderer. The
#:                  claim is about the reference's shading model, so the reference has to fail it too.
#:  ``evidence``    the artifact paths the two numbers above were measured in.
_COLOR_APPLICABILITY_REQUIRED = (
    "leaves",
    "mechanism",
    "max_shading_factor",
    "official_renderer_fails_the_same_check",
    "evidence",
)


def color_exactness_applicability(env, probes=None) -> dict:
    """Which leaves ``check_colors_apply_exactly``'s LITERAL-RGB bound may be asserted on, and why.

    The same shape as ``resolution_consistency_applicability`` and for the same reason. A style leaf
    whose renderer applies a documented lighting/material law that cannot reach unit gain can never
    put the requested triple in the frame -- not because the knob is wired wrongly, but because the
    OFFICIAL renderer cannot either, which is measurable and is measured. Judging a renderer against
    a relation its own reference does not satisfy measures the reference.

    Read, never invented. An env with no ``color_exactness_applicability`` is fully applicable, which
    is every cell but one, and the check's own verdict (``colors_not_applied``) is not touched here:
    the probe still runs, the distance still gets recorded, and only the JUDGE consults this.

    ``probes`` are the ``"<leaf path>-><target>"`` keys the check measured. Returns
    ``{"honoured": bool, "not_applicable_probes": [...], "declared_leaves": [...],
    "undeclared_leaves_still_judged": [...], "declaration": {...} | None, "why": str}``.
    """
    cls = env if isinstance(env, type) else type(env)
    decl = getattr(cls, "color_exactness_applicability", None)
    none = {
        "honoured": False,
        "not_applicable_probes": [],
        "declared_leaves": [],
        "undeclared_leaves_still_judged": sorted(probes or []),
        "declaration": None,
        "why": "no color_exactness_applicability is declared, so literal RGB equality is asserted on every RGB leaf",
    }
    if not isinstance(decl, dict) or not decl:
        return none

    missing = [k for k in _COLOR_APPLICABILITY_REQUIRED if decl.get(k) in (None, "", (), [], {})]
    if missing:
        return {
            **none,
            "declaration": dict(decl),
            "rejected_because": missing,
            "why": f"the declaration is missing {missing}, so it is REFUSED and every RGB leaf "
            f"stays judged; see _COLOR_APPLICABILITY_REQUIRED",
        }
    try:
        ceiling = float(decl["max_shading_factor"])
    except (TypeError, ValueError):
        ceiling = 1.0
    if not (0.0 < ceiling < 1.0):
        return {
            **none,
            "declaration": dict(decl),
            "rejected_because": ["max_shading_factor"],
            "why": f"max_shading_factor {decl['max_shading_factor']!r} is not a measured ceiling "
            f"strictly below 1, so the declaration is REFUSED and every leaf stays judged",
        }
    if decl["official_renderer_fails_the_same_check"] is not True:
        return {
            **none,
            "declaration": dict(decl),
            "rejected_because": ["official_renderer_fails_the_same_check"],
            "why": "the declaration does not assert that the OFFICIAL renderer fails this check "
            "too, so it is REFUSED and every leaf stays judged",
        }

    leaves = [str(x) for x in decl["leaves"]]
    probes = sorted(probes or [])
    covered = [k for k in probes if k.split("->", 1)[0] in leaves]
    rest = sorted({k.split("->", 1)[0] for k in probes} - set(leaves))
    return {
        "honoured": True,
        "not_applicable_probes": covered,
        "declared_leaves": sorted(leaves),
        "undeclared_leaves_still_judged": rest,
        "declaration": dict(decl),
        "why": f"{len(leaves)} named leaves are rendered through {decl['mechanism']}, whose "
        f"measured ceiling is {ceiling:.6f} < 1, so the requested triple is unreachable "
        f"for any correct implementation of the reference and the official renderer fails "
        f"this check too; evidence {list(decl['evidence'])}. Every other RGB leaf on this "
        f"env stays judged ({rest or 'there are none'}).",
    }


def partition_colour_exactness(result: dict) -> dict:
    """Split ``check_colors_apply_exactly``'s failures into the ones a caller may still ASSERT on.

    The check is deliberately unchanged: it probes every RGB leaf and reports every leaf that did not
    receive its colour literally. What an ASSERTION is entitled to do with that list is a different
    question, and this is the one place that answers it, so the GPU suite and the certifier cannot
    drift apart on it.

    ``asserted``          failures on leaves NOTHING declares inapplicable. A caller asserts this is
                          empty, exactly as it always has.
    ``not_applicable``    failures on leaves the class declares, each with the ``best_abs_diff`` the
                          probe measured. REPORTED, never asserted -- but recorded, so a knob that
                          comes off its primitive still moves a number.

    A result with no ``applicability`` (an older artifact, or an env with no declaration) puts every
    failure in ``asserted``: the conservative direction, and the previous behaviour exactly.
    """
    failures = list(result.get("colors_not_applied") or [])
    app = result.get("applicability") or {}
    best = result.get("best_abs_diff") or {}
    if not app.get("honoured"):
        return {
            "asserted": failures,
            "not_applicable": [],
            "honoured": False,
            "why": app.get("why") or "no honoured colour-exactness declaration, so every failing probe is asserted on",
        }
    excused = set(app.get("not_applicable_probes") or [])
    return {
        "asserted": [p for p in failures if p not in excused],
        "not_applicable": [{"probe": p, "best_abs_diff": best.get(p)} for p in failures if p in excused],
        "honoured": True,
        "why": app.get("why"),
    }


def check_colors_apply_exactly(env) -> dict:
    """An RGB knob set to a colour must put THAT colour in the frame, not merely change it.

    ``check_variations_do_something`` only asks whether a knob moves pixels; a knob wired to the wrong
    primitive, or scaled, or with its channels swapped, passes that and is still wrong. This asks
    whether the requested colour is literally present.
    """
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    st = env.sample_states(env.B)
    wrong = []
    best: dict[str, int] = {}
    for path, (off, w) in layout.items():
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        if w != 3 or not hasattr(node, "high") or int(np.max(node.high)) != 255:
            continue  # only RGB leaves have a checkable answer
        for target in ([17, 33, 200], [200, 17, 33]):  # two, so a channel swap cannot pass both
            sty = np.tile(env.default_style(), (env.B, 1))
            sty[:, off : off + w] = np.asarray(target, np.float32)
            f = env._render(st, style=sty).cpu().numpy()
            d = np.abs(f.astype(np.int16) - np.asarray(target, np.int16)).sum(-1)
            # REPORTED, never judged: how close the closest pixel came, in summed absolute channel
            # error. The verdict below is unchanged -- this is the number a leaf whose literal bound
            # is declared inapplicable still moves when a knob comes off its primitive, and the
            # number this check used to throw away, at the colour-ruling condition.
            best[f"{path}->{target}"] = int(d.min())
            if not (d == 0).any():
                wrong.append(f"{path}->{target}")
    return {
        "colors_not_applied": wrong,
        "best_abs_diff": best,
        "applicability": color_exactness_applicability(env, sorted(best)),
    }


#: How close a pixel has to be to count as agreeing inside a task object's footprint. The same 8/255
#: ``pixel_agreement`` already reports as ``within8_frac``, and for the same reason: an object's
#: footprint is mostly SILHOUETTE — a 19-pixel disc at 84 px has almost no interior — so bit-exactness
#: there would be measuring two rasterizers' coverage rules rather than whether the object is present.
#:
#: It feeds the two REPORTED ``presence_agree_*`` numbers and NOT the bound, because a fraction over a
#: silhouette-dominated set is not stable enough to carry one: on ``DMCReacherHard``'s ~15 footprint
#: pixels per frame the declared arm agrees on 60% of them on CPU and 26% on the GPU, which is the
#: rasterizers' edge rule moving and not the object appearing. ``presence_margin`` is built from the
#: mean ERROR instead — see the check.
PRESENCE_TOL = 8


def check_task_object_visibility(env, states=None, n: int = 16, seed: int = 0) -> dict:
    """Is a DECLARED task-relevant object drawn when, and only when, the env says it is — and does
    the official renderer agree? The complement to interior-pixel exactness, which cannot see this.

    WHY THIS EXISTS, measured rather than argued. ``interior_mask`` keeps only pixels whose 3x3
    neighbourhood in the reference is one flat colour, so an object that lands on a TEXTURED surface
    contributes no interior pixels at all and can appear or disappear without moving
    ``interior_exact_frac`` by anything. DM Control reacher is the measured case: two renders
    that differ by a whole red disc — mean |d| 0.459 against 0.171 — report interior exactness equal
    to SIXTEEN digits and an identical ``interior_px``
    (``tests/fixtures/regression/join/step5/reacher/parity_probe.json``). The frame's ``interior_frac`` is 0.000000
    at 84 px, so the strong assertion is not merely blunted there, it is undefined.

    Nothing here redefines or relaxes that metric. This measures a population the interior mask
    excludes BY CONSTRUCTION and asserts a separate bound on it:

    ``footprint``  the pixels our renderer attributes to the object, as the exact set where the frame
                   with the knob at its declared value differs from the frame with it flipped. A
                   footprint of zero means the declaration is inert and nothing else below has a
                   subject.
    ``per_row``    a style whose rows alternate between the two values must reproduce the all-declared
                   render on one set of rows and the all-flipped render on the other, bit for bit.
                   ``check_per_env_variation`` structurally cannot ask this of a ``Discrete`` leaf —
                   it skips anything without ``.low`` — and per-row visibility is the whole point of
                   putting visibility in the style rather than in a constructor argument.
    ``presence``   inside that footprint, how far our frame sits from the OFFICIAL one under each
                   value, POOLED OVER THE WHOLE STATE STACK -- one number per object, not one per
                   row. (``per_row_max_abs_diff`` is the only per-row quantity here, and
                   ``rows_with_footprint`` says how many frames contributed pixels to the pool at
                   all.) Pooling is deliberate: a small object contributes a couple of hundred pixels
                   per frame and a per-frame ratio would be noisy enough to need its own floor, while
                   the failure this catches is a property of the configuration and applies to every
                   frame at once. The env's declared default must be the arm that is CLOSER, by
                   ``presence_margin = 1 - mean|d|(declared) / mean|d|(flipped)``. That is the
                   question no other check in the suite asks: if we paint an object the reference does
                   not paint, the declared arm is the FAR one, the ratio inverts and the margin goes
                   sharply negative, while every existing number stays exactly where it was.

                   A RATIO OF MEAN ERRORS rather than a fraction of agreeing pixels, and the
                   difference is not cosmetic. An object's footprint is mostly silhouette, so the
                   fraction of it that lands within 8/255 is a statement about two rasterizers' edge
                   rules: on ``DMCReacherHard``'s ~15 footprint pixels per frame it is 0.60 on CPU and
                   0.26 on a 3090, for one unchanged renderer. The mean error is not: an object drawn
                   in the same place is a few levels away and an object one side does not draw at all
                   is ~100/255 away, and nothing about an edge rule moves that by a factor of ten.
                   Both fractions are still REPORTED, as ``presence_agree_*``, because they are what a
                   human reads when the margin is marginal.

    Applies to any env that declares a ``visibility.*`` leaf; ``{"skipped": ...}` otherwise, which is
    every env that has no task-relevant object it can be asked to hide.
    """
    paths = [p for p in env.variation_space.names() if p.split(".")[0] == "visibility"]
    if not paths:
        return {"skipped": "the env declares no visibility.* leaf"}
    if states is None:
        states = env.sample_states(int(n), rng=np.random.default_rng(int(seed)))
    states = np.asarray(states)
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    cls = type(env)
    official, why_ref = None, "no official renderer is declared"
    if getattr(cls, "reference_id", None) and hasattr(cls, "reference_env"):
        try:
            # through the process-wide cache, never cls.reference_env directly: a reference env bound
            # to a local here is dropped on return, and for the MuJoCo families that RELEASES an EGL
            # context while other envs still hold live ones. Same rule the parity test follows.
            official = cls.reference_render(reference_env(cls, env.size), states).astype(np.int16)
        except ImportError as e:  # the package is absent HERE, not broken
            why_ref = f"the official renderer is not installed in this interpreter ({e})"

    out: dict = {"n_states": int(len(states)), "tol": int(PRESENCE_TOL), "objects": {}}
    for path in paths:
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        declared = int(np.asarray(node.value).reshape(-1)[0])
        other = 1 - declared
        a = env._render(states, variation={path: declared}).cpu().numpy()
        b = env._render(states, variation={path: other}).cpu().numpy()
        M = (a != b).any(-1)  # the object's own pixels, exactly

        # per-row, through the STYLE vector rather than the variation dict: that is the form a dataset
        # stores and the form a per-environment appearance arrives in.
        off, w = layout[path]
        sty = np.tile(np.asarray(env.default_style(), np.float32), (len(states), 1))
        odd = np.arange(len(states)) % 2 == 1
        sty[:, off : off + w] = float(declared)
        sty[odd, off : off + w] = float(other)
        mix = env._render(states, style=sty).cpu().numpy()
        per_row = int(
            max(
                np.abs(mix[~odd].astype(np.int16) - a[~odd].astype(np.int16)).max(initial=0),
                np.abs(mix[odd].astype(np.int16) - b[odd].astype(np.int16)).max(initial=0),
            )
        )

        row = {
            "declared_value": declared,
            "footprint_px": int(M.sum()),
            "footprint_frac": float(M.mean()),
            "rows_with_footprint": int((M.reshape(len(states), -1).any(1)).sum()),
            "per_row_max_abs_diff": per_row,
        }
        if official is not None and M.any():
            da = np.abs(a.astype(np.int16) - official).max(-1)[M]
            db = np.abs(b.astype(np.int16) - official).max(-1)[M]
            row["presence_agree_declared"] = float((da <= PRESENCE_TOL).mean())
            row["presence_agree_flipped"] = float((db <= PRESENCE_TOL).mean())
            row["presence_mean_abs_declared"] = float(da.mean())
            row["presence_mean_abs_flipped"] = float(db.mean())
            # 1 - near/far. One when the declared arm is exact, 0 when the two are equally wrong,
            # and sharply negative when they are the wrong way round. The denominator cannot be zero
            # unless the flipped arm is bit-exact on the object's own pixels, which would mean the
            # object is not there at all -- reported as a margin of -inf via the guard below rather
            # than divided by.
            far = float(db.mean())
            row["presence_margin"] = (1.0 - float(da.mean()) / far) if far > 0 else None
            if far <= 0:
                row["why_no_margin"] = (
                    "the FLIPPED appearance is bit-exact against the official frame on the object's "
                    "own pixels, so the official renderer is drawing the flipped configuration and "
                    "the declared default is the wrong one"
                )
            row["reference_id"] = cls.reference_id
        else:
            row["presence_margin"] = None
            row["why_no_reference"] = (
                why_ref if official is None else "the object has an empty footprint on these states"
            )
        out["objects"][path] = row
    return out


def check_determinism(env) -> dict:
    st = env.sample_states(env.B)
    return {"identical": bool(np.array_equal(env._render(st).cpu().numpy(), env._render(st).cpu().numpy()))}


#: How an env declares the reference the step checker should drive. Every entry is optional except
#: ``reference_sim_state``, because there is no defensible guess for "where does this reference keep
#: its state" — a guessed attribute name gives a silent mismatch, which costs more than writing four
#: lines. The rest fall back to the shapes that are actually common.
STEP_REFERENCE_HOOKS = """
    sim_state_names        tuple[str, ...]        the FULL dynamical state being compared, which is
                                                  usually a SUPERSET of state_names (velocities are
                                                  not drawn, so they are not in the render state).
                                                  Defaults to state_names.
    sim_state()            -> (B, D) tensor       read it off the GPU env. Defaults to get_state().
    set_sim_state(v)       (B, D)                 write it. Defaults to set_state(v).
    reference_sim_state(ref) -> (D,) float64      REQUIRED. Read it off the reference.
    reference_step(ref, a)                        advance the reference one control step.
                                                  Defaults to ref.step(a).
    step_reference_reset(ref, seed) -> (D,)       reset and read. Defaults to reset(seed=)+read.
    step_reference_actions(rng, steps, hold)      the action sequence, (steps, action_dim). Defaults
                                                  to uniform over [action_low, action_high), resampled
                                                  every ``hold`` steps.
    step_reference_groups  ((name, dims, metric),...)  how the error is reported. metric is 'abs'
                                                  (max |diff| over the dims), 'euclidean' (vector
                                                  distance) or 'angle' (wrapped). Defaults to one
                                                  'abs' group per dimension, named after it.
    step_reference_control str                    the group that must agree to FLOAT PRECISION over
                                                  every step because nothing chaotic acts on it.
                                                  Defaults to none, and then the control is the
                                                  first-step error, which no divergence can inflate.
    step_reference_gate    str                    the group whose motion makes an episode informative;
                                                  episodes where it never moved are excluded from the
                                                  final-step statistics and counted. Defaults to none.
    step_reference_hold    int                    default ``hold``.
    step_reference_exact   bool                   the env CLAIMS bit-exactness, so exact_frac is the
                                                  headline number and anything below 1.0 is a failure.
"""


def _wrap_pi(d: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * d)))


def _group_error(diff: np.ndarray, dims, metric: str) -> np.ndarray:
    """``(episodes, steps, D)`` signed differences -> ``(episodes, steps)`` error for one group."""
    d = diff[..., list(dims)]
    if metric == "euclidean":
        return np.linalg.norm(d, axis=-1)
    if metric == "angle":
        return _wrap_pi(d).max(-1)
    return np.abs(d).max(-1)


def _default_actions(cls, rng, steps: int, hold: int) -> np.ndarray:
    adim = int(getattr(cls, "action_dim", 1))
    lo = np.broadcast_to(np.asarray(getattr(cls, "action_low", -1.0), np.float64), (adim,))
    hi = np.broadcast_to(np.asarray(getattr(cls, "action_high", 1.0), np.float64), (adim,))
    out = np.empty((int(steps), adim), np.float32)
    a = None
    for i in range(int(steps)):
        if i % max(1, int(hold)) == 0:
            a = rng.uniform(lo, hi).astype(np.float32)
        out[i] = a
    return out


def check_step_matches_reference(
    env, steps: int = 100, episodes: int = 16, seed: int = 0, hold: int | None = None
) -> dict:
    """How far a batched GPU ``step`` drifts from the official CPU simulator on identical actions.

    Rendering parity is a bit-exactness claim; simulation parity is only sometimes one, and the two
    must never be stated in the same words. Three regimes show up in this repo and this function
    reports all three the same way, so they can be compared:

      * **exact.** A deterministic discrete transition (MiniGrid's grid) can be bit-identical to the
        reference, and then the number to quote is ``exact_frac == 1.0`` over many (state, action)
        pairs — not a tolerance.
      * **near-exact.** A published closed-form integrator (the classic-control family) reimplemented
        in the same float64 arithmetic differs only where CUDA's ``sin``/``cos`` differ from glibc's by
        an ulp. The number to quote is a max deviation, and it is tiny.
      * **not exact, and cannot be.** Contact is chaotic (PushT), so two solvers that disagree by a
        micron at first touch disagree by centimetres later. The number to quote is a distribution over
        episodes plus a control that chaos cannot inflate.

    The reference is whatever the env DECLARES, through the hooks in ``STEP_REFERENCE_HOOKS``; this
    function knows nothing about any particular env. Returned keys:

      ``<group>_max_err``          max over every step and episode. The strongest statement.
      ``<group>_mean_err``         mean at the final step, over the episodes the gate kept.
      ``<group>_median_err``       median at the final step, ``<group>_p90_err`` its 90th percentile,
                                   ``<group>_median_err_at_25`` the same median a quarter of the way in.
      ``first_step_max_err``       the INTEGRATOR check: one step from a state the reference itself
                                   produced, so no amount of divergence can be hiding in it.
      ``control_max_err``          the declared control group over all steps, or the first-step error
                                   when an env declares no control.
      ``exact_frac``               fraction of compared numbers that are bit-identical.
      ``<gate>_touched``           episodes where the gate group actually moved. One that never moved
                                   agrees trivially and is evidence of nothing.

    ``agent_max_err`` and ``block_median_err`` are also always present. They are PushT's own group
    names, and the shared pytest asserts on them, so they are aliased to ``control_max_err`` and the
    overall final-step median for envs that have no such groups — one env's vocabulary should not
    decide whether another env can be checked at all.
    """
    from stable_envs._dynamics import has_dynamics

    cls = type(env)
    if not has_dynamics(env):
        raise ValueError(
            f"{cls.__name__} has no step(); there is no solver to compare. Envs that ship "
            f"a renderer alone are covered by the render-parity checks instead."
        )
    if not hasattr(cls, "reference_sim_state"):
        raise ValueError(
            f"{cls.__name__} ships a step() but declares no reference_sim_state(ref), so this check "
            f"cannot read the reference's state. Guessing an attribute name is what this function used "
            f"to do and it produces a silent mismatch rather than an error. Declare:\n"
            f"{STEP_REFERENCE_HOOKS}"
        )

    names = tuple(getattr(cls, "sim_state_names", None) or cls.state_names)
    D = len(names)
    groups = tuple(getattr(cls, "step_reference_groups", None) or tuple((n, (i,), "abs") for i, n in enumerate(names)))
    control = str(getattr(cls, "step_reference_control", "") or "")
    gate = str(getattr(cls, "step_reference_gate", "") or "")
    hold = int(getattr(cls, "step_reference_hold", 8) if hold is None else hold)
    steps, episodes = int(steps), int(episodes)

    read = cls.reference_sim_state
    advance = getattr(cls, "reference_step", None) or (lambda ref, a: ref.step(a))
    reset = getattr(cls, "step_reference_reset", None) or (
        lambda ref, sd: (ref.reset(seed=sd), np.asarray(read(ref), np.float64))[1]
    )
    make_acts = getattr(cls, "step_reference_actions", None) or (lambda rng, n, h: _default_actions(cls, rng, n, h))

    ref = reference_env(cls, env.size)
    rng = np.random.default_rng(seed)
    dev = env.device
    single = cls(num_envs=1, size=env.size, device=str(dev))
    get_sim = getattr(single, "sim_state", single._get_render_state)
    set_sim = getattr(single, "set_sim_state", single._set_render_state)

    diffs, exact, touched = [], [], []
    for ep in range(episodes):
        s0 = np.asarray(reset(ref, seed + ep), np.float64).reshape(D)
        acts = np.asarray(make_acts(rng, steps, hold), np.float32).reshape(steps, -1)

        cpu = np.empty((steps, D), np.float64)
        for i, a in enumerate(acts):
            advance(ref, a)
            cpu[i] = np.asarray(read(ref), np.float64).reshape(D)

        set_sim(torch.as_tensor(s0[None], dtype=torch.float64, device=dev))
        gpu = np.empty((steps, D), np.float64)
        for i, a in enumerate(acts):
            # ``step`` is stateful; what it RETURNS differs by cell (a StepResult TensorDict for a
            # migrated env, a packed tensor for the rest), so the reading is done by the
            # ``get_sim`` the caller supplied, which is the packed row on every path.
            single.step(torch.as_tensor(a, dtype=torch.float32, device=dev).view(1, -1), capture="states")
            gpu[i] = np.asarray(get_sim().detach().cpu().numpy(), np.float64).reshape(D)

        diffs.append(gpu - cpu)
        exact.append(gpu == cpu)
        if gate:
            g_dims = list(dict((n, d) for n, d, _ in groups)[gate])
            touched.append(bool(np.linalg.norm(cpu[-1, g_dims] - s0[g_dims]) > 1e-6))
    # The reference is NOT closed. It comes from the process-wide cache above, so closing it here
    # would (a) release an EGL context while other envs hold live ones, which the module docstring
    # forbids, and (b) hand the next caller a closed env out of the cache.

    diff = np.stack(diffs)  # (episodes, steps, D)
    m = np.array(touched, bool) if gate else np.ones(episodes, bool)
    sel = m if m.any() else np.ones_like(m)
    at25 = min(24, steps - 1)

    out = {
        "env": cls.__name__,
        "episodes": episodes,
        "steps": steps,
        "exact_claim": bool(getattr(cls, "step_reference_exact", False)),
        "exact_frac": float(np.stack(exact).mean()),
    }
    if gate:
        out[f"{gate}_touched"] = int(m.sum())
    per_group, first = {}, 0.0
    for name, dims, metric in groups:
        e = _group_error(diff, dims, metric)  # (episodes, steps)
        per_group[name] = e
        first = max(first, float(e[:, 0].max()))
        out[f"{name}_max_err"] = float(e.max())
        out[f"{name}_mean_err"] = float(e[sel][:, -1].mean())
        out[f"{name}_median_err"] = float(np.median(e[sel][:, -1]))
        out[f"{name}_p90_err"] = float(np.percentile(e[sel][:, -1], 90))
        out[f"{name}_median_err_at_25"] = float(np.median(e[sel][:, at25]))
    allg = np.stack(list(per_group.values())).max(0)  # worst group, per (episode, step)
    out["first_step_max_err"] = first
    out["max_err"] = float(allg.max())
    out["median_err"] = float(np.median(allg[sel][:, -1]))
    out["control_max_err"] = float(per_group[control].max()) if control else first
    # PushT's group names, which the shared pytest asserts on. Present for every env so that adding a
    # steppable env does not require the test to know its vocabulary.
    out.setdefault("agent_max_err", out["control_max_err"])
    out.setdefault("block_median_err", out["median_err"])
    return out


# ===================================================================== reproducibility
#
# Everything below asks the same question from a different angle: is the frame a function of
# ``(state, style, size)`` and NOTHING ELSE? The parity checks above compare our pixels against the
# official renderer's, which is a statement about correctness on one machine, in one process, at one
# batch size. The claim the repo actually makes to a user is stronger and different — delete your
# frames, keep the states, and get the same pixels back later, on another machine, in another process,
# in whatever batch your loader happens to produce. Each check here is one way that can be false while
# every parity number stays perfect.


def uniform_style(env, n: int, groups: int = 2, seed: int = 0):
    """``(n, style_dim)`` style with at most ``groups`` DISTINCT rows, or None if the env has no style.

    Bounded on purpose. Envs that bake a static plate per distinct appearance (the OGBench mazes) pay
    a full MuJoCo render per distinct row, so an n-wide ramp turns a one-second check into minutes and
    evicts the plate cache for everything after it. Two groups already falsify every property these
    checks are about: a batch that collapses to environment 0, an index that is permuted, a group that
    is scattered back into the wrong rows.
    """
    if env.style_dim == 0:
        return None
    rng = np.random.default_rng(seed)
    base = env.sample_style(int(groups), rng=rng)
    idx = np.arange(n) % int(groups)
    return base[idx]


def check_batch_size_invariance(env, states, style=None, sizes=(1, 7, 64), probes: int = 4) -> dict:
    """Frame ``i`` must not depend on how many other environments were rendered beside it.

    ``check_batch_invariance`` already compares a batch against single renders. This extends it to the
    sizes a real loader produces — including odd ones, because a kernel whose grid is computed as
    ``cdiv(H*W, BLOCK)`` and indexed with a 32-bit offset is exactly the kind of thing that is correct
    at 1, 64 and 1024 and wrong at 7. ``states`` is the full stack; every probe row is re-rendered
    inside a window of each size and compared against its row in the full-batch render.
    """
    n = len(states)
    full = env._render(states, style=style).detach().cpu().numpy()
    idx = np.linspace(0, n - 1, min(int(probes), n)).astype(int)
    worst, where = 0, ""
    for b in sizes:
        b = min(int(b), n)
        for i in idx:
            lo = max(0, min(int(i) - b // 2, n - b))
            sub = states[lo : lo + b]
            k = None if style is None else style[lo : lo + b]
            got = env._render(sub, style=k).detach().cpu().numpy()[int(i) - lo]
            d = int(np.abs(got.astype(np.int16) - full[int(i)].astype(np.int16)).max())
            if d > worst:
                worst, where = d, f"row {int(i)} in a batch of {b}"
    return {"max_abs_diff": worst, "worst_case": where, "batch_sizes": tuple(sizes), "n": n}


def check_chunked_render_matches(env, states, style=None, chunks=(1, 7, 64, 4096)) -> dict:
    """``render_states(..., chunk=c)`` must not depend on ``c``. The dataset path, not the env path.

    This is the entry point ``StateDataset`` and ``TrajectoryDataset.all_frames`` go through, and the
    chunk boundary is where a stale ``out`` view, a style row that is sliced with the wrong index, or a
    non-blocking copy that is read before it lands would show up — none of which the single-call render
    can reach.
    """
    from .._data import render_states

    ref = None
    worst, where = 0, ""
    for c in chunks:
        got = (
            render_states(env, states, size=env.size, chunk=int(c), style=style, device=str(env.device))
            .detach()
            .cpu()
            .numpy()
        )
        if ref is None:
            ref = got
            continue
        d = int(np.abs(got.astype(np.int16) - ref.astype(np.int16)).max())
        if d > worst:
            worst, where = d, f"chunk={c} vs chunk={chunks[0]}"
    return {"max_abs_diff": worst, "worst_case": where, "chunks": tuple(chunks)}


def _box_downsample(frames: np.ndarray, f: int) -> np.ndarray:
    """``(N, f*H, f*W, 3)`` uint8 -> ``(N,H,W,3)`` by an exact ``f x f`` box average, rounded once."""
    n, H, W, c = frames.shape
    x = frames.astype(np.float64).reshape(n, H // f, f, W // f, f, c)
    return np.clip(np.rint(x.mean(axis=(2, 4))), 0, 255).astype(np.uint8)


#: The protocol ``tests/gpu/test_reproducibility.py::DOWNSAMPLE_EXACT_FRAC`` was measured under, as
#: the test module's ``MEASURED_ON`` states it: "8 states per env from
#: sample_states(rng=default_rng(render_worker.STATE_SEED)), default appearance" -- i.e. that module's
#: ``inputs`` fixture, ``render_worker.sample_inputs(env, N)`` with ``N = 8``. The floor is a property
#: of a STATE SET as much as of a renderer: how much speckled sky and far floor is in view decides how
#: many pixels survive a 2x box filter, and the same renderer measures 0.6188 on one 64-state corpus,
#: 0.6241 on the first 8 of it and 0.6290 on these 8 (walker, both mujoco 3.11 and 3.12; see
#: ``tests/fixtures/regression/join/step6/texture-fidelity/findings.md`` section 6). A bound applied on a different
#: draw with a 0.005 margin fails for the draw and not for the renderer, which is why the check
#: carries its own protocol here rather than borrowing whatever states the caller has.
DOWNSAMPLE_PROTOCOL_N = 8


def downsample_protocol(states=None) -> dict:
    """The recorded protocol of the downsample bound, as a record a shard can carry beside the number.

    Pass the drawn states to pin them: now that the judged number is no longer measured on the frozen
    corpus (whose sha256 the shard already records), the draw itself has to be identifiable, so a
    later run can show it measured the same eight states and not a re-seeded ``sample_states``.
    """
    from . import render_worker as W

    out = {
        "n": DOWNSAMPLE_PROTOCOL_N,
        "state_seed": W.STATE_SEED,
        "draw": "render_worker.sample_inputs(env, n)",
        "style": "default appearance",
        "bound": "tests/gpu/test_reproducibility.py:DOWNSAMPLE_EXACT_FRAC (MEASURED_ON)",
    }
    if states is not None:
        out["states_sha256"] = frames_digest(np.ascontiguousarray(states, np.float32))
        out["n_states"] = int(len(states))
    return out


def downsample_protocol_states(env) -> np.ndarray:
    """The states ``DOWNSAMPLE_EXACT_FRAC`` was measured on for ``env``: the test fixture's own draw."""
    from . import render_worker as W

    states, _ = W.sample_inputs(env, DOWNSAMPLE_PROTOCOL_N)
    return np.ascontiguousarray(states, np.float32)


#: Keys of a ``resolution_consistency_applicability`` declaration that describe the declaration rather
#: than naming a band. Everything else in the dict is read as a band condition.
_APPLICABILITY_METADATA = frozenset(
    {
        "evidence",
        "measured_against",
        "mechanisms",
        "scope_note",
        "still_required",
        "note",
        "notes",
        "reference_versions",
    }
)

#: A band whose value starts with this is one the relation itself does not hold in, on the official
#: renderer's own evidence. Anything else is a band the bound is asserted in.
_NOT_APPLICABLE_PREFIX = "not_applicable"

#: The band condition that means "the whole cell", not a subset of its states.
_ALL_BANDS = frozenset({"all bands", "all", "every band"})

_APPLICABILITY_OPS = {
    ">=": lambda x, v: x >= v,
    "<=": lambda x, v: x <= v,
    ">": lambda x, v: x > v,
    "<": lambda x, v: x < v,
    "==": lambda x, v: x == v,
    "!=": lambda x, v: x != v,
}


def resolution_consistency_applicability(env, states=None) -> dict:
    """Which of ``states`` ``check_resolution_consistency``'s bound may be asserted on, and why.

    Two of the cells declare, on the class, that the 2x-render-and-box-downsample RELATION does
    not hold in some or all of their state space -- not because this renderer is wrong there, but
    because the OFFICIAL renderer is not resolution-consistent there either. Craftax draws its night
    noise field at the frame's own resolution, so the 2S field is a different field rather than a finer
    sampling of the same one; Craftax Full additionally resamples every texture PNG with
    ``Image.NEAREST`` at each size, so its 2S atlas is not a 2x replication of its S one. Both
    declarations carry the oracle experiment that measured it on the official renderer.

    Judging a renderer against a relation its own reference does not satisfy measures the reference,
    so the harness has to be able to read that declaration rather than discover it as a failure. It is
    read, never invented: an env with no ``resolution_consistency_applicability`` is fully applicable,
    which is every cell but two, and no bound value is touched here -- ``DOWNSAMPLE_EXACT_FRAC`` and
    ``DOWNSAMPLE_MARGIN`` stay exactly what the frozen fidelity spec records.

    A band is a key of the form ``"<state name> <op> <value>"`` resolved against ``env.state_names``,
    or one of ``"all bands"``/``"all"``/``"every band"`` for the whole cell. A band whose value begins
    ``not_applicable`` is one the relation does not hold in. A state is judged when it is inside at
    least one applicable band and outside every inapplicable one, so a declaration that names only the
    inapplicable band (which is the shape CraftaxClassic uses in spirit, and both shapes work) still
    leaves the complement judged.

    Returns ``{"applicable": bool, "why": str, "indices": list[int] | None, "declaration": dict,
    "bands": {...}}``. ``indices is None`` means "every state" -- the ordinary case, and the case a
    caller may skip subsetting for.
    """
    decl = getattr(type(env), "resolution_consistency_applicability", None)
    full = {
        "applicable": True,
        "why": "no resolution_consistency_applicability is declared, so the bound is asserted on every state",
        "indices": None,
        "declaration": None,
        "bands": {},
    }
    if not isinstance(decl, dict) or not decl:
        return full

    cited = {k: decl[k] for k in sorted(decl) if k in _APPLICABILITY_METADATA}
    bands = {k: v for k, v in decl.items() if k not in _APPLICABILITY_METADATA}
    out_bands = {
        k: ("not_applicable" if str(v).strip().lower().startswith(_NOT_APPLICABLE_PREFIX) else "applicable")
        for k, v in bands.items()
    }

    # "all bands: not_applicable" is the whole cell, and it is answered before any state is looked at:
    # there is no subset to fall back to and no draw to depend on.
    for k, v in bands.items():
        if k.strip().lower() in _ALL_BANDS and out_bands[k] == "not_applicable":
            return {
                "applicable": False,
                "indices": [],
                "declaration": cited,
                "bands": out_bands,
                "why": f"the class declares '{k}': {v}",
            }

    if states is None:
        return {
            **full,
            "declaration": cited,
            "bands": out_bands,
            "why": "band conditions are declared but no states were supplied to resolve them",
        }

    x = np.asarray(states, np.float64)
    names = list(getattr(env, "state_names", None) or [])
    applicable = np.zeros(len(x), bool)
    excluded = np.zeros(len(x), bool)
    unresolved = []
    for k, v in bands.items():
        if k.strip().lower() in _ALL_BANDS:
            (applicable if out_bands[k] == "applicable" else excluded)[:] = True
            continue
        parts = k.split()
        if len(parts) != 3 or parts[1] not in _APPLICABILITY_OPS or parts[0] not in names:
            unresolved.append(k)
            continue
        col = x[:, names.index(parts[0])]
        try:
            mask = _APPLICABILITY_OPS[parts[1]](col, float(parts[2]))
        except ValueError:
            unresolved.append(k)
            continue
        (applicable if out_bands[k] == "applicable" else excluded)[:] |= np.asarray(mask, bool)
    if unresolved:
        # A band nobody can resolve must not silently narrow or widen the judged set. Reporting it and
        # judging everything is the conservative direction: the bound stays asserted where it was.
        return {
            **full,
            "declaration": cited,
            "bands": out_bands,
            "why": f"band conditions {unresolved} could not be resolved against state_names, so "
            f"the bound is asserted on every state",
        }

    # A declaration that names ONLY inapplicable bands leaves their complement judged.
    if not any(status == "applicable" for status in out_bands.values()):
        applicable = ~excluded
    keep = np.flatnonzero(applicable & ~excluded)
    inapplicable = sorted(k for k, s in out_bands.items() if s == "not_applicable")
    if len(keep) == 0:
        return {
            "applicable": False,
            "indices": [],
            "declaration": cited,
            "bands": out_bands,
            "why": f"every drawn state falls in a band the class declares inapplicable ({inapplicable}); {cited}",
        }
    if len(keep) == len(x):
        return {
            "applicable": True,
            "indices": None,
            "declaration": cited,
            "bands": out_bands,
            "why": "no drawn state falls in a band the class declares inapplicable",
        }
    return {
        "applicable": True,
        "indices": [int(i) for i in keep],
        "declaration": cited,
        "bands": out_bands,
        "why": f"{len(keep)} of {len(x)} states are outside the bands the class declares "
        f"inapplicable ({inapplicable}); the bound is asserted on those and the rest are "
        f"recorded not_applicable with the class's cited evidence",
    }


def check_resolution_consistency(env, states, style=None, factor: int = 2) -> dict:
    """Render at ``factor x S`` and box-downsample to ``S``; how far is that from rendering at ``S``?

    These will NOT be identical and it would be wrong to assert that they are. Rendering at S asks
    "what colour covers this pixel's centre"; rendering at 2S and averaging asks "what is the mean
    colour over this pixel's area". They differ exactly where a shape edge crosses a pixel, which is
    the same place two rasterizers are entitled to disagree, and the difference is a real property of
    point sampling rather than a defect. The tile-atlas families differ for a second reason: their
    atlas is resampled from the source art at the requested tile size, so 2S is a genuinely different
    picture and not a supersampled one.

    What is worth asserting is that the disagreement stays where geometry is, so the numbers returned
    are the ones a bound can be written against: the fraction of pixels that agree exactly, the mean
    absolute difference, and the worst single channel. A regression that moves a shape shows up in
    ``exact_frac`` immediately; anti-aliasing does not move it at all.
    """
    S = int(env.size)
    a = env._render(states, size=S, style=style).detach().cpu().numpy()
    big = env._render(states, size=S * int(factor), style=style).detach().cpu().numpy()
    # A render at S, deliberately last. No env stores the requested size any more — the highway family
    # did, and one differently-sized call there rewrote every later frame and cascaded into four
    # unrelated dataset round-trip failures — so this is now a belt rather than a repair, and it is
    # kept because the cost is one frame and the failure it guards against is invisible at the point
    # where it lands.
    env._render(states[:1], size=S, style=None if style is None else style[:1])
    if big.shape[-3] != a.shape[-3] * factor or big.shape[-2] != a.shape[-2] * factor:
        return {"skipped": f"size={S * factor} is not a {factor}x frame ({big.shape} vs {a.shape})"}
    d = np.abs(_box_downsample(big, factor).astype(np.int16) - a.astype(np.int16))
    return {
        "exact_frac": float((d.max(-1) == 0).mean()),
        "mean_abs": float(d.mean()),
        "max_abs": int(d.max()),
        "size": S,
        "factor": int(factor),
    }


def check_input_form_invariance(env, states, style=None) -> dict:
    """numpy or torch, CPU or CUDA, float32 or float64, contiguous or not — one answer.

    Every one of these is a real call site. A loader hands numpy; a training step hands a CUDA tensor;
    a corpus loaded from parquet is float64 until someone narrows it; and ``frames[:, t]`` — a strided
    view into a ``(B, T, ...)`` buffer — is how a rollout indexes time, which has already crashed this
    repo once. The states are generated in float32 and widened, never narrowed, so float64 carries the
    same values exactly and a difference is the code path, not the arithmetic.
    """
    ref = env._render(states, style=style).detach().cpu().numpy()
    dev = env.device
    st32 = np.ascontiguousarray(np.asarray(states, np.float32))
    n, D = st32.shape

    # a non-contiguous numpy view with a real gap between rows, and its torch equivalent
    pad = np.zeros((n, 2, D), np.float32)
    pad[:, 0] = st32
    np_strided = pad[:, 0]
    t_pad = torch.zeros((n, 2, D), dtype=torch.float32, device=dev)
    t_pad[:, 0] = torch.as_tensor(st32, device=dev)
    cuda_strided = t_pad[:, 0]

    forms = {
        "numpy_f32": st32,
        "numpy_f64": st32.astype(np.float64),
        "numpy_strided": np_strided,
        "list": st32.tolist(),
        "torch_cpu_f32": torch.as_tensor(st32),
        "torch_cpu_f64": torch.as_tensor(st32).double(),
        "torch_cuda_f32": torch.as_tensor(st32, device=dev),
        "torch_cuda_f64": torch.as_tensor(st32, device=dev).double(),
        "torch_cuda_strided": cuda_strided,
        "torch_cuda_transposed": torch.as_tensor(st32, device=dev).t().contiguous().t(),
    }
    bad = {}
    for name, s in forms.items():
        try:
            got = env._render(s, style=style).detach().cpu().numpy()
        except Exception as e:  # noqa: BLE001
            bad[name] = f"{type(e).__name__}: {e}"
            continue
        d = int(np.abs(got.astype(np.int16) - ref.astype(np.int16)).max())
        if d:
            bad[name] = f"differs by {d}/255"
    return {"disagreements": bad, "forms": tuple(forms)}


def check_order_independence(env, states, style=None, seed: int = 0) -> dict:
    """``render(x[perm]) == render(x)[perm]``, and a duplicated row renders the same twice.

    A rasterizer that reads its geometry with the wrong batch stride, a grouped path that scatters
    results back with a stale index, an ``out`` buffer left partly unwritten — all three are invisible
    when every environment in the batch is asking for the same picture, and all three are caught here.
    Run with two distinct style groups so the grouping path is exercised: the OGBench renderer splits
    the batch by plate and writes each group back through an index, which is precisely the code an
    unpermuted batch never reaches.
    """
    n = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    base = env._render(states, style=style).detach().cpu().numpy()
    permuted = (
        env._render(np.asarray(states)[perm], style=None if style is None else np.asarray(style)[perm])
        .detach()
        .cpu()
        .numpy()
    )
    d_perm = int(np.abs(permuted.astype(np.int16) - base[perm].astype(np.int16)).max())

    # the same state twice in one batch must give the same frame twice
    dup = np.concatenate([np.asarray(states)[:1], np.asarray(states)[:1]], 0)
    kdup = None if style is None else np.concatenate([np.asarray(style)[:1]] * 2, 0)
    f = env._render(dup, style=kdup).detach().cpu().numpy()
    d_dup = int(np.abs(f[0].astype(np.int16) - f[1].astype(np.int16)).max())
    return {"permutation_max_abs_diff": d_perm, "duplicate_max_abs_diff": d_dup}


def check_render_is_pure(env, states, style=None) -> dict:
    """Rendering must not change the env, and restoring a state twice must be the same as once.

    The env holds ``_state`` and ``_style`` buffers that ``render`` reads. If a render writes to either
    — to normalise something, to cache something — then the second render of the same states differs
    from the first, every downstream comparison in this file becomes order-dependent, and a training
    loop that renders twice per step silently drifts. ``set_state(get_state())`` applied twice is the
    same claim from the other side: it is the operation a dataset performs on every batch.
    """
    env._set_render_state(
        np.asarray(states)[: env.B] if len(states) >= env.B else np.repeat(np.asarray(states)[:1], env.B, 0),
        style=None
        if style is None
        else (np.asarray(style)[: env.B] if len(style) >= env.B else np.repeat(np.asarray(style)[:1], env.B, 0)),
    )
    s0 = env._get_render_state().detach().cpu().numpy().copy()
    k0 = env.get_style().detach().cpu().numpy().copy()
    f0 = env._render().detach().cpu().numpy()

    env._render(states, style=style)  # an unrelated render, with explicit arguments
    s1 = env._get_render_state().detach().cpu().numpy().copy()
    k1 = env.get_style().detach().cpu().numpy().copy()

    env._set_render_state(env._get_render_state())
    env._set_render_state(env._get_render_state())
    s2 = env._get_render_state().detach().cpu().numpy().copy()
    f2 = env._render().detach().cpu().numpy()
    # `k0`/`k1` are `(B, style_dim)` and `style_dim` is legitimately 0 -- this file already guards
    # that case in `check_style_dims_matter`, `check_style_roundtrip` and `_grouped_style`. Without
    # the same guard here, `.max()` of an empty array raises "zero-size array to reduction operation
    # maximum which has no identity" and a styleless env cannot pass a check about RENDER PURITY that
    # has nothing to do with style. An env with no style trivially cannot have its style changed by a
    # render, so the honest answer for that case is 0.0, not a crash.
    return {
        "state_changed_by_render": float(np.abs(s1 - s0).max()),
        "style_changed_by_render": float(np.abs(k1 - k0).max()) if k0.size else 0.0,
        "state_changed_by_double_set": float(np.abs(s2 - s0).max()),
        "frame_changed_by_double_set": int(np.abs(f2.astype(np.int16) - f0.astype(np.int16)).max()),
    }


def check_set_state_copies(env) -> dict:
    """``set_state`` must COPY. Two consequences, both silent, both expensive.

    A caller that keeps the array it passed and then mutates it would otherwise be mutating the env's
    state — the corpus and the env alias. And PushT captures its whole control step into a CUDA graph
    that reads ``_state`` at a fixed address, so a ``set_state`` that rebound the attribute instead of
    writing through it would leave the graph replaying against dead memory: every ``set_state`` would
    be silently ignored, and the env would look like it had frozen.
    """
    st = env.sample_states(env.B, rng=np.random.default_rng(0))
    buf = np.array(st, np.float32)
    before = env._get_render_state().data_ptr()
    env._set_render_state(buf)
    stored = env._get_render_state().detach().cpu().numpy().copy()
    buf[:] = 0.0
    after = env._get_render_state().detach().cpu().numpy()
    return {
        "aliased_caller_buffer": float(np.abs(after - stored).max()),
        "state_tensor_rebound": bool(env._get_render_state().data_ptr() != before),
    }

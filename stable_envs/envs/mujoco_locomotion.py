"""Gymnasium's MuJoCo family — ten domains, one state vector each (``qpos``), in TWO fidelity tiers.

Reference: ``gymnasium.make("Reacher-v5", render_mode="rgb_array", width=S, height=S).render()``,
i.e. ``MujocoRenderer`` driving ``mjr_render`` at the default ``MjvOption``. The target is MuJoCo's
own EGL raster of a lit 3-D scene, so this family shares the plate + analytic-mover design that
``ogbench.py`` and ``dmcontrol.py`` use, and the same measured shading composition.

**Four are PARITY and six are VISUAL, and the two are kept apart mechanically rather than by
convention** (``docs/concepts/fidelity-tiers.md``). ``Reacher``, ``Pusher``, ``InvertedPendulum`` and
``InvertedDoublePendulum`` claim the official frame: every interior pixel bit-identical. The six
locomotion domains — ``HalfCheetah``, ``Hopper``, ``Walker2d``, ``Ant``, ``Humanoid``, ``Swimmer`` —
claim a DIFFERENT RENDER of the same scene, drawn by nvdiffrast, and declare what that costs in
``visual_fidelity``. They share the state, the reference procedure and this module and nothing else;
they are built on ``_GymMujocoFamily`` beside ``GymMujoco`` rather than on top of it.

**THE RENDER STATE IS ``qpos``, WHICH IS NOT THE OBSERVATION.** The locomotion envs default to
``exclude_current_positions_from_observation=True``, so their observation drops root coordinates the
renderer needs; the four here instead TRANSFORM ``qpos`` — Reacher's observation carries ``cos`` and
``sin`` of its two joint angles rather than the angles, and Pusher's carries the object and the goal
as body positions rather than as the slide coordinates ``qpos`` holds. Either way the observation is
neither a superset nor a subset of the render state, and ``state_names`` below names the ``qpos``
entries, read off the compiled model's joint layout rather than transcribed.

That ``qpos`` is the WHOLE render state is measured, not assumed (the frozen measurement record, all ten Gymnasium MuJoCo envs at 128 px, 8 states each): re-rendering the same
``qpos`` changes 0 pixels, drawing ``qvel`` from ``U[-100, 100]`` instead of zero changes 0 pixels,
and reaching the same ``qpos`` from the reverse rollout changes 0 pixels. Every ``qpos`` entry moves
pixels. The compact state is 2-24 floats across the ten, 2-11 for the four that ship here, against
240*240*3 = 172,800 bytes of frame.

WHICH FOUR ARE PARITY, AND WHY THE OTHER SIX CANNOT BE
-------------------------------------------------------

Parity: ``Reacher-v5``, ``Pusher-v5``, ``InvertedPendulum-v5``, ``InvertedDoublePendulum-v5``.
They have no textures at all (``ntex = 0``, ``nmat = 0``), no reflective surface, and a camera that
does not move, so the static half of the picture is one constant image and only 2-19 primitives are
cast per frame.

NOT parity, at either backend this repo has: ``HalfCheetah``, ``Hopper``, ``Walker2d``, ``Ant``,
``Humanoid``, ``Swimmer``. Their render state is also exactly ``qpos`` — the same measurement passes
on all six — and their models are also mesh-free primitives, so the obstacle is neither the state nor
the geometry. It is the picture, and after a second look on a second backend it is ONE thing about
the picture:

  * **The floor is a minification-filtered checker, and that is what decides it.** It covers 37-95%
    of the frame, and on five of the six the far half of it has been averaged into ONE colour by GL's
    filter: split the floor by image row and the farthest tenth has a luminance spread of 0.0-6.8
    about a single value of 93.7, against 74.9-90.1 spanning the full black-to-white range in the
    nearest tenth (``probe_minification.py``, 128 px). 93.7 is what the checker's two colours average
    to under that lighting. Any renderer that resolves one texel per ray keeps the contrast and
    paints speckle there — including ``mujoco_warp``, which builds every texture at one mip level and
    samples with no level of detail. Re-measured with every correction this repo knows about, MJWarp
    agrees with MuJoCo on **0.0017-0.0069% of far-field floor pixels** and 25-34% of the whole frame,
    against a 90% gate; take the textures off both sides and the same pipeline reaches 0.86-1.00
    interior exact, which is the flat-material parity ``backend3d.md`` reports.
  * **The camera rides the body**, which rules out THIS design and not the domain. All six render
    through the model's ``track`` camera (``mode="trackcom"``); hiding every moving geom and
    rendering eight states leaves the "static" frame differing by 202-255/255 over 29-57% of its
    pixels, so there is no plate to bake. A backend that draws the whole scene per frame does not
    care: MJWarp's ``trackcom`` camera lands on MuJoCo's to 0.0, once ``com_pos`` is in the frame
    loop (``mujoco_warp.kinematics`` does not write ``subtree_com``, and without it the camera is
    frozen at its ``put_data`` pose with no error).
  * **The floor is a mirror**, and it is a footnote. ``MatPlane`` carries ``reflectance="0.5"`` and
    MuJoCo's reflection pass does fire — removing it changes 1.0-3.3% of the pixels by up to
    81-128/255 — but it reaches **0.0000 of INTERIOR pixels on all six**, because a mirror image over
    a textured floor leaves no flat 3x3 neighbourhood anywhere.

Swimmer fails differently and fails anyway: its camera looks down from close range so its floor is
magnified rather than minified (far/near spread 0.911), and GL's magnification filter over a floor
that is 95% of the frame leaves it at 0.285 interior exact, the lowest of the six.

See ``docs/research/mujoco_locomotion.md`` Section 6 for the ladder that attributes this, for the two
escape hatches that were measured and closed (a pre-filtered floor texture; supersampling), and for
what would have to change.

WHAT THE SIX SHIP AS INSTEAD: THE VISUAL TIER
----------------------------------------------

The blocker above is a MISSING OPERATION rather than a wrong picture, and a third backend has it.
``stable_envs/render/nvdiff.py`` rasterizes through nvdiffrast, whose ``texture(...,
filter_mode="linear-mipmap-linear")`` samples a prefiltered pyramid off image-space uv derivatives —
exactly the level-of-detail a one-ray-per-pixel raytracer does not have. On the far tenth of this
floor that takes bit-exact agreement from **0.0000-0.0001 to 0.66-0.81** with the SAME pipeline either
way, the switch being ``filter_mode`` alone, and the luminance spread from 95 to 3-9 against a
reference of 0-3 (``docs/research/nvdiffrast.md`` Section 4.1). The no-mipmap control lands exactly in
``mujoco_warp``'s regime, which is what says the two were failing for one reason.

It is still not the official frame — see ``visual_fidelity`` on each class for the three numbers and
``VISUAL_NOTE`` for what is left — so the six are registered in the VISUAL tier, with
``parity_interior_min = None`` so the parity gate cannot be applied to them, cannot be relaxed for
them, and cannot be passed by them. They do not displace the four above and share no rendering code
with them.

Two things this module does that the nvdiffrast study did not, both measured here:

  * **The shadow is drawn.** All six light the scene with one directional light straight down plus
    MuJoCo's headlight, and the model light's shadow was the largest term left. ``_ShadowPass`` is one
    orthographic depth pass over the movers; it is worth mean |diff| 4.85 -> 3.79/255 and SSIM 0.959
    -> 0.969 on HalfCheetah, for 22% of the frame rate. What it is worth per env is in
    ``VISUAL_NOTE``.
  * **Supersampling is refuted here rather than inherited.** The parity four render at supersample 2
    because the reference is MSAA'd; on these six it makes every number worse (HalfCheetah mean 4.88
    -> 6.28, SSIM 0.960 -> 0.944), because supersampling also averages the floor TEXTURE and picks the
    mip level on the finer grid. ``default_supersample = 1``.

THE FREE CAMERA IS FROZEN AT THE FIRST RENDER, AND THAT IS WHY IT IS STYLE
--------------------------------------------------------------------------

None of the four has a camera in its XML, so ``MujocoRenderer`` falls back to ``camera_id = -1``,
builds a free camera, and ``OffScreenViewer._init_camera`` sets its ``lookat`` to
``median(data.geom_xpos)`` **at the moment the viewer is created** — which is the first ``render()``
call, at whatever state the env happens to be in. ``DEFAULT_CAMERA_CONFIG`` then overrides only what
it names (Reacher names ``trackbodyid``, which a free camera ignores). So gymnasium's own frame is a
function of ``qpos`` AND of the state its first render happened to see, and two processes that reset
differently get different pictures from the same ``qpos``.

That is a property of the reference, not something a renderer can fix, so it is made explicit
instead: the camera is part of the STYLE, its default is the pose gymnasium freezes after
``reset(seed=0)`` plus one ``render()``, and ``reference_env`` performs exactly that procedure so the
two sides are comparable by construction rather than by luck. ``_check_camera_matches`` asserts that
rather than trusting it.

WHAT IS DRAWN PER FRAME
-----------------------

The plate is baked through MuJoCo itself with the movers moved to a hidden geom/site group, so every
static pixel is the reference's own rasterization — 95.6-99.7% of the frame, exact by construction. The
movers (4 geoms for Reacher, 2 for InvertedPendulum, 3 geoms + 1 site for InvertedDoublePendulum, 19
for Pusher) are ray-cast analytically and composited against the plate's own ray parameter.

Pusher needs two more things and gets them the way ``dmcontrol.py`` does. Its model light is a
SPOTLIGHT (``cutoff=45``, ``exponent=10``, attenuation ``(1,0,0)``) with ``castshadow=1``, so the arm
darkens the table under it: a second plate is baked with that light's diffuse and specular zeroed,
which IS the colour a static surface has where the light does not reach, and a shadowed pixel selects
that plate. Nothing about MuJoCo's shadow MAP is modelled — only the shape of the shadow, by one
occlusion ray. The other three have no model light at all (``nlight = 0``); their only light is the
headlight, which is directional along the camera axis and casts no shadow.

MEASURED CONSTANTS
------------------

Read off ``MjvScene`` rather than the XML, because the XML does not contain them: with no material,
every geom here resolves to ``specular=0.5``, ``shininess=0.5``, ``emission=0``, ``reflectance=0``,
and the headlight to ambient 0.1 / diffuse 0.4 / specular 0.5. The specular exponent is
``128 * shininess``, i.e. 64. ``SPECULAR_SCALE`` is the one fitted number and it is carried from the
DM Control study, where an untextured BOX face at normal incidence pinned it to [0.04902, 0.04980)
against a nominal 0.05. It is re-measured here rather than inherited: swept against a nominal 1.0 it
improves or ties mover-pixel within-1 agreement on all four domains, by 0.739 -> 0.872 on
InvertedDoublePendulum and by a fraction of a point elsewhere.
"""

from __future__ import annotations

import numpy as np
import torch

from .._mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

from ..env import Env
from ..registry import _remember_historical_name, register
from ..render import _resources as _RES
from ..render import raycast3d as R3
from ..spaces import Box, Dict

#: ``mjvLight.type``: 0 is a spot/point light, 1 is directional. The headlight is always directional
#: and always at the camera; a model light with ``directional="false"`` (the default) is a spotlight
#: with a cutoff, an exponent and distance attenuation.
LIGHT_DIRECTIONAL = 1

#: Geom/site groups ``MjvOption`` shows by default — measured [1,1,1,0,0,0] for both. Group 5 is
#: therefore a place to park geometry that must not be drawn while the plate is baked.
VISIBLE_GROUPS = (0, 1, 2)
HIDDEN_GROUP = 5

#: MuJoCo's material defaults, used for every geom in this family because none of them carries a
#: material at all (``nmat = 0`` for three of the four; Pusher's is 0 too).
DEFAULT_SPECULAR, DEFAULT_SHININESS = 0.5, 0.5

#: Effective weight on the additive white specular term. Carried from ``dmcontrol.py``, where it was
#: measured by inverting the reference's own quantised colours on an untextured box face at normal
#: incidence — the one surface where MuJoCo's per-vertex shading and a per-pixel evaluation coincide
#: by construction. The lighting there is identical to the lighting here (headlight ambient .1 /
#: diffuse .4 / specular .5, material specular .5), which is why the number transfers; it is checked
#: rather than trusted by the frozen measurement record, which scores 1.0 and 0.988 on
#: mover-interior pixels for each of the four envs.
SPECULAR_SCALE = 0.988

#: class name -> (gymnasium id, human name). Only the four fixed-camera, texture-free domains; see
#: the module docstring for the six that are a documented NO-GO and why.
#: class name -> (parity_interior_min, extra parity_note), for the two envs that need a floor below
#: exactness. In both cases the count is what makes it a declared limit rather than a tolerance:
#: single pixels in millions, every one off by exactly 1/255 in one channel.
#:
#: ``MujocoPusher``'s entry is here because its original 1.0 was A LUCKY DRAW, and that is worth
#: stating plainly. It was measured as "0 disagreeing pixels out of 12,112,519 over 256 states" on
#: ONE unseeded sample of states; with the parity test seeded and run over 101 independent draws it
#: fails 15 of them, at 1-2 pixels of 3.03 M. Those two numbers are consistent — at that rate about
#: 47% of draws come back clean — so the first measurement was not wrong, it was a single sample of
#: a quantity that is sometimes zero, reported as if it were the quantity. A floor declared from one
#: draw of a rare event is a floor that will go red on someone else's machine, and it did.
#:
#: The mechanism is established and is NOT a bug in either renderer: float32 ``exp`` differs by one
#: ulp between torch's CPU implementation (1-ulp faithful) and ours (correctly rounded), and that
#: flips a pixel only where a smooth-shaded colour lands exactly on the boundary the reference
#: rounds the other way. It is the same mechanism ``parity_note`` already names for
#: InvertedPendulum. ``MujocoReacher`` (0 of 41 draws) and ``MujocoInvertedDoublePendulum`` (0 of 41)
#: are clean and declare nothing.
PARITY: dict[str, tuple[float, str]] = {
    "MujocoInvertedPendulum": (
        0.99999,
        " INTERIOR: 164 disagreeing interior pixels out of 673,455,908 over 201 seeded draws of 64 "
        "states at the shipped supersample 1, all inside the cart or pole capsule and all off by "
        "exactly 1/255; the count is IDENTICAL at supersample 2 and unchanged at supersample 4, so it "
        "is the per-vertex/per-pixel shading difference above and not a coverage rule. The worst "
        "single draw is 0.9999988, and the floor is set to 0.99999, which is 8.4x inside it.",
    ),
    "MujocoPusher": (
        0.99999,
        " INTERIOR: 8 disagreeing interior pixels out of 608,783,153 over 201 seeded draws of 64 "
        "states at the shipped supersample 1 — worst single draw 0.9999997, and most draws are "
        "exact — every one off by exactly 1/255 in a single channel. The cause is a one-ulp "
        "difference in float32 ``exp`` between torch's CPU implementation and this one, which reaches "
        "a pixel only where the specular term lands exactly on a rounding boundary — the same "
        "mechanism as InvertedPendulum above. Supersampling makes it WORSE, which is the clearest "
        "evidence available that it is a rounding boundary and not coverage: the same 201 draws at "
        "supersample 2 give 30 bad pixels against 8, because a blended pixel has more ways to land on "
        "a boundary than a covered one. The floor is set to 0.99999, which is 33x inside "
        "the measured worst draw. This env once declared 1.0 on the strength of a single unseeded "
        "draw that happened to come back clean; the floor is set from a draw COUNT, and that rule is "
        "why the supersample change was verified over 201 draws rather than over one large "
        "evaluation.",
    ),
}

ENVS: dict[str, str] = {
    "MujocoReacher": "Reacher-v5",
    "MujocoPusher": "Pusher-v5",
    "MujocoInvertedPendulum": "InvertedPendulum-v5",
    "MujocoInvertedDoublePendulum": "InvertedDoublePendulum-v5",
}

#: (env id, size) -> a live gymnasium env, its model/data, and the pristine appearance snapshot.
#: One per (env, size), built once and NEVER closed: the env owns an EGL context through its
#: ``MujocoRenderer``, and releasing one corrupts rendering in every other live context in the
#: process — the rule ``stable_envs/testing/conformance.py`` carries.
_ENV_CACHE: dict = {}
_SCENE_CACHE_MAX = 96
_SCENE_CACHE = _RES.BoundedCache(_SCENE_CACHE_MAX, name="mujoco_locomotion.scene")
#: Headless envs used only to roll out ``sample_states``. No ``render_mode``, so no viewer is ever
#: built and no GL context is created; that is why these can be cached per class without counting
#: against the context budget.
_SAMPLER_CACHE: dict = {}


def _gym():
    import gymnasium

    return gymnasium


# --------------------------------------------------------------------------- model reading


def _mat_rgb(m, matid: int, rgba) -> np.ndarray:
    """Base colour: the element's own rgba unless that is the compiler default and a material
    supplies one, which is the rule MuJoCo itself follows."""
    rgba = np.asarray(rgba, np.float64)
    if matid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[matid], np.float64)[:3]
    return rgba[:3]


def _spec_shin(m, matid: int) -> tuple[float, float]:
    if matid < 0:
        return DEFAULT_SPECULAR, DEFAULT_SHININESS
    return float(m.mat_specular[matid]), float(m.mat_shininess[matid])


def _norm(v):
    return v / torch.clamp(torch.sqrt((v * v).sum(-1, keepdim=True)), min=1e-12)


def _dot(a, b):
    """Unrolled for the reason ``raycast3d._dot`` is: a reduction whose CUDA kernel is chosen from
    the batch count sums its terms in a batch-dependent order, and one ulp there is a hit/miss flip
    on a silhouette ray."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


class _Lights:
    """The lights exactly as ``MjvScene`` reports them — the only place the headlight exists.

    Nothing is summed in advance. Three of these four domains have one directional light and could be
    folded into a single direction; Pusher has a spotlight beside the headlight, and folding would
    erase the cone falloff that is most of what the table looks like.
    """

    __slots__ = (
        "kind",
        "pos",
        "dir",
        "ambient",
        "diffuse",
        "specular",
        "attenuation",
        "cutoff",
        "exponent",
        "castshadow",
    )

    def __init__(self, scene):
        n = int(scene.nlight)
        g = lambda f: (
            np.stack(
                [
                    np.asarray(getattr(scene.lights[i], f), np.float64)  # noqa: E731
                    for i in range(n)
                ]
            )
            if n
            else np.zeros((0, 3))
        )
        self.pos, self.dir = g("pos"), g("dir")
        self.ambient, self.diffuse, self.specular = g("ambient"), g("diffuse"), g("specular")
        self.attenuation = g("attenuation")
        self.cutoff = np.array([float(scene.lights[i].cutoff) for i in range(n)])
        self.exponent = np.array([float(scene.lights[i].exponent) for i in range(n)])
        self.castshadow = np.array([int(scene.lights[i].castshadow) for i in range(n)])
        # mujoco >= 3.2 renamed the flag to a ``type`` enum; both spellings are read so this does not
        # become a version tripwire.
        self.kind = np.array(
            [int(getattr(scene.lights[i], "type", getattr(scene.lights[i], "directional", 1))) for i in range(n)]
        )

    def shadow_light(self) -> int:
        idx = np.flatnonzero(self.castshadow > 0)
        return int(idx[0]) if len(idx) else -1

    def to(self, device) -> dict:
        t = lambda x: torch.tensor(
            np.asarray(x, np.float64),
            device=device,  # noqa: E731
            dtype=torch.float32,
        )
        return dict(
            pos=t(self.pos),
            dir=t(self.dir),
            ambient=t(self.ambient),
            diffuse=t(self.diffuse),
            specular=t(self.specular),
            attenuation=t(self.attenuation),
            cutoff=[float(c) for c in self.cutoff],
            exponent=[float(e) for e in self.exponent],
            kind=[int(k) for k in self.kind],
            castshadow=[bool(c) for c in self.castshadow],
        )


class _Scene:
    """Everything a render needs that does not depend on the state."""

    __slots__ = (
        "plate",
        "plate_shadow",
        "static_t",
        "cam",
        "model",
        "data",
        "movers",
        "size",
        "lights",
        "prims",
        "shadow_light",
        "eps",
        "pristine",
        "rays",
        "shadow_candidates",
        "const",
    )


def _light_geometry(L: dict, i: int, p: torch.Tensor):
    """``(direction toward the light, attenuation)`` for light ``i`` at points ``p``.

    OpenGL fixed function, which is what ``mjr_render`` drives: a positional light attenuates by
    ``1/(k0 + k1 d + k2 d^2)`` and, inside its cutoff, by ``(spotdir . -l)^exponent``; a directional
    light does neither and has no position at all.
    """
    shape = (1,) * (p.dim() - 1) + (3,)
    if L["kind"][i] == LIGHT_DIRECTIONAL:
        l = (-_norm(L["dir"][i].view(shape))).expand_as(p)
        return l, torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)
    v = L["pos"][i].view(shape) - p
    dist = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
    l = v / dist.unsqueeze(-1)
    k = L["attenuation"][i]
    att = 1.0 / torch.clamp(k[0] + k[1] * dist + k[2] * dist * dist, min=1e-12)
    cut = L["cutoff"][i]
    if cut < 180.0:
        se = _dot(_norm(L["dir"][i].view(shape)).expand_as(l), -l)
        inside = se >= float(np.cos(np.deg2rad(cut)))
        att = torch.where(inside, att * torch.clamp(se, min=0) ** L["exponent"][i], torch.zeros_like(att))
    return l, att


def _shade(p, n, v, L, mat, spec, shininess, unlit=None):
    """``(...,3)`` colour in [0,1] for surface points ``p`` with unit normal ``n``, viewer dir ``v``.

    The composition is the one measured in ``docs/research/ogbench.md``: the specular term is WHITE
    and ADDITIVE, and it is added after the per-light diffuse sum rather than multiplied in.
    ``unlit`` masks the diffuse and specular of the shadow-casting light only, because a shadow does
    not remove ambient.
    """
    out = torch.zeros_like(p)
    highlight = torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
    exponent = 128.0 * shininess
    for i in range(len(L["kind"])):
        l, att = _light_geometry(L, i, p)
        lit = att
        if unlit is not None and L["castshadow"][i]:
            lit = att * (~unlit).to(att.dtype)
        ndl = torch.clamp(_dot(n, l), min=0)
        h = _norm(l + v)
        ndh = torch.clamp(_dot(n, h), min=0)
        out = out + mat * (att * L["ambient"][i][0]).unsqueeze(-1)
        out = out + mat * (lit * ndl * L["diffuse"][i][0]).unsqueeze(-1)
        highlight = highlight + lit * L["specular"][i][0] * ndh**exponent
    return torch.clamp(out + (highlight * spec * SPECULAR_SCALE).unsqueeze(-1), 0, 1)


def _upsample_rgb(img: torch.Tensor, ss: int) -> torch.Tensor:
    """Nearest-neighbour blow-up of a baked plate, chosen so the box average that follows is the
    IDENTITY on plate pixels: every sample inside an output pixel carries the same integer, so the
    mean is that integer and ``round`` returns it. The plate stays exact, not approximately so."""
    return img if ss == 1 else img.repeat_interleave(ss, 0).repeat_interleave(ss, 1)


class _GymMujocoFamily(Env):
    """What the TWO tiers of this family share: gymnasium, its state, and its reference procedure.

    Everything here is about the OFFICIAL env rather than about any renderer — how a gymnasium MuJoCo
    env is constructed so that its picture is reproducible, how one official frame is taken, what the
    state vector is called, where the states come from, and how the reference is driven. Both
    ``GymMujoco`` (the parity tier, plate + analytic ray-cast) and ``GymMujocoVisual`` (the visual
    tier, nvdiffrast) inherit it, so the two are compared against the reference by the same code
    rather than by two procedures that are meant to agree.

    It is a base class rather than a set of module functions because every one of these needs
    ``ENV_ID``, and a helper that takes the env id as an argument is a helper that can be called with
    the wrong one.
    """

    family = "mujoco"

    ENV_ID: str = ""

    #: ---- Public state --------------------------------------------------------
    #:
    #: The packed float32 row is unchanged: it IS ``qpos``, exactly the packed float32 layout every frozen
    #: digest and certification check speaks. The public State adds a name for it. One key and not
    #: several, because this family's state has no second half to separate from the first -- unlike
    #: DM Control there is nothing drawn per episode, so ``qpos`` covers every column and the layout
    #: is one entry with offset 0. It is assigned PER INSTANCE in ``_declare_state`` for the reason
    #: DM Control assigns its own there: ``nq`` is a property of the compiled model and is not known
    #: until one exists.
    #:
    #: NOTHING IS HIDDEN. ``qpos`` carries every column, so no field has offset ``None``,
    #: ``_hidden_state`` is never consulted, and ``pack``/``unpack`` round-trip the corpus bit for
    #: bit. There is no ``_live_state`` either: the packed row IS this family's live state --
    #: ``set_state`` writes it and ``render`` reads it -- so there is no more precise buffer to
    #: answer from. Velocities are not state here at all: gymnasium integrates them, this package
    #: renders from a pose, and a ``qvel`` column would be a dimension the frame never reads.
    #:
    #: NO DYNAMICS, and that is a declaration rather than an omission: both tiers ship a renderer
    #: over gymnasium's own MuJoCo physics and there is no batched simulator in this package for any
    #: of these domains, so ``step``/``transition`` raise ``Env``'s precise refusal and
    #: ``has_dynamics`` is False. ``reset`` and ``render`` work as usual.
    #:
    #: NO ``_close``. Everything this family holds -- the gymnasium bundle and its EGL context
    #: (``_ENV_CACHE``), the baked plates (``_SCENE_CACHE``), the nvdiffrast contexts
    #: (``_NVD_CACHE``), the headless sampler (``_SAMPLER_CACHE``) -- is keyed on (env id, size) and
    #: SHARED by every instance, and releasing an EGL context corrupts every other live one in the
    #: process (the rule ``testing/conformance.py`` carries). An instance owns no resource of its
    #: own, so ``close()`` buys exactly the contract's other half here: a closed env REFUSES to
    #: render rather than handing back frames drawn through something a caller believes is gone.

    #: Half of gymnasium's own 480x480 default, chosen so ``check_resolution_consistency``'s 2x
    #: render lands exactly on the official default rather than on an invented resolution. Parity is
    #: measured here; throughput at 84, 240 and 480 (``docs/research/mujoco_locomotion.md``).
    default_size = 240
    #: Random-action steps taken from a reset before a state is read. The reset distributions here
    #: are tiny — InvertedPendulum's is ``init_qpos + U[-0.01, 0.01]`` — so a parity number measured
    #: on resets alone would describe one pose. This is the env's OWN visited distribution under a
    #: random policy, which is the population every conformance number should be measured over.
    ROLLOUT_STEPS = 60

    def _declare_state(self, m) -> None:
        """``state_dim``/``state_names``/``state_layout`` off the compiled model, for BOTH tiers.

        Written once here rather than twice in the two constructors, which is where it was and where
        the two could have drifted: the parity tier and the visual tier draw the same domain and must
        agree about what its state IS, whatever draws it.
        """
        self.state_dim = int(m.nq)
        self.state_names = self._qpos_names(m)
        self.state_layout = ((("qpos",), self.state_dim, 0),)

    @classmethod
    def _fresh(cls, size: int):
        """A new official env at ``size``, driven through gymnasium's OWN first-render procedure.

        The three steps are load-bearing and in this order. ``offwidth``/``offheight`` must be raised
        before the viewer exists, because the offscreen framebuffer is sized from them when the
        ``MjrContext`` is built and MuJoCo's default 640x480 would refuse anything larger. Then
        ``reset(seed=0)``, then one ``render()``: that is the call that constructs the viewer, and
        the viewer's ``_init_camera`` reads ``median(data.geom_xpos)`` at that instant. Doing it any
        other way gives a different camera and therefore a different picture from the same qpos.
        """
        env = _gym().make(cls.ENV_ID, render_mode="rgb_array", width=int(size), height=int(size))
        u = env.unwrapped
        u.model.vis.global_.offwidth = max(int(u.model.vis.global_.offwidth), int(size))
        u.model.vis.global_.offheight = max(int(u.model.vis.global_.offheight), int(size))
        env.reset(seed=0)
        u.render()
        return env, u

    @staticmethod
    def _shot(u) -> np.ndarray:
        """One official frame, with this env's OWN GL context bound first.

        ``gymnasium``'s ``OffScreenViewer.render`` calls ``mjv_updateScene`` and ``mjr_render``
        without ever binding its context, and ``MujocoRenderer._get_viewer`` re-binds only when ONE
        renderer owns several viewers (the rgb_array + human case). So whatever context was last made
        current is the one this env's ``MjrContext`` is resolved against — and this class interleaves
        two envs by design, since the bundle env bakes plates while the conformance suite's reference
        env renders the frames being compared against them.

        MEASURED, and the shape of the result is why the line is here rather than in a comment
        (the frozen measurement record, four envs at 240 px, same env, same state,
        rendered before and after an interloper):

          * a plain ``mujoco.Renderer``: 0/255.
          * a segmentation ``mujoco.Renderer`` driven with the viewer's own ``MjvCamera``: 0/255.
          * a second gymnasium env of the same id and size: 0/255.
          * a gymnasium env whose model sets ``offsamples=1``: **0.4-8.1% of the pixels change, by up
            to 44-84/255**.
          * an explicit ``make_context_current()`` after any of them: back to 0/255.

        So the hazard is not "another context exists", it is "another context with a different
        multisample configuration became current", and every cheap test of the obvious kind passes
        while it is present. One ``eglMakeCurrent`` per frame closes it.
        """
        u.mujoco_renderer.viewer.make_context_current()
        return u.render()

    @staticmethod
    def _qpos_names(m) -> tuple[str, ...]:
        """One label per ``qpos`` entry, read off the model's joint layout rather than transcribed.

        A free joint contributes 7 entries and a ball joint 4, so a hand-written list is right until
        someone adds a joint. A state vector whose fields are mislabelled is worse than one with no
        labels, because everything downstream indexes the wrong number in silence.
        """
        import mujoco

        free = ("_x", "_y", "_z", "_qw", "_qx", "_qy", "_qz")
        ball = ("_qw", "_qx", "_qy", "_qz")
        out: list[str] = []
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            t = int(m.jnt_type[j])
            if t == int(mujoco.mjtJoint.mjJNT_FREE):
                out += [nm + s for s in free]
            elif t == int(mujoco.mjtJoint.mjJNT_BALL):
                out += [nm + s for s in ball]
            else:
                out.append(nm)
        return tuple(out)

    @classmethod
    def _sampler(cls):
        """A headless copy used only for rollouts. No ``render_mode``, so gymnasium never builds a
        viewer and no GL context is created — which is why caching it costs nothing against the
        context budget the conformance suite polices."""
        hit = _SAMPLER_CACHE.get(cls.ENV_ID)
        if hit is None:
            hit = _SAMPLER_CACHE[cls.ENV_ID] = _gym().make(cls.ENV_ID)
            hit.reset(seed=0)
        return hit

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """``(batch, nq)`` from the env's own reset distribution plus a random-policy rollout."""
        rng = rng or self.rng
        env = self._sampler()
        u = env.unwrapped
        lo = np.asarray(env.action_space.low, np.float64)
        hi = np.asarray(env.action_space.high, np.float64)
        out = np.empty((int(batch), self.state_dim), np.float64)
        for i in range(int(batch)):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            for _ in range(int(rng.integers(0, self.ROLLOUT_STEPS + 1))):
                _, _, term, trunc, _ = env.step(rng.uniform(lo, hi).astype(np.float32))
                if term or trunc:
                    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            out[i] = np.asarray(u.data.qpos, np.float64)
        return out.astype(np.float32)

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for the given ``qpos``, with the round-trip ASSERTED.

        ``qvel`` is set to zero and is provably irrelevant: drawing it from ``U[-100, 100]`` changes
        zero pixels on every one of the ten Gymnasium MuJoCo envs. The assertion is not decoration —
        without it, a state the reference did not reach produces a plausible parity number instead of
        a failure, and the mistake presents as a rendering discrepancy.
        """
        u = env.unwrapped
        nv = int(u.model.nv)
        out = []
        for st in np.asarray(states, np.float64):
            u.set_state(st, np.zeros(nv))
            got = np.asarray(u.data.qpos, np.float64)
            if np.abs(got - st).max() > 1e-9:
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested qpos (max "
                    f"{np.abs(got - st).max():.3e}); every parity number computed from it would be "
                    f"comparing two different scenes"
                )
            out.append(cls._shot(u).copy())
        return np.stack(out)


class GymMujoco(_GymMujocoFamily):
    """Base for the four fixed-camera Gymnasium MuJoCo domains. ``ENV_ID`` is set per subclass."""

    #: ONE ray per pixel. Same change and same evidence as ``DMControlSuite.default_supersample``,
    #: which carries the argument in full; the numbers below are this family's own.
    #:
    #: MSAA moves SILHOUETTE pixels and the strong parity assertion is about INTERIOR ones, so
    #: dropping it does not touch the claim. Over 201 seeded draws of 64 states at 240 px: interior
    #: disagreements are 2 of 124,179,562 at ss=1 against 3 of 124,179,562 at ss=2 on ``Pusher``, and
    #: 39 of 137,372,300 at BOTH supersamples on ``InvertedPendulum`` -- the ss=1 count is never the
    #: worse of the two on any of the four. ``Reacher`` and ``InvertedDoublePendulum`` stay at 0. The
    #: two bounded quantities keep their margins: worst all-pixel exactness 0.9671 against a floor of
    #: 0.90, worst mean 0.133 against a ceiling of 4.0.
    #:
    #: WHAT THIS GIVES UP: ``max_abs`` roughly doubles (108 -> 181 on Pusher, 44 -> 93 on Reacher) and
    #: mean |diff| rises about 2x, both unasserted. In exchange the family renders 2.5-2.8x faster,
    #: which is the largest supersample win in the registry because these cells cast over a 240 px
    #: frame where the movers cover a couple of percent of it.
    #:
    #: ``supersample=2`` remains available per call and per instance. See the frozen measurement record.
    default_supersample = 1
    #: Rays cast in one pass, so a large batch bounds its own memory rather than the caller's. Every
    #: quantity is (B, H*W*ss^2, ...) float32 and the arithmetic is per-ray, so chunking cannot change
    #: the answer — asserted by the reproducibility suite rather than assumed. Raised with
    #: ``DMControlSuite.RAY_BUDGET``, off the same sweep and for the same measured reason: once the
    #: casts are fused, what a chunk costs is a couple of hundred torch calls and two synchronisations,
    #: and these families run at 100% of one core. Pusher 4,156 -> 4,839 f/s and Reacher 7,599 ->
    #: 9,458 at 256 px batch 1024, for 1.8 GB more of peak device memory.
    RAY_BUDGET = 1 << 24
    #: NOT relaxed for two of the four, and that is a measurement rather than a hope: over 201
    #: independent draws of 64 states each (12,864 states) at the shipped ``default_supersample = 1``,
    #: interior disagreements are 0 out of 622,250,870 (Reacher) and 0 out of 725,869,529
    #: (InvertedDoublePendulum). Pusher and InvertedPendulum override it; see the ``PARITY`` table
    #: above for their counts and the mechanism, which is one ulp of float32 ``exp`` and not coverage.
    parity_interior_min = 1.0
    #: The residual this family DOES carry, stated where it cannot be lost. It is not a parity
    #: relaxation for three of the four — it does not reach the interior metric at all, because the
    #: interior mask keeps only pixels whose 3x3 neighbourhood in the reference is one flat colour
    #: and a shaded capsule almost never has one. It is the number the analytic shader is actually
    #: responsible for, so it is reported separately instead of being folded into an aggregate that
    #: the exact plate would dominate.
    parity_note = (
        "MOVER-PIXEL PARITY (supersample 1, against gymnasium at MuJoCo's default offsamples=4, 240 "
        "px, 32 states per env). The plate is baked through MuJoCo itself, so every static pixel is "
        "the reference's own rasterization and non-mover pixels are 99.76-99.93% bit-identical; the "
        "whole residual is inside the moving primitives, which are 0.3-4.4% of the frame. There the "
        "shader reaches 3-32% of pixels exact and 37-67% within 1/255 (mean 2.5-7.7/255), and on "
        "mover pixels eroded by one — so coverage is not in play — 6-34% exact and 53-72% within "
        "1/255 (mean 2.2-3.9/255). THOSE MOVER NUMBERS ARE THE PRICE OF ss=1 AND ARE STATED AS SUCH: "
        "at supersample 2 the identical protocol gives 16-35% / 64-78% (mean 2.0-3.1) and 20-37% / "
        "67-88% (mean 1.3-2.0). None of it reaches an assertion — the interior metric excludes a "
        "shaded capsule almost everywhere, and whole-frame exactness and mean keep their margins "
        "(worst 0.9670 against a 0.90 floor, worst mean 0.134 against a 4.0 ceiling, over 201 draws) "
        "— but a caller who is scored on mover pixels should pass supersample=2, which is still "
        "there. Against a NON-multisampled reference the same shader reaches "
        "71-90% within 1/255 (mean 1.27-1.86), which puts a large part of the gap in MSAA sample "
        "positions rather than in shading. The rest is MuJoCo shading curved primitives PER VERTEX "
        "over a tessellation while this is a per-pixel evaluation, and that diagnosis is measured "
        "rather than asserted: a least-squares fit of (ambient, diffuse, specular, exponent) on "
        "18,231 mover-interior pixels returns ambient 0.103 and diffuse 0.388 against the nominal "
        "0.1 and 0.4 — the diffuse law is right — and moves only the highlight, to a broader and "
        "weaker lobe (0.177 at exponent 32 against 0.25 at 64), which is what interpolating a "
        "per-pixel highlight across a triangle looks like. Adopting that lobe was tried at four "
        "settings and REFUTED as a family-wide constant: it improves InvertedDoublePendulum "
        "(within-1 0.872 -> 0.885) and InvertedPendulum (0.728 -> 0.736) and makes Reacher (0.685 -> "
        "0.668) and Pusher (0.729 -> 0.677) worse, so the documented law is kept."
    )

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        self._declare_state(self._model())
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        sc = self._scene()
        assert sc.model.nq == self.state_dim, f"{type(self).__name__}: nq changed under us"
        assert len(self.state_names) == self.state_dim
        self._check_scene_is_supported(sc)

    def _check_scene_is_supported(self, sc: _Scene) -> None:
        """Refuse to draw a scene this renderer cannot draw, instead of drawing it wrong.

        Every one of these holds for all four domains today and is cheap to check; each of them is
        also exactly what a future gymnasium release could change without anyone noticing, and the
        failure mode of each is a plausible-looking picture rather than an error.
        """
        m = sc.model
        assert int(m.ntex) == 0, (
            f"{type(self).__name__}: the model now carries {int(m.ntex)} textures, and this renderer "
            f"samples none — a textured surface would be painted its material colour"
        )
        for i in range(int(m.nmat)):
            assert float(m.mat_reflectance[i]) == 0.0, (
                f"{type(self).__name__}: material {i} is reflective (reflectance "
                f"{float(m.mat_reflectance[i])}), and this renderer draws no mirror"
            )
        for k in sc.prims.kind:
            assert int(k) in (R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX), (
                f"{type(self).__name__}: a moving geom is mjtGeom {int(k)}, which the analytic "
                f"caster has no closed-form intersection for"
            )

    # ---- model / reference plumbing --------------------------------------

    @classmethod
    def _bundle(cls, size: int):
        """``(gym env, unwrapped, MjModel, MjData, pristine)`` for this env at ``size``.

        The gymnasium env is the one whose ``MujocoRenderer`` bakes the plates, so the static half of
        every frame is the reference's own rasterization rather than a reimplementation of it. Built
        once per (env, size) and never closed.
        """
        key = (cls.ENV_ID, int(size))
        hit = _ENV_CACHE.get(key)
        if hit is not None:
            return hit
        env, u = cls._fresh(size)
        m, d = u.model, u.data
        pristine = dict(
            geom_rgba=np.array(m.geom_rgba, copy=True),
            site_rgba=np.array(m.site_rgba, copy=True),
            geom_group=np.array(m.geom_group, copy=True),
            site_group=np.array(m.site_group, copy=True),
            light_diffuse=np.array(m.light_diffuse, copy=True),
            light_specular=np.array(m.light_specular, copy=True),
            cam=cls._read_camera(u),
        )
        _ENV_CACHE[key] = (env, u, m, d, pristine)
        return _ENV_CACHE[key]

    @staticmethod
    def _read_camera(u) -> dict:
        cam = u.mujoco_renderer.viewer.cam
        return dict(
            lookat=np.array(cam.lookat, np.float64),
            distance=float(cam.distance),
            azimuth=float(cam.azimuth),
            elevation=float(cam.elevation),
            type=int(cam.type),
            fixedcamid=int(cam.fixedcamid),
        )

    @classmethod
    def _model(cls):
        return cls._bundle(cls.default_size)[2]

    @classmethod
    def _pristine(cls) -> dict:
        """The model AS FROZEN, which is the only thing an appearance default may be read from.

        The ``MjModel`` in ``_bundle`` is shared by every instance of the class and is mutated by
        every bake; between bakes it carries the LAST appearance drawn, not the official one. Reading
        a default off the live model therefore makes a new env's idea of "default" depend on what the
        previous one most recently rendered — invisible from inside one process, and caught only by
        the cross-process reproducibility check.
        """
        return cls._bundle(cls.default_size)[4]

    # ---- variation --------------------------------------------------------

    #: Style leaves that change the STATIC scene and therefore the baked plates. Everything else is a
    #: property of the movers and is applied per environment inside the ray-cast, so the common case
    #: — randomise the body, keep the scene — never splits the batch and never re-bakes.
    PLATE_PATHS = ("scenery.tint", "camera.distance_scale", "camera.azimuth", "camera.elevation")

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS reproduce the official picture exactly.

        The two colour knobs are TINTS (multipliers, default 1) rather than absolute colours, and
        that is not a shortcut. Every one of these scenes paints its parts different colours on
        purpose — Reacher's pink walls against its white ground, its blue arm against a green
        fingertip and a red target — and one absolute colour per group would erase exactly the
        distinctions the picture is made of. A tint moves the whole group while keeping them apart,
        and its default is an identity, so "the default style is the official appearance" is an
        identity a test can assert rather than a transcribed constant.

        Colours are float ``Box`` rather than ``RGBBox`` for the reason the OGBench mazes are: these
        are float material colours (0.7, 0.7, 0.0) that a uint8 channel cannot hold, and in a lit 3-D
        scene no material colour appears verbatim in any pixel anyway.
        """
        pr = self._pristine()
        return Dict(
            {
                "body": Dict(
                    {
                        "tint": Box(low=0.0, high=2.0, shape=(3,), init_value=[1.0, 1.0, 1.0]),
                        "scale": Box(low=0.3, high=3.0, shape=(1,), init_value=[1.0]),
                    }
                ),
                "scenery": Dict({"tint": Box(low=0.0, high=2.0, shape=(3,), init_value=[1.0, 1.0, 1.0])}),
                "camera": Dict(
                    {
                        "distance_scale": Box(low=0.5, high=2.0, shape=(1,), init_value=[1.0]),
                        # A window around the frozen default rather than a full turn. The range is
                        # not cosmetic: [0, 360] makes its two endpoints the SAME camera, and
                        # ``check_per_env_variation`` — which splits the batch between the two ends and
                        # requires two different pictures — would report a knob that works perfectly as
                        # collapsed.
                        "azimuth": Box(
                            low=float(pr["cam"]["azimuth"]) - 60.0,
                            high=float(pr["cam"]["azimuth"]) + 60.0,
                            shape=(1,),
                            init_value=[float(pr["cam"]["azimuth"])],
                        ),
                        "elevation": Box(low=-89.0, high=-5.0, shape=(1,), init_value=[float(pr["cam"]["elevation"])]),
                    }
                ),
            }
        )

    def _var(self, variation, path):
        if variation is not None and path in variation:
            return np.asarray(variation[path], np.float64).reshape(-1)
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        return np.asarray(node.value, np.float64).reshape(-1)

    # ---- scene bake -------------------------------------------------------

    def _plate_key(self, size, variation):
        return (self.ENV_ID, int(size), str(self.device)) + tuple(
            tuple(np.round(self._var(variation, p), 6)) for p in self.PLATE_PATHS
        )

    def _scene(self, size=None, variation=None) -> _Scene:
        size = int(size or self.size)
        key = self._plate_key(size, variation)
        hit = _SCENE_CACHE.get(key)
        if hit is not None:
            return hit
        return _SCENE_CACHE.put(key, self._bake(size, variation))

    @staticmethod
    def _visible(m, kind: str, pr) -> list[int]:
        grp = pr["geom_group"] if kind == "geom" else pr["site_group"]
        n = m.ngeom if kind == "geom" else m.nsite
        return [i for i in range(n) if int(grp[i]) in VISIBLE_GROUPS]

    def _movers(self, m, pr) -> tuple[list[int], list[int]]:
        """``(geom ids, site ids)`` whose world pose depends on ``qpos``: anything off the world body.

        Sites are in here because MuJoCo draws them at the default ``MjvOption`` and
        InvertedDoublePendulum hangs one off the tip of its second pole. Hiding only the moving GEOMS
        would leave a 0.01 m sphere painted at a fixed place in a "static" plate, which is the
        quietest possible way to break the state-to-frame claim.
        """
        g = [i for i in self._visible(m, "geom", pr) if int(m.geom_bodyid[i]) != 0]
        s = [i for i in self._visible(m, "site", pr) if int(m.site_bodyid[i]) != 0]
        return g, s

    def _apply_variation(self, m, u, pr, variation) -> None:
        """Absolute assignment from the pristine snapshot, never an increment, so baking B after A
        gives the same model as baking B first. ``distance_scale`` is the one that would otherwise
        compound: it is a MULTIPLIER on the camera's own distance."""
        m.geom_rgba[:] = pr["geom_rgba"]
        m.site_rgba[:] = pr["site_rgba"]
        m.geom_group[:] = pr["geom_group"]
        m.site_group[:] = pr["site_group"]
        m.light_diffuse[:] = pr["light_diffuse"]
        m.light_specular[:] = pr["light_specular"]
        tint = self._var(variation, "scenery.tint")
        gmov, smov = self._movers(m, pr)
        for i in self._visible(m, "geom", pr):
            if i not in gmov:
                m.geom_rgba[i, :3] = np.clip(pr["geom_rgba"][i, :3] * tint, 0, 1)
        for i in self._visible(m, "site", pr):
            if i not in smov:
                m.site_rgba[i, :3] = np.clip(pr["site_rgba"][i, :3] * tint, 0, 1)
        cam = u.mujoco_renderer.viewer.cam
        cam.lookat[:] = pr["cam"]["lookat"]
        cam.type = pr["cam"]["type"]
        cam.fixedcamid = pr["cam"]["fixedcamid"]
        cam.distance = pr["cam"]["distance"] * float(self._var(variation, "camera.distance_scale")[0])
        cam.azimuth = float(self._var(variation, "camera.azimuth")[0])
        cam.elevation = float(self._var(variation, "camera.elevation")[0])

    def _bake(self, size: int, variation) -> _Scene:
        import mujoco

        env, u, m, d, pr = self._bundle(size)
        self._apply_variation(m, u, pr, variation)

        gmov, smov = self._movers(m, pr)
        gstat = [i for i in self._visible(m, "geom", pr) if i not in gmov]
        sstat = [i for i in self._visible(m, "site", pr) if i not in smov]

        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        m.geom_group[gmov] = HIDDEN_GROUP
        m.site_group[smov] = HIDDEN_GROUP
        plate = self._shot(u).copy()

        # A second render with the movers still hidden, only to read the scene MuJoCo built: the
        # camera it derived (as two stereo eyes) and the lights it resolved, including the headlight,
        # which exists nowhere in the model.
        scn = u.mujoco_renderer.viewer.scn
        cam = R3.Camera.from_mjv(scn.camera[0], scn.camera[1])
        lights = _Lights(scn)
        sl = lights.shadow_light()

        # Plate 2: the shadow-casting light contributing only its ambient. That IS the colour of a
        # static surface the light does not reach, so MuJoCo's shadow MAP is never modelled — only
        # the shape of the shadow, by one occlusion ray.
        if sl >= 0:
            head = 1 if int(m.vis.headlight.active) else 0
            mi = sl - head
            if 0 <= mi < int(m.nlight):
                m.light_diffuse[mi] = 0.0
                m.light_specular[mi] = 0.0
            plate_shadow = self._shot(u).copy()
            m.light_diffuse[:] = pr["light_diffuse"]
            m.light_specular[:] = pr["light_specular"]
        else:
            plate_shadow = plate

        m.geom_group[:] = pr["geom_group"]
        m.site_group[:] = pr["site_group"]

        sc = _Scene()
        sc.size = size
        sc.model, sc.data = m, d
        sc.cam, sc.lights, sc.shadow_light = cam, lights, sl
        sc.movers = (gmov, smov)
        sc.pristine = pr
        sc.plate = torch.tensor(plate, device=self.device, dtype=torch.uint8)
        sc.plate_shadow = torch.tensor(plate_shadow, device=self.device, dtype=torch.uint8)
        sc.prims, _, _ = self._prims(m, d, gmov, smov)
        # The static depth is NOT cast here: it is needed on the SUPERSAMPLED grid, whose size is a
        # render-time argument, and one bake happens per sampled appearance — so casting it here
        # would pay for a buffer nobody asked for, once per style.
        sc.static_t = {}
        sc.rays = {}
        sc.shadow_candidates = {}
        sc.const = {}
        _, spos, _ = self._prims(m, d, gstat, sstat)
        # A length scale for the shadow ray offset, taken from the scene rather than assumed:
        # InvertedDoublePendulum's floor is 40 m across and Reacher's arena 0.6 m, so one absolute
        # epsilon cannot serve both.
        sc.eps = float(max(1e-6, 1e-5 * np.abs(spos).max())) if len(spos) else 1e-6
        return sc

    def _prims(self, m, d, geoms, sites):
        """``(Primitives, xpos (P,3), xmat (P,3,3))`` for a set of geoms AND sites.

        Sites carry their own type/size/rgba arrays, so they go through a parallel path rather than
        being pretended into geoms.
        """
        kind, size, rgb, spec, shin, pos, mat = [], [], [], [], [], [], []
        for i in geoms:
            mid = int(m.geom_matid[i])
            s, sh = _spec_shin(m, mid)
            kind.append(int(m.geom_type[i]))
            size.append(np.asarray(m.geom_size[i], np.float64))
            rgb.append(_mat_rgb(m, mid, m.geom_rgba[i]))
            spec.append(s)
            shin.append(sh)
            pos.append(np.asarray(d.geom_xpos[i], np.float64))
            mat.append(np.asarray(d.geom_xmat[i], np.float64).reshape(3, 3))
        for i in sites:
            mid = int(m.site_matid[i])
            s, sh = _spec_shin(m, mid)
            kind.append(int(m.site_type[i]))
            size.append(np.asarray(m.site_size[i], np.float64))
            rgb.append(_mat_rgb(m, mid, m.site_rgba[i]))
            spec.append(s)
            shin.append(sh)
            pos.append(np.asarray(d.site_xpos[i], np.float64))
            mat.append(np.asarray(d.site_xmat[i], np.float64).reshape(3, 3))
        if not kind:
            prims = R3.Primitives(
                kind=np.zeros(0, int),
                size=np.zeros((0, 3)),
                mat_rgb=np.zeros((0, 3)),
                tex_rgb=np.zeros((0, 3)),
                spec=np.zeros(0),
            )
            prims.shininess = np.zeros(0)
            return prims, np.zeros((0, 3)), np.zeros((0, 3, 3))
        prims = R3.Primitives(
            kind=np.array(kind),
            size=np.stack(size),
            mat_rgb=np.stack(rgb),
            tex_rgb=np.ones((len(kind), 3)),
            spec=np.array(spec),
        )
        prims.shininess = np.array(shin)
        return prims, np.stack(pos), np.stack(mat)

    def _rays(self, sc: _Scene, ss: int):
        """``(origin, directions)`` for the supersampled grid, cached per (scene, ss).

        ``Camera.rays`` builds the grid in NUMPY and copies it to the device, which is a
        state-independent constant — at 480 px and ss=2 it is 921,600 float64 rows rebuilt on every
        chunk of every render. Caching it is worth 2-5% at 240 px and considerably more at 480.
        """
        hit = sc.rays.get(int(ss))
        if hit is None:
            R = sc.size * int(ss)
            hit = sc.rays[int(ss)] = sc.cam.rays(R, R, self.device)
        return hit

    def _shadow_candidates(self, sc: _Scene, ss: int):
        """Indices of the STATIC pixels a mover's shadow could possibly change, cached per (scene, ss).

        Exact, not an approximation, and that is the whole reason it is allowed: a shadowed pixel
        selects ``plate_shadow`` instead of ``plate``, so wherever the two plates carry the SAME
        colour the occlusion test cannot change the frame no matter what it returns. Intersected with
        "a static surface is actually visible here", since a ray that hits nothing has no surface to
        shadow.

        Worth a lot on the one env that has a shadow at all: Pusher's spotlight only reaches the
        table, and 84% of its frame is background, so the occlusion pass — 19 primitives cast from
        every ray — runs on a sixth of the rays.
        """
        key = (int(ss), bool(self.SHADOW_COMPACTION))
        hit = sc.shadow_candidates.get(key)
        if hit is not None:
            return hit
        a = _upsample_rgb(sc.plate, ss).reshape(-1, 3).to(torch.int16)
        b = _upsample_rgb(sc.plate_shadow, ss).reshape(-1, 3).to(torch.int16)
        if self.SHADOW_COMPACTION:
            cand = ((a != b).any(-1)) & (self._static_t(sc, ss).view(-1) < R3.INF)
            idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
        else:
            idx = torch.arange(a.shape[0], device=a.device)  # the pass this compaction replaced
        rows = _upsample_rgb(sc.plate_shadow, ss).reshape(-1, 3).to(torch.float32)[idx]
        sc.shadow_candidates[key] = hit = (idx, rows.unsqueeze(0))
        return hit

    def _per_chunk_constants(self, sc: _Scene, ss: int):
        """``(o, d, plate_f, size, mat, spec, shin, L)``: everything a chunk needs that no state moves.

        The batch is rendered in ``RAY_BUDGET``-sized chunks, and every one of these was rebuilt per
        chunk. ``torch.tensor(x, device="cuda")`` copies from PAGEABLE host memory, which cannot be
        queued — the call blocks until the stream drains — so four of them per chunk is four stalls
        per chunk for numbers that were already known. Same fix, and the same measurement, as
        ``DMControlSuite._per_chunk_constants``.
        """
        hit = sc.const.get(int(ss))
        if hit is None:
            dev = self.device
            t = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)  # noqa: E731
            P = len(sc.prims.kind)
            hit = sc.const[int(ss)] = (
                *self._rays(sc, ss),
                _upsample_rgb(sc.plate, ss).reshape(1, -1, 3).to(torch.float32),
                t(sc.prims.size).view(1, P, 3),
                t(sc.prims.mat_rgb).view(1, P, 3),
                t(sc.prims.spec),
                t(sc.prims.shininess),
                sc.lights.to(dev),
            )
        return hit

    def _static_shadow_rays(self, sc: _Scene, ss: int):
        """``(idx, dark, o, d, lim)`` for the shadow test on static surfaces, cached per (scene, ss).

        THE SHADOW RAY IS A FUNCTION OF THE SCENE, NOT OF THE STATE: it starts at ``o + t*d`` for the
        STATIC ``t``, points at a light that does not move, and ends at it. Only whether a mover is in
        the way depends on the state, and that is the one fused kernel this leaves behind.
        """
        key = ("shadow", int(ss), bool(self.SHADOW_COMPACTION))
        hit = sc.const.get(key)
        if hit is not None:
            return hit
        cand, dark = self._shadow_candidates(sc, ss)
        o, d = self._rays(sc, ss)
        N = (sc.size * int(ss)) ** 2
        st = self._static_t(sc, ss).view(1, N)
        p = o.expand(1, N, 3)[:, cand] + st[:, cand].clamp(max=1e6).unsqueeze(-1) * d.view(1, N, 3)[:, cand]
        L = sc.lights.to(self.device)
        sl = sc.shadow_light
        shape = (1,) * (p.dim() - 1) + (3,)
        if L["kind"][sl] == LIGHT_DIRECTIONAL:
            rd = (-_norm(L["dir"][sl].view(shape))).expand_as(p)
            lim = torch.full(p.shape[:-1], R3.INF, device=p.device)
        else:
            v = L["pos"][sl].view(shape) - p
            lim = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
            rd = v / lim.unsqueeze(-1)
        # A ray that hits no static surface has no surface to shadow. With the compaction ON this is
        # implied — ``_shadow_candidates`` already intersects with ``st < INF`` — but with it OFF the
        # candidate set is the whole frame, and the occlusion test would then be free to darken sky
        # pixels that the compacted path never asks about. Carrying the mask explicitly is what makes
        # the two legs the same function, which ``tests/gpu/test_fast_paths.py`` checks and which caught
        # this.
        hit = sc.const[key] = (cand, dark, p + rd * sc.eps, rd, lim, st[:, cand] < R3.INF)
        return hit

    def _static_t(self, sc: _Scene, ss: int) -> torch.Tensor:
        """The static ray parameter on the SUPERSAMPLED grid, cached per (scene, ss).

        Cast rather than upsampled from the output-resolution buffer. The shadow test starts from
        ``o + t*d`` at every sample, so a nearest-neighbour ``t`` would put the sub-samples around a
        static silhouette on the wrong surface and scatter shadow pixels along every static edge.
        """
        hit = sc.static_t.get(int(ss))
        if hit is not None:
            return hit
        import mujoco

        m, d = sc.model, sc.data
        pr = sc.pristine
        gmov, smov = sc.movers
        gstat = [i for i in self._visible(m, "geom", pr) if i not in gmov]
        sstat = [i for i in self._visible(m, "site", pr) if i not in smov]
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        sprims, spos, smat = self._prims(m, d, gstat, sstat)
        R = sc.size * int(ss)
        t = R3.static_depth(sc.cam, sprims, spos, smat, R, R, self.device)
        sc.static_t[int(ss)] = t
        return t

    # ---- forward kinematics ----------------------------------------------

    def _fk(self, states: torch.Tensor, sc: _Scene):
        """``(B,nq)`` -> mover poses ``(B,P,3)`` / ``(B,P,3,3)`` through ``mj_kinematics``.

        A pure function of ``qpos``, a couple of microseconds a call, and a Python loop over the
        batch — which reads like the bottleneck and is not: the ray passes dominate by an order of
        magnitude at any size worth rendering.
        """
        import mujoco

        gmov, smov = sc.movers
        P = len(gmov) + len(smov)
        B = states.shape[0]
        q = states.detach().cpu().numpy().astype(np.float64)
        xp = np.empty((B, P, 3))
        xm = np.empty((B, P, 9))
        m, d = sc.model, sc.data
        # ``np.take(..., out=)`` rather than ``arr[ids]``: the fancy index allocates a fresh array per
        # row and the ``reshape(-1,3,3)`` allocates again, which measured 4.8 ms of a 10.4 ms batch-1024
        # call on the OGBench ant against 1.6 ms for the kinematics it surrounds. MuJoCo stores
        # ``*_xmat`` as (n, 9), so the reshape is one view at the end instead of B of them inside.
        ng, ns = np.asarray(gmov, np.intp), np.asarray(smov, np.intp)
        for i in range(B):
            d.qpos[:] = q[i]
            mujoco.mj_kinematics(m, d)
            if gmov:
                np.take(d.geom_xpos, ng, axis=0, out=xp[i, : len(gmov)])
                np.take(d.geom_xmat, ng, axis=0, out=xm[i, : len(gmov)])
            if smov:
                np.take(d.site_xpos, ns, axis=0, out=xp[i, len(gmov) :])
                np.take(d.site_xmat, ns, axis=0, out=xm[i, len(gmov) :])
        return (
            torch.tensor(xp, device=self.device, dtype=torch.float32),
            torch.tensor(xm.reshape(B, P, 3, 3), device=self.device, dtype=torch.float32),
        )

    # ---- ray-cast helpers -------------------------------------------------

    @staticmethod
    def _cast(sc, o, d, xpos, xmat, sizes):
        """Nearest mover hit over a ray bundle: ``(t, normal, primitive index)``."""
        # Primitives batched, rays tiled -- see raycast3d.cast_nearest for why it is not simply
        # batched over primitives, and for the two invariants that keep it bit-identical to this loop.
        return R3.cast_nearest(sc.prims.kind, sizes, xpos, xmat, o, d)

    @staticmethod
    def _occluded(sc, p, L, sl, xpos, xmat, sizes, skip=None, eps: float = 1e-6):
        """Does any mover block the segment from ``p`` to light ``sl``? ``(B,N)`` bool.

        ``skip`` is the primitive the ray already sits on, excluded so a convex mover does not shadow
        its own front face. That is exact rather than an epsilon: a ray leaving a convex surface
        toward the light re-enters the same primitive only where ``n.l <= 0``, which the diffuse term
        already zeroes.
        """
        shape = (1,) * (p.dim() - 1) + (3,)
        if L["kind"][sl] == LIGHT_DIRECTIONAL:
            d = (-_norm(L["dir"][sl].view(shape))).expand_as(p)
            lim = torch.full(p.shape[:-1], R3.INF, device=p.device)
        else:
            v = L["pos"][sl].view(shape) - p
            lim = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
            d = v / lim.unsqueeze(-1)
        o = p + d * eps
        return R3.any_blocking(sc.prims.kind, sizes, xpos, xmat, o, d, lim, skip=skip)

    # ---- render -----------------------------------------------------------

    @torch.no_grad()
    def render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8. Style may differ per environment.

        The batch is grouped by the plate-affecting half of the style: environments sharing a scenery
        tint and a camera share one bake and are rendered together. Body tint and body scale are
        per-environment inside the ray-cast, so randomising the body never splits the batch.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = s.shape[0]
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        key = torch.cat([sty[p].reshape(B, -1) for p in self.PLATE_PATHS], 1).cpu().numpy()
        uniq, inv = np.unique(np.round(key.astype(np.float64), 6), axis=0, return_inverse=True)
        offs, o = [], 0
        for p in self.PLATE_PATHS:
            w = int(sty.layout[p][1])
            offs.append((o, w))
            o += w
        tint = sty["body.tint"].reshape(B, 3).to(torch.float32)
        scale = sty["body.scale"].reshape(B, 1).to(torch.float32)
        for gi in range(len(uniq)):
            sub = np.flatnonzero(inv == gi)
            over = {p: uniq[gi][a : a + w] for p, (a, w) in zip(self.PLATE_PATHS, offs)}
            sc = self._scene(size, over)
            idx = torch.as_tensor(sub, device=self.device, dtype=torch.long)
            step = max(1, int(self.RAY_BUDGET // max(1, size * size * ss * ss)))
            for a in range(0, len(sub), step):
                part = idx[a : a + step]
                buf[part] = self._render_group(sc, s[part], size, ss, tint[part], scale[part])
        return buf.view(*lead, size, size, 3) if lead else buf

    #: Screen windows tried, as a divisor of the output edge. Smallest first. Mirrors the DM Control
    #: ladder, including the entries past 2: a half-frame window misses scenes whose movers are spread
    #: out but sparse, and missing by a few pixels costs the whole optimisation rather than part of it.
    MOVER_TILES = (8, 6, 4, 3, 2, 1.5, 1.25)

    #: Restrict the shadow-from-static pass to the samples where the lit and unlit plates disagree,
    #: and the mover shading to the rays that actually hit a mover. Class flags so
    #: ``tests/gpu/test_fast_paths.py`` can turn them off and compare against the passes they
    #: replaced; there is no reason for a caller to. Same two switches as ``DMControlSuite``.
    SHADOW_COMPACTION = True
    COVER_COMPACTION = True

    def _mover_idx(self, sc: _Scene, xpos: torch.Tensor, size: int, ss: int):
        """Flat ray indices covering every pixel the movers can touch, or None for the full frame.

        The shadow pass here was already compacted to the covered rays; the primary cast was not, and
        was still testing every primitive against every ray. Same conservative bound as DM Control's:
        the world AABB of each primitive's bounding sphere, all eight corners projected, padded a pixel.
        """
        span = R3.screen_span(sc.cam, sc.prims, xpos, size, size, xpos.device)
        if span is None:
            return None
        x0, y0, w, h = span
        for div in self.MOVER_TILES:
            tile = max(1, int(size / div))
            if tile >= size or w > tile or h > tile:
                continue
            return R3._expand_tile(
                (x0.clamp(0, size - tile), y0.clamp(0, size - tile)), tile, ss, size * ss, xpos.device
            )
        return None

    def _render_group(
        self, sc: _Scene, s: torch.Tensor, size: int, ss: int, tint: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        dev = self.device
        B = s.shape[0]
        R = size * ss
        N = R * R
        o, d, plate_f, psize, pmat, spec, shin, L = self._per_chunk_constants(sc, ss)
        oo = o.expand(B, N, 3)
        dd = d.view(1, N, 3).expand(B, N, 3)

        xpos, xmat = self._fk(s, sc)
        P = xpos.shape[1]
        sizes = (psize * scale.view(B, 1, 1)).expand(B, P, 3)
        mat_rgb = (pmat * tint.view(B, 1, 3)).clamp(0, 1)
        sl = sc.shadow_light
        st = self._static_t(sc, ss).view(1, N).expand(B, N)
        bi = torch.arange(B, device=dev).unsqueeze(1)

        # Restrict the primary cast to the movers' screen window. Outside it ``t_m`` is INF and
        # ``cover`` is False, so the composite is unchanged by construction; ``tests/gpu/
        # test_fast_paths.py`` asserts the two paths agree byte for byte.
        widx = self._mover_idx(sc, xpos, size, ss)
        if widx is None:
            oo_w, dd_w = oo, dd
            t_w, n_w, i_w = self._cast(sc, oo_w, dd_w, xpos, xmat, sizes)
            cover_w = t_w < st
        else:
            w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
            oo_w, dd_w = oo.gather(1, w3), dd.gather(1, w3)
            t_w, n_w, i_w = self._cast(sc, oo_w, dd_w, xpos, xmat, sizes)
            cover_w = t_w < st.gather(1, widx)

        # A WINDOW IS NOT A COVERAGE MASK. The window is a rectangle around the movers' projected
        # bound; on Pusher the movers COVER 2.1% of the frame and their rectangle is 25% of it. The
        # cast is one fused kernel and does not care, but the shadow ray and the ~75 elementwise ops
        # of shading are paid per ray, so narrow to the rays that actually landed on a mover, as the
        # union over the chunk. Restricting the SET cannot change a VALUE — every ray's colour is a
        # function of that ray alone — and outside the union ``cover`` is False for EVERY row, so
        # the composite takes the plate there whatever the shader would have said. (The union is
        # over window POSITIONS, not pixels: position j is a different pixel in each row, which is
        # exactly as sound and for the same reason.)
        if self.COVER_COMPACTION:
            js = torch.nonzero(cover_w.any(0), as_tuple=False).reshape(-1)
            widx = js.view(1, -1).expand(B, -1) if widx is None else widx[:, js]
            cover_w, t_w, n_w, i_w = cover_w[:, js], t_w[:, js], n_w[:, js], i_w[:, js]
            oo_w, dd_w = oo_w[:, js], dd_w[:, js]
        pm_w = oo_w + t_w.clamp(max=1e6).unsqueeze(-1) * dd_w
        unlit_w = None
        if sl >= 0 and P > 1:
            unlit_w = self._occluded(sc, pm_w, L, sl, xpos, xmat, sizes, skip=i_w, eps=sc.eps)
        col_w = _shade(pm_w, n_w, -dd_w, L, mat_rgb[bi, i_w], spec[i_w], shin[i_w], unlit_w) * 255.0

        base = plate_f.expand(B, N, 3).clone()
        if sl >= 0:
            cand, dark, ro, rd, lim, valid = self._static_shadow_rays(sc, ss)
            if cand.numel():
                occ = (
                    R3.any_blocking(
                        sc.prims.kind, sizes, xpos, xmat, ro.expand(B, -1, 3), rd.expand(B, -1, 3), lim.expand(B, -1)
                    )
                    & valid
                )
                base[:, cand] = torch.where(occ.unsqueeze(-1), dark, base[:, cand])

        # The movers go on LAST, straight into the plate canvas: ``where(cover, col, base)`` is the
        # identity wherever ``cover`` is False, and outside the index set it provably is.
        if widx is None:
            frame = torch.where(cover_w.unsqueeze(-1), col_w, base)
        else:
            w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
            frame = base.scatter_(1, w3, torch.where(cover_w.unsqueeze(-1), col_w, base.gather(1, w3)))
        frame = frame.view(B, size, ss, size, ss, 3).sum((2, 4)) / (ss * ss)
        return torch.clamp(torch.round(frame), 0, 255).to(torch.uint8)

    # ---- reference bridge -------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 240):
        """A FRESH official env, frozen exactly the way ``_bundle`` freezes its own.

        Never closed. ``test_state_is_sufficient`` drives one through two histories, and the env owns
        the EGL context its ``MujocoRenderer`` created — releasing it corrupts rendering in every
        other live context in the process.
        """
        env, u = cls._fresh(int(size))
        got = cls._read_camera(u)
        want = cls._pristine()["cam"]
        cls._check_camera_matches(got, want)
        env.unwrapped._se_size = int(size)
        return env

    @staticmethod
    def _check_camera_matches(got: dict, want: dict) -> None:
        """The reference's free camera must be the one the plate was baked through.

        Not decoration. ``_init_camera`` derives ``lookat`` from the state the env is in when its
        viewer is built, so a reference constructed by any other procedure silently looks somewhere
        else and every parity number computed from it compares two different scenes — which reads as
        a rendering discrepancy rather than as a setup error.
        """
        for k in ("distance", "azimuth", "elevation"):
            if abs(float(got[k]) - float(want[k])) > 1e-9:
                raise AssertionError(
                    f"reference camera {k} is {got[k]} but the plate was baked at {want[k]}; the "
                    f"free camera gymnasium freezes at the first render depends on the state it was "
                    f"in, so these two are looking at different scenes"
                )
        if np.abs(np.asarray(got["lookat"]) - np.asarray(want["lookat"])).max() > 1e-9:
            raise AssertionError(
                f"reference camera lookat is {got['lookat']} but the plate was baked at {want['lookat']}"
            )


def _public_mujoco_name(old: str) -> str:
    """``MujocoAnt`` → ``MuJoCoAnt``. Internal tables keep the historical keys."""
    return "MuJoCo" + old[len("Mujoco") :] if old.startswith("Mujoco") else old


def _hyphen_name(env_id: str) -> str:
    """``"HalfCheetah-v5"`` -> ``"mujoco-half-cheetah"``: the dataset name, from the gymnasium id."""
    pretty = env_id.split("-")[0]
    return "mujoco-" + "".join(("-" + c.lower()) if c.isupper() else c for c in pretty).lstrip("-")


def _make_classes():
    made = {}
    for cname, env_id in ENVS.items():
        floor, extra = PARITY.get(cname, (1.0, ""))
        public = _public_mujoco_name(cname)
        cls = register(
            type(
                public,
                (GymMujoco,),
                dict(
                    ENV_ID=env_id,
                    domain=env_id.split("-")[0].lower(),
                    parity_interior_min=floor,
                    parity_note=GymMujoco.parity_note + extra,
                    name=_hyphen_name(env_id),
                    reference_id=f"gymnasium/{env_id}",
                    __doc__=(
                        f"Gymnasium {env_id}. State is qpos; the frame is a measured pure function of "
                        f"it, and the camera gymnasium freezes at the first render is carried in the "
                        f"style."
                    ),
                ),
            )
        )
        _remember_historical_name(cname, cls)
        made[public] = cls
    return made


globals().update(_make_classes())


# --------------------------------------------------------------------------- the VISUAL tier
# The six locomotion domains the four above deliberately exclude. Same family, same state, same
# reference procedure (``_GymMujocoFamily``), a different renderer and a WEAKER claim; see
# ``docs/concepts/fidelity-tiers.md`` and the VISUAL TIER section of this module's docstring. Nothing
# below touches the four parity envs, and nothing below may be read as parity.

#: Why the six are not simply absent when nvdiffrast is not installed AND why this is not a module-
#: level ``raise``: the four parity envs above need nothing but gymnasium, and a module that raises
#: takes them with it. So the failure is captured here and the six are not registered — the same
#: outcome ``robocasa.py`` gets from its ``ImportError``, reached the only way a module with two
#: tiers in it can reach it. ``missing_visual_backend()`` reports the reason.
#:
#: The probe is the FAILING PATH, not ``importlib.util.find_spec``. nvdiffrast is a pip package whose
#: CUDA extension is built at install time and imported at ``nvdiffrast.torch`` import (``ops.py``
#: line 12, ``import _nvdiffrast_c``), so a machine can have the package present and unusable — wrong
#: torch ABI, no ``nvcc`` at install time, an extension built for another architecture. ``find_spec``
#: says yes to every one of those and the env then fails on first render.
try:
    import nvdiffrast.torch as _nvdr

    _ = _nvdr.RasterizeCudaContext  # the class we construct
    NVDIFF_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover
    NVDIFF_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

from ..render import nvdiff as ND  # noqa: E402  (imports neither mujoco nor nvdiffrast at module scope)

#: class name -> gymnasium id. The six that ``docs/research/mujoco_locomotion.md`` Section 6 rules out
#: of the parity tier, for the one measured reason named there.
VISUAL_ENVS: dict[str, str] = {
    "MujocoHalfCheetah": "HalfCheetah-v5",
    "MujocoHopper": "Hopper-v5",
    "MujocoWalker2d": "Walker2d-v5",
    "MujocoAnt": "Ant-v5",
    "MujocoHumanoid": "Humanoid-v5",
    "MujocoSwimmer": "Swimmer-v5",
}

#: Device memory nvdiffrast's instanced mode needs per world, fitted in
#: ``docs/research/nvdiffrast.md`` Section 5 on three scenes spanning 8.5 k to 579 k vertices and
#: predicting a 1.73 M-vertex RoboCasa kitchen to within 6 %. The first term is the rasterizer's own
#: buffers plus the interpolated attributes, the second the ``(B, V, 4)`` clip-space vertex buffer
#: that is the reason this backend's memory grows with the batch at all.
NVD_BYTES_PER_PIXEL, NVD_BYTES_PER_VERTEX = 239, 69
#: The render is chunked to this many bytes of transient device memory. It is a budget rather than a
#: batch because the batch that fits depends on the resolution AND the scene: at 240 px a locomotion
#: scene costs 14.1 MB a world, at 480 px it costs 55. The number matters more here than it does for
#: a raytracing backend, because nvdiffrast does not raise when it runs out — it returns ``CUDA
#: error: an illegal memory access`` and kills the interpreter (``nvdiffrast.md`` Section 5).
NVD_BUDGET_BYTES = 6 << 30

#: ``(env id, size, supersample, device)`` -> ``_Nvd``. Each entry owns an nvdiffrast context, a
#: triangle soup and the texture pyramids, so it is capped; 16 is two sizes for each of the six, which
#: is what one conformance run asks for (the default 240 plus the 2x render
#: ``check_resolution_consistency`` makes) and is the point below which the cache thrashes rather than
#: caches.
_NVD_CACHE = _RES.BoundedCache(16, name="mujoco_locomotion.nvd", on_evict=ND.close_cached_renderer)
_NVD_CACHE_MAX = 16


class _Nvd:
    """One nvdiffrast renderer for one ``(env, size, supersample)``, plus the model it is driven by."""

    __slots__ = (
        "br",
        "model",
        "data",
        "cam_id",
        "fovy",
        "znear",
        "zfar",
        "geoms",
        "lights",
        "chunk",
        "size",
        "ss",
        "shadow",
        "castshadow",
    )


def _camera_pose(pos: np.ndarray, mat: np.ndarray, offset: np.ndarray, pan: np.ndarray, tilt: np.ndarray):
    """``(B,3), (B,3,3)`` camera pose moved by a per-environment offset, pan and tilt.

    The offset is in the CAMERA's own frame and the two angles act on the right of the camera matrix,
    so both are what a mounting tolerance looks like rather than a world-space displacement — the same
    convention ``robocasa.py`` uses, and the reason the knob means the same thing on a camera that
    tracks a moving body as on one bolted to a robot. MuJoCo's camera looks down its local -z with +y
    up, so a pan is a rotation about local y and a tilt about local x.

    At the default (0, 0, 0) this is the identity in exact arithmetic — the rotations are identity
    matrices and the offset adds 0.0 — which is what makes ``default_style()`` reproduce the
    unstyled frame bit for bit rather than nearly.
    """
    n = len(pos)
    a = np.deg2rad(np.asarray(pan, np.float64)).reshape(n)
    b = np.deg2rad(np.asarray(tilt, np.float64)).reshape(n)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    ry = np.zeros((n, 3, 3))
    ry[:, 0, 0], ry[:, 0, 2], ry[:, 1, 1], ry[:, 2, 0], ry[:, 2, 2] = ca, sa, 1.0, -sa, ca
    rx = np.zeros((n, 3, 3))
    rx[:, 0, 0], rx[:, 1, 1], rx[:, 1, 2], rx[:, 2, 1], rx[:, 2, 2] = 1.0, cb, -sb, sb, cb
    return pos + np.einsum("bij,bj->bi", mat, np.asarray(offset, np.float64)), mat @ ry @ rx


#: Texels on a side of the shadow map. Chosen against the SCREEN rather than against MuJoCo's own
#: 4096: at 240 px the camera is ~3 m from the body at fovy 45, so an output pixel is ~1.0 cm on the
#: floor, and 256 texels over the ~4 m box the movers occupy is 1.6 cm — the same order, which is
#: what decides whether the shadow's EDGE is a pixel out. Its interior, which is where the error
#: lives, is exact at any resolution. Measured against 512 in
#: the frozen measurement record rather than assumed.
NVD_SHADOW_RES = 256
#: How far along the surface normal a receiver is pushed before it looks itself up, in TEXELS of the
#: shadow map, and a small constant depth slack in metres on top. Both are swept rather than chosen
#:: too little and a curved surface shadows
#: itself in stripes, too much and the shadow pulls away from the object casting it.
NVD_SHADOW_OFFSET_TEXELS = 1.0
NVD_SHADOW_DEPTH_BIAS = 0.002


class _ShadowPass:
    """The shadow of a DIRECTIONAL light, as one orthographic depth pass. MuJoCo draws it; we did not.

    ``docs/research/nvdiffrast.md`` Section 4.2 measures this as the largest remaining term after the
    minified floor: turning shadows off on the REFERENCE moves HalfCheetah's near floor band from
    0.427 to 0.753 exact. All six of these models light the scene with one directional light straight
    down (``dir="0 0 -1.3"``, ``castshadow`` defaulting to true) plus MuJoCo's headlight, which casts
    nothing — so the general shadow problem does not arise here and the special case is the cheapest
    one there is: a light that never moves, an orthographic projection, and one depth buffer.

    Three decisions, each measurable rather than a matter of taste:

    * **Only the movers cast.** The floor is a 40 m plane and putting it in the map would fill every
      texel, cost the pass its whole fill rate, and make the floor test its own depth against itself —
      which is the classic source of shadow acne and needs a bias to suppress. Leaving it out makes a
      texel with no caster mean exactly "nothing is above this", which is the answer we want.
    * **The box is fitted to the movers per batch**, not to the model's ``stat.extent``. A shadow map
      spread over a 40 m floor at 256 texels would have 16 cm texels and no shadow would survive it.
    * **Nearest, not PCF.** MuJoCo filters a 4096-texel map, whose edge is ~2 mm at this camera and
      therefore hard at 240 px; a 2x2 percentage-closer filter here would soften a 1.6 cm edge into a
      3 cm gradient, which is further from the reference rather than nearer.
    """

    __slots__ = (
        "scene",
        "ctx",
        "light",
        "res",
        "basis",
        "faces",
        "movers",
        "ldir",
        "gmax",
        "depth",
        "center",
        "half",
        "hz",
    )

    def __init__(self, c: _Nvd, light: int, res: int = NVD_SHADOW_RES):
        import torch

        self.scene, self.ctx, self.light, self.res = c.br.scene, c.br.ctx, int(light), int(res)
        f = np.asarray(c.lights.dir[light], np.float64).reshape(3)
        f = f / np.linalg.norm(f)
        self.ldir = -f  # unit vector TOWARD the light
        up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(f, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, f)
        # Columns right / up / BACK, the convention ``nvdiff.Camera`` uses: a camera looks down -z.
        self.basis = np.stack([right, up, -f], -1)
        mov = [i for i, g in enumerate(c.geoms) if int(c.model.geom_type[g]) != ND.PLANE]
        self.movers = np.asarray(mov, np.int64)
        keep = torch.as_tensor(np.isin(c.br.scene.fgeom.cpu().numpy(), mov), device=c.br.scene.face.device)
        self.faces = c.br.scene.face[keep].contiguous()
        #: The largest half-size any casting geom carries, so a capsule's body is inside the box its
        #: CENTRE put it in — the fit below is over geom origins, which are not their extents.
        self.gmax = float(max((np.abs(c.model.geom_size[c.geoms[i]]).max() for i in mov), default=0.0))
        self.depth = self.center = None

    def build(self, xpos, xmat, xp_np: np.ndarray) -> None:
        """Rasterize the movers from the light for this batch. ``xp_np`` is the same pose on the host.

        EVERY quantity here is per WORLD and none is per batch, which is not a style preference: a box
        fitted to the widest body in the batch would make a state's own frame depend on what it was
        rendered ALONGSIDE, and this repo's whole claim is that it does not. Measured on the first
        version, which took the max over the chunk: ``render(x)[i]`` differed from ``render(x[i])`` by
        up to 227/255.
        """
        import nvdiffrast.torch as dr
        import torch

        S = self.scene
        B = int(xpos.shape[0])
        pm = xp_np[:, self.movers]  # (B,M,3)
        ctr = pm.mean(1)
        # The box has to hold the movers AND reach the surface the shadow lands on, which is below
        # them; ``gmax`` is the largest geom radius, so a capsule's body is inside the box its centre
        # put it in.
        rad = np.abs(pm - ctr[:, None]).max((1, 2)) + self.gmax + 0.05  # (B,)
        drop = np.abs(ctr[:, 2]) + rad  # (B,)
        R = self.basis
        view = np.zeros((B, 4, 4))
        view[:, :3, :3] = R.T
        view[:, :3, 3] = -(ctr @ R)
        view[:, 3, 3] = 1.0
        proj = np.zeros((B, 4, 4))
        proj[:, 0, 0] = proj[:, 1, 1] = 1.0 / rad
        proj[:, 2, 2] = -1.0 / drop
        proj[:, 3, 3] = 1.0
        dev = xpos.device
        pos, _, _ = S.clip(
            xpos,
            xmat,
            torch.as_tensor(view, dtype=torch.float32, device=dev),
            torch.as_tensor(proj, dtype=torch.float32, device=dev),
        )
        rast, _ = dr.rasterize(self.ctx, pos, self.faces, resolution=[self.res, self.res], grad_db=False)
        # A texel with no caster is "nothing above this", i.e. the far plane, NOT z = 0.
        z = torch.where(rast[..., 3] > 0, rast[..., 2], torch.ones_like(rast[..., 2]))
        self.depth = z.unsqueeze(1).contiguous()  # (B,1,res,res)
        self.center = torch.as_tensor(ctr, dtype=torch.float32, device=dev)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev).view(B, 1, 1, 1)  # noqa: E731
        self.half, self.hz = t(rad), t(drop)

    def __call__(self, i: int, p, n):
        """``1.0`` where this light reaches the point and ``0.0`` where a mover blocks it."""
        import torch
        import torch.nn.functional as F

        if int(i) != self.light or self.depth is None:
            return None
        B = int(p.shape[0])
        R = torch.as_tensor(self.basis, dtype=p.dtype, device=p.device)
        l = torch.as_tensor(self.ldir, dtype=p.dtype, device=p.device).view(1, 1, 1, 3)
        ndl = (n * l).sum(-1, keepdim=True).clamp(0.05, 1.0)
        texel = 2.0 * self.half / self.res
        # NORMAL-OFFSET bias, and it is the whole difference between a shadow and a picture of
        # stripes. A texel of the map holds ONE depth for a square of world, so a curved surface is
        # compared against a depth taken up to a texel away from where it actually is, and a surface
        # tilted by theta from the light differs from its own neighbour by ``texel * tan(theta)``.
        # Pushing the sample point along the SURFACE NORMAL rather than adding a constant to the depth
        # is what keeps the shadow's own edge in place while the acne goes: measured on Hopper, the
        # first version of this pass put 80% of the pixels it changed on the BODY (where MuJoCo puts
        # 40%) and cost mean 4.51 -> 7.08/255. See ``nvd_probe6_shadowbias.py``.
        slope = torch.sqrt((1.0 - ndl * ndl).clamp_min(0.0)) / ndl
        p = p + n * (texel * NVD_SHADOW_OFFSET_TEXELS * (1.0 + slope))
        q = (p - self.center.view(B, 1, 1, 3)) @ R  # light-eye coordinates
        uv = q[..., :2] / self.half
        z = -q[..., 2:3] / self.hz
        zmap = F.grid_sample(self.depth, uv, mode="nearest", padding_mode="border", align_corners=False).permute(
            0, 2, 3, 1
        )
        outside = (uv.abs() > 1.0).any(-1, keepdim=True)
        return ((z <= zmap + NVD_SHADOW_DEPTH_BIAS / self.hz) | outside).to(p.dtype)


def _one_orientation(mat: np.ndarray) -> np.ndarray:
    """``(B,3,3)`` -> ``(1,3,3)`` when every world's camera has the SAME orientation, else unchanged.

    Not a micro-optimisation, and it is not about this env's arithmetic: ``nvdiff.BatchRenderer``
    builds the skybox's per-pixel ray grid from ``cam.mat`` in NUMPY, on the CPU, in float64, once per
    call — ``(B, H, W, 3)`` when the camera is per world and ``(1, H, W, 3)`` when it is shared, after
    which it is expanded. At 240 px and batch 256 that is a 353 MB host-side array and it costs more
    than the entire rasterization: MEASURED 217 f/s with a per-world matrix against 6.3 k with a
    shared one, on a scene where the skybox is 53 % of the frame.

    A ``trackcom`` camera TRANSLATES and never rotates, so the shared case is the default case here
    and the per-world case arises only from a per-environment ``camera.pan``/``camera.tilt``. The
    reduction is exact rather than approximate — the rows are compared for bitwise equality, and
    nvdiffrast's own ``expand`` of the shared grid produces the identical numbers the per-row
    computation would — so ``render(x)[i]`` is byte-identical either way. That is asserted, not
    assumed: ``check_per_env_variation`` renders half a batch at each end of the pan knob.
    """
    return mat[:1] if len(mat) > 1 and bool((mat == mat[:1]).all()) else mat


def _lights_for(base: ND.Lights, scale: np.ndarray, tint: np.ndarray) -> ND.Lights:
    """The scene's lights with a PER-WORLD intensity, without a render pass per world.

    ``nvdiff.BatchRenderer`` multiplies each light's ``ambient``/``diffuse``/``specular`` into a
    ``(B,H,W,3)`` colour, so an intensity carrying its own leading batch axis broadcasts through the
    shading law untouched: the arrays here are ``(L,B,1,1,3)`` where the renderer's own are ``(L,3)``.
    That is what keeps a batch of 64 different lightings ONE render rather than 64, which is the
    capability this tier exists for. It is also the one place this env leans on a shape the renderer
    does not declare, so it fails loudly rather than quietly if that changes — a shading law that
    reshaped the intensity to ``(1,1,1,3)`` would raise on the ``(B,1,1,3)`` array rather than
    silently render environment 0's lighting for everyone, and ``check_per_env_variation`` asserts the
    outcome either way.

    AMBIENT IS LEFT ALONE and only the diffuse and specular are scaled, which is what a light being
    dimmer or warmer does: ambient in the fixed-function pipeline is the scene's fill, not this
    light's output. Scaling it as well would make the knob an exposure control that no lighting
    change can produce.
    """
    w = np.asarray(scale, np.float64).reshape(-1, 1) * np.asarray(tint, np.float64).reshape(-1, 3)
    if bool((w == 1.0).all()):
        return base  # the default, and provably the official picture
    w = w[None, :, None, None, :]  # (1,B,1,1,3)
    b3 = lambda x: np.asarray(x, np.float64)[:, None, None, None, :]  # noqa: E731  (L,1,1,1,3)
    return ND.Lights(
        pos=base.pos,
        dir=base.dir,
        directional=base.directional,
        ambient=base.ambient,
        diffuse=b3(base.diffuse) * w,
        specular=b3(base.specular) * w,
        attenuation=base.attenuation,
        cutoff=base.cutoff,
        exponent=base.exponent,
        headlight=base.headlight,
    )


class GymMujocoVisual(_GymMujocoFamily):
    """Base for the six locomotion domains. VISUAL TIER — see the module docstring. ``ENV_ID`` per
    subclass."""

    #: Measured, not chosen: at 240 px against gymnasium's own MSAA'd frame, supersampling COSTS
    #: fidelity on every one of the six (HalfCheetah mean 4.88 -> 6.28/255, SSIM 0.960 -> 0.944, and
    #: the same direction on the other five). MuJoCo's ``offsamples=4`` anti-aliases geometry edges
    #: and nothing else; supersampling this renderer also averages the FLOOR TEXTURE, and it computes
    #: the mip level of detail at the finer grid, which sharpens exactly the surface that MuJoCo's
    #: mipmap has smoothed. So the cheap setting is also the accurate one here, which is not the case
    #: for the parity family above (``default_supersample = 2``).
    default_supersample = 1
    #: Draw the shadow-casting light's shadow map. A class attribute rather than a constructor
    #: argument because the declared ``visual_fidelity`` is measured WITH it, so an instance that
    #: turned it off would be an env whose numbers do not describe it; the ablation that measured what
    #: it is worth subclasses to flip it.
    SHADOWS = True
    fidelity_tier = "visual"
    parity_interior_min = None
    parity_note = ""

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        if NVDIFF_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                f"{type(self).__name__} renders through nvdiffrast, which this "
                f"interpreter cannot import: {NVDIFF_IMPORT_ERROR}"
            )
        self._declare_state(self._model())
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        self._check_scene_is_supported(self._nvd())

    # ---- model / renderer -------------------------------------------------

    @classmethod
    def _model(cls):
        """The compiled model, off the HEADLESS sampler env — no viewer, no GL context.

        It is the same compilation the reference renders through (one MJCF, one compiler, and
        ``_fresh`` changes only ``vis.global_.offwidth``/``offheight``, which are framebuffer sizes
        and reach no geometry), so this costs nothing and avoids owning an EGL context per env for a
        model read.
        """
        return cls._sampler().unwrapped.model

    def _nvd(self, size: int | None = None, ss: int | None = None) -> _Nvd:
        import mujoco

        size = int(self.size if size is None else size)
        ss = int(self.ss if ss is None else ss)
        key = (self.ENV_ID, size, ss, str(self.device))
        hit = _NVD_CACHE.get(key)
        if hit is not None:
            return hit
        m = self._model()
        c = _Nvd()
        c.model, c.data, c.size, c.ss = m, mujoco.MjData(m), size, ss
        c.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if c.cam_id < 0:  # pragma: no cover
            raise ValueError(
                f"{type(self).__name__}: {self.ENV_ID} has no 'track' camera, which is "
                f"the one gymnasium renders through by default (MujocoRenderer falls "
                f"back to camera_name='track'); there is nothing to be parity or visual "
                f"against"
            )
        c.fovy = float(m.cam_fovy[c.cam_id])
        c.znear = float(m.vis.map.znear) * float(m.stat.extent)
        c.zfar = float(m.vis.map.zfar) * float(m.stat.extent)
        c.br = ND.BatchRenderer(m, res=size, device=str(self.device), supersample=ss)
        # The lights and the skybox as ``mjv_updateScene`` RESOLVES them, at the camera the reference
        # uses. The headlight exists nowhere in the model and the model light inherits defaults, so
        # the scene MuJoCo builds is the only trustworthy source for either.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = int(c.cam_id)
        mujoco.mj_forward(m, c.data)
        _, scn = c.br.resolve_from_mujoco(m, c.data, camera=cam)
        c.lights = c.br.lights
        c.geoms = np.asarray(c.br.scene.geoms, np.int64)
        # WHICH light casts a shadow is read off the scene MuJoCo built, not off the model: the
        # headlight is not in the model at all, and it is scene light 0 when it is active, so a model
        # index used against the scene's light array is off by one exactly when the headlight is on.
        # ``nvdiff.Lights`` does not carry ``castshadow`` — it is a property of the light for the
        # SHADOW pass rather than for the shading law — so it is taken from the MjvScene here.
        c.castshadow = np.array([int(scn.lights[i].castshadow) for i in range(int(scn.nlight))])
        cast = [
            i
            for i in range(int(scn.nlight))
            if c.castshadow[i] and bool(c.br.lights.directional[i]) and not bool(c.br.lights.headlight[i])
        ]
        c.shadow = _ShadowPass(c, cast[0], NVD_SHADOW_RES) if (self.SHADOWS and cast) else None
        per_world = NVD_BYTES_PER_PIXEL * (size * ss) ** 2 + NVD_BYTES_PER_VERTEX * c.br.scene.V
        c.chunk = max(1, int(NVD_BUDGET_BYTES // max(1, per_world)))
        _NVD_CACHE.put(key, c)
        return c

    def _check_scene_is_supported(self, c: _Nvd) -> None:
        """Refuse to draw a scene this backend cannot draw, instead of drawing it wrong.

        Every one of these holds for all six domains today; each is also something a future gymnasium
        release could change without anyone noticing, and the failure mode of each is a
        plausible-looking picture rather than an error. The geom count is the one that matters most:
        ``nvdiff.Scene`` skips heightfields and SDF geoms SILENTLY, so a scene that grew one would
        simply be missing a surface.
        """
        m = c.model
        visible = [
            g
            for g in range(int(m.ngeom))
            if int(m.geom_group[g]) in ND.DEFAULT_GROUPS and ND.effective_rgba(m, g)[3] > 0.0
        ]
        assert len(c.geoms) == len(visible), (
            f"{type(self).__name__}: nvdiff.Scene drew {len(c.geoms)} of {len(visible)} visible "
            f"geoms — the missing ones are a geom type it skips (heightfield, SDF), and it skips "
            f"them silently, so the frame would simply be missing a surface"
        )
        assert int(m.nlight) >= 1 and int(m.vis.headlight.active), (
            f"{type(self).__name__}: the lighting this env's fidelity was measured under is the "
            f"model light plus MuJoCo's headlight; this model resolves to nlight={int(m.nlight)}, "
            f"headlight active={int(m.vis.headlight.active)}"
        )
        # A shadow-casting light this pass cannot draw is a SILENTLY missing shadow, which is the same
        # class of error as a silently skipped geom. ``_ShadowPass`` projects orthographically and is
        # therefore directional-only; all six of these models carry exactly one such light and no
        # other caster.
        L = c.br.lights
        unhandled = [i for i in range(len(L.pos)) if c.castshadow[i] and not L.headlight[i] and not L.directional[i]]
        assert not unhandled, (
            f"{type(self).__name__}: lights {unhandled} cast shadows and are not directional; "
            f"_ShadowPass draws an ORTHOGRAPHIC map and would leave their shadows out of the frame "
            f"without saying so"
        )

    # ---- variation --------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS are the official picture, and which vary PER WORLD.

        Every leaf here is applied inside ONE render: the camera pose because ``nvdiff.Camera``
        already carries a leading batch dimension (which is what a tracking camera needs anyway), and
        the light intensity through ``_lights_for``. A batch of B appearances is one rasterization,
        which is the capability the visual tier is for.

        What is deliberately NOT a knob, and the reason is the backend rather than the scene:
        anything that changes a geom's MATERIAL COLOUR or its SIZE. ``nvdiff.Scene`` holds both as
        per-geom constants baked at construction — the colour in ``mat_rgb``, the size in the triangle
        soup itself — so a per-environment value for either would mean one Scene, one nvdiffrast
        context and one render pass per distinct value. That is exactly the batch-splitting the parity
        family above pays for its plate and the thing this tier is supposed to avoid, so the knob is
        not offered rather than offered at a cost the name does not advertise. The parity four DO
        carry ``body.tint`` and ``body.scale``, because their movers are ray-cast per environment.
        """
        return Dict(
            {
                "camera": Dict(
                    {
                        # In the camera's own frame. Small, because this camera is 3-4 m from a 1 m body: 6 cm
                        # is ~6 px at 240, and a metre would be a different shot rather than a different
                        # camera.
                        "offset": Box(low=-0.06, high=0.06, shape=(3,), init_value=[0.0, 0.0, 0.0]),
                        "pan": Box(low=-8.0, high=8.0, shape=(1,), init_value=[0.0]),
                        "tilt": Box(low=-8.0, high=8.0, shape=(1,), init_value=[0.0]),
                    }
                ),
                "light": Dict(
                    {
                        "scale": Box(low=0.45, high=1.6, shape=(1,), init_value=[1.0]),
                        "tint": Box(low=0.6, high=1.4, shape=(3,), init_value=[1.0, 1.0, 1.0]),
                    }
                ),
            }
        )

    # ---- forward kinematics ----------------------------------------------

    def _pose(self, c: _Nvd, q: np.ndarray):
        """``(B,nq)`` -> geom poses ``(B,G,3)/(B,G,3,3)`` and the tracking camera ``(B,3)/(B,3,3)``.

        ``mj_kinematics`` alone is NOT enough and the failure is silent: a ``trackcom`` camera reads
        ``subtree_com``, which ``mj_kinematics`` does not write, so the camera would stay wherever it
        was last computed and the picture would be the right scene from the wrong place. ``mj_comPos``
        supplies it. That the three-call sequence reproduces ``mj_forward`` EXACTLY — 0.0 on geom
        poses and on the camera, all six envs, 16 states — is measured in
        the frozen measurement record, and it is worth doing because it skips the
        whole dynamics pipeline for a quantity that is pure kinematics.

        The camera is written into a preallocated row rather than appended, because ``d.cam_xpos[i]``
        is a VIEW into ``MjData``: appending it in a loop gives every state the LAST state's camera,
        which on a tracking camera is a plausible-looking picture of the right scene from the wrong
        place (frame mean 34.8/255 against 3.1, ``docs/research/nvdiffrast.md`` Section 2.4).
        """
        import mujoco

        m, d = c.model, c.data
        B, G = len(q), len(c.geoms)
        xp = np.empty((B, G, 3))
        xm = np.empty((B, G, 3, 3))
        cp = np.empty((B, 3))
        cm = np.empty((B, 3, 3))
        for i in range(B):
            d.qpos[:] = q[i]
            d.qvel[:] = 0
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_camlight(m, d)
            xp[i] = d.geom_xpos[c.geoms]
            xm[i] = d.geom_xmat[c.geoms].reshape(-1, 3, 3)
            cp[i] = d.cam_xpos[c.cam_id]
            cm[i] = d.cam_xmat[c.cam_id].reshape(3, 3)
        return xp, xm, cp, cm

    # ---- render -----------------------------------------------------------

    @torch.no_grad()
    def render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device. Style may differ per environment.

        THESE ARE NOT THE OFFICIAL PIXELS. See ``visual_fidelity`` for what they are instead.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        sty = self.resolve_style(style, variation, batch=B)
        c = self._nvd(size, ss)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        # ONE device-to-host read for the whole appearance, not one per leaf. Five ``sty[path].cpu()``
        # calls are five synchronisations, which is invisible at batch 1024 and is most of the frame
        # at batch 1 — and this env's headline is that a frame costs a fraction of a millisecond.
        flat = sty.values.detach().cpu().numpy()
        leaf = lambda p: flat[:, sty.layout[p][0] : sty.layout[p][0] + sty.layout[p][1]]  # noqa: E731

        q = s.detach().cpu().numpy().astype(np.float64)
        xp, xm, cp, cm = self._pose(c, q)
        cp, cm = _camera_pose(cp, cm, leaf("camera.offset"), leaf("camera.pan"), leaf("camera.tilt"))
        lscale, ltint = leaf("light.scale"), leaf("light.tint")
        xp_t = torch.as_tensor(xp, dtype=torch.float32, device=self.device)
        xm_t = torch.as_tensor(xm, dtype=torch.float32, device=self.device)
        # EVEN chunks, not "fill the budget and take what is left". Both are bit-identical (measured:
        # every batch size agrees with batch 1 to 0/255) and the ragged one is slower where it matters:
        # at 240 px the budget allows 457 worlds, so a batch of 512 becomes 457 + 55 and the tail pays
        # a whole launch for a tenth of the work — 12.4 k f/s against 14.0 k for the same frames.
        step = -(-B // max(1, -(-B // c.chunk)))
        for lo in range(0, B, step):
            hi = min(lo + step, B)
            cam = ND.Camera(cp[lo:hi], _one_orientation(cm[lo:hi]), c.fovy, c.znear, c.zfar)
            c.br.set_lights(_lights_for(c.lights, lscale[lo:hi], ltint[lo:hi]))
            if c.shadow is not None:
                c.shadow.build(xp_t[lo:hi], xm_t[lo:hi], xp[lo:hi])
                c.br.shadow = c.shadow
            buf[lo:hi] = c.br.render(xp_t[lo:hi], xm_t[lo:hi], cam)
        c.br.set_lights(c.lights)
        c.br.shadow = None
        return buf.view(*lead, size, size, 3) if lead else buf

    # ---- reference bridge -------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 240):
        """A FRESH official env, and an assertion that it is looking through the camera we are.

        Never closed: the env owns the EGL context its ``MujocoRenderer`` created, and releasing one
        corrupts rendering in every other live context in the process.

        Unlike the parity four there is no frozen free camera to reproduce — these six have a
        ``track`` camera in the XML and ``MujocoRenderer.render`` defaults to ``camera_name="track"``
        — so what has to be checked is the other thing: that gymnasium still resolves that default to
        the same camera this renderer drives. If it ever falls back to the free camera (id -1), every
        number measured here would describe two different shots and read as a rendering discrepancy.
        """
        import mujoco

        env, u = cls._fresh(int(size))
        m = u.model
        want = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        got = int(u.mujoco_renderer.camera_id)
        if got != want:
            raise AssertionError(
                f"{cls.__name__}: gymnasium rendered through camera {got} but this env draws camera "
                f"{want} ('track'); the two are different shots of the same scene and every fidelity "
                f"number computed from them would be meaningless"
            )
        u._se_size = int(size)
        return env


#: MEASURED through the env's own surface, by ``stable_envs.testing.conformance.check_visual_fidelity``
#: — the same function the conformance suite calls — over independent draws from ``sample_states`` at
#: the family's own 240 px. ``*_max`` / ``*_min`` are the thresholds the suite asserts and sit clear of
#: the worst draw, so a state draw cannot flake the gate; both are here so the margin is visible rather
#: than implied. ``interior_*`` are reported and NOT asserted: the visual tier does not claim
#: interior exactness, and turning a number reported for honesty into a target is how it stops being
#: reported.
_VIS_AT = (
    "240 px, supersample 1, 64 states x 4 independent draws from sample_states, one H200 "
    "(driver 590.48.01), nvdiffrast 0.4.0 / mujoco 3.9.0 / gymnasium 1.2.3, against "
    "gymnasium's own rgb_array render at the model's track camera (MuJoCo offsamples=4); "
    "the values below are the WORST of the four draws"
)
VISUAL_FIDELITY: dict[str, dict] = {
    "MujocoHalfCheetah": {
        "mean_abs": 4.089,
        "within8_frac": 0.8734,
        "ssim": 0.9666,
        "interior_mean_abs": 0.638,
        "interior_exact_frac": 0.594,
        "draw_spread": {"mean_abs": [3.955, 4.089], "ssim": [0.9666, 0.9677]},
        "mean_abs_max": 5.5,
        "within8_min": 0.83,
        "ssim_min": 0.94,
        "measured_at": _VIS_AT,
    },
    "MujocoHopper": {
        "mean_abs": 4.346,
        "within8_frac": 0.8462,
        "ssim": 0.9483,
        "interior_mean_abs": 0.491,
        "interior_exact_frac": 0.853,
        "draw_spread": {"mean_abs": [4.322, 4.346], "ssim": [0.9483, 0.9499]},
        "mean_abs_max": 5.8,
        "within8_min": 0.80,
        "ssim_min": 0.92,
        "measured_at": _VIS_AT,
    },
    "MujocoWalker2d": {
        "mean_abs": 3.861,
        "within8_frac": 0.8704,
        "ssim": 0.9467,
        "interior_mean_abs": 0.370,
        "interior_exact_frac": 0.799,
        "draw_spread": {"mean_abs": [3.814, 3.861], "ssim": [0.9467, 0.9491]},
        "mean_abs_max": 5.2,
        "within8_min": 0.83,
        "ssim_min": 0.92,
        "measured_at": _VIS_AT,
    },
    "MujocoAnt": {
        # The largest body in the family (10.3% of the frame) and the one the shadow pass is worth
        # most to: mean 6.21 -> 3.43/255 and SSIM 0.947 -> 0.977 against the same reference.
        "mean_abs": 3.430,
        "within8_frac": 0.8935,
        "ssim": 0.9771,
        "interior_mean_abs": 0.603,
        "interior_exact_frac": 0.642,
        "draw_spread": {"mean_abs": [3.329, 3.430], "ssim": [0.9771, 0.9790]},
        "mean_abs_max": 4.6,
        "within8_min": 0.85,
        "ssim_min": 0.95,
        "measured_at": _VIS_AT,
    },
    "MujocoHumanoid": {
        "mean_abs": 4.387,
        "within8_frac": 0.8390,
        "ssim": 0.9516,
        "interior_mean_abs": 0.623,
        "interior_exact_frac": 0.880,
        "draw_spread": {"mean_abs": [4.370, 4.387], "ssim": [0.9516, 0.9527]},
        "mean_abs_max": 5.8,
        "within8_min": 0.79,
        "ssim_min": 0.92,
        "measured_at": _VIS_AT,
    },
    "MujocoSwimmer": {
        # The best of the six on every column and the worst on interior exactness, which is the one
        # number the tier reports without asserting: its floor is 94.9% of the frame and MAGNIFIED
        # rather than minified, so almost every pixel of it has a flat 3x3 neighbourhood to be counted
        # in and a magnification filter to disagree about.
        "mean_abs": 1.972,
        "within8_frac": 0.9454,
        "ssim": 0.9810,
        "interior_mean_abs": 0.458,
        "interior_exact_frac": 0.542,
        "draw_spread": {"mean_abs": [1.931, 1.972], "ssim": [0.9810, 0.9826]},
        "mean_abs_max": 2.7,
        "within8_min": 0.91,
        "ssim_min": 0.96,
        "measured_at": _VIS_AT,
    },
}


VISUAL_NOTE = (
    "ONE DOMINANT RESIDUAL, and it is the same operation that rules these six out of the parity "
    "tier: TEXTURE MINIFICATION ON THE CHECKER FLOOR. MatPlane is a 100x100 checker at texrepeat 60 "
    "on a 20-40 m plane seen from ~3 m, so most of it is minified, and what MuJoCo draws out there is "
    "GL's mipmapped average -- luminance spread 0.0-6.8 about a single grey of 93.7 in the far tenth "
    "of the floor, against 74.9-90.1 spanning black to white in the nearest tenth. A renderer that "
    "resolves one texel per pixel paints speckle there instead, which is what caps mujoco_warp at "
    "0.0017-0.0069% of far-field floor pixels bit-exact and is why those six do not ship at parity. "
    "nvdiffrast samples a prefiltered pyramid off image-space uv derivatives and reaches 66-81% of "
    "those pixels bit-exact; the SAME pipeline with filter_mode switched to plain linear lands at "
    "0.00-0.01%, i.e. exactly in mujoco_warp's regime, which is the control that says the two were "
    "failing for one reason. What is left on that surface is that nvdiffrast's trilinear pyramid is "
    "not GL's, and that the 100x100 checker has to be RESAMPLED to 128x128 to have a pyramid at all "
    "(nvdiffrast refuses to halve an odd extent) -- a declared change to the picture, recorded per "
    "texture in nvdiff.prepare_texture(...)['resampled'], worth 3-9 levels of luminance spread "
    "against a reference of 0-3. SECONDARY RESIDUALS, in the order they cost pixels: SILHOUETTES, "
    "because MuJoCo multisamples at offsamples=4 and this rasterizer does not; and the SKYBOX, which "
    "is 52-55% of the frame and sits one level out on a smooth gradient. WHAT WAS CLOSED RATHER THAN "
    "DECLARED, because a residual that can be removed should not be documented instead: the model "
    "light's SHADOW, drawn by an orthographic depth pass over the moving geoms (_ShadowPass) -- worth "
    "mean 6.21 -> 3.50/255 and SSIM 0.947 -> 0.977 on Ant, 4.85 -> 4.09 and 0.959 -> 0.967 on "
    "HalfCheetah, 2.68 -> 1.91 and 0.968 -> 0.983 on Swimmer, for 22% of the frame rate. MEASURED AND "
    "REFUTED, recorded so they are not tried again: SUPERSAMPLING, the obvious answer to the MSAA "
    "residual, which makes every number worse here because it also averages the floor texture and "
    "picks the mip level on the finer grid (HalfCheetah mean 4.88 -> 6.28, SSIM 0.960 -> 0.944); "
    "ANISOTROPIC filtering, implemented behind BatchRenderer(aniso=N) and refuted at 4 and 16 taps "
    "(far-band exactness 0.757 -> 0.277 / 0.236 -- MuJoCo is doing plain isotropic trilinear); the "
    "floor's MIRROR, which MatPlane declares at reflectance 0.5 and which moves interior exactness by "
    "-0.012 to +0.005, in both directions, i.e. noise, on this backend and on mujoco_warp alike; and "
    "a shadow bias without a NORMAL OFFSET, which shadows the body instead of the floor and costs "
    "mean 4.51 -> 7.08 on Hopper. WHAT THIS TIER DOES NOT LICENSE: training on these frames and "
    "evaluating a policy on official gymnasium pixels. The two pixel distributions differ by the "
    "amounts above and nothing in this repo measures what that difference costs a policy. The four "
    "PARITY envs in this same module (Reacher, Pusher, InvertedPendulum, InvertedDoublePendulum) are "
    "not affected by any of this and still claim the official frame."
)


def missing_visual_backend() -> str | None:
    """Why the six visual-tier locomotion envs are absent on this machine, or ``None``.

    They need nvdiffrast, which the four parity envs in this module do not, so a machine without it
    loses six envs rather than ten and ``stable_envs.envs.import_errors()`` — which reports whole
    modules — cannot say so. This is the same reporter ``robocasa.missing_kitchens()`` is.
    """
    return NVDIFF_IMPORT_ERROR


def _make_visual_classes():
    made = {}
    if NVDIFF_IMPORT_ERROR is not None:  # pragma: no cover
        return made
    for cname, env_id in VISUAL_ENVS.items():
        vf = VISUAL_FIDELITY.get(cname)
        if vf is None:  # pragma: no cover
            continue
        public = _public_mujoco_name(cname)
        cls = register(
            type(
                public,
                (GymMujocoVisual,),
                dict(
                    ENV_ID=env_id,
                    domain=env_id.split("-")[0].lower(),
                    name=_hyphen_name(env_id),
                    reference_id=f"gymnasium/{env_id}",
                    visual_fidelity=vf,
                    visual_note=VISUAL_NOTE,
                    __doc__=(
                        f"Gymnasium {env_id}, rendered by nvdiffrast. State is qpos.\n\n"
                        f"VISUAL TIER — these are NOT the official pixels. The frame is a different "
                        f"rasterization of the same scene from the same qpos: same geometry, same "
                        f"tessellation, same track camera, same materials and lights. Measured against "
                        f"gymnasium's own render at 240 px: mean |diff| {vf['mean_abs']:.2f}/255, "
                        f"{100 * vf['within8_frac']:.1f}% of pixels within 8/255, SSIM {vf['ssim']:.4f}, and "
                        f"{100 * vf['interior_exact_frac']:.1f}% of interior pixels bit-exact against the "
                        f"100% every parity env in this registry declares.\n\n"
                        f"Suitable for generating training frames live. NOT a substitute for a stored dataset "
                        f"when a policy will be evaluated on official gymnasium pixels. See visual_note."
                    ),
                ),
            )
        )
        _remember_historical_name(cname, cls)
        made[public] = cls
    return made


globals().update(_make_visual_classes())

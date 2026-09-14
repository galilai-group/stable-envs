"""``stable_worldmodel``'s two torch-rendered navigation envs: ``swm/TwoRoom-v1`` and ``swm/Piecewise-v0``.

Both official envs already draw with torch on the CPU, one environment at a time, in float32. That is
an unusual reference to port and it changes what "parity" can mean here. There is no rasterizer to
match: the reference *is* arithmetic, so reproducing it batched on the GPU is a question of performing
the same operations in the same order and the same dtype, and the result is expected to be bit-exact
rather than exact-up-to-anti-aliasing. Measured, it is — see the numbers at the bottom of this
docstring.

What the official renderer actually does, per frame (``TwoRoomEnv._render_frame``):

    img   = uint8 (3,H,W) filled with background.color
    doors = wall stripe AND door span            -> painted with door.color
    walls = wall stripe AND NOT door span, plus the four borders -> painted with wall.color
    [target dot, only when rendering.render_target]
    agent dot

and each "dot" is **not a disc**. It is a normalised Gaussian alpha mask composited over the whole
image:

    dot   = exp(-((x-px)^2 + (y-py)^2) / (2 r^2));  dot /= dot.max()
    out   = img.float() * (1 - dot) + colour * dot;  img = out.to(uint8)

Five things were measured against the reference rather than read off it, and each of them changes
pixels:

  * **The Gaussian dot has no flat interior and a very large tail.** ``.to(torch.uint8)`` truncates,
    so a channel of 255 becomes 254 wherever ``1 - a`` is representably below 1.0 in float32 — i.e.
    wherever ``a > 2^-24`` — and that is a radius of **41.2 px** at the nominal radius of 7. Measured
    on the official frame with the agent at (60, 112): **4,921 pixels are touched** where a flat disc
    of radius 7 would cover 149, **3,304 of them are the single value 254**, and the green channel
    takes **102 distinct values** across the ramp. An implementation that draws a disc, or that
    rounds instead of truncating, is wrong over a 41-pixel radius while looking correct.
  * **The peak pixel is exactly the requested colour.** ``dot.max()`` normalises the mask, so the
    argmax pixel has ``a == 1.0`` exactly and ``out = 255*(1-1) + c*1 = c``. Measured: agent at
    (60, 112) puts ``(255, 0, 0)`` at row 112, column 60 — exactly, not within a rounding. That is
    what makes an RGB knob checkable at all here; it is a consequence of the normalisation rather
    than of the Gaussian, and it is why ``check_colors_apply_exactly`` can pass on an env whose
    every other pixel is a blend.
  * **``wall.border_color`` is declared by the official env and never read.** ``_wall_and_door_masks``
    folds the four borders into ``wall_mask``, and ``_render_frame`` paints ``wall_mask`` with
    ``wall.color``. Measured on the reference by setting it to (255,255,255) and to (0,0,0) and
    diffing the two frames: **max abs diff 0/255**. It is therefore NOT declared here; declaring it
    would advertise a knob that does nothing. Anything that ports the variation space by copying its
    leaf list inherits the dead knob.
  * **Doors are painted first and walls over them**, so a door whose span reaches a border is cut by
    the border rather than the other way round. The two masks are disjoint on the central stripe, so
    this is only visible at the borders — which is exactly where a wrong order hides.
  * **The borders are 4 px wide and sit INSIDE the 14 px margin**, at rows/columns 10..13 and
    210..213, with plain background outside them. Measured by reading column x=5 off the reference
    frame: the non-background rows are exactly ``[10, 11, 12, 13, 210, 211, 212, 213]``.
    ``BORDER_SIZE = 14`` is the margin, not the stroke; the stroke is the undocumented ``t = 4``
    local, and taking ``BORDER_SIZE`` for the stroke draws a 14 px frame in the wrong place.

``Piecewise`` is the same renderer without the central wall, plus an optional zone tint. Two of its
declared knobs are deliberately absent here and the reason is recorded rather than silently dropped:

  * ``rendering.render_bias_field`` rasterises a **matplotlib quiver** (anti-aliased arrowheads,
    alpha 0.85) and composites it. That is a second rasterizer with its own coverage rules inside an
    otherwise arithmetic renderer; it is off by default and it is not reproduced. An env that wants
    it should render the official frame.
  * ``zones.bias_*``, ``agent.speed`` and ``task.min_steps`` are DYNAMICS, not appearance — they
    change no pixel. They are carried as plain attributes (``env.speed``, ``env.zone_bias``) so
    ``step`` can use them, and are not declared as variations, because a knob in the variation space
    that changes nothing advertises support that does not exist.

The same reasoning splits each official env into two registered classes, exactly as MiniGrid splits
top-down from POV against one official id. ``rendering.render_target`` and ``rendering.render_zones``
both default to **0**, so under the official default the target and the zone tints are not drawn and
every knob that describes them would be inert. Rather than default them on (which would make "our
defaults are the official appearance" false) or declare dead knobs, each configuration is its own
class whose defaults reproduce the official env *in that configuration*:

    TwoRoom          swm/TwoRoom-v1                          (render_target=0, the shipped default)
    TwoRoomTarget    swm/TwoRoom-v1  render_target=True      (target dot drawn; state carries it)
    Piecewise        swm/Piecewise-v0                        (render_zones=0, the shipped default)
    PiecewiseZones   swm/Piecewise-v0 render_zones=1         (zone tints drawn)

Output size
-----------

The reference has exactly one resolution. Every length in it is in output pixels — a 4 px border, a
door half-extent in pixels, a Gaussian sigma in pixels, a wall centre at column 112 — and there is no
world-unit description underneath to re-sample at another rate. Rendering at 448 would not be the
same picture more finely sampled; it would be a different picture with a 4 px border on a 448 px
canvas. So ``render(size=...)`` refuses anything but 224 instead of quietly inventing a scaling rule
the reference does not have.

Dynamics
--------

All four ship ``step``, and unlike PushT this one CAN be exact: the transition is a clamp, a
comparison and an add in float32, with no contact solver and nothing chaotic. Measured against the
official CPU env over 16 episodes x 100 steps from reference-produced start states: **exact_frac
1.000, max error 0.0** for all four. Where the reference computes a clamp bound in Python floats
(``float(pos[0])``, ``bs + agent_r``) the bound is computed here in float64 and cast once, so the
two agree bit-for-bit rather than to a tolerance.

Measured parity and throughput
------------------------------

H200, 64 states per env from ``sample_states(rng=default_rng(3))``, default appearance, 224px, against
the official CPU renderer (``scratch/swm_nav_parity.py``). GPU frames/s at batch 512 rendering into a
caller-owned buffer; the official column is the same env driven one frame at a time, which is the only
way it runs.

    env              interior exact   all-pixel exact   max abs   GPU f/s   official f/s   speedup
    TwoRoom          100.000%         99.9981%  (61 px)      1     116,641      517.2       226x
    TwoRoomTarget    100.000%         99.9946% (172 px)      1      76,553      396.1       193x
    Piecewise        100.000%         99.9399% (193 px)      1     187,645      659.0       285x
    PiecewiseZones   100.000%         99.9994%  (18 px)      1     140,727      434.9       324x

Every remaining disagreement is a single unit on one channel at a place where the alpha ramp crosses
a truncation boundary. The count is what a one-ulp ``exp`` difference predicts, and it is 40x smaller
than the same port with ``exp`` in float32 (see ``_gaussian_dot``).

The interior column above is ONE DRAW (``default_rng(3)``) and rounds to 100% on it. Over 200 draws
per env it is 100% on most and 99.99991% on about one in eight; the next section is why, and it is the
reason ``parity_interior_min`` is declared at 0.999995 rather than left at 1.0.

Why the interior floor is 0.999995 and not 1.0
----------------------------------------------

The sentence that used to stand here — "every one of them is at a NON-interior pixel" — was true of
the four draws it was measured on and false in general. Over 200 seeded draws of 64 states per env,
one or two of them land on a pixel the interior mask admits: TwoRoom 9/200, TwoRoomTarget 16/200,
Piecewise 18/200, PiecewiseZones 5/200, never more than **2 pixels out of ~2.3 million interior
pixels**, never more than 1/255, always on ONE channel. The parity test drew its states unseeded, so
this was a 1-in-5 chance of a red suite on a different env each run, and seeding it to a clean draw
would have hidden a real disagreement rather than explained it.

The mechanism, measured by decomposition (``scratch/audit/swm_mechanism3.py``). Our renderer and the
reference differ in exactly two respects — the DEVICE and the PRECISION of the transcendental — so
build the frame stack with one factor changed at a time:

    ours on CUDA, exp in float64   vs   ours on CPU, exp in float64      0 pixels differ
    ours on CPU,  exp in float32   vs   the OFFICIAL renderer            0 pixels differ
    ours on CPU,  exp in float64   vs   ours on CPU, exp in float32      48-476 pixels differ

So the port is arithmetically exact: run on the CPU with the reference's own float32 ``exp``, it
reproduces the official frame **bit for bit**, and the device contributes nothing. One hundred percent
of the residual is the transcendental, and it is one ulp:

  * torch's CPU float32 ``exp`` is faithful to within an ulp, not correctly rounded. Ours is the
    correctly rounded float32 value (``exp`` in float64, rounded once), which is what makes the port
    40x closer overall — and on a small fraction of arguments the two differ by 1 ulp.
  * That ulp decides a whole pixel **only on the channel where the dot's colour equals the background**.
    There the composite is ``255*(1-a) + 255*a``, which is the identity in exact arithmetic and is not
    in float32: for ``a`` in the far tail the two products straddle 255, and whether the sum lands ON
    255 or one ulp below decides whether ``.to(uint8)`` truncates to 254. Measured at a disagreeing
    pixel: ``a = 0x384bebeb`` (ours) against ``0x384bebec`` (the reference) gives 254 against 255,
    with the other two channels bit-identical.
  * The interior mask cannot exclude these and is not wrong to admit them. Its premise — a flat 3x3
    neighbourhood in the reference means there is no rounding ambiguity — holds for a rasterizer of
    flat-coloured primitives and fails for a renderer whose output is a smooth per-pixel function: the
    Gaussian tail is so flat out at radius 25-31 px (where every disagreeing pixel measured sits, from
    ``1 - a`` = 0.9986 to 0.99995) that the reference is constant over a 5x5 or 7x7 patch while the
    underlying float is still moving. Measured, every disagreeing interior pixel sits at the centre of
    a patch the reference holds constant for 2-3 px in every direction.
  * The direction is one-sided for the same reason: the reference's own sporadic 254s break its 3x3
    flatness, so the mask admits "we say 254 where the reference says 255" and screens out the
    opposite case. Every interior disagreement observed is that way round.

Closing it would mean reproducing the host's vectorised ``expf`` bit for bit inside a GPU kernel —
a result that is not even fixed across CPUs, since it is whatever libm the interpreter was built
against returns. Using float32 ``exp`` on the GPU instead is measurably worse (2,561 pixels against
61 on TwoRoom; see ``_gaussian_dot``), because CUDA's ``expf`` and the CPU's are different functions.
So the floor is declared, with this note, at 0.999995 — five times the worst draw seen in 800, and
about 11 pixels of 2.3 million.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..core.env import StableEnv, register
from ..render import navdot
from ..spaces import Box, Dict, Discrete, RGBBox

#: The official canvas. Every constant below is in these pixels; see "Output size" above.
IMG = 224
#: ``BORDER_SIZE``: the margin the playfield is inset by. NOT the stroke width.
BORDER = 14
#: the undocumented ``t = 4`` local in ``_wall_and_door_masks`` / ``_border_mask``
BORDER_T = 4
WALL_CENTER = 112
MAX_DOOR = 3
#: ``door_margin`` in ``_apply_collisions`` — how far past a door's half-extent still counts as inside
DOOR_MARGIN = 1.75
#: the pushback the reference applies when it clamps the agent against the central wall
WALL_PUSHBACK = 0.5
GRID_N = 2                       # PiecewiseEnv's default grid_n
N_ZONES = GRID_N * GRID_N

AGENT_C = (255, 0, 0)
TARGET_C = (0, 255, 0)
WALL_C = (0, 0, 0)
DOOR_C = (255, 255, 255)
BG_C = (255, 255, 255)
BORDER_C = (0, 0, 0)
AGENT_R = 7.0
TARGET_R = 7.0
AGENT_SPEED = 5.0
WALL_THICKNESS = 10
WALL_AXIS = 1                    # 1 = vertical wall, 0 = horizontal
N_DOORS = 1
DOOR_POS = (49.0, 49.0, 49.0)
DOOR_SIZE = (14.0, 14.0, 14.0)
#: ``colorsys.hsv_to_rgb(i/4, 0.25, 1.0)`` scaled by 255 and TRUNCATED by ``int()``. Two of the four
#: are NOT the obvious pastel: zones 1 and 3 sit on an HSV sector boundary where ``f = 0.5``, so their
#: third channel is ``1 - 0.25*0.5 = 0.875 -> 223`` rather than ``1 - 0.25 = 0.75 -> 191``. Guessing
#: 191 for all of them costs 32/255 on the red channel of half the playfield, which is exactly what
#: the first parity run measured (exact_frac 0.617, max_abs 32) before these were read off the
#: reference's own variation space.
ZONE_C = ((255, 191, 191), (223, 255, 191), (191, 255, 255), (223, 191, 255))
#: ``2.0 * cos/sin(2 pi i / 4)`` — the per-zone motion bias. Dynamics only; never drawn.
ZONE_BIAS = ((2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (-0.0, -2.0))

#: agent.position / target.position bounds: ``[BORDER, IMG - BORDER - 1]``
POS_LO = float(BORDER)
POS_HI = float(IMG - BORDER - 1)
#: The batch is chunked inside ``render`` so peak float memory is bounded by the chunk rather than by
#: the batch: one float32 (chunk, 224, 224, 3) temporary is 600 kB per environment, so an unchunked
#: batch of 16k would ask for ~10 GB of temporaries for a 2.4 GB frame stack. Chunking cannot change
#: the answer because environments do not interact, and ``check_batch_size_invariance`` asserts that.
RENDER_CHUNK = 256


def _grids(device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(H,W)`` pixel-centre coordinates, exactly as ``torch.meshgrid(arange, arange, 'ij')``."""
    a = torch.arange(IMG, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(a, a, indexing="ij")
    return gy, gx


def _border_mask(device) -> torch.Tensor:
    """``(H,W)`` bool: the four 4 px strokes at rows/cols 10..13 and 210..213."""
    m = torch.zeros(IMG, IMG, dtype=torch.bool, device=device)
    b, t = BORDER, BORDER_T
    m[:, b - t:b] = True
    m[:, IMG - b:IMG - b + t] = True
    m[b - t:b, :] = True
    m[IMG - b:IMG - b + t, :] = True
    return m


def _zone_map(device) -> torch.Tensor:
    """``(H,W)`` long: PiecewiseEnv's zone index per pixel, -1 outside the playfield.

    A constant of the family: it depends on ``grid_n`` and ``BORDER_SIZE`` only, never on the state
    or on any variation, so it is built once and the variation only chooses the colours it indexes.
    """
    z = torch.full((IMG, IMG), -1, dtype=torch.long, device=device)
    play = IMG - 2 * BORDER
    for row in range(GRID_N):
        for col in range(GRID_N):
            y0 = BORDER + int(row * play / GRID_N)
            y1 = BORDER + int((row + 1) * play / GRID_N)
            x0 = BORDER + int(col * play / GRID_N)
            x1 = BORDER + int((col + 1) * play / GRID_N)
            z[y0:y1, x0:x1] = row * GRID_N + col
    return z


def _u(env):
    """The raw env behind whatever wrapper stack ``gymnasium.make`` put on it.

    ``conformance.check_step_matches_reference`` hands ``reference_sim_state`` the WRAPPED env while
    ``reference_render`` here has the unwrapped one; relying on gymnasium's attribute forwarding to
    paper over that works today and is deprecated, so the unwrapping is explicit and both call sites
    pass through the same function.
    """
    return getattr(env, "unwrapped", env)


def _trunc_color(c: torch.Tensor) -> torch.Tensor:
    """``int(value)`` per channel — the reference stores colours as uint8 and fills with ``int()``.

    Truncation rather than rounding, because that is what ``int()`` and ``float -> uint8`` both do,
    and because a style row is float32: a sampled 191.7 must paint 191 the way the official env's
    uint8 191 would, not 192.
    """
    return c.trunc()


def _gaussian_dot(gy, gx, pos, radius) -> torch.Tensor:
    """``(B,H,W)`` normalised Gaussian alpha, matching ``_gaussian_dot`` operation for operation.

    ``2.0 * std * std`` is evaluated in Python float64 in the reference and only then divides a
    float32 tensor, so it is computed in float64 here as well and cast once — doing it in float32
    would round the divisor differently and move the truncation boundary of the tail.

    **``exp`` is evaluated in float64 and rounded once.** That is not extra precision for its own
    sake; it is what makes the port bit-exact, and it was established by measurement rather than
    assumed. CUDA's float32 ``expf`` and the CPU's differ by up to one ulp, the alpha then multiplies
    a background of 255, and ``.to(uint8)`` truncates — so a one-ulp difference in ``exp`` flips a
    whole pixel wherever ``255*(1-a)`` sits within an ulp of an integer. Measured over 64 states per
    env against the official renderer (``scratch/swm_nav_exp_probe.py``):

        disagreeing pixels (and interior-exact fraction), 64 states per env:

        variant                       TwoRoom          TwoRoomTarget      Piecewise
        exp in float32                2,561  (1.0)     4,993 (0.9999982)  3,001 (0.9999988)
        exp in float64, round once       61  (1.0)       172 (1.0)          193 (1.0)
        dist2 and exp in float64      3,246 (0.9999976) 5,913 (0.9999969)  3,425 (0.9999980)

    The middle row is the answer and the third row is why: widening the *argument* as well makes it
    worse, because the reference's argument really is float32 and a "more accurate" input is a
    different input. Only the transcendental is wrong in float32, and rounding it once from float64
    gives the correctly rounded float32 value that the CPU reference also produces.
    """
    dx = gx.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
    dy = gy.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
    dist2 = dx * dx + dy * dy
    std = radius.double().clamp_min(1e-6)
    denom = (2.0 * std * std).to(torch.float32).view(-1, 1, 1)
    dot = torch.exp((-dist2 / denom).double()).to(torch.float32)
    m = dot.amax(dim=(1, 2), keepdim=True)
    return torch.where(m > 0, dot / m, dot)


#: WHY BOTH OF THIS MODULE'S STEP 6 OPTIMISATIONS ARE CPU-ONLY.
#:
#: ``_base_image`` keys its bake on the bytes of a style row and ``_blend_dot`` turns per-row windows
#: into groups, and BOTH decide on the host from values that live on the device: one ``.cpu()`` for
#: the key, one plus four scalar reductions for the windows, doubled for TwoRoomTarget's two dots.
#: On a CPU those are copies. On a GPU they are synchronisations in the middle of a render, and they
#: cost more than the work they save at every batch small enough that the saved work is small --
#: measured on a 3090 (job 6349274, gpu2113), before against after:
#:
#:     batch                1      16      64     256    1024    4096
#:     TwoRoom           1.25    0.48    0.55    1.41    1.40    1.41
#:     TwoRoomTarget     0.86    0.32    0.40    1.11    1.13    1.13
#:     Piecewise         0.58    0.19    0.35    0.99    0.98    0.98
#:     PiecewiseZones    0.85    0.27    0.43    1.14    1.14    1.14
#:
#: A 5.3x regression at batch 16 that a best-batch summary does not show is still a regression, and
#: section 7 names host/device synchronisation in a hot loop as a structural defect in its own right
#: rather than a cost to be netted off against a gain somewhere else. ``raster2d._row_groups``'s cost
#: model does not help here either: its two constants were measured on one CPU core, so on cuda it
#: accepts groupings that lose.
#:
#: So on ``device.type == "cuda"`` both paths run the arithmetic they replaced, unchanged, and the
#: GPU rows are identical code rather than an argument about which batches the qualifiers sample.
#: Recovering the large-batch gain there is a separate piece of work with its own measurements: it
#: needs the decisions made on the device (a device-side key, a device-side partition) or the cost
#: model re-calibrated on the card, and neither is this ticket.
_CUDA_IS_NOT_THE_CPU = True

#: The smallest alpha ``_alpha_blend`` can act on, MEASURED against that function over every uint8
#: contrast by ``scratch/step6/raster2d/alpha_bound.py``: 2.98e-8 = 2^-25, reached at img=1,
#: colour=0. The two directions are not symmetric and the asymmetry is the whole point. Painting a
#: bright colour onto a dark pixel needs ``a * |img - colour| >= 1`` before the truncation sees it,
#: i.e. ``a >= 1/255``. Darkening a bright one goes through ``img * (1 - a)``, and ``1 - a`` rounds
#: to exactly 1.0 in float32 for every ``a`` below the spacing at 1.0 -- so the threshold there is a
#: property of float32 and is four orders of magnitude smaller. A bound has to hold for the pixel
#: that is easiest to move, so this is the smaller of the two.
#:
#: The module docstring used to say the tail was effectively unbounded and that a Gaussian dot could
#: not be cropped. It can: below this alpha the blend is the identity, and a normalised Gaussian is
#: below it beyond 5.89 sigma.
_ALPHA_FLOOR = 2.0 ** -25

#: ``d/sigma`` at which the normalised dot drops under ``_ALPHA_FLOOR``, for a peak sitting exactly
#: on a sample. ``exp(-d^2 / 2 s^2) < T`` gives ``d > s * sqrt(2 ln(1/T))``.
_DOT_SIGMAS = math.sqrt(2.0 * math.log(1.0 / _ALPHA_FLOOR))


def _dot_window(pos, sigma):
    """``(k,4)`` int64 pixel windows outside which a normalised Gaussian dot cannot move a pixel.

    ``_gaussian_dot`` divides by the frame's own maximum, so the profile that reaches
    ``_alpha_blend`` is ``exp(-(d^2 - m^2) / 2 s^2)`` where ``m`` is the distance from the centre to
    the largest sample IN THE FRAME -- not to the centre itself. Two consequences, and the second is
    the one that is easy to get wrong:

      * for a centre inside the frame, ``m`` is at most half a pixel in each axis, so the normalised
        tail is the un-normalised one widened by a fraction of a sigma;
      * for a centre OUTSIDE it -- which ``render(states)`` permits, since a caller may pass any
        state and only ``step`` clamps -- ``m`` is the distance to the nearest in-frame sample and
        can be large. The normalisation then rescales the whole dot so that its in-frame maximum is
        1, and a bound that ignored that would crop a dot that is painting at full strength. So
        ``m`` is measured rather than assumed, and the window grows with it.

    The window always contains the in-frame argmax, which is what lets the caller take the maximum
    over the window and get the frame's maximum: the Gaussian decreases with distance from the
    centre, so its largest in-frame sample is the in-frame sample nearest the centre, and that
    sample is within ``m <= d`` of the centre by construction.
    """
    x, y = pos[:, 0], pos[:, 1]
    # the nearest in-frame SAMPLE; _grids puts samples on integers 0..IMG-1
    mx = x.round().clamp(0, IMG - 1)
    my = y.round().clamp(0, IMG - 1)
    m2 = (x - mx) ** 2 + (y - my) ** 2
    # Plus a pixel. The radius is exact for exact arithmetic and the reviewer's 108,000-state
    # boundary scan found no failure without it, but every term in it -- the sqrt, the log inside
    # _DOT_SIGMAS, the squares -- is float32 or float64 rounding away from the pixel it decides, and
    # the bound has no slack of its own. One pixel of window costs a row of samples and buys the
    # whole question.
    d = torch.sqrt((sigma.reshape(-1) * _DOT_SIGMAS) ** 2 + m2) + 1.0
    c0 = torch.ceil(x - d).clamp(0, IMG)
    c1 = torch.floor(x + d).clamp(0, IMG - 1) + 1
    r0 = torch.ceil(y - d).clamp(0, IMG)
    r1 = torch.floor(y + d).clamp(0, IMG - 1) + 1
    return torch.stack([r0, r1, c0, c1], dim=1).to(torch.int64)


def _alpha_blend(img_u8: torch.Tensor, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """``uint8 -> float32 -> img*(1-a) + c*a -> uint8``, exactly as the reference composites.

    Returning **uint8** is load-bearing and was the one thing this port initially got wrong. The
    reference's ``_alpha_blend`` ends in ``out.to(torch.uint8)``, and ``TwoRoomEnv._render_frame``
    calls it twice — target, then agent — so the target dot is QUANTISED before the agent dot is
    composited over it. Chaining the two blends in float32 and quantising once at the end is the
    obvious implementation and it looks identical by eye. Measured on TwoRoomTarget, 64 states,
    against the official frames: quantising between the blends leaves **172** disagreeing pixels,
    chaining in float32 leaves **35,957** — a 209x difference, all of it 1/255, none of it visible.
    """
    a = alpha.clamp(0.0, 1.0).unsqueeze(-1)
    out = img_u8.to(torch.float32)
    out = out * (1.0 - a) + _trunc_color(rgb).view(-1, 1, 1, 3) * a
    return out.to(torch.uint8)


class _SWMNavBase(StableEnv):
    """Shared machinery for the two torch-rendered swm navigation envs.

    The split between this and the subclasses follows the reference: everything up to and including
    the base image (background, structure, borders) is env-specific, and everything from the first
    alpha blend onwards is identical, because the official ``_render_frame`` bodies are identical
    from that point on.
    """

    default_size = IMG
    #: the reference has one resolution; see the module docstring
    fixed_size = IMG
    #: Interior exactness is 1.0 on most draws and 0.9999991 on about one in eight. The mechanism is
    #: named in the module docstring ("Why the interior floor is 0.999995 and not 1.0") and it is a
    #: one-ulp difference between the reference's float32 ``expf`` and the correctly rounded value,
    #: which flips a pixel only on the channel where the dot's colour equals the background. 0.999995
    #: is ~11 pixels of the ~2.3 million interior pixels in a 64-state draw; the worst of 800 seeded
    #: draws was 2.
    parity_interior_min = 0.999995
    parity_note = (
        "1 ulp of exp, not geometry. Run on the CPU with the reference's own float32 exp this "
        "renderer reproduces the official frame BIT FOR BIT and the device contributes nothing "
        "(measured both ways); the residual is entirely that our exp is correctly rounded and "
        "torch's CPU float32 exp is only faithful. That ulp changes a pixel only where the dot's "
        "colour equals the background, because 255*(1-a)+255*a is the identity in exact arithmetic "
        "and not in float32, and the 41 px Gaussian tail is flat enough that the reference's own 3x3 "
        "neighbourhood cannot see the quantisation contour. Worst of 800 seeded draws: 2 pixels of "
        "2.3 million, 1/255, one channel. See the module docstring.")
    action_dim = 2
    action_low = -1.0
    action_high = 1.0
    step_reference_checkable = True
    step_reference_exact = True
    step_reference_control = "agent"
    step_reference_gate = "agent"
    step_reference_hold = 8
    step_reference_groups = (("agent", (0, 1), "euclidean"),)

    #: DYNAMICS, not appearance: these change no pixel, so they are attributes rather than variation
    #: leaves. ``check_variations_do_something`` would (correctly) call them inert.
    speed = AGENT_SPEED

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gy, self._gx = _grids(self.device)
        self._border = _border_mask(self.device)

    # ---------------------------------------------------------------- appearance

    def _agent_space(self) -> dict:
        return {"color": RGBBox(init_value=AGENT_C),
                "radius": Box(low=7.0, high=14.0, shape=(1,), init_value=[AGENT_R])}

    def base_image(self, sty, n: int) -> torch.Tensor:
        """``(n,H,W,3)`` **uint8**: everything drawn before the first alpha blend.

        uint8 rather than float32 because that is the reference's own intermediate: it fills and
        masks a uint8 buffer and only widens it inside ``_alpha_blend``. Keeping the same dtype keeps
        the same truncation points and costs a quarter of the memory.
        """
        raise NotImplementedError

    def scene(self, states, variation=None, style=None):
        raise NotImplementedError(
            f"{type(self).__name__} is not drawn from vector primitives. The official renderer "
            f"composites a normalised Gaussian alpha mask over the whole frame, which is a per-pixel "
            f"function rather than a set of flat-coloured shapes; a Scene2D disc would reproduce the "
            f"7-pixel core and none of the 40-pixel tail. It overrides render() instead.")

    # ---------------------------------------------------------------- rendering

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        n = int(s.shape[0])
        t = int(size or self.size)
        if t != self.fixed_size:
            raise ValueError(
                f"{type(self).__name__} renders at {self.fixed_size}px only, and asking for {t} is not "
                f"a resolution change. Every length in the reference is in output pixels — a 4 px "
                f"border, a door half-extent in pixels, a Gaussian sigma in pixels, a wall at column "
                f"112 — with no world-unit description underneath, so there is nothing to re-sample. "
                f"Resize the frames afterwards if a different size is wanted; that is what "
                f"stable_worldmodel's own pipeline does.")
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}.render: supersample={supersample} has nothing to estimate. "
                f"Supersampling exists to approximate sub-pixel coverage of analytic geometry; this "
                f"frame is an exact per-pixel formula evaluated on the reference's own lattice.")
        sty = self.resolve_style(style, variation, batch=n)

        if out is not None:
            if not out.is_contiguous():
                raise ValueError(
                    f"{type(self).__name__}.render(out=): the destination must be contiguous. A "
                    f"strided view (frames[:, t] in a rollout buffer) would be written linearly from "
                    f"its own offset and the frames would land across neighbouring slices.")
            img = out.view(n, IMG, IMG, 3)
        else:
            img = torch.empty((n, IMG, IMG, 3), dtype=torch.uint8, device=self.device)

        for lo in range(0, n, RENDER_CHUNK):
            img[lo:min(lo + RENDER_CHUNK, n)] = self._render_chunk(
                s, sty, lo, min(lo + RENDER_CHUNK, n))
        return img.view(*lead, IMG, IMG, 3) if lead else img

    def _render_chunk(self, s, sty, lo: int, hi: int) -> torch.Tensor:
        k = hi - lo
        sl = lambda p: sty[p][lo:hi]
        base = self._base_image(sty, lo, hi)
        if navdot.available(self.device):
            pos, sig, col = self._dots(s[lo:hi], sl, k)
            return navdot.blend_dots(base.expand(k, IMG, IMG, 3).contiguous(), pos, sig, col)
        return self.blend_dots(base, s[lo:hi], sl, k)

    def _dots(self, s, sl, k):
        """``(k,ND,2)`` centres, ``(k,ND)`` radii, ``(k,ND,3)`` colours, IN COMPOSITING ORDER.

        The order is the contract: the reference blends the target first and the agent over it, and
        each blend quantises to uint8, so two dots swapped are a different picture where they
        overlap. One method per family rather than a flag, so a subclass that adds a dot declares it
        in the same place it declares how to draw it.
        """
        return (s[:, 0:2].reshape(k, 1, 2), sl("agent.radius").reshape(k, 1),
                sl("agent.color").reshape(k, 1, 3))

    #: Baked base images, keyed on the bytes of the style row that produced them. Bounded, and
    #: cleared wholesale rather than evicted one at a time — the same shape, and for the same
    #: reasons, as ``render/raster2d.py``'s ``_BASE_CACHE``.
    _BAKE_CACHE: dict = {}
    _BAKE_CACHE_MAX = 64

    def _base_image(self, sty, lo: int, hi: int) -> torch.Tensor:
        """``base_image`` for rows ``[lo, hi)``, built once per DISTINCT style row rather than per row.

        The base image — background fill, wall stripe, doors, zone tints, border — is a pure function
        of the style and of constants of the family. It is not a function of the state: the agent and
        the target arrive later, through ``blend_dots``. Measured on Oscar at one thread and batch 64
        it was 49% of TwoRoom's render, 32% of TwoRoomTarget's and 23% of Piecewise's, rebuilt from
        scratch on every call and materialised at ``(k,224,224,3)`` even when every row of the batch
        asked for the same picture. That is section 7's "repeated construction of static scene data",
        and the batch-uniform case — which is every render at the default appearance, so every frozen
        corpus and every benchmark row — is the one that pays for it k times over.

        So the rows are grouped by the bytes of their style, each distinct row is built ONCE through
        the subclass's own ``base_image``, and the result is expanded or gathered back to ``k``.

        **The key is the whole style row, not the leaves this env happens to read.** A key listing
        the relevant leaves would be smaller and would go stale the day somebody adds a leaf that
        feeds the base image and forgets to declare it — a wrong key is a stale picture, which is the
        one failure a cache must not have. Keying on everything can only cost a miss: two rows that
        differ in ``agent.color`` alone are baked twice and draw the same thing.

        Render purity holds because the key IS the content. Two calls with the same style get the
        same bytes and therefore the same image no matter what ran in between, and a call with a
        different style cannot reach the first one's entry. Batch and order invariance hold because a
        row's group is decided by its own bytes: a permuted batch yields the same groups and the same
        images, gathered back in the caller's order.

        The expanded result is a read-only view of the cached tensor by construction — an in-place
        write to an expanded tensor raises — and the only consumer, ``_alpha_blend``, widens it to
        float32 first, which copies.
        """
        k = hi - lo
        if self.device.type == "cuda":
            return self.base_image(lambda p: sty[p][lo:hi], k)          # see _CUDA_IS_NOT_THE_CPU
        rows = sty.values[lo:hi]
        # ONE host transfer, the same trick ``raster2d._static_base`` uses: the bytes are needed for
        # the key anyway, so the grouping comes free with them. On the CPU the transfer is a copy; on
        # a GPU it is a SYNCHRONISATION, which is why the branch above exists.
        blob = rows.detach().cpu().contiguous().numpy().tobytes()
        w = blob and len(blob) // k or 0
        keys = [blob[i * w:(i + 1) * w] for i in range(k)] if w else [b""] * k
        order: dict = {}
        for i, key in enumerate(keys):
            order.setdefault(key, []).append(i)
        if len(order) == k and k > 1:
            # every row is different: grouping buys nothing and the gather would cost something
            return self.base_image(lambda p: sty[p][lo:hi], k)

        cls = type(self).__name__
        plates = []
        for key, idx in order.items():
            ck = (cls, int(self.fixed_size), str(self.device), key)
            hit = _SWMNavBase._BAKE_CACHE.get(ck)
            if hit is None:
                first = lo + idx[0]
                hit = self.base_image(lambda p, a=first: sty[p][a:a + 1], 1)
                if len(_SWMNavBase._BAKE_CACHE) >= self._BAKE_CACHE_MAX:
                    _SWMNavBase._BAKE_CACHE.clear()
                _SWMNavBase._BAKE_CACHE[ck] = hit
            plates.append(hit)
        if len(plates) == 1:
            return plates[0].expand(k, IMG, IMG, 3)
        pick = torch.empty(k, dtype=torch.long, device=self.device)
        for g, (_, idx) in enumerate(order.items()):
            pick[torch.as_tensor(idx, device=self.device)] = g
        return torch.cat(plates, dim=0)[pick]

    def blend_dots(self, img, s, sl, k) -> torch.Tensor:
        """The alpha compositing half, identical in both official envs. Agent last, over everything."""
        return self._blend_dot(img, s[:, 0:2], sl("agent.radius"), sl("agent.color"))

    def _blend_dot(self, img, pos, sigma, colour) -> torch.Tensor:
        """``_alpha_blend(img, _gaussian_dot(...), colour)``, evaluated only where it can act.

        Same arithmetic, same lattice, fewer pixels. ``_dot_window`` gives the region outside which
        the normalised dot is below ``_ALPHA_FLOOR``, and below that alpha the blend is the identity,
        so every pixel outside the window keeps the value it already had — which is what ``img``
        already holds. The window is a BOUND, derived from the reference's own arithmetic and
        measured against it, so this cannot change a picture; ``tests/unit`` and the frozen replay
        hold that rather than leaving it as an argument.

        The batch is grouped by ``raster2d._row_groups``, the same partition the 2-D rasterizer uses
        and for the same reason: one window over a batch of scattered agents is the whole frame
        again, and quantising each row's window ORIGIN onto a grid of the window size gives a
        partition whose size is bounded by the frame rather than by the batch. Sharing that function
        means the repo has one grouping rule with one set of measured constants, rather than a second
        one that nothing calibrated.

        The maximum the normalisation divides by is taken over the WINDOW and is nonetheless the
        frame's maximum: a Gaussian decreases with distance from its centre, so its largest in-frame
        sample is the in-frame sample nearest the centre, and ``_dot_window`` always contains that
        sample. That is the one property of this crop that is not obvious, and it is why the window
        has to be built around the centre rather than around the frame.
        """
        from ..render.raster2d import _row_groups

        k = int(pos.shape[0])
        if self.device.type == "cuda":
            return _alpha_blend(img, _gaussian_dot(self._gy, self._gx, pos, sigma), colour)
        win = _dot_window(pos, sigma)
        wins = _row_groups(win.cpu().numpy(), *(int(v) for v in
                           (win[:, 0].min(), win[:, 1].max(), win[:, 2].min(), win[:, 3].max())))
        if len(wins) == 1 and wins[0][0] is None:
            r0, r1, c0, c1 = wins[0][1:]
            if (r1 - r0) * (c1 - c0) >= IMG * IMG:
                return _alpha_blend(img, _gaussian_dot(self._gy, self._gx, pos, sigma), colour)
        # ``clone``, never ``contiguous``. ``img`` may BE a cached base plate -- at k == 1 an
        # ``expand`` to its own shape returns the tensor itself and ``contiguous()`` on something
        # already contiguous returns it again, so the loop below would paint the agent into the
        # cache and every later render of that style would come back with a stale dot on it. Caught
        # by the order-invariance and purity checks in scratch/step6/raster2d/cpu_digest.py, which
        # is what they are for; ``clone`` copies unconditionally.
        out = img.expand(k, IMG, IMG, 3).clone()
        for rows, r0, r1, c0, c1 in wins:
            if r1 <= r0 or c1 <= c0:
                continue
            gy, gx = self._gy[r0:r1, c0:c1], self._gx[r0:r1, c0:c1]
            if rows is None:
                dot = _gaussian_dot(gy, gx, pos, sigma)
                out[:, r0:r1, c0:c1] = _alpha_blend(out[:, r0:r1, c0:c1], dot, colour)
            else:
                idx = torch.as_tensor(rows, device=out.device)
                dot = _gaussian_dot(gy, gx, pos[idx], sigma.reshape(-1)[idx])
                out[idx, r0:r1, c0:c1] = _alpha_blend(out[idx, r0:r1, c0:c1], dot, colour[idx])
        return out

    def _fill(self, k, color) -> torch.Tensor:
        """``(k,H,W,3)`` uint8 flat fill — the reference's ``img[c].fill_(int(color[c]))``."""
        c = _trunc_color(color).to(torch.uint8).view(k, 1, 1, 3)
        return c.expand(k, IMG, IMG, 3).contiguous()

    def _paint(self, img, mask, color) -> torch.Tensor:
        """Paint ``color`` where ``mask`` (``(k,H,W)`` bool or ``(H,W)`` bool) is set."""
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        c = _trunc_color(color).to(torch.uint8).view(-1, 1, 1, 3)
        return torch.where(m.unsqueeze(-1), c, img)

    # ---------------------------------------------------------------- reference bridge

    #: kwargs the reference env is constructed with; the subclass's whole reason for existing
    reference_kwargs: dict = {}

    @classmethod
    def reference_env(cls, size: int = IMG):
        import gymnasium
        import stable_worldmodel  # noqa: F401  (registers the swm/* ids)
        return gymnasium.make(cls.reference_id, **cls.reference_kwargs)

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        u.agent_position = torch.as_tensor(np.asarray(st, np.float32)[:2], dtype=torch.float32)

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for the given states.

        ``options={'variation': []}`` on the reset is load-bearing: the default is
        ``('agent.position', 'target.position')``, so a plain ``reset(seed=)`` RESAMPLES those leaves
        and the reference would be describing a different appearance from ours on every call. With an
        empty list the variation space is reset to exactly its init values, which is what our
        ``default_style()`` is defined to be.
        """
        u = env.unwrapped
        env.reset(seed=0, options={"variation": []})
        out = []
        for st in np.asarray(states):
            cls.reference_set_state(u, st)
            got = cls.reference_sim_state(u)
            err = float(np.abs(np.asarray(got, np.float64)
                               - np.asarray(st, np.float64)).max())
            if err > 1e-6:
                raise AssertionError(
                    f"the reference env did not reach the requested state (max err {err:.3e}); asked "
                    f"{np.asarray(st)}, got {np.asarray(got)}. Parity measured against this frame "
                    f"would be comparing two different scenes.")
            out.append(np.ascontiguousarray(u.render()))
        return np.stack(out)

    @staticmethod
    def reference_step(ref, a):
        return ref.step(np.asarray(a, np.float32))

    @classmethod
    def step_reference_reset(cls, ref, seed):
        ref.reset(seed=int(seed))
        return np.asarray(cls.reference_sim_state(ref.unwrapped), np.float64)

    # ---------------------------------------------------------------- dynamics helpers

    def _clamp_border(self, pos, radius):
        """The reference's border clamp, evaluated in float64 and cast once.

        ``_apply_collisions`` reads ``float(pos2[0])`` and clamps against ``bs + agent_r`` in Python
        floats before rebuilding a float32 tensor. Doing the bound in float32 here would round it
        differently for a non-integral radius and the two would disagree by an ulp at the wall.
        """
        r = radius.double()
        lo = (float(BORDER) + r).to(torch.float32)
        hi = (float(IMG - BORDER) - r).to(torch.float32)
        return torch.maximum(torch.minimum(pos, hi.unsqueeze(1)), lo.unsqueeze(1))


# ====================================================================== TwoRoom


class _TwoRoomBase(_SWMNavBase):
    reference_id = "swm/TwoRoom-v1"

    def _structure_space(self) -> dict:
        return {
            "background": Dict({"color": RGBBox(init_value=BG_C)}),
            "wall": Dict({
                "color": RGBBox(init_value=WALL_C),
                # swm declares these Discrete/MultiDiscrete. The stored form is a float32 row either
                # way (that is what makes style storable), and the renderer reads the number, so they
                # are Boxes with the same bounds here and the compat layer converts. ``wall.axis``
                # stays Discrete because it selects a branch rather than a length.
                "thickness": Discrete(35, start=7, init_value=WALL_THICKNESS),
                "axis": Discrete(2, init_value=WALL_AXIS),
            }),
            "door": Dict({
                "color": RGBBox(init_value=DOOR_C),
                "number": Discrete(3, start=1, init_value=N_DOORS),
                "position": Box(low=0.0, high=223.0, shape=(MAX_DOOR,), init_value=list(DOOR_POS)),
                "size": Box(low=1.0, high=21.0, shape=(MAX_DOOR,), init_value=list(DOOR_SIZE)),
            }),
        }

    def _masks(self, sl, k):
        """``(k,H,W)`` door and wall masks, in the reference's order and with its border folded in."""
        half = sl("wall.thickness").trunc().long().div(2, rounding_mode="floor").view(-1, 1, 1)
        axis = (sl("wall.axis") > 0.5).view(-1, 1, 1)
        gy = self._gy.unsqueeze(0)
        gx = self._gx.unsqueeze(0)
        lo = (WALL_CENTER - half).to(torch.float32)
        hi = (WALL_CENTER + half).to(torch.float32)
        stripe = torch.where(axis, (gx >= lo) & (gx <= hi), (gy >= lo) & (gy <= hi))

        pos = sl("door.position")
        size = sl("door.size")
        num = sl("door.number").trunc().view(-1, 1, 1)
        # the coordinate a door's extent is measured along: y for a vertical wall, x for a horizontal
        # one. Built once outside the loop; inside it, it is (k,H,W) float32 and three copies of it
        # is 300 MB at the render chunk.
        along = torch.where(axis, gy, gx)
        span = torch.zeros(k, IMG, IMG, dtype=torch.bool, device=self.device)
        for i in range(MAX_DOOR):
            c = pos[:, i].view(-1, 1, 1)
            s = size[:, i].view(-1, 1, 1)
            span |= ((along >= (c - s)) & (along <= (c + s))) & (num > i)

        door = stripe & span
        wall = (stripe & ~span) | self._border.unsqueeze(0)
        return door, wall

    def base_image(self, sl, k) -> torch.Tensor:
        img = self._fill(k, sl("background.color"))
        door, wall = self._masks(sl, k)
        img = self._paint(img, door, sl("door.color"))
        return self._paint(img, wall, sl("wall.color"))

    # -------------------------------------------------------------- dynamics

    def _in_any_door(self, coord, sl):
        pos, size, num = sl("door.position"), sl("door.size"), sl("door.number").trunc()
        hit = torch.zeros_like(coord, dtype=torch.bool)
        for i in range(MAX_DOOR):
            inside = (coord >= pos[:, i] - size[:, i] - DOOR_MARGIN) & \
                     (coord <= pos[:, i] + size[:, i] + DOOR_MARGIN)
            hit |= inside & (num > i)
        return hit

    @torch.no_grad()
    def step(self, action) -> torch.Tensor:
        """One control step. Exact against ``TwoRoomEnv.step`` — no contact, no chaos, just a clamp.

        The wall test is the reference's, including the detail that it uses the position BEFORE the
        move to decide which side the agent started on, and that a blocked move is clamped to the
        wall face with a 0.5 px pushback rather than reverted.
        """
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, 2)
        a = a.clamp(-1.0, 1.0)
        sty = self._style
        sl = lambda p: sty[p]
        r = sl("agent.radius").reshape(-1)
        pos = self._state[:, 0:2]
        nxt = self._clamp_border(pos + a * float(self.speed), r)

        half = sl("wall.thickness").trunc().long().div(2, rounding_mode="floor")
        axis1 = sl("wall.axis") > 0.5
        c = float(WALL_CENTER)
        rd = r.double()
        eff_lo = (c - half.double() - rd).to(torch.float32)
        eff_hi = (c + half.double() + rd).to(torch.float32)

        # index 0 along a vertical wall, 1 along a horizontal one; ``other`` is the coordinate the
        # door test is applied to
        moving = torch.where(axis1, nxt[:, 0], nxt[:, 1])
        started = torch.where(axis1, pos[:, 0], pos[:, 1]) < c
        other = torch.where(axis1, nxt[:, 1], nxt[:, 0])
        open_door = self._in_any_door(other, sl)
        block_lo = started & (moving > eff_lo) & ~open_door
        block_hi = (~started) & (moving < eff_hi) & ~open_door
        moved = torch.where(block_lo, eff_lo - WALL_PUSHBACK,
                            torch.where(block_hi, eff_hi + WALL_PUSHBACK, moving))
        new = torch.stack([torch.where(axis1, moved, nxt[:, 0]),
                           torch.where(axis1, nxt[:, 1], moved)], dim=1)
        self._state[:, 0:2] = new
        return self._state

    # -------------------------------------------------------------- reference bridge

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        return np.asarray(_u(env).agent_position.detach().cpu().numpy(), np.float64)


@register
class TwoRoom(_TwoRoomBase):
    """``swm/TwoRoom-v1`` at its shipped appearance: two rooms, a wall with doors, no target drawn.

    State is the agent position and nothing else, because that is all the frame reads under
    ``render_target = 0``. The goal image a swm workflow consumes is not a second kind of frame — the
    official env produces it as ``_render_frame(agent_pos=target_position)``, i.e. this same renderer
    at a different state — so a corpus of agent positions regenerates the goals too.
    """

    name = "swm-two-room"
    state_dim = 2
    state_names = ("agent_x", "agent_y")
    reference_kwargs = {"render_target": False}

    def make_variation_space(self) -> Dict:
        return Dict({"agent": Dict(self._agent_space()), **self._structure_space()})

    def step(self, action) -> torch.Tensor:
        """Declared on the concrete class so the shared step check actually runs.

        ``test_gpu_step_tracks_the_reference_simulator`` guards on ``"step" in cls.__dict__`` rather
        than on ``hasattr``, and deliberately so: pointing the step checker at an inherited step it
        cannot drive has aborted the interpreter rather than failing one test. The consequence is
        that a ``step`` defined only on a shared base is silently never checked — measured, this env
        family reported "no batched GPU simulator declared as checkable" while shipping one. One
        line per concrete class is the cost of being covered.
        """
        return super().step(action)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """The official reset distribution for ``agent.position``.

        ``Box(14, 209)^2`` with ``_constrain_agent_not_in_wall``: at the default thickness 10 and
        radius 7 that rejects ``x`` in ``[100, 124]``, so the agent always starts inside one of the
        two rooms rather than embedded in the wall. Sampling the raw box instead would spend a
        quarter of the parity states on a configuration the env never resets into.
        """
        rng = rng or self.rng
        n = int(batch)
        lo = WALL_CENTER - WALL_THICKNESS // 2 - AGENT_R
        hi = WALL_CENTER + WALL_THICKNESS // 2 + AGENT_R
        xs = np.empty(n, np.float32)
        filled = 0
        while filled < n:
            cand = rng.uniform(POS_LO, POS_HI, n).astype(np.float32)
            cand = cand[(cand < lo) | (cand > hi)]
            take = min(n - filled, cand.size)
            xs[filled:filled + take] = cand[:take]
            filled += take
        ys = rng.uniform(POS_LO, POS_HI, n).astype(np.float32)
        return np.stack([xs, ys], axis=1)


@register
class TwoRoomTarget(_TwoRoomBase):
    """``swm/TwoRoom-v1`` constructed with ``render_target=True``: the target dot is drawn.

    A separate class rather than a knob because ``rendering.render_target`` ships as 0. With it off,
    ``target.color``, ``target.radius`` and the target's coordinates change no pixel, so declaring
    them would advertise knobs that do nothing and put two dead dimensions in the state. Here they
    are all live, and the state carries the target because the frame reads it.
    """

    name = "swm-two-room-target"
    state_dim = 4
    state_names = ("agent_x", "agent_y", "target_x", "target_y")
    reference_kwargs = {"render_target": True}
    step_reference_groups = (("agent", (0, 1), "euclidean"), ("target", (2, 3), "euclidean"))

    def make_variation_space(self) -> Dict:
        return Dict({
            "agent": Dict(self._agent_space()),
            "target": Dict({"color": RGBBox(init_value=TARGET_C),
                            "radius": Box(low=7.0, high=14.0, shape=(1,), init_value=[TARGET_R])}),
            **self._structure_space(),
        })

    def step(self, action) -> torch.Tensor:
        """Declared on the concrete class so the shared step check actually runs.

        ``test_gpu_step_tracks_the_reference_simulator`` guards on ``"step" in cls.__dict__`` rather
        than on ``hasattr``, and deliberately so: pointing the step checker at an inherited step it
        cannot drive has aborted the interpreter rather than failing one test. The consequence is
        that a ``step`` defined only on a shared base is silently never checked — measured, this env
        family reported "no batched GPU simulator declared as checkable" while shipping one. One
        line per concrete class is the cost of being covered.
        """
        return super().step(action)

    def _dots(self, s, sl, k):
        """Target then agent: the reference's order, and visible where the two overlap."""
        return (torch.cat([s[:, 2:4].reshape(k, 1, 2), s[:, 0:2].reshape(k, 1, 2)], dim=1),
                torch.cat([sl("target.radius").reshape(k, 1),
                           sl("agent.radius").reshape(k, 1)], dim=1),
                torch.cat([sl("target.color").reshape(k, 1, 3),
                           sl("agent.color").reshape(k, 1, 3)], dim=1))

    def blend_dots(self, img, s, sl, k) -> torch.Tensor:
        """Target first, agent over it — the reference's order, and visible where the two overlap.

        Two windowed blends rather than two full-frame ones, and the order between them is what
        makes that sound: each blend takes the uint8 image the previous one produced, so cropping
        the first cannot move a pixel the second then reads differently — outside its own window the
        first blend is the identity, which is exactly what the second would have read anyway.
        """
        img = self._blend_dot(img, s[:, 2:4], sl("target.radius"), sl("target.color"))
        return super().blend_dots(img, s, sl, k)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        agent = TwoRoom.sample_states(self, batch, rng)
        target = rng.uniform(POS_LO, POS_HI, (int(batch), 2)).astype(np.float32)
        return np.concatenate([agent, target], axis=1)

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        u = _u(env)
        return np.concatenate([u.agent_position.detach().cpu().numpy(),
                               u.target_position.detach().cpu().numpy()]).astype(np.float64)

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        v = np.asarray(st, np.float32)
        u.agent_position = torch.as_tensor(v[:2], dtype=torch.float32)
        u.target_position = torch.as_tensor(v[2:4], dtype=torch.float32)


# ====================================================================== Piecewise


class _PiecewiseBase(_SWMNavBase):
    reference_id = "swm/Piecewise-v0"
    state_dim = 2
    state_names = ("agent_x", "agent_y")
    #: DYNAMICS only. The reference draws these as a matplotlib quiver and only when
    #: ``render_bias_field`` is on, which this env does not reproduce; ``step`` still needs them.
    zone_bias = ZONE_BIAS

    def _common_space(self) -> dict:
        return {"agent": Dict(self._agent_space()),
                "background": Dict({"color": RGBBox(init_value=BG_C)}),
                "border": Dict({"color": RGBBox(init_value=BORDER_C)})}

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """``Box(14, 209)^2`` — PiecewiseEnv puts no constraint on ``agent.position``."""
        rng = rng or self.rng
        return rng.uniform(POS_LO, POS_HI, (int(batch), 2)).astype(np.float32)

    @torch.no_grad()
    def step(self, action) -> torch.Tensor:
        """One control step: zone lookup, biased move, border clamp. Exact against the reference."""
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, 2)
        a = a.clamp(-1.0, 1.0)
        r = self._style["agent.radius"].reshape(-1)
        pos = self._state[:, 0:2]
        play = float(IMG - 2 * BORDER)
        # float64, because the reference computes ``int((float(pos[0]) - bs) / play * n)`` in Python
        # floats. A float32 evaluation lands on the other side of the zone boundary for positions
        # within an ulp of it, which picks a different bias and diverges immediately rather than by
        # an ulp — the one place in this env where the dtype of an intermediate is load-bearing.
        idx = ((pos.double() - float(BORDER)) / play * GRID_N).trunc().long().clamp(0, GRID_N - 1)
        zone = idx[:, 1] * GRID_N + idx[:, 0]
        bias = torch.as_tensor(self.zone_bias, device=self.device, dtype=torch.float32)[zone]
        nxt = pos + a * float(self.speed) + bias
        self._state[:, 0:2] = self._clamp_border(nxt, r)
        return self._state

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        return np.asarray(_u(env).agent_position.detach().cpu().numpy(), np.float64)


@register
class Piecewise(_PiecewiseBase):
    """``swm/Piecewise-v0`` at its shipped appearance: one open room, four invisible bias zones.

    The zones are the whole point of the env and they are invisible by default — ``render_zones``
    ships as 0 — so the frame is background, a 4 px border and the agent dot. That is a *feature* for
    a world model (the dynamics are not readable off the pixels) and it is why the tinted
    configuration is registered separately as ``PiecewiseZones`` rather than folded in here.
    """

    name = "swm-piecewise"
    #: ``PiecewiseEnv.__init__`` takes ``render_target`` and ``grid_n`` and NOT ``render_zones``: the
    #: zone tint is reachable only through the variation space. Passing it as a kwarg raises, which is
    #: how the asymmetry with ``TwoRoomEnv`` was found; ``PiecewiseZones`` sets the leaf instead.
    reference_kwargs: dict = {}

    def make_variation_space(self) -> Dict:
        return Dict(self._common_space())

    def step(self, action) -> torch.Tensor:
        """Declared on the concrete class so the shared step check actually runs.

        ``test_gpu_step_tracks_the_reference_simulator`` guards on ``"step" in cls.__dict__`` rather
        than on ``hasattr``, and deliberately so: pointing the step checker at an inherited step it
        cannot drive has aborted the interpreter rather than failing one test. The consequence is
        that a ``step`` defined only on a shared base is silently never checked — measured, this env
        family reported "no batched GPU simulator declared as checkable" while shipping one. One
        line per concrete class is the cost of being covered.
        """
        return super().step(action)

    def base_image(self, sl, k) -> torch.Tensor:
        return self._paint(self._fill(k, sl("background.color")), self._border, sl("border.color"))


@register
class PiecewiseZones(_PiecewiseBase):
    """``swm/Piecewise-v0`` with ``rendering.render_zones = 1``: each bias zone tinted.

    The zone map itself is a constant of the family — it depends on ``grid_n`` and the 14 px margin
    and on nothing else — so only the four colours are variation, and the zone rectangles are
    ``[14,112)`` and ``[112,210)`` on each axis, i.e. the tint stops 4 px short of the border on the
    far side because ``int(row * 196 / 2)`` truncates.
    """

    name = "swm-piecewise-zones"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zones = _zone_map(self.device)

    def make_variation_space(self) -> Dict:
        z = {f"color_{i}": RGBBox(init_value=ZONE_C[i]) for i in range(N_ZONES)}
        return Dict({**self._common_space(), "zones": Dict(z)})

    def step(self, action) -> torch.Tensor:
        """Declared on the concrete class so the shared step check actually runs.

        ``test_gpu_step_tracks_the_reference_simulator`` guards on ``"step" in cls.__dict__`` rather
        than on ``hasattr``, and deliberately so: pointing the step checker at an inherited step it
        cannot drive has aborted the interpreter rather than failing one test. The consequence is
        that a ``step`` defined only on a shared base is silently never checked — measured, this env
        family reported "no batched GPU simulator declared as checkable" while shipping one. One
        line per concrete class is the cost of being covered.
        """
        return super().step(action)

    def base_image(self, sl, k) -> torch.Tensor:
        img = self._fill(k, sl("background.color"))
        for i in range(N_ZONES):
            img = self._paint(img, self._zones == i, sl(f"zones.color_{i}"))
        return self._paint(img, self._border, sl("border.color"))

    @classmethod
    def reference_env(cls, size: int = IMG):
        """The official env with the zone tint switched on.

        ``render_zones`` has no constructor kwarg — unlike ``render_target`` it is reachable only
        through the variation space — so it is set as the leaf's INIT value, which survives the
        ``reset()`` that ``reference_render`` performs. Setting it as a plain value would be undone
        by that reset and the reference would silently render the untinted picture.
        """
        import gymnasium
        import stable_worldmodel  # noqa: F401
        env = gymnasium.make(cls.reference_id)
        env.unwrapped.variation_space["rendering"]["render_zones"].set_init_value(1)
        return env

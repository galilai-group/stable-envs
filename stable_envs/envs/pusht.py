"""PushT — the first env, and the one the repo's contract was derived from.

Reference: ``gym_pusht/PushT-v0`` (pymunk/Chipmunk physics, pygame rendering). The frame is a pure
function of ``[agent_x, agent_y, block_x, block_y, block_angle]``, which is exactly the compact state a
PushT corpus already stores — so the stored images in every PushT dataset are redundant with 5 floats
per step.

Every constant below was MEASURED against the reference, not read off the source, because the source
reads plausibly and is wrong in five separate places. Each is recorded here because each cost real
debugging time and each is invisible by eye:

  * **No y-flip.** ``pymunk.pygame_util.positive_y_is_up`` is False, so ``to_pygame`` is
    ``(int(x), int(y))`` and screen y equals world y. The gym_pusht draw override devotes a long
    docstring to flipping, which makes ``512 - y`` the natural reading; it puts the agent at row 348
    where the reference draws it at 162.
  * **The block transform is ``pos + R(theta) @ v_local``**, not ``pos + R @ (v - cog)``. Measured to
    5.7e-14 over 6 random poses against ``pymunk.Body.local_to_world``; the cog variant is off by
    44.6 px. The T *does* set a non-default centre of gravity (0, 45) — that governs the dynamics, not
    this transform, which is why the wrong one is tempting.
  * **``SpaceDebugColor.as_int()`` rounds.** ``light_color`` scales by 1.2, so the block fill is
    ``(143,163,184)``; truncation gives ``(142,163,183)`` and an off-by-one on every interior pixel.
  * **``draw_fat_segment(radius=2)`` paints a 5 px stroke**, not 4: it draws a width-4 line and then,
    because ``r > 2``, an orthogonal quad extended by ``radius`` plus round caps. Measured by pixel
    count — half-width 2.0 emits 8012 wall pixels against the reference's 10009; 2.5 matches.
  * **The block paints over the agent.** pymunk's ``debug_draw`` walks ``space.shapes``, so the order is
    a dict-iteration detail with no documentation. Resolved by rendering both orders against the
    reference on poses where the two shapes actually overlap — on random rollouts they almost never do,
    and the search came back a meaningless tie (0.9127 vs 0.9110) before the overlap poses were added.

Measured agreement with the reference at ss=1: **100% of interior pixels bit-identical**, 95.6% of all
pixels, the remainder being edge anti-aliasing where the reference is itself aliased (it draws at 512
and ``cv2.resize``s down, so neither renderer is ground truth at an edge).

Physics
-------

PushT is the one env here that also ships ``step`` — a batched GPU port of Chipmunk's solver, so
``env.unroll / env.step on device`` runs entirely on device. **Renderer parity and simulator parity are
different claims and this file must not blur them.** The renderer is bit-exact on interior pixels; the
simulator is not bit-exact and cannot be. Contact is chaotic, so two solvers that disagree by a micron
at first touch disagree by centimetres later — that is true of any replacement engine (MJX, Brax,
Jax2D, this one) and is a property of the problem, not a defect of the port. Measured against pymunk on
identical action sequences from identical states, 16 episodes x 100 steps: **median block position error
0.68 px after 100 steps** (mean 20.7 — a few episodes diverge on chaotic contact), and the *kinematic
agent* integrates to 3e-5 px, which is the integrator check: with no contacts acting on it, any
disagreement there would be an integration bug rather than a solver difference.

Three measured properties of PushT decide the design and two of them are surprising. Each was read off
the live Chipmunk space, not off the source, because the source reads plausibly and is misleading:

  * **Contacts are frictionless.** ``gym_pusht`` sets ``body.friction = 1`` on the T and the agent, but
    Chipmunk reads friction from the *shape*, and the shapes are left at the default. Queried on the
    live space: ``shape.friction == 0.0`` for both block shapes and the agent, and Chipmunk combines as
    ``sqrt(a*b)``. The contact is genuinely frictionless and the T can only be driven along contact
    normals. Implementing the friction the source *looks* like it asks for would produce an env that
    pushes very differently from every PushT dataset in existence.
  * **Velocity does not persist.** ``space.damping`` is 0.0 and Chipmunk applies ``v *= damping**dt``
    each substep, so ``0.0 ** 0.01 == 0.0`` zeroes the block's linear and angular velocity at the start
    of every substep. PushT is quasi-static: the T moves only while something touches it. That is why a
    solver with no warm-starting is adequate — there is no momentum to carry across substeps, so the
    block's velocity is a substep-local quantity and not part of the state.
  * **The agent is kinematic.** Infinite mass, velocity prescribed by ``v += (k_p*(target - p) +
    k_v*(0 - v))*dt`` with k_p=100, k_v=20, 10 substeps of dt=0.01; contacts never push it.

``mass = 1.0``, ``moment = 3000.0``, ``cog = (0, 45)``, read from the live body. The moment is 2x
``moment_for_poly(1, vertices1)`` — which evaluates to exactly 3000.0 — because ``add_tee`` computes
``inertia2`` from ``vertices1``: the stem's inertia is never used. That is a bug in the reference and it
is reproduced deliberately, because the target is the env the datasets came from, not a corrected
version of it.

Every constant above is re-checkable in about ten lines against ``PushT.reference_env()``: ``space``
gives ``damping``, ``iterations``, ``collision_slop`` and ``collision_bias``; ``block`` gives the mass
properties; ``agent.body_type`` is ``KINEMATIC``; and iterating ``space.shapes`` shows ``friction 0.0``
on all seven shapes and ``radius 2.0`` on the four wall segments. They were confirmed that way in this
repo's environment, not carried over on trust.

Two coordinate facts that cost debugging time. The body rotates about its **centre of gravity**, so the
solver carries ``c`` and ``theta``, while the reported and rendered ``block position`` is the body
*origin*: ``pos = c - R(theta) @ cog``. And the physics wall planes sit at 7 and 504 (segments at 5 and
506 with collision radius 2), whereas the renderer draws those segments with half-width **2.5** —
``draw_fat_segment`` paints a 5 px stroke, measured by pixel count. The two numbers describe different
things and making them agree would break one of them.

The agent's velocity is NOT in the compact PACKED state, exactly as in ``gym_pusht``'s own observation,
so ``set_state`` zeroes it — the condition the reference is in after ``reset``. Two rollouts from the
same stored state therefore agree; a rollout *resumed* mid-episode does not inherit the previous
velocity, and neither does the reference's.

That velocity is part of the public State — ``state["agent", "velocity"]`` — while leaving the
packed row at the same five columns. The two statements are not in tension and the distinction is the
whole point of ``pack``/``unpack``:

  * the PACKED row is what a corpus stores, what every renderer reads and what every frozen Step-4
    digest was computed from, and it is byte-for-byte the certified packed row;
  * the PUBLIC state is what the dynamics need to be a FUNCTION. While the velocity lived in an
    attribute, ``transition(state, action)`` was not pure — it silently consulted whatever the env had
    been doing — so two calls with the same ``state`` could disagree. Now it cannot.

``unpack`` fills the velocity with zeros, which is not a placeholder: it is exactly the condition
``set_state`` puts the env in, so a state loaded from a corpus behaves as it always did.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from tensordict import TensorDict

from ..env import Env
from ..registry import register
from ..render.scene import Scene2D, SceneBuilder
from ..spaces import Box, Dict, Discrete, RGBBox

SCREEN = 512.0

#: body-local T vertices, CCW, straight from ``pymunk.Poly.get_vertices()``
BAR = ((-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0))
STEM = ((-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0))
AGENT_RADIUS = 15.0
AGENT_INNER_RADIUS = 11.0  # round(radius - 4)
OUTLINE_HALF_W = 2.5
WALL_HALF_W = 2.5
GOAL_POSE = (256.0, 256.0, math.pi / 4)
WALLS = (
    ((5.0, 506.0), (5.0, 5.0)),
    ((5.0, 5.0), (506.0, 5.0)),
    ((506.0, 5.0), (506.0, 506.0)),
    ((5.0, 506.0), (506.0, 506.0)),
)

BG = (255, 255, 255)
GOAL_C = (144, 238, 144)  # LightGreen
WALL_C = (211, 211, 211)  # LightGray
BLOCK_FILL = (143, 163, 184)  # light_color(LightSlateGray), rounded
BLOCK_LINE = (119, 136, 153)  # LightSlateGray
AGENT_C = (65, 105, 225)  # RoyalBlue
AGENT_IN = (78, 126, 255)  # light_color(RoyalBlue)

# ------------------------------------------------------------------ physics (see the module docstring)

DT = 0.01  # space.step increment
SUBSTEPS = 10  # int(1 / (dt * control_hz)), control_hz = 10
K_P, K_V = 100.0, 20.0  # the agent's PD gains
MASS = 1.0
MOMENT = 3000.0  # 2x moment_for_poly(1, BAR) — the reference's own bug, reproduced
COG = (0.0, 45.0)  # block centre of gravity, body-local
SLOP = 0.1  # space.collision_slop
COLLISION_BIAS = 0.0017970074436457143  # space.collision_bias, == (1 - 0.1) ** 60
BIAS_COEF = 1.0 - COLLISION_BIAS**DT  # ~0.06125, Chipmunk's per-substep correction gain
SOLVER_ITERS = 10  # space.iterations
#: Arena half-planes ``n . p >= d``. The walls are Segments of collision radius 2 at 5 and 506, so the
#: contact surface is at 7 and 504 — NOT the 2.5 half-width the renderer draws (see the docstring).
WALL_PLANES = (((1.0, 0.0), 7.0), ((-1.0, 0.0), -504.0), ((0.0, 1.0), 7.0), ((0.0, -1.0), -504.0))
#: Deepest k vertices per wall. One point cannot stop the T rotating into a wall; two can.
N_WALL_CONTACTS = 2


def _cross2(a, b):
    """z-component of the 2-D cross product, batched over trailing dims."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@register
class PushT(Env):
    """T-block pushing. State ``[agent_x, agent_y, block_x, block_y, block_angle]`` in a 512 board."""

    name = "push-t"
    state_dim = 5
    state_names = ("agent_x", "agent_y", "block_x", "block_y", "block_angle")
    default_size = 96
    reference_id = "gym_pusht/PushT-v0"
    #: opt in to test_gpu_step_tracks_the_reference_simulator: this env ships a batched
    #: GPU simulator AND declares (below) how to drive its CPU reference
    step_reference_checkable = True

    #: How ``testing.conformance.check_step_matches_reference`` reports this env. The agent is the
    #: control: it is kinematic, so no contact acts on it and the two integrators must agree to float
    #: precision — a disagreement there is an integration bug, not solver divergence, and without its
    #: own group it would hide inside the block numbers. The block is the gate: an episode whose block
    #: was never touched agrees trivially and is evidence of nothing.
    step_reference_groups = (("agent", (0, 1), "euclidean"), ("block", (2, 3), "euclidean"), ("angle", (4,), "angle"))
    step_reference_control = "agent"
    step_reference_gate = "block"
    #: Targets are held for 8 steps because one resampled every step keeps the agent in transit and
    #: rarely produces sustained contact — the regime that actually exercises the solver.
    step_reference_hold = 8

    #: gym_pusht's action space: an ABSOLUTE target position for the agent in the 512-unit board, which
    #: is why the bounds are (0, 512) and not (-1, 1). ``rollout.RandomPolicy`` reads these three.
    action_dim = 2
    action_low = 0.0
    action_high = 512.0

    #: Capture the 10-substep env step into a CUDA graph on first use. The step is 10 substeps x 10
    #: solver iterations of small tensor ops, i.e. ~2,000 kernel launches, and uncaptured it measured
    #: 52.7, 52.8, 52.9 and 52.8 ms at batches 256, 1024, 4096 and 16384 — a 64x range of work at a
    #: constant cost, which is the signature of pure launch overhead and nothing else. Captured, the
    #: same steps take 8.07, 8.06, 9.02 and 15.25 ms, i.e. the step finally responds to how much work it
    #: is doing. Set False before the first ``step`` to keep the eager path (useful under a profiler,
    #: where a replayed graph shows up as one opaque node).
    use_step_graph = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f = dict(device=self.device, dtype=torch.float32)
        self._cog = torch.tensor(COG, **f)
        #: block vertices relative to the CoG; they double as contact arms, since the body rotates
        #: about the CoG. ``pos + R @ v_local`` (the render transform) and ``c + R @ (v_local - cog)``
        #: (this one) describe the same world points, because ``pos = c - R @ cog``.
        self._quads_c = torch.tensor([BAR, STEM], **f) - self._cog  # (2,4,2)
        self._wall_n = torch.tensor([w[0] for w in WALL_PLANES], **f)  # (4,2)
        self._wall_d = torch.tensor([w[1] for w in WALL_PLANES], **f)  # (4,)
        #: The agent's velocity is real dynamical state that the compact state does not carry (neither
        #: does gym_pusht's observation), so it lives here and ``set_state`` zeroes it.
        self._agent_v = torch.zeros(self.B, 2, **f)
        self._step_graph = None
        self._step_target = None
        self._step_graph_error = ""

    def make_variation_space(self) -> Dict:
        """Every knob that changes the picture without changing the physics state.

        Scales are multipliers on the reference geometry rather than absolute sizes, so ``1.0`` is
        exactly the reference and a parity test can assert that identity — an absolute default would
        make "matches gym_pusht" depend on a number nobody checks.

        These are APPEARANCE ONLY, and now that this env also ships ``step`` that is a trap worth
        naming: ``block.scale``, ``agent.scale`` and ``goal.position`` change what is drawn and not what
        is simulated, so a randomised ``block.scale`` renders a T of a different size while the solver
        keeps pushing the reference one. Randomise colours freely; randomise geometry only when nothing
        is stepping, or the frame advertises a world the dynamics do not share.
        """
        return Dict(
            {
                "agent": Dict(
                    {
                        "color": RGBBox(init_value=AGENT_C),
                        "inner_color": RGBBox(init_value=AGENT_IN),
                        "scale": Box(low=0.25, high=3.0, shape=(1,), init_value=[1.0]),
                    }
                ),
                "block": Dict(
                    {
                        "color": RGBBox(init_value=BLOCK_FILL),
                        "outline_color": RGBBox(init_value=BLOCK_LINE),
                        "scale": Box(low=0.25, high=3.0, shape=(1,), init_value=[1.0]),
                    }
                ),
                "goal": Dict(
                    {
                        "color": RGBBox(init_value=GOAL_C),
                        "position": Box(low=0.0, high=512.0, shape=(2,), init_value=list(GOAL_POSE[:2])),
                        "angle": Box(low=-math.pi, high=math.pi, shape=(1,), init_value=[GOAL_POSE[2]]),
                        "scale": Box(low=0.25, high=3.0, shape=(1,), init_value=[1.0]),
                    }
                ),
                "walls": Dict(
                    {
                        "color": RGBBox(init_value=WALL_C),
                        "visible": Discrete(2, init_value=1),
                    }
                ),
                "background": Dict({"color": RGBBox(init_value=BG)}),
                "rendering": Dict({"render_goal": Discrete(2, init_value=1)}),
            }
        )

    # geometry

    def _t_verts(self, pos, angle, scale):
        """(B,2,4,2) world vertices of the two quads. ``pos`` (B,2), ``angle`` (B,), ``scale`` (B,)."""
        local = torch.tensor([BAR, STEM], device=self.device, dtype=torch.float32)  # (2,4,2)
        local = local.unsqueeze(0) * scale.view(-1, 1, 1, 1)
        c, s = torch.cos(angle), torch.sin(angle)
        rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)
        return torch.einsum("bij,bqkj->bqki", rot, local) + pos.view(-1, 1, 1, 2)

    def scene(self, states, variation: dict | None = None, style=None) -> Scene2D:
        n = int(states.shape[0]) if hasattr(states, "shape") else len(states)
        return self._scene(states, self.resolve_style(style, variation, batch=n))

    def _scene(self, states, sty) -> Scene2D:
        """Build the scene from the compact state and a resolved per-environment style tensor."""
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32).reshape(-1, self.state_dim)
        B = s.shape[0]
        sb = SceneBuilder(B, device=self.device, vertices=4)

        # draw order is the reference's: goal, then walls, then agent, then block on top
        gpos = sty["goal.position"]
        gang = sty["goal.angle"]
        gscale = sty["goal.scale"]
        show_goal = sty["rendering.render_goal"] > 0.5
        gverts = self._t_verts(gpos, gang, gscale)
        gcol = sty["goal.color"]
        for q in range(2):
            sb.add_poly(gverts[:, q], gcol, enabled=show_goal)

        wcol = sty["walls.color"]
        show_walls = sty["walls.visible"] > 0.5
        for a, b in WALLS:
            sb.add_segment(a, b, WALL_HALF_W, wcol, enabled=show_walls)
        # Goal T and walls do not depend on the state, only on the variation, so they are baked into a
        # cached base image and skipped on every subsequent frame. They are 6 of the 9 primitives here.
        sb.mark_static()

        ascale = sty["agent.scale"]
        sb.add_disc(
            s[:, 0:2],
            AGENT_RADIUS * ascale,
            sty["agent.color"],
            inner_radius=AGENT_INNER_RADIUS * ascale,
            inner_color=sty["agent.inner_color"],
        )

        bscale = sty["block.scale"]
        bverts = self._t_verts(s[:, 2:4], s[:, 4], bscale)
        bfill = sty["block.color"]
        bline = sty["block.outline_color"]
        for q in range(2):
            sb.add_poly(bverts[:, q], bfill, outline_color=bline, outline_w=OUTLINE_HALF_W * bscale)

        return sb.build(background=sty["background.color"], world=(0.0, 0.0, SCREEN, SCREEN), flip_y=False)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """gym_pusht's reset distribution: agent U[50,450]^2, block U[100,400]^2, angle U[-pi,pi]."""
        rng = rng or self.rng
        return np.stack(
            [
                rng.uniform(50, 450, batch),
                rng.uniform(50, 450, batch),
                rng.uniform(100, 400, batch),
                rng.uniform(100, 400, batch),
                rng.uniform(-np.pi, np.pi, batch),
            ],
            axis=1,
        ).astype(np.float32)

    # ------------------------------------------------------------------ physics

    def set_state(self, states, style=None) -> PushT:
        """As the base class, and zero the agent's velocity — the state does not carry it.

        Restoring a stored state must put the env in one definite condition, and the only defensible
        one is the reference's after ``reset``: at rest. Leaving the velocity from whatever the env was
        doing before would make ``set_state(s)`` depend on history, which is precisely the failure this
        repo documents for ``gym_pusht._set_state`` on the position/angle order.
        """
        super().set_state(states, style=style)
        self._agent_v.zero_()
        return self

    def _rot(self, th: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(th), torch.sin(th)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)

    def _cog_of(self, pos: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """Body origin -> centre of gravity. The solver's coordinate."""
        return pos + torch.einsum("bij,j->bi", self._rot(th), self._cog)

    def _origin_of(self, c: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """Centre of gravity -> body origin. What ``get_state`` reports and the renderer draws."""
        return c - torch.einsum("bij,j->bi", self._rot(th), self._cog)

    def _block_verts_cog(self, th: torch.Tensor) -> torch.Tensor:
        """``(B,2,4,2)`` block vertices relative to the CoG, i.e. the contact arms."""
        return torch.einsum("bij,qkj->bqki", self._rot(th), self._quads_c)

    def _disc_quad_contacts(self, ap, c, rel):
        """Agent disc vs each quad -> ``(n_push (B,2,2), depth (B,2), arm (B,2,2))``.

        Signed distances to the four edge planes pick the separating feature: for a convex polygon, when
        the centre is outside, the closest point lies on the edge of maximal signed distance or at one
        of its endpoints, so a single clamped projection covers the edge and vertex cases together. When
        the centre is inside, the least-penetrating face is the contact.
        """
        cc = (ap - c).unsqueeze(1)  # (B,1,2), disc centre rel CoG
        a = rel  # (B,2,4,2)
        b = torch.roll(rel, -1, dims=2)
        e = b - a
        elen = e.norm(dim=-1).clamp_min(1e-9)
        n = torch.stack([e[..., 1], -e[..., 0]], -1) / elen.unsqueeze(-1)  # outward, CCW winding
        w = cc.unsqueeze(2) - a
        s = (w * n).sum(-1)  # (B,2,4)
        smax, imax = s.max(dim=2)
        gather = lambda t: torch.gather(t, 2, imax[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        a_i, e_i, n_i = gather(a), gather(e), gather(n)
        wv = cc - a_i
        t = ((wv * e_i).sum(-1) / (e_i * e_i).sum(-1).clamp_min(1e-9)).clamp(0, 1)
        q = a_i + t.unsqueeze(-1) * e_i  # closest point on the polygon
        dvec = cc - q
        dist = dvec.norm(dim=-1).clamp_min(1e-9)
        inside = smax <= 0
        n_out = torch.where(inside.unsqueeze(-1), n_i, dvec / dist.unsqueeze(-1))
        depth = torch.where(inside, AGENT_RADIUS - smax, AGENT_RADIUS - dist)
        arm = torch.where(inside.unsqueeze(-1), cc - n_i * smax.unsqueeze(-1), q)
        return -n_out, depth, arm  # n_push drives the BLOCK, i.e. away from the disc

    def _wall_contacts(self, c, rel):
        """Deepest ``N_WALL_CONTACTS`` block vertices per wall -> ``(n_push, depth, arm)``."""
        B = rel.shape[0]
        v = rel.reshape(B, 8, 2)
        world = v + c.unsqueeze(1)
        proj = torch.einsum("bvi,wi->bwv", world, self._wall_n)  # (B,4,8)
        depth = self._wall_d.view(1, 4, 1) - proj  # > 0 == penetrating
        d, idx = depth.topk(N_WALL_CONTACTS, dim=2)
        arm = torch.gather(v.unsqueeze(1).expand(-1, 4, -1, -1), 2, idx[..., None].expand(-1, -1, -1, 2))  # (B,4,k,2)
        n = self._wall_n.view(1, 4, 1, 2).expand(B, 4, N_WALL_CONTACTS, 2)
        k = 4 * N_WALL_CONTACTS
        return n.reshape(B, k, 2), d.reshape(B, k), arm.reshape(B, k, 2)

    def _substep(self, ap, av, c, th, target):
        """One 0.01 s substep, purely functional -> ``(agent_p, agent_v, cog, theta)``.

        Functional rather than mutating attributes because the whole 10-substep sequence is captured
        into a CUDA graph: capture needs stable addresses and no Python-side control flow, and rebinding
        locals keeps every intermediate inside the graph's private pool.

        Sequential-impulse with accumulated clamped normal impulses and Chipmunk's split-impulse
        positional correction — ``slop``, ``collision_bias`` and the iteration count all read off the
        live space. Contacts are solved Jacobi-style so a batch stays one kernel per iteration; with at
        most ten simultaneous contacts and no momentum to carry, that converges inside the budget.
        """
        # 1. agent PD. The agent is kinematic: contacts never modify this.
        av = av + (K_P * (target - ap) + K_V * (0.0 - av)) * DT

        # 2. damping: space.damping == 0 means the block starts every substep at rest.
        vc = torch.zeros_like(c)
        omega = torch.zeros_like(th)

        # 3. contacts
        rel = self._block_verts_cog(th)
        cn, cd, ca = self._disc_quad_contacts(ap, c, rel)
        wn, wd, wa = self._wall_contacts(c, rel)
        n = torch.cat([cn, wn], dim=1)  # (B,NC,2) push direction
        depth = torch.cat([cd, wd], dim=1)
        arm = torch.cat([ca, wa], dim=1)  # (B,NC,2) rel CoG
        # velocity of the OTHER body at the contact: the agent for the disc contacts, 0 for the walls
        v_other = torch.cat([av.unsqueeze(1).expand(-1, cn.shape[1], -1), torch.zeros_like(wa)], dim=1)
        active = depth > 0

        rn = _cross2(arm, n)
        k_eff = (1.0 / MASS + rn * rn / MOMENT).clamp_min(1e-9)
        bias = BIAS_COEF * (depth - SLOP).clamp_min(0.0) / DT

        lam = torch.zeros_like(depth)
        lam_b = torch.zeros_like(depth)
        vcb = torch.zeros_like(c)
        omb = torch.zeros_like(th)
        zero = torch.zeros_like(depth)
        for _ in range(SOLVER_ITERS):
            # real velocity: separation speed along n must be >= 0
            v_at = vc.unsqueeze(1) + torch.stack(
                [-omega.unsqueeze(1) * arm[..., 1], omega.unsqueeze(1) * arm[..., 0]], -1
            )
            vs = ((v_at - v_other) * n).sum(-1)
            new = (lam + torch.where(active, -vs / k_eff, zero)).clamp_min(0.0)
            dl = new - lam
            lam = new
            vc = vc + torch.einsum("bc,bci->bi", dl / MASS, n)
            omega = omega + (dl * rn / MOMENT).sum(1)

            # split impulse: the overlap correction lives in its own velocity channel (Chipmunk's
            # v_bias) so pushing bodies apart cannot inject energy into the real velocity.
            vb_at = vcb.unsqueeze(1) + torch.stack(
                [-omb.unsqueeze(1) * arm[..., 1], omb.unsqueeze(1) * arm[..., 0]], -1
            )
            vsb = (vb_at * n).sum(-1)
            newb = (lam_b + torch.where(active, (bias - vsb) / k_eff, zero)).clamp_min(0.0)
            dlb = newb - lam_b
            lam_b = newb
            vcb = vcb + torch.einsum("bc,bci->bi", dlb / MASS, n)
            omb = omb + (dlb * rn / MOMENT).sum(1)

        # 4. integrate
        return ap + av * DT, av, c + (vc + vcb) * DT, th + (omega + omb) * DT

    def _advance(self, state, av, target):
        """``(B,5)`` + agent velocity + PD target -> the same, one control step (10 substeps) later.

        The compact state is converted to the solver's CoG coordinate here and back at the end, once per
        env step rather than per substep, so ``get_state`` keeps returning the body ORIGIN that the
        renderer and gym_pusht report and there is only one representation of the truth.
        """
        ap, th = state[:, 0:2], state[:, 4]
        c = self._cog_of(state[:, 2:4], th)
        for _ in range(SUBSTEPS):
            ap, av, c, th = self._substep(ap, av, c, th, target)
        return torch.cat([ap, self._origin_of(c, th), th.unsqueeze(1)], dim=1), av

    def _capture_step(self) -> None:
        """Capture one env step into a CUDA graph, or record why it could not be.

        The graph reads ``self._state`` and ``self._agent_v`` at fixed addresses, so anything that
        REBINDS those attributes instead of writing through ``copy_`` would leave the graph replaying
        against dead memory — the env would look like it silently ignored every ``set_state``. The base
        class writes with ``copy_``; keep it that way.
        """
        try:
            self._step_target = torch.zeros(self.B, 2, device=self.device, dtype=torch.float32)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warm up autotune/allocator before capture
                    self._advance(self._state, self._agent_v, self._step_target)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                st, av = self._advance(self._state, self._agent_v, self._step_target)
                self._state.copy_(st)
                self._agent_v.copy_(av)
            self._step_graph = g
        except Exception as e:  # noqa: BLE001 - fall back, do not hide it
            self._step_graph = None
            self._step_graph_error = f"{type(e).__name__}: {e}"
            self.use_step_graph = False

    # ------------------------------------------------------------------ state / lifecycle

    #: The public state, grouped by entity, and how it maps onto the five packed columns.
    #: ``("agent", "velocity")`` carries offset ``None``: it is real dynamical state that the packed
    #: row has never carried (neither does gym_pusht's observation), which is what makes
    #: ``transition`` pure without moving a single column of the corpus layout.
    state_layout = (
        (("agent", "position"), 2, 0),
        (("agent", "velocity"), 2, None),
        (("block", "position"), 2, 2),
        (("block", "angle"), 1, 4),
    )

    #: Bounds for ``state_spec``: the 512-unit board, the reference's own action range for the
    #: velocity (the PD controller cannot exceed it by construction), and a full turn for the angle.
    state_bounds = {
        ("agent", "position"): (0.0, SCREEN),
        ("agent", "velocity"): (-SCREEN, SCREEN),
        ("block", "position"): (0.0, SCREEN),
        ("block", "angle"): (-math.pi, math.pi),
    }

    def _hidden_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """Where the one field the packed row does not carry actually lives.

        ``unpack`` fills the velocity with zeros, which is right for a state loaded from a corpus and
        wrong for the env's own live state after a step: ``step`` -> ``state`` -> ``transition`` must
        agree to the bit or the functional path is a different simulator. This is the whole of what a
        cell with hidden dynamics has to say; ``Env.state`` does the splicing.
        """
        return {("agent", "velocity"): self._agent_v[: self._live_rows].clone()}

    def _target(self, action) -> torch.Tensor:
        """The absolute PD target as ``(n, 2)`` float32 on this env's device."""
        return torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(-1, 2)

    @torch.no_grad()
    def transition(self, state: TensorDict, action) -> TensorDict:
        """``(state, action) -> next_state``. FUNCTIONAL: the env's own buffers are not touched.

        Bit-identical to ``step`` on the same device for the same state, because it is the same
        ``_advance`` over the same float32 inputs — the eager path, since a CUDA graph reads fixed
        addresses (``self._state``, ``self._agent_v``) and this call may not write them. ``step``
        keeps the captured path; ``tests/unit/test_pusht_api.py`` asserts the two agree to the
        bit on CPU and the oracle script asserts it again on CUDA.
        """
        rows = self.pack(state)
        av = torch.as_tensor(state["agent", "velocity"], device=self.device, dtype=torch.float32).reshape(-1, 2)
        st, av = self._advance(rows, av, self._target(action))
        out = self.unpack(st)
        out["agent", "velocity"] = av
        return out

    @torch.no_grad()
    def step(self, action) -> TensorDict:
        """Advance every environment one control step. ``action`` is ``(B,2)``, the absolute PD target.

        STATEFUL: it writes ``env.state``. Returns a :data:`~stable_envs.env.StepResult` carrying
        ``"state"`` AND NOTHING ELSE — PushT has no reward and no termination in stable-envs. gym_pusht's reward
        is a coverage score computed from the goal pose, which is an APPEARANCE knob here
        (``goal.position``/``goal.angle`` are variation, not physics), so a reward computed from it
        would be a different quantity wearing the same name. A ``"reward"`` of zeros would be worse:
        it reads as a measured value. The cell omits both (contract §5).

        Tracks pymunk to a median 0.68 px over 100 steps and is NOT bit-exact — see the module
        docstring. ``stable_envs.testing.conformance``'s ``check_step_matches_reference`` measures
        that rather than asserting it.
        """
        self._check_open("step")
        # ``(B, 2)`` and not ``(-1, 2)``: the captured graph's target buffer is ``(B, 2)`` and
        # ``copy_`` would BROADCAST a one-row target across every environment instead of refusing it.
        target = self._target(action).reshape(self.B, 2)
        replayed = False
        if self.use_step_graph and self.device.type == "cuda":
            if self._step_graph is None:
                self._capture_step()
            if self._step_graph is not None:
                self._step_target.copy_(target)
                self._step_graph.replay()
                replayed = True
        if not replayed:
            st, av = self._advance(self._state, self._agent_v, target)
            self._state.copy_(st)
            self._agent_v.copy_(av)
        return self._step_result(self.state)

    def _close(self) -> None:
        """Drop the captured CUDA graph. ``Env.close`` calls this once and is idempotent."""
        self._step_graph = None
        self._step_target = None

    # reference bridge, used by the parity tests

    @staticmethod
    def reference_env(size: int = 96):
        """The official env, with the pymunk>=7 collision-API shim gym_pusht needs."""
        import pymunk

        if not hasattr(pymunk.Space, "add_collision_handler"):

            class _H:
                _PHASES = ("begin", "pre_solve", "post_solve", "separate")

                def __init__(self, space, a, b):
                    object.__setattr__(self, "_space", space)
                    object.__setattr__(self, "_types", (a, b))

                def __setattr__(self, k, v):
                    if k in self._PHASES:
                        self._space.on_collision(*self._types, **{k: v})
                    else:
                        object.__setattr__(self, k, v)

            pymunk.Space.add_collision_handler = lambda self, a=0, b=0: _H(self, a, b)
        import gym_pusht  # noqa: F401
        import gymnasium

        return gymnasium.make(
            "gym_pusht/PushT-v0",
            obs_type="state",
            render_mode="rgb_array",
            observation_width=size,
            observation_height=size,
        )

    @staticmethod
    def reference_sim_state(ref) -> np.ndarray:
        """The reference's state in the same layout as ``get_state``. Read after ``reset``, because
        gym_pusht builds the bodies in ``_setup()``: probing before that reports "no agent" on a
        working env."""
        u = ref.unwrapped
        if not (hasattr(u, "agent") and hasattr(u, "block")):
            raise ValueError("gym_pusht's env exposes no `agent`/`block` bodies to read a state from")
        return np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)

    @staticmethod
    def step_reference_actions(rng, steps: int, hold: int) -> np.ndarray:
        """Absolute PD targets inside the arena, resampled every ``hold`` steps.

        The range is (60, 452) rather than the full (0, 512) action space: a target on the wall pins the
        agent against it for the whole hold, and an episode spent leaning on a wall measures the wall
        constraint rather than the solver.
        """
        out = np.empty((int(steps), 2), np.float32)
        tgt = None
        for i in range(int(steps)):
            if i % max(1, int(hold)) == 0:
                tgt = rng.uniform(60, 452, 2).astype(np.float32)
            out[i] = tgt
        return out

    @staticmethod
    def reference_set_state(u, st) -> None:
        """Place the reference env at ``st``. ANGLE FIRST, then position — the order is load-bearing.

        Chipmunk stores the body transform about its centre of gravity, and PushT's T has
        ``cog = (0, 45)``. Setting ``position`` and then ``angle`` therefore rotates the body about a
        point that is not where you just placed it, and the position you read back is NOT the one you
        wrote: measured, target (256,256) comes back as (284.99, 266.58) after a previous angle of 0.0,
        (241.63, 233.62) after 1.3, and (325.91, 202.86) after -2.0. The error is up to ~70 px and
        depends on the block's PREVIOUS angle, so the same requested state lands in a different place
        depending on history. Reversing the order round-trips to 2.8e-14 regardless of history.

        ``gym_pusht._set_state`` uses the broken order on purpose — its own comment says the angle
        "should theoretically be set first" but does the opposite "for compatibility with legacy data".
        Anything that restores PushT from a stored state through that path (dataset replay, planning
        resets, evaluation from a state file) is therefore placing the T somewhere other than the
        recorded pose, silently. This repo does not reproduce that bug: the state is the pose, and
        setting it puts the body there.
        """
        u.block.angle = float(st[4])
        u.block.position = [float(st[2]), float(st[3])]
        u.agent.position = [float(st[0]), float(st[1])]
        # BOTH bodies must be reindexed. Chipmunk caches each shape's world transform and only refreshes
        # it on a space step or an explicit reindex, so a manually-moved body keeps DRAWING at its old
        # pose while ``body.position`` already reports the new one. Reindexing only the block left the
        # agent painted at a fixed screen position (72.4, 56.5) for every state — the reference frames
        # silently stopped being a function of the agent's coordinates, which looked like a bug in our
        # renderer rather than in how the reference was being driven.
        u.space.reindex_shapes_for_body(u.block)
        u.space.reindex_shapes_for_body(u.agent)

    @staticmethod
    def reference_render(env, states) -> np.ndarray:
        """Official frames for the given states, by writing each state into the reference env.

        The round-trip is asserted rather than assumed: if the reference env does not actually end up
        at the requested state, every parity number computed from it compares two different scenes and
        the comparison is meaningless — which is precisely the failure mode the setter-order bug above
        produces, and it looks like a rendering discrepancy.
        """
        u = env.unwrapped
        env.reset(seed=0)
        out = []
        for st in np.asarray(states):
            PushT.reference_set_state(u, st)
            got = np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)
            err = np.abs(got - np.asarray(st, np.float64)).max()
            if err > 1e-6:
                raise AssertionError(
                    f"reference env did not reach the requested state (max err {err:.3e}); "
                    f"asked {np.asarray(st)}, got {got}. Parity measured against this frame would "
                    f"be comparing two different scenes."
                )
            out.append(u._render(visualize=False).copy())
        return np.stack(out)

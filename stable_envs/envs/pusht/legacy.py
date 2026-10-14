from __future__ import annotations

import math

import numpy as np
import torch
from tensordict import TensorDict

from ...env import Env
from ...registry import register
from ...render.scene import Scene2D, SceneBuilder
from ...spaces import Box, Dict, Discrete, RGBBox

SCREEN = 512.0

# Counterclockwise body-local vertices from pymunk.Poly.get_vertices().
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


DT = 0.01  # space.step increment
SUBSTEPS = 10  # int(1 / (dt * control_hz)), control_hz = 10
K_P, K_V = 100.0, 20.0  # the agent's PD gains
MASS = 1.0
MOMENT = 3000.0  # Match gym_pusht: both inertia terms use BAR, including the stem term.
COG = (0.0, 45.0)  # block centre of gravity, body-local
SLOP = 0.1  # space.collision_slop
COLLISION_BIAS = 0.0017970074436457143  # space.collision_bias, == (1 - 0.1) ** 60
BIAS_COEF = 1.0 - COLLISION_BIAS**DT  # ~0.06125, Chipmunk's per-substep correction gain
SOLVER_ITERS = 10  # space.iterations
# Collision planes use the wall radius of 2; rendering uses a separate half-width of 2.5.
WALL_PLANES = (((1.0, 0.0), 7.0), ((-1.0, 0.0), -504.0), ((0.0, 1.0), 7.0), ((0.0, -1.0), -504.0))
# Use two vertices per wall so a single contact cannot let the block rotate through it.
N_WALL_CONTACTS = 2


def _cross2(a, b):
    """Return the scalar 2-D cross product over the last dimension."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@register
class PushT(Env):
    """Compact PushT renderer with an approximate tensor-based contact solver.

    Packed render states contain agent position, block origin and block angle
    on a 512-unit board. Actions are absolute agent target positions.
    Use ``pusht.semantics.PushT`` for source-aligned simulation and replay State.
    """

    family = "pusht"
    domain = "pusht"

    name = "push-t"
    state_dim = 5
    state_names = ("agent_x", "agent_y", "block_x", "block_y", "block_angle")
    default_size = 96
    reference_id = "gym_pusht/PushT-v0"
    step_reference_checkable = True

    # Check kinematic agent motion separately from contact-sensitive block motion.
    step_reference_groups = (("agent", (0, 1), "euclidean"), ("block", (2, 3), "euclidean"), ("angle", (4,), "angle"))
    step_reference_control = "agent"
    step_reference_gate = "block"
    # Hold targets long enough to produce sustained contact.
    step_reference_hold = 8

    action_dim = 2
    action_low = 0.0
    action_high = 512.0

    # Disable before the first step to profile eager execution instead of CUDA graph replay.
    use_step_graph = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f = dict(device=self.device, dtype=torch.float32)
        self._cog = torch.tensor(COG, **f)
        # Contact arms are relative to CoG; rendering vertices are relative to the body origin.
        self._quads_c = torch.tensor([BAR, STEM], **f) - self._cog  # (2,4,2)
        self._wall_n = torch.tensor([w[0] for w in WALL_PLANES], **f)  # (4,2)
        self._wall_d = torch.tensor([w[1] for w in WALL_PLANES], **f)  # (4,)
        # Velocity is absent from the compact projection and must be retained separately for stepping.
        self._agent_v = torch.zeros(self.B, 2, **f)
        self._step_graph = None
        self._step_target = None
        self._step_graph_error = ""

    def make_variation_space(self) -> Dict:
        """Define appearance settings; geometry scales of one match the reference.

        Block/agent scale and goal pose affect rendering only. Changing these
        settings does not change the contact solver geometry.
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

    def _t_verts(self, pos, angle, scale):
        """Return world vertices ``(B, 2, 4, 2)`` for position, angle and scale batches."""
        local = torch.tensor([BAR, STEM], device=self.device, dtype=torch.float32)  # (2,4,2)
        local = local.unsqueeze(0) * scale.view(-1, 1, 1, 1)
        c, s = torch.cos(angle), torch.sin(angle)
        rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)
        return torch.einsum("bij,bqkj->bqki", rot, local) + pos.view(-1, 1, 1, 2)

    def scene(self, states, variation: dict | None = None, style=None) -> Scene2D:
        n = int(states.shape[0]) if hasattr(states, "shape") else len(states)
        return self._scene(states, self.resolve_style(style, variation, batch=n))

    def _scene(self, states, sty) -> Scene2D:
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32).reshape(-1, self.state_dim)
        B = s.shape[0]
        sb = SceneBuilder(B, device=self.device, vertices=4)

        # Match reference draw order: goal, walls, agent, then block.
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
        # Cache goal and walls, which depend on appearance but not on the current pose.
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
        """Sample the gym_pusht reset distribution for agent position, block position and angle."""
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

    def _set_render_state(self, states, variation=None, style=None) -> PushT:
        """Set the compact pose and zero agent velocity, which is absent from that projection."""
        super()._set_render_state(states, variation=variation, style=style)
        self._agent_v.zero_()
        return self

    def _rot(self, th: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(th), torch.sin(th)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)

    def _cog_of(self, pos: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """Convert body origin to the center-of-gravity coordinates used by the solver."""
        return pos + torch.einsum("bij,j->bi", self._rot(th), self._cog)

    def _origin_of(self, c: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """Convert center of gravity to the body origin used by the renderer."""
        return c - torch.einsum("bij,j->bi", self._rot(th), self._cog)

    def _block_verts_cog(self, th: torch.Tensor) -> torch.Tensor:
        """Return rotated block vertices relative to the center of gravity: ``(B, 2, 4, 2)``."""
        return torch.einsum("bij,qkj->bqki", self._rot(th), self._quads_c)

    def _disc_quad_contacts(self, ap, c, rel):
        """Return push normals ``(B, 2, 2)``, depths ``(B, 2)`` and contact arms ``(B, 2, 2)``.

        Outside each quad, project onto the edge with the largest signed distance.
        Inside it, use the least-penetrating face. Normals push the block away
        from the agent.
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
        """Return normals, depths and arms for the deepest block vertices at each wall."""
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
        """Advance 0.01 seconds and return agent position/velocity, block CoG and angle.

        Solve normal impulses in parallel and apply overlap correction through
        a separate bias velocity, leaving the physical velocity unchanged.
        """
        # The agent is kinematic; contacts must not modify its PD velocity.
        av = av + (K_P * (target - ap) + K_V * (0.0 - av)) * DT

        # Reference damping is zero, so the block starts each substep at rest.
        vc = torch.zeros_like(c)
        omega = torch.zeros_like(th)

        rel = self._block_verts_cog(th)
        cn, cd, ca = self._disc_quad_contacts(ap, c, rel)
        wn, wd, wa = self._wall_contacts(c, rel)
        n = torch.cat([cn, wn], dim=1)  # (B,NC,2) push direction
        depth = torch.cat([cd, wd], dim=1)
        arm = torch.cat([ca, wa], dim=1)  # (B,NC,2) rel CoG
        # Disc contacts see the agent velocity; wall contacts see a stationary body.
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
            # Normal impulses prevent approaching velocity at active contacts.
            v_at = vc.unsqueeze(1) + torch.stack(
                [-omega.unsqueeze(1) * arm[..., 1], omega.unsqueeze(1) * arm[..., 0]], -1
            )
            vs = ((v_at - v_other) * n).sum(-1)
            new = (lam + torch.where(active, -vs / k_eff, zero)).clamp_min(0.0)
            dl = new - lam
            lam = new
            vc = vc + torch.einsum("bc,bci->bi", dl / MASS, n)
            omega = omega + (dl * rn / MOMENT).sum(1)

            # Resolve overlap through bias velocity so positional correction does not add kinetic energy.
            vb_at = vcb.unsqueeze(1) + torch.stack(
                [-omb.unsqueeze(1) * arm[..., 1], omb.unsqueeze(1) * arm[..., 0]], -1
            )
            vsb = (vb_at * n).sum(-1)
            newb = (lam_b + torch.where(active, (bias - vsb) / k_eff, zero)).clamp_min(0.0)
            dlb = newb - lam_b
            lam_b = newb
            vcb = vcb + torch.einsum("bc,bci->bi", dlb / MASS, n)
            omb = omb + (dlb * rn / MOMENT).sum(1)

        return ap + av * DT, av, c + (vc + vcb) * DT, th + (omega + omb) * DT

    def _advance(self, state, av, target):
        """Advance one control step of ten substeps, converting block origin to CoG and back."""
        ap, th = state[:, 0:2], state[:, 4]
        c = self._cog_of(state[:, 2:4], th)
        for _ in range(SUBSTEPS):
            ap, av, c, th = self._substep(ap, av, c, th, target)
        return torch.cat([ap, self._origin_of(c, th), th.unsqueeze(1)], dim=1), av

    def _capture_step(self) -> None:
        """Capture one control step, retaining an error message if capture fails.

        Replay requires fixed addresses for state, velocity and target buffers.
        Update those tensors in place; rebinding them would leave the graph
        reading old storage.
        """
        try:
            self._step_target = torch.zeros(self.B, 2, device=self.device, dtype=torch.float32)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warm up autotune/allocator before capture
                    self._advance(self._render_state, self._agent_v, self._step_target)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                st, av = self._advance(self._render_state, self._agent_v, self._step_target)
                self._render_state.copy_(st)
                self._agent_v.copy_(av)
            self._step_graph = g
        except Exception as e:  # noqa: BLE001 - fall back, do not hide it
            self._step_graph = None
            self._step_graph_error = f"{type(e).__name__}: {e}"
            self.use_step_graph = False

    # Agent velocity has no packed column; retain it separately for full-State continuation.
    state_layout = (
        (("agent", "position"), 2, 0),
        (("agent", "velocity"), 2, None),
        (("block", "position"), 2, 2),
        (("block", "angle"), 1, 4),
    )

    state_bounds = {
        ("agent", "position"): (0.0, SCREEN),
        ("agent", "velocity"): (-SCREEN, SCREEN),
        ("block", "position"): (0.0, SCREEN),
        ("block", "angle"): (-math.pi, math.pi),
    }

    def _restore_state(self, state: TensorDict) -> None:
        self._set_render_state(self._pack_render_state(state))
        self._agent_v.copy_(state["agent", "velocity"])

    def _hidden_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """Supply live agent velocity when reconstructing full State from the compact pose."""
        return {("agent", "velocity"): self._agent_v[: self._live_rows]}

    def _target(self, action) -> torch.Tensor:
        """Convert absolute PD targets to device float32 rows ``(n, 2)``."""
        return torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(-1, 2)

    @torch.no_grad()
    def _step(self, action) -> TensorDict:
        """Advance absolute agent targets ``(B, 2)`` and return State.

        This approximate solver does not compute reward or termination. The
        source-aligned public PushT classes are defined in ``pusht.semantics``.
        """
        self._check_open("step")
        # Require B target rows: copy_ would otherwise broadcast one target across the captured batch.
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
            st, av = self._advance(self._render_state, self._agent_v, target)
            self._render_state.copy_(st)
            self._agent_v.copy_(av)
        return self._step_result(self.state)

    def _close(self) -> None:
        self._step_graph = None
        self._step_target = None

    @staticmethod
    def reference_env(size: int = 96):
        """Construct gym_pusht with a collision-handler adapter for Pymunk 7."""
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
        """Read agent position, block origin and angle after the reference has been reset."""
        u = ref.unwrapped
        if not (hasattr(u, "agent") and hasattr(u, "block")):
            raise ValueError("gym_pusht's env exposes no `agent`/`block` bodies to read a state from")
        return np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)

    @staticmethod
    def step_reference_actions(rng, steps: int, hold: int) -> np.ndarray:
        """Sample absolute PD targets in [60, 452], holding each for ``hold`` steps.

        Targets stay away from walls to encourage sustained agent–block contact.
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
        """Restore a reference pose, setting block angle before position.

        Rotation changes the body origin around its offset center of gravity.
        Reversing the assignment order makes the final position depend on the
        previous angle.
        """
        u.block.angle = float(st[4])
        u.block.position = [float(st[2]), float(st[3])]
        u.agent.position = [float(st[0]), float(st[1])]
        # Reindex both bodies after moving them; Pymunk rendering reads cached shape transforms.
        u.space.reindex_shapes_for_body(u.block)
        u.space.reindex_shapes_for_body(u.agent)

    @staticmethod
    def reference_render(env, states) -> np.ndarray:
        """Render reference poses after verifying that restoration reached each requested pose."""
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
                    f"asked {np.asarray(st)}, got {got}."
                )
            out.append(u._render(visualize=False).copy())
        return np.stack(out)

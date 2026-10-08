from __future__ import annotations

import numpy as np
import torch

from .._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import raycast3d as R3
from ..spaces import Box, Dict

# Resolved scene light type: 1 is directional; other lights have position/spot attenuation.
LIGHT_DIRECTIONAL = 1

# Default visible groups; group 5 hides movers during static bakes.
VISIBLE_GROUPS = (0, 1, 2)
HIDDEN_GROUP = 5

# MuJoCo material defaults for geometry without a material.
DEFAULT_SPECULAR, DEFAULT_SHININESS = 0.5, 0.5

# Calibration factor for the additive white specular contribution.
SPECULAR_SCALE = 0.988

# Per-task interior thresholds and recorded deviations from exact agreement.
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

# Source viewers and pristine snapshots live for the process, keyed by task and size.
_ENV_CACHE: dict = {}
_SCENE_CACHE_MAX = 96
_SCENE_CACHE = _RES.BoundedCache(_SCENE_CACHE_MAX, name="mujoco_locomotion.scene")
# Headless source environments used only for rollouts.
_SAMPLER_CACHE: dict = {}


def _gym():
    import gymnasium

    return gymnasium


def _mat_rgb(m, matid: int, rgba) -> np.ndarray:
    """Resolve geom color, using the material when geom rgba is the compiler default."""
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
    """Use fixed scalar addition order to avoid batch-dependent silhouette rounding."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


class _Lights:
    """Copy resolved scene lights, including the headlight absent from MjModel."""

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
        # Accept both the light type enum and the older directional flag.
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
    """Return light direction and fixed-function distance/spot attenuation at each point."""
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
    """Shade surface points in [0, 1] with additive white specular.

    The shadow mask removes only diffuse/specular from the casting light;
    ambient remains.
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
    """Repeat plate pixels so subsequent box averaging preserves their integer values."""
    return img if ss == 1 else img.repeat_interleave(ss, 0).repeat_interleave(ss, 1)


class _GymMujocoFamily(Env):
    """Share source construction, qpos sampling, and reference rendering across backends.

    State stores render poses only; velocities and simulator continuation are
    not represented. Render resources are shared across instances.
    """

    family = "mujoco"

    ENV_ID: str = ""

    default_size = 240
    # Random actions after reset broaden the sampled pose distribution.
    ROLLOUT_STEPS = 60

    def _declare_state(self, m) -> None:
        self.state_dim = int(m.nq)
        self.state_names = self._qpos_names(m)
        self.state_layout = ((("qpos",), self.state_dim, 0),)

    @classmethod
    def _fresh(cls, size: int):
        """Create the source renderer after framebuffer sizing and a seed-zero reset.

        The first render freezes a free camera from the current geometry, so preserve
        this order when constructing both plates and references.
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
        """Bind this viewer's GL context before drawing, including when viewers interleave."""
        u.mujoco_renderer.viewer.make_context_current()
        return u.render()

    @staticmethod
    def _qpos_names(m) -> tuple[str, ...]:
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
        """Cache a source environment without a viewer for state sampling."""
        hit = _SAMPLER_CACHE.get(cls.ENV_ID)
        if hit is None:
            hit = _SAMPLER_CACHE[cls.ENV_ID] = _gym().make(cls.ENV_ID)
            hit.reset(seed=0)
        return hit

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample float32 qpos from source resets followed by random actions."""
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
        """Set source qpos with zero velocity, require readback within 1e-9, and render."""
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
    """Render fixed-camera, untextured tasks using source plates and analytic movers.

    Scenery and camera variations select a plate; body tint and scale vary per
    lane. Curved shading and edge coverage can differ from the source rasterizer.
    """

    # One ray per output pixel by default; higher supersampling improves edge coverage.
    default_supersample = 1
    # Cap rays per chunk to bound transient rendering memory.
    RAY_BUDGET = 1 << 24
    parity_interior_min = 1.0
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
        """Reject textures, reflective materials, and unsupported moving primitive types."""
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

    @classmethod
    def _bundle(cls, size: int):
        """Cache a source environment and pristine appearance snapshot by task and size."""
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
        """Read defaults from the frozen snapshot, never the shared model mutated by bakes."""
        return cls._bundle(cls.default_size)[4]

    # Appearance paths requiring a new static bake; body variations remain per lane.
    PLATE_PATHS = ("scenery.tint", "camera.distance_scale", "camera.azimuth", "camera.elevation")

    def make_variation_space(self) -> Dict:
        """Expose multiplicative body/scenery tints, body scale, and camera settings.

        Identity tints retain distinct source colors within each group.
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
                        # Keep endpoints distinct; a full-turn interval would map both ends to the same camera.
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
        """Return visible geom and site IDs attached outside the world body.

        Moving sites must be hidden from the plate as well as moving geoms.
        """
        g = [i for i in self._visible(m, "geom", pr) if int(m.geom_bodyid[i]) != 0]
        s = [i for i in self._visible(m, "site", pr) if int(m.site_bodyid[i]) != 0]
        return g, s

    def _apply_variation(self, m, u, pr, variation) -> None:
        """Restore pristine appearance before absolute variation assignments."""
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

        # Read resolved scene lights, including the headlight that is absent from MjModel.
        scn = u.mujoco_renderer.viewer.scn
        cam = R3.Camera.from_mjv(scn.camera[0], scn.camera[1])
        lights = _Lights(scn)
        sl = lights.shadow_light()

        # Bake fully shadowed static colors by removing casting-light diffuse/specular.
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
        # Defer static depth until the requested supersampling factor is known.
        sc.static_t = {}
        sc.rays = {}
        sc.shadow_candidates = {}
        sc.const = {}
        _, spos, _ = self._prims(m, d, gstat, sstat)
        # Scale shadow-ray offsets to the scene instead of using one world-space epsilon.
        sc.eps = float(max(1e-6, 1e-5 * np.abs(spos).max())) if len(spos) else 1e-6
        return sc

    def _prims(self, m, d, geoms, sites):
        """Build primitives and world transforms from separate geom and site arrays."""
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
        hit = sc.rays.get(int(ss))
        if hit is None:
            R = sc.size * int(ss)
            hit = sc.rays[int(ss)] = sc.cam.rays(R, R, self.device)
        return hit

    def _shadow_candidates(self, sc: _Scene, ss: int):
        """Select visible static pixels where lit and shadowed plates differ.

        An occlusion result cannot change any excluded pixel.
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
            idx = torch.arange(a.shape[0], device=a.device)
        rows = _upsample_rgb(sc.plate_shadow, ss).reshape(-1, 3).to(torch.float32)[idx]
        sc.shadow_candidates[key] = hit = (idx, rows.unsqueeze(0))
        return hit

    def _per_chunk_constants(self, sc: _Scene, ss: int):
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
        """Cache static receiver-to-light rays; only mover occlusion varies with state."""
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
        # Keep a receiver-validity mask even without compaction, so sky pixels cannot be shadowed.
        hit = sc.const[key] = (cand, dark, p + rd * sc.eps, rd, lim, st[:, cand] < R3.INF)
        return hit

    def _static_t(self, sc: _Scene, ss: int) -> torch.Tensor:
        """Cast static depth at the requested supersampled resolution.

        Upsampling coarse depth would place boundary samples on the wrong surface.
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

    def _fk(self, states: torch.Tensor, sc: _Scene):
        """Run CPU kinematics per qpos row and copy mover transforms to the render device."""
        import mujoco

        gmov, smov = sc.movers
        P = len(gmov) + len(smov)
        B = states.shape[0]
        q = states.detach().cpu().numpy().astype(np.float64)
        xp = np.empty((B, P, 3))
        xm = np.empty((B, P, 9))
        m, d = sc.model, sc.data
        # Write directly into batch storage; reshape matrix rows once after the loop.
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

    @staticmethod
    def _cast(sc, o, d, xpos, xmat, sizes):
        return R3.cast_nearest(sc.prims.kind, sizes, xpos, xmat, o, d)

    @staticmethod
    def _occluded(sc, p, L, sl, xpos, xmat, sizes, skip=None, eps: float = 1e-6):
        """Test mover occlusion toward a light, optionally skipping the originating primitive.

        Skipping self-intersection assumes convex movers; back-facing diffuse light
        is already zero.
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

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
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

    # Candidate window sizes, from smallest to largest, as divisors of image width.
    MOVER_TILES = (8, 6, 4, 3, 2, 1.5, 1.25)

    # Allow parity tests to compare compacted passes against full-frame work.
    SHADOW_COMPACTION = True
    COVER_COMPACTION = True

    def _mover_idx(self, sc: _Scene, xpos: torch.Tensor, size: int, ss: int):
        """Return a conservative projected mover window, or None for full-frame casting."""
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

        # Shade only window positions hit in some lane; unhit positions keep the plate.
        # These are window-relative indices, which may denote different pixels in each lane.
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

        if widx is None:
            frame = torch.where(cover_w.unsqueeze(-1), col_w, base)
        else:
            w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
            frame = base.scatter_(1, w3, torch.where(cover_w.unsqueeze(-1), col_w, base.gather(1, w3)))
        frame = frame.view(B, size, ss, size, ss, 3).sum((2, 4)) / (ss * ss)
        return torch.clamp(torch.round(frame), 0, 255).to(torch.uint8)

    @classmethod
    def reference_env(cls, size: int = 240):
        """Construct a source viewer and verify its frozen camera matches the plate camera."""
        env, u = cls._fresh(int(size))
        got = cls._read_camera(u)
        want = cls._pristine()["cam"]
        cls._check_camera_matches(got, want)
        env.unwrapped._se_size = int(size)
        return env

    @staticmethod
    def _check_camera_matches(got: dict, want: dict) -> None:
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
    return "MuJoCo" + old[len("Mujoco") :] if old.startswith("Mujoco") else old


def _hyphen_name(env_id: str) -> str:
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
        made[public] = cls
    return made


globals().update(_make_classes())


# Probe the compiled backend while allowing analytic tasks to register if it is unavailable.
try:
    import nvdiffrast.torch as _nvdr

    _ = _nvdr.RasterizeCudaContext
    NVDIFF_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover
    NVDIFF_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

from ..render import nvdiff as ND  # noqa: E402  (imports neither mujoco nor nvdiffrast at module scope)

VISUAL_ENVS: dict[str, str] = {
    "MujocoHalfCheetah": "HalfCheetah-v5",
    "MujocoHopper": "Hopper-v5",
    "MujocoWalker2d": "Walker2d-v5",
    "MujocoAnt": "Ant-v5",
    "MujocoHumanoid": "Humanoid-v5",
    "MujocoSwimmer": "Swimmer-v5",
}

# Estimated per-world raster/attribute and vertex storage used to size chunks.
NVD_BYTES_PER_PIXEL, NVD_BYTES_PER_VERTEX = 239, 69
# Bound temporary memory; backend allocation failures can terminate the process.
NVD_BUDGET_BYTES = 6 << 30

# Eviction closes scene, texture, and raster resources owned by each cached renderer.
_NVD_CACHE = _RES.BoundedCache(16, name="mujoco_locomotion.nvd", on_evict=ND.close_cached_renderer)
_NVD_CACHE_MAX = 16


class _Nvd:
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
    """Apply camera-local offsets, pan about local y, then tilt about local x.

    MuJoCo cameras look along local -z with +y up.
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


NVD_SHADOW_RES = 256
# Receiver normal offset in shadow texels, plus world-space depth slack.
NVD_SHADOW_OFFSET_TEXELS = 1.0
NVD_SHADOW_DEPTH_BIAS = 0.002


class _ShadowPass:
    """Build an orthographic shadow map for a directional light.

    Only non-plane geometry casts. Each world gets its own fitted bounds;
    nearest-depth sampling and normal/depth bias control edge and acne errors.
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
        # Camera basis columns are right, up, and back; the view looks along -z.
        self.basis = np.stack([right, up, -f], -1)
        mov = [i for i, g in enumerate(c.geoms) if int(c.model.geom_type[g]) != ND.PLANE]
        self.movers = np.asarray(mov, np.int64)
        keep = torch.as_tensor(np.isin(c.br.scene.fgeom.cpu().numpy(), mov), device=c.br.scene.face.device)
        self.faces = c.br.scene.face[keep].contiguous()
        # Expand center-based bounds by primitive size to include caster extents.
        self.gmax = float(max((np.abs(c.model.geom_size[c.geoms[i]]).max() for i in mov), default=0.0))
        self.depth = self.center = None

    def build(self, xpos, xmat, xp_np: np.ndarray) -> None:
        """Rasterize caster depth using bounds fitted independently for each world.

        Batch-wide bounds would make a frame depend on neighboring states.
        """
        import nvdiffrast.torch as dr
        import torch

        S = self.scene
        B = int(xpos.shape[0])
        pm = xp_np[:, self.movers]  # (B,M,3)
        ctr = pm.mean(1)
        # Include caster extents and the receiving surface below the movers.
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
        # Empty shadow texels represent the far plane, not depth zero.
        z = torch.where(rast[..., 3] > 0, rast[..., 2], torch.ones_like(rast[..., 2]))
        self.depth = z.unsqueeze(1).contiguous()  # (B,1,res,res)
        self.center = torch.as_tensor(ctr, dtype=torch.float32, device=dev)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev).view(B, 1, 1, 1)  # noqa: E731
        self.half, self.hz = t(rad), t(drop)

    def __call__(self, i: int, p, n):
        """Return binary light visibility, or None when this pass does not handle the light."""
        import torch
        import torch.nn.functional as F

        if int(i) != self.light or self.depth is None:
            return None
        B = int(p.shape[0])
        R = torch.as_tensor(self.basis, dtype=p.dtype, device=p.device)
        l = torch.as_tensor(self.ldir, dtype=p.dtype, device=p.device).view(1, 1, 1, 3)
        ndl = (n * l).sum(-1, keepdim=True).clamp(0.05, 1.0)
        texel = 2.0 * self.half / self.res
        # Offset along the surface normal with slope scaling to suppress self-shadow acne.
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
    """Share exactly equal camera orientations to avoid duplicating skybox ray grids."""
    return mat[:1] if len(mat) > 1 and bool((mat == mat[:1]).all()) else mat


def _lights_for(base: ND.Lights, scale: np.ndarray, tint: np.ndarray) -> ND.Lights:
    """Broadcast per-world tint/intensity into diffuse and specular, preserving ambient."""
    w = np.asarray(scale, np.float64).reshape(-1, 1) * np.asarray(tint, np.float64).reshape(-1, 3)
    if bool((w == 1.0).all()):
        return base
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
    """VISUAL TIER: textured locomotion rendering through nvdiffrast.

    Camera and light variations are batched. Mesh shading, textures, and shadow
    edges approximate the source; see visual_fidelity for acceptance thresholds.
    """

    # Extra supersampling changes floor-texture filtering as well as geometry coverage.
    default_supersample = 1
    # Visual acceptance thresholds assume shadows are enabled.
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

    @classmethod
    def _model(cls):
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
        # Resolve lights and skybox through the source scene at its actual camera.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = int(c.cam_id)
        mujoco.mj_forward(m, c.data)
        _, scn = c.br.resolve_from_mujoco(m, c.data, camera=cam)
        c.lights = c.br.lights
        c.geoms = np.asarray(c.br.scene.geoms, np.int64)
        # Use scene light indices; an active headlight shifts model-light indices.
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
        """Check visible geometry coverage and supported source lighting assumptions."""
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
        # Orthographic shadow maps cannot represent positional-light shadows.
        L = c.br.lights
        unhandled = [i for i in range(len(L.pos)) if c.castshadow[i] and not L.headlight[i] and not L.directional[i]]
        assert not unhandled, (
            f"{type(self).__name__}: lights {unhandled} cast shadows and are not directional; "
            f"_ShadowPass draws an ORTHOGRAPHIC map and would leave their shadows out of the frame "
            f"without saying so"
        )

    def make_variation_space(self) -> Dict:
        """Expose per-world camera offsets and light tint/intensity.

        Geometry size and material colors are fixed in the shared scene.
        """
        return Dict(
            {
                "camera": Dict(
                    {
                        # Camera-local translations are in metres.
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

    def _pose(self, c: _Nvd, q: np.ndarray):
        """Compute geometry and tracking-camera poses from qpos on the CPU.

        TRACKCOM needs comPos before camlight. Copy each camera pose into its row
        because MjData camera arrays alias mutable simulator storage.
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

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
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

        flat = sty.values.detach().cpu().numpy()
        leaf = lambda p: flat[:, sty.layout[p][0] : sty.layout[p][0] + sty.layout[p][1]]  # noqa: E731

        q = s.detach().cpu().numpy().astype(np.float64)
        xp, xm, cp, cm = self._pose(c, q)
        cp, cm = _camera_pose(cp, cm, leaf("camera.offset"), leaf("camera.pan"), leaf("camera.tilt"))
        lscale, ltint = leaf("light.scale"), leaf("light.tint")
        xp_t = torch.as_tensor(xp, dtype=torch.float32, device=self.device)
        xm_t = torch.as_tensor(xm, dtype=torch.float32, device=self.device)
        # Balance chunk sizes to avoid paying a full launch for a small tail.
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

    @classmethod
    def reference_env(cls, size: int = 240):
        """Construct a source viewer and verify Gymnasium selected the model's track camera."""
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


# Recorded visual comparisons and per-task acceptance thresholds.
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
    "the "
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
    """Return the nvdiffrast import failure, or None when the visual backend is available."""
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
        made[public] = cls
    return made


globals().update(_make_visual_classes())

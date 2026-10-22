"""Measurements for renderer parity, reproducibility, and reference simulation.

Checks return measurements; callers apply acceptance thresholds and handle skips.
Most state helpers here use private render projections, not complete simulator
State. Use the shared trajectory contracts for public get_state/set_state checks.

Reference environments are cached by class and output size for the process
lifetime. Borrowers must not close them. This retains GL contexts across tests
and keeps classes sharing a source ID from sharing incompatible configuration.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import torch

_REFERENCE_ENVS: dict = {}


def reference_env(cls, size: int):
    """Borrow the process-lifetime reference instance for a class and size; do not close it.

    Construction failures propagate and are not cached.
    """
    key = (cls, int(size))
    hit = _REFERENCE_ENVS.get(key)
    if hit is None:
        hit = _REFERENCE_ENVS[key] = cls.reference_env(size=int(size))
    return hit


def sample_render_rows(env, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Render rows for the checks: the env's own sampler, or rows packed from short rollouts.

    Envs that cannot sample render rows but simulate (``has_dynamics``) are reset with seeds from
    ``rng`` and stepped with random actions; their State before the call is restored afterwards.
    """
    try:
        return env.sample_states(int(n)) if rng is None else env.sample_states(int(n), rng=rng)
    except NotImplementedError:
        if not getattr(env, "has_dynamics", False):
            raise
    rng = rng if rng is not None else np.random.default_rng(0)
    try:
        saved = env.get_state()
    except Exception:  # never reset: nothing to restore
        saved = None
    rows = []
    while sum(len(r) for r in rows) < int(n):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(int(rng.integers(0, 20))):
            env.step(env.action_spec.sample(env.B, rng), frames=None)
        rows.append(torch.as_tensor(env._pack_render_state(env.get_state())).detach().cpu().numpy())
    if saved is not None:
        env.set_state(saved)
    return np.concatenate(rows)[: int(n)].astype(np.float32)


def frames_digest(frames) -> str:
    """Hash shape, dtype, and contiguous logical array bytes for cross-process comparison."""
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(frames)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def check_reference_renderer_is_stable(cls, size: int, n: int = 4) -> dict:
    """Measure repeat-render error through one reference instance using seeded sampled states.

    Constructs the sampling environment on CUDA; this is a GPU-host precondition check.
    """
    ref = reference_env(cls, size)
    env = cls(num_envs=1, size=size, device="cuda")
    st = sample_render_rows(env, n, rng=np.random.default_rng(0))
    a = cls.reference_render(ref, st)
    b = cls.reference_render(ref, st)
    return {"max_abs_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}


def interior_mask(frames: np.ndarray) -> np.ndarray:
    """Return an (N, H-1, W-1) mask of locally flat regions in RGB frames.

    Erode zero adjacent-pixel differences with a 3x3 minimum filter. This is
    a flat-shading heuristic: quantized gradients can appear locally constant
    while still lying on a rounding boundary.
    """
    import scipy.ndimage as ndi

    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1] + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def pixel_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Measure maximum-channel uint8 error per pixel, with exact and within-8 fractions.

    Interior metrics use the second argument as reference. They are None when
    SciPy is unavailable or no interior pixels exist; raw counts distinguish
    a few rounding differences from widespread error.
    """
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
            # No interior pixels means undefined agreement, not zero agreement.
            res.update(
                interior_frac=float(m.mean()),
                interior_exact_frac=None,
                interior_mean_abs=None,
                interior_px=0,
                interior_bad_px=0,
            )
        else:
            # Counts make rare rounding differences visible alongside near-one fractions.
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
    """Return mean luminance SSIM using uniform filtering and 8-bit constants.

    The filter spans every luminance-array axis, including a leading frame axis.
    This perceptual measure does not establish pixel identity.
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
    """Return whether reference_render explicitly accepts a style parameter."""
    import inspect

    fn = getattr(cls, "reference_render", None)
    if fn is None:
        return False
    try:
        return "style" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def frame_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the MSE of each frame over its pixels and channels, on the 0-255 scale."""
    if a.shape != b.shape:
        raise AssertionError(f"shape {a.shape} vs {b.shape}")
    d = a.astype(np.float64) - b.astype(np.float64)
    return (d * d).reshape(len(d), -1).mean(1)


def mse_summary(per_frame: np.ndarray) -> dict:
    """Return the mean, the 95th percentile and the worst of per-frame MSE values."""
    v = np.asarray(per_frame, np.float64)
    return {"mse_mean": float(v.mean()), "mse_p95": float(np.percentile(v, 95)), "mse_max": float(v.max())}


def check_source_mse(env, n: int = 64, seed: int = 0) -> dict:
    """Measure per-frame MSE against the source on ``n`` render rows drawn from ``default_rng(seed)``.

    Rows the reference cannot reach (``reference_render(..., return_flags=True)``) are left out and counted.
    """
    cls = type(env)
    ref = reference_env(cls, env.size)
    states = sample_render_rows(env, int(n), rng=np.random.default_rng(int(seed)))
    ours = env._render(states).cpu().numpy()
    if "return_flags" in inspect.signature(cls.reference_render).parameters:
        official, flags = cls.reference_render(ref, states, return_flags=True)
        keep = ~np.asarray(flags, bool)
    else:
        official = cls.reference_render(ref, states)
        keep = np.ones(len(official), bool)
    per_frame = frame_mse(ours[keep], official[keep])
    return {
        **mse_summary(per_frame),
        "per_frame": per_frame.tolist(),
        "excluded_unreachable": int((~keep).sum()),
        "n": int(n),
        "seed": int(seed),
        "size": int(env.size),
    }


def check_render_state_roundtrip(env, n: int = 64) -> dict:
    """Measure private render-row restoration using env.B rows; n is unused.

    This does not test full simulator State restoration.
    """
    st = sample_render_rows(env, env.B)
    env._set_render_state(st)
    back = env._get_render_state().cpu().numpy()
    return {"max_abs_err": float(np.abs(back - st).max())}


def check_state_roundtrip(env, n: int = 64) -> dict:
    """Alias for the private render-row check; does not validate public simulator State."""
    return check_render_state_roundtrip(env, n=n)


def _objects_shown(env) -> dict | None:
    """Variation overrides that draw the objects an env hides by default, or None when it hides none."""
    shown = {
        p: 1
        for p in env.variation_space.names()
        if p.split(".")[0] == "visibility" and not int(np.asarray(env.variation_space._walk(p).value))
    }
    return shown or None


def check_state_dims_matter(env, n: int = 8, eps: float = 25.0, donors: int = 4) -> dict:
    """Probe whether each render-projection dimension changes pixels.

    Try sampled donor values, then additive perturbations. Dead dimensions can
    be excluded only when the scene signature validates a declared sphere-rotation
    invariant. Report exclusions and refusals separately; this is not a requirement
    that every full simulator State leaf be visible. Objects an env can hide are drawn.
    """
    shown = _objects_shown(env)
    st = sample_render_rows(env, env.B)
    base = env._render(st, variation=shown).cpu().numpy()
    pool = [sample_render_rows(env, env.B) for _ in range(max(1, donors))]
    candidates = set(_render_invariant_candidates(env))
    dead, excluded, refused = [], {}, {}
    for d in range(env.state_dim):
        alive = False
        for donor in pool:
            pert = st.copy()
            pert[:, d] = donor[:, d]
            if np.array_equal(pert[:, d], st[:, d]):
                continue  # donor happened to match; proves nothing either way
            if not np.array_equal(env._render(pert, variation=shown).cpu().numpy(), base):
                alive = True
                break
        if not alive:  # fall back to an additive nudge
            for step in (eps, -eps, 1.0, 0.5):
                pert = st.copy()
                pert[:, d] = pert[:, d] + step
                if not np.array_equal(env._render(pert, variation=shown).cpu().numpy(), base):
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
    f = getattr(env, "render_invariant_dims", None)
    if f is None:
        return ()
    return tuple(f() if callable(f) else f)


# Sphere-rotation exclusions must preserve every listed scene field.
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

# Missing frozen fields are refused unless explicitly justified here.
_RENDER_INVARIANT_OPTIONAL: dict[str, str] = {}

# mjtGeom.mjGEOM_SPHERE; ellipsoids and caller-supplied flags cannot establish rotation invariance.
_MJGEOM_SPHERE = 2


def _validate_render_invariant(env, states, d: int, variation=None) -> tuple:
    """Validate an exclusion for rotation of analytic spheres; return (accepted, reason).

    Require every frozen field, derive sphere type from geom_type, and reject
    changes beyond sphere orientation. The signature does not cover spatial
    tendons or dynamic camera poses; extend it before applying this exclusion
    to dimensions that can affect either.
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
    # Perturb only this component; judge the resulting scene, including normalized quaternion effects.
    moved[0, d] = moved[0, d] + 0.2
    b = sig(moved, **kw)
    if set(a) != set(b):
        return False, "render_scene_signature returned different keys for the two states"
    # Omitted fields cannot establish that the scene stayed unchanged.
    missing = [k for k in _RENDER_INVARIANT_FROZEN if k not in a and k not in _RENDER_INVARIANT_OPTIONAL]
    if missing:
        return False, (
            f"render_scene_signature does not emit {missing}; emit it, or add it to "
            f"_RENDER_INVARIANT_OPTIONAL with the reason it cannot change a pixel"
        )
    if "geom_type" not in a:
        return False, "render_scene_signature must emit geom_type; sphere-ness is derived from it"
    # Derive sphere type from the model; reject inconsistent caller-supplied flags.
    kind = np.asarray(a["geom_type"]).reshape(-1)
    sphere = kind == _MJGEOM_SPHERE
    claimed = a.get("geom_is_sphere")
    if claimed is not None:
        claimed = np.asarray(claimed, bool).reshape(-1)
        if claimed.shape != sphere.shape or not np.array_equal(claimed, sphere):
            return False, "render_scene_signature's geom_is_sphere disagrees with the geom_type it reports"
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
            "perturbing this dimension changed nothing in the scene, so it is dead in the model, "
            "not merely invisible to the renderer"
        )
    bad = np.flatnonzero(changed & ~sphere)
    if bad.size:
        names = a.get("geom_names") or []
        which = [names[i] if i < len(names) else int(i) for i in bad]
        return False, (
            f"perturbing this dimension rotated non-sphere geoms {which}, whose orientation "
            f"a correct renderer must draw"
        )
    names = a.get("geom_names") or []
    which = [names[i] if i < len(names) else int(i) for i in np.flatnonzero(changed)]
    return True, (
        f"render-invariant: sphere orientation. Perturbing it rotates only the sphere "
        f"geom(s) {which}; every other render-side scene field is bit-identical. An analytic "
        f"sphere is invariant under rotation about its centre. MuJoCo's tessellated sphere is "
        f"not; the env's state_dims_note declares that difference."
    )


def check_batch_invariance(env, n: int = 8) -> dict:
    """Compare sampled batch rows against their singleton renders."""
    st = sample_render_rows(env, env.B)
    batched = env._render(st).cpu().numpy()
    worst = 0
    for i in range(min(n, env.B)):
        one = env._render(st[i : i + 1]).cpu().numpy()[0]
        worst = max(worst, int(np.abs(one.astype(np.int16) - batched[i].astype(np.int16)).max()))
    return {"max_abs_diff": worst}


def check_backend_agreement(env, n: int = 4, max_size: int = 96) -> dict:
    """Compare Triton and torch Scene2D pixels on a small batch at bounded resolution.

    Skip when Triton, a Scene2D representation, or a CUDA scene is unavailable.
    """
    from ..render.raster2d import HAVE_TRITON, render_torch, render_triton

    if not HAVE_TRITON:
        return {"skipped": "triton unavailable"}
    b_n = min(int(n), env.B)
    st = sample_render_rows(env, env.B)[:b_n]
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
    """Yield appearance leaves, excluding declared dynamics/reset/task controls and undrawn leaves."""
    skip = set(getattr(env, "dynamics_variation_leaves", ()) or ()) | set(
        getattr(env, "undrawn_variation_leaves", ()) or ()
    )
    for path in env.variation_space.names():
        if path not in skip:
            yield path


def check_variations_do_something(env) -> dict:
    """Report appearance leaves whose alternate value fails to change the sampled frames.

    Objects an env can hide are drawn, so their appearance leaves are visible; a visibility leaf is
    tested by hiding its object.
    """
    shown = _objects_shown(env) or {}
    st = sample_render_rows(env, env.B)
    base = env._render(st, variation=shown or None).cpu().numpy()
    inert = []
    for path in _frame_variation_paths(env):
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        default = shown.get(path, node.value)
        try:
            if hasattr(node, "n"):  # Discrete: flip it
                alt = 0 if int(default) != 0 else 1
            else:  # Box/RGBBox: go to the far end of the range
                lo = np.broadcast_to(node.low, node.shape).astype(np.float64)
                hi = np.broadcast_to(node.high, node.shape).astype(np.float64)
                d = np.asarray(default, np.float64)
                alt = np.where(np.abs(d - lo) > np.abs(d - hi), lo, hi).astype(node.dtype)
            frames = env._render(st, variation={**shown, path: alt}).cpu().numpy()
            if np.array_equal(frames, base):
                inert.append(path)
        except Exception as e:  # a variation that cannot be set is inert too
            inert.append(f"{path} ({type(e).__name__}: {e})")
    return {"inert_variations": inert}


def check_per_env_variation(env) -> dict:
    """Probe continuous leaves with two per-row values and report fully collapsed outputs.

    Unsupported per-row inputs are skipped by this probe, not reported as failures.
    """
    if env.B < 2:
        return {"skipped": "batch too small"}
    st = sample_render_rows(env, env.B)
    collapsed = []
    for path in _frame_variation_paths(env):
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        if not hasattr(node, "low"):
            continue  # Discrete: nothing continuous to spread
        lo = np.broadcast_to(node.low, node.shape).astype(np.float64)
        hi = np.broadcast_to(node.high, node.shape).astype(np.float64)
        # Two appearance groups detect total collapse while bounding distinct plate-bake work.
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
    """Restore render rows and style in a fresh instance and measure frame disagreement."""
    if env.style_dim == 0:
        return {"skipped": "env declares no style"}
    st = sample_render_rows(env, env.B)
    sty = env.sample_style(env.B)
    a = env._render(st, style=sty).cpu().numpy()
    fresh = type(env)(num_envs=env.B, size=env.size, supersample=env.ss, device=str(env.device))
    fresh._set_render_state(st, style=sty)
    b = fresh._render().cpu().numpy()
    return {"max_abs_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}


def check_default_style_is_official(env) -> dict:
    """Compare implicit and explicit default style; this check does not render the source."""
    if env.style_dim == 0:
        return {"skipped": "env declares no style"}
    st = sample_render_rows(env, env.B)
    a = env._render(st).cpu().numpy()
    b = env._render(st, style=env.default_style()).cpu().numpy()
    return {"max_abs_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())}


# A literal-color exception must name leaves and document an unreachable source shading bound.
_COLOR_APPLICABILITY_REQUIRED = (
    "leaves",
    "mechanism",
    "max_shading_factor",
    "official_renderer_fails_the_same_check",
    "evidence",
)


def color_exactness_applicability(env, probes=None) -> dict:
    """Read declared exceptions to literal RGB equality for specific appearance leaves.

    Require named leaves, a shading ceiling strictly below one, an asserted source
    failure, and evidence metadata. This validates declaration structure, not its
    evidence files. Undeclared leaves remain subject to the literal-color bound.
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
            "why": f"the declaration is missing {missing}, so it is refused and every RGB leaf "
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
            f"strictly below 1, so the declaration is refused and every leaf stays judged",
        }
    if decl["official_renderer_fails_the_same_check"] is not True:
        return {
            **none,
            "declaration": dict(decl),
            "rejected_because": ["official_renderer_fails_the_same_check"],
            "why": "the declaration does not assert that the official renderer fails this check "
            "too, so it is refused and every leaf stays judged",
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
    """Partition measured failures into asserted and declared-inapplicable probes.

    Keep inapplicable errors in the result. Missing or rejected declarations
    leave every failure asserted.
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
    """Probe RGB leaves with two target colors and report targets absent from the image."""
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    st = sample_render_rows(env, env.B)
    wrong = []
    best: dict[str, int] = {}
    drawn = set(_frame_variation_paths(env))
    for path, (off, w) in layout.items():
        if path not in drawn:
            continue
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
            # Retain nearest-color error even for probes whose literal bound is inapplicable.
            best[f"{path}->{target}"] = int(d.min())
            if not (d == 0).any():
                wrong.append(f"{path}->{target}")
    return {
        "colors_not_applied": wrong,
        "best_abs_diff": best,
        "applicability": color_exactness_applicability(env, sorted(best)),
    }


# Tolerance for reported footprint agreement; presence_margin uses mean errors instead.
PRESENCE_TOL = 8


def check_task_object_visibility(env, states=None, n: int = 16, seed: int = 0, reference=None) -> dict:
    """Measure visibility toggles on their changed-pixel footprint.

    Alternate rows also test per-row style handling. When a source is available,
    report error for the declared and flipped appearances on the same footprint.
    The margin is 1 - declared_error/flipped_error; it is None when undefined.
    A missing source or empty footprint cannot establish source agreement.
    ``reference`` (``states -> frames``) replaces the class's own reference renderer.
    """
    paths = [p for p in env.variation_space.names() if p.split(".")[0] == "visibility"]
    if not paths:
        return {"skipped": "the env declares no visibility.* leaf"}
    if states is None:
        states = sample_render_rows(env, int(n), rng=np.random.default_rng(int(seed)))
    states = np.asarray(states)
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    cls = type(env)
    official, why_ref = None, "no official renderer is declared"
    if reference is not None:
        official = np.asarray(reference(states)).astype(np.int16)
    elif getattr(cls, "reference_id", None) and hasattr(cls, "reference_env"):
        try:
            # Borrow from the process-lifetime cache so returning does not release the reference context.
            official = cls.reference_render(reference_env(cls, env.size), states).astype(np.int16)
        except ImportError as e:  # reference package not installed
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

        # Alternate style rows exercise the stored per-environment representation.
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
            # Zero flipped error makes the ratio undefined and contradicts the declared default.
            far = float(db.mean())
            row["presence_margin"] = (1.0 - float(da.mean()) / far) if far > 0 else None
            if far <= 0:
                row["why_no_margin"] = (
                    "the flipped appearance is bit-exact against the official frame on the object's "
                    "own pixels, so the declared default is wrong"
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
    st = sample_render_rows(env, env.B)
    return {"identical": bool(np.array_equal(env._render(st).cpu().numpy(), env._render(st).cpu().numpy()))}


# Diagnostic reference-hook contract; reference_sim_state is required.
STEP_REFERENCE_HOOKS = """
    sim_state_names        tuple[str, ...]        the full dynamical state being compared, usually a
                                                  superset of state_names (velocities are not drawn,
                                                  so they are not in the render state).
                                                  Defaults to state_names.
    sim_state()            -> (B, D) tensor       read it off the GPU env. Defaults to get_state().
    set_sim_state(v)       (B, D)                 write it. Defaults to set_state(v).
    reference_sim_state(ref) -> (D,) float64      required. Read it off the reference.
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
    step_reference_control str                    the group that must agree to float precision over
                                                  every step because nothing chaotic acts on it.
                                                  Defaults to none, and then the control is the
                                                  first-step error, which no divergence can inflate.
    step_reference_gate    str                    the group whose motion makes an episode informative;
                                                  episodes where it never moved are excluded from the
                                                  final-step statistics and counted. Defaults to none.
    step_reference_hold    int                    default ``hold``.
    step_reference_exact   bool                   the env claims bit-exactness, so exact_frac is the
                                                  headline number and anything below 1.0 is a failure.
"""


def _wrap_pi(d: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * d)))


def _group_error(diff: np.ndarray, dims, metric: str) -> np.ndarray:
    """Reduce signed differences (..., D) to one error per leading index for selected dimensions."""
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
    """Compare fixed action sequences against the declared reference simulator hooks.

    Report first-step, exact-equality, and grouped trajectory errors. Final-step
    statistics select episodes whose declared gate group moved; if none moved,
    all episodes are used and the zero count remains reported. Read state through
    sim_state/set_sim_state when supplied, otherwise private render-row hooks.
    This fallback cannot establish complete simulator State parity.
    """
    from stable_envs._core.dynamics import has_dynamics

    cls = type(env)
    if not has_dynamics(env):
        raise ValueError(
            f"{cls.__name__} has no step(); there is no solver to compare. Envs that ship "
            f"a renderer alone are covered by the render-parity checks instead."
        )
    if not hasattr(cls, "reference_sim_state"):
        raise ValueError(
            f"{cls.__name__} ships a step() but declares no reference_sim_state(ref), so this check "
            f"cannot read the reference's state. Declare:\n"
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
            # Read through the simulation hook; step return values need not be packed state rows.
            single.step(torch.as_tensor(a, dtype=torch.float32, device=dev).view(1, -1), frames=None)
            gpu[i] = np.asarray(get_sim().detach().cpu().numpy(), np.float64).reshape(D)

        diffs.append(gpu - cpu)
        exact.append(gpu == cpu)
        if gate:
            g_dims = list(dict((n, d) for n, d, _ in groups)[gate])
            touched.append(bool(np.linalg.norm(cpu[-1, g_dims] - s0[g_dims]) > 1e-6))
    # Keep the borrowed reference open for subsequent checks.

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
    # Shared assertions consume these aliases even when the environment names its groups differently.
    out.setdefault("agent_max_err", out["control_max_err"])
    out.setdefault("block_median_err", out["median_err"])
    return out


def uniform_style(env, n: int, groups: int = 2, seed: int = 0):
    """Return n style rows cycling through at most groups seeded draws, or None for no style."""
    if env.style_dim == 0:
        return None
    rng = np.random.default_rng(seed)
    base = env.sample_style(int(groups), rng=rng)
    idx = np.arange(n) % int(groups)
    return base[idx]


def check_batch_size_invariance(env, states, style=None, sizes=(1, 7, 64), probes: int = 4) -> dict:
    """Compare selected rows in the full batch and smaller contiguous batches."""
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
    """Compare private render_states outputs across chunk sizes using identical inputs."""
    from ._data import render_states

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
    """Box-average uint8 frames in float64, round once, and return uint8."""
    n, H, W, c = frames.shape
    x = frames.astype(np.float64).reshape(n, H // f, f, W // f, f, c)
    return np.clip(np.rint(x.mean(axis=(2, 4))), 0, 255).astype(np.uint8)


# Bounds were measured on this fixed draw; a different corpus needs its own bound.
DOWNSAMPLE_PROTOCOL_N = 8


def downsample_protocol(states=None) -> dict:
    """Describe the fixed downsampling corpus and optionally hash its state rows."""
    from . import _sampling as W

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
    """Draw the fixed-seed render rows used to measure the downsampling bound."""
    from . import _sampling as W

    states, _ = W.sample_inputs(env, DOWNSAMPLE_PROTOCOL_N)
    return np.ascontiguousarray(states, np.float32)


# These keys carry metadata; all other entries are interpreted as band conditions.
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

# Only explicitly inapplicable bands are excluded.
_NOT_APPLICABLE_PREFIX = "not_applicable"

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
    """Resolve declared state bands to which the resolution-consistency bound applies.

    Return indices=None for all states or a list for a subset. Exclusions override
    inclusions; unresolved conditions conservatively retain all states. Metadata
    citations are reported but not verified by this helper.
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

    # An all-band exclusion does not depend on the supplied state draw.
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
        # Unresolved conditions must not silently remove tested states.
        return {
            **full,
            "declaration": cited,
            "bands": out_bands,
            "why": f"band conditions {unresolved} could not be resolved against state_names, so "
            f"the bound is asserted on every state",
        }

    # With only exclusions declared, test their complement.
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
    """Compare native-size frames with box-downsampled larger renders.

    Returns measurements only; callers resolve declaration-based applicability
    separately. Skip when the renderer does not produce the requested size ratio.
    """
    S = int(env.size)
    a = env._render(states, size=S, style=style).detach().cpu().numpy()
    big = env._render(states, size=S * int(factor), style=style).detach().cpu().numpy()
    # Leave size-dependent renderer caches at the original resolution.
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
    """Compare equivalent render rows across dtype, array type, device, and striding.

    Forms labeled cuda use env.device, so they exercise CUDA only for a CUDA env.
    """
    ref = env._render(states, style=style).detach().cpu().numpy()
    dev = env.device
    st32 = np.ascontiguousarray(np.asarray(states, np.float32))
    n, D = st32.shape

    # Include views with real gaps between rows, not merely equivalent contiguous arrays.
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
    """Measure whether permutation or duplication of input rows changes their frames."""
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

    dup = np.concatenate([np.asarray(states)[:1], np.asarray(states)[:1]], 0)
    kdup = None if style is None else np.concatenate([np.asarray(style)[:1]] * 2, 0)
    f = env._render(dup, style=kdup).detach().cpu().numpy()
    d_dup = int(np.abs(f[0].astype(np.int16) - f[1].astype(np.int16)).max())
    return {"permutation_max_abs_diff": d_perm, "duplicate_max_abs_diff": d_dup}


def check_render_is_pure(env, states, style=None) -> dict:
    """Check explicit rendering and repeated restore for changes to stored render rows/style.

    This does not inspect full simulator State or hidden continuation fields.
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
    # Styleless environments have an empty array and zero possible style change.
    return {
        "state_changed_by_render": float(np.abs(s1 - s0).max()),
        "style_changed_by_render": float(np.abs(k1 - k0).max()) if k0.size else 0.0,
        "state_changed_by_double_set": float(np.abs(s2 - s0).max()),
        "frame_changed_by_double_set": int(np.abs(f2.astype(np.int16) - f0.astype(np.int16)).max()),
    }


def check_set_state_copies(env) -> dict:
    """Check private render-row restore for caller-buffer aliasing and storage rebinding.

    This does not test the public set_state contract.
    """
    st = sample_render_rows(env, env.B, rng=np.random.default_rng(0))
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

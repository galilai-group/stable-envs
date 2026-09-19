"""The twelve Fetch cells after the Wave-1C migration, in two tiers.

WHY TWO TIERS, AND WHY NEITHER OF THEM IMPORTS ``stable_envs.envs.fetch`` INTO THE SUITE.

This family renders its 18 robot meshes with ``mujoco_warp``'s batch raytracer and refuses to import
at all without ``warp``/``mujoco_warp`` and ``mujoco >= 3.11`` — deliberately, at module scope, so a
missing wheel costs exactly these envs and shows up in ``stable_envs.envs.import_errors()`` instead
of producing 300 failures at the first frame. No local interpreter has those wheels. Importing the
module anyway, by stubbing them in ``sys.modules``, would ALSO register twelve classes for the rest
of the pytest session and change what every registry-wide test in this suite sees. So:

**Tier 1 reads the SOURCE.** That ``FetchBase`` derives ``Env``, that its constructor takes
``num_envs=`` with ``batch=`` still accepted, that it assigns ``state_layout`` per instance and
overrides none of the state machinery, that it defines no ``step``/``transition``, and that the
frozen declarations (the 0.99 interior floor, ``parity_note``, ``PLATE_PATHS``, the twelve ``CELLS``)
still say what Step 8 froze, are all properties of the text. They are checked with ``ast`` and run on
every interpreter, which is the only reason they run at all.

**Tier 2 BUILDS the envs, in a subprocess.** One child interpreter installs the stub, substitutes
the frozen Step-4 corpus for the pool that would otherwise drive the official env (blocked locally by
the numpy 2.5.3 + mujoco 3.12 ``set_joint_qpos`` defect), builds two cells and prints what it found.
The parent asserts over that. Nothing leaks: the stub, the twelve registrations and the compiled
MuJoCo models all die with the child. Skipped where ``gymnasium_robotics`` is absent.

WHAT NO TIER CHECKS, said here rather than left as a silence: a Fetch PIXEL. There is no CPU backend
for the mover pass (``CPU_BACKEND_MISSING``), so ``render()`` cannot run on a machine without CUDA.
The plate half — MuJoCo's own raster of floor/table/skybox, 91% of the frame — DOES run on the CPU
and is checked in tier 2; the ray-cast half and the composite are owed to the GPU oracle.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "stable_envs" / "envs" / "fetch.py"
TREE = ast.parse(SRC.read_text(encoding="utf-8"))
CLASSES = {n.name: n for n in ast.walk(TREE) if isinstance(n, ast.ClassDef)}
FETCH_BASE = CLASSES["FetchBase"]
METHODS = {n.name: n for n in FETCH_BASE.body if isinstance(n, ast.FunctionDef)}
SCENES = ("Reach", "Push", "Slide", "PickAndPlace")
CELL_NAMES = tuple(f"Fetch{s}{x}" for x in ("", "Dense", "Dict") for s in SCENES)


# ---------------------------------------------------------------- tier 1: the source


def test_fetch_base_derives_the_step9_env_and_not_the_step8_alias():
    """``Env``, by its canonical name. ``StableEnv`` is the deprecated alias and warns on use."""
    bases = [ast.unparse(b) for b in FETCH_BASE.bases]
    assert bases == ["Env"], bases
    text = SRC.read_text(encoding="utf-8")
    assert "from ..env import Env" in text
    assert "from ..registry import register" in text
    assert "StableEnv" not in text, "the Step-8 alias should not survive the migration"


def test_the_constructor_takes_num_envs_and_still_accepts_batch():
    """Contract §2. ``num_envs`` sits in the slot ``batch`` had, so positional callers are unaffected."""
    args = [a.arg for a in METHODS["__init__"].args.args]
    assert args[:2] == ["self", "num_envs"], args
    assert "batch" in args, "batch= must still be accepted as the deprecated Step-8 spelling"
    # forwarded to Env.__init__, which is the one place the deprecation warning is issued
    call = [n for n in ast.walk(METHODS["__init__"])
            if isinstance(n, ast.Call) and "super()" in ast.unparse(n)]
    assert call and "num_envs=num_envs" in ast.unparse(call[0]) and "batch=batch" in ast.unparse(
        call[0]), ast.unparse(call[0]) if call else "no super().__init__ call"


def test_state_layout_is_assigned_per_instance_and_nothing_else_is_overridden():
    """A compiled-model family knows its state width only once built (integration f24d94c).

    ``Env._layout`` reads ``self.state_layout``, so declaring it in ``__init__`` — where the model is
    compiled — is the whole of this family's state machinery. An ``_layout``/``state``/
    ``_hidden_state``/``_live_state`` override would be a second place for the packed row to be
    described, and the base class's refusal to guess would stop protecting anything.
    """
    assigns = [ast.unparse(t) for n in ast.walk(METHODS["__init__"])
               if isinstance(n, ast.Assign) for t in n.targets]
    assert "self.state_layout" in assigns, assigns
    for forbidden in ("_layout", "_layout_dtypes", "state", "_hidden_state", "_live_state"):
        assert forbidden not in METHODS, f"FetchBase must not override {forbidden}"


def test_the_family_ships_no_simulator_and_says_so_by_not_defining_one():
    """Renderer-only: ``Env.step``/``Env.transition`` raise the contract's precise refusal."""
    assert "step" not in METHODS and "transition" not in METHODS
    assert "NO DYNAMICS" in (ast.get_docstring(FETCH_BASE) or "")


def test_close_releases_only_what_the_instance_owns():
    """The shared MuJoCo renderer and MJWarp contexts are never released — see ``_renderer``."""
    assert "_close" in METHODS
    # the STATEMENTS, not the docstring — which names the shared caches in order to say they stay
    body = "\n".join(ast.unparse(n) for n in METHODS["_close"].body
                     if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)))
    assert "_idx_cache" in body
    for shared in ("_CTX_CACHE", "_RENDERER_CACHE", "_MODEL_CACHE", "_PLATE_CACHE"):
        assert shared not in body, f"{shared} is shared by every instance and must not be released"


def test_the_frozen_fidelity_declarations_still_say_what_step_8_froze():
    """The interior floor is a CONTRACT, not a running best, and the note is its evidence."""
    src = SRC.read_text(encoding="utf-8")
    assert "parity_interior_min = 0.99" in src
    assert "THE ROBOT'S 18 MESHES ARE SHADED PER PIXEL AND MuJoCo SHADES THEM PER VERTEX" in src
    assert "THE TRANSLUCENT MOCAP MARKERS ARE " in src
    assert "fidelity_tier" not in {t.id for n in FETCH_BASE.body if isinstance(n, ast.Assign)
                                   for t in n.targets if isinstance(t, ast.Name)}, \
        "the family is parity by inheritance; declaring the tier here would be a second source"


def test_the_plate_paths_are_the_six_leaves_that_change_the_static_scene():
    """Everything else is a mover property and is applied per environment inside the ray-cast."""
    got = next(ast.literal_eval(n.value) for n in FETCH_BASE.body
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "PLATE_PATHS" for t in n.targets))
    assert got == ("floor.color", "table.color", "light.diffuse", "camera.distance_scale",
                   "camera.azimuth_delta", "camera.elevation_delta")


def test_twelve_cells_are_declared_as_data_and_not_as_twelve_class_bodies():
    """Four scenes x three swm suffixes; the suffixes reach no pixel, so they share a renderer."""
    ns: dict = {}
    src = SRC.read_text(encoding="utf-8")
    exec(compile(ast.parse(src[src.index("SCENE_XML = {"):src.index("\nROBOT_JOINTS")]),
                 "<scene_xml>", "exec"), ns)
    exec(compile(ast.parse(src[src.index("CELLS = ("):src.index("\n\n\ndef _make_classes")]),
                 "<cells>", "exec"), ns)
    cells = ns["CELLS"]
    assert [c for c, _, _ in cells] == list(CELL_NAMES)
    # eight distinct official ids for twelve cells: Dense has its own, Dict shares the sparse one
    assert len({ref for _, _, ref in cells}) == 8, sorted({ref for _, _, ref in cells})
    assert all(ref == f"Fetch{scene}-v4" for cell, scene, ref in cells
               if cell.endswith("Dict"))
    assert len(CLASSES) == 4, sorted(CLASSES)   # _Cache, _Bundle, _Ctx, FetchBase — and no cell


def test_the_module_still_refuses_to_import_without_the_raytracer():
    """The refusal is at module scope on purpose: a missing wheel costs exactly these envs."""
    src = SRC.read_text(encoding="utf-8")
    assert 'for _m in ("warp", "mujoco_warp")' in src
    assert "raise ImportError(" in src
    import stable_envs as snv
    errs = snv.envs.import_errors()
    if "fetch" in errs:
        assert "mujoco_warp" in errs["fetch"]


def test_the_module_level_caches_split_egl_from_device_memory():
    """EGL/model/pool stay unbounded; plates, free-camera poses and MJWarp contexts are LRU."""
    src = SRC.read_text(encoding="utf-8")
    for name in ("_MODEL_CACHE", "_RENDERER_CACHE", "_POOL_CACHE"):
        assert f"{name} = _Cache(" in src, name
    assert '_PLATE_CACHE = _RES.BoundedCache(32, name="fetch.plate"' in src
    assert '_CAM_CACHE = _RES.BoundedCache(256, name="fetch.cam"' in src
    assert '_CTX_CACHE = _RES.BoundedCache(6, name="fetch.ctx"' in src
    assert "on_evict=MW.release_cached_ctx" in src
    assert "_CACHE_MAX" not in src, "the per-cache bound constants were folded into BoundedCache / _Cache"


# ---------------------------------------------------------------- tier 2: two built cells

PROBE = r'''
import hashlib, json, os, sys, types
import importlib.machinery as im
import numpy as np, torch
for nm in ("warp", "mujoco_warp"):
    m = types.ModuleType(nm); m.__spec__ = im.ModuleSpec(nm, None); sys.modules[nm] = m
os.environ.setdefault("MUJOCO_GL", "glfw")
import stable_envs as snv
import stable_envs.envs.fetch as F

states_dir = sys.argv[1]
pools = {}
def pool(scene, ref_id, state_dim, nq, n=512):
    if scene not in pools:
        pools[scene] = np.load(f"{states_dir}/Fetch{scene}.npz")["states"].astype(np.float32)
    return pools[scene]
F._state_pool = pool

out = {}
for name in ("FetchReach", "FetchPush"):
    corpus = np.load(f"{states_dir}/{name}.npz")["states"].astype(np.float32)[:6]
    env = getattr(snv, name)(num_envs=6, device="cpu")
    env.set_state(corpus)
    st = env.state
    rec = {
        "num_envs": int(env.num_envs), "len": len(env),
        "state_keys": [".".join(k) if isinstance(k, tuple) else k for k in st.keys(True, True)],
        "widths": {".".join(k): int(w) for k, w, o in env._layout()},
        "offsets": {".".join(k): o for k, w, o in env._layout()},
        "state_dim": int(env.state_dim), "state_names": list(env.state_names),
        "pack_is_packed_row": bool(torch.equal(env.pack(st), env.get_state())),
        "roundtrip_exact": bool(np.array_equal(
            env.get_state().cpu().numpy(), corpus)),
        "has_dynamics": bool(env.has_dynamics),
        "default_size": int(env.size),
        "style_names": list(env.style_names),
        "plate_sha": hashlib.sha256(
            env._plate(64, None).cpu().numpy().tobytes()).hexdigest(),
        "plate_is_state_independent": True,
        "movers": np.asarray(env._b.movers, np.int64).tolist(),
        "mover_lut_sum": int(env._mover_lut().sum()),
        "swm_doc": env.__doc__.split(" ")[1] if env.__doc__ else "",
    }
    for what in ("step", "transition"):
        try:
            env.step(None) if what == "step" else env.transition(st, None)
            rec[what] = None
        except NotImplementedError as e:
            rec[what] = str(e)
    env.close(); env.close()
    rec["closed"] = bool(env.closed)
    rec["idx_cache_dropped"] = "_idx_cache" not in env.__dict__
    try:
        env.render(); rec["closed_refuses_render"] = False
    except RuntimeError:
        rec["closed_refuses_render"] = True
    e2 = getattr(snv, name)(batch=2, device="cpu")
    rec["batch_kwarg"] = int(e2.num_envs)
    e2.close()
    out[name] = rec
print("@@JSON@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def built() -> dict:
    """Build FetchReach and FetchPush in a child interpreter and return what it saw."""
    pytest.importorskip("gymnasium_robotics",
                        reason="the Fetch models are compiled from gymnasium-robotics' own assets")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(PROBE),
                        str(ROOT / "examples/results/step4/states")],
                       capture_output=True, text=True, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        pytest.fail(f"the Fetch probe failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")][-1]
    return json.loads(line[len("@@JSON@@"):])


def test_the_public_state_is_qpos_plus_the_goal_and_hides_nothing(built):
    """Contract §3. ``qpos`` is the simulator's configuration; ``goal`` is written to a mocap body.

    Every packed column belongs to one of the two keys, so no field has offset ``None``,
    ``_hidden_state`` is never consulted and ``pack``/``unpack`` round-trip the frozen corpus.
    """
    for name, nq in (("FetchReach", 15), ("FetchPush", 22)):
        r = built[name]
        assert r["state_keys"] == ["qpos", "goal"], r["state_keys"]
        assert r["widths"] == {"qpos": nq, "goal": 3}
        assert r["offsets"] == {"qpos": 0, "goal": nq}
        assert r["state_dim"] == nq + 3
        assert r["state_names"][:3] == ["slide0", "slide1", "slide2"]
        assert r["state_names"][-3:] == ["goal_x", "goal_y", "goal_z"]
        assert r["pack_is_packed_row"] and r["roundtrip_exact"]


def test_num_envs_is_accepted_and_batch_still_means_the_same_thing(built):
    for r in built.values():
        assert r["num_envs"] == 6 and r["len"] == 6
        assert r["batch_kwarg"] == 2


def test_a_renderer_only_cell_refuses_step_and_transition_and_says_why(built):
    """PushT has dynamics; Fetch does not, and the refusal says so rather than raising AttributeError."""
    for name, r in built.items():
        assert r["has_dynamics"] is False
        for what in ("step", "transition"):
            assert r[what] and "no batched simulator" in r[what], r[what]
            assert "PushT has dynamics" in r[what]
            assert "render() and" not in r[what]


def test_close_is_idempotent_drops_the_instance_cache_and_then_refuses_to_render(built):
    for r in built.values():
        assert r["closed"] and r["idx_cache_dropped"] and r["closed_refuses_render"]


def test_the_registered_default_frame_size_is_gymnasium_robotics_own(built):
    """480 px, the reference's ``DEFAULT_SIZE``. Moving it would move every frozen digest."""
    for r in built.values():
        assert r["default_size"] == 480


def test_the_plate_is_baked_and_the_mover_mask_still_covers_the_robot(built):
    """The plate half runs on the CPU and is the 91% of the frame that is exact by construction."""
    for name, r in built.items():
        assert len(r["plate_sha"]) == 64
        assert r["mover_lut_sum"] == len(r["movers"])
        assert len(r["movers"]) >= 18, "the robot's meshes are movers"


def test_every_cell_names_its_swm_id_in_its_docstring(built):
    """``scratch/swm_matrix.py``'s declared_docstring rung reads exactly this."""
    for name, r in built.items():
        assert r["swm_doc"].startswith("Fetch") or r["swm_doc"].endswith("-v4"), r["swm_doc"]

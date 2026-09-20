"""What the OGBench family promises after the package layout.

TWO HALVES, for the reason ``tests/unit/test_envs_dmcontrol.py`` splits the same way. The STRUCTURAL
half needs no MuJoCo, no ``ogbench`` and no GL context: it is about where the code lives, which names
resolve, which objects are shared and what the classes DECLARE, and all of that is answerable from an
import. The BEHAVIOURAL half constructs the 25 maze cells and renders them on the CPU ray-caster, so
it skips cleanly without ``ogbench``.

The two manipulation cells are CUDA-only by declaration, so nothing here renders them. What IS
asserted about them here is everything that does not need a frame: the CPU refusal and its message,
the declarations, the variation space (which needs only the cached bundle) and the shape of the
public state. Their frames are owed to the GPU regression oracle.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

import stable_envs as snv  # noqa: E402
from stable_envs import registry as R  # noqa: E402
from stable_envs.envs import ogbench as OGB  # noqa: E402
from stable_envs.envs.ogbench import manipulation as MANIP  # noqa: E402
from stable_envs.envs.ogbench import maze as MAZE  # noqa: E402

pytestmark = pytest.mark.unit

MAZE_CELLS = tuple(f"{lo.capitalize()}maze{mz.capitalize()}{ds.capitalize()}" for lo, mz, ds in MAZE.CELLS)
MANIP_CELLS = ("OGBCube", "OGBScene")

#: A representative maze cell per BODY, so the behavioural half exercises all three state shapes and
#: both forward-kinematics paths (the closed-form point one and ``mj_kinematics`` for the legged
#: bodies) without paying for 25 MuJoCo bakes in a unit run.
BODY_SAMPLE = ("PointmazeMediumNavigate", "AntmazeMediumExplore", "HumanoidmazeMediumNavigate")


def _needs_ogbench():
    pytest.importorskip("mujoco")
    pytest.importorskip("ogbench")


# ===================================================================== structural: where things live


def test_the_import_path_is_the_package_and_it_carries_the_step8_names():
    """``envs/ogbench.py`` became the PACKAGE ``envs/ogbench/``, so there is no second path to alias.

    A module and a package cannot share a name in one directory, so the obligation a shim would have
    met — every package-level name still resolving — is met by the package's re-exports. The
    oracle checks the FULL list, read off the baseline source; this pins the ones that the tests, the
    probes and the docs actually reach for, so a broken re-export fails in the unit run and not only
    in a 27-cell replay.
    """
    assert hasattr(OGB, "__path__"), "stable_envs.envs.ogbench must be the package"
    for name in (
        "OGBenchMaze",
        "BODY_STATE",
        "CELLS",
        "MAT_SPECULAR",
        "SHADE_AMBIENT",
        "SHADE_DIFFUSE",
        "SHADE_SPECULAR",
        "SHADE_EXPONENT",
        "_NOMINAL",
        "_FREE",
        "_ANT_JOINTS",
        "_HUMANOID_JOINTS",
        "_SCENE_CACHE",
        "_SCENE_CACHE_MAX",
        "_ENV_CACHE",
        "_Scene",
        "_offsets",
        "_mat_and_tex",
        "_build_reference",
        "_scene_env",
        "_make_classes",
    ):
        assert hasattr(OGB, name), f"the exported name {name} no longer resolves off the package"
    for cell in MAZE_CELLS:
        assert getattr(OGB, cell) is R.REGISTRY[cell], f"{cell} is not the registered class"


def test_the_manipulation_module_exposes_the_names_the_tests_reach_for():
    """Canonical path: ``stable_envs.envs.ogbench.manipulation``."""
    for name in (
        "OGBManipBase",
        "SPEC",
        "VISIBLE_GROUPS",
        "IDENTITY_QUAT",
        "MJW_CHUNK",
        "_bundle",
        "_reference_env",
        "_button_tables",
        "_ctx_button_tables",
        "_context",
        "_appearance_digest",
        "_plate_rig",
        "_PlateRig",
        "_movers",
        "_cube_geoms",
        "_grid_texture",
        "_variation_defaults",
        "mj_quantise",
        "_seeded_action_space",
        "_PLATE_CACHE",
        "_CTX_CACHE",
        "_BUNDLE_CACHE",
        "_PLATE_RIGS",
        "_KEEPALIVE",
    ):
        assert hasattr(MANIP, name), f"{name} no longer resolves off ogbench.manipulation"
    assert "zero_bits" in inspect.getsource(MANIP._context)


def test_the_module_caches_are_one_object_each_across_the_package():
    """The R3 lesson, applied before it bites: a re-export must SHARE, not copy.

    ``tests/unit/test_ogbench_scene_cache.py`` saves, clears and restores ``OGB._SCENE_CACHE`` and
    then drives ``OGBenchMaze._scene``, which reads the cache out of ``maze``'s own globals. That
    only works because the two names bind ONE cache. A re-export written as a copy of
    ``maze._SCENE_CACHE``
    would leave the test passing vacuously — it would isolate a cache nothing uses — so the identity
    is asserted rather than assumed.
    """
    assert OGB._SCENE_CACHE is MAZE._SCENE_CACHE
    assert OGB._ENV_CACHE is MAZE._ENV_CACHE
    sentinel = ("__identity_probe__",)
    OGB._SCENE_CACHE.put(sentinel, 1)
    try:
        assert MAZE._SCENE_CACHE.get(sentinel) == 1, "a write through the package was not shared"
    finally:
        OGB._SCENE_CACHE.pop(sentinel, None)


def test_every_cell_reports_its_canonical_module():
    """``registry.spec`` derives the FAMILY from ``__module__``."""
    for cell in MAZE_CELLS:
        cls = R.REGISTRY[cell]
        assert cls.__module__ == "stable_envs.envs.ogbench"
        assert R.spec(cell)["family"] == "ogbench"
    for cell in MANIP_CELLS:
        cls = R.REGISTRY[cell]
        assert cls.__module__ == "stable_envs.envs.ogbench.manipulation"
        assert R.spec(cell)["family"] == "manipulation"
        import importlib

        assert getattr(importlib.import_module(cls.__module__), cell) is cls
    assert set(R.list_envs(family="ogbench")) == set(MAZE_CELLS)
    assert set(R.list_envs(family="manipulation")) == set(MANIP_CELLS)


def test_a_maze_cell_is_a_declaration_and_overrides_no_renderer_method():
    """No inheritance hierarchy grew: 25 cells over ONE base, and each leaf declares and nothing more."""
    for cell in MAZE_CELLS:
        cls = R.REGISTRY[cell]
        assert cls.__bases__ == (MAZE.OGBenchMaze,)
        callables = [k for k, v in vars(cls).items() if callable(v) and not k.startswith("__")]
        assert not callables, f"{cell} defines {callables}; a cell is a declaration"
    for cell in MANIP_CELLS:
        assert R.REGISTRY[cell].__bases__ == (MANIP.OGBManipBase,)


# ============================================================ structural: the preserved declarations


@pytest.mark.parametrize("cell", MAZE_CELLS)
def test_the_maze_declarations_are_still_exactly_what_step_8_declared(cell):
    """``parity_interior_min`` is NOT relaxed anywhere, and the shipped knobs did not move.

    1.0 on every one of the 25 was measured over 25 cells x 64 states; a tolerance here would be
    declaring a limitation the evidence does not show. The agent-pixel residual is real and lives in
    ``parity_note``, which is compared in full by the oracle and pinned for its key claims here.
    """
    cls = R.REGISTRY[cell]
    assert cls.parity_interior_min == 1.0, "the interior floor must not be relaxed"
    assert cls.fidelity_tier == "parity"
    assert cls.default_size == 224
    assert cls.default_supersample == 2, "the supersample every parity number was measured at"
    assert cls.TILE == 32
    assert cls.PLATE_PATHS == (
        "floor.tint",
        "walls.color",
        "camera.distance_scale",
        "camera.elevation",
        "rendering.decolor_floor",
    )
    assert "AGENT-PIXEL PARITY" in cls.parity_note
    assert cls.reference_id == f"ogbench/{cls.LOCO}maze-{cls.MAZE}-v0"


def test_the_fitted_shading_law_constants_did_not_move():
    """The four numbers the whole agent-pixel claim rests on, and the per-light split beside them."""
    assert (MAZE.SHADE_AMBIENT, MAZE.SHADE_DIFFUSE, MAZE.SHADE_SPECULAR, MAZE.SHADE_EXPONENT) == (
        0.078438,
        1.416189,
        0.230554,
        45.0,
    )
    assert MAZE._NOMINAL == dict(amb=(0.1, 0.0), dif=(0.4, 1.0), spec=(0.5, 0.1))
    assert MAZE.MAT_SPECULAR == 0.5


@pytest.mark.parametrize("cell", MANIP_CELLS)
def test_the_manipulation_declarations_are_still_the_measured_visual_ones(cell):
    """A visual cell's numbers AND its thresholds, and the note that says what the tier does not buy."""
    cls = R.REGISTRY[cell]
    assert cls.fidelity_tier == "visual"
    assert cls.parity_interior_min is None
    vf = cls.visual_fidelity
    for k in (
        "mean_abs",
        "within8_frac",
        "ssim",
        "mean_abs_max",
        "within8_min",
        "ssim_min",
        "measured_at",
        "reference_versions",
        "measured_with",
        "interior_frac",
        "interior_exact_frac",
        "all_exact_frac",
        "draw_spread",
    ):
        assert k in vf, f"{cell}: visual_fidelity is missing {k}"
    assert vf["mean_abs_max"] > vf["mean_abs"], "a ceiling must sit above the measurement"
    assert vf["within8_min"] < vf["within8_frac"] and vf["ssim_min"] < vf["ssim"]
    assert vf["reference_versions"] == {"mujoco": "3.12.0", "ogbench": "1.2.1"}
    assert vf["measured_with"] == {"mujoco_warp": "3.12.0"}
    assert "THREE declared residuals" in cls.visual_note
    assert cls.default_size == MANIP.SPEC[cls.CELL]["size"]
    assert "VISUAL TIER" in (cls.__doc__ or "").upper()


def test_the_scene_carries_the_button_texture_bake_note_and_the_cube_does_not():
    """The Scene's fourth residual, declared per geom because that is what the bake record is for."""
    assert "btn_metal" in R.REGISTRY["OGBScene"].visual_note
    assert "SCENE ONLY" in R.REGISTRY["OGBScene"].visual_note
    assert "btn_metal" not in R.REGISTRY["OGBCube"].visual_note


@pytest.mark.parametrize("cell", MANIP_CELLS)
def test_cpu_is_refused_rather_than_served_from_the_gpu(cell):
    """The honest-device-selection rule, with the measurement that justifies it in the message.

    Checked BEFORE anything is built, which is why this runs on a machine with no ``ogbench``: a
    caller who asked for a CPU env gets the reason rather than four seconds of construction.
    """
    with pytest.raises(ValueError, match="CPU3D-OGB-1") as e:
        R.REGISTRY[cell](num_envs=1, device="cpu")
    msg = str(e.value)
    assert "no CPU backend" in msg and "Use device='cuda'" in msg
    assert f"{MANIP.SPEC[cell.replace('OGB', '')]['visible_tri']:,}" in msg, (
        "the refusal must quote THIS cell's triangle count, not its sibling's"
    )
    with pytest.raises(ValueError, match="CPU3D-OGB-1"):
        snv.make(cell, num_envs=1, device="cpu")


# =========================================================== structural: the shared texel substitution


def test_the_plate_texel_helpers_are_the_shared_module_and_not_a_second_copy():
    """The plate texel helpers were lifted out of this family for ``dmcontrol`` to share.

    The copy is gone. ``mj_quantise`` is asserted to BE ``_plate_texels.quantise`` — the same
    function object, so the truncation rule cannot drift — and the two call sites are asserted to
    call the shared functions, because a comment saying they do is not enforcement. The behaviour
    those functions carry is unchanged and is checked by ``tests/unit/test_plate_texels.py``.
    """
    from stable_envs.envs import _plate_texels as PX

    assert MANIP.mj_quantise is PX.quantise
    assert "_texel_slots(" in inspect.getsource(MANIP._grid_texture)
    assert "_substitute(" in inspect.getsource(MANIP.OGBManipBase._apply_floor)
    assert "_texel_key(" in inspect.getsource(MANIP.OGBManipBase._floor_key)
    # The GL-context-current fix  must still be on the path a bake takes. It lives in
    # ``substitute`` now; a version of it that forgot the line would put back a 219/255 plate.
    body = inspect.getsource(PX.substitute).split('"""')[-1]  # the code, not the prose above it
    assert "make_current()" in body and "mjr_uploadTexture" in body
    assert body.index("make_current()") < body.index("mjr_uploadTexture"), (
        "the upload must be made under the renderer's OWN context"
    )


def test_the_manipulation_plate_is_grouped_by_floor_colour_and_nothing_else():
    assert MANIP.OGBManipBase.PLATE_PATHS == ("floor.color",)
    assert MANIP.MJW_CHUNK == 256


# =============================================================== structural: the public State layout


@pytest.mark.parametrize("loco,dim", [("point", 2), ("ant", 15), ("humanoid", 28)])
def test_the_maze_state_layout_covers_the_packed_row_and_hides_nothing(loco, dim):
    """The public State is the BODY's own structure and the packed row is untouched.

    ``qpos`` for a point is two slide joints; for the legged bodies it is a free joint (3 position +
    4 quaternion) followed by hinges. Every offset is a real column, so ``pack`` drops nothing,
    ``unpack`` invents nothing and every frozen Step-4 digest is computed from the same bytes.
    """
    layout = MAZE.OGBenchMaze.STATE_LAYOUT[loco]
    assert sum(w for _, w, o in layout if o is not None) == dim == len(MAZE.BODY_STATE[loco])
    assert all(o is not None for _, _, o in layout), "these cells have no hidden dynamical state"
    off = 0
    for key, w, o in layout:
        assert o == off, f"{loco}: {key} starts at {o}, not at {off}"
        off += w
    assert layout[0][0] == ("root", "position")
    if loco != "point":
        assert [k for k, _, _ in layout] == [("root", "position"), ("root", "orientation"), ("joints",)]


def test_the_maze_variation_taxonomy_is_the_step8_one():
    """The appearance knobs, leaf for leaf, including the one whose SWM counterpart is unsupported.

    ``rendering.decolor_floor`` is the OPPOSITE-SENSE twin of swm's ``floor.pixel_encoding``, which
    the VariationMap records as UNSUPPORTED-PENDING because an inverting relation has no
    representation there. That taxonomy is the thing this workstream must not move, so it is pinned
    here as well as compared by the oracle.
    """
    for loco in ("point", "ant", "humanoid"):
        space = MAZE.OGBenchMaze.make_variation_space(types.SimpleNamespace(LOCO=loco, MAZE="medium"))
        # ``names()`` is sorted, which is the packed-row order the oracle compares against the baseline.
        assert list(space.names()) == [
            "agent.color",
            "agent.scale",
            "camera.distance_scale",
            "camera.elevation",
            "floor.tint",
            "rendering.decolor_floor",
            "walls.color",
        ]
        assert np.allclose(space["agent"]["color"].value, [0.7, 0.5, 0.3])
        assert np.allclose(space["walls"]["color"].value, [0.6, 0.6, 0.6])
        assert np.allclose(space["camera"]["elevation"].value, [-90.0])
        assert int(np.asarray(space["rendering"]["decolor_floor"].value).reshape(-1)[0]) == 0


def test_the_manipulation_variation_taxonomy_is_the_two_mapped_leaves():
    """``cube.color`` and ``floor.color``, in the shapes the boundary was decided at.

    Needs the cached bundle (mujoco + ogbench + swm on the CPU) because the defaults are read off
    swm's own space rather than transcribed, which is what makes "our default IS the registered
    default" true by construction.
    """
    pytest.importorskip("stable_worldmodel")
    _needs_ogbench()
    for cell in ("Cube", "Scene"):
        space = MANIP.OGBManipBase.make_variation_space(types.SimpleNamespace(CELL=cell))
        assert list(space.names()) == ["cube.color", "floor.color"]
        assert space["cube"]["color"].shape == (3,)
        assert space["floor"]["color"].shape == (2, 3)


# ===================================================================== behavioural: the maze cells


@pytest.fixture(scope="module")
def _ogbench():
    _needs_ogbench()


@pytest.mark.parametrize("cell", BODY_SAMPLE)
def test_takes_the_step9_constructor_and_the_lifecycle(_ogbench, cell):
    env = R.REGISTRY[cell](num_envs=3, device="cpu")
    try:
        assert env.num_envs == 3 and len(env) == 3
        assert env.size == 224 and env.ss == 2
    finally:
        env.close()
    env.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        env.render()
    with R.REGISTRY[cell](num_envs=2, device="cpu") as e:
        assert e.num_envs == 2
    assert e.closed


@pytest.mark.parametrize("cell", BODY_SAMPLE)
def test_the_public_state_is_the_bodys_own_structure_and_packs_back_to_the_frozen_row(_ogbench, cell):
    with R.REGISTRY[cell](num_envs=4, device="cpu") as env:
        rows = torch.as_tensor(env.sample_states(4, np.random.default_rng(0)), device=env.device)
        env.set_state(rows)
        st = env.state
        keys = [".".join(k) for k, _, _ in env._layout()]
        assert keys[0] == "root.position"
        assert torch.equal(env.pack(st), env.get_state()), "pack(state) must BE the packed row"
        assert torch.equal(env.pack(env.unpack(env.get_state())), env.get_state())
        assert env._hidden_state() == {} and env._live_state() == {}
        # the keys name the packed columns in order, which is what makes the grouping a VIEW
        names, i = list(env.state_names), 0
        for key, w, _ in env._layout():
            assert env.state_names_of(key) == names[i : i + w]
            i += w


@pytest.mark.parametrize("cell", BODY_SAMPLE)
def test_a_renderer_only_cell_refuses_step_and_transition_and_says_why(_ogbench, cell):
    with R.REGISTRY[cell](num_envs=2, device="cpu") as env:
        assert env.has_dynamics is False
        st = env.reset(seed=0)
        for call, args in (("step", (torch.zeros(2, 1),)), ("transition", (st, torch.zeros(2, 1)))):
            with pytest.raises(NotImplementedError) as e:
                getattr(env, call)(*args)
            msg = str(e.value)
            assert cell in msg and "no batched simulator" in msg
            assert "env.render(states)" in msg, "the refusal must say what DOES work"
        assert env.render().shape == (2, 224, 224, 3)  # reset and render still work


@pytest.mark.parametrize("cell", BODY_SAMPLE)
def test_render_takes_a_state_a_packed_row_or_nothing(_ogbench, cell):
    with R.REGISTRY[cell](num_envs=2, device="cpu") as env:
        env.set_state(env.sample_states(2, np.random.default_rng(0)))
        st = env.state
        a, b, c = env.render(st), env.render(env.pack(st)), env.render()
        assert torch.equal(a, b) and torch.equal(a, c)
        assert a.shape == (2, 224, 224, 3) and a.dtype == torch.uint8
        assert env.observation_spec.shape == (224, 224, 3)


def test_the_dataset_siblings_of_one_scene_draw_the_same_frame(_ogbench):
    """The fact the plate cache is keyed on: ``navigate``/``stitch`` select a dataset, not a scene.

    It is also why the oracle replays fourteen cells on a sibling's frozen corpus — there is nothing
    else to replay, because the cells ARE the same renderer.
    """
    states = None
    digests = []
    for cell in ("PointmazeMediumNavigate", "PointmazeMediumStitch"):
        with R.REGISTRY[cell](num_envs=2, device="cpu") as env:
            if states is None:
                states = env.sample_states(2, np.random.default_rng(0))
            digests.append(env.render(torch.as_tensor(states)).cpu().numpy().tobytes())
    assert digests[0] == digests[1]

"""OGBench family: registration, declarations and the 25 maze cells on the CPU ray-caster.

The structural half needs no MuJoCo or ogbench. The behavioural half constructs maze cells and
skips without ``ogbench``. The manipulation cells are CUDA-only, so only their CPU refusal,
declarations and variation space are checked here.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from stable_envs import registry as R  # noqa: E402
from stable_envs.envs import ogbench as OGB  # noqa: E402
from stable_envs.envs.ogbench import manipulation as MANIP  # noqa: E402
from stable_envs.envs.ogbench import maze as MAZE  # noqa: E402

pytestmark = pytest.mark.unit

MAZE_CELLS = tuple(f"{lo.capitalize()}Maze{mz.capitalize()}{ds.capitalize()}" for lo, mz, ds in MAZE.CELLS)
MANIP_CELLS = ("OGBCube", "OGBScene")

#: A representative maze cell per BODY, so the behavioural half exercises all three state shapes and
#: both forward-kinematics paths (the closed-form point one and ``mj_kinematics`` for the legged
#: bodies) without paying for 25 MuJoCo bakes in a unit run.
BODY_SAMPLE = ("PointMazeMediumNavigate", "AntMazeMediumExplore", "HumanoidMazeMediumNavigate")


def _needs_ogbench():
    pytest.importorskip("mujoco")
    pytest.importorskip("ogbench")


# ===================================================================== structural: where things live


def test_maze_cells_are_reexported_from_the_package_as_the_registered_classes():
    assert hasattr(OGB, "__path__"), "stable_envs.envs.ogbench must be the package"
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
    assert "zero_bits" in inspect.getsource(MANIP._build_context)


def test_every_cell_reports_its_canonical_module():
    """Family is an explicit class attribute, not a module-path fragment."""
    for cell in MAZE_CELLS:
        cls = R.REGISTRY[cell]
        assert cls.__module__ == "stable_envs.envs.ogbench"
        meta = R.info(cell)
        assert meta["family"] == "ogbench"
        assert meta["domain"] == cls.LOCO + "maze"
    for cell in MANIP_CELLS:
        if cell not in R.REGISTRY:
            continue
        cls = R.REGISTRY[cell]
        assert cls.__module__ == "stable_envs.envs.ogbench.manipulation_semantics"
        meta = R.info(cell)
        assert meta["family"] == "ogbench"
        assert meta["domain"] == cls.CELL.lower()
        import importlib

        assert getattr(importlib.import_module(cls.__module__), cell) is cls
    assert set(MAZE_CELLS) <= set(R.list_envs(family="ogbench"))
    assert set(R.list_envs(family="ogbench", domain="pointmaze")) <= set(MAZE_CELLS)
    assert set(R.list_envs(family="manipulation")) == set()


def test_a_maze_cell_is_a_declaration_and_overrides_no_renderer_method():
    """Every cell composes the simulator over the renderer; only point/large/navigate swaps the scene."""
    for cell in MAZE_CELLS:
        cls = R.REGISTRY[cell]
        expected = (MAZE.MazeSemantics, MAZE.OGBenchMaze)
        if cell == "PointMazeLargeNavigate":
            expected = (MAZE._StatesScene, *expected)
        assert cls.__bases__ == expected
        callables = [k for k, v in vars(cls).items() if callable(v) and not k.startswith("__")]
        assert not callables, f"{cell} defines {callables}; a cell is a declaration"
    for cell in MANIP_CELLS:
        assert R.REGISTRY[cell].__bases__[1] is MANIP.CELL_CLASSES[cell]
        assert MANIP.CELL_CLASSES[cell].__bases__ == (MANIP.OGBManipBase,)


# ============================================================ structural: the preserved declarations


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
    assert "can differ from MuJoCo" in cls.visual_note and "visual_fidelity records" in cls.visual_note
    assert cls.default_size == MANIP.SPEC[cls.CELL]["size"]
    assert "VISUAL TIER" in (cls.__doc__ or "").upper()


def test_the_scene_carries_the_button_texture_bake_note_and_the_cube_does_not():
    """Only the Scene has UV-less button textures, so only its note declares their substitution."""
    assert "button textures" in R.REGISTRY["OGBScene"].visual_note
    assert "button textures" not in R.REGISTRY["OGBCube"].visual_note


@pytest.mark.parametrize("cell", MANIP_CELLS)
def test_frozen_renderer_cpu_is_refused_rather_than_served_from_the_gpu(cell):
    """The honest-device-selection rule, with the measurement that justifies it in the message.

    Checked BEFORE anything is built, which is why this runs on a machine with no ``ogbench``: a
    caller who asked for a CPU env gets the reason rather than four seconds of construction.
    """
    with pytest.raises(ValueError, match="requires device='cuda'"):
        MANIP.CELL_CLASSES[cell](num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="requires device='cuda'"):
        MANIP.CELL_CLASSES[cell](num_envs=1, device="cpu")


# =========================================================== structural: the shared texel substitution


def test_the_plate_texel_helpers_are_the_shared_module_and_not_a_second_copy():
    """The plate texel helpers were lifted out of this family for ``dmcontrol`` to share.

    The copy is gone. ``mj_quantise`` is asserted to BE ``plate_texels.quantise`` — the same
    function object, so the truncation rule cannot drift — and the two call sites are asserted to
    call the shared functions, because a comment saying they do is not enforcement. The behaviour
    those functions carry is unchanged and is checked by ``tests/unit/test_plate_texels.py``.
    """
    from stable_envs._core import plate_texels as PX

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
def test_the_maze_render_layout_covers_the_packed_row(loco, dim):
    """The body projection tiles the packed row contiguously."""
    layout = MAZE.OGBenchMaze.STATE_LAYOUT[loco]
    assert sum(w for _, w, o in layout if o is not None) == dim == len(MAZE.BODY_STATE[loco])
    assert all(o is not None for _, _, o in layout), "every renderer column has an offset"
    off = 0
    for key, w, o in layout:
        assert o == off, f"{loco}: {key} starts at {o}, not at {off}"
        off += w
    assert layout[0][0] == ("root", "position")
    if loco != "point":
        assert [k for k, _, _ in layout] == [("root", "position"), ("root", "orientation"), ("joints",)]


def test_the_maze_variation_taxonomy():
    """The appearance leaves, with swm's floor, encoding and light leaves at our defaults."""
    for loco in ("point", "ant", "humanoid"):
        space = MAZE.OGBenchMaze.make_variation_space(types.SimpleNamespace(LOCO=loco, MAZE="medium"))
        assert list(space.names()) == [
            "agent.color",
            "agent.scale",
            "camera.distance_scale",
            "camera.elevation",
            "floor.color",
            "floor.pixel_encoding",
            "floor.tint",
            "light.intensity",
            "rendering.decolor_floor",
            "walls.color",
        ]
        assert np.allclose(space["agent"]["color"].value, [0.7, 0.5, 0.3])
        assert np.allclose(space["walls"]["color"].value, [0.6, 0.6, 0.6])
        assert np.allclose(space["camera"]["elevation"].value, [-90.0])
        assert int(np.asarray(space["rendering"]["decolor_floor"].value).reshape(-1)[0]) == 0
        assert int(np.asarray(space["floor"]["pixel_encoding"].value).reshape(-1)[0]) == 1
        assert np.allclose(space["floor"]["color"].value, MAZE.CHECKER_RGB)
        assert np.allclose(space["light"]["intensity"].value, [1.0])


def test_the_manipulation_variation_taxonomy_is_the_two_mapped_leaves():
    """``cube.color`` and ``floor.color``, in the shapes the boundary was decided at.

    Needs the cached bundle (mujoco + ogbench on the CPU) because the defaults are read off
    the reference variation space rather than transcribed, which is what makes "our default IS the
    registered default" true by construction.
    """
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
def test_takes_num_envs_and_closes_idempotently(_ogbench, cell):
    env = R.REGISTRY[cell](num_envs=3, device="cpu")
    try:
        assert env.num_envs == 3 and len(env) == 3
        assert env.size == 224 and env.ss == 2
    finally:
        env.close()
    env.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        env._render()
    with R.REGISTRY[cell](num_envs=2, device="cpu") as e:
        assert e.num_envs == 2
    assert e.closed


@pytest.mark.parametrize("cell", BODY_SAMPLE)
def test_the_private_projection_packs_back_to_the_frozen_row(_ogbench, cell):
    with R.REGISTRY[cell](num_envs=4, device="cpu") as env:
        rows = torch.as_tensor(env.sample_states(4, np.random.default_rng(0)), device=env.device)
        env._set_render_state(rows)
        st = env._unpack_render_state(env._get_render_state())
        keys = [".".join(k) for k, _, _ in env._layout()]
        assert keys[0] == "root.position"
        assert torch.equal(env._pack_render_state(st), env._get_render_state())
        assert env._hidden_state() == {} and env._live_state() == {}
        names, i = list(env.state_names), 0
        for key, w, _ in env._layout():
            assert env.state_names_of(key) == names[i : i + w]
            i += w


@pytest.mark.parametrize("cell", BODY_SAMPLE)
def test_a_cell_steps_its_body_and_renders_the_qpos_projection(_ogbench, cell):
    with R.REGISTRY[cell](num_envs=2, device="cpu") as env:
        assert env.has_dynamics
        state = env.reset(seed=3)
        out = env.step(torch.zeros(2, env.action_dim), frames=None)
        assert out["state"].batch_size == torch.Size([2])
        rows = env._pack_render_state(state)
        assert rows.shape == (2, len(MAZE.BODY_STATE[env.LOCO]))
        assert torch.equal(env.render(state), env._render(rows))


@pytest.mark.parametrize("cell", BODY_SAMPLE)
def test_private_render_takes_a_projection_a_packed_row_or_nothing(_ogbench, cell):
    with R.REGISTRY[cell](num_envs=2, device="cpu") as env:
        env._set_render_state(env.sample_states(2, np.random.default_rng(0)))
        st = env._unpack_render_state(env._get_render_state())
        a, b, c = env._render(st), env._render(env._pack_render_state(st)), env._render()
        assert torch.equal(a, b) and torch.equal(a, c)
        assert a.shape == (2, 224, 224, 3) and a.dtype == torch.uint8
        assert env.frame_spec.shape == (224, 224, 3)


def test_the_dataset_siblings_of_one_scene_draw_the_same_frame(_ogbench):
    """``navigate``/``stitch`` select a dataset, not a scene, so the plate cache is shared."""
    states = None
    digests = []
    for cell in ("PointMazeMediumNavigate", "PointMazeMediumStitch"):
        with R.REGISTRY[cell](num_envs=2, device="cpu") as env:
            if states is None:
                states = env.sample_states(2, np.random.default_rng(0))
            digests.append(env._render(torch.as_tensor(states)).cpu().numpy().tobytes())
    assert digests[0] == digests[1]


@pytest.mark.parametrize("cell", ["PointMazeLargeNavigate", "AntMazeMediumNavigate"])
def test_a_recoloured_floor_and_dimmed_light_leave_the_default_plate_unchanged(_ogbench, cell):
    """Recoloured checkers upload texels into a shared renderer; the default bake must not see them."""
    with R.REGISTRY[cell](num_envs=1, device="cpu") as env:
        default = env._bake(env.size, None).plate
        varied = {"floor.color": [0.9, 0.1, 0.1, 0.1, 0.1, 0.9], "floor.pixel_encoding": [0], "light.intensity": [0.4]}
        assert not torch.equal(env._bake(env.size, varied).plate, default)
        assert torch.equal(env._bake(env.size, None).plate, default)

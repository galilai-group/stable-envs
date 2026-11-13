"""OGBench family: registration, declarations and the 25 maze cells on the CPU ray-caster.

The structural half needs no MuJoCo or ogbench. The behavioral half constructs maze cells and
skips without ``ogbench``. The manipulation cells are CUDA-only, so only their CPU refusal,
declarations and variation space are checked here.
"""

from __future__ import annotations

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

#: A representative maze cell per BODY, so the behavioral half exercises all three state shapes and
#: both forward-kinematics paths (the closed-form point one and ``mj_kinematics`` for the legged
#: bodies) without paying for 25 MuJoCo bakes in a unit run.
BODY_SAMPLE = ("PointMazeMediumNavigate", "AntMazeMediumExplore", "HumanoidMazeMediumNavigate")


def _needs_ogbench():
    pytest.importorskip("mujoco")
    pytest.importorskip("ogbench")


def test_maze_cells_are_reexported_from_the_package_as_the_registered_classes():
    assert hasattr(OGB, "__path__"), "stable_envs.envs.ogbench must be the package"
    for cell in MAZE_CELLS:
        assert getattr(OGB, cell) is R.REGISTRY[cell], f"{cell} is not the registered class"


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


@pytest.mark.parametrize("cell", MANIP_CELLS)
def test_the_manipulation_declarations_are_still_the_measured_visual_ones(cell):
    """A manipulation cell declares how it was measured and its pinned reference versions."""
    cls = R.REGISTRY[cell]
    vf = cls.visual_fidelity
    for k in ("measured_at", "reference_versions", "measured_with"):
        assert k in vf, f"{cell}: visual_fidelity is missing {k}"
    assert vf["reference_versions"] == {"mujoco": "3.17.0", "ogbench": "1.2.1"}
    assert vf["measured_with"] == {"mujoco_warp": "3.17.0"}
    assert "can differ from" in cls.visual_note and "visual_fidelity" in cls.visual_note
    assert cls.default_size == MANIP.SPEC[cls.CELL]["size"]


@pytest.mark.parametrize("cell", MANIP_CELLS)
def test_frozen_renderer_cpu_is_refused_rather_than_served_from_the_gpu(cell):
    """The MJWarp renderer classes refuse a CPU device before building anything."""
    with pytest.raises(ValueError, match="requires device='cuda'"):
        MANIP.CELL_CLASSES[cell](num_envs=1, device="cpu")


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
        assert int(np.asarray(space["floor"]["pixel_encoding"].value).reshape(-1)[0]) == 0
        assert np.allclose(space["floor"]["color"].value, MAZE.CHECKER_RGB)
        assert np.allclose(space["light"]["intensity"].value, [1.0])


def test_the_manipulation_variation_taxonomy_is_the_two_mapped_leaves():
    """The MJWarp renderer's space is ``cube.color`` (3,) and ``floor.color`` (2, 3)."""
    _needs_ogbench()
    for cell in ("Cube", "Scene"):
        space = MANIP.OGBManipBase.make_variation_space(types.SimpleNamespace(CELL=cell))
        assert list(space.names()) == ["cube.color", "floor.color"]
        assert space["cube"]["color"].shape == (3,)
        assert space["floor"]["color"].shape == (2, 3)


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


def test_culled_wall_shadows_and_split_mover_tiles_keep_every_byte(_ogbench, monkeypatch):
    """The state-observation scene draws the same frames as with every wall and one mover tile."""
    from stable_envs.render import raycast3d as R3

    casts = []
    composite = R3.composite

    def counted(*args, **kwargs):
        casts.append(kwargs.get("cast"))
        return composite(*args, **kwargs)

    monkeypatch.setattr(R3, "composite", counted)
    with R.REGISTRY["PointMazeLargeNavigate"](num_envs=4, device="cpu") as env:
        rows = torch.as_tensor(env.sample_states(40, np.random.default_rng(1)))
        near = rows[:12].clone()
        near[:, 2:4] = near[:, 0:2] + torch.linspace(-1.5, 1.5, 12).unsqueeze(1)
        rows = torch.cat([rows, near])
        fast = env._render(rows)
        single = torch.cat([env._render(rows[i : i + 1]) for i in (0, 41, 47)])
        assert any(c is not None for c in casts) and any(c is None for c in casts)
        monkeypatch.setattr(MAZE, "_CULL_WALLS", False)
        monkeypatch.setattr(MAZE, "_SPLIT_MOVERS", False)
        full = env._render(rows)
    assert torch.equal(fast, full)
    assert torch.equal(single, full[[0, 41, 47]])


def test_the_numba_lit_pass_draws_the_bytes_of_the_torch_lit_pass(_ogbench, monkeypatch):
    """Groups of 16, 16 and 9 rows, with the goal near the body in some rows."""
    pytest.importorskip("numba")
    calls = []
    plates_cpu = MAZE._LightPasses.plates_cpu

    def counted(self, *args):
        calls.append(len(args[2]))
        return plates_cpu(self, *args)

    monkeypatch.setattr(MAZE._LightPasses, "plates_cpu", counted)
    with R.REGISTRY["PointMazeLargeNavigate"](num_envs=4, device="cpu") as env:
        rows = torch.as_tensor(env.sample_states(29, np.random.default_rng(2)))
        near = rows[:12].clone()
        near[:, 2:4] = near[:, 0:2] + torch.linspace(-1.5, 1.5, 12).unsqueeze(1)
        rows = torch.cat([rows, near])
        kernel = env._render(rows)
        assert calls == [41]
        monkeypatch.setattr(MAZE._LightPasses, "cpu_kernel", lambda self: False)
        reference = env._render(rows)
    assert calls == [41]
    assert torch.equal(kernel, reference)


@pytest.mark.nightly
@pytest.mark.parametrize("cell", ["PointMazeLargeNavigate", "AntMazeMediumNavigate"])
def test_a_recoloured_floor_and_dimmed_light_leave_the_default_plate_unchanged(_ogbench, cell):
    """Recolored checkers upload texels into a shared renderer; the default bake must not see them."""
    with R.REGISTRY[cell](num_envs=1, device="cpu") as env:
        default = env._bake(env.size, None).plate
        varied = {"floor.color": [0.9, 0.1, 0.1, 0.1, 0.1, 0.9], "floor.pixel_encoding": [0], "light.intensity": [0.4]}
        assert not torch.equal(env._bake(env.size, varied).plate, default)
        assert torch.equal(env._bake(env.size, None).plate, default)


def test_a_cpu_maze_step_advances_and_draws_a_small_unlit_batch_on_one_torch_thread(_ogbench, monkeypatch):
    """Idle torch threads would spin beside the mujoco.batch threads; the count returns after the step."""
    seen = {}
    with R.REGISTRY["AntMazeMediumExplore"](num_envs=4, device="cpu", size=32) as env:
        sim_step, draw = env._simulator.step, MAZE.R3.composite

        def step(*args, **kwargs):
            seen["step"] = torch.get_num_threads()
            return sim_step(*args, **kwargs)

        def composite(*args, **kwargs):
            seen.setdefault("render", []).append(torch.get_num_threads())
            return draw(*args, **kwargs)

        monkeypatch.setattr(env._simulator, "step", step)
        monkeypatch.setattr(MAZE.R3, "composite", composite)
        env.reset(seed=0)
        before = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            env.step(torch.zeros(4, 8), frames=slice(None))
            assert torch.get_num_threads() == 2
        finally:
            torch.set_num_threads(before)
    assert seen["step"] == 1
    assert seen["render"][-1] == 1


def test_cpu_maze_frames_and_states_do_not_depend_on_the_torch_thread_count(_ogbench, monkeypatch):
    def rollout(threads, scoped):
        if not scoped:
            monkeypatch.setattr(MAZE.MazeSemantics, "native_step_threads", False)
            monkeypatch.setattr(MAZE, "_CPU_THREADED_ROWS", 0)
        before = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            with R.REGISTRY["AntMazeMediumExplore"](num_envs=3, device="cpu", size=48) as env:
                env.reset(seed=3)
                actions = torch.linspace(-1, 1, 3 * 5 * 8).reshape(3, 5, 8)
                out = env.step(actions, frames=slice(None))
                return out["frames"], out["states"]
        finally:
            torch.set_num_threads(before)
            monkeypatch.undo()

    frames, states = rollout(1, scoped=True)
    other_frames, other_states = rollout(4, scoped=False)
    assert torch.equal(frames, other_frames)
    assert (states == other_states).all()

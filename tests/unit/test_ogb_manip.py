"""CPU-checkable invariants of the OGBench Cube/Scene MJWarp renderer and its sampling.

`ogbench` is optional; every test skips cleanly without it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("ogbench")

try:
    import stable_envs.envs.ogbench.manipulation as ogb
except RuntimeError as e:  # the module refuses to register a cell without measured visual_fidelity
    pytest.skip(f"ogbench.manipulation is not measured in this build: {e}", allow_module_level=True)
except ImportError as e:
    pytest.skip(f"ogbench.manipulation's dependencies are absent: {e}", allow_module_level=True)


@pytest.mark.parametrize("cell,nq,nbuttons", [("Cube", 21, 0), ("Scene", 25, 2)])
def test_state_layout_is_qpos_goal_and_buttons(cell, nq, nbuttons):
    b = ogb._bundle(cell)
    assert b.nq == nq and b.nbuttons == nbuttons
    names = list(b.state_names)
    assert len(names) == nq + 3 + nbuttons
    assert names[:nq] == [f"qpos_{i}" for i in range(nq)]
    assert names[nq : nq + 3] == ["goal_x", "goal_y", "goal_z"]
    assert names[nq + 3 :] == [f"button_{i}" for i in range(nbuttons)]


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_mocap_quat_really_is_constant_in_the_registered_distribution(cell):
    """mocap_quat is not a state column because it never varies under the registered distribution."""
    env = ogb._reference_env(cell, role="sample")
    u = env.unwrapped
    seen = set()
    for s in range(4):
        u.np_random = np.random.default_rng(s)
        env.reset(seed=s)
        seen.add(tuple(np.round(u.data.mocap_quat.ravel(), 9)))
        for _ in range(3):
            env.step(u.action_space.sample())
            seen.add(tuple(np.round(u.data.mocap_quat.ravel(), 9)))
    assert len(seen) == 1, f"mocap_quat varies after all: {seen}"
    assert np.allclose(np.array(next(iter(seen))).reshape(-1, 4)[:, 0], 1.0)


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_sample_states_is_a_pure_function_of_its_rng(cell):
    f = _fake(cell)
    a = ogb.OGBManipBase.sample_states(f, 4, np.random.default_rng(0))
    b = ogb.OGBManipBase.sample_states(f, 4, np.random.default_rng(0))
    assert np.array_equal(a, b), f"max diff {np.abs(a - b).max()}"
    c = ogb.OGBManipBase.sample_states(f, 4, np.random.default_rng(1))
    assert not np.array_equal(a, c), "a different rng must give a different draw"


def _fake(cell):
    """An instance with only what `sample_states` needs: no CUDA, no renderer."""

    class _F(ogb.OGBManipBase):
        CELL = cell

        def __init__(self):
            self._b = ogb._bundle(cell)

    return _F()


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_sample_states_leaves_the_global_numpy_rng_alone(cell):
    fake = _fake(cell)
    np.random.seed(1234)
    before = np.random.get_state()[1][:8].copy()
    ogb.OGBManipBase.sample_states(fake, 2, np.random.default_rng(0))
    assert np.array_equal(np.random.get_state()[1][:8], before)


def test_button_tables_cover_every_combination_and_differ():
    b = ogb._bundle("Scene")
    assert b.button_tables is not None
    assert sorted(b.button_tables) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    g0, m0 = b.button_tables[(0, 0)]
    assert any(not np.array_equal(g0, g) or not np.array_equal(m0, m) for g, m in b.button_tables.values())


def test_button_deltas_are_the_same_whatever_task_the_env_starts_in():
    """`_ctx_button_tables` applies each bit combination as a delta against the all-zero table.

    The deltas must depend on the bits alone, not on the button state the env happened to start in.
    """
    b = ogb._bundle("Scene")
    u = b.unwrapped
    saved = list(getattr(u, "_cur_button_states", []))

    def deltas(tables):
        g0, m0 = tables[(0, 0)]
        return {k: (v[0] - g0, v[1] - m0) for k, v in tables.items()}

    want = deltas(b.button_tables)
    try:
        for start in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            u._cur_button_states = list(start)
            u._apply_button_states()
            got = deltas(ogb._button_tables(u, b.model, b.nbuttons))
            for bits in want:
                assert np.array_equal(got[bits][0], want[bits][0]), (start, bits, "geom_rgba delta")
                assert np.array_equal(got[bits][1], want[bits][1]), (start, bits, "mat_rgba delta")
        # the deltas must not all be zero, or the bits would not select colors
        assert any(np.any(g) or np.any(m) for g, m in want.values())
    finally:
        u._cur_button_states = list(saved) if saved else [0] * b.nbuttons
        u._apply_button_states()


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_the_mover_set_contains_the_cube_and_excludes_the_floor(cell):
    """The composite takes MJWarp only where a mover was hit; a floor in the set would hide the baked plate."""
    import mujoco

    b = ogb._bundle(cell)
    movers = set(b.movers)
    floor = mujoco.mj_name2id(b.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor >= 0 and floor not in movers
    assert set(int(g) for g in b.cube_geoms) <= movers


@pytest.mark.parametrize("cname", ["OGBCube", "OGBScene"])
def test_the_frozen_corpus_is_a_subset_of_the_pool_committed_beside_it(cname):
    """`meta['indices']` indexes the pool committed beside the corpus."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    d = root / "tests/fixtures/states/ogb_manipulation"
    sp, pp = d / f"{cname}.npz", d / f"{cname}_pool.npz"
    if not (sp.exists() and pp.exists()):
        pytest.skip("the corpora are written by the GPU measurement job")
    z, zp = np.load(sp, allow_pickle=False), np.load(pp, allow_pickle=False)
    states, pool = z["states"], zp["pool"]
    meta = json.loads(str(z["meta"]))
    idx = np.asarray(meta["indices"], int)
    assert len(idx) == len(states)
    assert np.array_equal(pool[idx], states), "the selection does not index this pool"
    assert states.shape[1] == meta["state_dim"]
    assert list(meta["state_names"]) and len(meta["state_names"]) == states.shape[1]


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_split_cuts_the_row_and_supplies_an_identity_quaternion(cell):
    b = ogb._bundle(cell)
    row = np.arange(len(b.state_names), dtype=np.float64)
    qpos, mp, mq, bits = ogb.OGBManipBase._split(_fake(cell), row)
    assert np.array_equal(qpos, row[: b.nq])
    assert np.array_equal(mp.ravel(), row[b.nq : b.nq + 3])
    assert np.array_equal(bits, row[b.nq + 3 :])
    assert np.array_equal(mq, [[1.0, 0.0, 0.0, 0.0]] * b.nmocap)


def test_the_bake_and_the_button_rule_touch_disjoint_material_rows():
    """The texture bake and the button color rule must write different material rows.

    Otherwise applying one as a delta over the other would undo it. Rows are measured against the
    all-zero-bits table, so the handle rows belong to the buttons and not to the bake.
    """
    import copy

    from stable_envs.render import mjwarp as MW

    b = ogb._bundle("Scene")
    zero = (0, 0)
    ref_g0, ref_m0 = b.button_tables[zero]

    u = b.unwrapped
    saved = list(getattr(u, "_cur_button_states", []))
    try:
        u._cur_button_states = list(zero)
        u._apply_button_states()
        m = copy.deepcopy(b.model)
    finally:
        u._cur_button_states = list(saved) if saved else [0, 0]
        u._apply_button_states()
    MW.prepare_model(m, ogb.VISIBLE_GROUPS)
    baked = MW.bake_unsamplable_textures(m, ogb.VISIBLE_GROUPS)
    assert baked["n_materials"] == 1, "the Scene has exactly one unsamplable material"

    bake_rows = set(np.nonzero(np.any(np.asarray(m.mat_rgba) != ref_m0, axis=1))[0].tolist())
    button_rows = set()
    for bits, (_, rm) in b.button_tables.items():
        button_rows |= set(np.nonzero(np.any(rm - ref_m0, axis=1))[0].tolist())
    assert bake_rows, "the bake must change a material row"
    assert button_rows, "the button rule must change material rows"
    assert not (bake_rows & button_rows), (
        f"bake rows {sorted(bake_rows)} and button rows {sorted(button_rows)} overlap"
    )


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_the_SPEC_state_dim_constant_matches_what_bundle_computes(cell):
    """`SPEC` carries the constant so that importing the module never builds the OGBench envs."""
    assert ogb.SPEC[cell]["state_dim"] == len(ogb._bundle(cell).state_names)


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_sample_states_returns_the_dtype_the_state_round_trips_in(cell):
    """Rows are float32, the dtype the env stores, so a state round trip is exact."""
    st = ogb.OGBManipBase.sample_states(_fake(cell), 2, np.random.default_rng(0))
    assert st.dtype == np.float32


def test_the_appearance_digest_ignores_sites_the_renderer_never_draws():
    """A task-changing reset moves undrawn target-marker sites, which must not invalidate a cached plate rig."""
    b = ogb._bundle("Scene")
    env = ogb._reference_env("Scene", role="sample")
    sitegroup = b.unwrapped._scene_option.sitegroup
    seen = set()
    for seed in range(10):
        env.reset(seed=seed)
        seen.add(ogb._appearance_digest(b.model, sitegroup))
    assert len(seen) == 1, f"the digest took {len(seen)} values over ten task-changing resets"


def test_the_appearance_digest_still_sees_a_site_that_is_drawn():
    """The mask is by site group, not a blanket skip of every site array."""
    b = ogb._bundle("Scene")
    m = b.model
    sitegroup = b.unwrapped._scene_option.sitegroup
    base = ogb._appearance_digest(m, sitegroup)
    group, pos = np.array(m.site_group, copy=True), np.array(m.site_pos, copy=True)
    try:
        m.site_group[0] = 0  # group 0 is drawn under this scene option
        m.site_pos[0] = m.site_pos[0] + 0.05
        moved = ogb._appearance_digest(m, sitegroup)
    finally:
        m.site_group[:] = group
        m.site_pos[:] = pos
    assert moved != base, "a drawn site moved and the digest did not change"
    assert ogb._appearance_digest(m, sitegroup) == base, "the restore did not put the model back"

"""What the OGBench manipulation cells promise, asserted on CPU.

The cells render through `mujoco_warp`, so their pixels cannot be exercised here. Everything that is
NOT the CUDA render can be, and the independent review of the first version found that the parts
worth testing were exactly the parts nothing tested: a composite rule that never used its own plate,
a colour table captured against the wrong baseline, a state draw that was not reproducible, and a
committed pool the corpus was not selected from. Each of those has a test below, written so it fails
on the defect rather than on its symptom.

`ogbench` is optional; every test skips cleanly without it. This package does not import
`stable_worldmodel`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("ogbench")

try:
    import stable_envs.envs.ogbench.manipulation as ogb
except RuntimeError as e:  # no MEASURED visual_fidelity yet: the module refuses to register
    pytest.skip(f"ogbench.manipulation is not measured in this build: {e}", allow_module_level=True)
except ImportError as e:
    pytest.skip(f"ogbench.manipulation's dependencies are absent: {e}", allow_module_level=True)


# --------------------------------------------------------------------------- state layout


@pytest.mark.parametrize("cell,nq,nbuttons", [("Cube", 21, 0), ("Scene", 25, 2)])
def test_state_layout_is_qpos_mocap_pos_and_bits(cell, nq, nbuttons):
    b = ogb._bundle(cell)
    assert b.nq == nq and b.nbuttons == nbuttons
    names = list(b.state_names)
    assert len(names) == nq + 3 + nbuttons
    assert names[:nq] == [f"qpos[{i}]" for i in range(nq)]
    assert names[nq : nq + 3] == [f"mocap_pos[{i}]" for i in range(3)]
    assert names[nq + 3 :] == [f"button_state_{i}" for i in range(nbuttons)]


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_neither_qvel_nor_mocap_quat_is_a_state_dim(cell):
    """Both were removed, for two different measured reasons; see the module docstring.

    `qvel` reaches no pixel. `mocap_quat` is CONSTANT under the registered distribution, so it
    cannot be shown to matter and the repo's `check_state_dims_matter` reports it dead.
    """
    names = list(ogb._bundle(cell).state_names)
    assert not any(n.startswith("qvel[") for n in names)
    assert not any(n.startswith("mocap_quat[") for n in names)


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_mocap_quat_really_is_constant_in_the_registered_distribution(cell):
    """The measurement that justifies dropping it, re-run small so the claim is not just prose."""
    env = ogb._reference_env(cell)
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


# --------------------------------------------------------------------------- reproducibility


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_sample_states_is_a_pure_function_of_its_rng(cell):
    """The property `tests/gpu/test_reproducibility.py` asserts, and the one the first version failed.

    Three upstream facts had to be worked around for this to hold, all verified in `ogb_manip`:
    `CubeEnv.reset` drops the `seed` it is given, `action_space` is a property that builds a fresh
    OS-seeded Box on every access, and `reset_variation_space` is called with `seed=None`.
    """
    f = _fake(cell)
    a = ogb.OGBManipBase.sample_states(f, 4, np.random.default_rng(0))
    b = ogb.OGBManipBase.sample_states(f, 4, np.random.default_rng(0))
    assert np.array_equal(a, b), f"max diff {np.abs(a - b).max()}"
    c = ogb.OGBManipBase.sample_states(f, 4, np.random.default_rng(1))
    assert not np.array_equal(a, c), "a different rng must give a different draw"


def _fake(cell):
    """An instance with only what `sample_states` needs -- no CUDA, no renderer."""

    class _F(ogb.OGBManipBase):
        CELL = cell

        def __init__(self):
            self._b = ogb._bundle(cell)

    return _F()


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_sample_states_leaves_the_global_numpy_rng_alone(cell):
    """It seeds the global generator (the only handle the reference exposes) and must put it back."""
    np.random.seed(1234)
    before = np.random.get_state()[1][:8].copy()
    ogb.OGBManipBase.sample_states(_fake(cell), 2, np.random.default_rng(0))
    assert np.array_equal(np.random.get_state()[1][:8], before)


# --------------------------------------------------------------------------- button tables


def test_button_tables_cover_every_combination_and_differ():
    b = ogb._bundle("Scene")
    assert b.button_tables is not None
    assert sorted(b.button_tables) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    g0, m0 = b.button_tables[(0, 0)]
    assert any(not np.array_equal(g0, g) or not np.array_equal(m0, m) for g, m in b.button_tables.values())


def test_button_deltas_are_the_same_whatever_task_the_env_starts_in():
    """The B2 regression, asserted on the quantity the code actually uses.

    `_ctx_button_tables` applies each combination as a DELTA against the all-zero table, and that is
    only correct if the delta is a function of the bits alone. The absolute tables are not: they also
    carry cube colours and the target marker's alpha, which `initialize_episode` and `post_step`
    move for reasons nothing to do with buttons, so asserting those would fail on unrelated drift.

    The defect this guards is the BASE, not the delta: the env's bits depend on which task `reset`
    picked -- measured starts include (0,0), (1,0) and (1,1) -- so a context copied without forcing
    them to zero applied correct deltas to a wrong base, silently, in roughly 40 % of processes.
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
        # and the deltas must not all be zero, or the bits are not colours
        assert any(np.any(g) or np.any(m) for g, m in want.values())
    finally:
        u._cur_button_states = list(saved) if saved else [0] * b.nbuttons
        u._apply_button_states()


def test_the_context_forces_all_zero_bits_before_copying_the_model():
    """The base itself, asserted from the source rather than from a GPU render.

    `_context` cannot be exercised without CUDA, so this checks the invariant it must maintain: that
    it forces the reference to all-zero bits around the deep copy and restores what it found.
    """
    import inspect

    src = inspect.getsource(ogb._context)
    assert "zero_bits" in src and "_apply_button_states" in src
    assert "saved_bits" in src, "the reference's own bits must be restored after the copy"
    assert src.index("_cur_button_states") < src.index("copy.deepcopy"), (
        "the bits must be forced BEFORE the copy is taken"
    )


# --------------------------------------------------------------------------- the composite rule


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_the_mover_set_excludes_the_floor_and_the_sky(cell):
    """B1's root cause, as a property of the mover set rather than of the GPU path.

    The composite must take MJWarp only where a MOVER was hit. The floor plane is a geom, so a
    `seg >= 0` rule takes MJWarp's floor over the baked plate everywhere -- the plate is then used
    only on sky, which is what the first version shipped.
    """
    import mujoco

    b = ogb._bundle(cell)
    m = b.model
    movers = set(b.movers)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor >= 0 and floor not in movers, "the floor must not be a mover"
    assert movers, "there must be movers"
    for g in movers:
        bid = int(m.geom_bodyid[g])
        moving = False
        while bid > 0:
            if int(m.body_dofnum[bid]) > 0 or int(m.body_mocapid[bid]) >= 0:
                moving = True
                break
            bid = int(m.body_parentid[bid])
        assert moving, f"geom {g} is in the mover set but sits on no moving body"


def test_the_lut_rule_selects_movers_and_lets_everything_else_fall_through():
    """The compositing arithmetic itself, on a synthetic segmentation -- no CUDA needed."""
    torch = pytest.importorskip("torch")
    ngeom, movers = 6, [2, 4]
    lut = torch.zeros(ngeom + 1, dtype=torch.bool)
    lut[torch.as_tensor(np.asarray(movers) + 1)] = True
    seg = torch.as_tensor([[-1, 0, 2], [3, 4, 5]], dtype=torch.int32)  # -1 = sky, 0 = floor
    mv = lut.index_select(0, (seg + 1).reshape(-1)).view(seg.shape)
    assert mv.tolist() == [[False, False, True], [False, True, False]]
    rgb = torch.full((2, 3, 3), 7, dtype=torch.uint8)
    plate = torch.full((2, 3, 3), 3, dtype=torch.uint8)
    comp = torch.where(mv.unsqueeze(-1), rgb, plate)
    assert comp[0, 0].tolist() == [3, 3, 3], "sky must come from the plate"
    assert comp[0, 1].tolist() == [3, 3, 3], "the floor must come from the plate"
    assert comp[0, 2].tolist() == [7, 7, 7], "a mover must come from MJWarp"


# --------------------------------------------------------------------------- declarations


@pytest.mark.parametrize("cname", ["OGBCube", "OGBScene"])
def test_the_declaration_is_complete_and_visual(cname):
    import stable_envs as se

    cls = se.registry.REGISTRY[cname]
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
        "draw_spread",
    ):
        assert k in vf, f"{cname}: visual_fidelity is missing {k}"
    assert vf["mean_abs_max"] > vf["mean_abs"], "a ceiling must sit above the measurement"
    assert vf["within8_min"] < vf["within8_frac"]
    assert vf["ssim_min"] < vf["ssim"]
    assert "VISUAL TIER" in (cls.__doc__ or "").upper()
    # The docstring may (and should) EXPLAIN that qvel is excluded; what it must not do is list it
    # as part of the state. Asserting the bare word is absent would forbid the explanation.
    doc = cls.__doc__ or ""
    assert "qvel)" not in doc and "qvel," not in doc.split("deliberately NOT")[0], (
        "the docstring must not present qvel as part of the state"
    )
    assert "(qpos, mocap_pos" in doc, "the docstring must state the real layout"


@pytest.mark.parametrize("cname", ["OGBCube", "OGBScene"])
def test_cpu_is_refused_rather_than_served_from_the_gpu(cname):
    import stable_envs as se

    with pytest.raises(ValueError, match="CPU3D-OGB-1"):
        se.make(cname, num_envs=1, device="cpu")


# --------------------------------------------------------------------------- the frozen corpus


@pytest.mark.parametrize("cname", ["OGBCube", "OGBScene"])
def test_the_frozen_corpus_is_a_subset_of_the_pool_committed_beside_it(cname):
    """B4: `meta['indices']` must index the pool that was actually kept."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    d = root / "tests/fixtures/regression/join/step7/ogb-manipulation/states"
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


# --------------------------------------------------------------------------- the upload slicing


@pytest.mark.parametrize("cell,nq,nbuttons", [("Cube", 21, 0), ("Scene", 25, 2)])
def test_the_render_upload_slices_match_the_state_layout(cell, nq, nbuttons):
    """The arithmetic `render()` uses to cut a state vector up, checked without CUDA.

    Job 6405900 died on exactly this and nothing here could have caught it: `render()` still sliced
    `mocap_quat` out of the state after B5 dropped it, so the slice was empty and the reshape raised
    `shape '[64, 1, 4]' is invalid for input of size 0`. The GPU path cannot run on CPU, but its
    INDEX arithmetic is pure and is asserted here against the same `state_names` the cell publishes.
    """
    b = ogb._bundle(cell)
    dim = len(b.state_names)
    assert dim == nq + 3 + nbuttons
    chunk = np.arange(2 * dim, dtype=np.float32).reshape(2, dim)

    mp = chunk[:, b.nq : b.nq + 3 * b.nmocap]
    assert mp.shape == (2, 3 * b.nmocap), "mocap_pos slice must be non-empty and the right width"
    assert mp.reshape(2, b.nmocap, 3).shape == (2, b.nmocap, 3)

    if nbuttons:
        bits = chunk[:, -nbuttons:]
        assert bits.shape == (2, nbuttons)
        assert b.nq + 3 * b.nmocap + nbuttons == dim, "bits must be the tail of the vector"
    else:
        assert b.nq + 3 * b.nmocap == dim, "with no bits the vector ends after mocap_pos"

    # and there is NO quaternion left to slice
    assert not any(n.startswith("mocap_quat[") for n in b.state_names)
    assert chunk[:, b.nq + 3 * b.nmocap : b.nq + 7 * b.nmocap].shape[1] == nbuttons or nbuttons == 0


def test_split_returns_an_identity_quaternion_rather_than_reading_one():
    b = ogb._bundle("Cube")
    f = _fake("Cube")
    st = np.arange(len(b.state_names), dtype=np.float64)
    qpos, mp, mq, bits = ogb.OGBManipBase._split(f, st)
    assert qpos.shape == (b.nq,) and mp.shape == (b.nmocap, 3)
    assert mq.shape == (b.nmocap, 4)
    assert np.array_equal(mq, ogb.IDENTITY_QUAT(b.nmocap))
    assert np.allclose(mq[:, 0], 1.0) and np.allclose(mq[:, 1:], 0.0)


def test_the_bake_and_the_button_rule_touch_disjoint_material_rows():
    """The conflict `_ctx_button_tables` asserts, checked on CPU against the right baseline.

    The bake replaces `btn_metal`; the button rule writes the two button geoms and the drawer and
    window handle materials. They must be disjoint, or applying one as a delta over the other undoes
    it. Job 6406972 died on this assertion firing SPURIOUSLY: it compared the context copy against a
    snapshot taken under whatever bits the env's last reset happened to choose, so on a Scene
    starting at (1,0) the handle rows looked like the bake's work. The baseline is the all-zero
    table, and this test pins both halves -- that the rows really are disjoint, and that the handle
    rows belong to the buttons rather than to the bake.
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
    """`SPEC[cell]['state_dim']` is a constant so that `_make_classes` never calls `_bundle`.

    It was `len(_bundle(cell).state_names)`, which built both OGBench environments at IMPORT: 4.4 s, and
    without `ogbench` the module raised and both ids dropped out of the registry (76
    instead of 78). The constant has to be kept honest by something, and this is it -- it runs only
    where ogbench IS importable, which is exactly where the truth is available.
    """
    assert ogb.SPEC[cell]["state_dim"] == len(ogb._bundle(cell).state_names)


@pytest.mark.parametrize("cell", ["Cube", "Scene"])
def test_sample_states_returns_the_dtype_the_state_round_trips_in(cell):
    """`check_state_roundtrip` spells its bound as exactly 0.0, so a narrowing cast is a failure.

    `StableEnv.set_state` stores float32. Returning float64 here made `set_state(get_state())` differ
    by the narrowing error -- 1.17e-07 on the Cube, 1.12e-07 on the Scene -- against a bound of
    exactly zero. Asserted on the dtype rather than on a round trip, because the round trip needs a
    constructed env and this does not.
    """
    st = ogb.OGBManipBase.sample_states(_fake(cell), 2, np.random.default_rng(0))
    assert st.dtype == np.float32, f"got {st.dtype}; the env stores float32"
    assert np.array_equal(st.astype(np.float32), st), "a float32 round trip must be the identity"


def test_the_appearance_digest_ignores_sites_the_renderer_never_draws():
    """A task-changing reset must not invalidate a cached plate rig.

    ``SceneEnv.reset`` writes ``site_pos`` for the per-task target markers -- measured here, rows 7
    and 9, both group 5 -- and ``scene_option.sitegroup`` is ``[1, 1, 1, 0, 0, 0]``, so those groups
    reach no pixel. Hashing them made ``_appearance_digest`` take four values over eight resets, and
    every one of those would fire ``_bake_plate``'s assertion on a rig reused after the task changed:
    a correct plate rejected for a marker nobody can see. Reproduced before the fix; this is the
    regression test for it, and it is the task-1 -> task-3 bake reduced to the quantity the bake
    actually asserts on.
    """
    b = ogb._bundle("Scene")
    env = ogb._reference_env("Scene", role="sample")
    sitegroup = b.unwrapped._scene_option.sitegroup
    seen = set()
    for seed in range(10):
        env.reset(seed=seed)  # lands on a different task id as the seed changes
        seen.add(ogb._appearance_digest(b.model, sitegroup))
    assert len(seen) == 1, (
        f"the digest took {len(seen)} values over ten task-changing resets, so a plate rig built "
        f"before one of them would be rejected by _bake_plate for a change no pixel can show"
    )


def test_the_appearance_digest_still_sees_a_site_that_is_drawn():
    """The mask is BY GROUP, not a blanket skip of every site array.

    Excluding all ``site_*`` state would also excuse a site that really is drawn and really did move,
    which is the kind of hole that makes an assertion worse than no assertion.
    """
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
    assert moved != base, "a DRAWN site moved and the digest did not notice"
    assert ogb._appearance_digest(m, sitegroup) == base, "the restore did not put the model back"

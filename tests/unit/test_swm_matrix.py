"""Behaviour the coverage matrix generator must not lose.

These are not tests of the matrix's CONTENTS -- those change whenever an env is added, which is the
whole point of generating the thing. They pin the four properties that make the generated page
trustworthy, each of which was violated by the first pass and fixed in the second:

  1. a failed probe reads as ``unknown_probe_failed``, never as ``absent``;
  2. an analytic integrator and a different contact solver are different physics classes;
  3. the same inputs produce the same bytes;
  4. a second census enriches rows and never deletes what the first one established.

Plus one that was never violated and is easy to violate later: a parameterised id whose default
matches must not be promoted to full configuration coverage.

The generator is a script in ``scripts/regression/``, so it is loaded by path rather than imported as a package.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MATRIX_PY = REPO / "scripts" / "regression" / "swm_matrix.py"


@pytest.fixture(scope="module")
def mx():
    if not MATRIX_PY.exists():
        pytest.skip(f"{MATRIX_PY} is absent")
    spec = importlib.util.spec_from_file_location("swm_matrix", MATRIX_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================ 1. unknown is not absent


def test_failed_construction_is_unknown_not_absent(mx):
    """The bug that made the first matrix wrong about sixteen ids.

    Four classic-control ids came back ``absent`` because the probe read ``unwrapped`` (a stock
    gymnasium env) instead of the swm wrapper that holds the variation space, and twelve Fetch ids
    came back ``absent`` because they did not construct at all. Both are now ``unknown_probe_failed``
    or a measured ``present``; neither may silently become ``absent`` again.
    """
    probe = {
        "make": "ERR AssertionError: ",
        "variation_space_status": "unknown_probe_failed",
        "frame_status_probe": "unknown_probe_failed",
        "variation_names": None,
    }
    status, why = mx.classify_frame(probe)
    assert status == "unknown", why
    assert "did not construct" in why

    default, params, _sem = mx.collect_default_and_params(
        "swm/Whatever-v0", probe, {"envs": [], "rung": None, "confidence": "none", "evidence": ""}, {}, {}
    )
    assert default["covered"] == "unknown"
    assert default["kind"] == "unknown_probe_failed"
    assert params["level"] == "unknown"


def test_declared_none_variation_space_is_absent_not_unknown(mx):
    """The other direction. An env that SAYS it has no variation space has answered the question.

    The Craftax wrappers define ``variation_space`` and set it to ``None``; that is a real answer and
    must not be laundered into "we could not tell", or a settled row would grow a fake next action.
    """
    probe = {
        "make": "ok",
        "variation_space_status": "absent",
        "variation_names": None,
        "frame_status_probe": "pixels",
        "frame_shape": [63, 63, 3],
    }
    status, _ = mx.classify_frame(probe)
    assert status == "pixels"
    assert probe["variation_space_status"] == "absent"


def test_symbolic_env_is_not_a_renderer_gap(mx):
    """No frame to draw is a classification, not a hole in coverage."""
    probe = {
        "make": "ok",
        "frame_shape": None,
        "frame_status_probe": "symbolic_no_frame",
        "reset": "ERR NotImplementedError: render is only supported for Pixels variants; "
        "symbolic envs do not produce a frame.",
    }
    status, _ = mx.classify_frame(probe)
    assert status == "symbolic_no_frame"
    default, params, _sem = mx.collect_default_and_params(
        "swm/X-v0", probe, {"envs": [], "rung": None, "confidence": "none", "evidence": ""}, {}, {}
    )
    assert default["covered"] == "n/a" and params["level"] == "n/a"


# ============================================================ 2. physics taxonomy


def _dec(**kw):
    base = {
        "has_step": False,
        "step_reference_exact": None,
        "step_reference_gate": None,
        "step_reference_control": None,
    }
    base.update(kw)
    return base


def test_analytic_integrator_and_contact_solver_are_different_classes(mx):
    """The distinction the first pass lost by calling both ``approx_fast_step``.

    Classic control transcribes a published closed form term for term and diverges only where the
    transcendentals do (measured at 1.1e-16 after one step). PushT runs a different contact solver
    and cannot be bit-exact at all. Grouping them would let a reader believe the classic-control
    dynamics are an approximation, which they are not.
    """
    near = mx.classify_physics(["CartPole"], {"CartPole": _dec(has_step=True)})
    approx = mx.classify_physics(
        ["PushT"], {"PushT": _dec(has_step=True, step_reference_gate="block", step_reference_control="agent")}
    )
    assert near["class"] == "near_exact_fast_step"
    assert approx["class"] == "approximate_fast_step"
    assert near["class"] != approx["class"]


def test_exact_step_requires_the_declaration(mx):
    """``exact`` is a claim an env makes and the conformance suite enforces, never an inference."""
    assert (
        mx.classify_physics(
            ["TwoRoom"], {"TwoRoom": _dec(has_step=True, step_reference_exact=True, step_reference_control="agent")}
        )["class"]
        == "exact_fast_step"
    )
    # ... and a bare `step` with no declaration is not promoted to exact
    assert mx.classify_physics(["X"], {"X": _dec(has_step=True)})["class"] != "exact_fast_step"


def test_no_step_and_no_declaration_are_different_answers(mx):
    """``reference_only`` means we looked; ``unknown`` means we could not."""
    assert mx.classify_physics(["X"], {"X": _dec()})["class"] == "reference_only"
    assert mx.classify_physics(["X"], {})["class"] == "unknown"
    assert mx.classify_physics([], {})["class"] == "n/a"


def test_physics_classes_are_the_declared_vocabulary(mx):
    assert set(mx.PHYSICS_CLASSES) == {
        "exact_fast_step",
        "near_exact_fast_step",
        "approximate_fast_step",
        "reference_only",
        "unknown",
        "n/a",
    }


# ============================================================ 3. multi-census merge is monotonic


def _census(tag: str, envs: dict) -> Path:
    return {
        "python": f"/fake/{tag}",
        "registry": sorted(envs),
        "n": len(envs),
        "import_errors": {},
        "declarations": envs,
    }


def test_second_census_enriches_and_never_deletes(mx, tmp_path):
    """Adding an interpreter that owns a family must only ever reduce uncertainty.

    This is the property the Oscar run depends on: the Mac census cannot import Fetch and the Oscar
    census cannot import DM Control, so the matrix is only correct if the union keeps both.
    """
    a = tmp_path / "census_a.json"
    b = tmp_path / "census_b.json"
    a.write_text(
        json.dumps(
            _census(
                "a",
                {
                    "DMCWalkerWalk": {"reference_id": "dm_control/walker-walk"},
                    "PushT": {"reference_id": "gym_pusht/PushT-v0"},
                },
            )
        )
    )
    b.write_text(
        json.dumps(
            _census(
                "b", {"FetchPush": {"reference_id": "FetchPush-v4"}, "PushT": {"reference_id": "gym_pusht/PushT-v0"}}
            )
        )
    )

    one = mx.read_censuses([a])
    both = mx.read_censuses(sorted([a, b]))

    assert set(one["declarations"]) == {"DMCWalkerWalk", "PushT"}
    assert set(both["declarations"]) == {"DMCWalkerWalk", "PushT", "FetchPush"}
    # nothing the first census established is lost
    assert set(one["declarations"]) <= set(both["declarations"])
    # and ownership accumulates rather than being overwritten
    assert both["owners"]["PushT"] == sorted(["census_a", "census_b"])


def test_census_merge_is_order_independent(mx, tmp_path):
    a = tmp_path / "census_a.json"
    b = tmp_path / "census_b.json"
    a.write_text(json.dumps(_census("a", {"X": {"reference_id": "x", "fidelity_tier": "parity"}})))
    b.write_text(
        json.dumps(_census("b", {"X": {"reference_id": "x", "fidelity_tier": "parity"}, "Y": {"reference_id": "y"}}))
    )
    assert mx.read_censuses(sorted([a, b]))["declarations"] == mx.read_censuses(sorted([b, a]))["declarations"]


def test_swm_probes_must_agree_on_the_id_list(mx, tmp_path):
    """Two probes of DIFFERENT stable_worldmodel versions would describe neither, so it is an error.

    Merging probes is only sound because the registry is built at import time from one source: what
    differs between interpreters is how far each could construct an id, never which ids exist.
    """

    def _probe(path, ids):
        path.write_text(
            json.dumps({"ids": ids, "probe": {i: {"make": "ok"} for i in ids}, "entry_points": {}, "kwargs": {}})
        )
        return path

    p1 = _probe(tmp_path / "p1.json", ["swm/A-v0", "swm/B-v0"])
    p2 = _probe(tmp_path / "p2.json", ["swm/A-v0"])
    with pytest.raises(SystemExit, match="disagree"):
        mx.read_swm_registries([p1, p2])


def test_swm_probe_merge_keeps_the_record_that_learned_most(mx, tmp_path):
    """Fetch constructs in one interpreter and not the other; the merge must take the one that did."""
    blind = tmp_path / "a_blind.json"
    seeing = tmp_path / "b_seeing.json"
    ids = ["swm/FetchPush-v3"]
    blind.write_text(
        json.dumps(
            {
                "ids": ids,
                "entry_points": {},
                "kwargs": {},
                "probe": {
                    "swm/FetchPush-v3": {
                        "make": "ERR AssertionError",
                        "variation_space_status": "unknown_probe_failed",
                        "frame_status_probe": "unknown_probe_failed",
                    }
                },
            }
        )
    )
    seeing.write_text(
        json.dumps(
            {
                "ids": ids,
                "entry_points": {},
                "kwargs": {},
                "probe": {
                    "swm/FetchPush-v3": {
                        "make": "ok",
                        "variation_space_status": "present",
                        "frame_status_probe": "pixels",
                        "frame_shape": [224, 224, 3],
                        "variation_names": ["object.color"],
                    }
                },
            }
        )
    )
    merged = mx.read_swm_registries(sorted([blind, seeing]))
    rec = merged["probe"]["swm/FetchPush-v3"]
    assert rec["variation_space_status"] == "present"
    assert rec["frame_shape"] == [224, 224, 3]
    assert merged["probe_source"]["swm/FetchPush-v3"] == "b_seeing"


# ============================================================ 4. parameter coverage


def test_default_match_does_not_imply_full_parameter_coverage(mx):
    """One registered id can build many scenes; rendering its default says nothing about the rest.

    ``swm/OGBMaze-v0`` renders its registered default (point / large) exactly and still leaves four
    of fifteen cells with no renderer. Promoting it to ``full`` on the strength of the default is the
    overclaim this field exists to prevent.
    """
    meta = mx.PARAMETERIZED["swm/OGBMaze-v0"]
    axes = meta["config_axes"]
    # declare only the registered default, and nothing else
    decls = {"PointmazeLargeNavigate": {"reference_id": "ogbench/pointmaze-large-v0"}}
    probe = {"make": "ok", "frame_status_probe": "pixels"}
    match = {
        "envs": ["PointmazeLargeNavigate"],
        "rung": "documented_manual",
        "confidence": "verified",
        "evidence": "",
        "primary": "PointmazeLargeNavigate",
    }
    default, params, _sem = mx.collect_default_and_params("swm/OGBMaze-v0", probe, match, decls, {})
    assert default["covered"] is True
    assert params["level"] == "partial"
    n_cells = len(axes["loco_env_type"]) * len(axes["maze_type"])
    assert params["configs_total"] == n_cells
    assert len(params["missing_configurations"]) == n_cells - 1


def test_parameter_coverage_is_computed_from_declarations(mx):
    """Declare every cell and the level becomes ``full`` -- the list is derived, not hand-kept."""
    meta = mx.PARAMETERIZED["swm/OGBMaze-v0"]
    decls = {}
    for loco in meta["config_axes"]["loco_env_type"]:
        for maze in meta["config_axes"]["maze_type"]:
            ref = meta["reference_pattern"].format(loco=loco, maze=maze)
            decls[f"{loco}{maze}"] = {"reference_id": ref}
    _, params, _sem = mx.collect_default_and_params(
        "swm/OGBMaze-v0",
        {"make": "ok", "frame_status_probe": "pixels"},
        {"envs": ["x"], "rung": "documented_manual", "confidence": "verified", "evidence": ""},
        decls,
        {},
    )
    assert params["level"] == "full"
    assert params["missing_configurations"] == []


def test_reward_difference_does_not_reduce_render_coverage(mx):
    """``swm/HumanoidDMControl-v0`` is ``humanoid-walk``; we declare ``humanoid-stand``.

    Step 1B blocked this row, and that was wrong. The two compile to a byte-identical render model,
    so our renderer draws this id's states correctly -- and in the pipeline this project runs,
    stable_worldmodel keeps producing the states while stable-envs supplies pixels only, so the
    reward never reaches the renderer. Render coverage is therefore TRUE and the reward difference is
    recorded on its own axis, where it informs anyone stepping the stable-envs env standalone
    without blocking renderer validation.
    """
    probe = {
        "make": "ok",
        "frame_status_probe": "pixels",
        "dmc_reference": {"reference_id": "dm_control/humanoid-walk", "how": "exact"},
        "swm_task_space": {
            "accepts_task": True,
            "default_task": "walk",
            "tasks": ["run", "stand", "walk"],
            "render_equivalence_classes": [["run", "stand", "walk"]],
            "distinct_render_configs": 1,
            "source": "humanoid._TASKS",
        },
    }
    decls = {"DMCHumanoidStand": {"reference_id": "dm_control/humanoid-stand"}}
    match = {
        "envs": ["DMCHumanoidStand"],
        "rung": "same_dmc_appearance",
        "confidence": "verified",
        "evidence": "",
        "primary": "DMCHumanoidStand",
    }
    default, params, sem = mx.collect_default_and_params("swm/HumanoidDMControl-v0", probe, match, decls, {})
    assert default["covered"] is True, "a reward-name difference must not reduce render coverage"
    assert default["kind"] == "same_scene_reward_differs"
    # ... and the difference is not swept away either
    assert sem["match"] is False
    assert sem["swm_task"] == "dm_control/humanoid-walk"


def test_render_shared_tasks_collapse_to_one_configuration(mx):
    """Eight walker tasks that draw one picture are one renderer configuration, not eight.

    Counting task NAMES would report seven missing renderer configurations that do not exist. The
    equivalence classes come from constructing each task and digesting its render model, so the
    collapse is measured.
    """
    probe = {
        "make": "ok",
        "frame_status_probe": "pixels",
        "dmc_reference": {"reference_id": "dm_control/walker-walk", "how": "by_class"},
        "swm_task_space": {
            "accepts_task": True,
            "default_task": "walk",
            "tasks": ["arabesque", "flip", "legs_up", "lie_down", "run", "stand", "walk", "walk-backward"],
            "render_equivalence_classes": [
                ["arabesque", "flip", "legs_up", "lie_down", "run", "stand", "walk", "walk-backward"]
            ],
            "distinct_render_configs": 1,
            "source": "walker._TASKS",
        },
    }
    decls = {"DMCWalkerWalk": {"reference_id": "dm_control/walker-walk"}}
    match = {"envs": ["DMCWalkerWalk"], "rung": "reference_id", "confidence": "verified", "evidence": ""}
    _, params, _ = mx.collect_default_and_params("swm/WalkerDMControl-v0", probe, match, decls, {})
    assert params["configs_total"] == 1
    assert params["missing_configurations"] == []
    assert params["tasks_total"] == 8


def test_render_changing_tasks_stay_separate(mx):
    """The other direction, and the control that shows the collapse is not indiscriminate.

    ``reacher``'s ``easy`` draws a bigger target than ``hard``, so the two are separate renderer
    configurations and a renderer for one does not cover the other.
    """
    probe = {
        "make": "ok",
        "frame_status_probe": "pixels",
        "dmc_reference": {"reference_id": "dm_control/reacher-hard", "how": "exact"},
        "swm_task_space": {
            "accepts_task": True,
            "default_task": "hard",
            "tasks": ["easy", "hard", "qpos_match"],
            "render_equivalence_classes": [["easy"], ["hard", "qpos_match"]],
            "distinct_render_configs": 2,
            "source": "reacher._TASKS",
        },
    }
    match = {"envs": ["DMCReacherHard"], "rung": "reference_id", "confidence": "verified", "evidence": ""}
    only_hard = {"DMCReacherHard": {"reference_id": "dm_control/reacher-hard"}}
    _, partial, _ = mx.collect_default_and_params("swm/ReacherDMControl-v0", probe, match, only_hard, {})
    assert partial["level"] == "partial"
    assert partial["missing_configurations"] == [["easy"]]

    both = dict(only_hard, DMCReacherEasy={"reference_id": "dm_control/reacher-easy"})
    _, full, _ = mx.collect_default_and_params(
        "swm/ReacherDMControl-v0", probe, dict(match, envs=["DMCReacherHard", "DMCReacherEasy"]), both, {}
    )
    assert full["level"] == "full"


def test_parameter_space_comes_from_the_swm_wrapper_not_the_suite(mx):
    """A wrapper with no ``task`` argument has no task parameter space, whatever the suite lists.

    ``AcrobotDMControlWrapper`` takes none while dm_control's suite lists two acrobot tasks; reading
    the suite invented a missing configuration that the registered id cannot even build.
    """
    probe = {
        "make": "ok",
        "frame_status_probe": "pixels",
        "dmc_reference": {"reference_id": "dm_control/acrobot-swingup_sparse", "how": "exact"},
        "swm_task_space": {"accepts_task": False, "why": "AcrobotDMControlWrapper.__init__ takes no `task` argument"},
    }
    decls = {"DMCAcrobotSwingup": {"reference_id": "dm_control/acrobot-swingup"}}
    match = {
        "envs": ["DMCAcrobotSwingup"],
        "rung": "same_dmc_appearance",
        "confidence": "verified",
        "evidence": "",
        "primary": "DMCAcrobotSwingup",
    }
    # a suite index that DOES list a second acrobot task must not create a missing configuration
    suite = {"acrobot": {"swingup": "aaa", "swingup_sparse": "aaa"}}
    _, params, _ = mx.collect_default_and_params("swm/AcrobotDMControl-v0", probe, match, decls, suite)
    assert params["level"] == "not_parameterized"
    assert "missing_configurations" not in params


# ============================================================ 5. the refusals stay refused


def test_cpu_throughput_is_never_inferred(mx):
    """A CPU path that runs is not a CPU benchmark, and nothing may quietly turn one into the other."""
    out = mx.collect_speed(["PushT"], {"PushT": {"se_gpu_fps": 1e6, "gpu_batch": 4096}})
    assert out["se_cpu_fps"] == mx.NEEDS


def test_dino_evidence_does_not_reach_downstream_level(mx):
    """Representation agreement is not task success, however good the cosine is."""
    dino = {
        "E": {
            "dino_cos_ours_vs_official": 0.0,
            "dino_cos_state_vs_state": 0.2,
            "dino_cos_blank_vs_official": 0.8,
            "dino_cos_jpeg95_vs_official": 0.03,
            "source": "x",
        }
    }
    got = mx.collect_validation(["E"], dino, None, "swm/NotPushT-v0")
    assert got["level"] == "dino_feature"
    assert all(e["kind"] != "downstream_task_success" for e in got["evidence"])


# ============================================================ 6. determinism, on the real inputs


def test_generation_is_byte_identical_on_repeat(mx, tmp_path):
    """The property that makes a diff on the generated page mean something.

    If regenerating churned bytes, nobody could tell a real coverage change from a rerun, and the
    page would stop being reviewable. The generator therefore calls no clock and no RNG -- the only
    time in the output is the ``probed_at`` its input recorded for itself.

    Runs against the committed inputs, with the live probe off so the test does not need CUDA, swm,
    or any optional dependency.
    """
    registries = sorted((REPO / "tests" / "fixtures" / "regression" / "census").glob("swm_registry*.json"))
    censuses = sorted((REPO / "tests" / "fixtures" / "regression" / "census").glob("census_*.json"))
    if not registries or not censuses:
        pytest.skip("committed matrix inputs are absent")

    class Args:
        swm_registry = [str(q) for q in registries]
        se_census = [str(q) for q in censuses]
        union = str(REPO / "tests/fixtures/regression/census/union.json")
        decode_all = str(REPO / "tests/fixtures/regression/references/decode_all.json")
        bench = str(REPO / "tests/fixtures/regression/references/bench_gpu_vs_cpu.json")
        dino = str(REPO / "tests/fixtures/regression/references/dino_agreement.json")
        lewm = str(REPO / "examples/lewm_pusht/results")
        swm_workflow = str(REPO / "examples/swm_compat/results/swm_workflow.json")
        live = False
        _variation_maps = {}

    first = json.dumps(mx.build(Args()), indent=1, sort_keys=False)
    second = json.dumps(mx.build(Args()), indent=1, sort_keys=False)
    assert first == second, "two builds on identical inputs disagreed"
    assert mx.markdown(json.loads(first)) == mx.markdown(json.loads(second))


def test_every_live_swm_id_appears_exactly_once(mx):
    """The join must not drop or duplicate a registered id, whatever the evidence says about it."""
    matrix_json = REPO / "tests" / "fixtures" / "regression" / "fidelity" / "swm_matrix.json"
    registries = sorted((REPO / "tests" / "fixtures" / "regression" / "census").glob("swm_registry*.json"))
    if not matrix_json.exists() or not registries:
        pytest.skip("committed matrix outputs are absent")
    live_ids = json.loads(registries[0].read_text())["ids"]
    rows = [r["swm_id"] for r in json.loads(matrix_json.read_text())["rows"]]
    assert sorted(rows) == sorted(live_ids)
    assert len(rows) == len(set(rows))

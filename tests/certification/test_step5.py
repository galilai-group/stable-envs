"""The Step 5 paired-gate arithmetic, on synthetic arms.

Nothing here plans, renders or loads a checkpoint. What is being tested is how two lists of episode
outcomes become a verdict, and that has to be checkable without a GPU and without the four hours of
allocation the real arms cost.

Three invariants run under all of it:

  * **no threshold lives in this code.** Margin, CI formula, control bounds, minimum episodes and
    the power guard are read from ``scratch/certification.json``, which Step 2 froze before any
    downstream number existed. ``test_no_threshold_is_hardcoded`` enforces it.
  * **a null result is not the claim.** A non-significant McNemar p is available for free from an
    underpowered run; the gate is the confidence interval, and the control arm is what makes the
    interval mean anything.
  * **the arms must be the same experiment.** Different episodes, different start steps or a
    different checkpoint make the discordant counts a comparison of two different things.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STEP5 = ROOT / "examples" / "downstream_step5"


def _load(rel: str, name: str):
    p = STEP5 / rel
    if not p.is_file():
        pytest.skip(f"{p} is not present in this checkout")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def S():
    return _load("step5_stats.py", "step5_stats")


@pytest.fixture(scope="module")
def P():
    import sys
    sys.path.insert(0, str(STEP5))
    return _load("ingest_pusht.py", "ingest_pusht")


@pytest.fixture(scope="module")
def G():
    import sys
    sys.path.insert(0, str(STEP5))
    return _load("paired_gate.py", "paired_gate")


@pytest.fixture(scope="module")
def SPEC():
    return json.loads((ROOT / "scratch" / "certification.json").read_text())["downstream"]


# ------------------------------------------------------------------ the frozen arithmetic


def test_the_gate_reads_the_frozen_step_2_policy(S, SPEC):
    """Not a copy of it. Step 4 made the same guarantee about Gate A and it is the reason a
    threshold cannot drift between the policy and the thing that applies it."""
    assert S.SPEC == SPEC
    assert S.SPEC["equivalence_margin_points"] == 5.0
    assert S.SPEC["min_paired_episodes"] == 300


def test_no_threshold_is_hardcoded(S):
    """Every number the verdict turns on must arrive through ``SPEC``.

    Scanned by AST rather than by eye: a literal margin typed into a comparison is exactly the edit
    that survives review and then quietly disagrees with the policy it claims to implement.
    """
    src = (STEP5 / "step5_stats.py").read_text()
    tree = ast.parse(src)
    banned = {5.0, 300, 0.19, -20.0, 0.01, 6, 50}
    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool) and node.value in banned:
                offenders.append((fn.name, node.value, node.lineno))
    assert not offenders, (
        f"threshold literals inside functions: {offenders}. Read them from SPEC instead")


def test_mcnemar_is_exact_at_the_corner_the_control_lands_in(S):
    """``c = 0`` is where the chi-square approximation is least trustworthy and where the
    destructive control is supposed to land, so the exact test is the one that has to be right."""
    assert S.mcnemar_exact(0, 0) == 1.0
    assert S.mcnemar_exact(12, 11) == pytest.approx(1.0)
    assert S.mcnemar_exact(46, 0) == pytest.approx(2.84e-14, rel=0.02)
    assert S.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)


def test_the_interval_reproduces_step_2s_own_worked_example(S, SPEC):
    """Step 2 recorded the PushT numbers it validated its rule against. Recomputing them from the
    formula is the cheapest possible check that this module implements THAT rule."""
    v = SPEC["validated_against_existing_evidence"]
    n, b, c = v["n"], v["b"], v["c"]
    off, ours = [1] * n, [1] * n
    for i in range(b):
        ours[i] = 0                      # official won, ours lost
    for i in range(b, b + c):
        off[i] = 0                       # ours won, official lost
    r = S.paired(off, ours)
    assert (r["b"], r["c"], r["n"]) == (v["b"], v["c"], 300)
    assert r["delta_points"] == pytest.approx(v["delta_points"], abs=0.01)
    assert r["ci"][0] == pytest.approx(v["ci"][0], abs=0.01)
    assert r["ci"][1] == pytest.approx(v["ci"][1], abs=0.01)
    assert r["mcnemar_p"] == pytest.approx(v["mcnemar_p"], abs=1e-9)


def test_wilson_stays_inside_the_unit_interval(S):
    """A 300-episode arm at 88% sits close enough to the ceiling that the textbook interval runs
    past 100, which is why the arms are reported with Wilson."""
    lo, hi = S.wilson(264, 300)
    assert 0 <= lo < 88.0 < hi <= 100
    lo, hi = S.wilson(50, 50)
    assert hi <= 100.0 and lo > 80


# ------------------------------------------------------------------ the verdict


def _arms(n=300, b=0, c=0):
    off = [1] * n
    ours = [1] * n
    for i in range(b):
        ours[i] = 0
    for i in range(b, b + c):
        off[i] = 0
    return off, ours


def test_equivalence_passes_only_with_a_degraded_control(S):
    off, ours = _arms(300, 12, 11)
    ctrl = [0] * 300
    assert S.gate_h(off, ours, ctrl, seeds=[i // 50 for i in range(300)],
                    per_seed_n=[50] * 6)["verdict"] == "PASS"


def test_without_a_degraded_control_the_result_is_inconclusive_not_a_pass(S):
    """The whole point of the control. Success here is computed from simulator state and never
    reads a pixel, so 'the score did not move' is evidence only if the score CAN move."""
    off, ours = _arms(300, 12, 11)
    g = S.gate_h(off, ours, [1] * 300, seeds=[i // 50 for i in range(300)], per_seed_n=[50] * 6)
    assert g["verdict"] == "INCONCLUSIVE"
    assert not g["control"]["degraded"]


def test_a_missing_control_is_also_inconclusive(S):
    off, ours = _arms(300, 12, 11)
    g = S.gate_h(off, ours, None, seeds=[i // 50 for i in range(300)], per_seed_n=[50] * 6)
    assert g["verdict"] == "INCONCLUSIVE" and g["control"]["present"] is False


def test_a_real_degradation_fails_rather_than_passing_quietly(S):
    """30 episodes lost and none won back: 10 points, twice the margin, and the interval says so."""
    off, ours = _arms(300, 30, 0)
    g = S.gate_h(off, ours, [0] * 300, seeds=[i // 50 for i in range(300)], per_seed_n=[50] * 6)
    assert g["verdict"] == "FAIL"
    assert g["equivalence"]["ci"][1] < -g["margin_points"] + 5


def test_a_noisy_experiment_is_inconclusive_rather_than_a_failure(S, SPEC):
    """High discordance widens the interval past the margin without any renderer difference at all.
    Calling that a FAIL would blame the renderer for the sample size."""
    n = 300
    off, ours = _arms(n, 40, 40)
    g = S.gate_h(off, ours, [0] * n, seeds=[i // 50 for i in range(n)], per_seed_n=[50] * 6)
    assert g["power_guard"]["underpowered"] is True
    assert g["verdict"] == "INCONCLUSIVE"
    assert g["power_guard"]["required_n"] > n


def test_a_short_run_is_a_design_shortfall_not_a_pass(S):
    off, ours = _arms(100, 2, 2)
    g = S.gate_h(off, ours, [0] * 100, seeds=[i // 50 for i in range(100)], per_seed_n=[50] * 2)
    assert g["verdict"] == "DESIGN_SHORTFALL"
    assert g["design"]["met"] is False


def test_the_design_requires_the_seeds_step_2_asked_for(S, SPEC):
    """300 episodes from one seed is not the design: the margin was justified against the
    SEED-TO-SEED spread, so a single seed cannot support it."""
    off, ours = _arms(300, 5, 5)
    g = S.gate_h(off, ours, [0] * 300, seeds=[42] * 300, per_seed_n=[300])
    assert g["design"]["met"] is False
    assert g["verdict"] == "DESIGN_SHORTFALL"


def test_unequal_arms_refuse_to_pair(S):
    with pytest.raises(ValueError):
        S.paired([1, 1, 1], [1, 1])


# ------------------------------------------------------------------ the PushT audit


def test_the_pusht_audit_reproduces_the_committed_artifact(P, S, SPEC):
    """Read from the RAW per-episode outcomes, not from the write-up, and checked against the
    numbers Step 2 recorded when it validated its own rule."""
    runs = P.load_runs(P.RESULTS)
    if not runs:
        pytest.skip("examples/lewm_pusht/results is not present in this checkout")
    rows = P.episodes(runs)
    off, seeds, _ = P.series(rows, "official")
    ours, _, _ = P.series(rows, "stable_envs")
    r = S.paired(off, ours)
    v = SPEC["validated_against_existing_evidence"]
    assert (r["b"], r["c"], r["n"]) == (v["b"], v["c"], v["n"])
    assert r["delta_points"] == pytest.approx(v["delta_points"], abs=0.01)


def test_the_pusht_arms_are_actually_paired(P):
    runs = P.load_runs(P.RESULTS)
    if not runs:
        pytest.skip("examples/lewm_pusht/results is not present in this checkout")
    assert P.check_pairing(runs)["paired"], "the committed PushT runs are not episode-for-episode"


def test_a_mismatched_episode_list_is_caught(P):
    """The check has to be able to fail, or it is decoration."""
    runs = P.load_runs(P.RESULTS)
    if not runs:
        pytest.skip("examples/lewm_pusht/results is not present in this checkout")
    bad = [dict(r) for r in runs]
    for r in bad:
        if r["arm"] == "stable_envs" and r["seed"] == 42:
            r["episodes_idx"] = list(reversed(r["episodes_idx"]))
    out = P.check_pairing(bad)
    assert not out["paired"] and any("episode indices" in p for p in out["problems"])


def test_the_control_arm_degradation_is_measured_paired(P, S):
    """The control ran on one seed. Comparing its 50 episodes against the 300-episode official
    aggregate would mix two different sets of tasks; the gate pairs it against the same 50."""
    runs = P.load_runs(P.RESULTS)
    if not runs:
        pytest.skip("examples/lewm_pusht/results is not present in this checkout")
    rows = P.episodes(runs)
    cseeds = sorted({r["seed"] for r in rows if r["arm"] == "blank"})
    ctrl, _, ce = P.series(rows, "blank")
    off, _, oe = P.series(rows, "official", seeds=cseeds)
    assert ce == oe
    r = S.paired(off, ctrl)
    assert r["delta_points"] < -20.0 and r["mcnemar_p"] < 0.01


# ------------------------------------------------------------------ the comparability diff


def _run(arm, seed=42, n=4, succ=None, ckpt="abc", horizon=5):
    return {
        "env": "swm/ReacherDMControl-v0", "task": "qpos_match", "model_family": "prejepa",
        "arm": arm, "seed": seed, "num_eval": n,
        "success_rate": 100.0 * sum(succ or [1] * n) / n,
        "episode_successes": succ or [1] * n,
        "episodes_idx": list(range(n)), "start_steps": [10] * n,
        "goal_offset": 25, "eval_budget": 50, "eval_seconds": 1.0,
        "config": {"plan_config": {"horizon": horizon}, "solver": {"num_samples": 300}},
        "provenance": {"checkpoint": {"policy": "reacher/prejepa", "files": {"w.pt": ckpt}},
                       "versions": {"torch": "2.14.0"}, "host": "node1", "gpu": "3090"},
    }


def test_a_checkpoint_difference_between_arms_is_fatal(G, tmp_path):
    """If the arms ran different weights, the discordant counts are a model comparison wearing a
    renderer's name."""
    for r in (_run("official"), _run("stable_envs", ckpt="deadbeef")):
        (tmp_path / f"{r['arm']}.json").write_text(json.dumps(r))
    out = G.compare_arms(G.load(tmp_path))
    assert not out["comparable"]
    assert any("checkpoint" in p for p in out["problems"])


def test_a_planner_difference_between_arms_is_fatal(G, tmp_path):
    for r in (_run("official"), _run("stable_envs", horizon=9)):
        (tmp_path / f"{r['arm']}.json").write_text(json.dumps(r))
    out = G.compare_arms(G.load(tmp_path))
    assert not out["comparable"] and any("plan_config" in p for p in out["problems"])


def test_the_host_and_gpu_may_differ_without_failing_the_comparison(G, tmp_path):
    """Arms run in different allocations. Where they ran is provenance, not an experimental
    variable -- and demanding it match would fail every real campaign."""
    a, b = _run("official"), _run("stable_envs")
    b["provenance"]["host"], b["provenance"]["gpu"] = "node9", "3090"
    for r in (a, b):
        (tmp_path / f"{r['arm']}.json").write_text(json.dumps(r))
    assert G.compare_arms(G.load(tmp_path))["comparable"]


def test_identical_arms_compare_clean(G, tmp_path):
    for r in (_run("official"), _run("stable_envs"), _run("blank", succ=[0, 0, 0, 0])):
        (tmp_path / f"{r['arm']}.json").write_text(json.dumps(r))
    out = G.compare_arms(G.load(tmp_path))
    assert out["comparable"] and out["arm_runs_checked"] == 2


# ------------------------------------------------------------------ the project verdict


@pytest.fixture(scope="module")
def R():
    p = ROOT / "scratch" / "step5" / "report.py"
    if not p.is_file():
        pytest.skip(f"{p} is not present in this checkout")
    import sys
    sys.path.insert(0, str(STEP5))
    spec = importlib.util.spec_from_file_location("step5_report", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_arm_is_not_a_project_verdict(R):
    """Step 2 named two representatives. A gate computed from the one that finished first would be
    the easiest possible way to publish a PASS."""
    assert R.project_verdict({"PushT": "PASS"}) == "INCOMPLETE"
    assert R.project_verdict({}) == "INCOMPLETE"


def test_a_failure_anywhere_fails_the_project(R):
    assert R.project_verdict({"PushT": "PASS", "Reacher": "FAIL"}) == "FAIL"
    assert R.project_verdict({"PushT": "FAIL", "Reacher": "INCONCLUSIVE"}) == "FAIL"


def test_an_inconclusive_arm_is_not_a_pass(R):
    assert R.project_verdict({"PushT": "PASS", "Reacher": "INCONCLUSIVE"}) == "INCONCLUSIVE"
    assert R.project_verdict({"PushT": "PASS", "Reacher": "DESIGN_SHORTFALL"}) == "INCONCLUSIVE"


def test_both_passing_is_the_only_pass(R):
    assert R.project_verdict({"PushT": "PASS", "Reacher": "PASS"}) == "PASS"


# ------------------------------------------------------------------ the destructive control


@pytest.fixture(scope="module")
def SW():
    import sys
    sys.path.insert(0, str(STEP5 / "reacher"))
    return _load("reacher/swap_renderer.py", "step5_swap_renderer")


def test_the_blank_arm_keeps_the_tensor_contract_it_is_supposed_to_break_nothing_else(SW):
    """Section 10: the control removes the INFORMATION and leaves shape, dtype and pipeline alone.
    A control that also changed the frame's shape would be testing the transform stack instead."""
    r = SW.Renderer(size=224, mode="blank")
    live = r.from_live_env(object())          # never touches the env in blank mode
    stored = r.from_state([[0.1, 0.2], [0.3, 0.4]])
    assert live.shape == (224, 224, 3) and live.dtype.name == "uint8"
    assert stored.shape == (2, 224, 224, 3) and stored.dtype.name == "uint8"
    assert live.min() == live.max() == 255, "blank frames must carry no scene information"


def test_the_blank_arm_needs_no_renderer_at_all(SW):
    """It must not quietly build a GPU renderer: the control has to be runnable in an interpreter
    where our renderer is unavailable, or it is not an independent arm."""
    r = SW.Renderer(size=84, mode="blank")
    assert r.env is None and r.slots is None


def test_as_placed_refuses_to_invent_a_target_position(SW):
    """If the corpus does not record where the target was, a stored frame cannot be regenerated
    with the target where the episode put it. Guessing would put a red disc in the wrong place in
    one arm only."""
    r = SW.Renderer(size=84, mode="blank")
    # blank short-circuits before the check, so drive the fields the stored-frame path reads
    r.mode, r.dataset_target_mode = "stable_envs", "as_placed"
    r.env, r.slots, r.nq = object(), (), 2
    with pytest.raises(KeyError):
        r.from_state([[0.0, 0.0]], target_pos=None)


def test_the_offscreen_placement_is_outside_any_reacher_scene(SW):
    """The reacher arm reaches 0.22 m and the camera sees roughly +/-0.3 m. The constant has to be
    far enough out that no state can bring the geom back into frame.

    Retained although ``hidden`` has superseded it: ``offscreen`` is the mechanism Step 5 ran under
    and ``examples/results/step5/reacher/`` is only reproducible while it still works.
    """
    assert min(abs(v) for v in SW.OFFSCREEN[:2]) > 1.0


def test_hiding_the_target_is_a_style_leaf_rather_than_a_placement(SW):
    """Step 6's replacement, asserted where the placement is: the harness must express "the
    reference does not draw its target" through the env's own appearance space, not by moving the
    geometry. ``visibility.target`` is the leaf ``DMCReacher*`` declares."""
    assert SW.VISIBILITY == "visibility.target"
    r = SW.Renderer(size=84, mode="blank", target_mode="hidden")
    assert r._variation("hidden") == {SW.VISIBILITY: 0}
    assert r._variation("as_placed") is None
    assert r._variation("offscreen") is None        # the placement does the work in that mode


def test_the_default_target_mode_is_the_representation_not_the_workaround(SW):
    r = SW.Renderer(size=84, mode="blank")
    assert r.target_mode == "hidden" and r.dataset_target_mode == "hidden"


def test_the_placements_and_modes_are_the_ones_the_arms_record(SW):
    """A mode the arm does not record is a mode nobody can tell was used.

    ``official_regen`` is the control that separates "our renderer" from "regenerated at all": it
    re-renders the stored frames with MuJoCo. The corpus was recorded with a floor texture the
    environment no longer draws, so a stored frame sits ~18/255 from ANY re-render of its own
    state -- and an arm that drops under this mode drops for reasons that are not ours.
    """
    assert set(SW.TARGET_MODES) == {"hidden", "offscreen", "as_placed"}
    assert set(SW.MODES) == {"stable_envs", "blank", "official_regen"}


def test_the_official_regeneration_control_builds_no_stable_envs_renderer(SW):
    """It must draw with MuJoCo and nothing else, or it is not a control."""
    import inspect
    src = inspect.getsource(SW.Renderer.__init__)
    i_official = src.index("official_regen")
    i_se = src.index("import stable_envs")
    assert i_official < i_se, (
        "the official_regen branch must be taken before the stable_envs import, or the control "
        "arm quietly constructs the renderer it exists to exclude")



def test_the_two_surfaces_can_place_the_target_differently(SW):
    """The live env hides its target and the stored corpus draws it. One policy for both surfaces
    would regenerate a goal image without the landmark the model steers by -- which is a change to
    the planner's objective wearing a renderer's name."""
    r = SW.Renderer(size=84, mode="blank", target_mode="offscreen",
                    dataset_target_mode="as_placed")
    assert r.target_mode == "offscreen" and r.dataset_target_mode == "as_placed"


def test_the_dataset_surface_defaults_to_the_live_policy_when_unset(SW):
    """Backwards-compatible: an older invocation that named one mode still gets one behaviour."""
    r = SW.Renderer(size=84, mode="blank", target_mode="as_placed")
    assert r.dataset_target_mode == "as_placed"


def test_an_unknown_placement_is_refused_on_either_surface(SW):
    for kwargs in ({"target_mode": "wherever"}, {"dataset_target_mode": "wherever"}):
        with pytest.raises(ValueError):
            SW.Renderer(size=84, mode="blank", **kwargs)


def test_the_power_guard_is_about_n_not_about_a_constant(S):
    """0.19 is the discordance a 300-episode run can resolve -- 3.8416*0.19/0.05^2 = 292 -- not a
    universal ceiling. Comparing against it at any n turns a powered FAIL into an INCONCLUSIVE,
    which is the comfortable answer rather than the true one."""
    off, ours = _arms(600, 132, 78)                      # discordance 0.35, required n = 538
    g = S.gate_h(off, ours, [0] * 600, seeds=[i // 50 for i in range(600)], per_seed_n=[50] * 12)
    assert g["power_guard"]["required_n"] == 538
    assert g["power_guard"]["underpowered"] is False, "600 >= 538 is powered"
    assert g["verdict"] == "FAIL"


def test_the_same_discordance_at_half_the_episodes_is_underpowered(S):
    """The identical effect at n=300 cannot resolve the margin, and must not be called a failure."""
    off, ours = _arms(300, 66, 39)
    g = S.gate_h(off, ours, [0] * 300, seeds=[i // 50 for i in range(300)], per_seed_n=[50] * 6)
    assert g["power_guard"]["underpowered"] is True
    assert g["verdict"] == "INCONCLUSIVE"

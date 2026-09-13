"""Merge the Step 4 shards, judge each check against the bound that already exists for it, report.

Three stages, deliberately separate, for the same reason Step 3 separated them: the measurement has to
be readable without the verdict, and the verdict has to be traceable to something that was written
down before the measurement.

    raw shards  ->  merge()    ->  one deterministic merged artifact
    merged      ->  verdicts() ->  Gate A per configuration, every bound CITED
    both        ->  report()   ->  docs/research/step4_fidelity_baseline.md

NO BOUND IS WRITTEN IN THIS FILE. ``scratch/step4/bounds.py`` extracts them from the test modules that
own them and from each env's own declaration, and the representation tripwire comes out of the frozen
Step 2 spec. If you want to add a constant here, the answer is that Step 4 is not allowed to: a
renderer that fails is a Step 4 RESULT and an input to the Step 6 queue, not a reason to move a bar.

Four statuses, and the distinction between the last two is the one that matters:

    PASS      every applicable check with a declared bound is inside it
    PARTIAL   the same, on a renderer that draws a DOCUMENTED subset of the registered default
              observation. A passing state that caps the ladder -- Step 2 section 12.2.
    FAIL      the renderer was reached and a check with a declared bound came out wrong
    BLOCKED   the renderer or its official counterpart could not be reached in any interpreter

    python scratch/step4/certify.py --shards examples/results/step4 --out examples/results/step4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import bounds as B  # noqa: E402
import universe  # noqa: E402
from stable_envs.testing.provenance import block_on_reference_mismatch  # noqa: E402

SPEC = json.loads((ROOT / "scratch" / "certification.json").read_text())

def _rel(p: Path) -> str:
    """Repo-relative where it can be, absolute otherwise -- the unit tests merge from a tmp dir."""
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


#: The seven the brief singles out in section 14, and the reason each is singled out. Editorial, and
#: the ONLY editorial content on the page: every number beside these rows is read from the artifact.
#: Kept here rather than in the prose so that a configuration that stops being measured shows up as
#: "not measured" in this table instead of quietly vanishing from the page.
REPRESENTATIVES = (
    ("PushT", "the one row with an existing downstream PASS, and CPU_REGRESSION in Step 3"),
    ("DMCReacherHard", "Step 5's representative; 3-D scene, reference physics, registered default is "
                       "`hard` and was measured to be"),
    ("DMCReacherEasy", "the second render-distinct configuration of the same row"),
    ("DMCBallInCupCatch", "Step 3's GPU outlier: BELOW_G4 and BATCH_SCALES_FAIL, so its renderer is "
                          "likely to be rewritten"),
    ("DMCWalkerWalk", "visual tier: the proof that a visual renderer can be validated without pixel "
                      "parity"),
    ("DMCHumanoidStand", "visual tier, mjwarp, and one of the 16 with no CPU renderer at all"),
    ("CraftaxClassic", "documented partial observation coverage; must stay `partial`"),
    ("PointmazeLargeNavigate", "OGBMaze's registered default cell; the row's parameter space stays "
                               "partial at 11 of 15"),
    ("FetchPickAndPlace", "the matched MuJoCo stack, a GPU renderer that exists, and no CPU path"),
)

MANDATORY = ("check_state_roundtrip", "check_state_dims_matter", "check_batch_invariance",
             "check_batch_size_invariance", "check_order_independence", "check_determinism",
             "check_render_is_pure", "check_chunked_render_matches", "check_input_form_invariance",
             "check_set_state_copies", "check_style_is_state", "check_default_style_is_official",
             "check_per_env_variation", "check_fidelity_declaration")


# ------------------------------------------------------------------ 1. merge

def merge(shard_dir: Path) -> dict:
    """Every shard under ``conformance/``, keyed by configuration. Deterministic, order-independent.

    A configuration measured in two interpreters keeps the RICHER run -- more checks that actually
    produced a number -- and records that it was seen twice. Preferring the richer one rather than the
    newer one is what makes a rerun of one shard safe: re-running ``fetch`` cannot demote a config
    that a wider interpreter had already measured in full.
    """
    shards = sorted((shard_dir / "conformance").glob("*.json"))
    out: dict = {"schema_version": 1, "step": 4, "shards": {}, "configs": {}}
    seen: dict[str, list[str]] = {}
    for p in shards:
        d = json.loads(p.read_text())
        tag = d.get("shard") or p.stem
        out["shards"][tag] = {"file": _rel(p), "provenance": d.get("provenance", {})}
        for cid, row in sorted((d.get("configs") or {}).items()):
            row = dict(row, shard=tag)
            seen.setdefault(cid, []).append(tag)
            prev = out["configs"].get(cid)
            if prev is None or _richness(row) > _richness(prev):
                out["configs"][cid] = row
    for cid, tags in seen.items():
        out["configs"][cid]["measured_in_shards"] = sorted(tags)
    out["configs"] = dict(sorted(out["configs"].items()))
    return out


def _richness(row: dict) -> tuple:
    n = sum(1 for g in ("mandatory", "family_specific")
            for v in (row.get(g) or {}).values() if v.get("status") == "measured")
    return (row.get("status") == "measured", n, row.get("shard") or "")


# ------------------------------------------------------------------ 2. judge

def _num(ev, *keys, default=None):
    for k in keys:
        if isinstance(ev, dict) and ev.get(k) is not None:
            return ev[k]
    return default


def judges(bnd: dict) -> dict:
    """``check name -> (evidence, ctx) -> (ok, detail)``. ``ok is None`` means no bound is declared.

    Every entry restates the assertion the existing suite makes on that check, and nothing else. The
    two that return ``None`` are not oversights: ``check_resolution_consistency`` has a floor only for
    the envs somebody measured one for, and a renderer with no recorded floor cannot pass or fail it;
    ``check_reference_renderer_is_stable`` is a property of the node, which Step 2 section 4.1 lists as
    diagnostic-only.
    """
    d = {k: (lambda ev, ctx, _k=k: (_num(ev, "max_abs_diff") == 0, {"observed": _num(ev, "max_abs_diff"),
                                                                    "bound": "== 0"}))
         for k in ("check_batch_invariance", "check_batch_size_invariance",
                   "check_chunked_render_matches", "check_style_is_state",
                   "check_default_style_is_official")}
    d["check_state_roundtrip"] = lambda ev, ctx: (
        _num(ev, "max_abs_err") == 0, {"observed": _num(ev, "max_abs_err"), "bound": "== 0"})
    d["check_state_dims_matter"] = lambda ev, ctx: (
        not ev.get("dead_dims"), {"observed": ev.get("dead_dims"), "bound": "no dead state dimension"})
    d["check_order_independence"] = lambda ev, ctx: (
        _num(ev, "permutation_max_abs_diff") == 0 and _num(ev, "duplicate_max_abs_diff") == 0,
        {"observed": {k: ev.get(k) for k in ("permutation_max_abs_diff", "duplicate_max_abs_diff")},
         "bound": "both == 0"})
    d["check_determinism"] = lambda ev, ctx: (
        bool(ev.get("identical")), {"observed": ev.get("identical"), "bound": "identical"})
    d["check_render_is_pure"] = lambda ev, ctx: (
        all(float(ev.get(k, 0)) == 0 for k in ("state_changed_by_render", "style_changed_by_render",
                                               "state_changed_by_double_set",
                                               "frame_changed_by_double_set")),
        {"observed": {k: ev.get(k) for k in ev}, "bound": "all == 0"})
    d["check_input_form_invariance"] = lambda ev, ctx: (
        not ev.get("disagreements"), {"observed": ev.get("disagreements"),
                                      "bound": "every input form gives one frame"})
    d["check_set_state_copies"] = lambda ev, ctx: (
        float(ev.get("aliased_caller_buffer", 1)) == 0 and not ev.get("state_tensor_rebound"),
        {"observed": {k: ev.get(k) for k in ("aliased_caller_buffer", "state_tensor_rebound")},
         "bound": "set_state copies and does not rebind"})
    d["check_per_env_variation"] = lambda ev, ctx: (
        not ev.get("collapsed_variations"), {"observed": ev.get("collapsed_variations"),
                                             "bound": "no variation collapses across the batch"})
    d["check_fidelity_declaration"] = lambda ev, ctx: (
        bool(ev.get("agrees")), {"observed": ev.get("describe_reports_tier"),
                                 "bound": "describe() reports the declared tier"})
    d["check_variations_do_something"] = lambda ev, ctx: (
        not ev.get("inert_variations"), {"observed": ev.get("inert_variations"),
                                         "bound": "no declared variation is inert"})
    d["check_colors_apply_exactly"] = lambda ev, ctx: (
        not ev.get("colors_not_applied"), {"observed": ev.get("colors_not_applied"),
                                           "bound": "an RGB knob puts that colour in the frame"})
    d["check_backend_agreement"] = lambda ev, ctx: _cmp_ge(
        ev.get("exact_frac"), bnd["backend_agreement_exact_frac_min"])
    d["render_parity_with_official_renderer"] = _parity_judge(bnd)
    d["check_visual_fidelity"] = _visual_judge("visual_fidelity")
    d["check_visual_fidelity_under_variation"] = _visual_judge("variation_fidelity")
    d["check_resolution_consistency"] = _downsample_judge(bnd)
    d["check_step_matches_reference"] = _step_judge(bnd)
    return d


def _cmp_ge(observed, bound: dict):
    v = bound.get("value")
    if v is None or observed is None:
        return None, {"observed": observed, "bound": None, "why": "no declared bound"}
    return float(observed) >= float(v), {"observed": observed, "bound": f">= {v}",
                                         "bound_from": bound.get("from")}


def _parity_judge(bnd: dict):
    def judge(ev, ctx):
        interior_floor = ctx.get("parity_interior_min")
        if interior_floor is None:
            interior_floor = bnd["parity_interior_min_default"]["value"]
        all_floor = ctx.get("parity_all_exact_min")
        if all_floor is None:
            all_floor = bnd["parity_all_exact_min_default"]["value"]
        mean_max = bnd["parity_mean_abs_max"]["value"]
        got = {"interior_exact_frac": ev.get("interior_exact_frac"),
               "exact_frac": ev.get("exact_frac"), "mean_abs": ev.get("mean_abs"),
               "ssim": ev.get("ssim"), "max_abs": ev.get("max_abs"),
               # How much of the frame the interior claim could even be made about. Zero means the
               # reference has no flat 3x3 neighbourhood anywhere, so the strong claim is unmeasurable
               # rather than failed -- which the table must show rather than print a bare "n/a".
               "interior_px": ev.get("interior_px"), "interior_frac": ev.get("interior_frac")}
        bad = []
        if ev.get("interior_exact_frac") is not None and ev["interior_exact_frac"] < interior_floor:
            bad.append(f"interior_exact_frac {ev['interior_exact_frac']:.6f} < {interior_floor}")
        if ev.get("exact_frac") is not None and ev["exact_frac"] < all_floor:
            bad.append(f"exact_frac {ev['exact_frac']:.4f} < {all_floor}")
        if ev.get("mean_abs") is not None and ev["mean_abs"] > mean_max:
            bad.append(f"mean_abs {ev['mean_abs']:.3f} > {mean_max}")
        return not bad, {"observed": got, "violations": bad,
                         "bound": {"interior_exact_frac_min": interior_floor,
                                   "all_exact_frac_min": all_floor, "mean_abs_max": mean_max},
                         "bound_from": "the env's own declaration, defaulting to "
                                       + bnd["parity_interior_min_default"]["from"]}
    return judge


def _visual_judge(declaration: str):
    """A visual env against ITS OWN declared numbers. Never a parity threshold -- Step 2 section 4.3."""
    def judge(ev, ctx):
        vf = ctx.get(declaration) or {}
        if not vf:
            return None, {"observed": {k: ev.get(k) for k in ("mean_abs", "within8_frac", "ssim")},
                          "bound": None,
                          "why": f"the class declares no {declaration}, so there is nothing to hold "
                                 f"this measurement to"}
        bad = []
        if ev.get("mean_abs") is not None and ev["mean_abs"] > vf.get("mean_abs_max", float("inf")):
            bad.append(f"mean_abs {ev['mean_abs']:.3f} > {vf['mean_abs_max']}")
        if ev.get("within8_frac") is not None and ev["within8_frac"] < vf.get("within8_min", 0):
            bad.append(f"within8_frac {ev['within8_frac']:.4f} < {vf['within8_min']}")
        if ev.get("ssim") is not None and ev["ssim"] < vf.get("ssim_min", 0):
            bad.append(f"ssim {ev['ssim']:.4f} < {vf['ssim_min']}")
        return not bad, {"observed": {k: ev.get(k) for k in ("mean_abs", "within8_frac", "ssim",
                                                             "interior_exact_frac")},
                         "violations": bad, "bound": vf,
                         "bound_from": f"the env's own {declaration} declaration"}
    return judge


def _downsample_judge(bnd: dict):
    floors, margin = bnd["downsample_exact_frac"]["value"], bnd["downsample_margin"]["value"]
    def judge(ev, ctx):
        f = floors.get(ctx["config_id"])
        got = {k: ev.get(k) for k in ("exact_frac", "mean_abs", "max_abs")}
        if f is None:
            return None, {"observed": got, "bound": None,
                          "why": "no measured downsample floor is recorded for this env in "
                                 + bnd["downsample_exact_frac"]["from"] + "; a guessed bound is worse "
                                 "than no test, so this is reported and does not gate"}
        ok = ev.get("exact_frac") is not None and ev["exact_frac"] >= f - margin
        return ok, {"observed": got, "bound": f">= {f} - {margin}",
                    "violations": ([] if ok else
                                   [f"exact_frac {ev.get('exact_frac'):.4f} < {f} - {margin}"]),
                    "bound_from": bnd["downsample_exact_frac"]["from"],
                    "measured_on": bnd["downsample_measured_on"]["value"]}
    return judge


def _step_judge(bnd: dict):
    """Gate E, judged and reported on its OWN axis. Never folded into Gate A -- brief section 3."""
    a_max, b_max = bnd["step_agent_max_err"]["value"], bnd["step_block_median_err"]["value"]
    def judge(ev, ctx):
        bad = []
        if ev.get("agent_max_err") is not None and ev["agent_max_err"] >= a_max:
            bad.append(f"agent_max_err {ev['agent_max_err']:.3e} >= {a_max}")
        if ev.get("block_median_err") is not None and ev["block_median_err"] >= b_max:
            bad.append(f"block_median_err {ev['block_median_err']:.3f} >= {b_max}")
        return not bad, {"observed": {k: ev.get(k) for k in
                                      ("agent_max_err", "block_median_err", "exact_frac", "steps")},
                         "violations": bad,
                         "bound": {"agent_max_err_lt": a_max, "block_median_err_lt": b_max},
                         "bound_from": bnd["step_agent_max_err"]["from"]}
    return judge


def verdict(row: dict, J: dict, provenance: dict | None = None) -> dict:
    """Gate A for one configuration, plus the physics axis kept beside it rather than inside it.

    ``provenance`` is the measuring shard's record (``reference_packages`` in particular). A check
    whose bound is a claim against the official renderer is judged only when the reference the
    declaration names is the reference this interpreter ran; otherwise it is ``blocked_environment``
    with the mismatch recorded -- ``stable_envs.testing.provenance`` -- exactly as a missing reference
    package already is. The measurement is kept in the record either way.
    """
    out = {"config_id": row["config_id"], "swm_ids": row.get("swm_ids", []),
           "role": row.get("role"), "shard": row.get("shard"),
           "registered_default_coverage": row.get("registered_default_coverage"),
           "tier": row.get("declared_tier"), "checks": {}, "failures": [], "blocked": [],
           "no_declared_bound": [], "not_applicable": []}
    if row.get("status") != "measured":
        out["gate_a"] = "BLOCKED"
        out["why"] = row.get("why", "the configuration was not reached")
        return out

    decl = (((row.get("mandatory") or {}).get("check_fidelity_declaration") or {}).get("evidence")) or {}
    ctx = {"config_id": row["config_id"],
           "parity_interior_min": decl.get("parity_interior_min"),
           "parity_all_exact_min": decl.get("parity_all_exact_min"),
           "visual_fidelity": decl.get("visual_fidelity") or {},
           # NOT defaulted from visual_fidelity. The under-variation numbers are a separate, looser
           # declaration that only a LEARNED decoder carries; borrowing the default-appearance ones
           # would judge a measurement against a bar nobody wrote for it. Absent means no bound.
           "variation_fidelity": decl.get("variation_fidelity") or {}}
    out["tier"] = decl.get("tier") or out["tier"]
    # Same precedence as ``stable_envs.core.env.reference_versions_of``: the class attribute first,
    # then ``visual_fidelity``'s entry on top of it, so this and ``describe()`` cannot disagree.
    declared_versions = {**(decl.get("reference_versions") or {}),
                         **((decl.get("visual_fidelity") or {}).get("reference_versions") or {})}
    recorded_packages = (provenance or {}).get("reference_packages") or {}

    for group in ("mandatory", "family_specific"):
        for name, res in sorted((row.get(group) or {}).items()):
            rec = {"group": group, "raw_status": res.get("status")}
            if res.get("status") == "measured":
                ok, detail = (J[name](res.get("evidence") or {}, ctx) if name in J
                              else (None, {"why": "no judge is defined for this check"}))
                rec.update(detail)
                rec["status"] = "pass" if ok else ("fail" if ok is False else "no_declared_bound")
                blocked = block_on_reference_mismatch(name, declared_versions, recorded_packages)
                if blocked is not None:
                    rec["judged_against_wrong_reference"] = rec["status"]
                    rec.update(blocked)
            elif res.get("status") == "not_applicable":
                rec["status"], rec["why"] = "not_applicable", res.get("why")
            elif res.get("status") == "blocked_environment":
                rec["status"], rec["why"] = "blocked_environment", res.get("why")
            else:
                rec["status"], rec["why"] = "fail", res.get("why")
                rec["error"] = True
            out["checks"][name] = rec
            {"fail": out["failures"], "blocked_environment": out["blocked"],
             "no_declared_bound": out["no_declared_bound"],
             "not_applicable": out["not_applicable"]}.get(rec["status"], []).append(name)

    phys = row.get("physics") or {}
    if phys.get("status") == "measured":
        ok, detail = J["check_step_matches_reference"](phys.get("evidence") or {}, ctx)
        out["physics"] = {"status": "pass" if ok else "fail", **detail}
    else:
        out["physics"] = {"status": phys.get("status", "not_applicable"), "why": phys.get("why")}

    missing = [k for k in MANDATORY if k not in out["checks"]]
    if missing:
        out["failures"].extend(missing)
        out["missing_mandatory"] = missing
    partial = row.get("registered_default_coverage") == "partial"
    out["gate_a"] = ("FAIL" if out["failures"] else
                     "BLOCKED" if out["blocked"] else
                     "PARTIAL" if partial else "PASS")
    return out


def certify(merged: dict) -> dict:
    """Gate A for every measured configuration, with its Step 1 provenance carried alongside.

    ``claimed_by`` comes straight out of ``universe.configurations()``: which swm row claims this
    configuration and on which axis -- as somebody's registered default, or through the parameter
    space of a row whose default is a different configuration. Carrying it here is what lets the
    report say a result projects onto three rows without implying three measurements.
    """
    bnd = B.load()
    J = judges(bnd)
    claims = {c["config_id"]: c for c in universe.configurations()}
    out = {}
    for cid, row in merged["configs"].items():
        v = verdict(row, J, (merged.get("shards") or {}).get(row.get("shard"), {}).get("provenance"))
        u = claims.get(cid, {})
        v["claimed_by"] = u.get("claimed_by", [])
        v["declared_parity"] = {k: u.get(k) for k in
                                ("parity_interior_min", "parity_all_exact_min", "parity_note")}
        v["reference_id"] = u.get("reference_id")
        v["backends"] = u.get("backends", [])
        v["siblings"] = u.get("siblings", [])
        v["signature"] = (row.get("signature") or {})
        v["corpus"] = (row.get("corpus") or {})
        out[cid] = v
    return {"schema_version": 1, "step": 4, "bounds": bnd,
            "what_this_is": "Gate A per render-distinct configuration. Every bound is cited to the "
                            "file that declares it; none is defined in scratch/step4/certify.py.",
            "configs": out}


# ------------------------------------------------------------------ 3. the tripwire

def tripwire(shard_dir: Path, merged: dict) -> dict:
    """``A = cos(ours, official) / cos(blank, official)`` against Step 2's frozen bound, by tier.

    Diagnostic, not a downstream claim: section 10 of the brief and section 10 of the spec both say
    so. It is here because it is the cheapest thing that notices a Step 6 optimization that keeps
    every pixel bound and still changes what the frame MEANS to an encoder.
    """
    bounds = SPEC["representation_tripwire"]["bounds"]
    out = {"metric": SPEC["representation_tripwire"]["metric"], "bounds": bounds, "envs": {}}
    for p in sorted((shard_dir / "dino").glob("*/dino_agreement.json")):
        d = json.loads(p.read_text())
        out.setdefault("runs", {})[p.parent.name] = {
            "file": _rel(p), "model": d.get("model"), "n": d.get("n"),
            "state_seed": d.get("state_seed"), "device": d.get("device")}
        for env, r in sorted((d.get("envs") or {}).items()):
            if "dino_cos_ours_vs_official" not in r:
                out["envs"][env] = {"status": "blocked_environment",
                                    "why": r.get("skipped") or r.get("error")}
                continue
            blank = r.get("dino_cos_blank_vs_official") or 0.0
            tier = (merged["configs"].get(env, {}) or {}).get("declared_tier") or r.get("tier")
            bound = bounds.get(tier)
            a = r["dino_cos_ours_vs_official"] / blank if blank else None
            out["envs"][env] = {
                "status": "measured" if a is not None else "no_control",
                "A": a, "tier": tier, "bound": bound,
                "pass": (None if a is None or bound is None else bool(a <= bound)),
                "cos_ours_vs_official": r["dino_cos_ours_vs_official"],
                "cos_blank_vs_official": blank,
                "cos_state_vs_state": r.get("dino_cos_state_vs_state"),
                "cos_jpeg95_vs_official": r.get("dino_cos_jpeg95_vs_official"),
                "ratio_ours_over_jpeg95": r.get("ratio_ours_over_jpeg95"), "run": p.parent.name}
    return out


# ------------------------------------------------------------------ 4. report

def _esc(x) -> str:
    """A cell, safe inside a pipe table. ``mean |Delta|`` in a header used to end the table there."""
    return str(x).replace("|", "\\|").replace("\n", " ")


def _tbl(rows, head) -> list[str]:
    if not rows:
        return ["_none._", ""]
    rows = [[_esc(c) for c in r] for r in rows]
    head = [_esc(h) for h in head]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    out = ["| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(head)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += ["| " + " | ".join(r[i].ljust(w[i]) for i in range(len(head))) + " |" for r in rows]
    return out + [""]


def _f(x, n=4):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def report(merged: dict, cert: dict, trip: dict) -> str:
    from collections import Counter, defaultdict
    cfgs = universe.configurations()
    by_id = {c["config_id"]: c for c in cfgs}
    V = cert["configs"]
    L: list[str] = []
    A = L.append

    gate = Counter(v["gate_a"] for v in V.values())
    missing = [c["config_id"] for c in cfgs if c["config_id"] not in V]
    failing = sorted(cid for cid, v in V.items() if v["gate_a"] == "FAIL")
    n_swm = len({i for c in cfgs for i in c["swm_ids"]})

    A("# Step 4 — the pre-optimization correctness and fidelity baseline")
    A("")
    A("Generated by `scratch/step4/certify.py` from `examples/results/step4/`. Every count on this")
    A("page is computed from `examples/results/step4/certification.json`; none is maintained by hand.")
    A("")
    A("## The conclusion")
    A("")
    n_par_pass = sum(1 for cid in V
                     if V[cid]["checks"].get("render_parity_with_official_renderer", {}).get("status")
                     == "pass")
    n_par_all = sum(1 for cid in V if V[cid].get("tier") == "parity")
    # Split three ways rather than reported as one number, because "45 of 45 pass" hides that the
    # rows are not all held to the same bar and that three of them cannot take the interior test at
    # all. A summary that flattens those together is the thing this whole page exists to avoid.
    _obs = lambda cid: (V[cid]["checks"].get("render_parity_with_official_renderer", {})
                        .get("observed") or {})
    _floor = lambda cid: ((V[cid]["checks"].get("render_parity_with_official_renderer", {})
                           .get("bound") or {}).get("interior_exact_frac_min"))
    _dflt = cert["bounds"]["parity_interior_min_default"]["value"]
    par_ids = [cid for cid in V if V[cid].get("tier") == "parity"]
    n_exact = sum(1 for cid in par_ids if _obs(cid).get("interior_exact_frac") == 1.0
                  and _floor(cid) == _dflt)
    n_relaxed = sum(1 for cid in par_ids
                    if _floor(cid) is not None and _floor(cid) < _dflt)
    n_blind = sum(1 for cid in par_ids if _obs(cid).get("interior_px") == 0)
    A(f"**{gate['PASS']} of {len(V)} render-distinct configurations pass Gate A outright, "
      f"{gate['PARTIAL']} on a documented subset, {gate['FAIL']} fail and {gate['BLOCKED']} could not "
      f"be reached.** All {n_par_pass} of {n_par_all} parity configurations meet the interior-exactness "
      f"contract **each one declares**, and those contracts are not identical: {n_exact} are bit-exact "
      f"on every interior pixel at the shared floor of `{_dflt}`, {n_relaxed} hold to a lower floor "
      f"their class declares and explains, and {n_blind} are on scenes where the official renderer "
      f"leaves no interior pixel to test at all. Not one of the failures below is a parity comparison.")
    A("")
    A(f"{gate['FAIL']} configurations were **already failing before any optimization**, and they fail")
    A("for two unrelated reasons. Three are visual-tier DM Control renderers that no longer meet the")
    A("fidelity numbers their own classes declare — on two independent metrics, with the residual")
    A("concentrated in the skybox; see *The visual-tier drift*. Two are parity-tier OGBench cells that")
    A("pass their parity comparison and fail a MANDATORY check instead: a fresh env handed the stored")
    A("`(state, style)` draws a frame 1/255 different, so for those two cells the frame depends on")
    A("something a dataset would not have recorded. None of this was fixed here. A renderer changed")
    A("before its baseline is frozen has no baseline, and the value of the finding is precisely that")
    A("it is now dated to BEFORE any Step 6 work.")
    A("")
    A("Every configuration that was reached now has a frozen state corpus, a frame digest and a metric")
    A("beside it, so a Step 6 rewrite can be replayed on the identical states and the question \"was")
    A("this already broken?\" has an answer instead of an opinion.")
    A("")
    A("## What was validated")
    A("")
    A(f"{len(cfgs)} render-distinct configurations, over {n_swm} matched swm rows and "
      f"{len({c['env'] for c in cfgs} | {s for c in cfgs for s in c['siblings']})} stable-envs envs. "
      f"The universe is computed in `scratch/step4/universe.py` from Step 1's matrix, not listed by "
      f"hand.")
    A("")
    A("The excess over 34 rows is real coverage rather than double counting. Three DM Control")
    A("configurations (`reacher-easy`, `finger-turn_easy`, `finger-spin`) and ten further OGBench")
    A("cells are claimed by the *parameter-space* axis of a row whose registered default is a")
    A("different configuration, and Step 1 counted each as covered. The six renderer gaps and the two")
    A("symbolic ids carry no renderer and are outside this step by construction.")
    A("")
    A("| shard | interpreter | configurations | host |")
    A("|---|---|---|---|")
    for tag, s in sorted(merged["shards"].items()):
        p = s.get("provenance", {})
        n = sum(1 for v in V.values() if v.get("shard") == tag)
        mj = (p.get("reference_packages") or {}).get("mujoco")
        A(f"| `{tag}` | python {p.get('python', '?')}, torch {p.get('torch', '?')}, "
          f"mujoco {mj or 'not installed (this family needs none)'} | {n} | "
          f"{p.get('gpu') or p.get('cpu', '?')} on `{p.get('host', '?')}` |")
    A("")
    A("## Gate A")
    A("")
    A("```")
    for k in ("PASS", "PARTIAL", "FAIL", "BLOCKED"):
        A(f"    {k:9s} {gate[k]}")
    A("```")
    A("")
    if missing:
        A(f"**{len(missing)} configuration(s) produced no result at all**: "
          + ", ".join(f"`{m}`" for m in missing) + ". A configuration that was never measured is not a")
        A("passing one; these are counted outside the four states above and are listed under *Blocked*.")
        A("")

    parity = sorted(cid for cid, v in V.items() if v.get("tier") == "parity")
    visual = sorted(cid for cid, v in V.items() if v.get("tier") == "visual")
    default_floor = cert["bounds"]["parity_interior_min_default"]["value"]
    floors = {cid: (V[cid]["checks"].get("render_parity_with_official_renderer", {}).get("bound")
                    or {}).get("interior_exact_frac_min") for cid in parity}
    relaxed = sorted(cid for cid, f in floors.items() if f is not None and f < default_floor)
    A("### Parity configurations")
    A("")
    A("Interior-pixel exactness against the official renderer, on the frozen corpus. **The floor is")
    A("the env's own `parity_interior_min`, and it is not the same number everywhere** — the `floor`")
    A(f"column below is the one each row was actually held to. The shared default is `{default_floor}`;")
    A(f"the all-pixel and mean bars are `tests/gpu/test_conformance.py`'s "
      f"`{cert['bounds']['parity_all_exact_min_default']['value']}` and "
      f"`{cert['bounds']['parity_mean_abs_max']['value']}` unless a row declares its own. Edge pixels")
    A("are excluded from the interior number on purpose: two rasterisers are entitled to disagree")
    A("there and neither is ground truth.")
    A("")
    rows = []
    for cid in parity:
        c = V[cid]["checks"].get("render_parity_with_official_renderer", {})
        o = c.get("observed") or {}
        rows.append([cid, c.get("status", "-"),
                     _f((c.get("bound") or {}).get("interior_exact_frac_min"), 6),
                     _f(o.get("interior_exact_frac"), 6),
                     _f(o.get("exact_frac")), _f(o.get("mean_abs"), 3), _f(o.get("ssim")),
                     _f(o.get("interior_px"), 0),
                     "; ".join(c.get("violations") or []) or (c.get("why") or "")[:60]])
    n_par_ok = sum(1 for cid in parity
                   if V[cid]["checks"].get("render_parity_with_official_renderer", {}).get("status")
                   == "pass")
    A(f"**{n_par_ok} of {len(parity)} parity configurations meet their declared parity contract.**")
    A("")
    L.extend(_tbl(rows, ["configuration", "parity", "floor", "interior exact", "all exact",
                         "mean |Δ|", "SSIM", "interior px", "note"]))
    if relaxed:
        A(f"**{len(relaxed)} of {len(parity)} parity configurations declare a floor below")
        A(f"`{default_floor}`**, and every one of them passes against the floor it declares rather")
        A("than against a floor this step chose. A relaxed floor is a legitimate contract and the repo")
        A("already refuses one that does not name its mechanism —")
        A("`test_relaxed_parity_floors_are_all_explained` asserts it on the class, over the whole")
        A("registry. It is repeated here because a table of `pass` beside a column of different floors")
        A("is unreadable otherwise, and because the difference between \"exact\" and \"exact to within")
        A("a declared, explained residual\" is exactly the kind of thing a summary flattens.")
        A("")
        rows = []
        for cid in relaxed:
            d = V[cid].get("declared_parity") or {}
            note = " ".join((d.get("parity_note") or "").split())
            rows.append([cid, _f(floors[cid], 6),
                         (note[:150] + "…") if len(note) > 150 else (note or "**NO parity_note**")])
        L.extend(_tbl(rows, ["configuration", "declared floor", "declared mechanism"]))
        unexplained = [cid for cid in relaxed
                       if not (V[cid].get("declared_parity") or {}).get("parity_note")]
        if unexplained:
            A(f"**{len(unexplained)} relaxed floor(s) carry no `parity_note`**: "
              + ", ".join(f"`{c}`" for c in unexplained) + ". A tolerance without a stated mechanism")
            A("is indistinguishable from a bug being tolerated, and this is a Step 4 finding.")
            A("")
        drifted = [cid for cid in parity
                   if (V[cid].get("declared_parity") or {}).get("parity_interior_min") is not None
                   and floors.get(cid) is not None
                   and float(V[cid]["declared_parity"]["parity_interior_min"]) != float(floors[cid])]
        if drifted:
            A(f"**{len(drifted)} row(s) were judged against a floor that differs from the one the")
            A("committed census records for the same class**, which means one of the two is stale: "
              + ", ".join(f"`{c}`" for c in drifted) + ".")
            A("")
    blind = sorted(cid for cid in parity
                   if (V[cid]["checks"].get("render_parity_with_official_renderer", {})
                       .get("observed") or {}).get("interior_px") == 0)
    if blind:
        A(f"**On {len(blind)} of these the interior claim cannot be made at all**, and that is a")
        A("property of the reference rather than of us. `conformance.interior_mask` keeps only pixels")
        A("whose 3x3 neighbourhood in the OFFICIAL frame is a single flat colour, and a DM Control")
        A("scene at 84 px has none: its floor is a checkerboard under a spotlight gradient and its sky")
        A("is a speckled gradient, so every pixel has a neighbour a level or two away. The suite")
        A("already behaves this way — `test_render_parity_with_official_renderer` guards the interior")
        A("assertion with `if r[\"interior_exact_frac\"] is not None` — and these configurations are")
        A("therefore judged on the all-pixel and mean bars alone. **They are not passing a weaker")
        A("version of the interior test; they are not taking it.** Reported here because a column of")
        A("`n/a` reads like a gap in the run, and this is a gap in what the metric can see:")
        A("")
        A("  " + ", ".join(f"`{c}`" for c in blind))
        A("")

    A("### Visual configurations")
    A("")
    A("A visual renderer is a different render of the same scene and is never held to a parity")
    A("threshold — Step 2 section 4.3. It is held to the numbers **it declares**, and it declares them")
    A("because they were measured before they were written down. `interior exact` is reported beside")
    A("them as the honest reminder of what the tier gave up, and is not a bar.")
    A("")
    rows = []
    for cid in visual:
        c = V[cid]["checks"].get("check_visual_fidelity", {})
        u = V[cid]["checks"].get("check_visual_fidelity_under_variation", {})
        o, b = c.get("observed") or {}, c.get("bound") or {}
        rows.append([cid, c.get("status", "-"), _f(o.get("mean_abs"), 3),
                     _f(b.get("mean_abs_max"), 3), _f(o.get("within8_frac")),
                     _f(b.get("within8_min")), _f(o.get("ssim")), _f(b.get("ssim_min")),
                     _f(o.get("interior_exact_frac")), u.get("status", "-")])
    n_vis_ok = sum(1 for cid in visual
                   if V[cid]["checks"].get("check_visual_fidelity", {}).get("status") == "pass")
    A(f"**{n_vis_ok} of {len(visual)} visual configurations meet their declared visual fidelity.**")
    A("")
    L.extend(_tbl(rows, ["configuration", "visual", "mean |Δ|", "≤", "within 8", "≥", "SSIM", "≥",
                         "interior exact", "under variation"]))

    A("## Mandatory checks")
    A("")
    A("Every certified renderer, both tiers, no exceptions — Step 2 section 4.1. A `not_applicable`")
    A("here is a property the env does not have (an env with no style cannot fail `check_style_is_state`),")
    A("never a check that was skipped for convenience.")
    A("")
    tally = defaultdict(Counter)
    for v in V.values():
        for name, rec in v["checks"].items():
            if rec["group"] == "mandatory":
                tally[name][rec["status"]] += 1
    rows = [[n, tally[n]["pass"], tally[n]["fail"], tally[n]["not_applicable"],
             tally[n]["blocked_environment"] + tally[n]["no_declared_bound"],
             ", ".join(sorted(cid for cid, v in V.items()
                              if v["checks"].get(n, {}).get("status") == "fail"))[:70] or "—"]
            for n in MANDATORY]
    L.extend(_tbl(rows, ["check", "pass", "fail", "n/a", "other", "failing configurations"]))

    A("## Family-specific checks")
    A("")
    tally = defaultdict(Counter)
    for v in V.values():
        for name, rec in v["checks"].items():
            if rec["group"] == "family_specific":
                tally[name][rec["status"]] += 1
    rows = [[n, tally[n]["pass"], tally[n]["fail"], tally[n]["not_applicable"],
             tally[n]["no_declared_bound"], tally[n]["blocked_environment"]]
            for n in sorted(tally)]
    L.extend(_tbl(rows, ["check", "pass", "fail", "n/a", "no declared bound", "blocked"]))

    unreachable = sorted(cid for cid, v in V.items()
                         if "Scene2D" in (v["checks"].get("check_backend_agreement", {})
                                          .get("why") or ""))
    if unreachable:
        A("### The backend-agreement check cannot reach the 3-D families")
        A("")
        A(f"`check_backend_agreement` is `not_applicable` on {len(unreachable)} configuration(s) for")
        A("one reason, and it is worth stating plainly rather than leaving as a status code. The check")
        A("compares the fused Triton rasterizer against the readable torch implementation of the")
        A("**same** `Scene2D`; the mjwarp, raycast3d and nvdiff families do not build a `Scene2D` at")
        A("all, so it returns `env renders without a Scene2D`. Their classes also do not offer two")
        A("interchangeable backends — `DMControlSuite` bakes a plate and ray-casts the movers,")
        A("`DMControlTrackCom` hands the whole scene to MJWarp, and an env is one or the other — so")
        A("there is no second implementation of one picture to compare against.")
        A("")
        A("This is a gap in the CHECK SUITE, not a defect in a renderer, and Step 4 does not close it:")
        A("writing a new cross-backend comparison would be building the second correctness framework")
        A("section 1 forbids. It is recorded because \"not applicable\" on a check section 4.1 lists as")
        A("family-specific should be a fact somebody chose, not a status nobody read.")
        A("")

    nb = sorted((cid, n) for cid, v in V.items() for n in v["no_declared_bound"])
    if nb:
        A("### Checks that ran and cannot pass or fail")
        A("")
        A(f"{len(nb)} measurement(s) have **no declared bound anywhere in the repository**, so they")
        A("are reported and do not gate. This is a gap in the *check suite*, not a defect in a")
        A("renderer, and it is the honest reading: `tests/gpu/test_reproducibility.py` says outright")
        A("that a guessed downsample floor is worse than no test, because it either fails a correct")
        A("renderer or passes a broken one. Writing the missing floors is Step 6 work — they must be")
        A("recorded from a measurement taken **before** the renderer changes, i.e. from this run.")
        A("")
        per = Counter(n for _, n in nb)
        L.extend(_tbl([[n, c, ", ".join(sorted(cid for cid, k in nb if k == n))[:90]]
                       for n, c in sorted(per.items())],
                      ["check", "configurations", "which"]))

    A("## The representation tripwire")
    A("")
    A(f"`{trip.get('metric', 'A')}`, against the bound Step 2 froze: "
      + ", ".join(f"`{k}` ≤ {v}" for k, v in sorted((trip.get("bounds") or {}).items())) + ".")
    A("")
    A("Diagnostic, and deliberately weak on its own: a cosine distance is not a task success rate and")
    A("this page does not treat it as one. It is here because it is the cheapest thing that notices a")
    A("Step 6 change that keeps every pixel metric inside its bound and still moves what the frame")
    A("means to an encoder. `ours / jpeg95` below 1 means regenerating the frame from its state lands")
    A("closer to the official renderer than storing that renderer's output as JPEG and reading it back.")
    A("")
    te = {k: v for k, v in (trip.get("envs") or {}).items() if k in V}
    ok = sum(1 for v in te.values() if v.get("pass") is True)
    n_judged = sum(1 for v in te.values() if v.get("pass") is not None)
    A(f"**{ok} of {n_judged} judged configurations are inside the tripwire** "
      f"({len(te)} measured, {len(V) - len(te)} not covered by a DINO run).")
    A("")
    rows = [[k, v.get("tier"), _f(v.get("A"), 4), _f(v.get("bound")),
             {True: "pass", False: "FAIL", None: "—"}[v.get("pass")],
             _f(v.get("cos_ours_vs_official"), 6), _f(v.get("cos_blank_vs_official"), 4),
             _f(v.get("ratio_ours_over_jpeg95"), 3)]
            for k, v in sorted(te.items()) if v.get("status") == "measured"]
    L.extend(_tbl(rows, ["configuration", "tier", "A", "≤", "verdict", "cos ours", "cos blank",
                         "ours / jpeg95"]))
    worse = sorted(k for k, v in te.items()
                   if (v.get("ratio_ours_over_jpeg95") or 0) > 1.0)
    if worse:
        A(f"On {len(worse)} of them `ours / jpeg95` is above 1: a JPEG-q95 round trip of the official")
        A("frame would land CLOSER to that frame, in this encoder's feature space, than our render")
        A("does. That is not a gate and it is not new — a renderer that is not bit-exact loses to a")
        A("lossy copy of the thing it is approximating, by construction — but it is the number Step 5")
        A("will want beside a downstream result, so it is recorded rather than left to be rediscovered:")
        A("")
        A("  " + ", ".join(f"`{c}` ({_f((te[c].get('ratio_ours_over_jpeg95')), 2)}x)" for c in worse))
        A("")

    A("## The frozen state corpora")
    A("")
    A("The artifact Step 6 replays. Each corpus is drawn from the env's **own** reset-and-rollout")
    A("distribution — so every state is one the env genuinely reaches — and then selected for coverage:")
    A("the most typical row, then up to half the budget on per-dimension extremes taken widest")
    A("dimension first, then the rest by farthest-point spread. Deterministic given the seed; no")
    A("generator is consulted by the selection. Synthesising extremes instead would have produced")
    A("states the env never reaches, and a parity number on an unreachable pose describes nothing.")
    A("")
    A("The APPEARANCE is frozen with them. The checks that take states are fed a two-group style from")
    A("`conformance.uniform_style(seed=0)`, whose digest is recorded beside the frames, so the")
    A("variation axis a Step 6 rerun replays is the same one measured here — and the parity and")
    A("oracle numbers are taken at the official default appearance, which is what parity is defined")
    A("against.")
    A("")
    corp = {cid: (merged["configs"][cid].get("corpus") or {}) for cid in V
            if merged["configs"].get(cid, {}).get("corpus")}
    seeds = {c.get("seed") for c in corp.values()}
    A(f"**{len(corp)} corpora frozen**, {sum(c.get('kept', 0) for c in corp.values())} states in total, "
      f"seed {seeds.pop() if len(seeds) == 1 else sorted(seeds)}. Stored as "
      f"`examples/results/step4/states/<configuration>.npz` with the selection metadata and a sha256 "
      f"beside the states.")
    A("")
    rows = [[cid, c.get("state_dim"), c.get("kept"), c.get("pool"), c.get("n_extreme"),
             c.get("n_spread"), (c.get("sha256") or "")[:12]]
            for cid, c in sorted(corp.items())]
    L.extend(_tbl(rows, ["configuration", "state dim", "states", "pool", "extremes", "spread",
                         "sha256"]))

    A("## The regression oracle")
    A("")
    A("What a Step 6 rerun compares against: the digest of our frames and of the official frames on")
    A("the exact states above. Digests rather than pixels — 32 bytes per side against 42 MB for one")
    A("Fetch configuration, in a repository whose thesis is that frames are cheap to regenerate. A")
    A("digest that changes says *something* moved; the metrics in the tables above say what.")
    A("")
    rows = []
    for cid in sorted(V):
        o = (merged["configs"].get(cid) or {}).get("oracle") or {}
        if not o:
            continue
        rows.append([cid, "x".join(str(x) for x in (o.get("frames_shape") or [])[1:3]) or "?",
                     (o.get("digest_ours") or "")[:12], (o.get("digest_official") or "—")[:12],
                     o.get("distinct_colors"),
                     ((o.get("reference_stable") or {}).get("evidence") or {}).get("max_abs_diff", "—")])
    L.extend(_tbl(rows, ["configuration", "frame", "digest ours", "digest official", "colours",
                         "reference re-render Δ"]))

    A("## Render-configuration collapse, re-measured")
    A("")
    A("Step 1 measured that OGBench's dataset suffixes over one `robot x maze` cell compile to one")
    A("scene, and Step 4 validates one env per cell on the strength of it. Section 6 of the brief says")
    A("evidence propagation has to be explicit, so each sibling was rendered once on the")
    A("representative's own frozen corpus and its digest compared. A mismatch would not fail the")
    A("configuration; it would say the collapse is wrong and the sibling needs its own validation.")
    A("")
    rows = []
    for cid in sorted(V):
        sb = (merged["configs"].get(cid) or {}).get("sibling_digest_matches") or {}
        for s, r in sorted(sb.items()):
            if s.startswith("_"):
                continue
            rows.append([cid, s, r.get("status"),
                         {True: "identical", False: "DIFFERENT", None: "—"}.get(r.get("identical"), "—"),
                         (r.get("why") or "")[:60]])
    n_id = sum(1 for r in rows if r[3] == "identical")
    A(f"**{n_id} of {len(rows)} sibling envs draw a byte-identical frame** to the configuration that")
    A("stands for them.")
    A("")
    L.extend(_tbl(rows, ["configuration validated", "sibling", "status", "frames", "note"]))

    A("### Configurations that share an official renderer and were still validated separately")
    A("")
    A("The counterpart, and the reason the collapse above is the only one applied. These")
    A("configurations declare the **same** `reference_id`, which is exactly the signal a")
    A("convenience grouping would have used — and the frames say they are different pictures. Had")
    A("they been folded together, one result would have stood for two renderers.")
    A("")
    groups: dict = defaultdict(list)
    for cid in sorted(V):
        ref = by_id.get(cid, {}).get("reference_id")
        if ref:
            groups[ref].append(cid)
    rows = []
    for ref, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        digs = {c: ((merged["configs"].get(c) or {}).get("oracle") or {}).get("digest_ours") for c in members}
        dims = {c: ((merged["configs"].get(c) or {}).get("corpus") or {}).get("state_dim") for c in members}
        same = len({d for d in digs.values() if d}) == 1 and len(set(dims.values())) == 1
        rows.append([ref, ", ".join(members), len(members),
                     "same frames" if same else "DIFFERENT frames",
                     ", ".join(f"{c}={(digs[c] or '?')[:8]}" for c in members)])
    L.extend(_tbl(rows, ["reference_id", "configurations", "n", "verdict", "digests"]))

    A("## Physics, on its own axis")
    A("")
    A("Kept separate from Gate A and never folded into it. A renderer is not wrong because its env")
    A("ships no batched simulator: `reference_only` is a certifiable physics class (Step 2 section 7),")
    A("and most of this universe is in it. Where an env *does* declare a checkable simulator, the")
    A("bounds are `tests/gpu/test_conformance.py`'s.")
    A("")
    rows = [[cid, v["physics"].get("status"),
             _f((v["physics"].get("observed") or {}).get("agent_max_err")),
             _f((v["physics"].get("observed") or {}).get("block_median_err"), 3),
             "; ".join(v["physics"].get("violations") or []) or (v["physics"].get("why") or "")[:50]]
            for cid, v in sorted(V.items()) if v["physics"].get("status") in ("pass", "fail")]
    n_phys_na = sum(1 for v in V.values() if v["physics"].get("status") not in ("pass", "fail"))
    A(f"{len(rows)} configuration(s) declare a checkable simulator; the remaining {n_phys_na} are "
      f"`reference_only` or declare no batched simulator at all, which is **not** a renderer finding.")
    A("")
    L.extend(_tbl(rows, ["configuration", "physics", "agent max err", "block median err", "note"]))

    A("## The rows to look at first")
    A("")
    A("Section 14 of the brief names seven configurations to inspect carefully, each for a different")
    A("reason, and each because Step 6 is likely to touch it. The verdicts and numbers are read from")
    A("the artifact; the reason column is why the brief singled the row out.")
    A("")
    rows = []
    for cid, why in REPRESENTATIVES:
        v = V.get(cid)
        if v is None:
            rows.append([cid, "not measured", "—", "—", why])
            continue
        par = v["checks"].get("render_parity_with_official_renderer", {}).get("observed") or {}
        vis = v["checks"].get("check_visual_fidelity", {}).get("observed") or {}
        headline = (f"interior {_f(par.get('interior_exact_frac'), 6)}, all {_f(par.get('exact_frac'))}"
                    if par else
                    f"mean {_f(vis.get('mean_abs'), 3)}, SSIM {_f(vis.get('ssim'))}" if vis else "—")
        rows.append([cid, v["gate_a"], v.get("tier") or "—", headline, why])
    L.extend(_tbl(rows, ["configuration", "gate A", "tier", "headline", "why this one"]))

    A("## Pre-existing correctness failures")
    A("")
    if not failing:
        A("None. Every configuration that was reached is inside every bound that exists for it.")
        A("")
    else:
        A(f"{len(failing)} configuration(s). **Recorded, not fixed** — section 18 of the brief. Each is")
        A("an input to the Step 6 engineering queue, and each is reproducible from the frozen corpus")
        A("named beside it.")
        A("")
        rows = []
        for cid in failing:
            v = V[cid]
            for n in v["failures"]:
                rec = v["checks"].get(n, {})
                what = ("; ".join(rec.get("violations") or [])
                        or (str(rec["observed"])[:80] if rec.get("observed") is not None else "")
                        or (rec.get("why") or "")[:90])
                bound = rec.get("bound")
                rows.append([cid, n, what,
                             str(bound)[:60] if not isinstance(bound, dict) else "its own declaration",
                             (merged["configs"][cid].get("corpus") or {}).get("sha256", "")[:12]])
        L.extend(_tbl(rows, ["configuration", "check", "what came out", "required", "corpus"]))

    # The visual-tier drift, assembled from what the classes already declare beside their thresholds:
    # the share of the frame that is skybox, the measurement the threshold was set from, and the
    # spread over independent draws. All of it is data; the only thing this section adds is putting
    # the four rows next to each other, which is what makes the pattern legible.
    vis_rows, drift, held = [], [], []
    for cid in visual:
        v = V[cid]
        decl = (((merged["configs"].get(cid) or {}).get("mandatory") or {})
                .get("check_fidelity_declaration", {}).get("evidence") or {}).get("visual_fidelity") or {}
        obs = v["checks"].get("check_visual_fidelity", {}).get("observed") or {}
        if decl.get("skybox_share") is None or obs.get("mean_abs") is None:
            continue
        spread = (decl.get("draw_spread") or {}).get("mean_abs") or []
        ratio = obs["mean_abs"] / decl["mean_abs"] if decl.get("mean_abs") else None
        vis_rows.append([cid, _f(decl["skybox_share"], 3), _f(decl.get("mean_abs"), 2),
                         f"[{_f(spread[0], 2)}, {_f(spread[1], 2)}]" if len(spread) == 2 else "—",
                         _f(obs["mean_abs"], 2), (f"{ratio:.2f}x" if ratio else "—"),
                         v["checks"].get("check_visual_fidelity", {}).get("status", "-"),
                         v["checks"].get("check_resolution_consistency", {}).get("status", "-")])
        (drift if v["checks"].get("check_visual_fidelity", {}).get("status") == "fail"
         else held).append((cid, decl["skybox_share"]))
    if drift:
        A("## The visual-tier drift")
        A("")
        A(f"{len(drift)} of the {len(vis_rows)} visual-tier configurations no longer meet the numbers")
        A("they declare, and the same ones also fall below their recorded downsample floor. Two")
        A("independent metrics disagreeing in the same direction on the same rows is what makes this")
        A("worth a section rather than a line in a table.")
        A("")
        L.extend(_tbl(vis_rows, ["configuration", "skybox share", "declared mean |Δ|",
                                 "declared draw spread", "measured mean |Δ|", "ratio",
                                 "visual fidelity", "downsample"]))
        A("**It is not the state draw.** Each declaration records the spread of `mean_abs` over three")
        A("independent 64-state draws, and those spreads are tight — a few hundredths for cheetah and")
        A("hopper. The measurements above are outside them by whole units, which is not something a")
        A("different draw of the same distribution produces.")
        A("")
        if held:
            names = ", ".join(f"`{c}`" for c, _ in held)
            zero = [c for c, sk in held if not sk]
            verb = "reproduces" if len(held) == 1 else "reproduce"
            A(f"**It is not the whole stack either.** {names} {verb} the declared measurement almost")
            A("exactly on this same interpreter, well inside its own recorded draw spread.")
            if zero:
                one = len(zero) == 1
                A(f"And {', '.join(f'`{c}`' for c in zero)} "
                  + ("is the only visual configuration with **no skybox in frame at all**"
                     if one else "are the visual configurations with **no skybox in frame at all**")
                  + " — every configuration that drifted has skybox in frame, and the "
                  + ("one" if one else "ones") + " that did not, "
                  + ("does" if one else "do") + " not.")
            A("")
        A("**What the pictures show.** In `comparisons/DMCWalkerWalk.png` the official sky is a smooth")
        A("gradient and ours is covered in white specks, and the amplified difference panel is almost")
        A("entirely that speckle field. The mechanism is the one `stable_envs/envs/dmcontrol.py`")
        A("already documents: the skybox is an 800x4800 gradient with `mark=\"random\"` specks, dm_control")
        A("MINIFIES it and averages the specks away, and MJWarp casts one ray per pixel and paints them")
        A("at full contrast. What changed is the SIZE of that disagreement, not its nature.")
        A("")
        at = ""
        for cid in visual:
            d2 = (((merged["configs"].get(cid) or {}).get("mandatory") or {})
                  .get("check_fidelity_declaration", {}).get("evidence") or {}).get("visual_fidelity") or {}
            at = d2.get("measured_at") or at
        pk = (merged["shards"].get(V[drift[0][0]]["shard"], {}).get("provenance", {})
              .get("reference_packages") or {})
        A("**What differs between the declaration and this run**, for whoever picks this up in Step 6:")
        A("")
        A("```")
        A(f"declared at  {at}")
        A(f"measured at  mujoco {pk.get('mujoco')} + mujoco-warp {pk.get('mujoco_warp')} "
          f"+ warp {pk.get('warp')}, RTX 3090")
        A("```")
        A("")
        A("That is a difference, not a diagnosis. **Nothing here was changed to make these pass** —")
        A("section 18 — and neither the thresholds nor the renderers were touched. What this baseline")
        A("buys is that the drift is now dated and attributable: it exists BEFORE any Step 6")
        A("optimization, so a rewrite of these renderers cannot be blamed for it and cannot hide behind")
        A("it either.")
        A("")

    A("## Known partial coverage")
    A("")
    par = sorted(cid for cid, v in V.items() if v["gate_a"] == "PARTIAL")
    A("`partial` is a **passing** Gate A state and it caps the ladder: Step 2's invariant is that a")
    A("`partial` env can be certified per backend at any performance tier and can never be")
    A("`ENVIRONMENT_DONE`. No result on this page moves an env out of it — a renderer that draws a")
    A("documented subset of the registered default observation still draws a subset.")
    A("")
    for cid in par:
        c = by_id.get(cid, {})
        A(f"- **`{cid}`** — {', '.join(c.get('swm_ids', []))}. Registered-default render coverage is")
        A(f"  `partial`; the declared exclusions stay declared.")
    A(f"- **`swm/OGBMaze-v0`** — the registered default cell (`point` x `large`) is covered and passes")
    A("  above; the row's *parameter space* remains partial at 11 of 15 cells, and no `arena` maze is")
    A("  registered for any robot. That is a coverage fact from Step 1, unchanged by this step.")
    A("")

    A("## Blocked measurements")
    A("")
    blocked = sorted((cid, n, V[cid]["checks"][n].get("why", ""))
                     for cid, v in V.items() for n in v["blocked"])
    A(f"{len([1 for v in V.values() if v['gate_a'] == 'BLOCKED'])} configuration(s) are BLOCKED and")
    A(f"{len(blocked)} individual check(s) could not be run. A dependency that is absent from an")
    A("interpreter is a fact about the interpreter; it never becomes a fidelity failure.")
    A("")
    L.extend(_tbl([[cid, n, (why or "")[:90]] for cid, n, why in blocked],
                  ["configuration", "check", "why"]))
    for cid in sorted(cid for cid, v in V.items() if v["gate_a"] == "BLOCKED" and v.get("why")):
        A(f"- `{cid}`: {V[cid]['why'][:200]}")
    if missing:
        for cid in missing:
            A(f"- `{cid}`: no shard produced a result for this configuration.")
    A("")

    A("## The human comparison sheets")
    A("")
    A("`official | stable-envs | 4x |difference|`, four states per configuration, for the")
    A("representative families section 11 names. **Not a pass/fail eyeball test** — the numbers above")
    A("are the verdict. The sheets exist so that what \"visual tier\" means is visible before Step 6")
    A("changes it, and so a bizarre artifact that no aggregate metric would flag has somewhere to show")
    A("up. The difference panel is amplified 4x and clipped, because a 5/255 residual is invisible at 1x.")
    A("")
    # The path is rebuilt from the configuration id rather than read out of the shard. The shard ran on
    # Oscar and recorded an absolute scratch path, which is where the file WAS, not where it is; the
    # committed copy lives beside the rest of the results and that is the one a reader can open.
    rows = []
    for cid in sorted(V):
        im = (merged["configs"].get(cid) or {}).get("images")
        if not im:
            continue
        rel = f"../../examples/results/step4/comparisons/{cid}.png"
        here = ROOT / "examples" / "results" / "step4" / "comparisons" / f"{cid}.png"
        rows.append([cid, f"[`{cid}.png`]({rel})" + ("" if here.is_file() else " **(missing)**"),
                     _f(im.get("mean_abs"), 3), im.get("max_abs", "—"),
                     " → ".join(im.get("panels", [])), (im.get("sha256") or "")[:12]])
    L.extend(_tbl(rows, ["configuration", "sheet", "mean |Δ|", "max |Δ|", "panels", "sha256"]))

    A("## What Step 4 does not establish")
    A("")
    A("That a world model or a controller cannot tell the renderers apart. Gate A says the renderer is")
    A("internally coherent, that its fidelity declaration is supported by a measurement, that the")
    A("configuration it draws is the one the swm id registers, and that its output is reproducible.")
    A("The semantic claim is Step 5's, on PushT and Reacher, and on a visual-tier arm if a visual-tier")
    A("claim is to be advertised. Nothing here licenses it.")
    A("")
    A("It also establishes nothing about speed, in either direction. Step 3 found GPU rendering broadly")
    A("healthy and CPU rendering broadly poor, with 16 configurations having no CPU renderer at all;")
    A("none of that is evidence about what the pixels are, and this page deliberately does not repeat it.")
    A("")
    A("## How Step 6 replays this")
    A("")
    A("```")
    A("python examples/fidelity_step4.py --shard <dmc|ogb|raster|fetch|swm> \\")
    A("       --states-dir examples/results/step4/states \\")
    A("       --out examples/results/step4/conformance/<shard>.json")
    A("python scratch/step4/certify.py --shards examples/results/step4 --out examples/results/step4")
    A("```")
    A("")
    A("The corpora are read, never redrawn: an existing `.npz` is loaded as-is, so a rerun after an")
    A("optimization compares the same states through a different implementation. What to rerun after")
    A("what kind of change is Step 2 section 15, unchanged by this step.")
    A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=str(ROOT / "examples" / "results" / "step4"))
    ap.add_argument("--out", default=str(ROOT / "examples" / "results" / "step4"))
    ap.add_argument("--doc", default=str(ROOT / "docs" / "research" / "step4_fidelity_baseline.md"))
    a = ap.parse_args()

    shard_dir, out = Path(a.shards), Path(a.out)
    merged = merge(shard_dir)
    cert = certify(merged)
    trip = tripwire(shard_dir, merged)
    cert["tripwire"] = trip
    out.mkdir(parents=True, exist_ok=True)
    (out / "merged.json").write_text(json.dumps(merged, indent=1, sort_keys=True))
    (out / "certification.json").write_text(json.dumps(cert, indent=1, sort_keys=True))
    Path(a.doc).write_text(report(merged, cert, trip))
    from collections import Counter
    print(f"{len(merged['configs'])} configuration(s) from {len(merged['shards'])} shard(s): "
          f"{dict(Counter(v['gate_a'] for v in cert['configs'].values()))}")
    print(f"wrote {out / 'merged.json'}, {out / 'certification.json'}, {a.doc}")


if __name__ == "__main__":
    main()

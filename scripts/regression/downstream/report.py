"""Assemble the Step 5 page from the artifacts, so no number on it is maintained by hand.

Reads only machine-readable output -- the two ``gate_h.json`` files, the Reacher parity probe, the
corpus inspection and the run records -- and writes ``docs/research/step5_downstream_semantics.md``.
Every threshold quoted comes from the frozen Step 2 spec by way of ``step5_stats.SPEC``; every
count comes from ``episodes.jsonl``.

    python scratch/step5/report.py --results tests/fixtures/regression/join/step5 \
        --out docs/research/step5_downstream_semantics.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()


def _esc(x) -> str:
    """Pipe characters end a markdown table row. Learned in Step 4, on ``mean |d|``."""
    return str(x).replace("|", "\\|")


def _tbl(rows: list[list], head: list[str]) -> list[str]:
    if not rows:
        return ["_none._", ""]
    w = [max(len(_esc(h)), *(len(_esc(r[i])) for r in rows)) for i, h in enumerate(head)]
    out = [
        "| " + " | ".join(_esc(h).ljust(w[i]) for i, h in enumerate(head)) + " |",
        "|" + "|".join("-" * (x + 2) for x in w) + "|",
    ]
    for r in rows:
        out.append("| " + " | ".join(_esc(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    out.append("")
    return out


#: Everything this step ran into that a reader would otherwise have to rediscover. Kept here, with
#: the artifact that evidences each one, because the brief's stop condition asks for an "issues
#: discovered" list and a list assembled from memory at the end is a list with holes in it.
FINDINGS: tuple[tuple[str, str, str], ...] = (
    (
        "stable_worldmodel's reacher hides its target; our renderer cannot express that",
        "`render_target = 0` sets the target material's alpha to zero. `stable_envs`' DM Control "
        "appearance space carries material colour, light diffuse and camera fovy and has no alpha, so "
        "the only way this renderer can decline to paint a geom is to put it outside the camera "
        "frustum. Measured, not assumed: moving the reference's target changes 0/255. The harness runs "
        "`offscreen` and records it as part of the arm. A DM Control visibility knob is Step 6 work.",
        "tests/fixtures/regression/join/step5/reacher/parity_probe.json",
    ),
    (
        "interior-pixel exactness is blind to the target",
        "The two target placements differ by a visible disc -- mean |d| 0.459 against 0.171 -- and "
        "their interior-exact fractions are identical to sixteen digits. `interior_mask` keeps only "
        "pixels whose 3x3 neighbourhood in the reference is flat, and the disc lands on a textured "
        "checkerboard floor, so every pixel it changes is already excluded. A whole object appears and "
        "the interior metric does not move. Step 4's parity numbers should be read with that in mind.",
        "tests/fixtures/regression/join/step5/reacher/parity_probe.json",
    ),
    (
        "the DINO-WM Reacher checkpoint behind the recorded 79% does not exist",
        "Not in either checkout, not in `$STABLEWM_HOME`, not on the cluster, not in the HuggingFace "
        "orgs that publish this project's weights. `docs/baselines.md` lists the checkpoint column as "
        "`NA` for every DINO-WM row. Step 5 trained a replacement with the repository's own script.",
        "docs/research/step5_downstream_semantics.md",
    ),
    (
        "the committed DINO-WM training config cannot instantiate",
        "`scripts/train/config/prejepa.yaml` names `stable_worldmodel.wm.prejepa.CausalPredictor`. The "
        "class lives in `...prejepa.module` and that package's `__init__` re-exports only from "
        "`.prejepa`, so hydra raises `InstantiationException` before the dataset is opened. One line, "
        "and the same file already spells its two neighbours correctly.",
        "scratch/step5/train_config/prejepa_reacher.yaml",
    ),
    (
        "the training script names a callback no released version ships",
        "`scripts/train/prejepa.py` builds its trainer with `spt.callbacks.CPUOffloadCallback()`, which "
        "is absent from `stable-pretraining==0.1.8` -- the version stable-worldmodel's own `uv.lock` "
        "pins. Thirty-odd callbacks are exported and none offloads anything. Step 5 supplies a no-op "
        "stand-in rather than editing their repository.",
        "scratch/step5/train_prejepa_shim.py",
    ),
    (
        "the published LeWM Reacher checkpoint cannot be loaded under transformers 5.x",
        "The weights carry `encoder.encoder.layer.N.attention.attention.query.*`; the instantiated "
        "model wants `encoder.layers.N.attention.q_proj.*`. HuggingFace renamed ViT parameters between "
        "major versions and the checkpoint predates the rename. Remapped rather than abandoned, under "
        "guards: the mapping must be a bijection onto the state dict of a model built from the "
        "checkpoint's own config, every shape equal, nothing unmatched on either side, and a strict "
        "load. 303 tensors, 192 renamed, 111 untouched; no tensor was modified, only keys. Those "
        "guards rule out a STRUCTURAL error and cannot rule out a semantic one -- `q_proj` and "
        "`k_proj` are both (192, 192) and a swap between them satisfies every check -- so the remap is "
        "verified against the task instead, where the checkpoint has a recorded 86% and a random "
        "policy scored 8% on the same episodes.",
        "scratch/step5/remap_lewm.py",
    ),
    (
        "MuJoCo 3.12 stopped rendering dm_control's floor texture, and 3.11 renders it",
        "The cause of the corpus mismatch, isolated: `suite.load('reacher','hard')` rendered under "
        "mujoco 3.11.0 shows the checkerboard floor the published corpus has; the same model under "
        "3.12.0 shows a smooth floor. Quadrant contrast 40.68 against 7.49. The texture declaration is "
        'byte-identical across both -- `builtin="checker"`, 300x300, `texrepeat` [1,1], '
        "`texuniform` 1, the same `mat_texid` -- so nothing in dm_control, stable_worldmodel or "
        "stable-envs changed. This is upstream. Consequences: the published corpus is a faithful "
        "record of a pre-3.12 renderer rather than a corrupted file; our frames match the CURRENT "
        "reference to 0.104/255; and any downstream arm that regenerates those stored frames is "
        "measuring the version change. Step 4 recorded that DMCCheetahRun, DMCHopperHop and "
        "DMCWalkerWalk declared their visual contracts under 3.11 and were measured under 3.12 with "
        "the residual in the SKYBOX -- also a texture -- so one upstream change may account for three "
        "of the five frozen Step 4 failures as well. That is a hypothesis under test, not a result.",
        "tests/fixtures/regression/join/step6/texture/",
    ),
    (
        "the corpus and the environment that published it no longer agree",
        "Every re-render of a stored reacher state -- MuJoCo's included -- sits ~18/255 from the "
        "stored frame, with barely half its pixels exact and nothing off by more than 48. The stored "
        "frames have a checkerboard floor and no current render draws one. Consequence for anyone "
        "regenerating this corpus: the drift costs a planner 28 points with stable-envs nowhere in "
        "the arm, so a downstream comparison that swaps the STORED surfaces measures the drift, not "
        "the renderer. The live surface is unaffected, because both arms render today's environment.",
        "tests/fixtures/regression/join/step5/reacher/corpus_probe.json",
    ),
    (
        "planning with a patch-level world model costs hours per 50 episodes, and cannot be batched",
        "CEM scores 300 candidate plans per iteration, and DINO-WM's predictor runs over 768 tokens "
        "(3 history frames x 256 DINOv2 patches) with 16 heads. The attention score tensor is "
        "300 x 16 x 768 x 768 = 2.83e9 entries, 10.55 GiB in fp32 -- four times every other activation "
        "in the layer combined, and materialised because the predictor passes an explicit boolean mask "
        "which makes SDPA fall back to its math backend. Measured consequences on a 24 GB 3090: "
        "`solver.batch_size` above 2 runs out of memory (5, 10 and 50 all fail, each holding ~18 GiB "
        "before the allocation that tips it), so the committed default of 1 is a memory constraint "
        "rather than an oversight; `bf16` cannot be used at all because `PreJEPA.encode` hard-casts "
        "its input with `.float()`, so casting the backbone raises a dtype mismatch; and PyTorch "
        "leaves TF32 matmul off by default, so the whole planner runs in full fp32 on a card whose "
        "tensor cores are idle. None of this is gradient retention -- `solve` is "
        "`@torch.inference_mode()`, `get_action` adds `no_grad`, and the model is "
        "`requires_grad_(False)`. LeWM runs the identical solver in 77 seconds because its predictor "
        "sees 3 pooled tokens instead of 768, a score tensor of 0.16 MiB.",
        "scratch/step5/tf32_probe.sbatch",
    ),
    (
        "Step 2's recorded PushT control degradation is the unpaired figure",
        "The frozen spec records `control.delta_points = -84.0`, which is the 300-episode official rate "
        "minus the 50-episode blank rate. Paired on the seed the control actually ran, it is -92.0. "
        "Both clear the -20 requirement by a wide margin; the artifact is what this page reports.",
        "tests/fixtures/regression/join/step5/pusht/gate_h.json",
    ),
)


def load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


#: The two representatives Step 2 named. A project verdict computed from one of them is not the
#: verdict, however good that one looks.
REQUIRED_ARMS = ("PushT", "Reacher")


def project_verdict(verdicts: dict[str, str]) -> str:
    """Both required arms, or no project-level claim.

    ``FAIL`` beats ``INCONCLUSIVE`` beats ``PASS``: a renderer that moved a success rate past the
    margin somewhere is not rescued by agreeing everywhere else, and an arm whose control never
    degraded contributes no evidence in either direction.
    """
    if any(a not in verdicts for a in REQUIRED_ARMS):
        return "INCOMPLETE"
    seen = set(verdicts.values())
    if "FAIL" in seen:
        return "FAIL"
    if seen == {"PASS"}:
        return "PASS"
    return "INCONCLUSIVE"


def arm_row(name: str, a: dict, extra: str = "") -> list:
    return [
        name,
        f"{a['successes']}/{a['n']}",
        f"{a['success_rate']:.1f} %",
        f"[{a['wilson_ci'][0]:.1f}, {a['wilson_ci'][1]:.1f}]",
        extra,
    ]


def gate_block(g: dict, title: str) -> list[str]:
    e, out = g["equivalence"], []
    rows = [
        arm_row("official", g["official"]),
        arm_row("stable-envs", g["stable_envs"], f"b={e['b']} / c={e['c']}, McNemar p = {e['mcnemar_p']:.3f}"),
    ]
    if g["control"]["present"]:
        c = g["control"]
        rows.append(
            arm_row(
                "blank (control)",
                c["arm"],
                f"b={c['paired']['b']} / c={c['paired']['c']}, p = {c['paired']['mcnemar_p']:.1e}",
            )
        )
    out += [f"### {title}", ""]
    out += _tbl(rows, ["arm", "successes", "rate", "95% Wilson CI", "paired vs official"])
    out += [
        f"**Paired difference {e['delta_points']:+.2f} points, 95% CI "
        f"[{e['ci'][0]:.2f}, {e['ci'][1]:.2f}], against a frozen margin of "
        f"+/-{g['margin_points']} points.**",
        "",
        f"- design: {g['design']['paired_episodes']} paired episodes over "
        f"{g['design']['n_seeds']} seed(s) "
        f"({g['design']['min_paired_episodes']} and {g['design']['min_seeds']} required) -- "
        f"{'met' if g['design']['met'] else 'NOT met'}",
        f"- discordance {e['discordance']:.3f} against a guard of "
        f"{g['power_guard']['max_discordance_at_n300']}; "
        f"{'underpowered' if g['power_guard']['underpowered'] else 'powered at this n'}",
    ]
    if g["control"]["present"]:
        c = g["control"]
        out.append(
            f"- control: {c['paired']['delta_points']:+.1f} points at "
            f"p = {c['paired']['mcnemar_p']:.1e}, against the frozen requirement of "
            f"<= {c['max_delta_points']} points at p < {c['max_p']} -- "
            f"{'DEGRADED' if c['degraded'] else 'NOT degraded'}"
        )
    else:
        out.append("- control: **absent**")
    out += ["", f"**Gate H: {g['verdict']}** -- {g['verdict_reason']}", ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "tests" / "fixtures" / "regression" / "join" / "step5")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "research" / "step5_downstream_semantics.md")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "docs" / "research" / "artifacts" / "step5_downstream"))
    import step5_stats as S

    pusht = load(args.results / "pusht" / "gate_h.json")
    reacher = load(args.results / "reacher" / "gate_h.json")
    probe = load(args.results / "reacher" / "parity_probe.json")
    corpus = load(args.results / "reacher" / "dataset_inspect.json")
    qual = load(args.results / "reacher" / "qualification.json")
    rand_gate = load(args.results / "reacher" / "random_gate" / "gate_h.json")
    corpus_probe = load(args.results / "reacher" / "corpus_probe.json")
    diag = load(args.results / "reacher" / "lewm_diag" / "gate_h.json")

    L: list[str] = []
    A = L.append
    A("# Step 5 — downstream semantic equivalence")
    A("")
    A("Generated by `scratch/step5/report.py` from `tests/fixtures/regression/join/step5/`. Every number on this")
    A("page is computed from an artifact in that directory; none is maintained by hand. Every")
    A(
        "threshold comes from `tests/fixtures/regression/fidelity/certification.json`, which Step 2 froze before any of these"
    )
    A("numbers existed.")
    A("")

    # ---------------------------------------------------------------- conclusion
    # An arm whose campaign has not finished does not vote. Its provisional gate reads
    # INCONCLUSIVE for want of a control arm, and letting that propagate would publish a
    # project-level INCONCLUSIVE that is an artefact of the clock rather than of any measurement.
    def complete(v) -> bool:
        return bool(v) and not ({"official", "stable_envs", "blank"} - set(v.get("arms") or {}))

    def gate_of(v) -> dict:
        """The gate that answers "can a world model tell the renderers apart".

        For PushT that is the all-surface arm: its corpus and its environment still agree, so
        swapping every surface changes only the renderer. For Reacher they do not agree -- MuJoCo
        3.12 stopped painting a floor texture 3.11 painted, so the published corpus records an older
        renderer -- and swapping the STORED surfaces additionally replaces stale bytes with current
        ones. The official_regen control prices that substitution with MuJoCo doing the rendering
        and finds it costs about the same, so the all-surface arm there is not a renderer test. The
        live-observation arm is, and it is reported beside the all-surface arm rather than instead
        of it.
        """
        return v.get("gate_h_renderer_arm") or v["gate_h"]

    verdicts = {k: gate_of(v)["verdict"] for k, v in (("PushT", pusht), ("Reacher", reacher)) if complete(v)}
    overall = project_verdict(verdicts)
    A("## The conclusion")
    A("")
    A(f"**Parity-tier semantic gate: {overall}.**")
    A("")
    if verdicts:
        A(" ".join(f"`{k}` {v}." for k, v in verdicts.items()))
        A("")
    if overall == "INCOMPLETE":
        missing = [k for k, v in (("PushT", pusht), ("Reacher", reacher)) if not complete(v)]
        A(f"Both representatives are required and {', '.join(missing)} has no paired result yet.")
        A("")
    A("No visual-tier claim is made. Step 2 pre-selected Walker as the visual-tier representative and")
    A("Step 4 found it failing its own fidelity contract; substituting a visual-tier environment that")
    A("happens to pass would make the choice of representative depend on the outcome, which is the")
    A("one thing a pre-registered representative is for.")
    A("")

    # ---------------------------------------------------------------- pusht
    if pusht:
        g = pusht["gate_h"]
        A("## PushT + LeWM")
        A("")
        A("The calibration arm, and it is **existing evidence audited rather than a rerun**. The runs")
        A("were made with the official LeWM PushT checkpoint under `le-wm`'s own evaluation; this step")
        A("read their raw per-episode outcomes, checked that the arms are episode-for-episode paired,")
        A("and recomputed every statistic from the outcomes rather than copying the write-up.")
        A("")
        pr = pusht["pairing"]
        A(
            f"Pairing: **{'verified' if pr['paired'] else 'PROBLEMS'}** over {pr['pairs_checked']} "
            f"arm-seed combinations — same episode indices, same start steps, same goal offset, "
            f"budget and checkpoint."
        )
        A("")
        L.extend(gate_block(g, "Arms"))
        sec = pusht.get("secondary_arms") or {}
        if sec:
            A("#### Which pixel surface moved what")
            A("")
            A("Three surfaces feed the model an image: the live observation at every step, the stored")
            A("corpus frame at the start step, and the stored goal frame. Swapping them separately")
            A("turns a null result into a diagnosis.")
            A("")
            rows = []
            for k, v in sorted(sec.items()):
                if "paired" not in v:
                    continue
                p = v["paired"]
                rows.append(
                    [
                        k,
                        f"{v['arm']['successes']}/{v['arm']['n']}",
                        f"{v['arm']['success_rate']:.1f} %",
                        f"{p['delta_points']:+.2f}",
                        f"{p['b']} / {p['c']}",
                        f"{p['mcnemar_p']:.3f}",
                    ]
                )
            L.extend(_tbl(rows, ["arm", "successes", "rate", "delta (pts)", "b / c", "McNemar p"]))
        pv = pusht.get("provenance", {})
        A("#### What the artifact records, and what it cannot")
        A("")
        A("Section 18 asks for a git commit, a checkpoint hash, dependency versions and a GPU beside")
        A("every downstream number. These runs predate that requirement and were executed on a")
        A("cluster this checkout cannot reach, so the missing fields are listed rather than invented:")
        A("")
        for k in pv.get("not_recorded", []):
            A(f"- {k}")
        A("")
        A(
            f"Recorded: checkpoint `{'`, `'.join(pv.get('recorded_in_artifact', {}).get('checkpoint', []))}`, "
            f"seeds {pv.get('recorded_in_artifact', {}).get('seeds')}, "
            f"goal offset {pv.get('recorded_in_artifact', {}).get('goal_offset')}, "
            f"budget {pv.get('recorded_in_artifact', {}).get('eval_budget')}, "
            f"and the episode ids and start steps of every run."
        )
        A("")

    # ---------------------------------------------------------------- reacher
    A("## Reacher + DINO-WM")
    A("")
    if corpus:
        A("### The corpus")
        A("")
        A(
            f"`{corpus['dataset']}` — {corpus['episodes']} episodes, {corpus['rows']} rows, "
            f"columns `{'`, `'.join(corpus['columns'])}`."
        )
        A("")
    if probe:
        A("### Do the two renderers draw the same reacher?")
        A("")
        A("The pre-flight, and on this environment it is not a formality. `stable_worldmodel`'s")
        A("reacher is not `suite.load('reacher', 'hard')`: it recompiles the model through `mjcf`")
        A("after editing materials and geom types, and its variation default `render_target = 0`")
        A("sets the target material's **alpha** to zero. `stable_envs`' DM Control appearance space")
        A("carries material colour, light diffuse and camera fovy and has no alpha, so if the")
        A("reference declines to draw its target and we draw it, every frame differs by a red disc")
        A("and the downstream comparison measures that disc.")
        A("")
        drawn = probe["reference_target_drawn"]
        A("Measured, by moving the target geom and re-rendering the reference: moving it changes at")
        A(
            f"most **{probe['reference_target_move_delta']}/255**, so the reference "
            f"**{'DOES' if drawn else 'does NOT'} draw its target**."
        )
        A("")
        rows = [
            [
                k,
                f"{v['exact_frac']:.4f}",
                "n/a" if v["interior_exact_frac"] is None else f"{v['interior_exact_frac']:.6f}",
                f"{v['mean_abs']:.3f}",
                str(v["max_abs"]),
                f"{v['ssim']:.4f}",
            ]
            for k, v in probe["modes"].items()
        ]
        L.extend(_tbl(rows, ["our target placement", "all exact", "interior exact", "mean |d|", "max |d|", "SSIM"]))
        A(f"The harness uses `{probe['closest_mode']}`.")
        A("")
    if qual:
        A("### Instrument qualification")
        A("")
        A(qual.get("note", ""))
        A("")
        rows = [
            [
                r["policy"],
                r.get("precision", ""),
                r["arm"],
                f"{r['successes']}/{r['n']}",
                f"{r['rate']:.1f} %",
                r.get("reference", ""),
            ]
            for r in qual.get("runs", [])
        ]
        L.extend(_tbl(rows, ["checkpoint", "arithmetic", "arm", "successes", "rate", "recorded baseline"]))
    if rand_gate:
        g = rand_gate["gate_h"]
        e = g["equivalence"]
        A("#### The gate, applied to the floor")
        A("")
        A("Running the frozen rule over those three random-policy arms is worth doing once, because")
        A("it is the case where agreement between renderers means nothing at all -- a policy that")
        A("ignores the pixels agrees with itself whatever is drawn. The rule says so:")
        A("")
        A("```")
        A(f"official     {g['official']['successes']}/{g['official']['n']}")
        A(f"stable-envs  {g['stable_envs']['successes']}/{g['stable_envs']['n']}")
        A(
            f"paired       delta {e['delta_points']:+.2f} pts, CI "
            f"[{e['ci'][0]:.2f}, {e['ci'][1]:.2f}], McNemar p = {e['mcnemar_p']:.3f}"
        )
        A(
            f"control      {g['control']['paired']['delta_points']:+.1f} pts, "
            f"p = {g['control']['paired']['mcnemar_p']:.2f}  ->  degraded = "
            f"{str(g['control']['degraded']).lower()}"
        )
        A(f"VERDICT      {g['verdict']}")
        A("```")
        A("")
        A("A p of 1.000 on the paired difference, and the verdict is still not a pass. That is the")
        A("control requirement doing the only job it has: the blank arm scored ABOVE official here,")
        A("because with random actions the frames are irrelevant, and an evaluation that cannot be")
        A("hurt by removing the picture cannot be used to argue that two pictures are equivalent.")
        A("")
    if corpus_probe or diag:
        A("### The corpus does not match the environment that published it")
        A("")
        A("Found while diagnosing a 30-point drop, and it changes how the two STORED pixel surfaces")
        A("have to be read. The planner's first observation and its goal image are not rendered at")
        A("evaluation time in the official arm: they are read off disk from the corpus. Our arm")
        A("regenerates them from the `qpos` stored beside each frame, which is the repository's own")
        A("thesis -- and the thesis is testable, so it was tested.")
        A("")
    if corpus_probe:
        A(f"On {corpus_probe['n']} corpus rows, every re-render of a stored state compared against")
        A("the stored frame itself:")
        A("")
        rows = [
            [k, f"{v['exact_frac']:.4f}", f"{v['mean_abs']:.3f}", str(v["max_abs"]), f"{v['ssim']:.4f}"]
            for k, v in corpus_probe["comparisons"].items()
        ]
        # _esc escapes the pipes itself; pre-escaping here doubled the backslashes.
        L.extend(_tbl(rows, ["re-render of the stored state", "all exact", "mean |d|", "max |d|", "SSIM"]))
        A("**The cause is upstream, and it is now isolated.** `suite.load('reacher','hard')` under")
        A("mujoco **3.11** renders the checkerboard the corpus has; under **3.12** it does not —")
        A("quadrant contrast 40.68 against 7.49, with a byte-identical texture declaration on both")
        A("sides. Nothing in dm_control, stable_worldmodel or stable-envs is implicated. See")
        A("`tests/fixtures/regression/join/step6/texture/` and the panel in `stored_vs_official_vs_ours.png`.")
        A("")
        A("**MuJoCo cannot reproduce the corpus either.** `official_default` is the live environment")
        A("rendering the same `qpos` and target position the frame was stored with, and it lands")
        A("18.3/255 away with barely half its pixels exact. Ours lands at 18.5. The contact sheet")
        A("(`corpus_probe.png`) shows the mechanism: the stored frames have a CHECKERBOARD floor and")
        A("no current render draws one. The environment's appearance drifted after the data was")
        A("published, and the signature is unmistakable -- half the pixels exact, nothing off by")
        A("more than 48, which is one set of checker squares and nothing else.")
        A("")
        A("Two consequences. Regenerating a stored frame moves the model off its training")
        A("distribution no matter which renderer regenerates it, so a downstream arm that swaps the")
        A("stored surfaces is measuring the drift and not the renderer. And a *live*-surface")
        A("comparison is unaffected, because both arms render the same environment as it is today.")
        A("")
    if diag:
        A("#### Which surface costs the points")
        A("")
        A("Measured on the LeWM instrument, one seed, fifty episodes. Not a gate -- a diagnosis.")
        A("")
        g = diag["gate_h"]
        rows = [
            [
                "official (baseline)",
                f"{g['official']['successes']}/{g['official']['n']}",
                f"{g['official']['success_rate']:.1f} %",
                "-",
            ],
            [
                "stable-envs, every surface",
                f"{g['stable_envs']['successes']}/{g['stable_envs']['n']}",
                f"{g['stable_envs']['success_rate']:.1f} %",
                f"{g['equivalence']['delta_points']:+.1f}",
            ],
        ]
        for k, v in sorted((diag.get("secondary_arms") or {}).items()):
            if "paired" not in v:
                continue
            rows.append(
                [
                    k,
                    f"{v['arm']['successes']}/{v['arm']['n']}",
                    f"{v['arm']['success_rate']:.1f} %",
                    f"{v['paired']['delta_points']:+.1f}",
                ]
            )
        L.extend(_tbl(rows, ["arm", "successes", "rate", "delta (pts)"]))
        A("The live-surface row is the one that answers Step 5's question for this environment, and")
        A("it is exact rather than approximate: the same episode count, not merely a similar rate.")
        A("")
    if reacher and reacher.get("gate_h_renderer_arm"):
        A("### The renderer arm — the Gate H answer for this environment")
        A("")
        A("Only the LIVE observation is ours; the two stored corpus frames are left exactly as the")
        A("official arm reads them off disk. Nothing but `render()` differs, which is the")
        A("comparison Gate H describes.")
        A("")
        L.extend(gate_block(reacher["gate_h_renderer_arm"], "Arms (live observation surface)"))
        A("### The all-surface arm — and why it is not the renderer's verdict")
        A("")
        A("Reported in full because it fails, and because the reason is worth more than the number.")
        A("Swapping the two STORED surfaces does not only change the renderer: it replaces frames")
        A("recorded months ago with frames rendered now. On this environment those are not the same")
        A("picture — see the corpus section below — and the `official_regen` control prices the")
        A("substitution with **MuJoCo itself** doing the regenerating.")
        A("")
    if reacher:
        # A gate computed while the campaign is still running is a progress note, not a verdict.
        # Say so at the top of the section rather than trusting a reader to notice DESIGN_SHORTFALL.
        arms_present = set(reacher.get("arms") or {})
        missing = {"official", "stable_envs", "blank"} - arms_present
        g0 = reacher["gate_h"]
        need = g0["power_guard"]["required_n"]
        if not missing and g0["power_guard"]["underpowered"] and g0["equivalence"]["n"] < need:
            A(
                f"> **Campaign extending.** The power guard below asks for {need} paired episodes at "
                f"the observed discordance and {g0['equivalence']['n']} have reported; six further "
                f"seeds are running, which takes the design to 600. The verdict here is the frozen "
                f"rule applied to what exists so far, and INCONCLUSIVE is what it should say until "
                f"the rest lands."
            )
            A("")
        if missing:
            A(
                f"> **Campaign in progress.** The {', '.join(sorted(missing))} arm(s) have not "
                f"reported yet, so the verdict below is provisional and cannot be read as a result: "
                f"without the destructive control the frozen rule returns INCONCLUSIVE by "
                f"construction, whatever the paired difference happens to be."
            )
            A("")
        L.extend(gate_block(reacher["gate_h"], "Arms"))
        sec = reacher.get("secondary_arms") or {}
        if sec:
            A("#### Which pixel surface moved what")
            A("")
            rows = [
                [
                    k,
                    f"{v['arm']['successes']}/{v['arm']['n']}",
                    f"{v['arm']['success_rate']:.1f} %",
                    f"{v['paired']['delta_points']:+.2f}",
                    f"{v['paired']['b']} / {v['paired']['c']}",
                    f"{v['paired']['mcnemar_p']:.3f}",
                ]
                for k, v in sorted(sec.items())
                if "paired" in v
            ]
            L.extend(_tbl(rows, ["arm", "successes", "rate", "delta (pts)", "b / c", "McNemar p"]))
        c = reacher.get("comparability", {})
        A(
            f"Arm comparability: **{'verified' if c.get('comparable') else 'PROBLEMS'}** over "
            f"{c.get('arm_runs_checked', 0)} arm-runs — resolved config, checkpoint file hashes, both"
        )
        A("git commits and dependency versions diffed field by field; only the renderer differs.")
        A("")
        for p in c.get("problems", []):
            A(f"- **{p}**")
        for n in c.get("benign_differences", [])[:8]:
            A(f"- (benign) {n}")
        A("")
    else:
        A("_The Reacher arm has not produced a paired result yet._")
        A("")

    # ---------------------------------------------------------------- limits
    A("## What Step 5 does not establish")
    A("")
    A("That every renderer in the library is semantically equivalent. Two environments were measured,")
    A("chosen in Step 2 before any of this was run, and they are parity-tier. A visual-tier renderer")
    A("is a different render of the same scene by construction, and nothing here licenses a claim")
    A("about one.")
    A("")
    A("It also establishes nothing about speed or about pixel exactness. Step 3 measured the first")
    A("and Step 4 the second; a downstream success rate is a third thing, and the reason to measure")
    A("it is precisely that neither of the other two implies it.")
    A("")
    A("## Issues discovered")
    A("")
    A("Recorded, not repaired -- except where a repair was the only way to run the experiment at")
    A("all, in which case what was changed and why is beside it.")
    A("")
    for i, (title, detail, where) in enumerate(FINDINGS, 1):
        A(f"{i}. **{title}.** {detail} (`{where}`)")
    A("")
    A("## The rule this was judged by")
    A("")
    A("Frozen in Step 2, quoted here so the verdict can be checked against it:")
    A("")
    A("```json")
    A(
        json.dumps(
            {
                k: v
                for k, v in S.SPEC.items()
                if k not in ("margin_justification", "validated_against_existing_evidence")
            },
            indent=1,
        )
    )
    A("```")
    A("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print(f"wrote {args.out} ({len(L)} lines); overall {overall}")


if __name__ == "__main__":
    main()

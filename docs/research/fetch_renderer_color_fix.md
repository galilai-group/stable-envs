# Fetch renderer color correction

Starting checkpoint: `repair/environment-semantics`, `df80c9c54590afa9fc0482a4dcb241d21f6b63a3`.
Pinned SWM: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
Scope: all 12 Fetch registrations, with independent per-registration checks. No shared State/RenderState redesign, shader changes, semantic promotions, or push.

## Why the comparison was wrong

The exact FetchPushDict witness is seed 17, case 6, step 49 (frame 40). Its source reset options explicitly select table and object RGB `[0.35, 0.35, 0.35]`.

1. **Benchmark omitted appearance.** `collect.render_kwargs` returned an empty mapping for Fetch. The CPU corpus saved full simulator State, but Fetch appearance lives separately in the existing style API. The GPU collector constructed a fresh environment and rendered those States with its default white table. Numeric simulator-State equality did not establish appearance equality. The old varied-frame Fetch MSEs therefore were not a valid matched-appearance renderer comparison.
2. **Source reset side effects were missing.** Pinned SWM's `FetchWrapper._apply_visual_variations` detaches table, floor, and object materials on its first active visual reset, including when only table or object color is requested. The floor's underlying geometry RGB is 0.5 rather than its material RGB 0.2. Our reset did not reproduce this change.
3. **Applying two colors replaced the first.** Reset called `Env.set_variation` separately for object and table. Each mapping call starts from defaults, so the second call discarded the first color. This also mishandled partial color resets.

The collector now snapshots a detached style row alongside each simulator State. Fetch reset derives the affected geometry groups from the installed upstream model, follows the source material-detachment/color updates, and writes the resulting effective colors together into the existing style. Geometry colors persist through repeated resets as in the source. No rendering pass was added.

## Evidence and results

CPU source-color and continuation checks: **351 two-lane reset cases**, covering each registration independently, untouched appearance, table-only, object-only where supported, combined and sampled colors, and successive partial/no-visual resets. Each reset is followed by three paired simulator steps.

GPU comparison: **432 archived full States**, each rendered twice without mutating supplied or live State. Original source CPU/GPU images and action/state archives are reused unchanged. The two newly measured candidate variants separate appearance-transfer correction alone from the complete reset-color correction. **333 no-visual-variation frames are byte-identical to the original candidate images.**

All **1,728** candidate/source MSE comparisons were independently recomputed locally from downloaded raw RGB arrays; all 165 exported PNG RGB hashes and gallery links were checked.

Maximum per-frame normalized RGB MSE against source GPU (each of the three registrations per scene was measured separately):

| Scene | Old, with appearance mismatch | Transfer corrected only | Both corrections |
|---|---:|---:|---:|
| PickAndPlace | 0.02948557 | 0.01978110 | 0.00040646 |
| Push | 0.02751139 | 0.01166068 | 0.00055631 |
| Reach | 0.02784319 | 0.01165884 | 0.00047218 |
| Slide | 0.05637268 | 0.01187186 | 0.00055056 |

For the exact FetchPushDict frame 40: **0.0275113941 → 0.0003684384**, a 98.66% reduction. The corrected corpus maximum above is a different frame. Source CPU/GPU MSE on frame 40 is 0.0000358980.

This is not a claim of pixel-perfect Fetch. Edge sampling, translucent markers/axes, shading, and material-property residuals remain. PickAndPlace retains the existing reset-only limitation; CPU candidate mesh rendering remains unsupported. No performance benchmark or acceptance promotion was performed.

## Artifacts and execution

- [Per-registration Fetch CSV](semantic_recovery_evidence/fetch_renderer_colors/fetch_color_differences.csv)
- [Updated 42-registration CSV](semantic_recovery_evidence/fetch_renderer_colors/renderer_differences_42_environments.csv): only Fetch candidate comparisons are updated; all other rows and source images retain prior campaign provenance.
- [Source witness](semantic_recovery_evidence/fetch_renderer_colors/FetchPushDict-0040-source_gpu.png), [old candidate](semantic_recovery_evidence/fetch_renderer_colors/FetchPushDict-0040-old.png), [fixed candidate](semantic_recovery_evidence/fetch_renderer_colors/FetchPushDict-0040-fixed.png).
- [Local validation](semantic_recovery_evidence/fetch_renderer_colors/local-validation.json) and [terminal evidence](semantic_recovery_evidence/fetch_renderer_colors/terminal-evidence.txt).
- Complete gallery, raw RGB arrays, CPU assertions, per-frame styles/hashes and two candidate measurement records, preserved outside macOS temporary cleanup: `/Users/samib/stable-envs/.worktrees/render-config-fixes/tmp/renderer-evidence/fetch-colors/`.
- Oscar root: `/oscar/scratch/sboughan/se-semantic-recovery/fetch-color-fix-df80c9c5-20260928T032545Z`.
- Source corpus: `render-rerun-df80c9c5-20260928T021357Z/run-6750648_2/results` under the same Oscar recovery root.
- Job 6751747: CPU audit PASS; first GPU diagnostic failed before rendering because it incorrectly asserted equality of fresh resets across simulator MuJoCo 3.8.1 and renderer MuJoCo 3.11 profiles.
- Job 6751848: corrected diagnostic renders the exact archived CPU States; both GPU variants PASS, job COMPLETED 0:0.
- Job 6751874: independent export/verification PASS, job COMPLETED 0:0.
- Local targeted tests: 36 passed / 56 dependency skips initially; the updated verifier suite passes 17/17 including the new appearance-archive regression. Live Oscar source-color and GPU checks did not skip registrations.

The accepted matrix and earlier evidence files remain untouched.

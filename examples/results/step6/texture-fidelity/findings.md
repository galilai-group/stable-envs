# Step 6A — texture / reference-fidelity diagnosis (workstream `texture-fidelity`)

**Classification: FINDING** — measured diagnosis; no renderer patch is warranted; a declaration-provenance
decision is (§8, §10).

- Baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`)
- Branch: `step6/texture-fidelity` (HEAD = the commit carrying this file)
- Authorization: diagnosis only. **Nothing under `stable_envs/` was modified.** Only scripts under
  `scratch/step6/texture-fidelity/` and artifacts under `examples/results/step6/texture-fidelity/`.

## Verdict in one paragraph

Step 4's Gate-A `check_visual_fidelity` failures on `DMCCheetahRun`, `DMCHopperHop` and `DMCWalkerWalk` are
**(B) reference-renderer version drift**, and the drift is **in the floor, not the skybox**. MuJoCo 3.12.0
changed how a texture is mapped onto a finite plane (changelog 3.12.0 item 15, commit `cc7fb98c`: *"Added
explicit texture coordinates to built-in geometries (Plane, Box, …). For finite planes, textures are now
anchored to the bottom-left corner instead of the center. This will cause the most common visual breakage, as
common procedural checker textures will be phase shifted."*). `mujoco_warp` — the renderer stable-envs draws
these scenes with — implements the pre-3.12 mapping in both 3.11.0 and 3.12.0 (`sample_texture`, PLANE branch,
byte-identical across the two releases; its own comment says it replicates `render_gl3.c settexture`), and our
frames are pixel-identical across the two arms (mean |diff| 0.000). The compiled texture bytes, the material
table, the suite XMLs and dm_control's render engine are identical across the two versions. Replaying the
frozen Step-4 corpora under the declaration's own stack (mujoco 3.11.0 + dm_control 1.0.44 + mujoco-warp
3.11.0, RTX 3090) puts all three back inside their declared numbers — cheetah 4.99 / 0.882 / 0.661 against a
declared 5.01 / 0.882 / 0.661 — and replaying under 3.12 on two different 3090 nodes reproduces Step 4's
failure to 0.03/255. Per surface, the skybox residual is the same under both versions (4.0–4.2/255, 91–92%
within 8: exactly what the declaration's note measured), and the floor goes from 5–12/255 under 3.11 to 19–33
under 3.12. The three failing domains are the ones whose floor plane shifts by a non-integer number of checker
periods (long thin planes, half-width 0.8–1.0); the control `DMCHumanoidStand` has a square 100×100 floor,
shifts by whole periods, and passes under both. `check_resolution_consistency` is a separate, smaller finding:
that check never reads the reference, gives the same number under both MuJoCo versions, reproduces its
recorded bound to 6 decimals under the bound's own 8-state protocol on both arms, and fails on walker only
because the bound is applied to a 64-state corpus it was not measured on with a 0.005 margin.

## 1. Exact versions compared

| arm | mujoco | dm_control | mujoco_warp | warp | torch | gymnasium | card / driver / node | SLURM |
|---|---|---|---|---|---|---|---|---|
| declaration (`_AT` in `stable_envs/envs/dmcontrol.py`) | 3.11.0 | (1.0.44 pair) | 3.11.0 | — | — | — | H200, driver 590.48.01 | pre-repo |
| Step 4 frozen (`examples/results/step4/conformance/dmc.json` provenance) | 3.12.0 | present (1.0.45) | 3.12.0 | 1.17.0 | 2.14.0+cu126 | 1.3.0 | RTX 3090, 575.57.08, gpu2105 | 6176998 |
| **arm 3.12**, replay + localize + ablation | 3.12.0 | 1.0.45 | 3.12.0 | 1.17.0 | 2.14.0+cu126 | 1.3.0 | RTX 3090, 575.57.08, gpu2108 | 6343431 |
| **arm 3.11**, replay + localize + probe; **arm 3.12** localize + probe again; cross (one node) | 3.11.0 / 3.12.0 | 1.0.44 / 1.0.45 | 3.11.0 / 3.12.0 | 1.17.0 | 2.14.0+cu126 | 1.3.0 | RTX 3090, 575.57.08, gpu2113 | 6343699 |
| local model/texture check (CPU, no GL) | 3.11.0 / 3.12.0 | 1.0.44 / 1.0.45 | 3.11.0 / 3.12.0 (source diff only) | 1.17.0 | — | — | Apple M5 | — |
| local official-frame drift (qualitative) | 3.11.0 / 3.12.0 | 1.0.44 / 1.0.45 | — | — | — | — | Apple GL, `MUJOCO_GL=glfw` | — |

Venvs on Oscar: `$SCR/venvs/se-vis311` (built by job 6343431 from `visual311_normgpu.sbatch`; `gymnasium`
1.3.0 added on the login node afterwards — the original build line omitted a core `pyproject.toml`
dependency, so the first 3.11 arm imported nothing) and `$SCR/venvs/se-step5` (3.12 arm). Same driver as
Step 4 on every arm; Step 3/4's card class (`--constraint=geforce3090`).

## 2. Declaration / model comparison — what is identical and what changed

Checked on the Mac in both interpreters, `suite.load(domain, task, task_kwargs={"random": 0})` + `reset()`
(`scratch/step6/texture-fidelity/texdigest.py`; `local/model_textures_mujoco311{0,20}.json`), and re-checked
inside both Oscar arms (`localize*/localize.json` → `textures`; `cross.json` → `same_texture_bytes`,
`same_material_table` = true on every config):

| item | 3.11.0 | 3.12.0 |
|---|---|---|
| `skybox` texture (type 2 = skybox, 800×4800×3, `builtin="gradient" mark="random"`) — sha256 of `tex_data` | `f6a21dd7d85c1e5d…` | **identical** |
| `grid` texture (type 0 = 2d, 300×300×3, `builtin="checker" mark="edge"`) — sha256 of `tex_data` | `d6efb0698f2ef40f…` | **identical** |
| material table (`mat_texid`, `texrepeat=[1,1]`, `texuniform=1`, rgba, reflectance, shininess, specular) | — | **identical** |
| dm_control suite XMLs (`cheetah/hopper/walker/humanoid/reacher.xml`, `common/{materials,skybox,visual}.xml`) | 1.0.44 | **byte-identical** in 1.0.45 |
| dm_control `mujoco/engine.py` (the `physics.render` path) | 1.0.44 | **byte-identical** in 1.0.45 |
| segmentation buffer at the frozen states (which pixel is sky / floor / body) | — | **identical** across arms |
| `mujoco_warp/_src/render.py::sample_texture` PLANE branch (`uv = (0.5·x, −0.5·y)`, offset −0.5, `× texrepeat`; comment: "Replicate MuJoCo's OBJECT_PLANE texgen … render_gl3.c settexture") | 3.11.0 | **identical** in 3.12.0 (the only 3.11→3.12 diff in `render.py` is the Gaussian-splat path) |
| our frames (mujoco_warp 3.11 vs 3.12, same node) | — | mean 0.000, 99.98% exact (the residual is mujoco_warp's known run-to-run 1-level noise) |
| MuJoCo classic renderer, finite-plane texture mapping | GL texgen, centred | **explicit texcoords anchored bottom-left** (3.12.0 item 15, `cc7fb98c`) |
| official frames (3.11 vs 3.12, same node, frozen states) | — | cheetah 1.99, hopper 2.41, walker 10.02, humanoid 0.79 /255 — all of it floor (§5) |

So the declaration is the same bytes on both sides and stable-envs draws the same picture on both sides; the
one thing that moved is libmujoco's plane mapping. This also reframes the Step-5 reacher finding
(`examples/results/step6/texture/`): 3.12 does not "stop painting" the checker — the 0.6 m × 0.6 m reacher
plane, re-anchored to its corner, now lies inside a single checker cell, which at 84 px reads as a flat floor
(quadrant spread 40.7 → 7.5 on Oscar; 36.0 → 4.5 locally).

Why the control is immune: the floors are `cheetah` 100×0.8, `hopper` 50×1, `walker` 250×0.8 (all
`texrepeat 1 1`, `texuniform=true`), `humanoid` 100×100. Re-anchoring from centre to corner shifts the checker
phase by the plane's half-extent: on the 100 / 50 / 250 axis that is a whole number of periods, on the
0.8 / 1.0 axis it is not — and that is the camera-facing axis. Humanoid shifts by whole periods on both.

## 3. What Step 4 actually measured, and against what

`examples/fidelity_step4.py` renders the official frames **live** (`cls.reference_render` → dm_control
`physics.render` under whatever mujoco is installed); `check_visual_fidelity` draws its own 64 states from
`env.rng` re-seeded to `CHECK_SEED = 4041`; the frozen corpus feeds the oracle digests and
`check_resolution_consistency`. The bounds it was judged against carry these provenances:

| check | bound source | measured under | applied in Step 4 under |
|---|---|---|---|
| `check_visual_fidelity` | `VISUAL_FIDELITY[...]` + `_AT`, `stable_envs/envs/dmcontrol.py` | mujoco 3.11.0 + mujoco-warp 3.11.0, H200, 64 states × 3 draws (`scratch/dmcontrol/visual_calib.py`) | mujoco 3.12.0 + mujoco_warp 3.12.0, RTX 3090 |
| `check_resolution_consistency` | `DOWNSAMPLE_EXACT_FRAC` + `MEASURED_ON`, `tests/gpu/test_reproducibility.py` | mujoco 3.9, H200, torch 2.11+cu130, **8 states** from `render_worker.STATE_SEED` | mujoco 3.12.0, RTX 3090, **64 frozen corpus states** |

## 4. Visual / metric comparison against the frozen bounds

`gate.py` applies the bounds recorded in `examples/results/step4/certification.json` verbatim to each arm's
replay shard (`gate.json`). `check_visual_fidelity` = mean_abs / within8_frac / ssim on the seed-4041 draw:

| config | declared (3.11, H200) | ceiling | Step 4 (3.12, gpu2105) | arm 3.12 (gpu2108) | **arm 3.11 (gpu2113)** |
|---|---|---|---|---|---|
| DMCCheetahRun | 5.01 / 0.882 / 0.661 | ≤6.5 / ≥0.84 / ≥0.62 | 6.54 / 0.796 / 0.582 FAIL | 6.51 / 0.797 / 0.584 FAIL | **4.99 / 0.882 / 0.661 PASS** |
| DMCHopperHop | 4.35 / 0.905 / 0.684 | ≤5.7 / ≥0.86 / ≥0.64 | 6.58 / 0.791 / 0.544 FAIL | 6.55 / 0.792 / 0.546 FAIL | **4.37 / 0.904 / 0.684 PASS** |
| DMCWalkerWalk | 8.07 / 0.700 / 0.826 | ≤10.5 / ≥0.65 / ≥0.77 | 16.01 / 0.552 / 0.533 FAIL | 15.95 / 0.553 / 0.533 FAIL | **7.86 / 0.714 / 0.824 PASS** |
| DMCHumanoidStand (control) | 6.26 / 0.735 / 0.868 | ≤8.2 / ≥0.69 / ≥0.82 | 6.30 / 0.730 / 0.866 PASS | 6.33 / 0.729 / 0.865 PASS | 6.27 / 0.732 / 0.867 PASS |

The 3.11 arm lands inside the declaration's own recorded draw spread on every metric (cheetah mean
[4.986, 5.032], hopper [4.327, 4.364], walker [7.636, 8.302]). The 3.12 arm on a second 3090 node reproduces
Step 4 to within 0.03/255 on every config: the failure is real under 3.12 and not a node artefact
(`reference_stable` max_abs_diff = 0 on all nodes).

`check_resolution_consistency` (exact_frac; floor = `DOWNSAMPLE_EXACT_FRAC` − 0.005; reference never used):

| config | bound − margin | Step 4 (gpu2105) | arm 3.12 (gpu2108 and gpu2113) | arm 3.11 (gpu2113) |
|---|---|---|---|---|
| DMCCheetahRun | 0.52885 | 0.5208 FAIL | 0.5304 pass | 0.5304 pass |
| DMCHopperHop | 0.54522 | 0.5350 FAIL | 0.5526 pass | 0.5526 pass |
| DMCWalkerWalk | 0.62395 | 0.6031 FAIL | 0.6188 FAIL | 0.6188 FAIL |
| DMCHumanoidStand | 0.59865 | 0.6088 pass | 0.6095 pass | 0.6095 pass |

MuJoCo version has no effect on this check (identical to 4 decimals across arms, nodes and processes). §6.

## 5. Residual localisation (per surface, off MuJoCo's own segmentation buffer at the identical states)

`scratch/step6/texture-fidelity/localize.py`; masks: `skybox` = segmentation objid < 0, `floor` = plane
geoms, `bodies` = the rest; `resid` = that region's share of Σ|diff|. Seed-4041 draw (the frozen corpus is
within 0.4/255 of it in every cell; both are in `localize*/localize.json`).

**ours vs official, per arm (gpu2113, job 6343699; the gpu2108 3.12 numbers are identical):**

| config | arm | whole mean | skybox share / mean / within8 / resid | floor share / mean / within8 / resid | bodies share / mean / resid |
|---|---|---|---|---|---|
| DMCCheetahRun | 3.11 | 4.99 | 0.826 / **4.14** / 0.923 / 0.69 | 0.132 / **7.81** / 0.726 / 0.21 | 0.042 / 12.9 / 0.11 |
| | 3.12 | 6.51 | 0.826 / **4.14** / 0.922 / 0.53 | 0.132 / **19.26** / 0.084 / 0.39 | 0.042 / 12.9 / 0.08 |
| DMCHopperHop | 3.11 | 4.37 | 0.833 / **4.03** / 0.924 / 0.77 | 0.142 / **5.01** / 0.836 / 0.16 | 0.025 / 12.3 / 0.07 |
| | 3.12 | 6.55 | 0.833 / **4.02** / 0.924 / 0.51 | 0.142 / **20.37** / 0.041 / 0.44 | 0.025 / 12.3 / 0.05 |
| DMCWalkerWalk | 3.11 | 7.86 | 0.534 / **4.05** / 0.914 / 0.28 | 0.395 / **12.45** / 0.447 / 0.63 | 0.071 / 11.1 / 0.10 |
| | 3.12 | 15.95 | 0.534 / **4.05** / 0.913 / 0.14 | 0.395 / **32.96** / 0.041 / 0.82 | 0.071 / 11.0 / 0.05 |
| DMCHumanoidStand | 3.11 | 6.27 | 0.000 / — | 0.960 / **5.81** / 0.743 / 0.89 | 0.040 / 17.6 / 0.11 |
| | 3.12 | 6.33 | 0.000 / — | 0.960 / **5.87** / 0.740 / 0.89 | 0.040 / 17.6 / 0.11 |

The declaration's own note (`DMControlTrackCom` docstring / `VISUAL_NOTE`) measured the skybox at
**4.1–4.7/255 with 89–92% within 8** and the floor at **4.3–8.2/255** (walker's floor is the minified far band
the note also names). Under both versions the skybox sits exactly there and the bodies do not move; the floor
goes 7.8 → 19.3, 5.0 → 20.4, 12.5 → 33.0 and only 4–8% of its pixels stay within 8. The brief's premise that
"the residual was concentrated in the skybox" is true only as a pixel count (the skybox is 53–83% of the
frame and carries ~4/255 everywhere, in both arms); the *change* since the declaration is entirely floor.

**Which side moved (same node, same states, `cross/cross.json`):**

| config | official 3.11 vs 3.12: whole / skybox / floor / bodies | ours 3.11 vs 3.12 | ours(3.11) vs official(3.12) |
|---|---|---|---|
| DMCCheetahRun | 1.99 / **0.04** / **14.75** / 0.16 | 0.000 | 6.51 (= arm 3.12) |
| DMCHopperHop | 2.41 / **0.02** / **16.90** / 0.09 | 0.000 | 6.55 |
| DMCWalkerWalk | 10.02 / **0.04** / **25.21** / 0.70 | 0.000 | 15.95 |
| DMCHumanoidStand | 0.79 / — / 0.82 / 0.10 | 0.000 | 6.33 |

The reference's skybox does not move between versions; its floor moves by 15–25/255 on the three failing
domains and by 0.8 on the control, and the sheets (`cross/*_cross.png`, `local/official_3110_vs_3120_appleGL.png`)
show the checker's light and dark cells swapped with only the `mark="edge"` lines staying put. Our renderer
draws the same frame under both. The Apple-GL run of the same comparison (`local/official_drift_frozen_corpus_appleGL.json`)
gives the same split (floor carries 97.7–99.6% of the drift; skybox 0.01–0.06), so this is a property of
libmujoco, not of one GL driver.

**`STABLE_ENVS_LIGHTSPEC=0` ablation (stock `mujoco_warp` lighting, 3.12 arm, `localize312_nolightspec/`):**
bit-identical to the default run on every cell. stable-envs' own render-path specialisation is not a factor.

## 6. `check_resolution_consistency` — a bound-provenance finding, not a texture one

- The check compares our S render with our 2S render box-downsampled; the reference is never called and
  `tex_data` is identical across versions, so MuJoCo's plane change cannot enter it, and it does not: the
  number is identical to 4 decimals under 3.11 and 3.12, on gpu2108 and gpu2113, in six separate processes.
- `resprobe.py` (`resprobe311.json`, `resprobe312.json`) re-measures under the bound's **own** protocol
  (8 states from `render_worker.STATE_SEED`, default appearance) and reproduces the recorded
  `DOWNSAMPLE_EXACT_FRAC` **to six decimals on both arms**: cheetah 0.533854 (recorded 0.533850), hopper
  0.550223 (0.550220), walker 0.628951 (0.628950), humanoid 0.603653 (0.603650). The bound is sound and the
  renderer has not moved since it was measured — on an H200 under mujoco 3.9, which is a stronger statement
  of renderer stability than the brief asked for.
- On the frozen 64-state corpus the same check gives 0.5304 / 0.5526 / 0.6188 / 0.6095; the first 8 corpus
  states give 0.5322 / 0.5514 / 0.6241 / 0.6177. The quantity depends on the state set by ±0.01 (how much
  speckled sky and far floor is in view), and the margin is 0.005. Walker's "failure" is therefore the bound
  being applied to states it was not measured on.
- Where the inexact pixels are: skybox 76% / 81% / 58% on cheetah / hopper / walker (the speckled gradient, as
  the `DOWNSAMPLE_EXACT_FRAC` note predicts), floor 15% / 14% / 24%.
- **Open question:** Step 4's own values (0.5208 / 0.5350 / 0.6031 / 0.6088 on gpu2105, job 6176998) are not
  reproduced by any of the six later runs of the same script on the same corpus (0.5304 / 0.5526 / 0.6188 /
  0.6095). The provenance record has `git_commit: null` (the `$SCR/stable-envs` snapshot is not a git
  checkout), so the likeliest explanation is that Step 4 ran on a code snapshot that differed from the
  baseline in how the 2S context is built; it cannot be settled from here. It does not change the verdict
  (cheetah and hopper pass the recorded bound on the corpus today; walker fails it by 0.005 for the state-set
  reason above).

## 7. Causal verdict per config

| config | `check_visual_fidelity` | `check_resolution_consistency` |
|---|---|---|
| DMCCheetahRun | **B — reference-renderer version drift** (MuJoCo 3.12.0 `cc7fb98c` finite-plane texture re-anchoring). Passes under 3.11 (4.99 vs ceiling 6.5); skybox residual unchanged between versions (4.14); floor 7.8 → 19.3; official floor moves 14.8, official skybox 0.04; ours moves 0.000; model bytes identical; `mujoco_warp` plane UV unchanged | passes today on the corpus (0.5304 ≥ 0.5289); bound reproduced exactly under its own protocol; Step 4's 0.5208 not reproduced (open) |
| DMCHopperHop | **B** — same evidence: passes under 3.11 (4.37 vs 5.7); skybox 4.02 both arms; floor 5.0 → 20.4; official floor moves 16.9, skybox 0.02 | passes today (0.5526 ≥ 0.5452); same as above |
| DMCWalkerWalk | **B** — passes under 3.11 (7.86 vs 10.5); skybox 4.05 both arms; floor 12.5 → 33.0 carrying 82% of the residual; official floor moves 25.2, skybox 0.04; largest floor share of the three, hence the largest failure | **bound-provenance defect** (not A, not B): 0.6188 on the 64-state corpus vs 0.6241 on its first 8 vs 0.6290 recorded on the 8-state protocol; margin 0.005 < state-set variation |
| DMCHumanoidStand (control) | passes under both; square floor → whole-period shift → official drift 0.8 | passes |

No (A) stable-envs defect is demonstrated. Our renderer draws the same picture under both MuJoCo versions,
that picture matches the reference it was declared against to within the declared draw spread, and the
residual it carries under 3.11 (skybox one-tap sampling, floor minification) is the residual the declaration
already names. The one upstream change (`cc7fb98c`) explains all three visual failures and the reacher
finding; it does not explain the resolution failures, which have a different, smaller cause (§6).

## 8. Recommended action per config

- **Cheetah, Hopper, Walker — `check_visual_fidelity`: do not touch the renderer, do not weaken thresholds.**
  Neither "match 3.12's floor phase" nor "loosen `mean_abs_max`" is right: the first would make stable-envs
  disagree with the 3.11 reference it was calibrated against and with `mujoco_warp` upstream (which still
  implements the centred mapping), the second would hide a 2.5–4× floor error behind a number. The clean fix
  is provenance: certify against the reference version the declaration names (pin `mujoco==3.11.0`,
  `dm_control==1.0.44`, `mujoco-warp==3.11.0` in the certification venv), OR re-declare under 3.12 with a
  new `_AT` string and the honest 3.12 numbers (mean 6.5 / 6.5 / 16.0, within-8 0.80 / 0.79 / 0.55, SSIM
  0.58 / 0.55 / 0.53 — Step 4 and both 3.12 arms agree) and a third mechanism in `VISUAL_NOTE` ("MuJoCo
  ≥ 3.12 anchors finite-plane textures to the corner; `mujoco_warp` does not; the floor checker is phase
  shifted"). Coordinator's call. Watch `mujoco_warp` upstream: if a later release adopts the corner
  anchoring, the 3.12 numbers drop back on their own and the pin can move.
- **Walker — `check_resolution_consistency`:** re-record `DOWNSAMPLE_EXACT_FRAC` for the five visual-tier
  DMC cells on the protocol Step 4 applies it with (the frozen corpus) or apply it on the protocol it was
  measured with (`render_worker.sample_inputs`, N = 8), and give this family a margin sized from the measured
  state-set spread (≥ 0.01). Owner: whoever owns `tests/gpu/test_reproducibility.py`; not a texture edit.
  Cheetah and Hopper pass this check on the corpus today; their Step-4 FAIL is the unreproduced-run question
  in §6.
- **DMCHumanoidStand:** no action.
- **Reacher (Step 5's stored-frame finding):** same upstream cause; same policy answer.

## 9. Was the declaration tied to an older reference version? — yes, explicitly

- `stable_envs/envs/dmcontrol.py`, `_AT` — the `measured_at` field of every `VISUAL_FIDELITY` entry for
  `DMCCheetahRun`, `DMCWalkerWalk`, `DMCHopperHop`, `DMCHumanoidStand`: *"… H200 / driver 590.48.01 /
  MUJOCO_GL=egl, mujoco 3.11.0 + mujoco-warp 3.11.0, against dm_control's physics.render at camera 0"*;
  `_SWIMMER_AT`: *"mujoco 3.11.0 / warp 1.16.0"*.
- `pyproject.toml`, extra `dmcontrol = ["dm-control>=1.0.44", "mujoco>=3.11"]` with the comment *"3.11.0 +
  1.0.44 is the pair this repo's DM Control parity numbers were taken under"* — a floor, not a pin, which is
  what let Step 4 certify under 3.12.
- `tests/gpu/test_reproducibility.py`, `MEASURED_ON` for `DOWNSAMPLE_EXACT_FRAC`: *"H200 … mujoco 3.9; 8
  states per env from sample_states(rng=default_rng(render_worker.STATE_SEED))"*.
- `examples/results/step4/conformance/dmc.json` → `provenance.reference_packages`: `mujoco 3.12.0`,
  `mujoco_warp 3.12.0` — Step 4's own record of the mismatch.

## 10. Proposed provenance / version policy (for the coordinator)

1. A fidelity declaration is a tuple (numbers, reference package versions, card). Store the versions as
   fields, not prose — e.g. `VISUAL_FIDELITY[...]["reference"] = {"mujoco": "3.11.0", "dm_control": "1.0.44",
   "mujoco_warp": "3.11.0"}` — and have the certifier compare them with the interpreter's
   `importlib.metadata` versions and record `blocked_environment: reference version mismatch` rather than
   `fail` when they differ, the same rule Step 4 already applies to a missing reference ("a dependency failure
   is not a fidelity failure").
2. Pin, do not floor, the reference pair used for certification, and bump it only together with a
   re-declaration commit that records the new numbers and the upstream change that motivated them.
3. Every conformance artifact already records `reference_packages`; `certify.py` should refuse to judge a
   `visual_fidelity` bound whose declared reference version differs from the artifact's.
4. For reference-free bounds (`DOWNSAMPLE_EXACT_FRAC`) the protocol (state seed, N, card class) is part of the
   bound: apply it on the protocol it was measured with, or re-measure on the one it is applied with.

## 11. SLURM jobs, blockers, artifacts

| job | what | outcome |
|---|---|---|
| 6328158 | Sami's `scratch/step6/visual311.sbatch` on `3090-gcondo` (would also have failed: `--configs` space-separated where `fidelity_step4.py` takes one comma-separated string; `PYTHONPATH` at the stale `$SCR/stable-envs` checkout) | cancelled by me once 6343431 was RUNNING (authorized duplicate) |
| 6343431 | `visual311_normgpu.sbatch`, gpu2108 (RTX 3090, 575.57.08), 6 min: built `se-vis311`; 3.11 arm failed at `import stable_envs` (venv lacked core dep `gymnasium`); 3.12 arm complete (replay, localize, lightspec ablation) | `visual312/`, `localize312/`, `localize312_nolightspec/` |
| 6343518 | second copy of the same script, submitted in the seconds 6343431 was flipping PENDING→RUNNING | cancelled by me immediately (own job, never ran) |
| 6343699 | `arms_rerun.sbatch`, gpu2113 (RTX 3090, 575.57.08), 3 min: 3.11 replay + localize + resprobe; 3.12 localize + resprobe (same card); cross | `visual311/`, `localize311/`, `localize312/` (overwritten, same numbers), `resprobe31{1,2}.json`, `cross/` |

Blockers: none remaining. The 3.11 arm cost one extra job because the `se-vis311` build line lacked
`gymnasium`; fixed in `visual311_normgpu.sbatch` for the record and on the venv itself.

Artifacts (`examples/results/step6/texture-fidelity/`): `gate.json` (frozen bounds applied per arm),
`visual311/`, `visual312/` (replay shards + `comparisons/*.png`), `localize311/`, `localize312/`,
`localize312_nolightspec/` (per-surface JSON + `sheets/*.png`; the `frames/*.npz` stacks stay on Oscar and
in the local worktree, not committed), `cross/` (same-card cross-arm JSON + sheets), `resprobe311.json`,
`resprobe312.json`, `local/` (Mac evidence: texture-byte digests, official drift on the frozen corpus,
reset-frame probes, side-by-side sheet). Scripts (`scratch/step6/texture-fidelity/`):
`visual311_normgpu.sbatch`, `arms_rerun.sbatch`, `resprobe.sbatch`, `localize.py`, `cross.py`, `gate.py`,
`resprobe.py`, `texdigest.py`, `localcorpus.py`, `localrender.py`.

Report-format items not covered above: profile-before — not applicable (diagnosis workstream, no timing
claim); intervention — none implemented; tests run — none needed for a no-code-change finding (the replay
harness is `examples/fidelity_step4.py` itself, run under both arms); Step-4 replay — done (§4); tiers
before/after — unchanged; pixels changed — no; semantic follow-up — none (no renderer change).

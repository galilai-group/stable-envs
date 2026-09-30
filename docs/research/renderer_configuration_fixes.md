# Maze, Finger and Scene renderer configuration corrections

Starting HEAD: `df80c9c54590afa9fc0482a4dcb241d21f6b63a3`. Pinned stable-worldmodel source: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
Worktree: `/Users/samib/stable-envs/.worktrees/render-config-fixes`, branch `repair/render-config-fixes`.
The preceding Fetch corrections are preserved. No semantic-status promotion, shared State/RenderState redesign, or push was performed.

## Source mismatches corrected

- **OGBMaze:** the registered source defaults to `ob_type='states'`, whereas our renderer forced `pixels`. Pixel mode replaces the XML checker with a position-encoding gradient, removes the goal marker and torso light, and alters global ambient lighting. The historical point/large/navigate renderer now selects the actual registered configuration. Its existing `task.goal_pos` State leaf reaches the renderer's private five-coordinate projection (agent XY plus goal XYZ). The full simulator State schema is unchanged. Reference and sampling helpers use that projection too. Scene and plate caches include observation mode so pixel-observation siblings retain their original configuration. The actual state-mode light table replaces the two-light pixel-mode calibration for this registration.
- **FingerDMControl:** source `render_target=0` changes the target material's alpha, hiding both `target` and `tip`. Our visibility control hid only `target`, leaving the red tip. The registered class now defaults to hidden and groups both sites under its existing visibility control.
- **OGBScene:** the source assigns each button's geometry color independently, but both geoms share `btn_red`. MuJoCo honors the geometry override; MJWarp uses the material RGB. The render-only model now duplicates that material for each button and synchronizes the copies with the source-derived geometry colors. Texture, specular, shininess, reflectance and emission attributes are preserved. Simulator models and the frozen render algorithm are unchanged.

No shader implementation or additional rendering pass was introduced. The Maze fix does restore the source's previously omitted goal geometry and lighting; this work does not establish performance neutrality.

## Matched-frame evidence

The new candidate images use the exact archived full States and appearances. Source CPU/GPU images are reused unchanged from the isolated fresh renderer campaign, not regenerated in these jobs. Maze and Finger are measured on both candidate backends; Scene's candidate CPU renderer remains unavailable.

| Registration | Frames per candidate backend | Old maximum GPU/source-GPU MSE | Corrected maximum |
|---|---:|---:|---:|
| `swm/OGBMaze-v0` | 42 | 0.1029564672 | 0.0004145475 |
| `swm/FingerDMControl-v0` | 24 | 0.0003442044 | 0.0001597850 |
| `swm/OGBScene-v0` | 132 | 0.0102497852 | 0.0091995301 |

MSE is mean squared RGB difference after normalization to [0,1]; these are maxima over the retained corpus, not acceptance thresholds. Maze improves about 99.6%, Finger 53.6%, and Scene 10.2% by this metric. Exact figures, all backend comparisons and per-frame hashes are in the linked evidence.

Each candidate capture is repeated at unchanged State and checked for exact repeatability and no supplied/live State mutation. A four-lane Scene batch with independently colored buttons and all four bit combinations agrees with individual renders. All **57 Cube control frames remain byte-identical** to the original candidate captures.

The local exporter independently recomputes **1,284 candidate/source MSE comparisons** from **321 candidate captures**, validates all 1,284 downloaded RGB-array hashes, checks the original maxima against the previous CSV, checks unchanged source CPU/GPU gaps, and round-trips the exported PNGs. The updated 42-registration table changes only the three affected rows, retaining the preceding Fetch corrections and all other rows.

## Limits and remaining work

These corrections do not establish pixel parity. Scene/Cube still lose dynamic floor shadows in the static-plate composite. Walker reflections are still missing. Antialiasing, material/shading differences and other recorded renderer gaps remain. Maze's remaining error includes agent/goal shading and effects beyond the configuration correction; the other maze variants and the full visual-variation space were not exhaustively re-certified. The source Maze GPU's previously documented tiny repeatability difference remains disclosed in the CSV.

## Artifacts

- [Updated 42-registration CSV](semantic_recovery_evidence/renderer_configuration_fixes/renderer_differences_42_environments.csv)
- [Before/after image gallery](semantic_recovery_evidence/renderer_configuration_fixes/images.html)
- [Per-registration summary](semantic_recovery_evidence/renderer_configuration_fixes/summary.json)
- [Per-frame evidence](semantic_recovery_evidence/renderer_configuration_fixes/evidence.json)
- [Independent local verification](semantic_recovery_evidence/renderer_configuration_fixes/local-validation.json)
- [Terminal job evidence](semantic_recovery_evidence/renderer_configuration_fixes/terminal-evidence.txt)
- [Cleanup investigation](semantic_recovery_evidence/renderer_configuration_fixes/cleanup-investigation.txt)

Oscar campaign: `/oscar/scratch/sboughan/se-semantic-recovery/render-config-df80c9c5-20260928`. Source corpus: `render-rerun-df80c9c5-20260928T021357Z` under the same recovery root. Finger/Maze use the accepted MuJoCo 3.11 renderer profile; Scene/Cube use MuJoCo/MJWarp 3.12.0, OGBench 1.2.1 and Warp 1.17.0. Exact profiles are embedded in evidence.json. Every revision has an immutable source overlay and SHA-256 manifest.

Raw captures and job scripts are retained locally under `tmp/render-config-20260928/` in the durable worktree. The older Fetch and original renderer galleries were also preserved under `tmp/renderer-evidence/`.

Final terminal successes: Maze CPU `6753124_0`, Maze GPU `6753125_0`, Scene GPU `6753097_2`, Finger GPU `6752924_1`, Cube control `6752924_3`, and focused CPU tests `6753107_0` / `6753107_1` are all `COMPLETED 0:0`. The focused CPU suites pass 11 tests without skips, and the local benchmark verifier passes 17. Finger CPU's 24 checked captures come from `6752923_1`; that job failed its subsequent misconfigured test, which passed after explicit CPU selection in the final CPU suite. The captured images themselves passed repeatability, State nonmutation and independent local recomputation.

The final Scene tests also compare MuJoCo's independently resolved scene colors for all button bit patterns, independent lock colors, exact default grey 0.5, and nearby nondefault grey 0.50001. Exact-default grey preserves MuJoCo's material fallback. Final Maze GPU captures differ slightly from the intermediate run's OpenGL plate on another GPU; the table uses the final run and each run passes exact within-process repeatability.

## Diagnostic failures retained

The initial Maze trial exposed two assumptions specific to pixel mode: all non-wall geoms move with the agent, and the scene has two lights. Subsequent fixes added the existing State's goal position and actual source light table; neither assertion was relaxed to accept incorrect images. The initial Finger CPU model test accidentally selected the class's CUDA default; it was rerun with explicit CPU selection. The initial Scene test compared material IDs across models after inserting two materials; it now compares the same named arm materials. A broad pytest selector also selected Finger under Scene's incompatible dm-control profile; the final run selects the intended tests explicitly. Failed runs remain in the campaign and are not used as terminal success evidence.

The added Maze reference-helper test initially let its reference object be garbage-collected, tripping the existing GL-context lifetime guard. It now retains that object until process exit; the guard remains enabled and the final CPU/GPU jobs both pass.

The frozen `stable_envs/envs/ogbench/manipulation.py` SHA-256 remains `bac4b1ea3e29a04faa5eeec33de0c212cb87bbf488ae0466ea762bd3a140576a`.

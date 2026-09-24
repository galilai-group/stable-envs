# Cube/Scene checkpoint handoff — one failed Oscar gate

Cube and Scene remain **INCOMPLETE**. CPU semantics and named RenderState/model
integration passed locally. Oscar job **6669720** failed before rendering because
my selective archive omitted a transitive test fixture. There is no GPU pixel
acceptance evidence, and neither family is promoted. A corrected bundle has
passed its local unpacked CPU gate and is ready for resubmission review.

## Git and checkpoint identity

- Worktree: `/private/tmp/stable-envs-semantic-recovery`.
- Branch: `repair/environment-semantics`.
- Starting clean HEAD / original CPU evidence / prior ending checkpoint:
  `72612563590f12a081d940394496795a6c024577`.
- CPU implementation: `14456339ec7f1b28ad698fc3158fe9e4ed8f6d1e`.
- Dependency audit: `75ef812977423fa059b857b36542119e101a430f`.
- RNG feasibility: `434cc72974360ab98e47cad7aebcc8b1b62bb1de`.
- Prior documentation: `03adc1ab79bae71e3e62db2e5c4cb19f20723166`.
- Pinned SWM oracle: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
- Submission source and corrected archive both use committed tree
  `294675f039e9b7f867c4bff15aac2b5f2e572bf6`.
- The final evidence commit contains this handoff. Its exact SHA is returned in
  the final response and can be read with `git rev-parse HEAD`; it cannot be
  embedded in its own hashed tree. Worktree is clean after that commit.
- No push occurred. The main checkout and other family worktrees were not edited.

Commits before the final evidence checkpoint:

```text
0aaa72fed5cc1a63c46af02aa448bebf6e279838 Close Cube and Scene CPU bookkeeping and characterize atomic restore
2f4b2a6e5c0bc1ed8455d8235c1eb0e26ce70ff7 Integrate named Cube and Scene RenderState with the frozen renderer
457afa0b401a3fa757c94eb8a8b6a3de3074585c Record Cube and Scene local renderer gates and immutable Oscar runner
294675f039e9b7f867c4bff15aac2b5f2e572bf6 Verify complete independent Cube and Scene Oscar CPU test selections
```

The final commit records the actual failed GPU evidence, pending matrix/map
status, corrected archive preparation and this handoff. No prior checkpoint was
rewritten or squashed. The extra small local-runner commit corrected family test
selection before the single submission; 45 Cube tests and 48 Scene tests cover
all 92 distinct cases, with the shared isolation case run for each family.

The complete changed-file inventory is
[changed-files.txt](semantic_recovery_evidence/ogb_renderer_closeout/changed-files.txt).
The scope review and expected versus actual paths are in
[final-diff-review.json](semantic_recovery_evidence/ogb_renderer_closeout/final-diff-review.json).
All evidence and changed source digests are in
[SHA256SUMS](semantic_recovery_evidence/ogb_renderer_closeout/SHA256SUMS).

## CPU and renderer conclusions

| Registration | CPU 3.8.1 | Local CPU 3.12 | Local RenderState/model | Oscar CPU | GPU pixels | Promotion |
| --- | --- | --- | --- | --- | --- | --- |
| swm/OGBCube-v0 | PASS | PASS | PASS | collection failed | not run | INCOMPLETE |
| swm/OGBScene-v0 | PASS | PASS | PASS | collection failed | not run | INCOMPLETE |

Restore remains the audited atomic reconstruction path. Median same-model
restore latency is Cube **181.4 ms (B=1), 591.2 ms (B=3)** and Scene **211.8 ms,
705.6 ms**, about **150–193×** step. Cold/warm median, p95, throughput,
changed-configuration timings and staged costs are in
[CPU closeout](ogbench_cpu_closeout.md). A fingerprint-validated buffer-swap fast
path is proposed for later review; no restore optimization was implemented.

The candidate unified profile is MuJoCo 3.12.0, OGBench 1.2.1, NumPy 2.2.6,
Gymnasium 1.2.3, dm-control 1.0.41; GPU pins mujoco-warp 3.12.0 and warp-lang
1.17.0. Oscar installed these exact versions with Torch 2.14.0+cu126 and
TensorDict 0.14.2. Installation is not semantic validation: Linux CPU tests failed
collection, so the unified GPU profile remains unaccepted. The prior 3.8.1 CPU
claim remains valid. There is no cross-version bit-identity or GPU physics claim.

The final RenderState has **14 named numeric leaves for Cube / 16 for Scene**,
covering family/asset/schema signature, camera/resolution/supersampling, qpos,
goal mocap position/quaternion, size, compiled agent/camera/light/floor appearance,
current cube/target RGB, fixed target alpha, and Scene locks/button bits. It owns
storage, validates the public batch, and is rejected by `set_state`. Full schema,
compiled-versus-pending distinction and runtime ownership:
[renderer report](ogbench_renderer_closeout.md) and
[schema](semantic_recovery_evidence/ogb_renderer_closeout/renderstate-schema.json).

Scene's old frozen constructor omitted upstream's required `env_type`. The
caller now supplies a correctly configured independent compatible model, then
inherits the frozen drawing implementation. Production uses OGBench assets and
the narrow stable-envs compatibility layer. Source MuJoCo's visual-tier residuals
remain distinct from the required exact frozen-baseline comparison.

Frozen renderer SHA-256:
`bac4b1ea3e29a04faa5eeec33de0c212cb87bbf488ae0466ea762bd3a140576a`.
It is unchanged; no thresholds, camera corrections, shared contract, or baseline
was modified. Frozen process-lifetime EGL ownership remains explicit, including
its memory limitation for many distinct static configurations.

## Actual Oscar evidence

- Job **6669720**, `gpu` / `norm-gpu`, no requeue; NVIDIA L40S on `gpu2708`.
- Terminal state **FAILED**, exit **1:0**, elapsed **00:00:57**.
- Exact terminal marker: `OGB_CUBE_SCENE_GPU_GATE_FAIL`.
- Cube and Scene CPU exits: **4** (collection); renderer exits: **99** (skipped).
- Failure: missing `tests/fixtures/regression/support_matrix.json` imported by
  the harness crosswalk from `conftest`. This is a bundle packaging failure,
  not a measured renderer mismatch. No candidate/baseline frames exist.
- Submitted source archive SHA-256:
  `21bff09fc8d3e9f0b14bba291736af5dba82c42a4a7bea3cae89d2f162dab27e`.
- Pinned oracle archive SHA-256:
  `9d8fefcd19337522f3f5cf37379703c48fc2974e958f22f9c849a8c3a9b33b7d`.
- Real downloaded remote evidence archive SHA-256:
  `14e52fc42e7278f53138902c7f57db75992112ddc51139888738b4bde4333c61`.
- Local versus remote archive/log hashes and all **12** evidence files verified.
  The downloaded archive, extracted bytes and SLURM log are retained under
  `semantic_recovery_evidence/ogb_renderer_closeout/remote/`.
- Only **one job** was submitted. An earlier staging bundle was superseded
  before submission and never ran. No automatic retry or requeue occurred.

## Validation and preservation

The 3.8.1 closure gate passed **86 tests** before rendering changes. The 3.12
CPU suite passed **86**, and combined with new RenderState tests **92**. Final
boundary checks passed **6** after integer override validation. Model inputs
matched the pinned oracle for **35/35** fixture cases. The corrected archive,
extracted separately from the checkout and importing its own source/oracle,
passed **92 tests, zero skips**. This is CPU-only validation on the Mac.

Shared State/harness/readiness/taxonomy/matrix tests passed **129**, with **9**
expected skips, after final map changes. Strict MkDocs, Ruff format/lint,
shell syntax and whitespace checks passed. Raw logs and downloaded evidence
are kept verbatim and excluded from whitespace normalization. Exact executable
test commands, results and logs:
[commands.json](semantic_recovery_evidence/ogb_renderer_closeout/commands.json)
and `scripts/recovery/ogbench_renderer_gate.sbatch`.

Preservation audit confirms **33 accepted rows**, **170 artifact hashes**,
**11 terminal logs**, **485 pinned source files**, and the Fetch/RNG/dependency/
CPU manifests (75/27/15/34 entries). Both original CPU implementation modules
are byte-identical. OGBench Maze, Fetch, PushT, Craftax, Rocket, all unrelated
renderers and shared contracts are unchanged. All production ASTs contain zero
stable-worldmodel imports. There is no OGBench fork or copied complete SWM
Cube/Scene implementation. Prior acceptance was revalidated by unchanged rows,
artifact/terminal hashes and shared regression tests; prior GPU suites were not
rerun outside this single-job scope.

Final actual matrix: **33 VERIFIED / 9 INCOMPLETE**. Remaining incomplete:

- `swm/CraftaxClassicPixels-v1`
- `swm/CraftaxClassicSymbolic-v1`
- `swm/CraftaxPixels-v1`
- `swm/CraftaxSymbolic-v1`
- `swm/OGBCube-v0`
- `swm/OGBScene-v0`
- `swm/PFRocketLanding-v0`
- `swm/PushT-Discrete-v1`
- `swm/PushT-v1`

## Exact resumption after review

The corrected bundle is fully prepared locally, **not submitted**, at
`/private/tmp/ogb-renderer-294675f0-complete`. Its source archive contains the
entire tracked tree at the same source commit, fixing transitive harness data
omissions without changing candidate code, profile, oracle, comparisons or
runner. The fresh extraction passed all 92 CPU/RenderState tests.

Corrected source SHA-256:
`a002c3aa72ea16a841858ef1d53123009f40ff681e2afa0dcf594235b9fe46c2`.
Runner SHA-256:
`88aa1d237de31ab2f9a783d1ab26473e9186af39e1b1cdd1014dc7272790b134`.
The oracle hash is unchanged. See
[corrected manifest](semantic_recovery_evidence/ogb_renderer_closeout/corrected-bundle.json)
and [checksums](semantic_recovery_evidence/ogb_renderer_closeout/corrected-bundle-SHA256SUMS).

Review the concrete packaging failure and corrected bundle before authorizing
another Oscar job under the one-job limit. After that review, use a fresh remote
folder; do not overwrite or resubmit the failed bundle. The prepared commands are:

```bash
ssh -S /private/tmp/stable-envs-oscar-control sboughan@sshcampus.ccv.brown.edu \
  'mkdir /oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete'
scp -o ControlPath=/private/tmp/stable-envs-oscar-control \
  /private/tmp/ogb-renderer-294675f0-complete/{source.tar.gz,oracle.tar.gz,gate.sbatch,bundle.json,SHA256SUMS} \
  sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete/
ssh -S /private/tmp/stable-envs-oscar-control sboughan@sshcampus.ccv.brown.edu \
  'cd /oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete && sha256sum -c SHA256SUMS && test ! -e submission.txt && sbatch --parsable --chdir=/oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete --output=/oscar/scratch/sboughan/se-semantic-recovery/ogb-renderer-294675f0-complete/slurm-%j.log gate.sbatch | tee submission.txt'
```

Record the new ID without automatic retries. Require exact per-family success
markers plus `OGB_CUBE_SCENE_GPU_GATE_PASS`, download the actual remote archive,
verify remote/local hashes and every evidence entry, inspect frames/comparisons
and lifecycle results, then run per-registration promotion/readiness gates.
Keep each registration INCOMPLETE until its entire acceptance standard passes.
No other family work or push is part of this resumption. Stop for review here.

# Documentation cleanup plan

Review checkpoint: `d424517f7766a16bcb3554469b0f42704c7d0fc5`, branch `sami-dev`.
Craftax and archived research/evidence are outside this cleanup.

## Rules for each file

1. Read the implementation before deciding whether it needs a module docstring.
   Default to no header. Keep one only when the file has a contract or architecture
   that readers cannot readily infer from its public objects.
2. Rewrite retained documentation from scratch. Aim for roughly 20% of maintained
   files with headers, usually much shorter than the approximately 50-line ceiling.
   This is a coverage budget, not a reason to fill space or erase useful API docs.
3. Document public inputs, outputs, ownership, precision, device placement, and
   meaningful failure conditions where they belong. Remove private docstrings that
   merely repeat a name or translate the next statement into English.
4. Keep comments to one or two lines explaining a current constraint. Preserve
   ordering, aliasing, RNG, rounding, synchronization, and lifetime requirements
   that could otherwise cause silent bugs. Remove chronology, benchmark stories,
   completion claims, and section banners.
5. Do not move everything removed into another document. Git already retains it.
   Preserve licenses and notices. Check `__doc__` consumers before removing CLI
   descriptions or generated documentation.

The local reference projects support selective placement: stable-worldmodel has
module headers in 39/118 library files, while stable-pretraining has 86/132.
They are style references, not evidence that fewer total docstring lines is always
better. The 20% target is our chosen constraint.

## Completed first batch

| File | Header decision | Total file lines, before → after |
| --- | --- | ---: |
| `stable_envs/envs/fetch.py` | Remove; explain renderer/simulator composition on the class | 1,285 → 916 |
| `stable_envs/envs/pusht.py` | Remove; distinguish the compact solver from public PushT locally | 715 → 511 |
| `stable_envs/envs/navigation.py` | Remove; retain local numeric and RNG constraints | 2,146 → 1,506 |
| `stable_envs/envs/__init__.py` | Remove; discovery calls explain the file | 22 → 5 |
| `stable_envs/env.py` | Retain a 17-line shared contract | 1,358 → 985 |

Total: **1,603 lines removed**, including 1,107 docstring lines. One of five files
retains a header; every standalone comment block is at most two lines. The private
float32 renderer projection is explicitly distinguished from full simulator State.

One Fetch test asserted the obsolete docstring phrase “NO DYNAMICS.” It now checks
that registered classes combine `FetchSemantics` with the renderer base instead.
No production execution, signatures, thresholds, or error messages changed.

## Follow-up checkpoint

The next pass starts from `df2d46728e5162571e89963627fad98233372e07` on `sami-dev`.
It rewrites **74 additional library files** plus **11 shared contract/helper files**.
Of those 74 library files, **14 retain headers (19%)**, with a maximum of nine
lines. Every standalone comment block in the reviewed library files is at most
two lines. Runtime code, diagnostic strings, generated class metadata, source
pins, fidelity thresholds, and committed evidence remain unchanged.

The review also corrected inaccurate documentation: discovery records import
failures; composite validation does not retain leaf conversions; render projections
cannot restore simulator State; and a skipped node-health probe is not a successful
renderer check. Numerical and ownership requirements stay beside the relevant code.

| Plan area | Completed in this pass | Still to review |
| --- | --- | --- |
| Renderers | The five priority files, plus `scene`, `primitives`, `navdot`, and `_plate_texels` | CPU3D, raycast, fused/classic gfxdraw, light-spec, tilemap, RenderNet |
| Stepping | Shared stepping, process request ownership, five process executors, four execution selectors, step graphs | Family simulation adapters |
| Core | Specs, spaces, registry, dynamics, configuration, style, GL setup, package exports | None of the listed core files; `core/` is no longer a tracked package |
| Environments | Six priority modules; DMC domain declarations, semantic wrappers, model/shading/tendon helpers, exports, OpenGL path | DMC base/suite/trackcom; other environment families and their adapters |
| Data/testing | `_data.py` and every module under `stable_envs/testing/` | No remaining files in this area |
| Tests | All eight contract files and three top-level helpers; classic State consolidation; discovery behavior checks; visibility-collapse negative control repaired | Reference helpers and remaining test modules; other redundancy recommendations |
| Tools | CLI descriptions retained and smoke-checked for supervisor/render worker | Maintained scripts, examples, benchmarks, root configuration |

This is a substantial checkpoint, **not completion of the whole-tree header target**:
83/148 non-Craftax library files still have headers (56%). The six large priority
environment modules are complete, but the remaining families still need individual
implementation review. Craftax and immutable research drivers remain outside scope.

Validation for this checkpoint:

- Executable ASTs match the starting commit for all 74 production files and all
  11 documentation-only test/helper edits. The three intentional test changes have
  separate behavioral checks; no production implementation changed.
- Renderer checks: 180 passed, 1 skipped. API/process/core checks: 609 passed,
  95 skipped. Constructor/declaration subset: 24 passed. Some later targeted runs
  overlap these suites, so these are suite results, not a unique-test census.
- Consolidated classic tests: the same 12 cases pass before and after. Shared
  contracts, semantic harness, classic tests, and discovery checks: 31 passed,
  77 skipped. Live-source profiles were not requested; CUDA cases require Oscar.
- Additional CPU helper checks passed 128 cases; the visibility-collapse case
  exposed a stale fixture method reference, now repaired. All seven visibility
  cases pass, including the injected per-row collapse.
- Utility/provenance checks: 117 passed with three failures reproduced at the
  starting commit: an archived driver import path, a removed HumanoidStand identity,
  and a reference-arm record that no longer regenerates identically. Evidence was
  not regenerated or overwritten.
- Safety harness: 11 passed; its timeout test fails identically at the starting
  commit in this environment. Other broad static checks expose stale module paths,
  existing source-identity symbols, and local cache directories; these are not
  reasons to erase tests or local work.
- Broader graphics checks cannot run here: GLFW initialization aborted one run,
  and 13 DMC scene checks failed at macOS CGL initialization with an invalid
  CoreGraphics connection. No GPU parity or performance result is claimed from
  this documentation-only pass.
- Ruff, compilation, whitespace checks, and CLI help checks pass. MkDocs builds
  with the same four pre-existing research-link warnings.

The [redundancy audit](testing-redundancy-audit.md) records the completed classic
consolidation and the remaining precision-sensitive replay work.

## Review order

Each row is a review batch, but edits still proceed one file at a time. Header
decisions below are starting positions to confirm against each implementation.

| Order | Modules | Initial header decision and focus |
| --- | --- | --- |
| 1 | `render/mjwarp.py`, `render/raster2d.py`, `render/gfxdraw.py`, `render/nvdiff.py`, `render/_resources.py` | Keep short overviews only where backend ownership or raster conventions span functions. Put stream, buffer, and rounding rules beside the operations they constrain. |
| 2 | `_stepping.py`, `_adapters/process_requests.py`, `_adapters/fetch_process.py`, `_adapters/dmcontrol_process.py`, other process executors | Keep a short overview for shared sequence/executor contracts. Remove family headers that repeat them; retain snapshot ownership, terminal masking, and worker-lifetime rules. |
| 3 | `core/`, `specs.py`, `spaces.py`, `registry.py`, `_dynamics.py`, public package exports | Usually no header. Keep shape, broadcasting, dtype, and registration requirements on the public objects. Check generated class docs and discovery consumers. |
| 4 | Remaining `envs/` modules and `_adapters/`, family by family | Usually no header. Start with Jumanji, HumanoidBench, MetaWorld, PGX, MuJoCo locomotion, and RoboCasa: their headers alone total 1,120 lines. Preserve family-specific source semantics locally, without status narratives. |
| 5 | `_data.py`, `data.py`, policy/rollout utilities, `testing/` | Retain only cross-function contracts: allocation ownership, streaming, corpus schema, evidence validation. Put user-facing examples in the existing guide, without duplicating them in every helper. |
| 6 | `tests/contracts/`, `tests/helpers/`, then unit/integration/GPU/regression tests | Usually no header. Use test names for the behavior and short comments for unusual counterexamples. Apply the separate redundancy audit before polishing tests likely to be consolidated. |
| 7 | Maintained `scripts/`, `examples/`, `benchmarks/`, root configuration | Remove prose introductions by default, but preserve usable `--help` where it consumes `__doc__`. Check commands and examples against the current API. Leave immutable research drivers/evidence untouched. |

Paths in the table are relative to `stable_envs/` except the test/tool directories.
After the first batch, 143/148 non-Craftax library files had headers. About
30 retained headers would meet the target for that scope. Track tests and tools
separately so many tiny test files cannot disguise verbose library documentation.

## Validation and completion

- Compare parsed executable ASTs before/after, ignoring actual docstrings and
  explicitly reviewed generated `__doc__` values. Do not normalize other strings.
- Compile edited files; run formatting/lint and whitespace checks. Recount headers
  and comment blocks, but use human review to judge clarity.
- Run affected API/doc-consumer tests. Build the documentation and inspect any new
  warnings. CLI files additionally need a `--help` smoke check.
- Documentation-only edits with unchanged executable AST do not need new GPU
  measurements. Any behavioral test cleanup must preserve its CPU/GPU, source,
  fixture, threshold, and failure-detection coverage.

First-batch verification: executable AST equivalence for all five production files;
Ruff and whitespace checks pass; 336 focused checks pass (84 skipped), plus 40 audit
checks (5 CUDA cases skipped). MkDocs builds with four existing research-link warnings.
A broader run found two missing-Matplotlib failures and seven declaration-test
failures; all nine reproduce in an isolated copy of the starting commit. This is
not a claim that the entire suite passes. See the [test audit](testing-redundancy-audit.md).

# Shared testing cleanup checkpoint

Baseline: `sami-dev` at `4a8fb346f6b196de4b2f7c3c2852db8eec045ee4`.
The acceptance matrix remains **38/38 non-Craftax registrations** in its documented
scopes. This checkpoint changes testing infrastructure, not environment APIs,
physics, renderers, acceptance thresholds or previously accepted artifacts.

## Fresh independent source regressions

| CPU profile | Registrations | Required cases | Result |
|---|---|---:|---|
| native | CartPoleControl, PendulumControl | 24/24 | PASS |
| fetch | FetchReach | 16/16 | PASS |
| rocket | PFRocketLanding | 16/16 | PASS |

These four registrations exercise the shared State, simulation, task and online
State-capture contracts. Each profile uses seeds 17/42, batch 2 and its declared
action dtypes. Online checks include single actions and eight-action chunks with
none/all/final/reverse/empty State slices. Family-specific contact, termination,
Variation, renderer and multiprocessing acceptance gates remain separate and
required. This is **4/38 registrations migrated to the representative profiles**,
not a fresh all-environment acceptance or performance sweep.

Each profile directory contains its expected case set, exact candidate file
hashes, per-case progress, source checkout/runtime identity, results and its own
write-once seal. The source was the clean pinned SWM checkout at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. Tests ran locally on Apple M5, with the
isolated package versions in `cpu-source-runtime.json`. No GPU physics or CUDA
rendering is claimed by these CPU profiles.

## Measurement and migration controls

The local combined run recorded **317 passed, 2 failed, 5 skipped, 1 deselected**.
The two failures are both in the legacy `test_reference_provenance.py`:

- A direct registry lookup still expects the historical `DMCHumanoidStand` name.
- Its historical reference-arm regeneration consequently differs from the
  committed report.

Both failures reproduce on an untouched detached worktree of the starting commit:
**16 passed, the same 2 failed**. See `baseline-provenance.txt`. They were not
hidden, marked xfail, or fixed by rewriting historical evidence. The five skips
are four CUDA-only checks and one missing optional Matplotlib dependency. The
deselected test is the explicitly excluded legacy Craftax renderer-only check.
`checks.xml` and `checks.txt` retain the full outcomes.

The new migration controls include valid evidence, a deliberately corrupted
frame, missing frames/cases, false witnesses and wrong metrics. Old and new
Rocket verifiers agree on those outcomes. Additional controls reject duplicate
case IDs, missing visual groups, corrupted native corpus hashes/schema and
partially selected or dependency-blocked required profiles. The latter stay
nonzero even when unrelated frozen tests pass.

Shared metrics reverified **204 saved CPU frames and 204 saved GPU frames** from
the accepted Rocket corpus:

| Historical pixels | Mean normalized MSE | Maximum |
|---|---:|---:|
| CPU | 0 | 0 |
| CUDA | 0.0000462285830201 | 0.0000970718393567 |

This is read-only reanalysis of historical GPU output, not a fresh GPU run. Oscar
rejected SSH authentication during this checkpoint. All 76 artifacts in Rocket's
original `closure-v4/SHA256SUMS` still matched their committed hashes.

## Repository and remaining work

Repository checks now use Git-index membership, while checking working contents.
The existing ignored local scratch directory and unrelated untracked handoffs
were left untouched. Ruff and the maintained-file diff checks passed.

The strict docs build still stops at the same four archive-link warnings recorded
in the design review; see `docs-build.txt`. No new warning came from these changes.

The next migration is the common reached-state corpus format and the remaining
family-specific acceptance/benchmark drivers. Keep the two legacy Humanoid
provenance failures visible during that migration. Contributor commands and the
scope of each new profile are documented in [Testing](../../testing.md).

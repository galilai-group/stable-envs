# Saved-State testing cleanup: classic-control cohort

Initial candidate: `cc81789cb468ef2c5caffda000a27de44bf7a681`. Final candidate:
`d2d10f04` on `sami-dev` adds the measured frozen-corpus policy and verified-source
import ordering fix.
This cohort changes testing infrastructure, fixtures, CI selection and documentation.
It does not modify production simulation/rendering or the accepted registration matrix.

## What changed

- A sealed numeric State/task trajectory format with typed nested leaves, source
  and producer hashes, actions and source task outputs. Loading checks complete
  file integrity, registration identity and schema before constructing owned State.
- A source-only generator for all five classic-control defaults. Every fresh array
  is byte-identical to its preserved historical counterpart (seed 17, B=3, T=32).
- Acrobot, MountainCar and MountainCarContinuous now share the four live contracts
  already used by CartPole/Pendulum. The native profile requires 56 cases, including
  seeds 17/42 and each declared action dtype. Shared live profiles cover 7/38
  non-Craftax registrations; the remaining 31 still use their existing family drivers.
- Disk consumers restore reset and reached checkpoints (0, 7, 24) into different
  destination histories, verify eight-step continuations, and exercise T=1/T=8
  capture. Expected terminal lane endpoints and invalid task slots come from the
  source corpus, with no extra source physics in the replay test.

The first exploratory capture test incorrectly assumed that repeated scalar calls
would stop after done. The historical corpus deliberately includes post-terminal
scalar calls, while a public chunk freezes each done lane. The test was corrected
using source done flags to derive each lane's endpoint, validity and zero padding;
no implementation behavior or comparison tolerance was changed. Native rendering
also retains the source's fixed resolution; the final test uses those defaults.

## Local validation

- Pinned-source profile: **56/56 required cases pass**, plus **15 corpus checks**.
  `native/` is independently sealed; the five CUDA cases are explicitly deselected
  locally and are required in the separate cluster gate.
- Focused CPU suite: **114 passed**, five CUDA cases deselected. Includes typed State,
  full-State contracts, shared semantic harness, inventory/repository checks,
  corruption controls and preserved old/new verifier comparisons.
- Sixteen corpus controls reject missing/changed artifacts, schema/dtype changes,
  duplicate NPZ members, missing source identity, wrong registrations, overlapping
  State paths, object/nonfinite arrays and invalid done flags; snapshot ownership
  and overwrite refusal are exercised too.
- Ruff and diff whitespace checks pass. Strict documentation build retains the
  same four pre-existing broken links; no new warnings.
- Original Rocket 76-artifact closure, initial cleanup 81-artifact seal, previous
  CUDA 16-artifact seal, and the new 17-artifact corpus seal all verify unchanged.
  The clean candidate profile's 200 file hashes match the committed files.

`final-native.xml`, `final-cpu.xml`, their logs, the inventory and the local verification records
retain the detailed counts, source identity and runtime. The two known legacy
reference-provenance failures documented in the prior cleanup are outside this
focused suite and remain unresolved; this report does not claim a green full suite.

## Cross-platform diagnosis

The first cluster run (7011634, RTX 3090) passed 11 cases and failed four at an
exact float64 comparison: Pendulum and Acrobot continuations on CPU and CUDA.
The first reported differences were 2.22e-16 and 1.11e-16. Failed evidence is
retained under `oscar/7011634`; it was not overwritten or relabeled.

Independent source-only audits on Linux repeated the same source commit and
32-step action plans as the Mac corpus. Maximum source-versus-source differences
were 7.994e-15 (Pendulum reward), 4.885e-15 (Pendulum velocity), and 1.11e-16
(Acrobot velocity). CartPole also showed last-bit drift over the full trajectory;
both MountainCar variants were exact. All float32 outputs, RNG, counters and flags
were exact. Raw Linux source trajectories, per-leaf diagnostics, source hashes and
runtime math identity are retained in the diagnostic jobs.

The frozen regression policy now permits absolute error `1e-12` only for float64
continuation/task values, with relative tolerance zero. Immediate restoration,
float32/discrete fields, original-versus-migrated corpus bytes, and live source
contracts retain exact comparisons. Four negative controls require larger float64
errors, a one-ULP float32 change, an integer RNG change and a changed flag to fail.
No registration acceptance or pixel tolerance changed.

The first diagnostic's live profile was BLOCKED because the root test configuration
imported an installed source package before the explicitly pinned checkout could
load. It correctly refused that mismatch. Diagnostic 7011805 loaded the verified
source first and passed **56/56 exact live checks**. The final implementation defers
the legacy optional import when a source profile is requested; an injected wrong
installed package must remain unexecuted in its regression test. Both diagnostic
runs are retained, including the first blocked result.

## Final Oscar gate

Job **7011859** completed with exit **0:0** on **gpu1505, NVIDIA A40**, using
`gpu` / `norm-gpu`. All **76 required tests passed**, zero skipped/failed/crashed,
in one supervisor segment. The run verified all **1,633 archived file hashes**
against candidate `d2d10f04047997ba15efd316d63dd5244b80036a` before testing.

| Required checks | Passed |
|---|---:|
| Exact same-runtime live-source contracts | 56 |
| Full 32-step historical corpus replay | 5 |
| Byte-identical old/new corpus controls | 5 |
| CPU disk restoration, continuation and frame capture | 5 |
| CUDA disk restoration, continuation and frame capture | 5 |
| Total | 76 |

The terminal log, exact required/passed node lists, JUnit, supervisor events,
hardware/runtime identity and submission scripts are under `oscar/7011859`.
`verification-final.json` independently checks their counts, source/candidate
hashes and prior evidence seals locally. Both condo duplicates (7011622 and
7011860) were cancelled while pending at the group GPU quota. The final pass is
an **A40 correctness result**, not an RTX 3090 performance measurement.

## Scope and remaining work

This format currently stores State/task trajectories, not ground-truth pixels.
Frame checks compare chunk capture to direct candidate rendering at the independently
source-derived States. They do **not** establish new source pixel MSE or speedups.
Existing registration-specific acceptance gates, thresholds and evidence remain
required. Source pixels/camera/fidelity metadata, other family migrations and
renderer benchmark consumers are future steps. No old drivers were retired.

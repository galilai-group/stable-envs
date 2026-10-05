# Shared source contracts for all 38 non-Craftax registrations

Candidate: `48a8d67a` on `sami-dev`; baseline: `e27edd56`.
This checkpoint expands the common regression profiles from 7 to 38 registrations
while reducing maintained executable Python. Production APIs and runtime code
are unchanged. Craftax remains excluded.

## Results

**38/38 registrations pass on both CPU and CUDA: 1,152/1,152 required cases,
with no skips or failures.** All seven final Slurm jobs completed with exit `0:0`.

| Profile | Registrations | CPU passed/required | CUDA passed/required | Job |
|---|---:|---:|---:|---|
| native | 5 | 56/56 | 56/56 | `7012544_0` |
| fetch | 12 | 192/192 | 192/192 | `7012544_1` |
| dmc | 12 | 192/192 | 192/192 | `7012544_2` |
| navigation | 3 | 48/48 | 48/48 | `7012544_3` |
| ogbench | 3 | 48/48 | 48/48 | `7012544_4` |
| pusht | 2 | 24/24 | 24/24 | `7012544_5` |
| rocket | 1 | 16/16 | 16/16 | `7012544_6` |

See the [per-registration CSV](registrations.csv), [machine-readable summary](summary.json)
and [terminal accounting](slurm-accounting.txt). Fetch, navigation and OGBench ran
on RTX 3090s; native, DMC, PushT and Rocket ran on A40s, all through `gpu` /
`norm-gpu`. The `3090-gcondo` duplicates remained queued and were canceled.

The [sealed evidence archive](evidence.tar.gz) (containing `oscar-evidence.tar.gz`) retains full sealed CPU/CUDA results,
per-case records, JUnit/logs, runtime identities, source hashes, failed/blocked or
superseded runs, and submission manifests. All fourteen final result seals were
verified after download; required node IDs match exactly. It omits extracted
code, dependency installations and caches; code is reconstructible from the
recorded commit and file manifest.

## What the shared tests establish

Every registration runs four contracts at seeds 17 and 42, batch 2, and its declared
action dtypes. Continuous actions cover float32 and float64; discrete actions use
int64. There are 576 required cases per device, 1,152 across CPU and CUDA.

- State: independent source comparison; owned snapshots; restoration into another
  history; atomic rejection of a missing State leaf; eight source-matched future steps.
- Simulation: reset and 24 reached transitions against independent source fields.
- Task: exact public reward, termination, truncation and the registration's other
  exposed task outputs against source. Available observation/info fields are included.
- Online State capture: T=1 and T=8; no, all, final, reverse-strided and empty State
  selections; source-matched endpoints, terminal-lane validity/padding and ownership.

CUDA runs explicitly check tensor placement. Physics remains on CPU for these
profiles. Frames are omitted; this is not a new renderer-parity, GPU-physics,
multiprocessing or throughput claim. Specialized contact, lifecycle, Variation,
rendering and process acceptance gates remain required and retain their old cases.
The accepted registration matrix and historical evidence are unchanged.

All live comparisons remain exact, including floating-point leaves. The existing
`1e-12` frozen float64 cross-machine policy is unchanged and is not used to loosen
these live gates. Missing dependencies, partial selections, duplicate results or
wrong source versions cannot produce a passing profile.

The source checkout is the clean SWM pin
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. Each result records package versions and
source hashes; candidate manifests verify the exported test/runtime files before
execution. Cube/Scene keep their accepted controlled-entropy compatibility scope.
PushT and Rocket use independent native-engine witnesses alongside public replay
State restoration; they are not relabeled native engine-snapshot APIs.

## Code removed and retained

Across maintained Python in `stable_envs`, `tests` and `scripts`, the candidate
adds 504 lines and removes 702: **198 fewer lines overall**. Evidence, documentation
and compressed historical source copies are excluded from that code count.

Five DMC oracle implementations (591 lines) become one (195 lines), with a small
source-derived map for domain-specific model fields. Capture no longer imports a
candidate schema. Existing callers, benchmark bridges and hash lists use the
shared helper. The archived old/new proof compares all twelve defaults at both
seeds through reset, stepping, different-history restoration, internal reset
boundaries and cached Gaussian RNG state: 24 exact comparisons of all State/task
tensors. See `migration/result.json` and `migration/identity.json` in the
[sealed evidence archive](evidence.tar.gz).

Cube/Scene's independent oracle assertion is extracted from its unit-test module
and reused by the shared bridge and online source gate. Forty-eight existing
source-oracle tests pass after extraction. The navigation source bridge now honors
inactive lanes in chunk tests, preserving its ordinary scalar-step behavior.

Registration data is centralized; new registrations reuse the four test functions.
Family-specific bridges remain small and independent. Specialized acceptance
scripts are retained until their stronger cases can be migrated without weakening
their obligations.

## Local checks and exploratory evidence

- Final cleanup/corpus controls: 76 passed; five CUDA cases explicitly deselected
  on the Mac (fresh cluster validation is recorded separately).
- Final local Fetch profile: 192/192 passed.
- Cube/Scene oracle-extraction checks: 48 passed, 38 unrelated cases deselected.
- Additional repository/import audit: 38 passed, 2 skipped, 5 failed. These are
  pre-existing inputs: two assertions reference the removed `dynamics.py`, one
  applies an obsolete blanket SWM-symbol ban, one cannot initialize Mac CGL, and
  one treats ignored bytecode-only directories as live shims. All 155 tracked
  runtime/test Python inputs to these checks match the baseline byte-for-byte;
  see `local/preexisting-import-contracts.json` in the evidence archive. This checkpoint
  does not claim a clean repository-wide suite.
- Strict documentation build retains the same four historical broken-link warnings
  (three script links outside the documentation root and one missing trace archive).
  No additional warning was introduced by this checkpoint.

The local exploratory archive retains blocked imports, unavailable Mac CGL,
the navigation terminal-mask test failure and corrected runs. Initial Oscar Fetch
was correctly BLOCKED because its robotics installation was absent from
`PYTHONPATH`; the final runner includes the pinned installation. Superseded runs
and canceled duplicates are retained, not counted as acceptance. The final runner
executes CPU and CUDA pytest processes independently within each four-core job;
these correctness runs are not performance measurements.

The archive includes its own `SHA256SUMS` for all raw report artifacts. Extract
it into a new directory and use `stable_envs.testing.evidence.verify_seal` there;
extract its `oscar-evidence.tar.gz` to inspect each independently sealed device run.
Generated submission scripts and repeated manifests live only in the archive.

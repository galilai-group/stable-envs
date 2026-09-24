# Cube/Scene CPU bookkeeping and restore performance

Starting clean branch `repair/environment-semantics` at
`72612563590f12a081d940394496795a6c024577` (CPU evidence and ending checkpoint).
CPU implementation is `14456339ec7f1b28ad698fc3158fe9e4ed8f6d1e`.
Dependency audit: `75ef812977423fa059b857b36542119e101a430f`;
RNG checkpoint: `434cc72974360ab98e47cad7aebcc8b1b62bb1de`;
documentation checkpoint: `03adc1ab79bae71e3e62db2e5c4cb19f20723166`.
Exact pre-change matrix/map entries and frozen renderer hash are in
[start.json](semantic_recovery_evidence/ogb_renderer_closeout/start.json).
No push occurred.

The final CPU closure rerun passed **86/86** cases under MuJoCo 3.8.1 before
renderer changes. The implementation map and only the two relevant matrix rows
now acknowledge completed production CPU implementation and passed CPU gates,
while both registrations remain **INCOMPLETE**. No incompatible status enum was
added. Renderer/GPU acceptance is still pending; matrix **33 VERIFIED / 9 INCOMPLETE**.
Prior source-preflight evidence is preserved as historical evidence, not current status.

## Restore characterization

Local Apple M5, Python 3.12.11, MuJoCo 3.8.1, OGBench 1.2.1, NumPy 2.2.6,
Gymnasium 1.2.3, dm-control 1.0.41, Torch 2.10.0, TensorDict 0.14.2.
Wall-clock timings include public batched State creation. Five first calls on
fresh adapters measure cold-instance latency after imports; they are not claimed
as fresh-process startup measurements. Thirty warm calls per operation measure
median, p95, maximum and throughput. B=1 and B=3 are separate runs.

The first exploratory run overlapped CPU correctness tests and is retained with
`-concurrent` filenames. The table uses the subsequent run without those suites.
Full cold/warm distributions and construction/compilation/validation/copy costs:
[performance.json](semantic_recovery_evidence/ogb_renderer_closeout/performance.json).

| Family | B | Step median ms | Get median ms | Same restore median / p95 ms | Changed restore median / p95 ms | Same restores/s (batches) | Same/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cube | 1 | 1.185 | 0.721 | 181.389 / 233.281 | 180.849 / 231.985 | 5.513 | 153.1× |
| Cube | 3 | 3.074 | 1.765 | 591.219 / 615.094 | 593.307 / 617.705 | 1.691 | 192.4× |
| Scene | 1 | 1.401 | 0.705 | 211.752 / 262.988 | 212.261 / 265.135 | 4.722 | 151.2× |
| Scene | 3 | 3.753 | 1.771 | 705.623 / 783.148 | 702.140 / 759.757 | 1.417 | 188.0× |

Both restore paths construct and compile fresh models by design. The same-model
path is about 150–193 times normal step and costs nearly the same as changing
physical size. Staged timings isolate validation, constructor, compilation and
State-row copying; the remaining bucket includes MJCF/model assembly, forwarding,
field assignment and the atomic swap. These are local characterization numbers,
not Oscar or benchmark-certification claims.

Proposed later optimization, **not implemented**: after validating the whole batch,
compare compiled model fingerprints and persistent configuration. For exact matches,
restore into preallocated private data buffers against the existing model; preserve
all cached fields and RNG-last ordering, compare the reconstructed full State,
then atomically swap buffers. Retain temporary reconstruction for changed models.
Any implementation must re-run malformed-State and injected late-failure rollback,
contact/warmstart/controller replay, RNG and independent-instance gates. A direct
mutation shortcut without rollback would violate the current atomic contract.

## Commands

From `/private/tmp/stable-envs-semantic-recovery`:

```bash
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. \
 /private/tmp/ogb-manip-authority-swm-lock/bin/python -c 'import sys; sys.path.append("/private/tmp/stable-envs-semantic-recovery/.venv/lib/python3.12/site-packages"); import pytest; sys.exit(pytest.main(["-q", "-rA", "tests/unit/test_ogbench_manipulation_cpu.py"]))'
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=. \
 /private/tmp/ogb-manip-authority-swm-lock/bin/python \
 docs/research/semantic_recovery_evidence/ogb_renderer_closeout/performance.py
```

Results: `86 passed` and `RESTORE_CHARACTERIZATION_PASS`. Exact full-State
restoration was checked during the physical-configuration timings. No production
simulator algorithm, restore optimization, shared contract, frozen renderer,
accepted registration or prior artifact was changed by this bookkeeping step.

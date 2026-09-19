# Step 9 — Workstream O (archaeology / dead-code classification)

**Branch** `step9/cleanup` · **parent** `step9/integration` @ `786993b` ·
**Step-8 baseline** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.

**Result: INVENTORY.** This wave classifies the leftover research and compatibility surface. It does
**not** delete the Step-8 behavioral oracle, frozen corpora, or historical scratch. A v0 draft keeps
the shims that still have live callers; removal is a finite post-v0 ticket after those callers move.

## Ruling for v0

Wave 3 of the plan says to audit `scratch/`, `examples/results/`, shims, and deprecated aliases.
The binding constraint is: **do not destroy the Step-8 behavioral oracle.** For the first public
cut that means:

- research evidence stays in-tree and is classified `ARCHIVE_RESEARCH`;
- compatibility shims stay importable and unadvertised;
- only user-facing falsehoods (the README `[all]` extra) are corrected now;
- live test imports are retargeted to canonical modules only after Workstream K lands.

Removing `StableEnv` / `core` / `compat` / `swm_nav` tonight would break hundreds of historical
tests and scratch probes without changing product behavior. That is not OSS quality; it is a
breakage.

## Classification

| item | class | why |
|---|---|---|
| `examples/results/step{4,5,6,7,8}/` | ARCHIVE_RESEARCH | frozen oracles, Gate A, support matrix |
| `examples/results/step9/oracle/` | KEEP_INTERNAL | immutable Step-8 snapshot for Step-9 diffs |
| `docs/research/` | KEEP_PUBLIC | family studies and closure packets |
| `scratch/step{3,4,5,6,7,8}/` | ARCHIVE_RESEARCH | certifiers, probes, sbatch history |
| `scratch/step9/` | KEEP_INTERNAL | per-workstream CPU/GPU oracles |
| `scratch/certification.json` | KEEP_INTERNAL | machine-readable certification spec |
| `stable_envs/testing/` | KEEP_PUBLIC | `numpy.testing`-style measurement library |
| `stable_envs/rollout.py` | KEEP_PUBLIC | `snv.sample` / policy helpers; not `env.rollout` |
| `stable_envs/data.py` | KEEP_PUBLIC | state datasets; lazy top-level |
| `stable_envs/spaces.py` | KEEP_INTERNAL | SWM-shaped variation spaces; used by interop |
| `stable_envs/core/` | KEEP_INTERNAL | Step-8 import path; same objects as `env`/`registry` |
| `stable_envs/compat/` | KEEP_INTERNAL | `sys.modules` aliases onto `_interop` |
| `stable_envs/envs/swm_nav.py` | KEEP_INTERNAL | alias of `navigation` |
| `stable_envs/envs/simple_point_maze.py` | KEEP_INTERNAL | alias of `navigation` |
| `stable_envs/envs/ogb_manip.py` | KEEP_INTERNAL | alias of `ogbench.manipulation`; still the family name |
| `stable_envs/envs/craftax_full.py` | KEEP_INTERNAL | alias of `_craftax_full` (21 names differ; no merge) |
| `snv.StableEnv` / `REGISTRY` / `register` / `batch=` | KEEP_INTERNAL | warn once; same objects |
| `_LEGACY_MODULES` (`snv.spaces`, `snv.render`, …) | KEEP_INTERNAL | resolve, not advertised in `dir(snv)` |
| README `[all]` extra | DELETE_SUPERSEDED | extra does not exist; corrected in this commit |

## Files changed in this commit

- `README.md` — stop advertising a nonexistent `[all]` extra.
- `CHANGELOG.md` — v0.1.0 draft notes.
- this file.

## Coordinator requests (after K and the GPU gate)

1. Live tests import canonical modules (`env` / `interop` / `_interop`). Shims stay for scratch
   and for the alias-identity tests.
2. When `ogb_manip` goes, move `manipulation._MODULE` and `scratch/step4/universe.py::shard_of`
   together (Workstream D R2).
3. Do not delete `examples/results/` or `scratch/` for v0.
4. Consumer `close()` / `BoundedCache` wiring on MJWarp/nvdiff family caches is done
   (`release_cached_ctx` / `close_cached_renderer` / rocket `owns_scene`). Remaining CPU
   clear-on-full caches are the same LRU.

## §17 ten items

1. **Files.** README, CHANGELOG, this findings file. Nothing under `stable_envs/` product code.
2. **API.** Documentation only: `[all]` is not a real extra.
3. **Step-8 IDs.** None.
4. **Tests.** None in this commit; the inventory is the product.
5. **Certification.** Untouched.
6. **Frames.** Untouched.
7. **Variation/state.** Untouched.
8. **Performance.** Untouched.
9. **Resources.** Untouched.
10. **Unexplained oracle differences.** Zero; no oracle was run.

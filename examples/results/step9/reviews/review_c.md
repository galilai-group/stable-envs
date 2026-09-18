APPROVE

# Step 9 — independent review of Workstream C (Fetch)

**Reviewed** `step9/envs-fetch` at **86df7ee95380e824e6ac44dde7af8f332ee41384** — the author's delivery
`0ef8a97` plus the coordinator's merge of `step9/integration` (`312a631`) — diffed against the Step-8
behavioural baseline `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.
**Reviewer host** Apple M5 Mac (`cis1b1zs2j3`), CPU only, no CUDA device on this ticket, `MUJOCO_GL=glfw`,
`-p no:cacheprovider`. **Interpreters used**: `/Users/samib/se6-venv` (the unit suite), `/Users/samib/se8-fetch-venv`
(gymnasium-robotics 1.4.2, mujoco 3.12.0, numpy 2.5.3, tensordict 0.14.2 — confirmed present myself),
`/Users/samib/se8-vm-venv`. Nothing was edited, committed, stashed, reset, rebased or pushed on the
branch; no Oscar/ssh contact; no GPU job. The oracle harness created and removed its own detached
`git worktree` of `21cb306` under `$TMPDIR`; it cleaned up.

## Verdict

**APPROVE.** I re-ran the CPU oracle myself from scratch at 86df7ee (not just read the committed
JSON) and it still reports zero unexplained differences over all 41 keys and 12 cells against a
freshly built `21cb306` worktree. I independently re-verified every quantitative claim in
`findings.md` that could be checked without CUDA — the taxonomy totals, the registry/SWM alias
resolution, the two coordinator-visible test-suite deltas, the `_CAM_CACHE` capacity discrepancy the
author self-disclosed (E4), Ruling 2 (no `state`/`_layout`/`_hidden_state`/`_live_state` override),
Ruling 3 (`unpack` copies — I mutated a live unpacked `State` in place and confirmed the env's own
buffer did not move), and Workstream I's four re-exports — and none of it needed to be taken on
trust. REGRESSION count: **0**. One process gap (no committed GPU sbatch section for this family) is
real and is detailed below; it does not block this merge under Ruling 8, but the coordinator has to
close it before Wave-1's GPU oracle and closure §20 can run. Two minor findings, neither blocking.

## CPU oracle — re-run by me, from scratch, into a temp dir

```
cd /Users/samib/stable-envs-step9-envs-fetch
MUJOCO_GL=glfw /Users/samib/se8-fetch-venv/bin/python scratch/step9/envs-fetch/oracle.py \
    --out <tmp scratchpad dir> --name oracle_review.json
```
The script builds its own fresh detached `git worktree` of `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
and re-runs the probe against it, side by side with a fresh run against my checkout at 86df7ee — it
does **not** diff against the committed `oracle.json` (which the author correctly notes was produced
before the integration merge, at head `c40be4d`, before B and G landed). My run:

```
"baseline_sha": "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
"head_sha":     "86df7ee95380e824e6ac44dde7af8f332ee41384"
"identical": true
"unexplained_differences": {}
"taxonomy": {"mapped": 33, "partial": 12, "unsupported": 12, "dead_upstream": 15, "not_applicable": 51}
"taxonomy_ok": true, "swm_aliases_ok": true, "functional_ok": true
ORACLE: PASS (frames owed to the GPU oracle)
```
All twelve cells (`FetchReach/Push/Slide/PickAndPlace` × `""/Dense/Dict`) printed a plate digest and
a rows digest; none of them moved relative to what the author's own printed table shows for the same
cells (`FetchReach`/`FetchPush`/`FetchSlide`/`FetchPickAndPlace` groups share a plate digest per
scene as expected, since the plate does not depend on the reward/obs-container suffix).

**This is the single most important number in this review: at 86df7ee — after B (DMControl as a
package) and G (2-D render-backend consolidation: gfxdraw, raster2d, scene, navdot, tilemap, prng)
both landed on integration — nothing moved.** That is also structurally guaranteed and not a
coincidence: `stable_envs/envs/fetch.py` imports only `..env`, `..registry`, `..render.mjwarp` and
`..spaces` (verified by grepping every `^import`/`^from` line in the file); it touches none of G's
five 2-D backend modules and none of B's DMControl package. `git diff --stat 21cb306..86df7ee` shows
`stable_envs/render/mjwarp.py` with **zero** diff lines anywhere in the whole 169-file, 88k-line
delta — the module Fetch actually renders through is untouched by every other Wave-1 workstream so
far. The CPU-checkable 41/41 keys were never at risk of drifting, and my fresh run confirms they did
not.

I counted `COMPARED` in `scratch/step9/envs-fetch/oracle.py` directly rather than trusting the prose:
`awk '/^COMPARED = \{/,/^\}/' … | grep -c '^\s*"'` → **41**, matching the claim exactly.

## Contract conformance — verified live, not read off the source

I do not have a way to construct the *official* `gymnasium.make("Fetch*-v4", ...)` reference env in
`se8-fetch-venv` — that path is genuinely blocked by the numpy 2.5.3 + mujoco 3.12 `set_joint_qpos`
defect (Step 3), and I did not try to work around that. I **did** construct real `stable_envs.envs.fetch`
classes directly, the same way the author's own `scratch/step9/envs-fetch/oracle.py` FUNCTIONAL block
and `tests/unit/test_envs_fetch.py` tier 2 do (stub `warp`/`mujoco_warp` in `sys.modules`, substitute
`F._state_pool` with a loader over the frozen Step-4 corpus, build with `device="cpu"`). This is a
real build of `FetchReach`/`FetchPush`/`FetchSlide`, not a mock:

```python
env = snv.FetchReach(num_envs=4, device="cpu")
env.set_state(corpus)                      # frozen Step-4 corpus, 4 states
rows_before = env.get_state().clone()
st = env.unpack(env.get_state())
st["qpos"][:] = -999.0                     # mutate the unpacked State in place
rows_after = env.get_state()
```
Result: `env buffer unchanged after mutating unpacked State's qpos in place: True`;
`state.qpos.data_ptr() == env.get_state().data_ptr(): False`. **Ruling 3 holds for this family**:
`unpack` returns a real copy, not a view aliasing the env's packed buffer — confirmed by mutating live
data, not by reading `Env.unpack`'s `copy=True` comment (which I also read, at `stable_envs/env.py:939`,
and which is inherited unmodified — Fetch overrides neither `pack` nor `unpack`).

`registry.make/spec/list_envs` and the SWM aliases, also built live:
```
list_envs() fetch-family entries: 12 names, FetchReach..FetchPickAndPlaceDict
spec('FetchReach')['swm_ids'] == ('swm/FetchReach-v3',)
make('FetchPush', num_envs=2, device='cpu') -> FetchPush, num_envs=2
SWM_IDS entries containing 'FetchPush': {'swm/FetchPush-v3': ('FetchPush',)}
with snv.FetchSlide(num_envs=1, device='cpu') as e2: ...   # opens, then e2.closed == True on exit
```
I then re-ran this for all twelve cells (not just the two spot-checked above): every one of the
twelve resolves to exactly one `swm/Fetch*-v3` id, `spec()["family"] == "fetch"` and
`spec()["module"] == "stable_envs.envs.fetch"` for all twelve — confirming `__module__` pinning in
`_make_classes` (§3 of findings.md) actually does what it claims and the family is not split.

`num_envs=`/`batch=`, `step`/`transition` refusal, `close()` idempotence and closed-env-refuses-render
were already exercised live by the author's own FUNCTIONAL probe inside my from-scratch oracle
re-run (`functional_ok: true` above covers all twelve cells, not a sample); I read that probe's source
(`scratch/step9/envs-fetch/oracle.py:228-286`) to confirm it actually asserts what it claims —
`torch.equal` on `pack`/`unpack` round trips, `NotImplementedError` catches for `step`/`transition`,
`env.closed` after a double `close()`, and a `RuntimeError` catch on `render()` post-close — rather
than assuming a green flag means what the docstring says it means.

**Ruling 2** — `FetchBase` overrides none of `state`/`_layout`/`_layout_dtypes`/`_hidden_state`/
`_live_state`. Verified two ways: (1) `grep -n "def state\b\|_hidden_state\|_live_state\|def pack\|def unpack" stable_envs/envs/fetch.py`
returns nothing for any of those names as a definition in this file; (2) the shipped test
`test_state_layout_is_assigned_per_instance_and_nothing_else_is_overridden` does the same check via
`ast` over the class body, which I read and confirmed asserts the right thing rather than a tautology.
The only state machinery in this family is `self.state_layout = (...)` assigned once in `__init__`,
which is exactly the hook Ruling 2 sanctions.

## Workstream I's requirement — re-exports and the fetch.py:453 line pointer

`MW.HIDDEN_GROUP`, `MW.DEFAULT_GROUPS` (imported here as `VISIBLE_GROUPS`), `MW.gl_rounding` and
`MW.create_render_context` are all present, unrenamed, at their documented call sites
(`stable_envs/render/mjwarp.py:97,99,563,649`) and `render/mjwarp.py` has zero diff lines in the
whole Wave-1-so-far delta — C did not touch it, and nothing else that has merged touched it either.
**Confirmed re-exported and unbroken.**

**`fetch.py:453` is now stale, and this is worth recording precisely.** At the Step-8 baseline
(`21cb306`), line 453 of `stable_envs/envs/fetch.py` is:
```python
if len(_CTX_CACHE) >= _CTX_CACHE_MAX:
    _CTX_CACHE.clear()
```
— exactly the line Workstream I's consumer change list cites (`examples/results/step9/render-mjwarp-nvdiff/findings.md:106`:
`envs/fetch.py | 453 | for _c in _CTX_CACHE.values(): MW.release_render_context(_c.rc)`). At 86df7ee,
line 453 is `m.vis.global_.offheight = max(int(m.vis.global_.offheight), 4 * FetchBase.default_size)`
— an unrelated line inside `_build_bundle`, because C's `_Cache` refactor **removed** the per-call
`if len(...) >= MAX: clear()` idiom from `_context()` entirely and folded it into
`_Cache.__setitem__` (`stable_envs/envs/fetch.py:205-207`), which is now shared by all six caches, not
just `_CTX_CACHE`. **A mechanical `sed`/line-number patch at fetch.py:453 would now land in the wrong
place and edit unrelated model-compilation code.** The correct site for I's Wave-2 fix is
`_Cache.__setitem__`'s `self.clear()` call — but that call is generic across all six caches
(`_CTX_CACHE`, `_PLATE_CACHE`, `_MODEL_CACHE`, `_RENDERER_CACHE`, `_POOL_CACHE`, `_CAM_CACHE`), and
only `_CTX_CACHE` entries hold something that needs `MW.release_render_context` — so I's fix can no
longer be a one-line insertion at a fixed line number; it needs either a per-cache `on_evict` hook
(which the handoff's own "rulings owed" section says is being considered for `BoundedCache` anyway)
or a `_CTX_CACHE`-specific override. **Not a defect in C's patch** — the refactor is exactly what the
brief asked for and the coordinator's own `_CTX_CACHE`/`_PLATE_CACHE` bound semantics are preserved
byte for byte (see next paragraph) — but the consumer change list Sami and I will hand to Workstream I
in Wave 2 needs updating, or it will silently patch the wrong nine lines.

**The one real behavioural micro-difference the cache refactor makes, and the author disclosed it
correctly (E4).** I read the diff hunk by hunk rather than trusting the summary. Old `_CTX_CACHE`
(`if len >= MAX: clear()` before insert) and old `_PLATE_CACHE` (same idiom) are byte-for-byte
equivalent to the new `_Cache.__setitem__`'s `len(self) >= self.maxsize` check. Old `_CAM_CACHE` used
`if len(_CAM_CACHE) > _CAM_CACHE_MAX: clear()` — **strict `>`**, checked *before* the insert that
grows it to 257 — so it topped out one entry higher than its stated cap of 256 before the *next* call
cleared it. The new unified cache uses `>=`, so `_CAM_CACHE` now tops out at exactly 256. This is the
only place a cache's observable capacity actually changed, it is a capacity difference only (the
answer for a given key is a pure function of the model and camera parameters — a cache miss just
recomputes it), and the author flagged it themselves as E4 rather than leaving it for a reviewer to
find. I confirm E4 is accurate and complete.

## The ten §17 items, my own numbers

1. **Files moved/removed/created** — confirmed via `git diff --stat b3f3a81..0ef8a97`: exactly
   `stable_envs/envs/fetch.py` (201 insertions / 68 deletions, then a 16/12 follow-up commit),
   `tests/unit/test_envs_fetch.py` (new, 306 lines, 17 tests), `scratch/step9/envs-fetch/oracle.py`
   (new, 474 lines), `examples/results/step9/envs-fetch/{findings.md,oracle.json}` (new). Zero files
   moved, zero removed. `registry.py`, `env.py`, `dynamics.py`, `specs.py`, `_config.py`,
   `__init__.py`, `pyproject.toml` — every coordinator-owned file — has **zero** diff lines in
   `b3f3a81..0ef8a97` (checked directly, not inferred): the family-ownership rule was followed.
2. **API changes** — additive only, verified live above (`num_envs=`, `batch=` still accepted,
   `{qpos, goal}` State, `close()`/context manager). No Step-8 call site breaks.
3. **Step-8 IDs affected** — all twelve `swm/Fetch*-v3` ids; behaviourally none, confirmed by my own
   from-scratch CPU oracle re-run (41/41 keys, `identical: true`) and my own live registry/SWM-alias
   check across all twelve cells.
4. **Targeted tests** — 17 new tests in `tests/unit/test_envs_fetch.py`, all 17 pass on their own in
   `se8-fetch-venv` (I ran them: `17 passed in 2.08s`, 0 EGL contexts — this family opens none on
   CPU). Full unit suite on this branch (my own run, `se6-venv`): **1543 passed, 205 skipped, 112
   deselected in 114.55s**. I isolated C's own contribution by running the same command on
   `step9/integration` at `312a631` (the commit C merged, i.e. everything except C):
   **1533 passed, 198 skipped, 112 deselected**. Delta from C: **+10 passed, +7 skipped** — matching
   the author's tier-1/tier-2 split exactly (10 AST-based tests run everywhere; 7 more skip in
   `se6-venv` for lack of `gymnasium_robotics` and would run in `se8-fetch-venv`). This is my own
   number, produced by running the suite twice on two different worktrees, not a re-statement of the
   author's `1385 → 1395` (which was measured at a different, pre-integration baseline and is
   internally consistent with mine).
5. **Certification checks** — the frozen Step-4 corpus replay covers everything a CPU can reach (§4
   above); Gate A / the GPU conformance suite / tier measurements genuinely need CUDA and there is
   **no existing `tests/gpu` file that even mentions this Fetch family** (`grep -rln Fetch tests/gpu/*.py`
   → only `test_reproducibility.py`'s `MiniGridPOVFetch`, an unrelated cell) — owed, and correctly
   scoped as fully owed rather than partially covered.
6. **Default-frame comparison** — OWED, confirmed structurally: `_context()`/`_frame()` (the mover
   ray-cast) is gated by a module-scope `ImportError` on `warp`/`mujoco_warp` (lines 133-143,
   byte-identical to Step 8 except the `Env`/`StableEnv` import swap — I diffed lines 1-150 against
   `21cb306` directly). Real `warp`/`mujoco_warp` are not importable on this Mac, so `render()` cannot
   execute in either arm; the plate half (`_bake_plate`, MuJoCo's own raster, 91% of the frame by the
   author's own measurement) is the only part of `render()` a CPU can produce, and it is compared
   (`plate_default_sha256`, `plate_varied_sha256`, and the 64-way `style_group_plate_sha256`) — I
   re-ran this myself and it is identical.
7. **Variation/state comparison** — identical (my own run): style layout, the default appearance
   value by value, taxonomy per leaf and as totals (33/12/12/15/51), `state_dim`/`state_names`, the
   corpus round trip. `background.color` is absent from `make_variation_space()` (grepped directly:
   no `background` key anywhere in the space dict) — Ruling 9 holds, nothing was re-frozen.
8. **Performance for hot paths** — not measurable on CPU (every hot path is CUDA-only); I agree with
   the author that measuring it here would be theatre, not evidence. Genuinely owed to the GPU arm —
   see the new section below, since there is currently no instrument to collect it.
9. **Resource lifetime** — `_close()` drops only `_idx_cache`; grepped the method body via `ast` (same
   check the shipped test does) and confirmed `_CTX_CACHE`/`_RENDERER_CACHE`/`_MODEL_CACHE`/
   `_PLATE_CACHE` never appear in it. My own unit-suite run's resource guard: **20 EGL contexts
   created, 0 released, 20 live** — unchanged in kind from what G's reviewer measured (16, from
   DMControl only) — the fetch tests open zero.
10. **Unexplained oracle differences** — **zero**, all 41 keys, all 12 cells, on my own from-scratch
    re-run at 86df7ee.

## GPU evidence owed and the missing sbatch

C's findings correctly identify the frame diff, the tier measurements and Gate A as owed to the
post-Wave-1 GPU oracle (Ruling 8: no per-workstream GPU jobs). But unlike A, B, G, H and I, **C did
not commit a `scratch/step9/envs-fetch/*.sbatch`** — `find scratch/step9 -iname '*sbatch*'` on this
worktree returns only `render-3d/{oracle_selfcompare,hmg_determinism,gpu_oracle}.sbatch`,
`envs-classic-nav/gpu.sbatch` and `render-2d/gpu_oracle.sbatch`, `envs-dmcontrol/gpu.sbatch` — nothing
under `envs-fetch/`. The debt findings.md books is real, but there is currently no instrument on this
branch to collect it, which the coordinator has confirmed and asked me to scope rather than write.

**1. What produces no frame on CPU, by name, and why.** All twelve cells' `render()` output, in full —
not a subset. `_build_bundle` compiles the model on any device (CPU-reachable and already exercised by
the oracle), and `_bake_plate` (the static floor/table/skybox raster, 91% of the frame) executes on
CPU through plain `mujoco.Renderer` and is already covered by my CPU re-run. What cannot run at all
without CUDA is `_context()`/`_frame()` — the MJWarp batch ray-cast that draws the 18 robot meshes and
the four scenes' movers — because the module refuses to import at all (`ImportError` at
`stable_envs/envs/fetch.py:133-143`) unless `warp` and `mujoco_warp` are both importable and
`mujoco.__version__ >= 3.11`; those two packages are not present in any local venv (`se8-fetch-venv`
included — confirmed: `find_spec("warp")`/`find_spec("mujoco_warp")` both `None` there, which is why
the CPU oracle has to stub them in `sys.modules` to import the module at all rather than actually
running the ray-cast). So the cells with no CPU frame are: `FetchReach`, `FetchPush`, `FetchSlide`,
`FetchPickAndPlace`, and their `Dense`/`Dict` variants — all twelve — and the specific code paths owed
are `_context`, `_frame`, `_camera_batch`'s composite consumer, the `MJW_CHUNK`-chunked upload/
supersample loop, and the final `torch.where(mv.unsqueeze(-1), rgb, pl)` composite. The plate and the
grouping arithmetic that feeds that composite are not owed; they are already proven identical on CPU.

**2. Which venv arm is required, and this is a real, currently-unconfirmed gap.** Building *any* Fetch
cell — not just rendering one — unconditionally imports `gymnasium_robotics` at model-compile time:
`_asset_dir()` (`stable_envs/envs/fetch.py:232-234`) does `import gymnasium_robotics as gr; return
os.path.join(os.path.dirname(gr.__file__), "envs", "assets", "fetch")` to locate the MJCF the model is
compiled from, and `_build_bundle` calls it unconditionally in `__init__` — this is not confined to the
`reference_env()`/`reference_render()` path the findings correctly flag as separately owed. So the GPU
venv needs **all** of: `gymnasium_robotics` (to locate the asset XML), `mujoco >= 3.11`, `warp`,
`mujoco_warp`, CUDA-capable `torch`, and `tensordict`/`loguru` (core `stable_envs` deps since Wave 0).
Per the coordinator's own handoff inventory (§3 of `docs/research/step9_handoff.md`), the closest
Oscar venv is `se-s3b-fetch311` ("gymnasium-robotics + mujoco 3.11: the only interpreter where Fetch
builds") — but the handoff does not say it has `warp`/`mujoco_warp` installed, and `se7-ogb`
("ogbench + mjwarp") is not recorded as having `gymnasium_robotics`. **I cannot confirm from this Mac,
without ssh, that any existing Oscar venv satisfies the full combination** — this needs a check (and
possibly a `uv pip install` of `warp mujoco_warp` into `se-s3b-fetch311`, mirroring how the
`envs-dmcontrol/gpu.sbatch` job installs `tensordict`/`loguru` into `se-vis311` on the fly if missing)
before the sbatch can run at all. I flag this rather than guess it away.

**3. What a correct section must assert, modelled on `envs-dmcontrol/gpu.sbatch` and
`render-3d/gpu_oracle.sbatch`** (both run baseline-vs-branch in the same job on the same card, which
is the only valid form here — a cross-card digest comparison is not a valid oracle per the handoff's
own §3a note):
   - Provenance block first: host, job id, `git rev-parse HEAD` of the branch tree, confirmation the
     baseline SHA `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` is present in the tree's history, GPU
     name/driver, and `mujoco`/`mujoco_warp`/`warp`/`torch`/`gymnasium_robotics` versions — this family
     is unusually version-sensitive (`parity_note`'s numbers were measured at a specific
     mujoco/mujoco_warp pair, the same way B's certification semantics are pinned to mujoco 3.11.0 +
     mujoco-warp 3.11.0).
   - **The oracle itself, extended to compare `render()`, not just the plate.** `oracle.py --python
     <cuda python>` needs no code change per R1 — it already builds every env and the frame comparison
     is a stated TODO in `COMPARED`. Both arms (a detached `21cb306` worktree and the branch) run in
     the same job; compare the full `(H, W, 3)` uint8 frame digest per cell, per one of the 64 frozen
     corpus states minimum (all 64 if the job's time budget allows — this job is estimated 2-30 min per
     the handoff's own sizing note), at the default appearance. This is the first time the actual
     ray-cast + composite pipeline — the only GPU-only code this patch touches through `_style_groups`,
     `_restore`, `_cached`, `_Cache` — will run at all since the migration.
   - **At least one non-default style draw**, not just the default appearance: the grouping refactor
     (`_style_groups`) is the one piece of `render()` this patch actually rewrote, and the CPU oracle
     already proved the arithmetic identical for 64 distinct appearances cut into 64 groups — the GPU
     arm needs to run that same batch through the real composite once to confirm the *pixels* the
     regrouped plates/cameras feed into `torch.where(mv, rgb, pl)` still land correctly, not just that
     the grouping indices match.
   - **A live parity check against the reference**, if the venv's mujoco is ≤ 3.11 (avoiding the local
     `set_joint_qpos` defect): `reference_env()`/`reference_render()` for at least one scene, scored
     against `parity_interior_min = 0.99`, to settle whether the numbers in `parity_note`
     (0.998814/0.998800/0.998782, "re-measured 2026-08-11") still hold on this commit — they have not
     been re-verified by anyone since Step 8 as far as this delta shows.
   - **Enough distinct camera appearances in one job to cross the `_CAM_CACHE` 256-entry boundary**
     (E4 above) on both arms, confirming a cache clear mid-batch does not change a frame — the author's
     claim that "the answer is a pure function of the model and camera parameters" is architecturally
     sound but has never been checked with a real render.
   - **Paired GPU timings**, alternating baseline/branch on the same card in the same job (the pattern
     both model sbatches use), for at least 2-3 cells spanning both `nq` widths (e.g. `FetchReach`
     [15+3] and `FetchPush` or `FetchPickAndPlace` [22+3]) — this is the only source of real numbers
     for §17 item 8, which is currently and correctly unmeasured.
   - **Optionally, and only if time permits**: run the oracle with `--device cpu` on this GPU node,
     where `warp`/`mujoco_warp` are actually importable for the first time anywhere in this project.
     `docs/research/swm_matrix.md`'s own `PROBE_LIMITATIONS["fetch_cpu"]` entry records that the one
     prior CPU attempt (Step 1b, an Oscar RTX 3090 node) failed on the *reference* env's
     `set_joint_qpos` before the renderer was ever reached, so `CPU_BACKEND_MISSING` for the ray-cast
     itself is inherited as a Step 3 finding and has never actually been exercised end-to-end. This is
     not owed by C — Step 8 already carries this status forward unchanged — but the GPU job is the
     first opportunity to actually settle it rather than continue to carry it as `CPU_BACKEND_MISSING`
     on an untested path.

**Does the missing sbatch block this merge? No.** Ruling 8 explicitly removed per-workstream GPU jobs
and centralised evidence collection into one post-Wave-1 job the coordinator assembles; C's CPU oracle
(41/41 keys, re-verified by me from scratch) is exactly the evidence Ruling 8 asks a Wave-1 workstream
to produce locally, and the frame/timing/certification debt is honestly and specifically declared
rather than hidden. But it is real, uncollected work: unlike A/B/G/H/I, there is no draft section for
the coordinator to assemble into the combined job, and (per item 2) there may not even be a venv ready
to run it on yet. I'd treat this as a checklist item for `docs/research/step9_handoff.md` before the
"ONE GPU oracle over all 42 ids" job is assembled, not as a reason to withhold APPROVE from the code
that is in front of me.

## Defects

1. **`fetch.py:453` in Workstream I's consumer change list is stale post-refactor** (documentation/
   coordination only, not a code defect in C). Severity: low. Does not block — it blocks a future
   mechanical patch from landing correctly, not this merge. Detailed above; the coordinator should
   update the consumer change list before Wave 2 assigns I's fix.
2. **No `scratch/step9/envs-fetch/*.sbatch`**, unlike every other Wave-1 workstream reviewed so far.
   Severity: low for this merge (Ruling 8 does not require it per workstream), but it is a genuine gap
   in what the coordinator can currently assemble for the post-Wave-1 GPU job. Does not block. See the
   dedicated section above for what such a section needs to assert and the unresolved venv question.

Nothing above changes the verdict: **APPROVE**.

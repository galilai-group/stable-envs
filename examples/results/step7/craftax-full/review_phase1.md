# Step 7 W2 (craftax-full, target swm/CraftaxPixels-v1) — independent review, phase 1

Reviewed HEAD `59ab465027ad65b7f7dccbfccd9af528ec75633d` on `step7/craftax-full` (worktree
`/Users/samib/stable-envs-step7-craftax-full`), 9 commits on `step7/integration` `ada4075`. Code under
`stable_envs/` and `tests/` is byte-identical to `35127de` (`git diff --stat 35127de..59ab465 -- stable_envs tests` is
empty); the later three commits add result artifacts, scratch scripts, findings text and one docs row in
`docs/research/swm_coverage.md`. Nothing was edited, committed, pushed or submitted; no ssh to Oscar. Baseline built as a
detached worktree of `ada4075` in the scratchpad (removed afterwards). Environments: `/Users/samib/se7-craftax-venv`
(jax 0.11.1, `jax_threefry_partitionable=True`, craftax 1.6.1, torch 2.14) for everything touching the reference,
`/Users/samib/se6-venv` for the rest, `MUJOCO_GL=glfw`; `PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider` so nothing was
written into the author's worktree.

## Verdict: APPROVE-WITH-CHANGES

The renderer does what is claimed, and more strongly than the author's own evidence shows: it is byte-identical to the
un-jitted craftax 1.6.1 renderer on my own 2,009-state random rollout extracted with **my own** EnvState-to-window
extractor (not the author's bridge), on 90 level-shifted/band-forced real states covering all nine levels, and on 49
hand-built edge states, at 10 px and 16 px; the committed corpus reproduces the two digests the Mac, x86-64 and RTX 3090
records carry; every omitted `EnvState` field moves 0 pixels; the frame is a pure function of the 544 slots across
histories; the dead-dim check has power under this sampler; CraftaxClassic is untouched in fact as well as in the diff.
Six things must change before merge. None changes a pixel, the frozen corpus or any digest.

### Required changes

1. **Module docstring (lines 60-62) states a limit that is false.** "a projectile may in principle share a tile with a
   mob, and that is not representable here" -- measured: `proj_on_mob_tiles` (six projectiles placed on mob tiles) is
   byte-identical to the official frame at 10 and 16 px; the `ov` table is the `mob x projectile` product, so it is
   representable by construction. The actual limits are **two mobs on one tile** (max abs 113/138 on my pool) and **two
   projectiles on one tile** (187/186), plus the bridge-side budget of `MAX_MOB_PROJ + MAX_PLAYER_PROJ = 6` (an
   `EnvState` has `mob_projectiles.mask` (9,3) and `player_projectiles.mask` (9,3): a 7th cannot exist; the bridge
   drops it, max abs 171 vs ours, and ours draws it). State those three, and put the >6 limit in the **class**
   docstring as the author proposed.
2. **"4 gathers and 3 selects" (module docstring line 20; `_render_atlas` docstring line 1003 "four gathers and three
   selects").** The frame is one background gather plus three stencil layers, each a colour lookup + mask lookup +
   `where`: seven `index_select`s and three selects. The author flagged this; I agree.
3. **Module docstring line 44 cites `tests/unit/test_craftax_full_night.py`, which does not exist.** The pin is
   `tests/unit/test_craftax_full.py::test_the_vignette_is_a_reciprocal_multiply_and_not_a_divide`.
4. **`certification_state_distribution["uint8_fast_path_share"] = 2/6` is wrong; the sampler yields 1/4.** Band 2
   (`i%6==2`) never lands on a sleeping row (`i%4==3`), but band 3 (`i%6==3`) alternates sleeping/awake, and a sleeping
   row takes the float chain. Measured at `sample_states(64, rng(20260915))`: 0.25 (16 rows); the corpus's own
   `coverage.uint8_fast_path: 14` of 64 agrees with 1/4, not 1/3. The author's test hides the 0.083 error behind
   `abs(...) < 0.1`. Fix the value to `1/4`, reword the note ("band 2 and the awake half of band 3"), and tighten the
   tolerance to the one-row remainder the other assertions use. Related labelling nit: `certify.py::band()` tags a row
   "fast_uint8" by lighting alone, so findings §12's "fast (21)" counts 7 sleeping rows that do not take the fast path.
5. **findings.md:889 cites `scratch/step7/craftax-classic-full/paired_verdict.py`**, which exists only on W1's branch.
   On this branch `tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves` goes from 11 broken
   citations (baseline, pre-existing) to 12 -- this is the one regression the branch introduces in `tests/unit`. Cite
   the branch's own `scratch/step7/craftax-full/paired_verdict.py`.
6. **`check_state_dims_matter` at batch 1 reports `inv_ench_bow` dead in 3 of 200 seeds** (seeds 100, 202, 248; the
   author's seeds 0..11 reproduce `[]`). Mechanism confirmed directly: the reference draws
   `arrow_enchantment_textures[bow_enchantment * (arrows > 0)]`, so with `arrows == 0` in the single sampled row the
   dim moves 0 pixels (with `arrows == 1` it moves 2). Probability 1/100 per band-0 row; 0 hits in 60 seeds at batch 2.
   This is the fifth conditionally-dead dimension of exactly the kind the sampler already guards four times, and the
   guard is corpus-neutral: lift `arrows` to >= 1 on band-0 rows post hoc (no RNG draw consumed) -- the 512-row pool
   has **no** band-0 row with `arrows == 0`, so `pool.npz`, `states.npz`, both digests and all three cluster records are
   unchanged. Either add that one line (preferred) or document the conditional in the sampler docstring and the
   findings' "dead_dims [] at batch 1" claim.

### Checklist, with what was run and observed

**1. Parity.** Scripts `w2_common.py` (my extractor, written from `craftax/craftax/renderer.py`), `w2_parity.py`,
`w2_adversarial.py`.
- Committed corpus (64x544, sha `b0c38605…`): ours vs un-jitted, **10 px exact_frac 1.0 max_abs 0, digest
  `b3dcba4dea708d15…`; 16 px 1.0 / 0, `43414708ce31a3d7…`** -- identical to `certify_cpu_mac_corpus.json`, the x86-64
  record (node2352, job 6397697) and the CUDA record (gpu2106, RTX 3090, job 6397699). Also reproduced inside
  `fidelity_step4.family_specific` (`render_parity_with_official_renderer`: exact 1.0, interior 1.0, ssim 1.0).
- My rollout: seed 777, 2,000 random-policy steps, 8 resets, 2,009 states (levels [0] only; 8 day / 1,275 dusk / 726
  night; 288 sleeping; 1,441 with a mob, 325 with a projectile on screen; no tile ever held two entities). Jitted
  screen on all 2,009 at 10 px: exact 0.999992, max 1, 38 rows differ (the jit gap). **Un-jitted, every 5th state
  (402): 10 px 1.0 / 0; 16 px 1.0 / 0.**
- Level-shifted/band-forced (90 states from 5 rollout bases x 9 levels, at the up-ladder and at a random position,
  random `light_level`, 30% sleeping; 53 rows with `light_map < 1` in view): **1.0 / 0 at both sizes**.
- Adversarial (52 cases): exactly 6 projectiles in all four orientation classes incl. diagonal/zero directions, full mob
  budget, projectiles on mob tiles, projectile on the centre tile, mob on the centre tile, the opaque ranged-5 sprite
  on the centre, torches/ladders open and blocked incl. under a mob and at the centre, DARKNESS blocks, every block id
  once incl. NECROMANCER, a light-map ramp 0..1 on level 0, the map corner (OOB + light-0 padding) on levels 0 and 3,
  level 8 with the necromancer vulnerable / not / with a mob present / dark+sleeping, levels 1-8 at the up-ladder, xp 12
  (clamps to 9: matched), health 0.4/7.9/0, all counts at 99 with every tool/armour/enchantment/spell set, arrows
  0 vs 5 with bow enchantment 2, `light_level` at 0.5 / nextafter / 0 / 1-eps / 0.25, keys (0,0), (2^32-1, 2^32-1),
  (2^31, 2^31), mixed, sleeping at night, four facings at dusk. **49 representable rows 1.0 / 0 at 10 and 16 px.** The
  three that differ are the deliberately unrepresentable ones (two projectiles on a tile, two mobs on a tile, count
  100 -- unreachable, the simulator clips at 99).
- Jitted vs un-jitted on 402 real states: max float |diff| **3.05e-05** (the declaration's "up to 1.53e-05" is a
  phase-0-pool number and should say so), uint8: 10/402 rows differ by 1, 392 identical. `reference_env` returns the
  un-jitted function, asserts `jax_threefry_partitionable`, builds `do_night_noise=True`; `parity_note` names
  un-jitted, UINT8, both flags (pinned by `test_the_declaration_names_what_the_parity_claim_depends_on`).

**2. Window sufficiency and purity.** 33-35 perturbations of every omitted field on four states (level-0 day, level-0
night, level-3 dark sleeping, level-8 with the necromancer in view): `timestep`, `achievements`, the five recover/
hunger/thirst/fatigue/mana floats, `is_resting`, `potion_mapping`, `chests_opened`, growing-plant positions/age/mask,
`down_ladders`, `up_ladders`, `boss_progress`, `fractal_noise_angles`, `map`/`item_map`/`light_map` off-screen and on
other levels, `mob_map`, mob `health`/`attack_cooldown`, a masked off-screen mob, a mob on another level at the
player's x/y, `monsters_killed` on another level, `state_rng` where no noise is drawn, `light_level` off level 0,
`boss_timesteps` with no necromancer in view -- **every one moved 0 pixels**. Purity across histories:
`official(bridge(extract(s)))` (the author's bridge writes the window into a `reset(PRNGKey(0))` template) equals
`official(s)` on 99 states (rollout + adversarial): **1.0 / 0**. Bytes/state 544x4 = 2,176; frame 42,900 (19.7x) /
109,824 (50.5x) as declared.

**3. Sampler and dead dims** (`w2_gate_a.py`, `w2_dims200.py`). `dead_dims == []` at batch 64 in the battery, and at
batch 1/2/4/16 x seeds 100-103 except the batch-1 hit in item 6. Power: a subclass whose render zeroes `inv_books`,
`light_r0c0`, `rng_half3`, `proj_r8c10` before rendering is reported with exactly those four dead at batch 1, 2 and 4
-- the four sampler devices do not hide a truly dead dim. `sample_states` is reproducible (same rng -> identical
array). Declared shares vs `sample_states(64)`: level-0 day/dusk 0.172, night 0.328 (declared 1/6, 1/6, 1/3 with the
remainder), underground lit/dark 0.172/0.156, sleeping 0.25 -- all as declared; `uint8_fast_path_share` is item 4.

**4. Gate A and F.** `examples/fidelity_step4.py`'s own `mandatory()` + `family_specific()` with `reseed(env)` between
checks on the frozen corpus: all 14 mandatory checks measured and clean (roundtrip 0, dims [], batch/order/chunked/
input-form/batch-size 0, determinism identical, render-pure 0/0/0/0, set_state_copies 0/not rebound, style-is-state 0,
default-style-official 0, per-env variation [], fidelity declaration parity/1.0/1.0); family-specific: parity 1.0/0,
variations [] , colours [], backend_agreement/visual/task-visibility not_applicable for the right reasons,
resolution_consistency 0.7945 (protocol draw) / 0.8411 (frozen corpus). The reseed trap is real (rendering after
`check_render_is_pure` without `reseed` changes the frame), but the author's `certify.py` takes parity on a fresh env
before its Gate A so its numbers are unaffected; its 18 checks are the 18 in the record, and the CUDA record adds
`check_reference_renderer_is_stable = 0`. The `not_applicable` declaration is a class attribute carrying the evidence
path (`phase0/resolution_and_dead_dims.json`: official 0.4895 overall, world 0.4817 in daylight, 0.0991 at night, strip
0.94, max abs 191), both mechanisms, `craftax 1.6.1 un-jitted, jax 0.11.1, partitionable=True`, the scope note and the
still-required clause; `stable_envs/testing` and every threshold untouched (0 files in the diff). Gate F: facade obs
(544,) float32, `render()` (130,110,3) uint8 and (208,176,3) at 16, `step()` raises `NotImplementedError` by design,
`swap_renderer` exists, `out=` and leading-dims paths identical, mixed plain/lit batches equal per-row renders. The real
SWM wrapper (loaded from `/Users/samib/stable-worldmodel/.../craftax/craftax.py`): `obs_shape (130,110,3)` float32 in
[0,1] = `render_craftax_pixels(state)/255` of the **whole** frame (strip rows non-zero, night applied), no action repeat,
`reset`/`step` jitted; ours == `round(obs*255)` on 5/6 sampled steps and == the un-jitted render on 6/6 (the 6th is a
sleeping frame whose 15,579 half-integer pixels do not survive the `/255 -> *255` round trip; jitted == un-jitted there).
SWM's `render()` truncates -- the pre-existing shard-wide note from W1.

**5. Reuse and scope.** `git diff --quiet ada4075..HEAD -- stable_envs/envs/craftax.py stable_envs/render/prng.py`:
identical; the `stable_envs` diff is `envs/_craftax_full_tex.py` and `envs/craftax_full.py` only, both new; imports are
`..render.prng`, `..core.env`, `..spaces`, `._craftax_full_tex`. `night_vignette` is **bitwise** equal to craftax's
`night_noise_intensity_texture` at 10 and 16 px (the naive divide differs on 2,265 / 13,956 elements). All 27 baked
tables at both sizes regenerate byte-identical from `gen_consts.tables_for` against craftax 1.6.1. On the real atlas
every overlay alpha (item, player, three mob classes, projectile, digits, three enchantment sets) is in {0,1} at 10 and
16 px and every texture is integral, so the stencil-select composite is exact; `ItemType.NONE`'s alpha is 1.0 in
craftax (opaque `debug.png`) and 0 in the bake. xp clamps (xp 12 matched). `do_night_noise=True` as ruled.

**6. Regression.** Frozen Step-4 `states/CraftaxClassic.npz` (64x127, zero-extended to 149 with `light_level=1`)
rendered on the branch and on `ada4075`: world-rows digest `1aeabbe8bc6a9a2e…` on both (== the frozen record in
`step4/merged.json`), full 144-row frames `np.array_equal` True; W1's corpus (`craftax-classic-full/corpus/states.npz`)
digest `ee7af0104c7d67a8…` on both trees, arrays equal. `tests/unit`: branch **930 passed, 7 skipped, 2 failed**;
`ada4075` **905 passed, 7 skipped, 2 failed**; the same two pre-existing failures (`test_examples…render_mjwarp.py --help`,
`test_every_cited_scratch_path_resolves`), with the branch adding one citation to the latter (item 5).

**7. Hygiene.** `git log --format=%B ada4075..HEAD | grep -ci -E "co-authored-by|generated with"` = 0; author `sami-bg`
on all 9 commits; `examples/results/step4|5|6`, `stable_envs/testing`, `stable_envs/render`: 0 files changed;
`_pool.pkl` (27 MB) untracked, not in `git ls-files`; committed results 408 KB (`pool.npz` 164 KB, `states.npz` 21 KB,
`_craftax_full_tex.py` 41 KB, golden 2 KB). Working tree at review time carried the author's uncommitted
`gate_f_swm.py` / `gate_f_swm_nowrapper.json`; not reviewed.

**8. Performance record.** `bench_cpu_mac.json`, `bench_cpu_node2352…json`, `bench_cuda_gpu2106…json`,
`paired_verdict_node2352…json`, findings §16/§17.2: local numbers say "NOT a certification number"; both `S_cpu`
denominators are recorded with "NEITHER certifies anything yet … Escalated with W1's"; the slower band is flagged
(`level0_night`, 6.75x on the Mac, 20.7x on node2352 at B=256, 20.8x on the 3090 at B=4096). The paired CPU tier
(1,337 f/s at B=64/10 px, `cores_cpu` 0.356, `S_cpu` 3.81 jitted / 1,870 un-jitted) is gated USABLE by the author's
three gates, but note it was taken at loadavg 38.9 on a 48-core node and the same job's diagnostics show contention
(16 px B=256 at 265 f/s below B=16 at 589): read the absolute tier as a lower bound, as the findings already imply.

### Non-blocking notes for the coordinator
- Both Step-7 corpora (`craftax-full/corpus/states.npz`, W1's too) carry only a `states` key; `fidelity_step4.corpus_for`
  reads `z["meta"]` unconditionally, so the Step-8 freeze needs the Step-4 meta block written into them.
- `test_a_window_carrying_more_projectiles…`'s last assertion compares against an all-zero state and so cannot fail for
  the reason its comment gives; asserting the 7-projectile frame differs from the 6-projectile one would test it.
- The x86-64 (`node2352`) and CUDA (`gpu2106`) records were not re-derived here; their digests equal what I measured.

### What only the cluster can verify
The CUDA path itself (no local GPU; the recorded job 6397699 gives the same digests and a clean Gate A including
`check_reference_renderer_is_stable`), the CPU tier on a batch node (6397916), the x86 unit suite, and the owed
`examples/swm_compat/run_swm_workflow.py` run with `stable_worldmodel` installed.

### Files
Scratchpad: `w2_common.py`, `w2_parity.py` (+`.json/.log`), `w2_adversarial.py` (+`.json`), `w2_tables.py` (+`.json`),
`w2_gate_a.py` (+`.json`), `w2_dims200.py` (+`.json`), `w2_gate_f.py` (+`.json`), `w2_regress.py`, `w2_swm.py` (+`.json`),
`unit_branch2.log`, `unit_base2.log`, `cited_branch.txt`, `cited_base.txt`.

---

# Re-check before merge — HEAD `de31e893ea70488a77717b954fd0b4d8d4321010`

Commits `ab42355` (the six required changes) and `de31e89` (the Gate-F probe script, its sbatch, and a dangling
section reference in findings) on top of `59ab465`. `git diff --stat ab42355..de31e89 -- stable_envs tests` is empty;
`git diff 59ab465..HEAD -- stable_envs tests` touches only `envs/craftax_full.py` (docstrings, the declaration, the
sampler guard) and `tests/unit/test_craftax_full.py`. Trailer grep 0; author sami-bg on all 11 commits; 0 files under
`examples/results/step4|5|6`, `stable_envs/testing`, `stable_envs/render`; `_pool.pkl` untracked; worktree clean apart
from it. Read-only; nothing edited, committed, pushed or submitted.

## Verdict: APPROVE

**The six required changes are satisfied, verified by reading the diff and by re-running.**
1. Module docstring now says a mob and a projectile on one tile ARE representable, names the real limit (a second entity
   of the same kind on one tile, with the mechanism and both measured ranges: author 156 / 237 / 253, mine 113-187) and
   the 6-projectile bridge budget; the class docstring carries the budget with the `(9, 3)` mask shapes and points at the
   boundary pool and the test. Matches what I measured.
2. Both docstrings: one background gather + three stencil layers = seven `index_select`s, three selects.
3. The vignette pin is cited as `test_craftax_full.py::test_the_vignette_is_a_reciprocal_multiply_and_not_a_divide`.
4. `uint8_fast_path_share = 0.25` with the mechanism (rows 2, 8, 9 of the 12-row period) in the comment and note;
   tolerance `<= 1/64`. Re-measured at `sample_states(64, rng(0))`: **0.25**. Findings item 4 records the fix. One
   discrepancy with the summary I was given: `scratch/step7/craftax-full/certify.py::band()` is **not** changed or
   documented on the tip (no diff since `59ab465`, no docstring) and findings §12 still heads the column "fast (21)".
   Scratch-only, non-blocking; note it so the "lighting classifier" wording is not assumed to exist.
5. findings.md:889 now cites `scratch/step7/craftax-full/paired_verdict.py`;
   `test_every_cited_scratch_path_resolves` reports **11** broken (the baseline's 11), so the branch no longer adds one.
6. `sample_states` clamps `arrows >= 1` on band-0 rows (post hoc, no RNG draw), with the measurement in the comment, and
   `test_the_arrow_guard_leaves_the_frozen_corpus_untouched` asserts the recorded pool/kept sha256. Re-run
   `check_state_dims_matter` at batch 1 on seeds **100, 202, 248** (the three hits), 0..11 and 300..339: **no dead dim**.
   Regenerated the 512 pool from `corpus.json`'s seed: `pool_sha256` and `sha256` **equal the recorded values**, and
   `pool[indices]` is array-equal to the committed `states.npz` — the corpus is unchanged.

**Digests unchanged.** Committed corpus vs un-jitted craftax 1.6.1 re-taken on the tip: 10 px 1.0 / 0
`b3dcba4dea708d15…`, 16 px 1.0 / 0 `43414708ce31a3d7…` (ours == official), identical to the Mac, x86-64 and RTX 3090
records; the only change in `certify_cpu_mac_corpus.json` is `reference_seconds_per_frame`.

**Tests.** `tests/unit` on the tip: **931 passed, 7 skipped, 2 failed** — the same two pre-existing failures as
`ada4075` (905/7/2): `test_examples…render_mjwarp.py --help` and `test_every_cited_scratch_path_resolves` (11 broken on
both). The rewritten 6-vs-8 projectile assertion is real (checks only tiles 6 and 7 move, on the batch-indexed frame).

**Nits landed.** "up to 1.53e-05" is now "on the 49-state phase-0 pool" with my 3.05e-05 recorded; the CPU tier is
labelled a lower bound (§17.2, §18); the Step-8 carry-overs (corpus `meta` block; W1's denominator) are in §18.1.

**Gate-F jitted-frame observation.** `gate_f/gate_f_swm_nowrapper.json` (craftax's own env, which the wrapper wraps;
the wrapper layer itself is Oscar job 6398628): `round(obs*255)` vs ours 0.99963 exact, max abs 1, 39/41 identical;
against the un-jitted reference 1.0 / 0 on 8. My own run of the real `CraftaxPixelsWrapper` (loaded by file, §4 above)
agrees: ours == the un-jitted render on every sampled step and == `round(obs*255)` except where the frame holds
half-integers (a sleeping row) — and there the jitted and un-jitted frames were bit-identical, so part of any residual
in this protocol is the `/255 -> *255` round trip rather than fusion. Judgement: **no declaration wording beyond what
Classic carries is required.** The mechanism is the one `parity_note` already states in full (jitted vs un-jitted,
uint8 claim, `reference_env` un-jitted), it is shard-wide (every jax-backed SWM id jits `step`), and Classic's
approved declaration says nothing about the wrapper. If the coordinator wants it explicit, the honest form is one
sentence in `parity_note` — "the SWM wrapper jits `step`, so a live workflow observes the jitted frame; the certified
claim is against the un-jitted renderer, and the two differ by at most 1 uint8 step on a small fraction of pixels" —
and it belongs to both Craftax cells as a Step-8 carry-over, not to this branch alone. Not a defect.

Re-check artifacts: this section's commands are in the session log (dead-dim seeds, pool/kept sha, digests, pytest).

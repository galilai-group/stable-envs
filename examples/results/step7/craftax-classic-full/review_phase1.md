# Step 7 W1 (craftax-classic-full) — independent review, phase 1

Reviewed HEAD `a1ebf57af2b912ef84f5a888270d184d20490195` on `step7/craftax-classic-full` (worktree
`/Users/samib/stable-envs-step7-craftax-classic-full`), 10 commits on baseline `76dc5116`. Code/test/corpus
content is identical to 53fcb15 (later commits touch sbatch scripts and result artifacts only; verified with
`git diff 8e4506c..HEAD --stat` and `git diff --name-only 8e4506c..HEAD -- stable_envs tests examples` = 0).
Worktree clean at review time. Nothing edited, committed, pushed or submitted; no ssh to Oscar was needed.

Environments: `/Users/samib/se7-craftax-venv` (jax 0.11.1, craftax 1.6.1, torch 2.14) for jax/craftax work;
`/Users/samib/se6-venv` for the rest; `MUJOCO_GL=glfw`. Baseline built as a detached worktree of 76dc5116 in the
scratchpad (removed afterwards).

## Verdict: APPROVE-WITH-CHANGES

The renderer does what is claimed: byte-identical to un-jitted craftax 1.6.1 on my own state pool (fresh seed plus
hand-built boundary rows) at 7 px and 16 px; the Step-4 world rows replay to the frozen digest; the PRNG is
bit-exact on every fresh draw including `jax.random.bits` low bits, int32-boundary keys and counter offsets; the
two shared-file edits are minimal and weaken nothing. Four things must change before merge, none of which alters a
pixel.

### Required changes

1. **`stable_envs/render/prng.py`: `uniform(minval, maxval)` is not bit-exact against jax and the docstring says
   it is.** jax's own `uniform(key, shape, f32, 32., 127.)` fuses `floats*(max-min)+min` into a single-rounding FMA
   inside its jitted body; `prng.uniform` does mul-then-add. Measured (key `DEADBEEF/CAFEBABE`, 64x64):
   jax `uniform(minval,maxval)` == float64-emulated FMA on 4096/4096; jax `uniform()*95+32` (two un-jitted ops)
   differs from jax `uniform(minval,maxval)` on 508/4096; `prng.uniform(k, s, 32, 127)` matches the *former*, not
   the latter (also mismatched for (-100, 100); matches for (-1, 1) and (0.25, 0.75) where the span is a power of two).
   Craftax Classic and full Craftax both draw `uniform(key, shape) * 95 + 32` with the default range, so **parity is
   unaffected** — but W2 is told to consume this module verbatim and the docstring makes a bit-level claim for the
   range arguments. Fix: either drop `minval`/`maxval` from the interface, or document them as mul-then-add
   semantics ("`uniform(k,s)*(maxval-minval)+minval`, NOT jax's fused `uniform(k,s,minval,maxval)`") and add a test
   pinning that identity. The existing `test_uniform_is_in_range_and_is_not_degenerate` only checks range.

2. **The daylight fast path is not "untouched" (findings §9, craftax.py docstring); per-call overhead was added and
   part of it is avoidable.** Same machine, one thread, daylight/awake/unit-gain states, baseline vs branch:

   | B | t | baseline | branch |
   |---|---|---|---|
   | 1 | 16 | 0.061 ms (16.3k f/s) | 0.166 ms (6.0k f/s) |
   | 16 | 16 | 0.148 ms (108k f/s) | 0.309 ms (52k f/s) |
   | 256 | 7 | 0.696 ms (368k f/s) | 1.231 ms (208k f/s) |
   | 256 | 16 | 1.460 ms (175k f/s) | 2.092 ms (122k f/s) |

   About 0.42 ms of the B=256/16 px delta is the frame growing 112->144 rows (legitimate, it is the coverage
   change). The remaining ~0.1 ms fixed cost per call is: `CraftaxClassic._strip_grid` rebuilding `cell_of`/`blank`
   with 32 indexed assignments on every render (measured 39.7 us/call on CPU; on CUDA that is ~34 tiny launches and
   two allocations per frame — the per-frame-launch pattern AGENT_COMMON lists), `_decode` (46.5 us at B=1), and on
   the lit path `torch.tensor(NIGHT_TINT, device=dev)` / `torch.tensor([0,0,8])` built per call (H2D copies on
   CUDA). Fix: cache `_strip_grid` per device like `_VIGNETTE_CACHE`/`_StripAssets`, hoist the two constants, and
   replace the "measures the same as before the patch" sentence with the measured before/after. The CPU_FAST bar
   (S_cpu >= 5 against a ~5k f/s decode bar) is still cleared by >20x on this band, so this is not a tier risk; it
   is an accuracy-of-claims and a GPU-hygiene item.

3. **Wording of the jitted-reference claim.** `parity_note`, the module docstring and `reference_env`'s docstring say
   every jit-differing pixel is on a sleeping row. On my pool (52 states, 16 px) the jitted reference differs from
   the un-jitted one on 9 rows: 8 sleeping, and **row 42: awake, `light_level = 0.49999997`**, 45 px, max abs 1
   (7 px: same rows). Awake rows at `light == 1.0` never differ (consistent with the Step-4 jitted parity). The
   mechanism is "XLA fusion moves float32 by ~1e-5 and a pixel sitting at a uint8 rounding boundary flips", which
   applies to any lit row; sleeping rows are where it is frequent because the desaturation halves the range. Reword
   to "on the 64-state pool every differing pixel was on a sleeping row; the mechanism applies to any lit row" in
   the three places. The decision itself (un-jitted reference, uint8 claim) is right and is the one I would make.

4. **findings.md §8.4 / §11 say the Step-4 digests "cannot be compared until the CUDA leg runs".** They can and do:
   see Regression below — both the baseline CPU render and the branch's world rows digest to the frozen
   `1aeabbe8bc6a9a2ee1fdb8ffc27a737bd659630dc6e3a1b98f6bae002c894a4d`. Update the two sections and drop it from
   the owed list. Also "(4 commits)" in the header is stale (10).

### Checklist, with what was run

**1. PRNG.** Golden file provenance: re-ran `gen_prng_golden.py` in the craftax venv to a scratch path -> output
byte-identical to `tests/data/prng_threefry_golden.json` (sha256 `40990c34…`; jax 0.11.1, partitionable True,
impl threefry2x32 recorded in the file). `tests/unit/test_craftax_prng.py` imports only numpy/torch/json (no jax).
Fresh probe (`prng_probe.py`): 36 keys (incl. `FFFFFFFF/0`, `0/FFFFFFFF`, `80000000/80000000`, `7FFFFFFF/80000001`,
25 random, 4 `PRNGKey(seed)`) x 9 shapes (incl. odd counts 13, 1001, 3-D (2,3,4), and 112x144/63x77/49x63):
`random_bits` vs `jax.random.bits(key, shape, uint32)` **324/324** (this checks the low 9 bits the golden file never
sees), `uniform` **324/324**, `uniform*95+32` (the exact craftax expression) **324/324**; counter-offset chunking
(5000+7000+rest of 16128) rebuilds the whole field; with `jax_threefry_partitionable=False` jax and prng.py
disagree (expected). Shift/mask: `_rotl32` keeps `d` bits after `>> (32-d)` with mask `(1<<d)-1`, matching a
python-int rotation on 517 values (author's test, re-run); `uniform` uses `(bits >> 9) & 0x7FFFFF | 0x3F800000`
which is jax's `bits >> (32-23) | float_bits(1.0)` with the extra mask needed because torch's int32 `>>` is
arithmetic — correct. int32 boundaries: negative-pattern keys and `_counter` past 2**31 (int64 arange narrowed)
are covered by the 324/324 and by the parity rows with key halves 65535/32768. Exception: item 1 above.

**2. Parity.** `parity_probe.py`, un-jitted reference, my pool of 52 states (40 from `sample_states(rng(424242))`
+ 12 boundary rows: light in {0.5, nextafter(0.5,0), 0.0, nextafter(1,0), 1.0, 0.25, 0.75}, keys
`FFFFFFFF/FFFFFFFF`, `80000000/7FFFFFFF`, `0/0`, all sixteen counts = 9, sleeping alternating; 16 night / 24 dusk /
12 day / 27 sleeping): **7 px exact_frac 1.000000 max_abs 0; 16 px 1.000000 / 0**. Author's seed 20260915
reproduced locally: digests `288630d8d823c0b9` (7 px) and `820c52a906ae839a` (16 px), ours == official, identical
to `certify_cpu_mac.json` and to the Oscar x86-64 record `certify_cpu_node2416…json` (jax 0.11.1, partitionable
True). Committed corpus (64x149): 1.0 / 0 at both sizes. Jitted reference: 99.80% (7 px) / 99.77% (16 px), max
abs 1, rows characterised in item 3. `reference_env` returns the un-jitted function and raises if
`jax_threefry_partitionable` is off; `reference_versions = {craftax 1.6.1, jax 0.11.1}` and `parity_note` name the
flag, the un-jitted choice and the uint8 nature of the claim. Honest.

**3. Reductions.** Intensity skip: `dark = light < 0.5`; the reference's `max(2*(0.5-d), 0)` is exactly 0 for
d >= 0.5 and `(1-0)*world + 0*noise` is `world` for finite noise, so the skip is exact; a row at
`nextafter(0.5, 0)` is *not* skipped (intensity 5.96e-8) and is in the parity pool (exact). Key-sensitivity
probe: changing the key at light 0.5 / 0.75 / 1.0 changes 0 pixels. Aliasing: `night_and_sleep` leaves its
`world` input bit-identical, is idempotent, and its output does not alias the input; in `_render_atlas` the
in-place ops act on `night = world.clone()` and `lum` (fresh), and `px[:, :w] = ...` copies back. Daylight fast
path: author's `test_daylight_is_the_identity_and_keeps_the_uint8_fast_path` passes; my probe renders 64
daylight/awake rows alone (uint8 gather) and inside a batch with one sleeper (float chain) at 7/16/32 px:
identical, max abs 0; and the float chain applied to the whole 816-tile float table at light=1.0 is bitwise the
table, whose rounding is the uint8 table.

**4. Regression.** Shared-code footprint: `git diff --name-only 76dc5116..HEAD -- stable_envs` = `envs/craftax.py`,
`render/prng.py` (new) only; no tilemap/`_Composite`-shared file, scene, core or testing module touched; the two
flagged files are `examples/fidelity_step4.py` and `tests/unit/test_declarations.py`. `examples/results/step4|5|6`:
0 files changed. Step-4 replay (`replay_step4.py`): frozen `states/CraftaxClassic.npz` (64x127) rendered on the
baseline worktree on CPU -> `frames_digest` = `1aeabbe8…` == frozen `oracle.digest_ours`/`digest_official`
(taken on a 3090 — CPU and CUDA agree). Branch: rows zero-extended to 149 with `light_level = 1.0` -> full frame
(64,144,144,3), strip max 0, world rows `[:112]` digest = `1aeabbe8…`, and `np.array_equal(baseline, branch
world rows)` True. So the world-rows-only obligation is met and no CUDA leg is needed for it. Unit tests
(`tests/unit`, `--ignore test_examples.py --ignore test_nvdiff_texgen.py`): branch **754 passed, 7 skipped,
1 failed**; baseline **700 passed, 6 skipped, 1 failed**; the one failure on both is
`test_repo_contracts.py::test_every_cited_scratch_path_resolves` (pre-existing).

**5. Shared files.** `fidelity_step4.py`: `"jax"` added to the tuple that records `reference_packages`; the loop
records `md.version` when installed, `__version__`/"present" as fallback, `None` when absent — same treatment as
every other package, no check weakened. `test_declarations.py`: `parity_note` is used only when
`visual_fidelity["measured_at"]` is empty. Negative tests run against the test module itself: a parity env with
`reference_versions` and no `parity_note` -> FLAGGED (`measured_at says ''`); a parity env whose note names
craftax 1.5.0 -> FLAGGED; a visual env without `measured_at` is refused at `register()` by
`stable_envs/core/env.py` before this test runs, so the change cannot let a visual env skip `measured_at`.
Minimal and honest.

**6. sample_states stratification.** Band = `i % 4` (night / dusk / 1.0 / raw cycle), `sleeping[::4] = False`,
`sleeping[3::4] = True` for batch > 3. `check_state_dims_matter` substitutes donor values and then nudges
(+25, -25, +1, +0.5) and only declares a dim dead if *every* perturbation leaves the frame unchanged, so a
dimension the renderer truly ignores fails regardless of which rows were sampled — stratification can only stop
the check from reporting a *conditionally* dead dim (facing while asleep, key at light >= 0.5), both of which are
genuine reference properties. Ran the check at B = 1, 2, 4, 5 x 6 seeds each: `dead_dims = []` every time.

**7. Gate F.** `StableEnvGym("CraftaxClassic", device="cpu")`: obs (149,) float32, `render()` (144,144,3) uint8,
info keys unchanged; at `size=7` render is (63,63,3), which is exactly
`stable_worldmodel/envs/craftax/craftax.py::CraftaxClassicPixelsWrapper`'s declared `obs_shape=(63,63,3)` — the
official observation is `render_craftax_pixels(state, 7) / 255.0`, i.e. the whole 63-row frame (49 world + 14 strip
rows, night and sleep applied), so strip/night are indeed part of the official observation. Note (pre-existing,
out of W1 scope): SWM's `render()` does `(clip(obs)*255).astype(uint8)` (truncation) while this repo rounds
half-to-even; the parity claim is against the rounded raw renderer as declared in the docstring.

**8. Hygiene.** `git log --format=%B 76dc5116..HEAD | grep -ci -E "co-authored-by|generated with"` = 0; author
`sami-bg` on all 10 commits; step4/5/6 results untouched; corpus 20 KB (states.npz 6.6 KB), phase0 72 KB, golden
64 KB, certify 2 x ~8 KB — modest.

**9. Policy.** `prng.py` is generic (no env knowledge; shapes/offsets/keys only); the night chain and vignette live in
`craftax.py` as module-level, shape-parameterised functions with a `light_map` hook for W2 (full Craftax draws the
same `uniform(state_rng, shape)*95+32` field — checked in its renderer source). No CUDA fallback: device comes from
the env, every tensor is built on `blk.device`/`world.device`; there is no `.cuda()`, no device string branch, and
nothing raises on cuda — the cuda path is the same code (unexecuted here: no local GPU; the author's `04_gpu.sbatch`
owes it). The `plain` decision is one host read; `night_and_sleep` adds two more (`dark.any()`, `sleeping.any()`),
acknowledged in the code. Per-call launch overhead on cuda: item 2.

### arm64 vs x86-64 threefry agreement
My local check cannot by itself prove cross-architecture agreement, but it does not need to: the frame digests I
measured on this arm64 Mac (`288630d8d823c0b9` / `820c52a906ae839a`) equal the x86-64 Oscar record in
`certify_cpu_node2416…json`, and the golden vectors are architecture-independent sha256 expectations, so the
unit-test pass in job 6394533 is a direct x86 confirmation. Mechanistically, `threefry2x32` is int32
two's-complement `+`, `^`, `<<`, `>>`+mask, which torch defines identically on every CPU; the only float ops are an
exact bitcast/subtract and, in `uniform`, mul-then-add (no FMA since they are separate kernels). Settled.

### Non-blocking notes for the coordinator
- `certify.py`'s Gate A records `check_per_env_variation: skipped, batch too small` because the harness builds the env
  at its default batch; Step 4 measured it. The Step-8 certification run should use the corpus batch.
- The `check_resolution_consistency` / tier-per-band questions the author routed to Sami are correctly described;
  the measured night numbers (0.59 at 0.45, 0.28 at 0.10, mixed 0.757) reproduce on Oscar.
- Findings claim 832/37 for tests/unit; on this machine with the same deselections it is 754/7 (the difference is
  collection, not failures).

### Files
Probe scripts and logs are in the scratchpad: `prng_probe.py`, `parity_probe.py` (+`.log`), `jit_probe.py`,
`replay_step4.py`, `probes_cpu.py`, `perf_probe.py`, `unit_new.log`, `unit_base.log`, `golden_regen/`.

---

# Re-check before merge — HEAD `7a1a1387f73b3e0e95dbe7c0cb8515807c0deb2a`

Commits 978372b / b399fbb / 7a1a138 on top of a1ebf57. Worktree clean; trailer grep 0; author sami-bg;
`examples/results/step4|5|6` and `stable_envs/testing` untouched (0 files in `git diff --name-only 76dc5116..HEAD`).

## Verdict: APPROVE

**(1) The four required changes are satisfied.**
- `prng.uniform` signature is now `(keys, shape, offset=0)`; range args gone (positional/keyword calls raise
  `TypeError`), `test_uniform_takes_no_range_arguments` pins the signature, docstring records the FMA mechanism and
  the 508/4096 measurement; `test_the_scaled_night_field_matches` pins `uniform()*95+32` against jax-generated
  digests. Re-ran my fresh probe on the new API: `random_bits` vs `jax.random.bits` **324/324**, `uniform` 324/324,
  `*95+32` 324/324, offset chunking exact. The `[0,1)` output is bit-identical to before (the removed
  `*1.0+0.0`/clamp were no-ops).
- `_strip_grid` cached per device (`_GRID_CACHE`, returns `blank` pre-unsqueezed), colour constants cached
  (`_CONST_CACHE`/`_post_constants`), `_decode` does one `round()` over the tail block and returns key halves,
  `key_from_halves` runs only in `_post` (lit path). Semantics unchanged: parity re-run identical (below).
  `test_the_strip_grid_cache_is_not_written_through` guards the cache. **Quiet timing, min of 3 interleaved
  rounds, baseline (76dc5116 worktree) -> tip, one thread, daylight/awake/unit gain:**

  | B | t | baseline ms | tip ms | baseline x 144/112 | residual |
  |---|---|---|---|---|---|
  | 1 | 16 | 0.061 | 0.101 (was 0.166) | 0.078 | +0.023 |
  | 16 | 16 | 0.150 | 0.215 (was 0.309) | 0.193 | +0.022 |
  | 256 | 7 | 0.671 | 0.888 (was 1.231) | 0.863 | +0.025 |
  | 256 | 16 | 1.428 | 1.815 (was 2.092) | 1.836 | -0.021 |

  The per-frame cost is now the row growth; the residual is a flat ~0.02 ms per call (the 22 extra decoded slots and
  the strip tile-id math), which agrees with the author's 1.4396 -> 1.8706 / +0.0197 ms. At or near baseline
  apart from row growth: yes.
- parity_note, module docstring and `reference_env` docstring now say "on this pool", name the awake
  `light_level = 0.49999997` counter-example and state the mechanism as "any lit row can straddle".
- findings §8.4 now records the CPU comparison against the frozen record (`1aeabbe8…`, mean 93.8446, 277 colours,
  identical) and §11 lists the CUDA leg as confirmation only; the "(4 commits)" header is gone.

**(2) API change breaks nothing else.** `grep -rn "prng\." stable_envs tests examples scratch/step7` — the only
callers are `stable_envs/envs/craftax.py:537` (`prng.uniform(keys, (h, w)) * 95 + 32`) and `:928`
(`prng.key_from_halves`), plus `tests/unit/test_craftax_prng.py`; all other `uniform(` hits are numpy `rng.uniform`.
`tests/unit` on the tip (`--ignore test_examples.py --ignore test_nvdiff_texgen.py`): **760 passed, 7 skipped,
1 failed** — the same pre-existing `test_every_cited_scratch_path_resolves` (baseline 700/6/1).

**(3) Parity still byte-identical.** Same 52-state pool as phase 1 (fresh seed + boundary rows), un-jitted
craftax 1.6.1: **7 px exact_frac 1.000000 max_abs 0; 16 px 1.000000 / 0**; author's seed digests
`288630d8d823c0b9` / `820c52a906ae839a` reproduced; committed corpus 1.0 / 0; jitted-reference rows unchanged
(9 rows incl. the awake boundary row).

**(4) The oracle experiment is a faithful instance of the check.** `oracle_resolution.py` uses S = `env.size` = 16,
2S = 32, `cf._box_downsample(big, 2)` (the check's own function, unchanged in the untouched testing module), and the
check's statistic `(d.max(-1) == 0).mean()`; the states are the frozen Step-7 corpus split by band plus the check's
own `downsample_protocol_states(env)` draw; the official side is craftax's renderer **un-jitted** with each state's
own `state_rng` written back by `reference_render` (same rounding as parity), and the 32 px textures come from
craftax's `load_all_textures_given_size(32)` substituted into the `load_all_textures()[size]` lookup the renderer
performs — every 2S pixel is craftax's. I reproduced it independently (`mini_oracle.py`): corpus night official world
**0.1634 / ours 0.1634**, official strip 0.6815 / ours 1.0000; corpus daylight 1.0000 / 1.0000, strip 0.6933 / 1.0000;
protocol draw official frame 0.7440, world 0.7639 (= ours 0.7639), strip 0.6743; and on my own fresh pool night
0.1043 / 0.1043, day 1.0 / 1.0, strip 0.67-0.69 / 1.0. The committed JSON's numbers match the author's table
exactly. The applicability declaration (`resolution_consistency_applicability`) names the mechanism, the evidence
path, and `craftax 1.6.1 un-jitted, jax 0.11.1, jax_threefry_partitionable=True`; it is a class attribute only,
`stable_envs/testing` is untouched, the frozen threshold is untouched, and
`test_the_declared_resolution_applicability_matches_measurement` fails if our side ever meets the bound below 0.5
(staleness) or misses it at/above 0.5. Honest.

**(5) The strip observation is characterised correctly.** The official strip scores 0.68-0.69 in *every* band,
daylight included (band-independent, so it is geometry, not lighting): craftax re-lays the strip out per size with
icon `int(0.8t)` / digit `int(0.6t)` (12/9 at 16 px, 25/19 at 32 px), so its 32 px strip is not a 2x replication of
its 16 px strip; ours resamples the finished cell, so it is (1.0000), and `test_the_strip_is_resolution_consistent_in_every_band`
asserts it. Two facts from my run refine the wording rather than contradict it: (a) at daylight the official 32 px
*world* rows are exactly the 2x replication of its 16 px ones, and our 32 px world rows are **100% parity with
craftax's constructed 32 px world rows even at night** (same threefry field at (7*32, 9*32)); (b) our 32 px strip
differs from craftax's constructed 32 px strip on ~18% of strip pixels (0.81-0.82 exact), by design. So "a size
which has no official frame to be parity against" is better read as "no *shipped* textures" — craftax can build a
32 px frame and this renderer matches its world layer exactly there while deliberately not matching its strip
re-layout. A wording nit, not a defect.

### Note for the coordinator (not a defect of this branch)
The certifier still evaluates `check_resolution_consistency` on the protocol draw, which is a mixed 8-state pool
(ours 0.8164, official 0.7440), and `stable_envs/testing` was deliberately left alone; until the Step-8 certifier
reads `resolution_consistency_applicability` (or judges the check on the `light >= 0.5` subset), the frozen 0.995
bound will be recorded as failed on this config for a reason the oracle shows is the reference's.

Re-check artifacts in the scratchpad: `prng_probe2.py`, `parity_probe2.log`, `mini_oracle.py`/`.log`,
`unit_new2.log`, timing output in the session log.

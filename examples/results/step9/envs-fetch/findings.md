# Step 9 — Wave 1, Workstream C: the Fetch family

**Workstream:** `step9/envs-fetch` · **Parent:** `step9/integration` at `b3f3a81`, no rebase ·
**Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only — no CUDA device in this ticket.
**Interpreters:** `/Users/samib/se8-fetch-venv` (gymnasium-robotics 1.4.2, mujoco 3.12, numpy 2.5.3;
`tensordict` 0.14.2 installed for this work — see §10 E3), `/Users/samib/se6-venv` (the CPU suite),
`/Users/samib/se8-vm-venv` (SWM). `MUJOCO_GL=glfw`, `-p no:cacheprovider`.

**Result: PATCH.** All twelve Fetch cells are on the Wave-0 foundation. The oracle compares **41
keys** per cell through a detached worktree of `21cb306` and this branch in the same venv and finds
**zero unexplained differences**: `describe()`, the style layout and the **official default
appearance**, the variation taxonomy (**33 mapped / 12 partial / 12 unsupported / 15 dead / 51
n/a**, asserted as totals) and `classification()` leaf for leaf, every fidelity declaration
including `parity_note` verbatim, the registry rows including `module`, the twelve `swm/Fetch*-v3`
aliases, `state_dim`/`state_names`, the frozen Step-4 corpus round trip, `sample_states`, **the
baked plate** (default, varied, and the 64 plates a mixed batch cuts itself into), the mover mask and
the recovered camera. Local unit suite **1385 → 1395**, no pre-existing test changed its result.

**THE FRAME DIFF IS OWED, and it is owed for a reason that is a property of this family and not of
this patch.** Fetch's mover pass is `mujoco_warp`'s batch raytracer: CUDA-only, no CPU backend
(`CPU_BACKEND_MISSING`, Step 3, preserved). Neither tree can make a Fetch pixel on this Mac, so the
byte-for-byte frame comparison every other Wave-1 workstream reports is **explicitly owed to the
post-Wave-1 GPU oracle, which the coordinator runs**. `scratch/step9/envs-fetch/oracle.py` takes
`--python` and needs no other change there. Nothing in this document is a frame claim.

---

## 1. Files created / changed / removed

| file | what happened |
|---|---|
| `stable_envs/envs/fetch.py` | 1,038 → 1,178 lines, migrated and de-duplicated (§2, §3) |
| `tests/unit/test_envs_fetch.py` | **new**, 17 tests in two tiers (§5) |
| `scratch/step9/envs-fetch/oracle.py` | **new**, the two-tree comparison (§4) |
| `examples/results/step9/envs-fetch/oracle.json` | **new**, its record |
| `examples/results/step9/envs-fetch/findings.md` | this file |

Removed: nothing. Nothing moved module, so no import path changed and `registry_table()[i]["module"]`
is `stable_envs.envs.fetch` in both trees (compared, §4).

**The line count grew (+140) and none of it is duplication.** +58 of Step-9 machinery and its
explanation (the `num_envs=` constructor, the per-instance `state_layout`, `_close`, the state
comment block), +82 of docstring on the four folded helpers and the class, saying what each one is
and what the duplication it replaced actually was. Four separate idioms were **deleted** (§3).

## 2. API changes

**Additive.** Nothing a Step-8 caller did stopped working; the packed row, the declarations and the
registered defaults are unchanged.

```python
env = snv.FetchPush(num_envs=64, device="cuda")   # num_envs=, and batch= still accepted
st  = env.state                    # State: {"qpos": (nq,), "goal": (3,)}
rows = env.pack(st)                # the frozen packed row, byte for byte
px  = env.render(st) == env.render(rows) == env.render()      # CUDA only, as before
env.step(a)                        # NotImplementedError: no batched simulator (below)
env.close(); with snv.FetchReach(...) as env: ...
```

**The public State is `qpos` plus the goal, grouped by what the two halves ARE.** `qpos` is the
simulator's configuration; `goal` is the task target, which this env does **not** write to the
simulator at all — it writes it to a MOCAP body (`se:goalbody`), because MJWarp's `kinematics` does
not recompute `geom_xpos` for world-body geometry and a worldbody goal sphere stayed frozen at the
origin. That is a different kind of quantity from a joint coordinate, and the two keys say so.

**`state_layout` is an INSTANCE attribute and this family overrides nothing of the state machinery.**
`nq` is 15 on Reach (no block, no free joint) and 22 on the three scenes that carry one, so the
layout is a property of the COMPILED MODEL and not of the class — the same fact that made Workstream
B file R1. The foundation took R1 (`f24d94c` made `Env._layout`/`_layout_dtypes` instance methods
reading `self.state_layout`), so `__init__` assigns it where the model is compiled and there is no
override: not `state`, not `_layout`, not `_hidden_state`, not `_live_state`.
`test_state_layout_is_assigned_per_instance_and_nothing_else_is_overridden` pins that from the source.

**Nothing is hidden, and there is no `_live_state`.** Every column of the packed row belongs to one
of the two keys, so no field has offset `None`, `_hidden_state` is never consulted, and `pack`/
`unpack` round-trip the frozen corpus bit for bit (asserted per cell in §4 and §5). The packed
float32 row IS this family's live state — `set_state` writes it and `render` reads it — so unlike
classic control's float64 simulator there is no more precise buffer to answer from.
`mocap_pos`/`mocap_quat` are still **not** state, for the module docstring's measured reason: the
only pixels they reach are the four translucent markers this backend cannot draw.

**NO DYNAMICS, declared rather than omitted.** `FetchBase` defines neither `step` nor `transition`,
so `Env`'s precise refusal fires (`"FetchPush.step: FetchPush ships a renderer and no batched
simulator … PushT has dynamics; FetchPush does not"`) and `has_dynamics` is False. `reset` and
`render` work as usual. No simulator was invented and no zero reward is fabricated.

**`close()` releases what the INSTANCE owns and nothing else** — the per-env device tensors
(`_idx_cache`: the arm material indices, the pristine light and the mover LUT). The `mujoco.Renderer`
in `_RENDERER_CACHE`, the MJWarp contexts in `_CTX_CACHE` and the compiled models in `_MODEL_CACHE`
are **not** released, which is the family's policy rather than an oversight: they are shared by every
instance at the same (scene, resolution, worlds), and releasing an EGL context corrupts every other
live one in the process — reproduced while writing this env, and why `_renderer` says "never closed".
What `close()` buys is the contract's other half: a closed env refuses to render.

**`CPU_BACKEND_MISSING` stays.** No CPU backend was added, none was implied, and `device="cpu"` is
neither newly refused nor newly promised: it behaves exactly as it did at Step 8.

## 3. Duplication removed — four idioms, all mechanical

| what | before | after |
|---|---|---|
| module caches | six dicts, each with its own look-up/build/store; three ALSO bounded, and the bound checked in two different places — `_CTX_CACHE`/`_PLATE_CACHE` cleared before the insert (holding `maxsize`), `_CAM_CACHE` after (holding `maxsize + 1`) | one `_Cache(name, maxsize)`; `get_or_build` is the only way anything is stored, and the bound is applied before the insert as the two that matter already did |
| the pristine restore | two copies of the same assignment list in `_bake_plate`, already drifted — the entry copy omitted `cam_pos`/`cam_quat` | one `_restore(m, pr)` |
| the grouping decision | the same five lines in `render` (plate leaves) and `_camera_batch` (camera leaves), one deriving the leaf offsets and one hard-coding `r[0:1]`, `r[1:2]`, `r[2:3]` | one `_style_groups(sty, paths, B) -> (uniq, inv, offs)`; `_camera_batch` cuts its row with the returned offsets |
| the per-env tensor caches | `_idx`, `_light0` and `_mover_lut` each open-coded `self.__dict__.setdefault("_idx_cache", {})` + get + build + store | one `_cached(key, build)`; `close()` drops the one dict |

**What was NOT touched, deliberately:** the plate bake and its cache key, the mover-mask composite
(`torch.where(mv.unsqueeze(-1), rgb, pl)`), the supersample path, the chunk loop and `MJW_CHUNK`, the
`gl_rounding` capture, the surgical model build (`_build_bundle`), the variation space, the
`_check_plate_is_constant` assertion, the reference bridge, and every declaration.

**The twelve cells stay DATA, not twelve class bodies** (the brief's "one family, data over
subclasses"). `CELLS` is unchanged in name and shape — `scratch/swm_matrix.py` reads it — and
`_make_classes` now pins `__module__` to this module explicitly rather than inheriting `type()`'s
default from the calling frame. That default is currently correct; `registry.spec` derives an env's
FAMILY from the last component of `__module__`, and a family of twelve is not registered on a
coincidence of where a factory happens to be defined. A `__all__` was added.

## 4. The oracle — `scratch/step9/envs-fetch/oracle.py`

Twelve cells, the frozen Step-4 corpora (`examples/results/step4/states/Fetch*.npz`, 64 states each),
two interpreters, one venv, **41 compared keys**, run:

```
MUJOCO_GL=glfw /Users/samib/se8-fetch-venv/bin/python scratch/step9/envs-fetch/oracle.py
→ "identical": true, "unexplained_differences": {}, taxonomy_ok, swm_aliases_ok, functional_ok
   ORACLE: PASS (frames owed to the GPU oracle)
```

`examples/results/step9/envs-fetch/oracle.json`, head `c40be4d`. What it compares, and what each
difference would have meant:

| group | keys |
|---|---|
| the consumer's view | `describe()` |
| **the default appearance** | `style_names`, `style_dim`, `default_style_sha256`, and the default **value by value** — the thing Sami's ruling forbids moving |
| the taxonomy | `variation_names` per cell; `classification()` for all twelve `swm/Fetch*-v3` maps, leaf for leaf; and the **totals asserted against the Step-8 line**: 33 mapped / 12 partial / 12 unsupported / 15 dead_upstream / 51 not_applicable |
| fidelity | `fidelity_tier`, `render_decoder`, `parity_interior_min` (0.99), `parity_all_exact_min`, `parity_note` **verbatim**, `visual_fidelity`, `variation_fidelity`, `reference_versions` |
| registry | the full row including `module`; the twelve swm aliases resolve one-to-one |
| state | `state_dim`, `state_names`, `corpus_states`, `roundtrip_exact`, `rows_sha256`, `sample_states_sha256` |
| **the plate** | `plate_default_sha256`, `plate_varied_sha256` (a non-default appearance touching **every** leaf in `PLATE_PATHS`), `plate_shape`, `plate_paths` |
| **the grouping** | `style_groups_uniq_sha256`, `style_groups_inv`, `style_groups_offsets`, and `style_group_plate_sha256` — **the 64 plates a 64-environment batch of DIFFERENT sampled appearances actually bakes** |
| the composite | `movers`, `static`, `mover_lut_sha256` |
| the camera | `campos_sha256`, `camquat_sha256`, `cam_defaults` |
| the model | `geom_names`, `scene`, `ref_id`, `size`, `supersample`, `default_size` |

**The plate and the grouping are the load-bearing CPU evidence.** The plate is MuJoCo's own raster of
floor/table/skybox — 91% of the frame, exact by construction — and it is the one half of this
renderer that runs without CUDA. Comparing it covers the plate cache key, the pristine restore, the
recovered free camera and every per-leaf model mutation. The grouping comparison is the direct test
of the one piece of `render()` this patch rewrote: the migrated tree answers with `_style_groups`,
and the baseline, which has no such function, has the same five lines inlined into the probe from
`render()` — so the two IMPLEMENTATIONS are compared over the same sampled style. 64 distinct
appearances cut into 64 groups with identical inverse indices, identical offsets and 64 identical
plate digests, per cell.

### Two substitutions, applied identically to both arms

1. **`warp`/`mujoco_warp` are stubbed in `sys.modules`** so the module's import-time dependency check
   passes. Nothing the script calls touches either — the whole ray-cast path is skipped — and the
   stub is installed in both arms.
2. **`_state_pool` is replaced by a loader for the frozen Step-4 corpus.** The shipped pool builds
   itself by driving `gymnasium.make("Fetch*-v4")`, which is **UNMEASURED-locally**: it raises
   `AssertionError` from `gymnasium_robotics/utils/mujoco_utils.py::set_joint_qpos` under numpy 2.5.3
   + mujoco 3.12 — the upstream defect Step 3 already recorded at
   `scratch/swm_matrix.py::PROBE_LIMITATIONS["fetch_cpu"]`, reproduced here from a bare
   `gymnasium.make("FetchReach-v4")`. The corpus is the frozen record of what that pool produced on a
   working interpreter, so this substitutes the DATA, not the code path: `sample_states`'s own
   indexing, seeding and dtype are the shipped ones and are still compared.

### What is owed, stated as a debt rather than left as a silence

| owed | why | to whom |
|---|---|---|
| `render()` for all twelve cells at 480 px over the frozen corpora, byte for byte; and the GPU timings (plan §8) | no CPU backend for the mover pass (`CPU_BACKEND_MISSING`); neither tree can make a Fetch pixel on this Mac | the post-Wave-1 GPU oracle, run by the coordinator (`oracle.py --python <cuda python>`) |
| `reference_env()` / `reference_render()` / the shipped `_state_pool` | UNMEASURED-locally: the numpy 2.5.3 + mujoco 3.12 `set_joint_qpos` defect above. **Untouched by this migration** | a mujoco ≤ 3.11 interpreter, as Step 3 already recorded |
| the chunked render path (`MJW_CHUNK`, `_upload`, the composite, supersampling) | CUDA-only code; the grouping and plate halves it feeds are compared above | the same GPU oracle |

## 5. Tests

`tests/unit/test_envs_fetch.py`, 17 tests, two tiers — **and neither tier imports
`stable_envs.envs.fetch` into the suite**. Doing so needs the warp stub, and the stub would also
register twelve classes for the rest of the pytest session and change what every registry-wide test
in this suite sees.

| tier | how | what it pins |
|---|---|---|
| 1 (10 tests, run everywhere) | `ast` over the source | `FetchBase` derives `Env` and `StableEnv` is gone; `num_envs=` first with `batch=` still accepted and both forwarded; `state_layout` assigned per instance and **no** `_layout`/`state`/`_hidden_state`/`_live_state` override; no `step`/`transition`; `_close` names the instance cache and none of the shared ones; the 0.99 floor and both halves of `parity_note`; the six `PLATE_PATHS`; twelve cells over eight official ids declared as data; the module-scope import refusal; the six caches are one `_Cache` |
| 2 (7 tests, `gymnasium_robotics` required) | builds FetchReach + FetchPush in a **child interpreter** (stub + corpus pool), asserts over its JSON | the two-key state and its exact widths/offsets (15+3, 22+3); `pack` is the packed row; the corpus round trip; `num_envs`/`len`/`batch=`; the renderer-only refusal for both methods; `close` idempotent, drops `_idx_cache`, then refuses to render; the 480 px default; the plate bakes and the mover LUT covers the robot's meshes; each cell names its swm id in its docstring |

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1395 passed, 202 skipped, 112 deselected in 135.04s
```

Baseline for that command on this machine at `b3f3a81`, measured before any edit:
**1385 passed, 195 skipped, 112 deselected**. 1385 + 10 (tier 1) = 1395, and the 7 skips are tier 2,
which has no `gymnasium_robotics` in `se6-venv`. **No pre-existing test changed its result, and no
pre-existing test file was edited.**

```
MUJOCO_GL=glfw /Users/samib/se8-fetch-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_variation_maps_fetch.py tests/unit/test_envs_fetch.py   → 27 passed, 49 skipped
MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python   -m pytest ... (same two files)
                                                                          → 20 passed, 56 skipped
```

`test_variation_maps_fetch.py` alone is **10 passed, 49 skipped** in BOTH venvs and in BOTH trees
(baseline worktree and this branch) — identical, including which tiers skip. Its tier 3 needs a built
env and a GPU and skips here, as it does at Step 8.

## 6. Performance

**Not measured, and it should not be.** Every hot path in this family (the ray-cast, the chunked
upload, the composite) is CUDA-only and cannot execute on this host. The four folded helpers are
outside the timed region or strictly fewer operations than what they replace: `_style_groups` runs
once per render instead of twice (the same work), `_cached` removes one `setdefault` per access,
`_Cache` removes a `len()` per miss, `_restore` adds two array copies per **bake** (not per frame),
which are no-ops on an already-pristine model. The GPU timings are owed with the frames (§4).

## 7. Resource lifetime

`close()` is explicit, idempotent and never required for correctness from `__del__`; it drops the
per-instance device tensors and leaves the shared EGL/MJWarp/model caches alone (§2). A closed env
refuses `render`, `reset` and `step` with `Env`'s message. The suite's resource guard reports
**0 EGL contexts created / 0 released** across the Fetch tests in every venv, unchanged: nothing in
this workstream's tests reaches the ray-caster.

## 8. The §17 ten-item report

1. **files moved/removed/created** — §1. Created 4, changed 1, removed 0, moved 0.
2. **API changes** — §2. Additive: `num_envs=`, the two-key State, `close()`. `batch=` still works.
3. **Step-8 IDs affected** — the twelve `swm/Fetch*-v3`, behaviorally **none**: every comparable key
   is identical and the frames are owed rather than changed (§4).
4. **targeted tests** — §5. 17 new; 1385 → 1395 locally; 10/49 unchanged in both SWM venvs.
5. **applicable certification checks** — the frozen Step-4 corpus replay for all twelve cells on
   every key a CPU can reach (§4), and the Step-8 variation taxonomy asserted as totals. Gate A, the
   GPU conformance suite and the tier measurements need CUDA: **owed** (§4).
6. **default-frame comparison** — **OWED** (§4), and the reason is the family's, not the patch's. The
   default APPEARANCE that produces those frames is compared and identical, as is the baked plate
   that is 91% of them.
7. **variation/state comparison** — identical: `style_names`, `style_dim`, the default appearance
   value by value, `variation_names`, `classification()` per id, the taxonomy totals, `state_dim`,
   `state_names`, the corpus round trip (§4).
8. **performance comparison for hot paths** — not measurable on this host (§6); owed with the frames.
9. **resource-lifetime comparison** — §7. Explicit `close`, idempotent, closed env refuses; guard
   counts unchanged at 0/0.
10. **unexplained oracle differences** — **zero**, on all 41 comparable keys, for all twelve cells.

## 9. Requests for the coordinator / foundation

**R1 — run the GPU arm.** `scratch/step9/envs-fetch/oracle.py --python <a CUDA python with
mujoco_warp>` on a node with one GPU. It needs no edit: the warp stub is a no-op where the real
module exists, and the `_state_pool` substitution is the frozen corpus either way (and can be dropped
there by deleting the four-line patch if the interpreter has mujoco ≤ 3.11). Add `render()` digests
to `COMPARED` — the probe already builds every env — and this becomes the full frame oracle. This is
the single outstanding piece of evidence for this workstream.

**R2 — no foundation change is requested.** The instance-method `_layout` from Workstream B's R1
(`f24d94c`) was exactly what this family needed and nothing else was missing: `Env.__init__`'s
`batch=`/`num_envs=` handling, `_step_result`, `no_dynamics`, `_check_open` and `close` all fit a
compiled-model renderer-only family without an override. This is a data point for R1's generality:
two independent families with model-derived state widths needed the same thing and now share it.

**R3 — nothing is requested of Workstream I.** This family imports exactly three names from the
render layer: `MW.HIDDEN_GROUP`, `MW.DEFAULT_GROUPS`, `MW.gl_rounding` and `MW.create_render_context`
(four). None was touched here and all four must stay re-exported from `stable_envs.render.mjwarp`.
Recorded rather than patched, as the brief instructs.

**R4 — `tensordict` was installed into `/Users/samib/se8-fetch-venv`** (0.14.2, the version the
foundation put into the other two venvs). Without it `import stable_envs` fails there on
`dynamics.py`, and `se8-fetch-venv` is the ONLY local interpreter with `gymnasium_robotics`, i.e. the
only one where a Fetch model can be compiled at all. Flagged because installing into a shared venv is
a coordinator-owned decision, and it followed the Wave-0 precedent rather than a ruling.

**R5 — the skybox half of `background.color` stays PARTIAL and the Fetch defaults were not
re-frozen**, per Sami's ruling. Nothing in this patch touches the leaf, the map or the default, and
the oracle compares the default appearance value by value to prove it. The documented limitation and
its escalation (swm flattens the skybox; this family's is a 16-colour gradient, so no flat-fill leaf
reproduces the compiled appearance and adding one would move a frozen pixel oracle) stand unchanged
in `make_variation_space`'s docstring.

## 10. Escalations

| # | what | state |
|---|---|---|
| E1 | the frame comparison is owed to the GPU oracle | **ACCEPTED-AS-OWED**, R1. Listed in `oracle.json` under `owed_comparisons`, printed by the script, and stated in §4 and §8.6 rather than left as a silence. |
| E2 | live Fetch construction through the official env is **UNMEASURED-locally** (numpy 2.5.3 + mujoco 3.12 `set_joint_qpos`) | Pre-existing upstream defect, recorded at Step 3. The oracle works around it identically in both arms with the frozen corpus and says so; `reference_env`/`reference_render` are untouched. |
| E3 | `tensordict` installed into `se8-fetch-venv` | R4. |
| E4 | `_CAM_CACHE` now holds at most 256 entries where it held 257 | A cache capacity, not a contract: the answer for a given key is a pure function of the model and the camera parameters, and the frame is bit-identical at every cache size (the same argument `MJW_CHUNK` already records for chunking). Called out because it is the one numeric difference the dedup makes. |

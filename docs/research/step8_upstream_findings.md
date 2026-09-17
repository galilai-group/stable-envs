# Step 8 — consolidated SWM / upstream findings bundle

Workstream D (`step8/swm-integration`). Every item here is a defect or surprising behaviour in
`stable_worldmodel` or one of ITS dependencies (`gymnasium-robotics`, `ogbench`, `PyFlyt`,
`craftax`), not in `stable-envs`. None was fixed here: plan section 12 says to fix upstream
behaviour only if stable-envs cannot truthfully integrate without doing so, and every id below
already has an honest, documented status in
[`step8_swm_integration.md`](step8_swm_integration.md) (`MISMATCH(upstream)` where B measured it,
`UNMEASURED` or `NOT_APPLICABLE` elsewhere) that does not depend on any of these being fixed.

Two source layers, merged and deduplicated:

1. the plan-section-12 named items (found in the Step-7 per-family findings, or read directly out
   of the installed `stable_worldmodel` source on this Mac);
2. every `upstream_findings` / `buckets.UPSTREAM_FINDING` entry other Step-8 workstreams (B, C, F,
   H) already recorded in `examples/results/step8/dashboard.yaml`, carried forward here so this is
   the ONE place a reader looks for "what is broken upstream" rather than six.

## Plan section 12's named items

| id | finding | evidence |
|---|---|---|
| U1 | SWM `reset(seed)` behavior in OGB wrappers: `reset()` ACCEPTS a `seed` kwarg and discards it (`super().reset(options=options, *args, **kwargs)` never forwards it) in both `cube_env.py` and `scene_env.py`, so `self.np_random` is never reseeded. Measured: two `reset(seed=3)` draws on a fresh env differ by up to 2.98 in `qpos`. | `examples/results/step7/ogb-manipulation/findings.md:1026-1038` |
| U2 | `action_space` property reconstruction: `ogbench/manipspace/envs/env.py:246` builds a new, OS-seeded `Box` on every access (`u.action_space is u.action_space` → `False`). `ManipSpaceEnv.initialize_episode` calls `self.step(self.action_space.sample())` **twice** per reset, consuming two actions nothing outside can seed. | `examples/results/step7/ogb-manipulation/findings.md:1041-1046` |
| U3 | `reset_variation_space(seed=None)`: `cube_env.py`/`scene_env.py`'s `reset()` calls it with `seed=None`, and `stable_worldmodel/spaces.py:38` then seeds the variation space from OS entropy before resampling default leaves. Measured: `cube.start_position` differs between two identically-seeded resets. | `examples/results/step7/ogb-manipulation/findings.md:1048-1053` |
| U4 | `SceneEnv.reset(options={'state': ...})` calls `self.set_state(qpos, qvel)` without the `button_state_*` kwargs `SceneEnv.set_state` itself requires, and raises `KeyError: 0`. swm's own documented reset-from-state path does not work for `OGBScene`. | `examples/results/step7/ogb-manipulation/findings.md:1054-1057` |
| U5 | dead `sky_color`: `environment.sky_color` is offered in the Rocket variation space and read nowhere. | `examples/results/step7/rocket-landing/findings.md:1137-1140,1304-1306` |
| U6 | dead `_modify_rocket_urdf`: exists in the upstream Rocket wrapper but is never called; the URDF decoder is not rebuilt per episode. | `examples/results/step7/rocket-landing/findings.md:1131-1136,1304-1306` |
| U7 | PyFlyt flame recolor reset behavior: the ignition flame is recoloured on every state transition and never reset back — an upstream reset defect. | `examples/results/step7/rocket-landing/findings.md:1307-1308` |
| U8 | one-substep link/camera lag: PyFlyt/PyBullet render the frame after `step()` one substep behind the link/camera pose the step just integrated to. | `examples/results/step7/rocket-landing/findings.md:1309-1311` |
| U9 | Craftax wrapper JIT semantics: `stable_worldmodel/envs/craftax/craftax.py:CraftaxWrapper.__init__` builds `self._reset_fn = jax.jit(_reset, ...)` and `self._step_fn = jax.jit(_step, ...)`, so a LIVE SWM workflow always observes the JIT-produced frame. The certified stable-envs parity claim (100% exact, max \|diff\| 0) is measured against the **un-jitted** renderer; the jitted reference itself disagrees with its un-jitted self by up to 1.5e-05 in float32, which is invisible in uint8 on most frames and can flip one uint8 step on a sleeping/lit row (measured: 99.7854% exact at 7 px / 99.7225% at 16 px for Classic, max \|diff\| 1; 48/49 states byte-identical for Full, 1/49 differing on 0.0023% of pixels, max \|diff\| 1). Declared as a class attribute this workstream added: `CraftaxClassic.SWM_WRAPPER_GATE_F`, `CraftaxFull.SWM_WRAPPER_GATE_F`. | `stable_worldmodel/envs/craftax/craftax.py` (read directly, this Mac); `stable_envs/envs/craftax.py`; `stable_envs/envs/craftax_full.py`; `examples/results/step8/semantic/identity_recheck.json` |

## Carried forward from other Step-8 workstreams

| id | source | finding |
|---|---|---|
| U10 | C | `swm` OGB `modify_mjcf_model`: `agent_color_changed = np.allclose(current, desired)` is missing the negation, so the dirty flag is never set from that comparison alone; `agent.color` is dead when varied alone, live only when co-varied with another leaf that does set the flag. |
| U11 | C | `swm` `OGBMaze` in its registered `'states'` observation mode: `custom_renderer` is rebuilt only for `'pixels'` mode, so all 5 variation leaves move 0 px when the maze is used in `'states'` mode. |
| U12 | C | Rocket `pad.color` is declared `200` but a plain reset gives white; `OGBMaze` `pixel_encoding` is declared `0` but constructed as `1`. Declared vs. constructed defaults disagree upstream. |
| U13 | B | `swm` Fetch `_apply_visual_variations` unbinds three materials for the env's whole lifetime (19–32% of the frame), so any wrapper-level parity measurement is measuring the wrapper's own persistent side effect, not a fresh reset. |
| U14 | B | `swm` Fetch's declared variation-space leaf DEFAULTS are not the official Fetch appearance — this is the mechanism behind the `MISMATCH(upstream)` recorded in `swm_wrapper_reset_appearance` for all twelve `swm/Fetch*-v3` ids in `step8_swm_integration.md`. |
| U15 | B | `swm`'s Fetch variation space accepts `variation_values` list entries and uses them unconverted in at least one path, raising `AttributeError` for a caller who follows the documented shape. |
| U16 | B | `gymnasium-robotics` + `numpy 2.5.3` + `mujoco 3.12` cannot construct a Fetch env at all (a pybind enum vs. `np.int32` mismatch) — an environment/dependency-compatibility defect, not an appearance one. |
| U17 | H | `stable_worldmodel` registers four Craftax ids but ships no collector, training/planning config, checkpoint or baseline row for any of them — there is no existing downstream consumer the Craftax semantic follow-up could validate against, which is why H classified that follow-up as "not meaningfully testable downstream" rather than run a paired Gate H that would not test the claim. |
| U18 | F | `mujoco_warp` 3.13.0 ships a UV-less-mesh guard bundled with four independent render-path changes — fixes the crash Fetch needs but is not a drop-in bump (moves every frozen digest; see `docs/research/step8_reference_provenance.md`). |
| U19 | F | `gymnasium-robotics`' `set_joint_qpos` asserts under `mujoco >= 3.12`, so the official Fetch reference is unbuildable on any interpreter with mujoco 3.12+; certification needs `mujoco==3.11.0` specifically, which is a floor stable-envs' own `pyproject` `dmcontrol` extra does not currently pin (open decision for Sami, see `examples/results/step8/dashboard.yaml` buckets.DOCUMENTED_LIMITATION). |

## A note on `scratch/swm_matrix.json` (dashboard item, source E)

The dashboard recorded: *"`scratch/swm_matrix.json` stale: 6 of 42 rows have empty
`stable_envs.envs` (Workstream D's file)."* Re-checked at this HEAD: **8** of 42 rows have empty
`stable_envs.envs` now (`swm/CraftaxClassicSymbolic-v1`, `swm/CraftaxPixels-v1`,
`swm/CraftaxSymbolic-v1`, `swm/ManipulatorDMControl-v0`, `swm/OGBCube-v0`, `swm/OGBScene-v0`,
`swm/PFRocketLanding-v0`, `swm/QuadrupedDMControl-v0`) — two of those (the symbolic ids) are
CORRECTLY empty (no renderer, no stable-envs match), so the stale count is still 6, matching the
dashboard's own number. That legacy Step-3/7 tool (`scratch/swm_matrix.py`, 2287 lines, live env
introspection, GPU-only for four of the six stale rows) was not re-run: this workstream's
deliverable, `examples/results/step8/swm/gate_f_matrix.json` (built by
`scratch/step8/swm/build_gate_f.py`), already resolves the registry mapping for all 42 ids
correctly — including these six — from `examples/results/step8/support_matrix.json`'s
`join_evidence`, and is the current, re-generable, authoritative source. Patching the legacy file's
six rows in place was judged not worth doing twice; `STEP9_CLEANUP` note added in
`examples/results/step8/swm/findings.md`.

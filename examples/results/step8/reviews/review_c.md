# Independent Step-8 C review

Reviewed 2026-09-16 by a fresh reviewer. No workstream files were edited, reset, stashed, or committed.

**Committed candidate `a09b4074f59de06db10945e2384dc98bbdcbdb13`: APPROVE-WITH-CHANGES.**
**Candidate plus the existing dirty `stable_envs/envs/ogb_manip.py`: BLOCK.**
Neither state has an APPROVE verdict and neither may merge under coordinator-plan section 18.

Baseline: `43ac95238f7472137619f3e69e7636e11dd088b1`.
Dirty patch observed: 278 insertions / 26 deletions, 22,578 raw diff bytes,
SHA256 `b44808ed56e48bb2cfde5000959629eeab95b6437076a6c8cde59b22ace7ab60`.
The takeover archive has the same patch text plus one trailing newline (22,579 bytes,
SHA256 `41eb1fd383ff0af8752e503fdb19c801fad6911420e3298f0b61b40e66065330`).

## Findings requiring changes

1. **Dirty patch: a custom cube color persists into subsequent default renders.**
   `_cube_colours` returns `None` when the requested RGB is close to default;
   `render` then skips `_upload_cube_colours`. `_context` caches its per-world
   `geom_rgba`, so OGBCube retains the previous custom RGB. OGBCube has no Scene
   button-table upload that could reset it. This also crosses renderer instances
   sharing the module cache and can depend on chunk/batch history. A CPU buffer
   reproduction using the real helper methods produced `default upload: None`
   and cached RGBA `[[[1.0, 0.0, 0.0, 1.0]]]` after requesting the default
   `[0.96, 0.26, 0.33]`. Restore pristine defaults when required; test
   custom -> default, mixed -> default, default -> custom -> default, cross-instance
   cache reuse, and changed batch/chunk sizes on Oscar. The `np.allclose` shortcut
   also silently suppresses small nondefault colors; any shortcut needs an exact
   effective-color contract rather than an unmeasured tolerance.

2. **Dirty patch is not reflected in maps, declarations, tests, or evidence.**
   It introduces a nine-dimensional OGB appearance space while `newcells.py`
   still classifies cube/floor color unsupported and the tests explicitly pin
   empty OGB maps. Committed GPU evidence says `style_dim: 0`; findings say no
   renderer changed. Finish the authorized cube/floor scope by updating mappings,
   classifications, targeted tests, and generated reports. Record the declaration
   delta explicitly; preserve frozen corpora and do not replace an oracle.
   Claims in `_grid_texture` and `_apply_floor` of byte-identical texture/bake
   reproduction need saved probe artifacts tied to the completed code.

3. **Committed OGBMaze ours-only classification is incomplete.**
   Its variation space exposes `floor.tint` and `rendering.decolor_floor`, but
   neither is mapped nor listed in `_OGBMAZE_OURS_ONLY`. Only agent.scale and
   the two camera leaves are documented there. Add explicit reasons and an
   assertion covering every own-side leaf, not just SWM-side completeness.

4. **After A merges, do not map pixel_encoding by inversion alone.**
   Current `ogbench.py` implements `decolor_floor=1` by filling the texture with
   constant `[60,64,72]`; SWM `pixel_encoding=0` restores a two-color checker.
   `newcells.py` acknowledges this unresolved semantic difference, while
   findings.md calls the missing affine transform the only blocker. A's affine
   capability resolves representability, not visual equivalence. Measure both
   settings against the actual SWM configuration, then implement a justified
   mapping or accurately retain the semantic limitation. Update the misleading
   findings language either way.

5. **GPU evidence is useful but cannot satisfy the final merge gate as stored.**
   All five GPU JSON files have `provenance.git_sha: null`; they identify trees
   by mutable PYTHONPATH directory and tag only. They lack the required exact
   baseline SHA field. The default evidence draws eight states from seed 0;
   it is not an affected frozen-corpus replay. Default OGB and Rocket digests
   do match between their recorded baseline/branch arms, and Rocket's five
   mapped colors all have nonzero pixel movement. Missing optional families
   appear as errors in the other family's venv and must be interpreted with
   the complementary artifact, not as a universal pass. Bind any rerun to
   immutable source SHA/content manifests and dependencies. Run the affected
   Gate-A/Gate-F checks and frozen replay, especially for the completed renderer
   patch, without claiming old evidence covers new code.

## Independent checks

- `/Users/samib/se6-venv/bin/python -m pytest tests/unit/test_variation_maps_newcells.py -q`:
  **58 passed, 5 skipped**. The skips require live SWM. These tests verify the
  committed mapping declarations, not correctness of the unfinished renderer.
- `scratch/step8/variations-newcells/generate_report.py --check`: **PASS**.
  Generated classification JSON/leaf table match the committed classification
  source. This does not make them match the dirty renderer's new style space.
- CPU reproduction of stale cube RGBA: **defect reproduced** without a GPU,
  using `_cube_colours` and `_upload_cube_colours` with a mock context buffer.
- A live-SWM test attempt in `se7-ogb-venv` with `MUJOCO_GL=glfw` exited **134**
  before pytest emitted results. This is not a pass, nor evidence of a C code
  regression. No GPU/render validation was attempted locally.
- Static inspection confirms the committed candidate changes no renderer or
  style declaration; its code changes add maps and register them. The dirty
  patch is precisely the authorized OGB cube.color/floor.color implementation
  scope and must be preserved and finished, not discarded.

## Ten-part merge gate

| Requirement | Committed candidate | Candidate plus dirty renderer |
|---|---|---|
| 1. Explicit scope | PASS: C family variation classification/maps | PASS: authorized OGB cube/floor colors |
| 2. Before/after diagnosis | PASS with above taxonomy correction required | INCOMPLETE: default-reset defect and evidence claims must be resolved |
| 3. Targeted unit tests | PASS: 58/5; add own-side completeness test | MISSING targeted new-color/reset/cache tests |
| 4. Affected Gate-A/Gate-F | PARTIAL: inventory/registration tests; no complete affected run found | MISSING for new style dimensions, default preservation, isolation, mapping |
| 5. Affected frozen-corpus replay | MISSING: eight sampled states are not frozen replay | MISSING; preserve old oracle and explain style declaration delta |
| 6. Pixel-movement classification | No renderer change; sampled recorded defaults identical, SHA linkage incomplete | MISSING completed-code measurement; default should remain unchanged, new styles intentionally change pixels |
| 7. Hot-path performance regression | NOT APPLICABLE: declaration-only committed code | MISSING: host transfers/grouping and per-world uploads alter render path |
| 8. Independent verdict | APPROVE-WITH-CHANGES | BLOCK |
| 9. Semantic follow-up decision | None needed for unchanged renderer | Must be recorded after isolation/default evidence; do not presume unchanged from old findings |
| 10. Dashboard update | Existing entry records delivery but not this verdict | Must record unfinished patch and failing default-reset behavior |

The cheap authorized patch should be completed. Additional light/camera/size and
Scene lock_color decisions remain coordinator release-scope decisions; this review
does not approve declaring feasible required leaves permanently unsupported.

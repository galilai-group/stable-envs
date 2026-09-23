# Implementation source map and accepted-scope closure

Starting integration: `repair/environment-semantics` at
`a5ef2e47662d352022468a1765f06d261497033a`, clean. The actual inventory is
21 VERIFIED (12 DMC, 5 classic control, 3 Navigation, 1 Maze) and 21 INCOMPLETE
(12 Fetch, 2 OGBench manipulation, 4 Craftax, 1 Rocket, 2 PushT).

The canonical machine-readable map is
[semantic_recovery_implementation_map.json](semantic_recovery_implementation_map.json).
It indexes all 42 registrations from the actual pinned SWM registration table,
constructor defaults, exact wrapper method bodies and source URLs. Named State
schemas come from the independent immutable corpora; candidate restoration
implementations are indexed alongside them. Unaccepted implementations are
explicitly unproved. Historical lock versions and tested profile versions are
separate provenance fields, not interchangeable authority claims.

The closure audit checked every existing semantic corpus manifest and every
recovery SHA256SUMS file. All resolve and match. Every verified class is present
in its accepted renderer evidence; separately stored baseline/candidate results
match. Actual terminal success markers were checked. Five older original Oscar
logs were retrieved into `semantic_recovery_evidence/implementation_closure/`;
this closes local evidence packaging gaps without rerunning accepted gates.

Production Python ASTs match the latest covering gates: all DMC at `07d7098`
(job 6628919 also reran Reacher, first-family, continuation and aligned gates),
classic control at `98c65cb`, Navigation at `6ee8d8cb`, and Maze at `7f2f3ae`.
Maze's textual differences are formatting only. Acrobot's corrected sparse
identity uses its later identity proof and translated frozen cell name, not the
obsolete Swingup spelling. Humanoid is Walk; Cartpole is sparse Swingup; Maze
is Point/Large/Navigate. Renderer-only siblings are not counted as accepted.
No closure defect or status discrepancy was found, and no accepted production
code or gate was rerun or changed.

Bounded Variation, task/reset-option, renderer and dependency-profile limitations
remain attached to each row. DMC uses delegated CPU physics even in CUDA tests.
The historical audit hash and renderer baseline pointer remain unchanged.

For future Fetch work, the complete documented PCG64 structure is treated as one
atomic named State component. The earlier checkpoint's individual cached-word
minimality question is not an acceptance requirement under the current user
instruction. Existing experiments and failed diagnostics remain unmodified.

Reproduce the original Phase A map and closure checks at commit `f51d26eb`
(before the later Fetch promotion) with:

```sh
.venv/bin/python scripts/recovery/build_implementation_map.py --swm /Users/samib/stable-worldmodel
```

This audit is committed separately from Fetch implementation. No push is made.

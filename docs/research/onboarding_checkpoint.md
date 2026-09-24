# Contributor onboarding checkpoint

Start: clean `repair/environment-semantics` at
`e1b7e5a364b02386e1d3ed6ace29c3d22c7826f3`, the exact containing PushT checkpoint.
PushT evidence commit remains `6873b7d608cfe0fb7777fb35680e859e2160f511`;
Fetch promotion remains `b45500ca84d8c6b3216d90196acac747e8fb3d5b`.
The containing documentation commit's SHA is recorded in the subsequent handoff.

Added `docs/adding/{index,environment,task,renderer}.md`; retained
`docs/adding-an-env.md` as a compatibility page. Updated navigation, README,
CONTRIBUTING, quickstart/home, State/dynamics/physics/lifecycle/ownership concepts,
Env API and testing guidance. Examples use accepted CartPole and Fetch behavior.
Full State, private RenderState, Frame and downstream learner observation are
separate; no new shared class or abstraction is proposed.

The pre-change strict MkDocs build failed on ten archived source/directory links.
`docs/_hooks.py` supplies downloadable source assets and evidence directory indexes
in the built site, and rewrites those links only in generated pages. Original
hashed research documents remain byte-identical. Strict validation remains enabled.
A stale testing-page anchor now points to the new authority guidance.

Validation:

- Existing closure audit: 33 accepted registrations, 170 distinct artifact hashes,
  11 terminal logs and all 75 Fetch-manifest files pass, before and after edits.
- Initial shared checks: 42 passed, one optional ALE warning.
- Final targeted suite: 57 passed, one optional ALE warning.
- All eight Python blocks in the four new guides, State/dynamics/lifecycle concepts
  and quickstart execute successfully in the accepted Fetch CPU profile.
- `python -m mkdocs build --strict --config-file mkdocs.yaml --site-dir /private/tmp/stable-envs-onboarding-site`:
  PASS, including links/anchors and generated archive indexes.
- `ruff check docs/_hooks.py`, `ruff format --check docs/_hooks.py`, `git diff --check`: PASS.
- Production, tests, package metadata, shared contract, acceptance matrix/map and
  all earlier evidence: unchanged. No actual shared-API contradiction was found;
  stale documentation was corrected to match the accepted implementation.

Targeted suite command:

```sh
.venv/bin/python -m pytest -q tests/unit/test_repo_contracts.py tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py tests/unit/test_native_classic_semantics.py tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py
```

Raw documentation/test/example logs and their SHA256 manifest are in
`semantic_recovery_evidence/onboarding/`. Example execution extracted each fenced
Python block and executed it independently with
`/private/tmp/fetch-authority-swm-lock/bin/python`, `SDL_VIDEODRIVER=dummy`, and
`PYTHONPATH` pointing at the integration checkout. This validates examples; it
is not new family acceptance evidence.

Status remains **33 VERIFIED / 9 INCOMPLETE**. No push. OGBench implementation work
begins only after this documentation checkpoint is committed cleanly.

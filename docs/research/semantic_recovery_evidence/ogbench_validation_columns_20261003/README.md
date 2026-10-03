# Column-wise Cube/Scene validation candidate

Base a388ee63fbbbc7039ea014ee91e6b9cec9d84e40. Production is unchanged.
The prior render-entry profile measured thousands of per-lane Tensor.cpu calls,
TensorDict unbinding and scalar checks. This candidate validates shape/dtype,
transfers and owns each complete host column once, checks finiteness on CPU,
then presents row views to the exact unchanged semantic validation tail.
RNG, task, lifecycle, quaternion, range and dependency/profile checks are retained.
Rows cannot alias input buffers or overlap another lane. No renderer, State
schema, process route or shared step API changes.

Local tests: 78 pass, ten CUDA cases skipped. Every leaf's presence, shape and
dtype is tested, plus all nonempty floating leaves' NaN/Inf rejection, semantic
final-lane rejection, noncontiguous inputs, metadata batch override, ownership,
one-transfer-per-column counts, varied State, cached RNG and restored continuation.
Existing process tests include all-shard failed-restore isolation and worker death.
The initial 10-pass/10-skip report predates import-format cleanup of the frozen
reference; the full report matches the final files. The legacy implementation
is derived exactly from the base with only recorded import rewrites.

Each cluster task runs 88 CPU/CUDA tests and independently gates one registration
with the ordinary eight-owner constructor on CPU/software GL and CUDA/NVIDIA.
Twelve public cases/device cover chunk/single actions, slices, owned snapshots,
unseeded reset, variations, malformed render/restore rejection and validation
while worker stepping is in flight. The paired benchmark uses one population;
only the parent validator switches to the frozen implementation. Worker stepping
is unchanged, and restoration stays outside timing. It does not hold two
simulator populations or claim an upstream speedup.

B64/128, T1/8, all States retained and no/final/all frames receive six samples
in reversed strategy order. Separate validation, projection and saved full-State
rendering measurements expose the remaining costs. The slice API and default
renderers are unchanged. CPU and CUDA acceptance plus measured performance are
required before applying the candidate to production. Reconstruct by exporting
the base commit, overlaying candidate/, then verifying source_manifest.json.

Array 6970790 is running independently for Cube and Scene. Both tasks have
completed the 88-test CPU/CUDA suite with zero failures or skips; XML is under
cluster-tests/. Public frame/State gates and performance sweeps are still
pending. These unit tests alone do not enable the candidate in production.
